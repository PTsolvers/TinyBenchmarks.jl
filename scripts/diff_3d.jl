using TinyKernels
using BenchmarkTools

using CUDA
@static if CUDA.functional()
    using TinyKernels.CUDABackend
end

using AMDGPU
@static if AMDGPU.functional()
    using TinyKernels.ROCBackend
end

@tiny function comp_flux!(qx, qy, qz, A, C, _dx, _dy, _dz)
    ix, iy, iz = @indices()
    if (ix <= size(qx, 1) && iy <= size(qx, 2) - 2 && iz <= size(qz, 3) - 2)
        @inbounds qx[ix, iy, iz] = -(C[ix, iy+1, iz+1] + C[ix+1, iy+1, iz+1]) * 0.5 * _dx * (A[ix+1, iy+1, iz+1] - A[ix, iy+1, iz+1])
    end
    if (ix <= size(qy, 1) - 2 && iy <= size(qy, 2) && iz <= size(qz, 3) - 2)
        @inbounds qy[ix, iy, iz] = -(C[ix+1, iy, iz+1] + C[ix+1, iy+1, iz+1]) * 0.5 * _dy * (A[ix+1, iy+1, iz+1] - A[ix+1, iy, iz+1])
    end
    if (ix <= size(qz, 1) - 2 && iy <= size(qz, 2) - 2 && iz <= size(qz, 3))
        @inbounds qz[ix, iy, iz] = -(C[ix+1, iy+1, iz] + C[ix+1, iy+1, iz+1]) * 0.5 * _dz * (A[ix+1, iy+1, iz+1] - A[ix+1, iy+1, iz])
    end
    return
end

@tiny function comp_mbal!(A, qx, qy, qz, dt, _dx, _dy, _dz)
    ix, iy, iz = @indices()
    if (ix <= size(A, 1) - 2 && iy <= size(A, 2) - 2 && iz <= size(A, 3) - 2)
        @inbounds A[ix+1, iy+1, iz+1] += -dt * (_dx * (qx[ix+1, iy, iz] - qx[ix, iy, iz])
                                              + _dy * (qy[ix, iy+1, iz] - qy[ix, iy, iz])
                                              + _dz * (qz[ix, iy, iz+1] - qz[ix, iy, iz]))
    end
    return
end

function compute!(fun1!, fun2!, A, qx, qy, qz, C, dt, _dx, _dy, _dz, nt)
    for it in 1:nt
        ev1 = fun1!(qx, qy, qz, A, C, _dx, _dy, _dz; ndrange=size(A))
        wait(ev1)
        ev2 = fun2!(A, qx, qy, qz, dt, _dx, _dy, _dz; ndrange=size(A))
        wait(ev2)
        # update_halo!(A) # here
    end
    return
end

function compute!(fun1!, fun2!, A, qx, qy, qz, C, dt, _dx, _dy, _dz, nt, ranges)
    for it in 1:nt
        ev1 = fun1!(qx, qy, qz, A, C, _dx, _dy, _dz; ndrange=size(A))
        wait(ev1)
        inn_ev2 =  fun2!(A, qx, qy, qz, dt, _dx, _dy, _dz; ndrange=ranges[1])
        out_ev2 = [fun2!(A, qx, qy, qz, dt, _dx, _dy, _dz; ndrange=ranges[i], priority=:high) for i in 2:lastindex(ranges)]
        wait(out_ev2)
        # update_halo!(A) # here
        wait(inn_ev2)
    end
    return
end

function main(; device)
    # physics
    lx = ly = lz = 10.0
    c0 = 2.0
    nt = 2
    # numerics
    nx = ny = nz = 512#1024
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    dt = min(dx, dy, dz)^2 / c0 / 6.1
    b_w = (32, 16, 2)
    nIO = 6
    _dx, _dy, _dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    # init arrays
    A  = device_array(Float64, device, nx, ny, nz); copyto!(A, rand(nx, ny, nz))
    C  = device_array(Float64, device, nx, ny, nz); fill!(C, c0)
    qx = device_array(Float64, device, nx-1, ny-2, nz-2); copyto!(qx, zeros(nx-1, ny-2, nz-2))
    qy = device_array(Float64, device, nx-2, ny-1, nz-2); copyto!(qy, zeros(nx-2, ny-1, nz-2))
    qz = device_array(Float64, device, nx-2, ny-1, nz-1); copyto!(qz, zeros(nx-2, ny-1, nz-1))
    # compute ranges
    ranges = ((b_w[1]+1:nx-b_w[1], b_w[2]+1:ny-b_w[2], b_w[3]+1:nz-b_w[3]),
              (1:b_w[1]          , 1:ny              , 1:nz              ),
              (nx-b_w[1]+1:nx    , 1:ny              , 1:nz              ),
              (b_w[1]+1:nx-b_w[1], 1:ny              , 1:b_w[3]          ),
              (b_w[1]+1:nx-b_w[1], 1:ny              , nz-b_w[3]+1:nz    ),
              (b_w[1]+1:nx-b_w[1], 1:b_w[2]          , b_w[3]+1:nz-b_w[3]),
              (b_w[1]+1:nx-b_w[1], ny-b_w[2]+1:ny    , b_w[3]+1:nz-b_w[3]))
    # action
    println("testing Diffusion step 3D")
    kernel_comp_flux! = comp_flux!(device)
    kernel_comp_mbal! = comp_mbal!(device)
    TinyKernels.device_synchronize(device)
    # warmup
    compute!(kernel_comp_flux!, kernel_comp_mbal!, A, qx, qy, qz, C, dt, _dx, _dy, _dz, nt, ranges)
    compute!(kernel_comp_flux!, kernel_comp_mbal!, A, qx, qy, qz, C, dt, _dx, _dy, _dz, nt)
    TinyKernels.device_synchronize(device)
    # split
    t_nt = @belapsed compute!($kernel_comp_flux!, $kernel_comp_mbal!, $A, $qx, $qy, $qz, $C, $dt, $_dx, $_dy, $_dz, $nt, $ranges)
    t_eff = sizeof(eltype(A)) * nIO * length(A) * 1e-9 / (t_nt / nt)
    println("  split    - time (s) = $(round(t_nt/nt, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
    TinyKernels.device_synchronize(device)
    # no split
    t_nt = @belapsed compute!($kernel_comp_flux!, $kernel_comp_mbal!, $A, $qx, $qy, $qz, $C, $dt, $_dx, $_dy, $_dz, $nt)
    t_eff = sizeof(eltype(A)) * nIO * length(A) * 1e-9 / (t_nt / nt)
    println("  no split - time (s) = $(round(t_nt/nt, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
    TinyKernels.device_synchronize(device)
    return
end

@static if CUDA.functional()
    println("running on CUDA device...")
    main(;device=CUDADevice())
end

@static if AMDGPU.functional()
    println("running on AMD device...")
    main(;device=ROCBackend.ROCDevice())
end
