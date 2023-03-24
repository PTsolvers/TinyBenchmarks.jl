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

@tiny function comp_flux!(qx, qy, A, C, _dx, _dy)
    ix, iy = @indices()
    if (ix <= size(qx, 1) && iy <= size(qx, 2) - 2)
        @inbounds qx[ix, iy] = -(C[ix, iy+1] + C[ix+1, iy+1]) * 0.5 * _dx * (A[ix+1, iy+1] - A[ix, iy+1])
    end
    if (ix <= size(qy, 1) - 2 && iy <= size(qy, 2))
        @inbounds qy[ix, iy] = -(C[ix+1, iy] + C[ix+1, iy+1]) * 0.5 * _dy * (A[ix+1, iy+1] - A[ix+1, iy])
    end
    return
end

@tiny function comp_mbal!(A, qx, qy, dt, _dx, _dy)
    ix, iy = @indices()
    if (ix <= size(A, 1) - 2 && iy <= size(A, 2) - 2)
        @inbounds A[ix+1, iy+1] += -dt * (_dx * (qx[ix+1, iy] - qx[ix, iy]) + _dy * (qy[ix, iy+1] - qy[ix, iy]))
    end
    return
end

function compute!(fun1!, fun2!, A, qx, qy, C, dt, _dx, _dy, nt)
    for it in 1:nt
        ev1 = fun1!(qx, qy, A, C, _dx, _dy; ndrange=size(A))
        wait(ev1)
        ev2 = fun2!(A, qx, qy, dt, _dx, _dy; ndrange=size(A))
        wait(ev2)
        # update_halo!(A) # here
    end
    return
end

function compute!(fun1!, fun2!, A, qx, qy, C, dt, _dx, _dy, nt, ranges)
    for it in 1:nt
        ev1 = fun1!(qx, qy, A, C, _dx, _dy; ndrange=size(A))
        wait(ev1)
        inn_ev2 =  fun2!(A, qx, qy, dt, _dx, _dy; ndrange=ranges[1])
        out_ev2 = [fun2!(A, qx, qy, dt, _dx, _dy; ndrange=ranges[i], priority=:high) for i in 2:lastindex(ranges)]
        wait(out_ev2)
        # update_halo!(A) # here
        wait(inn_ev2)
    end
    return
end

function main(; device)
    # physics
    lx = ly = 10.0
    c0 = 2.0
    nt = 2
    # numerics
    nx, ny = 4096*8, 4096*8
    dx, dy = lx / nx, ly / ny
    dt = min(dx, dy)^2 / c0 / 4.1
    b_w = (32, 16)
    nIO = 5
    _dx, _dy = 1.0 / dx, 1.0 / dy
    # init arrays
    A  = device_array(Float64, device, nx, ny); copyto!(A, rand(nx, ny))
    C  = device_array(Float64, device, nx, ny); fill!(C, c0)
    qx = device_array(Float64, device, nx-1, ny-2); copyto!(qx, zeros(nx-1, ny-2))
    qy = device_array(Float64, device, nx-2, ny-1); copyto!(qy, zeros(nx-2, ny-1))
    # compute ranges
    ranges = ((b_w[1]+1:nx-b_w[1] , b_w[2]+1:ny-b_w[2]),
              (1:b_w[1]           , 1:ny              ),
              (nx-b_w[1]+1:nx     , 1:ny              ),
              (b_w[1]+1:nx-b_w[1] , 1:b_w[2]          ),
              (b_w[1]+1:nx-b_w[1] , ny-b_w[2]+1:ny    ))
    # action
    println("testing Diffusion step 2D")
    kernel_comp_flux! = comp_flux!(device)
    kernel_comp_mbal! = comp_mbal!(device)
    TinyKernels.device_synchronize(device)
    # warmup
    compute!(kernel_comp_flux!, kernel_comp_mbal!, A, qx, qy, C, dt, _dx, _dy, nt, ranges)
    compute!(kernel_comp_flux!, kernel_comp_mbal!, A, qx, qy, C, dt, _dx, _dy, nt)
    TinyKernels.device_synchronize(device)
    # split
    t_nt = @belapsed compute!($kernel_comp_flux!, $kernel_comp_mbal!, $A, $qx, $qy, $C, $dt, $_dx, $_dy, $nt, $ranges)
    t_eff = sizeof(eltype(A)) * nIO * length(A) * 1e-9 / (t_nt / nt)
    println("  split    - time (s) = $(round(t_nt/nt, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
    TinyKernels.device_synchronize(device)
    # no split
    t_nt = @belapsed compute!($kernel_comp_flux!, $kernel_comp_mbal!, $A, $qx, $qy, $C, $dt, $_dx, $_dy, $nt)
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
