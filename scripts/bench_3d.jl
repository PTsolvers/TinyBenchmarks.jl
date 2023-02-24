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

@tiny function memcopy_triad!(A, B, C, s)
    ix, iy, iz = @indices()
    @inbounds A[ix, iy, iz] = B[ix, iy, iz] + s * C[ix, iy, iz]
    return
end

@tiny function diff_step!(A, B, C, s)
    ix, iy, iz = @indices()
    if (ix>1 && ix<size(A,1) && iy>1 && iy<size(A,2) && iz>1 && iz<size(A,3))
        @inbounds A[ix, iy, iz] = B[ix, iy, iz] + s / 50.0 * (C[ix, iy, iz] * (
                                  -((-(B[ix+1, iy, iz] - B[ix, iy, iz])) - (-(B[ix, iy, iz] - B[ix-1, iy, iz])))
                                  -((-(B[ix, iy+1, iz] - B[ix, iy, iz])) - (-(B[ix, iy, iz] - B[ix, iy-1, iz])))
                                  -((-(B[ix, iy, iz+1] - B[ix, iy, iz])) - (-(B[ix, iy, iz] - B[ix, iy, iz-1])))))
    end
    return
end

function compute!(fun!, A, B, C, s, ranges, nrep)
    for ir in 1:nrep
        inner_event  =  fun!(A, B, C, s; ndrange=ranges[1])
        outer_events = [fun!(A, B, C, s; ndrange=ranges[i], priority=:high) for i in 2:lastindex(ranges)]
        wait(outer_events)
        wait(inner_event)
    end
    return
end

function compute!(fun!, A, B, C, s, nrep)
    for ir in 1:nrep
        event = fun!(A, B, C, s; ndrange=size(A))
        wait(event)
    end
    return
end

function main(; device)
    nx = ny = nz = 1024
    A = device_array(Float64, device, nx, ny, nz)
    B = device_array(Float64, device, nx, ny, nz)
    C = device_array(Float64, device, nx, ny, nz)
    copyto!(B, rand(nx, ny, nz))
    fill!(C, 2.0)
    s = 1.5
    nrep = 2
    b_w = (32, 16, 2)
    # compute ranges
    ranges = ((b_w[1]+1:nx-b_w[1], b_w[2]+1:ny-b_w[2], b_w[3]+1:nz-b_w[3]),
              (1:b_w[1]          , 1:ny              , 1:nz              ),
              (nx-b_w[1]+1:nx    , 1:ny              , 1:nz              ),
              (b_w[1]+1:nx-b_w[1], 1:ny              , 1:b_w[3]          ),
              (b_w[1]+1:nx-b_w[1], 1:ny              , nz-b_w[3]+1:nz    ),
              (b_w[1]+1:nx-b_w[1], 1:b_w[2]          , b_w[3]+1:nz-b_w[3]),
              (b_w[1]+1:nx-b_w[1], ny-b_w[2]+1:ny    , b_w[3]+1:nz-b_w[3]))

    println("testing Memcopy triad 3D")
    kernel_memcopy_triad! = Kernel(memcopy_triad!, device)
    TinyKernels.device_synchronize(device)
    # warmup
    compute!(kernel_memcopy_triad!, A, B, C, s, ranges, 2)
    compute!(kernel_memcopy_triad!, A, B, C, s, 2)
    TinyKernels.device_synchronize(device)
    # split
    t_nrep = @belapsed compute!($kernel_memcopy_triad!, $A, $B, $C, $s, $ranges, $nrep)
    t_eff = sizeof(eltype(A)) * 3 * length(A) * 1e-9 / (t_nrep / nrep)
    println("  split    - time (s) = $(round(t_nrep/nrep, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
    TinyKernels.device_synchronize(device)
    # no split
    t_nrep = @belapsed compute!($kernel_memcopy_triad!, $A, $B, $C, $s, $nrep)
    t_eff = sizeof(eltype(A)) * 3 * length(A) * 1e-9 / (t_nrep / nrep)
    println("  no split - time (s) = $(round(t_nrep/nrep, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")

    println("testing Diffusion step 3D")
    kernel_diff_step! = Kernel(diff_step!, device)
    TinyKernels.device_synchronize(device)
    # warmup
    compute!(kernel_diff_step!, A, B, C, s, ranges, 2)
    compute!(kernel_diff_step!, A, B, C, s, 2)
    TinyKernels.device_synchronize(device)
    # split
    t_nrep = @belapsed compute!($kernel_diff_step!, $A, $B, $C, $s, $ranges, $nrep)
    t_eff = sizeof(eltype(A)) * 3 * length(A) * 1e-9 / (t_nrep / nrep)
    println("  split    - time (s) = $(round(t_nrep/nrep, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
    TinyKernels.device_synchronize(device)
    # no split
    t_nrep = @belapsed compute!($kernel_diff_step!, $A, $B, $C, $s, $nrep)
    t_eff = sizeof(eltype(A)) * 3 * length(A) * 1e-9 / (t_nrep / nrep)
    println("  no split - time (s) = $(round(t_nrep/nrep, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
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
