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
    ix, iy = @indices()
    @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    return
end

@tiny function diff_step!(A, B, C, s)
    ix, iy = @indices()
    if (ix>1 && ix<size(A,1) && iy>1 && iy<size(A,2))
        @inbounds A[ix, iy] = B[ix, iy] + s / 50.0 * (C[ix, iy] * (
                              -((-(B[ix+1, iy] - B[ix, iy])) - (-(B[ix, iy] - B[ix-1, iy])))
                              -((-(B[ix, iy+1] - B[ix, iy])) - (-(B[ix, iy] - B[ix, iy-1])))))
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
    nx, ny = 4096*4, 4096*4
    A = device_array(Float64, device, nx, ny)
    B = device_array(Float64, device, nx, ny)
    C = device_array(Float64, device, nx, ny)
    copyto!(B, rand(nx, ny))
    fill!(C, 2.0)
    s = 1.5
    nrep = 2
    b_w = (32, 16)
    # compute ranges
    ranges = ((b_w[1]+1:nx-b_w[1] , b_w[2]+1:ny-b_w[2] ),
              (1:b_w[1]           , 1:ny               ),
              (nx-b_w[1]+1:nx     , 1:ny               ),
              (b_w[1]+1:nx-b_w[1] , 1:b_w[2]           ),
              (b_w[1]+1:nx-b_w[1] , ny-b_w[2]+1:ny     ))

    println("testing Memcopy triad 2D")
    kernel_memcopy_triad! = memcopy_triad!(device)
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

    println("testing Diffusion step 2D")
    kernel_diff_step! = diff_step!(device)
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
    main(; device=CUDADevice())
end

@static if AMDGPU.functional()
    println("running on AMD device...")
    main(; device=ROCBackend.ROCDevice())
end
