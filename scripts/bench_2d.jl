using TinyKernels
using BenchmarkTools

include("setup_benchs.jl")

# Select based upon your local device (:CPU, :CUDA, :AMDGPU, :Metal)
backend = :CPU

@setup_benchs()

@tiny function kernel_memcopy_triad!(A, B, C, s)
    ix, iy = @indices()
    @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    return
end

@tiny function kernel_diff_step!(A, B, C, s)
    ix, iy = @indices()
    if (ix>1 && ix<size(A,1) && iy>1 && iy<size(A,2))
        @inbounds A[ix, iy] = B[ix, iy] + s / 50.0 * (C[ix, iy] * (
                              -((-(B[ix+1, iy] - B[ix, iy])) - (-(B[ix, iy] - B[ix-1, iy])))
                              -((-(B[ix, iy+1] - B[ix, iy])) - (-(B[ix, iy] - B[ix, iy-1])))))
    end
    return
end

function compute!(fun!, A, B, C, s, ranges, nrep)
    for _ in 1:nrep
        inner_event  =  fun!(A, B, C, s; ndrange=ranges[1])
        outer_events = [fun!(A, B, C, s; ndrange=ranges[i], priority=:high) for i in 2:lastindex(ranges)]
        wait(outer_events)
        wait(inner_event)
    end
    return
end

function compute!(fun!, A, B, C, s, nrep)
    for _ in 1:nrep
        event = fun!(A, B, C, s; ndrange=size(A))
        wait(event)
    end
    return
end

function main(::Type{DAT}; device) where DAT
    nx, ny = 4096*4, 4096*4
    A = device_array(DAT, device, nx, ny)
    B = device_array(DAT, device, nx, ny)
    C = device_array(DAT, device, nx, ny)
    copyto!(B, rand(DAT, nx, ny))
    fill!(C, DAT(2.0))
    s = DAT(1.5)
    nrep = 2
    b_w = (32, 16)
    # compute ranges
    ranges = ((b_w[1]+1:nx-b_w[1] , b_w[2]+1:ny-b_w[2] ),
              (1:b_w[1]           , 1:ny               ),
              (nx-b_w[1]+1:nx     , 1:ny               ),
              (b_w[1]+1:nx-b_w[1] , 1:b_w[2]           ),
              (b_w[1]+1:nx-b_w[1] , ny-b_w[2]+1:ny     ))

    println("testing Memcopy triad 2D")
    memcopy_triad! = kernel_memcopy_triad!(device)
    TinyKernels.device_synchronize(device)
    # warmup
    compute!(memcopy_triad!, A, B, C, s, ranges, 2)
    compute!(memcopy_triad!, A, B, C, s, 2)
    TinyKernels.device_synchronize(device)
    # split
    t_nrep = @belapsed compute!($memcopy_triad!, $A, $B, $C, $s, $ranges, $nrep)
    t_eff = sizeof(eltype(A)) * 3 * length(A) * 1e-9 / (t_nrep / nrep)
    println("  split    - time (s) = $(round(t_nrep/nrep, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
    TinyKernels.device_synchronize(device)
    # no split
    t_nrep = @belapsed compute!($memcopy_triad!, $A, $B, $C, $s, $nrep)
    t_eff = sizeof(eltype(A)) * 3 * length(A) * 1e-9 / (t_nrep / nrep)
    println("  no split - time (s) = $(round(t_nrep/nrep, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")

    println("testing Diffusion step 2D")
    diff_step! = kernel_diff_step!(device)
    TinyKernels.device_synchronize(device)
    # warmup
    compute!(diff_step!, A, B, C, s, ranges, 2)
    compute!(diff_step!, A, B, C, s, 2)
    TinyKernels.device_synchronize(device)
    # split
    t_nrep = @belapsed compute!($diff_step!, $A, $B, $C, $s, $ranges, $nrep)
    t_eff = sizeof(eltype(A)) * 3 * length(A) * 1e-9 / (t_nrep / nrep)
    println("  split    - time (s) = $(round(t_nrep/nrep, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
    TinyKernels.device_synchronize(device)
    # no split
    t_nrep = @belapsed compute!($diff_step!, $A, $B, $C, $s, $nrep)
    t_eff = sizeof(eltype(A)) * 3 * length(A) * 1e-9 / (t_nrep / nrep)
    println("  no split - time (s) = $(round(t_nrep/nrep, digits=5)), T_eff (GB/s) = $(round(t_eff, digits=2))")
    return
end

println("running on $backend device...")
main(eletype; device)