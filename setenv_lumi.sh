#!/bin/bash

module load CrayEnv
# module load PrgEnv-cray
module load craype-accel-amd-gfx90a # MI250x
module load rocm

## no longer needed
# module load cray-mpich
# export JULIA_AMDGPU_DISABLE_ARTIFACTS=1
# export JULIA_AMDGPU_MAX_SCRATCH=128

## Instead use preferences:
# AMDGPU.enable_artifacts!(false)
# AMDGPU.Runtime.set_max_scratch!(128)

# add https://github.com/luraess/ImplicitGlobalGrid.jl#lr/amdgpu-0.4.x-support
# MPIPreferences.use_system_binary(; library_names=["libmpi_cray"], mpiexec="srun")

## And set following to circumvent precompile conflict between login and compute node
# JULIA_PKG_PRECOMPILE_AUTO=0

# ROCm-aware MPI set to 1, else 0
export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_ROCMAWARE_MPI=1

# Needs to know about location of GTL lib
export LD_PRELOAD=${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so

echo "ENV setup done"
