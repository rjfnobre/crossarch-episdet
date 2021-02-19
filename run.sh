#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
echo "############## Executing on DevCloud"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/gsl/lib

# export SYCL_BE=PI_OPENCL
# cat /proc/cpuinfo
sycl-ls

./bin/episdet datasets/1024SNPs_4096samples.csv
#./bin/episdet datasets/4096SNPs_8192samples.csv

echo "############## Execution completed"

