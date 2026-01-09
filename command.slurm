#!/bin/bash
#SBATCH -o ./log/%j.out			    #指定作业的标准输出
#SBATCH -e ./log/%j.err			    #指定作业的错误输出
#SBATCH -J gmres			    #指定作业名称
#SBATCH -t 02:00:00			    #指定作业最大运行30分钟
#SBATCH -N 1				    #指定节点数
#SBATCH --nodelist=hepnode1

source ../../spack/share/spack/setup-env.sh
spack load glog
spack load sentencepiece
spack load googletest
spack load armadillo
rm -r build
mkdir build
cd build
cmake ..
make
cd sampler/simple_chat
/home/lyl/local/NVIDIA-Nsight-Compute-2025.4/ncu --target-processes all \
    --kernel-name SingleDecodeWithKVCacheKernel \
    --set full \
    ./simple_chat ../../../../../models/bf16/model.bin ../../../../../models/llama-2-7b/tokenizer.model

