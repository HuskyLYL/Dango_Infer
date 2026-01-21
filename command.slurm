#!/bin/bash
#SBATCH -o ./log/%j.out			    #指定作业的标准输出
#SBATCH -e ./log/%j.err			    #指定作业的错误输出
#SBATCH -J llama_tp4                         #指定作业名称
#SBATCH -t 02:00:00                         #指定作业最大运行时间
#SBATCH -N 2                                #节点数
#SBATCH --ntasks=4                          #总进程数
#SBATCH --ntasks-per-node=2                 #每节点2进程
#SBATCH --nodelist=hepnode1,hepnode2        #节点列表

source ../../spack/share/spack/setup-env.sh
spack load glog
spack load sentencepiece
spack load googletest
spack load armadillo
spack load nccl
spack load openmpi
#network env for nccl
#export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=br0
export NCCL_P2P_DISABLE=0


set -e
CATKIN_WS="$(pwd)"
CATKIN_WS_BUILD="$CATKINWS/build"

CATKIN_WS_MODEL_DIR="$CATKIN_WS/models"
CATKIN_WS_MODEL_BIN="$CATKIN_WS_MODEL_DIR/bf16/model.bin"
CATKIN_WS_TOKENIZER="$CATKIN_WS_MODEL_DIR/llama-2-7b/tokenizer.model"


rm -rf "$CATKIN_WS_BUILD"
cmake -S "$CATKIN_WS" -B "$CATKIN_WS_BUILD"
cmake --build "$CATKIN_WS_BUILD" -j

srun --mpi=pmix_v5 "$CATKIN_WS_BUILD/sampler/simple_chat/simple_chat" "$CATKIN_WS_MODEL_BIN" "$CATKIN_WS_TOKENIZER"
srun --mpi=pmix_v5 "$CATKIN_WS_BUILD/sampler/tp_parallel_simple_chat/tp_parallel_simple_chat" "$CATKIN_WS_MODEL_BIN" "$CATKIN_WS_TOKENIZER"







#/home/lyl/local/NVIDIA-Nsight-Compute-2025.4/ncu --target-processes all \
#    --kernel-name SingleDecodeWithKVCacheKernel \
#    --set full \
#    ./simple_chat ../../../../../models/bf16/model.bin ../../../../../models/llama-2-7b/tokenizer.model
