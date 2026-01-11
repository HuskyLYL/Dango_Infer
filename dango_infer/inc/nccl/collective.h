#ifndef DANGO_INCLUDE_NCCL_COLLECTIVE_H_
#define DANGO_INCLUDE_NCCL_COLLECTIVE_H_
#include "nccl/base.h"
#include "tensor/tensor.h"



//nccl 一些全局ID
//初始化配置





namespace nccl
{
    
    void TensorAllGather(const tensor::Tensor& input);
    void TensorAllReduce(const tensor::Tensor& input);







}



#endif  
