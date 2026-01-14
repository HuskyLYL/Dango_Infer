#include "op/parallel_embedding.h"
#include "nccl/collective.h"
#include "kernel/cuda/embedding.cuh"
#include "kernel/kernels_interface.h"
#include "op/layer.h"
#include "base/base.h"
namespace op 
{
  ParallelEmbeddingLayer::ParallelEmbeddingLayer():LayerParam( "Embedding") 
  {
      reset_weight_size(1);
      reset_input_size(1);
      reset_output_size(1);
  }


  base::Status ParallelEmbeddingLayer::check() const 
  {

    

    const auto& input_tensor = get_input(0);

    const auto& weight_tensor = get_weight(0);

    const auto& output_tensor = get_output(0);



    int32_t input_size = input_tensor.size();
    int32_t weight_dim =  weight_tensor.get_dim(1);
    base::DataType data_type = weight_tensor.data_type();

    switch (data_type) 
    {
      case base::DataType::kDataTypeFp32:
        if (weight_dim % 4 != 0)
          return base::error::InvalidArgument("Embedding fp32 input size must be a multiple of 4.");
        break;
      case base::DataType::kDataTypeBf16:
        if (weight_dim % 8 != 0)
          return base::error::InvalidArgument("Embedding bf16 input size must be a multiple of 8.");
        break;
      default:
        return base::error::InvalidArgument("Unsupported data type in the embedding layer.");
    }

    base::deviceId device_id = weight_tensor.getDeviceId();

    

    CHECK_GT(input_size, 0);

    
  
    // input tokens must be int32
    if (input_tensor.data_type() != base::DataType::kDataTypeInt32)
      return base::error::InvalidArgument("Embedding input tensor must be int32 tokens.");

    base::Status status = check_tensor_with_dim(input_tensor, device_id, base::DataType::kDataTypeInt32, input_size);
    if (!status) 
    {
      LOG(ERROR) << "The input tensor error in the embedding layer.";
      return status;
    }



    status = check_tensor_with_dim(output_tensor, device_id, output_tensor.data_type(),input_size,weight_dim);
    if (!status) 
    {
      LOG(ERROR) << "The output tensor error in the embedding layer.";
      return status;
    }

  
    base::setDevice(device_id);

    return base::error::Success();
  }

  base::Status ParallelEmbeddingLayer::forward(cudaStream_t stream) 
  {

    base::Status status = check();

    if (!status) 
      return status;


    auto input_tensor = get_input(0);
    auto weight_tensor = get_weight(0);
    auto output_tensor =  get_output(0);
    base::DataType data_type = weight_tensor.data_type();
    if (data_type == base::DataType::kDataTypeFp32)
      f32x4_kernel_cu::get_parallel_embedding_kernel()(input_tensor, weight_tensor, output_tensor, stream ? stream : nullptr);
    else if (data_type == base::DataType::kDataTypeBf16)
      bf16x8_kernel_cu::get_parallel_embedding_kernel()(input_tensor, weight_tensor, output_tensor, stream ? stream : nullptr);

    if (stream)
      CUDACHECK(cudaStreamSynchronize(stream));

    nccl::TensorAllReduce(output_tensor);

    return base::StatusCode::kSuccess;
  }

  base::Status ParallelEmbeddingLayer::set_weight(int32_t idx, const tensor::Tensor& weight)
  {
    CHECK_EQ(idx, 0);
    CHECK_GT(nccl::G_MPI_SIZE, 0);

    const int32_t rows = weight.get_dim(0);
    const int32_t cols = weight.get_dim(1);
    CHECK_EQ(rows % nccl::G_MPI_SIZE, 0)
        << "Embedding vocab rows must be divisible by world size";

    const int32_t rows_per_rank = rows / nccl::G_MPI_SIZE;
    const size_t elem_bytes = base::DataTypeSize(weight.data_type());
    const size_t offset_bytes = static_cast<size_t>(rows_per_rank) * static_cast<size_t>(cols) * elem_bytes
        * static_cast<size_t>(nccl::G_MPI_RANK);

    // Create a local tensor view of this rank's vocab slice.
    void* offset_ptr = static_cast<char*>(const_cast<void*>(weight.get_buffer()->ptr())) + offset_bytes;
    tensor::Tensor local_weight({rows_per_rank, cols}, weight.getDeviceId(), weight.data_type(), offset_ptr);

    return LayerParam::set_weight(idx, local_weight);
  }

  base::Status ParallelEmbeddingLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
    const void* weight_ptr, base::deviceId device_id, base::DataType weight_data_type)
  {
    CHECK_EQ(idx, 0);
    CHECK_GT(nccl::G_MPI_SIZE, 0);
    CHECK_EQ(dims.size(), 2u);

    const int32_t rows = dims[0];
    const int32_t cols = dims[1];
    CHECK_EQ(rows % nccl::G_MPI_SIZE, 0)
        << "Embedding vocab rows must be divisible by world size";

    const int32_t rows_per_rank = rows / nccl::G_MPI_SIZE;
    const size_t elem_bytes = base::DataTypeSize(weight_data_type);
    const size_t offset_bytes = static_cast<size_t>(rows_per_rank) * static_cast<size_t>(cols) * elem_bytes
        * static_cast<size_t>(nccl::G_MPI_RANK);

    const char* src = static_cast<const char*>(weight_ptr) + offset_bytes;

    tensor::Tensor local_weight({rows_per_rank, cols}, device_id, weight_data_type, nullptr);

    // Copy the slice into local_weight buffer.
    void* dst_ptr = local_weight.get_buffer()->ptr();
    const size_t bytes_to_copy = static_cast<size_t>(rows_per_rank) * static_cast<size_t>(cols) * elem_bytes;

    if (device_id == base::CPUID)
        memcpy(dst_ptr, src, bytes_to_copy);
    else
        CUDACHECK(cudaMemcpy(dst_ptr, src, bytes_to_copy, cudaMemcpyDefault));

    return LayerParam::set_weight(idx, local_weight);
  }
}  // namespace op
