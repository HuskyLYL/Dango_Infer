#include"nccl/collective.h"



namespace nccl
{
    
    void TensorAllGather(const tensor::Tensor& input)
    {
        ncclDataType_t dtype;
        switch (input.data_type()) 
        {
            case base::DataType::kDataTypeFp32:
                dtype = ncclFloat32;
                break;
            case base::DataType::kDataTypeBf16:
                dtype = ncclBfloat16;
                break;
            default:
                LOG(FATAL) << "TensorAllGather only supports fp32/bf16, got "
                           << static_cast<int>(input.data_type());
                abort();
        }

        auto buffer = input.get_buffer();
        CHECK(buffer != nullptr);


        const size_t total_elems = input.size();
        CHECK_GT(total_elems, 0);
        CHECK_GT(G_MPI_SIZE, 0);
        CHECK_EQ(total_elems % static_cast<size_t>(G_MPI_SIZE), 0u)
            << "TensorAllGather expects tensor size divisible by world size";
        
        //计算当前同步的偏移值
        const size_t elems_per_rank = total_elems / static_cast<size_t>(G_MPI_SIZE);
        const size_t elem_bytes = base::DataTypeSize(input.data_type());
        const size_t offset_bytes =
            elems_per_rank * elem_bytes * static_cast<size_t>(G_MPI_RANK);

        void* recv_ptr = buffer->ptr();
        const void* send_ptr = static_cast<const char*>(buffer->ptr()) + offset_bytes;

        NCCLCHECK(ncclAllGather(send_ptr, recv_ptr, elems_per_rank, dtype, COMM, 0));
        CUDACHECK(cudaDeviceSynchronize());


    }

    void TensorAllReduce(const tensor::Tensor& input)
    {
        ncclDataType_t dtype;
        switch (input.data_type())
        {
            case base::DataType::kDataTypeFp32:
                dtype = ncclFloat32;
                break;
            case base::DataType::kDataTypeBf16:
                dtype = ncclBfloat16;
                break;
            default:
                LOG(FATAL) << "TensorAllReduce only supports fp32/bf16, got "
                           << static_cast<int>(input.data_type());
                abort();
        }

        auto buffer = input.get_buffer();
        CHECK(buffer != nullptr);

        const size_t total_elems = input.size();
        CHECK_GT(total_elems, 0);

        //原地进行 all-reduce：在同一个缓冲区内对各个 rank 的数据求和
        void* data = buffer->ptr();
        NCCLCHECK(ncclAllReduce(data, data, total_elems, dtype, ncclSum, COMM, 0));
        CUDACHECK(cudaDeviceSynchronize());
    }





}
