#include<kernel/flashInfer/single_decode_with_kv_cache.cuh>
#include<flashInfer/decode.cuh>
#include <cuda_bf16.h>
#include <glog/logging.h>
namespace flashinfer
{

  


    void single_decode_with_kv_cache(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
        int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
        const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
        const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
        cudaStream_t stream)
    {


        unsigned int num_qo_heads = head_num;
        unsigned int head_dim_qk = head_size;
        unsigned int head_dim_vo = head_size;
        unsigned int kv_len, num_kv_heads;
        kv_len = pos+1;
        num_kv_heads = head_num/kv_mul;

        const uint32_t q_stride_n = num_qo_heads * head_dim_qk;
        const uint32_t q_stride_h = head_dim_qk;
        const uint32_t kv_stride_n = num_kv_heads * head_dim_vo;
        const uint32_t kv_stride_h = head_dim_vo;
        uint32_t kv_chunk_size = 0;
        cudaError_t status;
        

    
    
        switch (query_tensor.data_type())
        {
            case base::DataType::kDataTypeFp32: 
            {
                const float* q = query_tensor.ptr<float>();
                const float* k = key_cache_tensor.ptr<float>(layer_index * seq_len * kv_dim);
                const float* v = value_cache_tensor.ptr<float>(layer_index * seq_len * kv_dim);
                float* o = const_cast<float*>(mha_out.ptr<float>());
                status = SingleDecodeWithKVCacheDispatched<float>(head_size,stream,
                    num_qo_heads,num_kv_heads ,kv_len,
                    q,k,v,o,
                    q_stride_n,q_stride_h,kv_stride_n,kv_stride_h 
                );
                break;
            }

            case base::DataType::kDataTypeBf16: 
            {
                const __nv_bfloat16* q = query_tensor.ptr<__nv_bfloat16>();
                const __nv_bfloat16* k = key_cache_tensor.ptr<__nv_bfloat16>(layer_index * seq_len * kv_dim);
                const __nv_bfloat16* v = value_cache_tensor.ptr<__nv_bfloat16>(layer_index * seq_len * kv_dim);
                __nv_bfloat16* o =  const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());
                status = SingleDecodeWithKVCacheDispatched<__nv_bfloat16>(head_size,stream,
                    num_qo_heads,num_kv_heads ,kv_len,
                    q,k,v,o,
                    q_stride_n,q_stride_h,kv_stride_n,kv_stride_h 
                );

                break;
            }
        
            default:
            {
                LOG(ERROR) << "single_decode_with_kv_cache: unsupported data type "
                           << static_cast<int>(query_tensor.data_type());
                status = cudaErrorInvalidValue;
                break;
            }
        }

     
        
        CHECK(status == cudaSuccess);   
    }
}
