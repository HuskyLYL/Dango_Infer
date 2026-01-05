#include<flashInfer/vec_dtypes.cuh>
#include<flashInfer/state.cuh>

#include <cstdint>
#include <glog/logging.h>
#include<flashInfer/pos_enc.cuh>
#include<flashInfer/vec_dtypes.cuh>
#include<flashInfer/cp_async.cuh>
#include<flashInfer/utils.cuh>
#include<flashInfer/state.cuh>
#include<tensor/tensor.h>
#include<base/base.h>





namespace flashinfer
{


    template <uint32_t vec_size, typename T>
    __device__ __inline__ void update_local_state(const T* smem, const float* s,
                                                   uint32_t compute_stage_idx,
                                                   state_t<vec_size>& st, uint32_t tx,uint32_t bdx, uint32_t tile_size) 
    {
        #pragma unroll
        for (uint32_t j = 0; j < tile_size; ++j) 
        {
            vec_t<float, vec_size> v_vec;
            v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
            #pragma unroll
            for (uint32_t i = 0; i < vec_size; ++i) 
            {
                st.o[i] = st.o[i] + s[j] * v_vec[i];
            }
        }
    }







    template<uint32_t vec_size>
    __device__ __inline__ void sync_state(state_t<vec_size>& st, float* smem, float* smem_md,
                                               const uint32_t tx, const uint32_t ty, const uint32_t tz,
                                                uint32_t bdx, uint32_t bdy, uint32_t bdz) 
    {
        if (bdz > 1) 
        {
            uint32_t head_dim = bdx * vec_size;

            st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    
            smem_md[(tz * bdy + ty) * 2] = st.m;
            smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
            __syncthreads();
            st.init();
            #pragma unroll
            for (uint32_t j = 0; j < bdz; ++j) 
            {
                float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
                vec_t<float, vec_size> oz;
                oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
                st.merge(oz, mz, dz);
            }
            

        }
    }








    template <uint32_t vec_size,  typename T>
    __device__ __forceinline__ void compute_qk(
        const uint32_t batch_idx, const T* smem,
        const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
        uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx, float* s,
        state_t<vec_size>& st, const uint32_t tx, const uint32_t ty, const uint32_t tz,uint32_t tile_size,uint32_t bdx) 
    {
        float m_prev = st.m;

        #pragma unroll
        for (uint32_t j = 0; j < tile_size; ++j) 
        {
            vec_t<float, vec_size> k_vec;
            // do not apply rotary embedding
            k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
            

            s[j] = 0.f;
            #pragma unroll
            for (uint32_t i = 0; i < vec_size; ++i) 
                s[j] += q_vec[i] * k_vec[i];

            #pragma unroll
            for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) 
                s[j] += math::shfl_xor_sync(s[j], offset);

            const uint32_t pos = kv_idx_base + tz * tile_size + j;

            const float sm_scale_log2 = math::log2e / sqrtf(static_cast<float>(bdx * vec_size));
            s[j] *= sm_scale_log2;

            bool mask = true;

            //bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
            //                               kv_head_idx);
            s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf;
            st.m = max(st.m, s[j]);
        }


        float o_scale = math::ptx_exp2(m_prev - st.m);
        st.d *= o_scale;

        #pragma unroll
        for (uint32_t j = 0; j < tile_size; ++j) 
        {
            s[j] = math::ptx_exp2(s[j] - st.m);
            st.d += s[j];
        }

        #pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) 
            st.o[i] = st.o[i] * o_scale;
        
    }


    uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) 
    {
        if (group_size == 8U) 
            if (sizeof_dtype == 1U) 
                return 256U;  // not enough registers for 512 threads
            else 
                return 512U;
        else 
            return 128U;
        
    }


    template <typename T, uint32_t num_stages_smem,uint32_t vec_size>
    __global__ void SingleDecodeWithKVCacheKernel(
        const T* q, const T* k, const T* v, T* o, const uint32_t q_stride_n, const uint32_t q_stride_h,
        const uint32_t kv_stride_n, const uint32_t kv_stride_h, const uint32_t kv_len,
        uint32_t num_qo_heads, uint32_t kv_chunk_size, uint32_t tile_size_per_bdx, 
        uint32_t bdx, uint32_t bdy, uint32_t bdz) 
    {
        uint32_t head_dim = bdx * vec_size;
        uint32_t kv_head_idx = blockIdx.y;
        uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
        uint32_t kv_chunk_idx = blockIdx.x;

        extern __shared__ uint8_t smem[];
        const uint32_t seq_len = kv_len;
        T* k_smem = (T*)smem;
        T* v_smem =
            (T*)(smem + num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim * sizeof(T));
        float* smem_md =
            (float*)(smem +
                     2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim * sizeof(T));

        uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
        vec_t<float, vec_size> q_vec;
        vec_t<float, vec_size> freq;

        //装载我们的Q了
        q_vec.cast_load(q + qo_head_idx * q_stride_h + tx * vec_size);

        __syncthreads();

        uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
        kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
        uint32_t chunk_end = chunk_start + kv_chunk_size;

        // preload k tiles and v tiles
        uint32_t producer_kv_idx_base = chunk_start;
        constexpr uint32_t vec_bits = sizeof(T) * vec_size * 8;
        #pragma unroll
        for (uint32_t iter = 0; iter < num_stages_smem; ++iter) 
        {
            for (uint32_t j = 0; j < tile_size_per_bdx; ++j) 
            {
                cp_async::pred_load<vec_bits, cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kNoFill>(
                    k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
                        tx * vec_size,
                    k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
                        kv_head_idx * kv_stride_h + tx * vec_size,
                    producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
            }
            cp_async::commit_group();
            for (uint32_t j = 0; j < tile_size_per_bdx; ++j) 
            {
                cp_async::pred_load<vec_bits, cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kFillZero>(
                    v_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
                        tx * vec_size,
                    v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
                        kv_head_idx * kv_stride_h + tx * vec_size,
                    producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
            }
            cp_async::commit_group();
            producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
        }

        // pipelining k/v tiles loading and state updating
        uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
        state_t<vec_size> st_local;
        //float s[bdy * tile_size_per_bdx];

        float* s = reinterpret_cast<float*>(smem_md + 2* bdy * bdz );

        #pragma unroll 2
        for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) 
        {
            // compute qk
            cp_async::wait_group<2 * num_stages_smem - 1>();
            __syncthreads();
            compute_qk<vec_size, T>(
                /*batch_idx=*/0,
                k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
                consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz, kv_chunk_size, qo_head_idx,
                kv_head_idx, s, st_local, tx, ty, tz, bdy * tile_size_per_bdx, bdx);
            __syncthreads();

            // load k
            for (uint32_t j = 0; j < tile_size_per_bdx; ++j) 
            {
                cp_async::pred_load<vec_bits, cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kNoFill>(
                    k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
                        tx * vec_size,
                    k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
                        kv_head_idx * kv_stride_h + tx * vec_size,
                    producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
            }
            cp_async::commit_group();

            // update m/d/o state
            cp_async::wait_group<2 * num_stages_smem - 1>();
            __syncthreads();
            update_local_state<vec_size>(
                v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx,
                st_local, tx,bdx, bdy * tile_size_per_bdx);
            __syncthreads();

            // load v
            for (uint32_t j = 0; j < tile_size_per_bdx; ++j) 
            {
                cp_async::pred_load<vec_bits, cp_async::PrefetchMode::kPrefetch, cp_async::SharedMemFillMode::kFillZero>(
                    v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
                        tx * vec_size,
                    v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
                        kv_head_idx * kv_stride_h + tx * vec_size,
                    producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
            }
            cp_async::commit_group();

            stage_idx = (stage_idx + 1) % num_stages_smem;
            producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
            consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
        }
        cp_async::wait_group<0>();
        __syncthreads();

        // sync local state of all warps inside a threadblock
        sync_state<vec_size>(st_local, reinterpret_cast<float*>(smem), smem_md, tx, ty, tz, bdx, bdy, bdz);
        #pragma unroll
        for (size_t i = 0; i < vec_size; ++i) 
        {
            float d_rcp = (st_local.m != -math::inf) ? math::ptx_rcp(st_local.d) : 0.f;
            //return output * d_rcp;
            st_local.o[i] = st_local.o[i]*d_rcp;
            //variant.OutputTransform(params, st_local.o[i], /*batch_idx=*/0, /*qo_idx=*/0,
            //                                        qo_head_idx, st_local.m, st_local.d, /*scale=*/1.0f);
        }

        st_local.o.cast_store(o + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);

    }


    template <typename T>
    cudaError_t SingleDecodeWithKVCacheDispatched(uint32_t head_dim,cudaStream_t stream,
        const uint32_t num_qo_heads,const uint32_t num_kv_heads ,const  uint32_t  kv_len,
        const T* q,const T* k,const T* v ,T* o,
        const uint32_t q_stride_n, const uint32_t q_stride_h, const uint32_t kv_stride_n,const uint32_t kv_stride_h ,uint32_t kv_chunk_size

    
    ) 
    {


        const uint32_t seq_len = kv_len;

        constexpr uint32_t vec_size = 16UL / sizeof(T);
        const uint32_t bdx = head_dim/ vec_size;
        auto compute_capacity = GetCudaComputeCapability();
        CHECK(bdx <= 32U);

        const uint32_t group_size = num_qo_heads / num_kv_heads;

        const uint32_t bdy = group_size;
        const uint32_t num_threads = std::max(get_heuristic_num_threads(group_size, sizeof(T)), bdx * bdy);

        const uint32_t bdz = num_threads / (bdx * bdy);
        const uint32_t tile_size_per_bdx = group_size == 1 ? (sizeof(T) == 1 ? 2U : 8U) : 1U;

        DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, 
        {
            const uint32_t smeme_size = 2U * NUM_STAGES_SMEM * bdy * tile_size_per_bdx * bdz * head_dim * sizeof(T) + 2U * bdy * bdz * sizeof(float)+bdy * tile_size_per_bdx*sizeof(float);

            dim3 nblks = dim3(1, num_kv_heads);
            dim3 nthrs = dim3(bdx, bdy, bdz);
            //uint32_t kv_chunk_size = seq_len;

            if(stream)
            
                SingleDecodeWithKVCacheKernel<T,NUM_STAGES_SMEM,vec_size><<<nblks,nthrs,smeme_size,stream>>>(
                    q, k, v, o, q_stride_n, q_stride_h,
                    kv_stride_n, kv_stride_h, kv_len,
                    num_qo_heads, kv_chunk_size, tile_size_per_bdx, 
                    bdx,  bdy, bdz); 
            else
            
                SingleDecodeWithKVCacheKernel<T,NUM_STAGES_SMEM,vec_size><<<nblks,nthrs,smeme_size>>>(
                    q, k, v, o, q_stride_n, q_stride_h,
                    kv_stride_n, kv_stride_h, kv_len,
                    num_qo_heads, kv_chunk_size, tile_size_per_bdx, 
                    bdx,  bdy, bdz); 
            

                

        });
        
        return cudaSuccess;
    }










    
}
