#include<flashInfer/vec_dtypes.cuh>
#include<flashInfer/state.cuh>

namespace flashinfer
{

    template<uint32_t vec_size>
    __device__ __inline__ void sync_state(state_t<vec_size>& st, float* smem, float* smem_md,
                                               const uint32_t tx, const uint32_t ty, const uint32_t tz,
                                                uint32_t bdx, uint32_t bdy, uint32_t bdz) 
    {
        if constexpr (bdz > 1) 
        {
            constexpr uint32_t head_dim = bdx * vec_size;

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

    
            s[j] *= variant.sm_scale_log2;

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


    const uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) 
    {
        if (group_size == 8U) 
            if (sizeof_dtype == 1U) 
                return 256U;  // not enough registers for 512 threads
            else 
                return 512U;
        else 
            return 128U;
        
    }










    
}
