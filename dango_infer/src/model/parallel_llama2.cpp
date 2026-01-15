#include "model/parallel_llama2.h"
#include "nccl/base.h"

namespace model
{
    ParallelLLama2Model::ParallelLLama2Model(base::TokenizerType tokenizer_type,
        std::string token_path, std::string model_path,
        base::DataType data_type, bool is_quant_model)
        : LLama2Model(tokenizer_type, std::move(token_path), std::move(model_path),
            data_type, is_quant_model)
    {
        // Placeholder: parallel-specific initialization can be added later.
    }

    void ParallelLLama2Model::init_mem()
    {
        tensor::Tensor input_tokens(1, base::CPUID, base::DataType::kDataTypeInt32);

        tensor::Tensor input_embeddings(1,config_->dim_,device_id_,data_type_);

        // Partition sin/cos cache along seq_len per rank.
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        CHECK_EQ(config_->head_size_ % nccl::G_MPI_SIZE, 0)
            << "head_size must be divisible by world size in ParallelLLama2Model.";

        const int32_t seq_per_rank = config_->seq_len_ / nccl::G_MPI_SIZE;
        tensor::Tensor sin_cache(config_->head_size_ * seq_per_rank,device_id_,data_type_);

        tensor::Tensor cos_cache(config_->head_size_ * seq_per_rank,device_id_,data_type_);

        CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
        CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

        CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
        CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

        tensor::Tensor rms_output(config_->dim_,device_id_,data_type_);
        CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
        CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
        CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
        CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

        tensor::Tensor w1_output(config_->hidden_dim_, device_id_, data_type_);
        tensor::Tensor w3_output(config_->hidden_dim_, device_id_, data_type_);

        CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
        CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

        // kv cache
        tensor::Tensor key_cache(config_->layer_num_, config_->seq_len_,config_->kv_dim_, device_id_,data_type_);

        tensor::Tensor value_cache(config_->layer_num_, config_->seq_len_,config_->kv_dim_, device_id_,data_type_);

        CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
        CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

        // Wq query output
        tensor::Tensor query(config_->dim_,device_id_,data_type_);
        CHECK(insert_buffer(ModelBufferType::kQuery, query));

        // Pos tensor
        tensor::Tensor pos_tensor(1, base::CPUID,base::DataType::kDataTypeInt32);
        CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

        // Attention output
        tensor::Tensor attn(config_->head_num_, config_->seq_len_, device_id_,data_type_);
        CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
        CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

        // final forward output
        tensor::Tensor forward_output(config_->vocab_size_, device_id_, data_type_);

        tensor::Tensor forward_output_cpu(config_->vocab_size_, base::CPUID, data_type_);


        CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));


        CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
    }

    void ParallelLLama2Model::create_param_layers()
    {
        CHECK(!is_quant_model_);
        CHECK(llama_layers_ != nullptr);

        llama_layers_->embedding_layer_ = std::make_shared<op::ParallelEmbeddingLayer>();



      const void* weight_embedding = raw_model_data_->weight(0);

      llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                            weight_embedding, base::CPUID,data_type_);

      llama_layers_->embedding_layer_->to_device(device_id_);






      // create all matmul layer
      int32_t dim = config_->dim_;
      size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
      // create weight matrix for query
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto wq = std::make_shared<op::RowMatmulLayer>(false);
          wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          wq->to_device(device_id_);
          llama_layers_->wq_layers_.push_back(wq);
          
          pos += dim * dim;
      }

      // create weight matrix for key
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto wk = std::make_shared<op::RowMatmulLayer>(false);
          wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          wk->to_device(device_id_);
          llama_layers_->wk_layers_.push_back(wk);
          pos += config_->kv_dim_ * dim;
      }

      // create weight matrix for value
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto wv = std::make_shared<op::RowMatmulLayer>(false);
          wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          wv->to_device(device_id_);
          llama_layers_->wv_layers_.push_back(wv);
          pos += config_->kv_dim_ * dim;
      }

      // create weight matrix for output
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto wo = std::make_shared<op::ColomParallelMatmulLayer>(true);
          wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          wo->to_device(device_id_);
          llama_layers_->wo_layers_.push_back(wo);
          pos += dim * dim;
      }

      // skip ffn rmsnorm
      pos += config_->layer_num_ * dim;

      // w1 layers
      int32_t hidden_dim = config_->hidden_dim_;
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
            LOG(INFO)<<"#######################################"<<endl;
          auto w1 = std::make_shared<op::RowMatmulLayer>(false);
          w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos),base::CPUID,data_type_);
          w1->to_device(device_id_);
          llama_layers_->w1_layers_.push_back(w1);
          pos += dim * hidden_dim;
      }

      // w2 layers
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto w2 = std::make_shared<op::ColomParallelMatmulLayer>(true);
          w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          w2->to_device(device_id_);
          llama_layers_->w2_layers_.push_back(w2);
          pos += dim * hidden_dim;
      }

      // w3 layers
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto w3 = std::make_shared<op::RowMatmulLayer>(false);
          w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          w3->to_device(device_id_);
          llama_layers_->w3_layers_.push_back(w3);
          pos += dim * hidden_dim;
      }

      // skip final rms weight
      pos += dim;
      // skip freqs_cos and freqs_sin weight
      pos += config_->seq_len_ * config_->head_size_;

      llama_layers_->cls_layer_ =std::make_shared<op::RowMatmulLayer>(true);
  
      if (config_->is_shared_weight_) 
      // using token embedding weight
          llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},this->raw_model_data_->weight(0), base::CPUID,data_type_);
      else 
          llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},this->raw_model_data_->weight(pos), base::CPUID,data_type_);

      llama_layers_->cls_layer_->to_device(device_id_);
      // create rmsnorm layer
      size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          std::shared_ptr<op::RmsNormLayer> rms_norm_layer =std::make_shared<op::RmsNormLayer>();
          const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
          rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, base::CPUID,data_type_);
          rms_norm_layer->to_device(device_id_);
          llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
          rmsnorm_pos += config_->dim_;
      }

      // skip attention.wq attention.wk attention.wv attention.wo
      rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
      rmsnorm_pos += config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
      rmsnorm_pos += config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
      rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          std::shared_ptr<op::RmsNormLayer> rms_norm_layer = std::make_shared<op::RmsNormLayer>();
          const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
          rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, base::CPUID,data_type_);
          rms_norm_layer->to_device(device_id_);
          llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
          rmsnorm_pos += config_->dim_;
      }

      // skip ffn.w1 ffn.w2 ffn.w3
      rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
      rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
      rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

      std::shared_ptr<op::RmsNormLayer> rms_final_layer = std::make_shared<op::RmsNormLayer>();

      const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
      rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, base::CPUID,data_type_);
      rms_final_layer->to_device(device_id_);
      llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
  }

    void ParallelLLama2Model::create_nonparam_layers()
    {
        CHECK(llama_layers_ != nullptr);

        llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(config_->dim_, config_->kv_dim_, config_->head_size_);

        //pos 是计算cacheo算到多少层    但是cache会预留一个大的空间,所以这里不需要担心
        llama_layers_->mha_layer_ = std::make_shared<op::Paralle_MultiHeadAttenton>(0, config_->kv_mul_, config_->kv_dim_, 
        config_->seq_len_, config_->head_num_,config_->head_size_);

        llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>();

        llama_layers_->swiglu_layer_ = std::make_shared<op::ParallelSwiGLULayer>();
    }
}  // namespace model
