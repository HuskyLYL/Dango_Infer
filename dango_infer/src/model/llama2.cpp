#include "model/llama2.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "kernel/cuda/rope_kernel.cuh"
#include "base/tick.h"
namespace model 
{

 

    LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,std::string model_path, base::DataType data_type,bool is_quant_model)
    : Model(data_type,tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),std::move(model_path), is_quant_model) { }



    base::Status LLama2Model::init(base::deviceId device_id) 
    {
        using namespace base;
    
        if (token_path_.empty()) 
            return error::PathNotValid(token_path_);
    
        if (device_id == base::CPUID ) 
            return error::InternalError("The cpu device do not support .");
    

        //这里先初始化了device_id_
        device_id_ = device_id;

        if(device_id_ != base::CPUID)

            base::setDevice(device_id_);

        Status read_status = gen_model_from_file();

        if (!read_status) 
        
            return read_status;
    

            
        init_mem();

        if (data_type_ == base::DataType::kDataTypeFp32)
        {
            base_kernel_cu::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                get_buffer(ModelBufferType::kSinCache),
                get_buffer(ModelBufferType::kCosCache));
        }
        else if (data_type_ == base::DataType::kDataTypeBf16)
        {
            bf16_kernel_cu::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                get_buffer(ModelBufferType::kSinCache),
                get_buffer(ModelBufferType::kCosCache));
        }
        else
        {
            return error::InvalidArgument("Unsupported data type for sin/cos cache.");
        }

        // ensure cache build finishes before later kernels
        cudaDeviceSynchronize();



        sampler_ = std::make_unique<sampler::ArgmaxSampler>();
        return error::Success();
    }



  base::Status LLama2Model::forward(const tensor::Tensor& input, 
    const tensor::Tensor& pos_tensor,int& next) const 
  {
    if (input.is_empty()) 
      return base::error::InvalidArgument("The input tensor is empty.");

    if (device_id_ == base::CPUID && is_quant_model_) 
      return base::error::InternalError("Unsupported int8 quant in the cpu device");

    if(base::g_enable_debug_log)
    input.print("Begin:");


    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) 
    {
      if(base::g_enable_debug_log)
        LOG(INFO)<<"layer"<<layer_idx<<":\n";
      attention_rms(layer_idx, input);
      // attention (wq wk wv @ input)
      attention_qkv(layer_idx, pos_tensor);
      // multi-head attention
      attention_mha(layer_idx, pos_tensor);
      // feed forward
      feed_forward(layer_idx, input);
    }
    cls_logits(input);
    return base::error::Success();
  }

  void LLama2Model::create_nonparam_layers() 
  {
      //这里实际上都是在初始化一下我们带参数的层，只需要模型的
      CHECK(llama_layers_ != nullptr);

      llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(config_->dim_, config_->kv_dim_, config_->head_size_);

      //pos 是计算cacheo算到多少层    但是cache会预留一个大的空间,所以这里不需要担心
      llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(0, config_->kv_mul_, config_->kv_dim_, 
      config_->seq_len_, config_->head_num_,config_->head_size_);

      llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>();

      llama_layers_->swiglu_layer_ = std::make_shared<op::SwiGLULayer>();
  }



  void LLama2Model::create_param_layers() 
  {
      CHECK(!is_quant_model_);
      CHECK(llama_layers_ != nullptr);

      llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>();

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
          auto wq = std::make_shared<op::MatmulLayer>();
          wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          wq->to_device(device_id_);
          llama_layers_->wq_layers_.push_back(wq);
          
          pos += dim * dim;
      }

      // create weight matrix for key
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto wk = std::make_shared<op::MatmulLayer>();
          wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          wk->to_device(device_id_);
          llama_layers_->wk_layers_.push_back(wk);
          pos += config_->kv_dim_ * dim;
      }

      // create weight matrix for value
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto wv = std::make_shared<op::MatmulLayer>();
          wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          wv->to_device(device_id_);
          llama_layers_->wv_layers_.push_back(wv);
          pos += config_->kv_dim_ * dim;
      }

      // create weight matrix for output
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto wo = std::make_shared<op::MatmulLayer>();
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
          auto w1 = std::make_shared<op::MatmulLayer>();
          w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos),base::CPUID,data_type_);
          w1->to_device(device_id_);
          llama_layers_->w1_layers_.push_back(w1);
          pos += dim * hidden_dim;
      }

      // w2 layers
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto w2 = std::make_shared<op::MatmulLayer>();
          w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          w2->to_device(device_id_);
          llama_layers_->w2_layers_.push_back(w2);
          pos += dim * hidden_dim;
      }

      // w3 layers
      for (int32_t i = 0; i < config_->layer_num_; ++i) 
      {
          auto w3 = std::make_shared<op::MatmulLayer>();
          w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), base::CPUID,data_type_);
          w3->to_device(device_id_);
          llama_layers_->w3_layers_.push_back(w3);
          pos += dim * hidden_dim;
      }

      // skip final rms weight
      pos += dim;
      // skip freqs_cos and freqs_sin weight
      pos += config_->seq_len_ * config_->head_size_;

      llama_layers_->cls_layer_ =std::make_shared<op::MatmulLayer>();
  
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

  void LLama2Model::init_mem() 
  {
    // token ids must stay int32 regardless of model precision to avoid buffer underruns
    tensor::Tensor input_tokens(1, base::CPUID, base::DataType::kDataTypeInt32);

    tensor::Tensor input_embeddings(1,config_->dim_,device_id_,data_type_);

    tensor::Tensor sin_cache(config_->head_size_ * config_->seq_len_,device_id_,data_type_);

    tensor::Tensor cos_cache(config_->head_size_ * config_->seq_len_,device_id_,data_type_);

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
    tensor::Tensor key_cache(config_->layer_num_, config_->seq_len_,config_->kv_dim_, 0,data_type_);

    tensor::Tensor value_cache(config_->layer_num_, config_->seq_len_,config_->kv_dim_, 0,data_type_);

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

  base::Status LLama2Model::create_layers() 
  {
    using namespace base;
    if (!llama_layers_) 
        llama_layers_ = std::make_unique<LLama2Layers>();

    //在这里，是加载我们有权重的模型层
    if (!is_quant_model_) 
        create_param_layers();
    else 
        return error::InternalError("Create the embedding layer for the llama model failed!");

    create_nonparam_layers();

    if (!llama_layers_->embedding_layer_) 

      return error::InternalError("Create the embedding layer for the llama model failed!");


    if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) 

      return error::InternalError("Create the rmsnorm layers for the llama model failed!");


    if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
        llama_layers_->wk_layers_.size() != config_->layer_num_ ||
        llama_layers_->wv_layers_.size() != config_->layer_num_ ||
        llama_layers_->wo_layers_.size() != config_->layer_num_) 
    {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }


    //这里都是在检查我们的模型层有没有被正确的初始化罢了
    for (int32_t i = 0; i < config_->layer_num_; ++i) 
    {
      if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
          !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i)) {
        return error::InternalError(
            "Create the matmul layer in the attention and ffn attention layers for "
            "the llama model "
            "failed.");
      }
    }

    if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
        llama_layers_->w2_layers_.size() != config_->layer_num_ ||
        llama_layers_->w3_layers_.size() != config_->layer_num_) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) 
      if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
          !llama_layers_->w3_layers_.at(i)) 
        return error::InternalError(
            "Create the matmul layer in the feedforward layers for the llama model "
            "failed.");
    if (!llama_layers_->rope_layer_) 
      return error::InternalError("Create the rope layer for the llama model failed!");


    if (!llama_layers_->add_layer_) 
      return error::InternalError("Create the add layer for the llama model failed!");


    if (!llama_layers_->mha_layer_) 
      return error::InternalError("Create the mha layer for the llama model failed!");


    if (!llama_layers_->swiglu_layer_) 
      return error::InternalError("Create the SwiGLU layer for the llama model failed!");

    return error::Success();
  }

  //传入的实际上是CPU上的数据
  //拷贝了一个tensor的对象,然后将数据转移到CPU之上
  op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) const 
  {

    //  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    //tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);

    //这个是在CPU上的
    auto input_tokens = get_buffer(ModelBufferType::kInputTokens);

    auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);


    if (input_tokens.size() != tokens.size()) 
    {
      //这里的dim_应该就是我们的正常的分词器的大小
      //tensor 是管理我们的更底层的Buffer类的
      //Buffer受到智能指针的管理，离开作用域即被释放
      input_tokens.reshape({static_cast<int32_t>(tokens.size())});

      input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});


    }






    for (int32_t i = 0; i < tokens.size(); ++i) 
    {
      //这里读取指针的时候，我们用了reinterpret_cast
      //T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
      //index这里本质上是调用的强制指针转换类型去
      input_tokens.index<int32_t>(i) = tokens.at(i);
    }

  
    tensor::Tensor input_tokens_cu = input_tokens;
    input_tokens_cu.to_device(device_id_);

    LOG_IF(FATAL, !llama_layers_->embedding_layer_) << "The embedding layer in the llama2 model is null pointer.";
    STATUS_CHECK( llama_layers_->embedding_layer_->forward(input_tokens_cu, input_embeddings));
    op::EmbeddingOutput output(input_tokens, input_embeddings);

    return output;
  }

  void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const 
  {
    CHECK(llama_layers_ != nullptr);
    // attn rmsnorm
    tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);

    if (!rmsnorm_layer) 
      LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
    
    STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
    if(base::g_enable_debug_log)
      rmsnorm_output.print("rmsnorm_output:");

  }

  void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const 
  {
    CHECK(llama_layers_ != nullptr);
    // kv cache
    tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
    int32_t pos = pos_tensor.index<int32_t>(0);
    // wq wk wv @ input


    const auto& [key, val] = slice_kv_cache(layer_idx, pos);
    // query
    const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
    CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

    auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

    // key
    const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
    CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
    // value
    const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
    CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

    // rope
    CHECK_NE(llama_layers_->rope_layer_, nullptr)
        << "The RoPE layer in the attention block is null pointer.";
    STATUS_CHECK(llama_layers_->rope_layer_->forward(
        query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
  }

  base::Status LLama2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                    bool is_prompt, int& next) const 
  {
    auto status = forward(input, pos_tensor, next);
    if (!status) 
      return status;
    next = post_processing(pos_tensor, is_prompt);
    return base::error::Success();
  }

  void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const 
  {
    CHECK(llama_layers_ != nullptr);
    // mha
    tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    // VAL = [val1,val2,...val t]
    // output @ VAL = 最终的结果
    tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

    tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

    const auto& mha_layer = llama_layers_->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
    int pos = pos_tensor.index<int32_t>(0);


    //智能指针的安全转换类型
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

    // wo @ attention output
    tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
    const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
    CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
    STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
    if(base::g_enable_debug_log)
      attn_output.print("attn_output:");
  }

  void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const 
  {
    CHECK(llama_layers_ != nullptr);
    // residual add
    CHECK_NE(llama_layers_->add_layer_, nullptr)<< "The add layer in the feedforward block is null pointer";
    STATUS_CHECK(llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));
    if(base::g_enable_debug_log)
      input.print("add_layer:");

    // ffn rmsnorm
    tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
    const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
    CHECK_NE(ffn_rmsnorm, nullptr) << "The final rmsnorm layer in the feedforward block is null pointer";
    STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));
    if(base::g_enable_debug_log)
      ffn_norm_output.print("rmsnorm_layers");

    // w1
    tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
    const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
    CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    if(base::g_enable_debug_log)
      w1_output.print("w1");

    // w3
    tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
    const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
    CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));
    if(base::g_enable_debug_log)
      w3_ouput.print("w3");

    // SwiGLU
    CHECK_NE(llama_layers_->swiglu_layer_, nullptr) << "The swiglu layer in the feedforward block is null pointer";
    STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));
    if(base::g_enable_debug_log)
      w1_output.print("swiglu_layer:");

    // w2
    tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
    const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
    CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
    STATUS_CHECK(w2_layer->forward(w1_output, w2_output));
    if(base::g_enable_debug_log)
      w2_output.print("w2");

    // residual add
    CHECK_NE(llama_layers_->add_layer_, nullptr) << "The add layer in the feedforward block is null pointer";
    STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
    if(base::g_enable_debug_log)
      input.print("feed_forward:");
  }

  void LLama2Model::cls_logits(const tensor::Tensor& input) const 
  {
    CHECK(llama_layers_ != nullptr);
    const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
    CHECK_NE(norm, nullptr);
    STATUS_CHECK(norm->forward(input, input));

    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    CHECK_NE(llama_layers_->cls_layer_, nullptr);
    STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
    if(base::g_enable_debug_log)
      forward_output.print("cls_logits");
  }

  int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt,cudaStream_t stream) const 
  {
    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    int32_t next = 0;
    if (is_prompt) 
      next = -1;
    else 
      next = static_cast<int32_t>(sampler_->sample(forward_output, stream));

    if(base::g_enable_debug_log)
    {
        
        int32_t BBQ = static_cast<int32_t>(sampler_->sample(forward_output, stream));
        LOG(INFO)<<"next"<<BBQ<<"\n";

    }
      
    return next;
  }

}  // namespace model
