#include <gtest/gtest.h>
#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama2.h"
#include <cstdint>
#include <string>



TEST(test_model, encode) 
{

    std::string sentence = "Hello!";

    const char* checkpoint_path = "/home/ty/project-src/Dango_Infer/models/stories110M.bin"; 

    const char* tokenizer_path = "/home/ty/project-src/Dango_Infer/models/tokenizer.model";

    model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,checkpoint_path,base::DataType::kDataTypeFp32, false);

    model.init(0);

    auto tokens = model.encode(sentence);

    int32_t prompt_len = static_cast<int32_t>(tokens.size());

    EXPECT_EQ(tokens[0],1);

    EXPECT_EQ(tokens[1],15043);

    EXPECT_EQ(tokens[2],29991);

    EXPECT_EQ(prompt_len, static_cast<int32_t>(tokens.size()));

    EXPECT_GT(prompt_len, 0);
}


TEST(test_model, embedding) 
{

    std::string sentence = "Hello!";

    const char* checkpoint_path = "/home/ty/project-src/Dango_Infer/models/stories110M.bin"; 

    const char* tokenizer_path = "/home/ty/project-src/Dango_Infer/models/tokenizer.model";

    model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,checkpoint_path, base::DataType::kDataTypeFp32,false);

    model.init(0);

    auto tokens = model.encode(sentence);

    //model.embedding

    const auto& prompt_embedding = model.embedding(tokens);


}




