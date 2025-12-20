#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace model 
{

    struct Llama2ChatTemplate 
    {
        std::string B_INST = "[INST]";
        std::string E_INST = "[/INST]";
        std::string B_SYS  = "<<SYS>>\n";
        std::string E_SYS  = "\n<</SYS>>\n\n";
    };

    // 一个“轻量 session”，只管拼 prompt 字符串
    class PromptSession 
    {
    public:
        explicit PromptSession(Llama2ChatTemplate t = {});

        // 设置 system（可选）。为空表示不使用 system block
        void set_system(std::string system);
        const std::string& system() const;

        // 追加一轮 user / assistant
        void add_user(std::string user);
        void add_assistant(std::string assistant);

        // 清空历史（system 保留/清空由参数决定）
        void clear_history();
        void reset_all();

        // 历史条数（不含 system）
        size_t history_size() const;

        // 生成最终 prompt 字符串：
        // (u,a) pairs -> "[INST] u [/INST] a␠"
        // last user   -> "[INST] u_last [/INST]"
        //
        // 要求：history 必须以 user 结尾（等待模型生成 assistant）
        std::string build() const;


    private:
        Llama2ChatTemplate t_;
        std::string system_;                 // optional
        std::vector<std::string> users_;     // user turns
        std::vector<std::string> assistants_;// assistant turns (may be <= users_-1)

        static std::string Trim(std::string_view s);

        // 保证 add_user/add_assistant 的调用顺序合理
        void validate_push_user() const;
        void validate_push_assistant() const;

        // build 前做校验：至少有一个 user，且 assistants_ == users_-1
        void validate_build() const;
};

} // namespace model