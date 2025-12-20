#include "model/llama2_prompt.h"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace model 
{

    PromptSession::PromptSession(Llama2ChatTemplate t) : t_(std::move(t)) {}

    void PromptSession::set_system(std::string system) { system_ = std::move(system); }

    const std::string& PromptSession::system() const { return system_; }

    void PromptSession::clear_history() 
    {
        users_.clear();
        assistants_.clear();
    }

    void PromptSession::reset_all() 
    {
        system_.clear();
        clear_history();
    }

    size_t PromptSession::history_size() const 
    {
        // 这里用“消息条数”概念：user + assistant 总数
        return users_.size() + assistants_.size();
    }

    std::string PromptSession::Trim(std::string_view s) 
    {
        size_t b = 0, e = s.size();
        while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
        return std::string(s.substr(b, e - b));
    }

    void PromptSession::validate_push_user() const 
    {
    // 允许：连续对话中 user 数要么等于 assistant 数，要么等于 assistant 数 + 1（未完成）
    // push_user 之前，必须满足 users_ == assistants_（上一轮已闭合）或者 users_==assistants_+? 实际上不能 > assistants_+1
        if (users_.size() > assistants_.size() + 1) 
            throw std::runtime_error("Invalid state: too many user turns without assistant replies.");
        
    // 更严格：如果 users_ == assistants_ + 1，说明已经有一个未回答的 user，还没生成 assistant，不能再 add_user
        if (users_.size() == assistants_.size() + 1) 
            throw std::runtime_error("Cannot add another user: the last user hasn't been answered by assistant yet.");
    
    }

    void PromptSession::validate_push_assistant() const 
    {
        // assistant 必须对应某个已存在但尚未配对的 user：assistants_ 必须比 users_ 少 1
        if (users_.empty()) 
            throw std::runtime_error("Cannot add assistant: no user message exists.");
    
        if (assistants_.size() >= users_.size())
            throw std::runtime_error("Cannot add assistant: no pending user to answer.");
    }

    void PromptSession::add_user(std::string user) 
    {
        validate_push_user();
        users_.push_back(std::move(user));
    }

    void PromptSession::add_assistant(std::string assistant) 
    {
        validate_push_assistant();
        assistants_.push_back(std::move(assistant));
    }

    void PromptSession::validate_build() const 
    {
        if (users_.empty()) 
            throw std::runtime_error("Cannot build prompt: no user message.");

        // build() 用来“等模型回答最后一个 user”，所以必须：assistants_ == users_-1
        if (assistants_.size() + 1 != users_.size()) 
            throw std::runtime_error("Cannot build prompt: history must end with an unanswered user (assistants = users - 1).");

    }

    std::string PromptSession::build() const 
    {
        validate_build();

        std::string out;
        out.reserve(4096);

        //第一次生成
        if(assistants_.empty())
        {
            out += t_.B_INST;
            out += " ";

            if (!system_.empty()) 
            {
                out += t_.B_SYS;
                out += system_;
                out += t_.E_SYS;
            }

            out += Trim(users_[0]);
            out += " ";
            out += t_.E_INST;

        // 如果有 assistant0（即 users_.size()>=2 时，assistants_.size()>=1），就拼上

            return out;
        }
        //out += " ";
        //out += Trim(assistants_[0]);
        //out += " "; // 注意：历史轮次末尾保留一个空格，便于与下一段 [INST] 分隔

        // 中间完整轮次 (u_i, a_i)，从 i=1 开始到 i=assistants_.size()-1
        //for (size_t i = 1; i < assistants_.size(); ++i) 
        //{
        //    out += t_.B_INST; out += " ";
        //    out += Trim(users_[i]);
        //    out += " ";
        //    out += t_.E_INST; out += " ";
        //    out += Trim(assistants_[i]);
        //    out += " ";
        //}

        // 最后一条 user（未回答）
        const size_t last_u = users_.size() - 1;
        out += t_.B_INST; out += " ";
        out += Trim(users_[last_u]);
        out += " ";
        out += t_.E_INST;

        return out;
    }

} // namespace llama2