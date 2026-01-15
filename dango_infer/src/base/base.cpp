#include "base/base.h"

#include <string>
namespace base 
{
  // Default false: enable to turn on debug-only outputs.
  bool g_enable_debug_log = false;

  //状态编码的构造类
  Status::Status(int code, std::string err_message)
    : code_(code), message_(std::move(err_message)) {}

  //status符号构造
  Status& Status::operator=(int code) 
  {
    code_ = code;
    return *this;
  }

  bool Status::operator==(int code) const 
  {
    if (code_ == code) 
      return true;
    else 
      return false;
  }

  bool Status::operator!=(int code) const 
  {
    if (code_ != code) 
      return true;
    else 
      return false;
  }

  //status状态重定义
  Status::operator int() const { return code_; }

  Status::operator bool() const { return code_ == kSuccess; }

  //获取修改status信息

  int32_t Status::get_err_code() const { return code_; }

  const std::string& Status::get_err_msg() const { return message_; }

  void Status::set_err_msg(const std::string& err_msg) { message_ = err_msg; }





  //打包信息DEBUG信息编码发送
  namespace error 
  {
    Status Success(const std::string& err_msg) { return Status{kSuccess, err_msg}; }

    Status Running(const std::string& err_msg) { return Status{kRunning, err_msg}; }

    Status Failed(const std::string& err_msg) { return Status{kFailed, err_msg}; }

    Status FunctionNotImplement(const std::string& err_msg) { return Status{kFunctionUnImplement, err_msg}; }

    Status PathNotValid(const std::string& err_msg) { return Status{kPathNotValid, err_msg}; }

    Status ModelParseError(const std::string& err_msg) { return Status{kModelParseError, err_msg}; }

    Status InternalError(const std::string& err_msg) { return Status{kInternalError, err_msg}; }

    Status InvalidArgument(const std::string& err_msg) { return Status{kInvalidArgument, err_msg}; }

    Status KeyHasExits(const std::string& err_msg) { return Status{kKeyValueHasExist, err_msg}; }
  }  // namespace error

  //重定义错误信息
  std::ostream& operator<<(std::ostream& os, const Status& x) 
  { 
    os << x.get_err_msg();
    return os;
  }
  
}
