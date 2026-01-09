#ifndef DANGO_INCLUDE_BASE_BASE_H_
#define DANGO_INCLUDE_BASE_BASE_H_
#include <cuda_fp16.h>  
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <string>


//用于抑制编译器未使用的警告
#define UNUSED(expr)          \
    do                        \
    {                         \
        (void)(expr);         \
    }while (0)


#define CUDA_CALL(expr)                                                         \
    do                                                                          \
    {                                                                           \
        cudaError_t _cuda_err = (expr);                                         \
        if (_cuda_err != cudaSuccess)                                           \
        {                                                                       \
            LOG(FATAL) << "CUDA error: " << cudaGetErrorString(_cuda_err)       \
                 << " | expr: " << #expr                                        \
                 << " | file: " << __FILE__                                     \
                 << " | line: " << __LINE__;                                    \
        }                                                                       \
    } while (0)

// cudaProfilerPause is only available on newer toolkits; fall back to Stop when missing.
inline cudaError_t cuda_profiler_pause_compat()
{
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    return cudaProfilerPause();
#else
    return cudaProfilerStop();
#endif
}



namespace model 
{
    //model 推理中缓存的BUFFER类型
    enum class ModelBufferType 
    {
        kInputTokens = 0,
        kInputEmbeddings = 1,
        kOutputRMSNorm = 2,
        kKeyCache = 3,
        kValueCache = 4,
        kQuery = 5,
        kInputPos = 6,
        kScoreStorage = 7,
        kOutputMHA = 8,
        kAttnOutput = 9,
        kW1Output = 10,
        kW2Output = 11,
        kW3Output = 12,
        kFFNRMSNorm = 13,
        kForwardOutput = 15,
        kForwardOutputCPU = 16,
        kSinCache = 17,
        kCosCache = 18,
    };
}

namespace base 
{
    
    using deviceId = int;

    // Global switch: true enables extra debug outputs across the project.
    extern bool g_enable_debug_log;

    // avoid C++17 inline variable requirement on nvcc default standard
    static constexpr deviceId CPUID = static_cast<deviceId>(999999);

    //数据精度
    enum class DataType : uint8_t 
    {
        kDataTypeUnknown = 0,
        kDataTypeFp32 = 1,
        kDataTypeInt8 = 2,
        kDataTypeInt32 = 3,
        kDataTypeBf16 = 4,
    };

    //模型类型
    enum class ModelType : uint8_t 
    {
        kModelTypeUnknown = 0,
        kModelTypeLLama2 = 1,
    };

    //内联函数，输入枚举类，返回实际存储大小
    inline size_t DataTypeSize(DataType data_type) 
    {
        if (data_type == DataType::kDataTypeFp32) 
            return sizeof(float);

        else if (data_type == DataType::kDataTypeInt8) 
            return sizeof(int8_t);

        else if (data_type == DataType::kDataTypeInt32) 
            return sizeof(int32_t);

        else if(data_type == DataType::kDataTypeBf16)
            return sizeof(__half);  
        else 
            return 0;
    }

    //不可拷贝父类
    class NoCopyable 
    {
    protected:
        NoCopyable() = default;

        ~NoCopyable() = default;

        NoCopyable(const NoCopyable&) = delete;

        NoCopyable& operator=(const NoCopyable&) = delete;
    };

    //枚举类：状态类型
    enum StatusCode : uint8_t 
    {
        kSuccess = 0,
        kFunctionUnImplement = 1,
        kPathNotValid = 2,
        kModelParseError = 3,
        kInternalError = 5,
        kKeyValueHasExist = 6,
        kInvalidArgument = 7,
        kRunning = 8,
        kFailed = 9,

    };

    // 分词器类型
    enum class TokenizerType 
    {
        kEncodeUnknown = -1,
        kEncodeSpe = 0,
        kEncodeBpe = 1,
    };

    class Status 
    {
    public:
        Status(int code = StatusCode::kSuccess, std::string err_message = "");

        Status(const Status& other) = default;

        Status& operator=(const Status& other) = default;

        Status& operator=(int code);

        bool operator==(int code) const;

        bool operator!=(int code) const;

        operator int() const;

        operator bool() const;

        int32_t get_err_code() const;

        const std::string& get_err_msg() const;

        void set_err_msg(const std::string& err_msg);

    private:
        int code_ = StatusCode::kSuccess;
        std::string message_;
    };

    inline void setDevice(deviceId device_id)
    {
    
        if (device_id == CPUID) 
            return;
        int current = -1;
        cudaError_t err = cudaGetDevice(&current);
        if (err != cudaSuccess) 
            LOG(FATAL) << "cudaGetDevice failed: " << cudaGetErrorString(err);
        if (current == static_cast<int>(device_id))
            return;
        err = cudaSetDevice(static_cast<int>(device_id));
        if (err != cudaSuccess) 
            LOG(FATAL) << "cudaSetDevice(" << device_id << ") failed: " << cudaGetErrorString(err);
        return ;

    }



    namespace error 
    {
        #define STATUS_CHECK(call)                                                                  \
        do                                                                                          \
        {                                                                                           \
            const base::Status& status = call;                                                      \
            if (!status)                                                                            \
            {                                                                                       \
                const size_t buf_size = 512;                                                        \
                char buf[buf_size];                                                                 \
                snprintf(buf, buf_size - 1,                                                         \
                    "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__,     \
                    __LINE__, int(status), status.get_err_msg().c_str());                           \
                LOG(FATAL) << buf;                                                                  \
            }                                                                                       \
        } while (0)

        Status Success(const std::string& err_msg = "");

        Status Running(const std::string& err_msg = "");

        Status Failed(const std::string& err_msg = "");

        Status FunctionNotImplement(const std::string& err_msg = "");

        Status PathNotValid(const std::string& err_msg = "");

        Status ModelParseError(const std::string& err_msg = "");

        Status InternalError(const std::string& err_msg = "");

        Status KeyHasExits(const std::string& err_msg = "");

        Status InvalidArgument(const std::string& err_msg = "");

    } 


    std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_BASE_H_
