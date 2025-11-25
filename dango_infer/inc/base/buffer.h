#ifndef DANGO_INCLUDE_BASE_BUFFER_H_
#define DANGO_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base 
{
  class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> 
  {
  private:

    deviceId device_id_ = 0;

    size_t byte_size_ = 0;

    void* ptr_ = nullptr;

    //决定是否分配
    bool use_external_ = false;

    std::shared_ptr<DeviceAllocator> allocator_ = nullptr;

  public:

    explicit Buffer() = default;

    

    //实例化初始
    explicit Buffer(size_t byte_size,deviceId device_id=CPUID);



    //托管化初始
    explicit Buffer(size_t byte_size,void* ptr,deviceId device_id);





    deviceId getDeviceId() const;

    virtual ~Buffer();

    void copy_from(const Buffer& buffer,cudaStream_t stm = nullptr) const;

    void copy_from(const Buffer* buffer,cudaStream_t stm = nullptr) const;



    


    void* ptr();

    const void* ptr() const;

    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;



    std::shared_ptr<Buffer> get_shared_from_this();

    bool is_external() const;

  };
} 

#endif