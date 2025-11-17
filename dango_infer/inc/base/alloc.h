#ifndef DANGO_INCLUDE_BASE_ALLOC_H_
#define DANGO_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base.h"
namespace base 
{
    //枚举类，设备的拷贝类型
    enum class MemcpyKind 
    {
        kMemcpyCPU2CPU = 0,
        kMemcpyCPU2CUDA = 1,
        kMemcpyCUDA2CPU = 2,
        kMemcpyCUDA2CUDA = 3,
    };


    class DeviceAllocator 
    {
    public:
        virtual void release(void* ptr) const = 0;

        virtual void* allocate(size_t byte_size) const = 0;

        explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

        virtual DeviceType device_type() const { return device_type_; }

        virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, bool need_async,void* stream = nullptr) const;

        virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

    private:
        
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
    };


    class CPUDeviceAllocator : public DeviceAllocator 
    {
    public:
        explicit CPUDeviceAllocator();

        void* allocate(size_t byte_size) const override;

        void release(void* ptr) const override;
    };


    //开一个设备分配器的目的是为了统一资源池
    class CUDADeviceAllocator : public DeviceAllocator 
    {
    public:
        explicit CUDADeviceAllocator();

        void* allocate(size_t byte_size) const override;

        void release(void* ptr) const override;

        //在CUDA中，标记一个event确保
        void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
            MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,void* event = nullptr) const override;

    private:

        deviceId device_id_;

        mutable std::map<int, size_t> no_busy_cnt_;

        mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;

        mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
    };


    struct CudaMemoryBuffer 
    {
        void* data;
        size_t byte_size;
        bool busy;

        CudaMemoryBuffer() = default;

        CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
            : data(data), byte_size(byte_size), busy(busy) {}
    };


    //CPU和GPU设备管理器的工厂类
    //全局CPU和GPU的设备管理器各初始化一个
    //通过智能指针进行管理
    class CPUDeviceAllocatorFactory 
    {
    public:
        static std::shared_ptr<CPUDeviceAllocator> get_instance() 
        {
            if (instance == nullptr) 
                instance = std::make_shared<CPUDeviceAllocator>();
            return instance;
        }
    private:
        static std::shared_ptr<CPUDeviceAllocator> instance;
    };


    class CUDADeviceAllocatorFactory 
    {
    public:
        static std::shared_ptr<CUDADeviceAllocator> get_instance() 
        {
            if (instance == nullptr) 
                instance = std::make_shared<CUDADeviceAllocator>();
            return instance;
        }
    private:
        static std::shared_ptr<CUDADeviceAllocator> instance;
    };
} 
#endif  