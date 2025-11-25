#ifndef DANGO_INCLUDE_BASE_ALLOC_H_
#define DANGO_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include <unordered_map>
#include <cuda_runtime_api.h>
#include "base.h"
namespace base 
{

    enum class MemcpyKind 
    {
        kMemcpyCPU2CPU = 0,
        kMemcpyCPU2CUDA = 1,
        kMemcpyCUDA2CPU = 2,
        kMemcpyCUDA2CUDA = 3,
        kMemcpyGPU2GPU = 4,
    };

    struct CudaMemoryBuffer 
    {
        void* data;
        bool busy;
        size_t byte_size;

        CudaMemoryBuffer() = default;

        CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
            : data(data), byte_size(byte_size), busy(busy) {}
    };

    struct MemcpyTask
    {
        const void* src_ptr;
        const void* dest_ptr;
        int src_dev_id ;
        int dest_dev_id;
        cudaStream_t stream = nullptr;

        size_t byte_size;
        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU;

        

        MemcpyTask(const void* dest , size_t size , MemcpyKind kind)
            : dest_ptr(dest),  byte_size(size), memcpy_kind(kind){}

        MemcpyTask(const void* dest ,const void* src , size_t size , MemcpyKind kind)
            :  dest_ptr(dest), src_ptr(src), byte_size(size), memcpy_kind(kind){}

        MemcpyTask( const void* dest ,const void* src , deviceId dev_id,const size_t size , MemcpyKind kind)
            :  dest_ptr(dest),src_ptr(src), src_dev_id(dev_id),byte_size(size), memcpy_kind(kind){}

        MemcpyTask( const void* dest ,const void* src , deviceId d_dev_id,const deviceId s_dev_id,size_t size , MemcpyKind kind)
            :  dest_ptr(dest),src_ptr(src), dest_dev_id(d_dev_id),src_dev_id(s_dev_id), byte_size(size), memcpy_kind(kind){}



        MemcpyTask(const void* dest , size_t size , MemcpyKind kind, cudaStream_t stm )
            : dest_ptr(dest), byte_size(size), memcpy_kind(kind),stream(stm){}

        MemcpyTask( const void* dest ,const void* src , size_t size , MemcpyKind kind, cudaStream_t stm )
            : dest_ptr(dest),src_ptr(src),  byte_size(size), memcpy_kind(kind),stream(stm){}

        MemcpyTask( const void* dest ,const void* src , int dev_id,size_t size , MemcpyKind kind, cudaStream_t stm )
            : dest_ptr(dest), src_ptr(src), src_dev_id(dev_id),byte_size(size), memcpy_kind(kind),stream(stm){}

        MemcpyTask( const void* dest ,const void* src , int dev_id,int d_dev_id,size_t size , MemcpyKind kind, cudaStream_t stm )
            :  dest_ptr(dest),src_ptr(src), dest_dev_id(d_dev_id),src_dev_id(dev_id),byte_size(size), memcpy_kind(kind),stream(stm){}




    };


    class DeviceAllocator 
    {
    public:
        virtual void release(void* ptr) const = 0;

        virtual void* allocate(size_t byte_size) const = 0;

        explicit DeviceAllocator(deviceId device_id) : device_id_(device_id) {}

        deviceId getDeviceId() const;

    protected:

        deviceId device_id_ = CPUID;
    };





    class CPUDeviceAllocator : public DeviceAllocator 
    {
    public:
        explicit CPUDeviceAllocator();

        void* allocate(size_t byte_size) const override;

        void release(void* ptr) const override;
    };




    class CUDADeviceAllocator : public DeviceAllocator 
    {
    public:
        explicit CUDADeviceAllocator(deviceId device_id);

        void* allocate(size_t byte_size) const override;

        void release(void* ptr) const override;

    private:
        mutable size_t no_busy_cnt_;

        mutable std::vector<CudaMemoryBuffer> big_buffers_;

        mutable std::vector<CudaMemoryBuffer> cuda_buffers_;
    };



     
    class DeviceAllocatorFactory 
    {
    public:
        static std::shared_ptr<DeviceAllocator> get_instance(deviceId device_id = CPUID)
        {
            if(device_id == CPUID)
            {
                if(cpu_instance == nullptr) 
                    cpu_instance = std::make_shared<CPUDeviceAllocator>();
                return cpu_instance;
            }
            if (!cuda_instances.count(device_id)) 
                cuda_instances[device_id] = std::make_shared<CUDADeviceAllocator>(device_id);
            return cuda_instances[device_id];
        }

        static void memcpy(MemcpyTask& task);

        static void memset_zero(MemcpyTask& task);



    private:
        static std::shared_ptr<CPUDeviceAllocator> cpu_instance ;
        static std::unordered_map<deviceId ,std::shared_ptr<CUDADeviceAllocator> >  cuda_instances; 
    };





} 
#endif  