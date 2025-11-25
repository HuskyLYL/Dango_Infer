#include "base/buffer.h"
#include <glog/logging.h>
namespace base 
{






    //实例化初始化
    Buffer::Buffer(size_t byte_size,deviceId device_id)
        :byte_size_(byte_size),device_id_(device_id)
    {

        allocator_ = DeviceAllocatorFactory::get_instance(device_id_);
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
    }


    //托管类型初始化
    Buffer::Buffer(size_t byte_size,void* ptr,deviceId device_id)
        :byte_size_(byte_size),device_id_(device_id),ptr_(ptr){}






    Buffer::~Buffer() 
    {

        if (!use_external_&&ptr_ && allocator_) 
        {
            allocator_->release(ptr_);
            ptr_ = nullptr;
        }
    }

    void* Buffer::ptr() 
    {
        return ptr_;
    }

    const void* Buffer::ptr() const 
    {
        return ptr_;
    }

    size_t Buffer::byte_size() const 
    {
        return byte_size_;
    }

    std::shared_ptr<DeviceAllocator> Buffer::allocator() const 
    {
        return allocator_;
    }

    deviceId Buffer::getDeviceId() const
    {
        return device_id_;
    }





    //尽可能让两个tensor的内存相当吧
    void Buffer::copy_from(const Buffer& buffer,cudaStream_t stm) const 
    {

        CHECK(buffer.ptr_ != nullptr);

        size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
  
        const deviceId& buffer_device = buffer.getDeviceId();
        const deviceId& current_device = this->getDeviceId();

        

        if (buffer_device == CPUID && current_device == CPUID) 
        {
            MemcpyTask memcpyTask(this->ptr_ ,buffer.ptr(),byte_size , MemcpyKind::kMemcpyCPU2CPU);
            return DeviceAllocatorFactory::memcpy(memcpyTask);  
        }

            
  
        else if (current_device == CPUID) 
        {
            if(stm == nullptr)
            {
                MemcpyTask memcpyTask( this->ptr_,buffer.ptr() , buffer.getDeviceId(),byte_size , MemcpyKind::kMemcpyCUDA2CPU);
                return DeviceAllocatorFactory::memcpy(memcpyTask);
            }
            MemcpyTask memcpyTask( this->ptr_,buffer.ptr() , buffer.getDeviceId(),byte_size , MemcpyKind::kMemcpyCUDA2CPU,stm);
            return DeviceAllocatorFactory::memcpy(memcpyTask);

        }
        
            
        
        else if (buffer_device ==CPUID) 
        {
            if(stm == nullptr)
            {
                MemcpyTask memcpyTask( this->ptr_,buffer.ptr() , this->getDeviceId(), byte_size , MemcpyKind::kMemcpyCPU2CUDA);
                return DeviceAllocatorFactory::memcpy(memcpyTask); 
            }
            MemcpyTask memcpyTask( this->ptr_,buffer.ptr() , this->getDeviceId(), byte_size , MemcpyKind::kMemcpyCPU2CUDA,stm);
            return DeviceAllocatorFactory::memcpy(memcpyTask); 
        }

            

        else 
        {
            if(buffer_device == current_device)
            {
                if(stm == nullptr)
                {
                    MemcpyTask memcpyTask(this->ptr_,buffer.ptr() ,current_device, byte_size , MemcpyKind::kMemcpyCUDA2CUDA);
                    return DeviceAllocatorFactory::memcpy(memcpyTask); 
                }
                MemcpyTask memcpyTask(this->ptr_,buffer.ptr() ,current_device, byte_size , MemcpyKind::kMemcpyCUDA2CUDA,stm);
                return DeviceAllocatorFactory::memcpy(memcpyTask); 
            }
            else
            {
                if(stm == nullptr)
                {
                    MemcpyTask memcpyTask(this->ptr_,buffer.ptr() ,current_device, buffer_device ,byte_size , MemcpyKind::kMemcpyGPU2GPU);
                    return DeviceAllocatorFactory::memcpy(memcpyTask); 
                }
                MemcpyTask memcpyTask(this->ptr_,buffer.ptr() ,current_device, buffer_device ,byte_size , MemcpyKind::kMemcpyGPU2GPU,stm);
                return DeviceAllocatorFactory::memcpy(memcpyTask); 

            }     
        }
    } 


    
    void Buffer::copy_from(const Buffer* buffer,cudaStream_t stm) const 
    {

        CHECK(buffer->ptr_ != nullptr);

        size_t byte_size = byte_size_ < buffer->byte_size_ ? byte_size_ : buffer->byte_size_;
  
        const deviceId& buffer_device = buffer->getDeviceId();
        const deviceId& current_device = this->getDeviceId();

        

        if (buffer_device == CPUID && current_device == CPUID) 
        {
            MemcpyTask memcpyTask(this->ptr_ ,buffer->ptr(),byte_size , MemcpyKind::kMemcpyCPU2CPU);
            return DeviceAllocatorFactory::memcpy(memcpyTask);  
        }

            
  
        else if (current_device == CPUID) 
        {
            if(stm == nullptr)
            {
                MemcpyTask memcpyTask( this->ptr_,buffer->ptr() , buffer->getDeviceId(),byte_size , MemcpyKind::kMemcpyCUDA2CPU);
                return DeviceAllocatorFactory::memcpy(memcpyTask);
            }
            MemcpyTask memcpyTask( this->ptr_,buffer->ptr() , buffer->getDeviceId(),byte_size , MemcpyKind::kMemcpyCUDA2CPU,stm);
            return DeviceAllocatorFactory::memcpy(memcpyTask);

        }
        
            
        
        else if (buffer_device ==CPUID) 
        {
            if(stm == nullptr)
            {
                MemcpyTask memcpyTask( this->ptr_,buffer->ptr() , this->getDeviceId(), byte_size , MemcpyKind::kMemcpyCPU2CUDA);
                return DeviceAllocatorFactory::memcpy(memcpyTask); 
            }
            MemcpyTask memcpyTask( this->ptr_,buffer->ptr() , this->getDeviceId(), byte_size , MemcpyKind::kMemcpyCPU2CUDA,stm);
            return DeviceAllocatorFactory::memcpy(memcpyTask); 
        }

            

        else 
        {
            if(buffer_device == current_device)
            {
                if(stm == nullptr)
                {
                    MemcpyTask memcpyTask(this->ptr_,buffer->ptr() ,current_device, byte_size , MemcpyKind::kMemcpyCUDA2CUDA);
                    return DeviceAllocatorFactory::memcpy(memcpyTask); 
                }
                MemcpyTask memcpyTask(this->ptr_,buffer->ptr() ,current_device, byte_size , MemcpyKind::kMemcpyCUDA2CUDA,stm);
                return DeviceAllocatorFactory::memcpy(memcpyTask); 
            }
            else
            {
                if(stm == nullptr)
                {
                    MemcpyTask memcpyTask(this->ptr_,buffer->ptr() ,current_device, buffer_device ,byte_size , MemcpyKind::kMemcpyGPU2GPU);
                    return DeviceAllocatorFactory::memcpy(memcpyTask); 
                }
                MemcpyTask memcpyTask(this->ptr_,buffer->ptr() ,current_device, buffer_device ,byte_size , MemcpyKind::kMemcpyGPU2GPU,stm);
                return DeviceAllocatorFactory::memcpy(memcpyTask); 

            }     
        }
    }


    std::shared_ptr<Buffer> Buffer::get_shared_from_this() 
    {
        return shared_from_this();
    }

    bool Buffer::is_external() const 
    {
        return this->use_external_;
    }

}  // namespace base