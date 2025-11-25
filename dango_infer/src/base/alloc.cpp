#include "base/alloc.h"



namespace base 
{

    




    deviceId DeviceAllocator::getDeviceId() const { return device_id_;}
    
    
    void DeviceAllocatorFactory::memcpy(MemcpyTask& task)
    {
        CHECK_NE(task.src_ptr, nullptr);

        CHECK_NE(task.dest_ptr, nullptr);

        if(!task.byte_size) return;



        

        if (task.memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) std::memcpy(const_cast<void*>(task.dest_ptr), const_cast<void*>(task.src_ptr), task.byte_size);

        else if(task.memcpy_kind == MemcpyKind::kMemcpyGPU2GPU)
        {

            if(!task.stream) cudaMemcpyPeer(const_cast<void*>(task.dest_ptr), task.dest_dev_id, const_cast<void*>(task.src_ptr), task.src_dev_id, task.byte_size);

            else
            {
          
                setDevice(task.dest_dev_id);

                cudaMemcpyPeerAsync(const_cast<void*>(task.dest_ptr), task.dest_dev_id , const_cast<void*>(task.src_ptr) , task.src_dev_id,task.byte_size,task.stream);

            }
        }

        else
        {
            cudaMemcpyKind kind;
            switch (task.memcpy_kind) 
            {
                case MemcpyKind::kMemcpyCPU2CUDA:
                    setDevice(task.src_dev_id);
                    kind = cudaMemcpyHostToDevice;
                    break;
                case MemcpyKind::kMemcpyCUDA2CPU:
                    setDevice(task.src_dev_id);
                    kind = cudaMemcpyDeviceToHost;
                    break;
                case MemcpyKind::kMemcpyCUDA2CUDA:
                    setDevice(task.src_dev_id);
                    kind = cudaMemcpyDeviceToDevice;
                    break;
                default:
                    LOG(FATAL) << "Unknown memcpy kind: " << int(task.memcpy_kind);
                    return; 
            }
            if (!task.stream) cudaMemcpy(const_cast<void*>(task.dest_ptr), const_cast<void*>(task.src_ptr), task.byte_size, kind);

            else

                cudaMemcpyAsync(const_cast<void*>(task.dest_ptr), const_cast<void*>(task.src_ptr), task.byte_size, kind, task.stream);

        }
  
    }





    void DeviceAllocatorFactory::memset_zero(MemcpyTask& task ) 
    {

        if (task.memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) 
            std::memset(const_cast<void*>(task.src_ptr), 0, task.byte_size);
  
        else 
        {   
            if (task.stream) 
            {
                
                setDevice(task.src_dev_id);

                cudaMemsetAsync(const_cast<void*>(task.src_ptr), 0, task.byte_size, task.stream);

            } 
            else 
                cudaMemset(const_cast<void*>(task.src_ptr), 0, task.byte_size);
        }
    }

} 