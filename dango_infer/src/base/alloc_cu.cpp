#include "base/alloc.h"
namespace base 
{

  CUDADeviceAllocator::CUDADeviceAllocator(deviceId device_id) :DeviceAllocator(device_id) {}


  void* CUDADeviceAllocator::allocate(size_t byte_size) const 
  {
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    CHECK(state == cudaSuccess);
    if(id != device_id_)
      cudaSetDevice(device_id_);


    if (byte_size > 1024 * 1024) 
    {
      
      int sel_id = -1;
      for (int i = 0; i < big_buffers_.size(); i++) 
      {
        if (big_buffers_[i].byte_size >= byte_size && !big_buffers_[i].busy &&
          big_buffers_[i].byte_size - byte_size < 1 * 1024 * 1024) 
          if (sel_id == -1 || big_buffers_[sel_id].byte_size > big_buffers_[i].byte_size) 
            sel_id = i;
      }

      if (sel_id != -1) 
      {
        big_buffers_[sel_id].busy = true;
        return big_buffers_[sel_id].data;
      }

      void* ptr = nullptr;
      state = cudaMalloc(&ptr, byte_size);
      if (cudaSuccess != state) 
      {
        char buf[256];
        snprintf(buf, 256,
          "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
          "left on  device.",
        byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
      }
      big_buffers_.emplace_back(ptr, byte_size, true);
      return ptr;
    }

    
    for (int i = 0; i < cuda_buffers_.size(); i++) 
    {
      if (cuda_buffers_[i].byte_size >= byte_size && !cuda_buffers_[i].busy) 
      {
        cuda_buffers_[i].busy = true;
        no_busy_cnt_ -= cuda_buffers_[i].byte_size;
        return cuda_buffers_[i].data;
      }
    }
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) 
    {
      char buf[256];
      snprintf(buf, 256,
        "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
        "left on  device.",
        byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    cuda_buffers_.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  void CUDADeviceAllocator::release(void* ptr) const 
  {
    if (!ptr) 
      return;
    


  cudaError_t state = cudaSuccess;



  if (no_busy_cnt_ > 1024 * 1024 * 1024) 
  {
    std::vector<CudaMemoryBuffer> temp;
    for (int i = 0; i < cuda_buffers_.size(); i++) 
    {
      if (!cuda_buffers_[i].busy) 
      {
        state = cudaSetDevice(device_id_);
        state = cudaFree(cuda_buffers_[i].data);
        CHECK(state == cudaSuccess)
          << "Error: CUDA error when release memory on device " << device_id_;
      } 
      else 
        temp.push_back(cuda_buffers_[i]);
    }
    cuda_buffers_.clear();
    cuda_buffers_ = temp;
    no_busy_cnt_ = 0;
  }



    for (int i = 0; i < cuda_buffers_.size(); i++) 
    {
      if (cuda_buffers_[i].data == ptr) 
      {
        no_busy_cnt_ += cuda_buffers_[i].byte_size;
        cuda_buffers_[i].busy = false;
        return;
      }
    }
    
    for (int i = 0; i < big_buffers_.size(); i++) 
    {
      if (big_buffers_[i].data == ptr) 
      {
        big_buffers_[i].busy = false;
        return;
      }
    }
  
  state = cudaSetDevice(device_id_);
  state = cudaFree(ptr);

  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

  


}  // namespace base