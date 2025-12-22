//张量库，用于传递多维数据等
#include "tensor/tensor.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <glog/logging.h>
#include <numeric>
#include <sstream>

namespace tensor 
{
  template <typename T, typename Tp>
  static size_t reduce_dimension(T begin, T end, Tp init) 
  {
    if (begin >= end) 
      return 0;
    size_t size = std::accumulate(begin, end, init, std::multiplies<>());
    return size;
  }  

  static size_t data_type_size(base::DataType data_type) 
  {
    switch (data_type) 
    {
      case base::DataType::kDataTypeFp32: 
        return 4;
      case base::DataType::kDataTypeInt8: 
        return 1;
      case base::DataType::kDataTypeInt32: 
        return 4;
      case base::DataType::kDataTypeBf16:
        return sizeof(__nv_bfloat16);
      default: 
        LOG(FATAL) << "Unknown data type size for " << int(data_type);
        return 0;
    }
  }


  base::deviceId Tensor::getDeviceId() const
  {
    CHECK_NE(buffer_, nullptr);
    return this->buffer_->getDeviceId();
  }




  Tensor::Tensor(int32_t dim0, base::deviceId device_id,base::DataType data_type,void* ptr)
    :data_type_(data_type)
  {
    dims_.push_back(dim0);
    size_ = dim0;
    std::shared_ptr<base::DeviceAllocator> alloc;

    if(ptr == nullptr)
    { 
      switch (device_id) 
      {
        case base::CPUID:  
          alloc = base::DeviceAllocatorFactory::get_instance();
          break;
        default:           
          alloc = base::DeviceAllocatorFactory::get_instance(device_id);
          break;
      }
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_,device_id);
      this->buffer_ = buffer;
    }
    else
    {
      //代理buffer 不会进行释放
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_, ptr, device_id);
      this->buffer_ = buffer;
    }
  }


  Tensor::Tensor(int32_t dim0, int32_t dim1, base::deviceId device_id,base::DataType data_type,void* ptr)
    :data_type_(data_type)
  {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0*dim1;
    std::shared_ptr<base::DeviceAllocator> alloc;
    if(ptr == nullptr)
    { 
      switch (device_id) 
      {
        case base::CPUID:  
          alloc = base::DeviceAllocatorFactory::get_instance();
          break;
        default:           
          alloc = base::DeviceAllocatorFactory::get_instance(device_id);
          break;
      }
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_,device_id);
      this->buffer_ = buffer;
    }
    else
    {
      //代理buffer 不会进行释放
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_,ptr, device_id);
      this->buffer_ = buffer;
    }
  }


  Tensor::Tensor(int32_t dim0, int32_t dim1, int32_t dim2,base::deviceId device_id,base::DataType data_type,void* ptr)
    :data_type_(data_type)
  {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0*dim1*dim2;
    std::shared_ptr<base::DeviceAllocator> alloc;
    if(ptr == nullptr)
    { 
      switch (device_id) 
      {
        case base::CPUID:  
          alloc = base::DeviceAllocatorFactory::get_instance();
          break;
        default:           
          alloc = base::DeviceAllocatorFactory::get_instance(device_id);
          break;
      }
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_,device_id);
      this->buffer_ = buffer;
    }
    else
    {
      //代理buffer 不会进行释放
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_, ptr,device_id);
      this->buffer_ = buffer;
    }
  }

  Tensor::Tensor(int32_t dim0, int32_t dim1, int32_t dim2,int32_t dim3,base::deviceId device_id,base::DataType data_type,void* ptr)
    :data_type_(data_type)
  {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = dim0*dim1*dim2*dim3;
    std::shared_ptr<base::DeviceAllocator> alloc;
    if(ptr == nullptr)
    { 
      switch (device_id) 
      {
        case base::CPUID:  
          alloc = base::DeviceAllocatorFactory::get_instance();
          break;
        default:           
          alloc = base::DeviceAllocatorFactory::get_instance(device_id);
          break;
      }
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_,device_id);
      this->buffer_ = buffer;
    }
    else
    {
      //代理buffer 不会进行释放
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_, ptr, device_id);
      this->buffer_ = buffer;
    }
  }



  Tensor::Tensor(std::vector<int32_t> dims,base::deviceId device_id,base::DataType data_type,void* ptr)
    :data_type_(data_type),dims_(std::move(dims))
  {
    size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
    std::shared_ptr<base::DeviceAllocator> alloc;
    if(ptr == nullptr)
    { 
      switch (device_id) 
      {
        case base::CPUID:  
          alloc = base::DeviceAllocatorFactory::get_instance();
          break;
        default:           
          alloc = base::DeviceAllocatorFactory::get_instance(device_id);
          break;
      }
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_,device_id);
      this->buffer_ = buffer;
    }
    else
    {
      //代理buffer 不会进行释放
      std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_,  ptr, device_id);
      this->buffer_ = buffer;
    }
  }


  size_t Tensor::size() const { return this->size_; }


  int32_t Tensor::get_dim(int32_t idx) const 
  {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->dims_.size());
    return this->dims_.at(idx);
  }



  //如果tensor管理的是代理的数据，那么就会拷贝到新的buffer上
  void Tensor::to_device(base::deviceId device_id,cudaStream_t stm) 
  {
    CHECK_NE(buffer_, nullptr);

    base::deviceId newDeviceId = device_id;
    base::deviceId currentDeviceId = this->getDeviceId();

    if(currentDeviceId == newDeviceId)
      return ;

    std::shared_ptr<base::Buffer> buffer = 
      std::make_shared<base::Buffer>(data_type_size(data_type_) * size_,  newDeviceId);


    //这里好危险啊
    buffer->copy_from(this->buffer_.get(),stm);
    this->buffer_ = buffer;

}







  //重新帮顶buffer
  bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) 
  {
    if (!buffer) 
    {
      LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
      return false;
    }

    
    size_t byte_size = this->byte_size();
    if (byte_size > buffer->byte_size()) 
    {
      LOG(ERROR) << "The size of buffer is too small for the tensor!";
      return false;
    }
    buffer_ = buffer;
    return true;
  }


  void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) 
  {
    this->data_type_ = data_type;
    this->dims_ = dims;
    this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
    this->buffer_ = nullptr;
  }




  const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

  int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

  base::DataType Tensor::data_type() const { return data_type_; }


  void Tensor::reshape(const std::vector<int32_t>& dims) 
  {
    size_t size = reduce_dimension(dims.begin(), dims.end(), 1);

    if (!buffer_) 
    {
      this->dims_ = dims;
      this->size_ = size;
      return;
    }

    if (size > size_) 
    {
      auto new_buffer = std::make_shared<base::Buffer>(size * base::DataTypeSize(this->data_type_),this->getDeviceId());
      new_buffer->copy_from(buffer_.get());
      this->buffer_ = new_buffer;
    }

    this->dims_ = dims;
    this->size_ = size;
  }

  std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

  Tensor Tensor::clone() const 
  {
    Tensor new_tensor = *this;
    size_t byte_size = this->byte_size();
    new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, this->getDeviceId());
    new_tensor.buffer_->copy_from(buffer_.get());
    return new_tensor;
  }

  void Tensor::print(const std::string& name) const 
  {
    // Shallow copy tensor metadata, then move that view to host.
    Tensor host_tensor = *this;
    host_tensor.to_device(base::CPUID);

    std::ostringstream oss;
    if (!name.empty()) 
      oss << name << " ";

    oss << "[";
    switch (host_tensor.data_type()) 
    {
      case base::DataType::kDataTypeFp32: 
      {
        const float* data = host_tensor.ptr<float>();
        for (size_t i = 0; i < host_tensor.size(); ++i) 
        {
          if (i > 0) 
            oss << ", ";
          oss << data[i];
        }
        break;
      }
      case base::DataType::kDataTypeBf16: 
      {
        const __nv_bfloat16* data = host_tensor.ptr<__nv_bfloat16>();
        for (size_t i = 0; i < host_tensor.size(); ++i) 
        {
          if (i > 0) 
            oss << ", ";
          oss << __bfloat162float(data[i]);
        }
        break;
      }
      default: 
        LOG(WARNING) << "Tensor::print only supports fp32 and bf16, current data type: "
                     << static_cast<int>(host_tensor.data_type());
        return;
    }
    oss << "]";

    LOG(INFO) << oss.str();
  }


  size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }

  std::vector<size_t> Tensor::strides() const 
  {
    std::vector<size_t> strides;
    if (!dims_.empty()) 
    {
      for (int32_t i = 0; i < dims_.size() - 1; ++i) 
      {
        size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
        strides.push_back(stride);
      }
      strides.push_back(1);
    }
    return strides;
  }

  bool Tensor::is_empty() const 
  {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
  }
}  // namespace tensor
