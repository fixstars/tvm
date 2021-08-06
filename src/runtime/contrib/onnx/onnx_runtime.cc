/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file onnx_runtime.cc
 */
#include "onnx_runtime.h"

#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

#define TVM_DTYPE_DISPATCH(type, DType, ...)    \
  if (type == DataType::Float(64)) {            \
    typedef double DType;                       \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Float(32)) {     \
    typedef float DType;                        \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Float(16)) {     \
    typedef uint16_t DType;                     \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Int(64)) {       \
    typedef int64_t DType;                      \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Int(32)) {       \
    typedef int32_t DType;                      \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Int(16)) {       \
    typedef int16_t DType;                      \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Int(8)) {        \
    typedef int8_t DType;                       \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::UInt(64)) {      \
    typedef uint64_t DType;                     \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::UInt(32)) {      \
    typedef uint32_t DType;                     \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::UInt(16)) {      \
    typedef uint16_t DType;                     \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::UInt(8)) {       \
    typedef uint8_t DType;                      \
    { __VA_ARGS__ }                             \
  } else {                                      \
    LOG(FATAL) << "unknown data type " << type; \
  }

DataType ONNXElementType2TVMDType(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::Float(32);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::Int(32);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::Int(64);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return DataType::Int(16);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DataType::Int(8);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::UInt(8);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::Float(16);
    default:
      LOG(FATAL) << "this onnx data type not support yet: " << dtype;
      return DataType::Float(32);
  }
}

PackedFunc ONNXRuntime::GetFunction(const std::string& name,
                                    const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int in_idx = args[0];
      ICHECK_GE(in_idx, 0);
      ICHECK_LT(in_idx, session_->GetInputCount());
      this->SetInput(in_idx, args[1]);
    });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int out_idx = args[0];
      ICHECK_GE(out_idx, 0);
      ICHECK_LT(out_idx, session_->GetOutputCount());
      *rv = this->GetOutput(out_idx);
    });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Run(); });
  } else {
    return PackedFunc();
  }
}

void ONNXRuntime::Init(const std::string& onnx_model_bytes, Device dev,
                       const int& intra_op_num_threads) {
  // Save the model buffer into heap
  const char* buffer = onnx_model_bytes.c_str();
  const size_t buffer_size = onnx_model_bytes.size();
  model_buf_ = std::make_unique<char[]>(buffer_size);
  std::memcpy(model_buf_.get(), buffer, buffer_size);

  Ort::SessionOptions session_options;

  // Set basic options
  session_options.SetIntraOpNumThreads(intra_op_num_threads);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

  // Check if cuda is enabled
  if (dev.device_type == kDLCUDA) {
    OrtCUDAProviderOptions cuda_options{dev.device_id};
    session_options.AppendExecutionProvider_CUDA(cuda_options);
  }

  // Create a onnx runtime session
  session_ = std::make_unique<Ort::Session>(env_, model_buf_.get(), buffer_size, session_options);

  // Create a memory info for tensor allocation
  memory_info_ = std::make_unique<Ort::MemoryInfo>("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  input_tensors_.resize(session_->GetInputCount());
  output_tensors_.resize(session_->GetOutputCount());

  device_ = dev;
}

void ONNXRuntime::SetInput(int index, DLTensor* data_in) {
  std::vector<int64_t> shape;
  for (tvm_index_t i = 0; i < data_in->ndim; ++i) {
    shape.push_back(data_in->shape[i]);
  }
  NDArray tmp = NDArray::Empty(shape, data_in->dtype, device_);
  tmp.CopyFrom(data_in);
  input_tensors_[index] = tmp;
}

void ONNXRuntime::Run() {
  // Get input_names & output_names
  Ort::AllocatorWithDefaultOptions ort_alloc;

  size_t input_cnt = session_->GetInputCount();
  size_t output_cnt = session_->GetOutputCount();

  // Get input names
  std::vector<char*> input_names;
  for (size_t i = 0; i < input_cnt; ++i) {
    input_names.push_back(session_->GetInputName(i, ort_alloc));
  }

  // Get output names
  std::vector<char*> output_names;
  for (size_t i = 0; i < output_cnt; ++i) {
    output_names.push_back(session_->GetOutputName(i, ort_alloc));
  }

  // Convert input_tensors_(NDArray) -> Ort::Value
  std::vector<Ort::Value> input_values;
  for (size_t i = 0; i < input_cnt; ++i) {
    const NDArray& in = input_tensors_[i];
    DataType dtype(in->dtype);
    TVM_DTYPE_DISPATCH(dtype, DType, {
      input_values.push_back(Ort::Value::CreateTensor<DType>(*memory_info_, (DType*)in->data,
                                                             GetDataSize(in.ToDLPack()->dl_tensor),
                                                             in->shape, in->ndim));
    });
  }

  // Run a inference & save outputs
  auto outputs =
      session_->Run(Ort::RunOptions{nullptr}, &input_names[0], &input_values[0],
                    session_->GetInputCount(), &output_names[0], session_->GetOutputCount());

  // Copy output Ort::Values to output_tensors_(NDArray)
  for (size_t i = 0; i < input_cnt; ++i) {
    const Ort::Value& v = outputs[i];
    Ort::TensorTypeAndShapeInfo info = v.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = info.GetElementType();
    DataType dtype = ONNXElementType2TVMDType(type);

    std::vector<int64_t> shape = info.GetShape();
    int64_t size = 1;
    for (auto& s : shape) {
      size *= s;
    }

    NDArray ret;
    if (device_.device_type == kDLCPU) {
      ret = NDArray::Empty(shape, dtype, device_);
    } else {
      // NOTE: pre-allocated output buffers have not been implemented yet.
      // So, ouput Ort::Values are allocated on CPU.
      ret = NDArray::Empty(shape, dtype, Device{kDLCPU, 0});
    }

    TVM_DTYPE_DISPATCH(dtype, DType, {
      DType* dest = static_cast<DType*>(ret->data);
      const DType* src = v.GetTensorData<DType>();
      for (int64_t i = 0; i < size; ++i) {
        dest[i] = src[i];
      }
    });

    output_tensors_[i] = ret;
  }

  // Free input & output names
  for (size_t i = 0; i < input_cnt; ++i) {
    ort_alloc.Free(input_names[i]);
  }
  for (size_t i = 0; i < output_cnt; ++i) {
    ort_alloc.Free(output_names[i]);
  }
}

NDArray ONNXRuntime::GetOutput(int index) const { return output_tensors_[index]; }

Module ONNXRuntimeCreate(const std::string& onnx_model_bytes, Device dev,
                         const int& intra_op_num_threads) {
  auto exec = make_object<ONNXRuntime>();
  exec->Init(onnx_model_bytes, dev, intra_op_num_threads);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.onnx_runtime.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = ONNXRuntimeCreate(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("target.runtime.onnx").set_body_typed(ONNXRuntimeCreate);
}  // namespace runtime
}  // namespace tvm
