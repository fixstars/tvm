# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_TARGET_ONNX)
  message(STATUS "Build with contrib.codegen_onnx")
  file(GLOB ONNX_CONTRIB_SRC src/runtime/contrib/onnx/onnx_module.cc)
  list(APPEND RUNTIME_SRCS ${ONNX_CONTRIB_SRC})
endif(USE_TARGET_ONNX)

if(USE_ONNX_RUNTIME)
  message(STATUS "Build with contrib.onnx_runtime")

  if(IS_DIRECTORY ${USE_ONNX_RUNTIME})
    set(ONNXRT_ROOT_DIR ${USE_ONNX_RUNTIME})
    message(STATUS "Custom ONNX RT path: " ${USE_ONNX_RUNTIME})
  endif()

  find_library(ONNXRT_LIB onnxruntime HINTS ${ONNXRT_ROOT_DIR} PATH_SUFFIXES lib)
  if(NOT ONNXRT_LIB)
      message(ERROR "Could not find ONNX RT.")
  endif()

  message(STATUS "ONNXRT_LIB: " ${ONNXRT_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${ONNXRT_LIB})

  if (NOT IS_DIRECTORY ${ONNXRT_ROOT_DIR})
    # NOTE: assume the include dir is located ${ONNXRT_LIB_DIR}/../include
    get_filename_component(ONNXRT_LIB_DIR ${ONNXRT_LIB} DIRECTORY)
    set(ONNXRT_ROOT_DIR ${ONNXRT_LIB_DIR}/..)
  endif()

  include_directories(${ONNXRT_ROOT_DIR}/include/onnxruntime/core/session)
  include_directories(${ONNXRT_ROOT_DIR}/include/onnxruntime)

  file(GLOB ONNX_CONTRIB_SRC src/runtime/contrib/onnx/onnx_runtime.cc)
  list(APPEND RUNTIME_SRCS ${ONNX_CONTRIB_SRC})

endif(USE_ONNX_RUNTIME)
