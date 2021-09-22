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
"""ONNX runtime that load and run onnx models."""
import tvm._ffi

from ..rpc import base as rpc_base


def create(onnx_model_bytes, device, providers="CPUExecutionProvider", intra_op_num_threads=1):
    """Create a runtime executor module given a onnx model and device.
    Parameters
    ----------
    onnx_model_byte : bytes
        The onnx model to be deployed in bytes string format.
    device : Device
        The device to deploy the module. It can be local or remote when there
        is only one Device.
    providers: str
        The semicolon-separated list of the ONNX runtime execution providers.
        The caller of this func must set appropriate providers
    intra_op_num_threads : int
        this controls the number of threads to use to run the model
    Returns
    -------
    onnx_runtime : ONNXModule
        Runtime tflite module that can be used to execute the tflite model.
    """
    device_type = device.device_type

    runtime_func = "tvm.onnx_runtime.create"

    if device_type >= rpc_base.RPC_SESS_MASK:
        fcreate = device._rpc_sess.get_function(runtime_func)
    else:
        fcreate = tvm._ffi.get_global_func(runtime_func)

    return ONNXModule(fcreate(bytearray(onnx_model_bytes), device, providers, intra_op_num_threads))


class ONNXModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : Module
        The internal tvm module that holds the actual tflite functions.

    Attributes
    ----------
    module : Module
        The internal tvm module that holds the actual tflite functions.
    """

    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]

    def set_input(self, index, value):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key
        """
        self._set_input(index, value)

    def run(self):
        """run forward execution of the model"""
        self._run()

    def get_output(self, index):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index
        """
        return self._get_output(index)
