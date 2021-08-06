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
from collections import OrderedDict

import numpy as np
import onnxruntime as ort
import pytest
import tvm
import tvm.relay.testing
from tvm import rpc
from tvm.contrib import onnx_runtime, utils
from tvm.contrib.target.onnx import to_onnx


def func_to_onnx(mod, params, name):
    onnx_model = to_onnx(mod, params, name, path=None)
    return onnx_model.SerializeToString()


def get_data(in_data_shapes, dtype="float32"):
    in_data = OrderedDict()
    for name, shape in in_data_shapes.items():
        in_data[name] = np.random.uniform(size=shape).astype(dtype)
    return in_data


def _create_onnx_model():
    if not tvm.runtime.enabled("onnx"):
        print("skip because onnx runtime is not enabled...")
        return
    if not tvm.get_global_func("tvm.onnx_runtime.create", True):
        print("skip because onnx runtime is not enabled...")
        return

    mod, params = tvm.relay.testing.resnet.get_workload(1, 1000, num_layers=18)
    return func_to_onnx(mod, params, "test_resnet")


@pytest.mark.skip("skip because accessing output tensor is flakey")
def test_local():
    if not tvm.runtime.enabled("onnx"):
        print("skip because onnx runtime is not enabled...")
        return
    if not tvm.get_global_func("tvm.onnx_runtime.create", True):
        print("skip because onnxx runtime is not enabled...")
        return

    onnx_fname = "model.onnx"
    onnx_model = _create_onnx_model()
    temp = utils.tempdir()
    onnx_model_path = temp.relpath(onnx_fname)
    open(onnx_model_path, "wb").write(onnx_model)

    # inference via onnxruntime python apis
    in_data_shapes = OrderedDict({"data": (1, 3, 224, 224)})
    in_data = get_data(in_data_shapes, dtype="float32")

    sess = ort.InferenceSession(onnx_model_path)
    input_names = {}
    for input, data in zip(sess.get_inputs(), in_data.values()):
        input_names[input.name] = data
    output_names = [output.name for output in sess.get_outputs()]
    onnx_output = sess.run(output_names, input_names)

    # inference via tvm onnx runtime
    with open(onnx_model_path, "rb") as model_fin:
        runtime = onnx_runtime.create(model_fin.read(), tvm.cpu(0))
        runtime.set_input(0, tvm.nd.array(in_data["data"]))
        runtime.run()
        out = runtime.get_output(0).numpy()
        np.testing.assert_equal(out, onnx_output[0])


def test_remote():
    if not tvm.runtime.enabled("onnx"):
        print("skip because onnx runtime is not enabled...")
        return
    if not tvm.get_global_func("tvm.onnx_runtime.create", True):
        print("skip because onnxx runtime is not enabled...")
        return

    onnx_fname = "model.onnx"
    onnx_model = _create_onnx_model()
    temp = utils.tempdir()
    onnx_model_path = temp.relpath(onnx_fname)
    open(onnx_model_path, "wb").write(onnx_model)

    # inference via onnxruntime python apis
    in_data_shapes = OrderedDict({"data": (1, 3, 224, 224)})
    in_data = get_data(in_data_shapes, dtype="float32")

    sess = ort.InferenceSession(onnx_model_path)
    input_names = {}
    for input, data in zip(sess.get_inputs(), in_data.values()):
        input_names[input.name] = data
    output_names = [output.name for output in sess.get_outputs()]
    onnx_output = sess.run(output_names, input_names)

    # inference via remote tvm onnx runtime
    server = rpc.Server("127.0.0.1")
    remote = rpc.connect(server.host, server.port)
    a = remote.upload(onnx_model_path)

    with open(onnx_model_path, "rb") as model_fin:
        runtime = onnx_runtime.create(model_fin.read(), remote.cpu(0))
        runtime.set_input(0, tvm.nd.array(in_data["data"], remote.cpu(0)))
        runtime.run()
        out = runtime.get_output(0).numpy()
        np.testing.assert_equal(out, onnx_output[0])

    server.terminate()


if __name__ == "__main__":
    test_local()
    test_remote()
