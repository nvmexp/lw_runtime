/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LWIDIA_LIBS_TEST_LWDNN_UTIL_H_
#define LWIDIA_LIBS_TEST_LWDNN_UTIL_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/types/variant.h"
#include "lwdnn/include/lwdnn.h"
#include "lwda_util.h"
#include "lwdnn.pb.h"

// Provides wrappers to perform lwDNN colwolution ops defined by proto messages.

namespace lwidia_libs_test {

// Returns Status from lwDNN status.
Status GetStatus(lwdnnStatus_t);

namespace detail {
struct LwdnnHandleDeleter {
  void operator()(lwdnnHandle_t) const;
};
struct TensorDescriptorDeleter {
  void operator()(lwdnnTensorDescriptor_t) const;
};
struct FilterDescriptorDeleter {
  void operator()(lwdnnFilterDescriptor_t) const;
};
struct ColwolutionDescriptorDeleter {
  void operator()(lwdnnColwolutionDescriptor_t) const;
};
}  // namespace detail

// RAII wrappers for lwDNN handles.
using LwdnnHandle = std::unique_ptr<lwdnnContext, detail::LwdnnHandleDeleter>;
using TensorDescriptor =
    std::unique_ptr<lwdnnTensorStruct, detail::TensorDescriptorDeleter>;
using FilterDescriptor =
    std::unique_ptr<lwdnnFilterStruct, detail::FilterDescriptorDeleter>;
using ColwolutionDescriptor =
    std::unique_ptr<lwdnnColwolutionStruct,
                    detail::ColwolutionDescriptorDeleter>;

// Specifies one colwolution algorithm.
using ColwolutionAlgo =
    absl::variant<lwdnnColwolutionFwdAlgo_t, lwdnnColwolutionBwdDataAlgo_t,
                  lwdnnColwolutionBwdFilterAlgo_t>;

// Creates a lwDNN handle.
LwdnnHandle CreateLwdnnHandle();

// Creates lwDNN tensor descriptor from proto.
TensorDescriptor CreateTensorDescriptor(proto::TensorDescriptor proto);

// Returns true iff left and right are equal.
bool TensorDescriptorEqual(const TensorDescriptor& left,
                           const TensorDescriptor& right);

// Returns the number of elements in tensor, including strided elements.
size_t GetTensorNumElements(const TensorDescriptor& tensor);

// Returns the size of the tensor in bytes.
size_t GetTensorSizeInBytes(const TensorDescriptor& tensor);

// Returns the data type of tensor.
lwdnnDataType_t GetTensorDataType(const TensorDescriptor& tensor);

// Allocates device memory for tensor and initializes it with random values.
StatusOr<DeviceMemory> CreateTensorData(const TensorDescriptor& tensor,
                                        double lower, double upper,
                                        const RandomGenerator& rand_gen);

// Blends the data from src tensor with dst tensor.
Status TransformTensor(const LwdnnHandle& handle, double alpha, double beta,
                       const TensorDescriptor& src_desc,
                       const DeviceMemory& src_data,
                       const TensorDescriptor& dst_desc,
                       const DeviceMemory& dst_data);

// Copies the data from src tensor to dst tensor.
Status TransformTensor(const LwdnnHandle& handle,
                       const TensorDescriptor& src_desc,
                       const DeviceMemory& src_data,
                       const TensorDescriptor& dst_desc,
                       const DeviceMemory& dst_data);

// Copies the data from src tensor to dst tensor. Also colwerts src data type
// to dst data type if required.
Status ColwertAndTransformTensor(const LwdnnHandle& handle, double alpha,
                                 double beta, const TensorDescriptor& src_desc,
                                 const DeviceMemory& src_data,
                                 const TensorDescriptor& dst_desc,
                                 const DeviceMemory& dst_data);

string GetTensorDebugString(const TensorDescriptor& desc,
                            const DeviceMemory& data, bool print_values);

// Creates lwDNN filter descriptor from proto.
FilterDescriptor CreateFilterDescriptor(const proto::FilterDescriptor& proto);

// Returns true iff left and right are equal.
bool FilterDescriptorEqual(const FilterDescriptor& left,
                           const FilterDescriptor& right);

// Returns the number of elements in filter, including strided elements.
size_t GetFilterNumElements(const FilterDescriptor& filter);

// Returns the size of the filter in bytes.
size_t GetFilterSizeInBytes(const FilterDescriptor& filter);

// Returns the data type of filter.
lwdnnDataType_t GetFilterDataType(const FilterDescriptor& filter);

// Allocates device memory for filter and initializes it with random values.
StatusOr<DeviceMemory> CreateFilterData(const FilterDescriptor& filter,
                                        const RandomGenerator& rand_gen);

// Creates lwDNN colwolution descriptor from proto.
ColwolutionDescriptor CreateColwolutionDescriptor(
    proto::ColwolutionDescriptor proto);

// Returns true iff left and right are equal.
bool ColwolutionDescriptorEqual(const ColwolutionDescriptor& left,
                                const ColwolutionDescriptor& right);

// Creates an output tensor desciptor for the given parameters. Unless it's a
// 4D tensor, format must be TENSOR_NCHW and the generated output tensor is
// fully packed.
StatusOr<TensorDescriptor> CreateOutputDescriptor(
    const proto::TensorFormat& format, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ColwolutionDescriptor& colwolution);

// Returns the proto's output tensor if one is present, otherwise forwards to
// the function above.
StatusOr<TensorDescriptor> CreateOutputDescriptor(
    const proto::ColwolutionConfig& proto, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ColwolutionDescriptor& colwolution);

// Returns the number of bytes in device_memory_limit_mb flag that have not yet
// been allocated through AllocateDeviceMemory().
size_t GetAvailableDeviceMemoryBytes();

// Returns the workspace limit in bytes specified by the proto, or otherwise
// the device memory limit specified by the corresponding flag minus the number
// of bytes that have already been allocated.
StatusOr<size_t> GetWorkspaceLimit(const proto::ColwolutionConfig& proto);

// Returns the workspace size in bytes required to perform a colwolution with
// the given parameters.
StatusOr<size_t> GetWorkspaceSize(const LwdnnHandle& handle,
                                  const TensorDescriptor& input,
                                  const FilterDescriptor& filter,
                                  const ColwolutionDescriptor& colwolution,
                                  const TensorDescriptor& output,
                                  const ColwolutionAlgo& algo);

// Returns all algorithms that successfully return a workspace size no larger
// than the workspace_limit.
std::vector<ColwolutionAlgo> GetSupportedColwolutionAlgos(
    const LwdnnHandle& handle, const proto::ColwolutionDirection& direction,
    const TensorDescriptor& input, const FilterDescriptor& filter,
    const ColwolutionDescriptor& colwolution, const TensorDescriptor& output,
    size_t workspace_limit);

// Returns the fastest algorithm.
StatusOr<ColwolutionAlgo> FindColwolutionAlgo(
    const LwdnnHandle& handle, const proto::ColwolutionDirection& direction,
    const TensorDescriptor& input_desc, const DeviceMemory& input_data,
    const FilterDescriptor& filter_desc, const DeviceMemory& filter_data,
    const ColwolutionDescriptor& colwolution_desc,
    const TensorDescriptor& output_desc, const DeviceMemory& output_data,
    size_t workspace_limit);

// Performs colwolution.
Status RunColwolution(const LwdnnHandle& handle, const ColwolutionAlgo& algo,
                      double alpha, double beta,
                      const TensorDescriptor& input_desc,
                      const DeviceMemory& input_data,
                      const FilterDescriptor& filter_desc,
                      const DeviceMemory& filter_data,
                      const ColwolutionDescriptor& colwolution_desc,
                      const TensorDescriptor& output_desc,
                      const DeviceMemory& output_data,
                      const DeviceMemory& workspace, size_t workspace_size);

struct Colwolution {
  TensorDescriptor input_desc;
  FilterDescriptor filter_desc;
  TensorDescriptor output_desc;
  ColwolutionDescriptor colw_desc;
  DeviceMemory input_data;
  DeviceMemory filter_data;
  DeviceMemory output_data;
};

// Colwenience function to create everything needed to run a colwolution
// specified by proto.
StatusOr<Colwolution> CreateColwolution(const proto::ColwolutionConfig& proto,
                                        double data_lower, double data_upper,
                                        const RandomGenerator& rand_gen);

}  // namespace lwidia_libs_test

// This operator<< is in the global namespace in order to be found by ADL.
// That's important so it can be called from any namespace, not just the
// namespace it was declared in.
//
// ColwolutionAlgo is an instantiation of the absl::variant template class with
// types from the global namespace (::lwdnnColwolutionFwdAlgo_t etc.). The fact
// that the typedef is in the lwidia_libs_test namespace is irrelevant for ADL.
// It's undefined behavior to overload functions in the std namespace, and the
// same should apply for the namespace of the C++17 std::variant implementation
// used here. We are therefore left with implementing this function in the
// global namespace (the namespace of the template arguments).
std::ostream& operator<<(std::ostream& str,
                         const lwidia_libs_test::ColwolutionAlgo& algo);

#endif  // LWIDIA_LIBS_TEST_LWDNN_UTIL_H_
