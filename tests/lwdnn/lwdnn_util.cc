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

#include "lwdnn_util.h"

#include <sstream>

#include "gflags/gflags.h"
#include "base/googleinit.h"
#include "lwda_util.h"

DEFINE_int32(lwda_device, 0, "The LWCA device id to use");
DEFINE_int32(device_memory_limit_mb, 4096,
             "Maximum device memory to use for workspace after tensors have "
             "been allocated, in megabytes. Negative values specify an offset "
             "from the memory available at startup. Defaults to 4096.");

namespace lwidia_libs_test {
namespace {
size_t device_memory_limit_bytes = 0;

// Flag validator that initializes device_memory_limit_bytes.
bool ValidateDeviceMemoryLimit(const char*, int device_memory_limit_mb) {
  int device_id = FLAGS_lwda_device;
  int device_count = 0;
  CHECK_OK_STATUS(GetStatus(lwdaGetDeviceCount(&device_count)));
  CHECK_LT(device_id, device_count) << "Invalid LWCA device";
  CHECK_OK_STATUS(GetStatus(lwdaSetDevice(device_id)));
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CHECK_OK_STATUS(GetStatus(lwdaMemGetInfo(&free_bytes, &total_bytes)));
  auto limit_bytes = static_cast<ptrdiff_t>(device_memory_limit_mb) << 20;
  CHECK_GE(free_bytes, std::abs(limit_bytes))
      << "Available device memory is smaller than specified limit.";
  if (limit_bytes < 0) {
    // Use available device memory less flag value.
    device_memory_limit_bytes = free_bytes + limit_bytes;
  } else {
    // Use flag value.
    device_memory_limit_bytes = limit_bytes;
  }
  static bool result = [&] {
    lwdaDeviceProp device_prop;
    CHECK_OK_STATUS(
        GetStatus(lwdaGetDeviceProperties(&device_prop, device_id)));
    auto get_version_string = [](size_t version) {
      std::ostringstream oss;
      oss << version / 1000;
      version %= 1000;
      oss << "." << version / 100;
      version %= 100;
      oss << "." << version;
      return oss.str();
    };
    LOG(INFO) << "Running lwDNN v" << get_version_string(lwdnnGetVersion())
              << " for LWCA " << get_version_string(lwdnnGetLwdartVersion())
              << " on " << device_prop.name;
    return true;
  }();
  return result;
}
}  // namespace

REGISTER_MODULE_INITIALIZER(lwdnn_util, {
  // This runs during static initialization, before flags are parsed. Register
  // the validator to be called during gflags::ParseCommandLineFlags().
  gflags::RegisterFlagValidator(&FLAGS_device_memory_limit_mb,
                                &ValidateDeviceMemoryLimit);
});

Status GetStatus(lwdnnStatus_t status) {
  if (status == LWDNN_STATUS_SUCCESS) {
    return OkStatus();
  }
  const char* str = lwdnnGetErrorString(status);
  return ErrorStatus("lwDNN error '") << str << "'";
}

namespace detail {
void LwdnnHandleDeleter::operator()(lwdnnHandle_t handle) const {
  CHECK_OK_STATUS(GetStatus(lwdnnDestroy(handle)));
}

void TensorDescriptorDeleter::operator()(
    lwdnnTensorDescriptor_t descriptor) const {
  CHECK_OK_STATUS(GetStatus(lwdnnDestroyTensorDescriptor(descriptor)));
}

void FilterDescriptorDeleter::operator()(
    lwdnnFilterDescriptor_t descriptor) const {
  CHECK_OK_STATUS(GetStatus(lwdnnDestroyFilterDescriptor(descriptor)));
}

void ColwolutionDescriptorDeleter::operator()(
    lwdnnColwolutionDescriptor_t descriptor) const {
  CHECK_OK_STATUS(GetStatus(lwdnnDestroyColwolutionDescriptor(descriptor)));
}
}  // namespace detail

LwdnnHandle CreateLwdnnHandle() {
  lwdnnHandle_t result;
  CHECK_OK_STATUS(GetStatus(lwdnnCreate(&result)));
  return LwdnnHandle(result);
}

namespace {
TensorDescriptor CreateTensorDescriptor() {
  lwdnnTensorDescriptor_t result;
  CHECK_OK_STATUS(GetStatus(lwdnnCreateTensorDescriptor(&result)));
  return TensorDescriptor(result);
}

// Returns the strides for a fully packed tensor.
std::array<int, LWDNN_DIM_MAX> GetFullyPackedStrides(const int* dims,
                                                     int rank) {
  std::array<int, LWDNN_DIM_MAX> result;
  for (int i = rank - 1, stride = 1; i >= 0; --i) {
    result[i] = stride;
    stride *= dims[i];
  }
  return result;
}
}  // namespace

TensorDescriptor CreateTensorDescriptor(proto::TensorDescriptor proto) {
  CHECK_EQ(proto.data_type_oneof_case(), proto::TensorDescriptor::kDataType);
  CHECK_EQ(!proto.stride_size(),
           proto.format_oneof_case() == proto::TensorDescriptor::kFormat);
  int rank = proto.dimension_size();
  auto data_type = static_cast<lwdnnDataType_t>(proto.data_type());
  auto result = CreateTensorDescriptor();
  if (proto.stride_size()) {
    CHECK_EQ(rank, proto.stride_size());
    CHECK_OK_STATUS(GetStatus(lwdnnSetTensorNdDescriptor(
        result.get(), data_type, rank, proto.dimension().data(),
        proto.stride().data())));
  } else if (rank == 4) {
    CHECK_OK_STATUS(GetStatus(lwdnnSetTensor4dDescriptor(
        result.get(), static_cast<lwdnnTensorFormat_t>(proto.format()),
        data_type, proto.dimension(0), proto.dimension(1), proto.dimension(2),
        proto.dimension(3))));
  } else {
    CHECK_EQ(proto.format(), proto::TENSOR_NCHW);
    auto strides = GetFullyPackedStrides(proto.dimension().data(), rank);
    CHECK_OK_STATUS(GetStatus(
        lwdnnSetTensorNdDescriptor(result.get(), data_type, rank,
                                   proto.dimension().data(), strides.data())));
  }
  return result;
}

namespace {
struct TensorDescriptorData {
  lwdnnDataType_t data_type;
  int rank;
  int dimensions[LWDNN_DIM_MAX];
  int strides[LWDNN_DIM_MAX];
};

bool operator==(const TensorDescriptorData& left,
                const TensorDescriptorData& right) {
  return left.data_type == right.data_type && left.rank == right.rank &&
         std::equal(left.dimensions, left.dimensions + left.rank,
                    right.dimensions) &&
         std::equal(left.strides, left.strides + left.rank, right.strides);
}

TensorDescriptorData GetTensorDescriptorData(
    const lwdnnTensorDescriptor_t& tensor) {
  TensorDescriptorData data;
  CHECK_OK_STATUS(GetStatus(
      lwdnnGetTensorNdDescriptor(tensor, LWDNN_DIM_MAX, &data.data_type,
                                 &data.rank, data.dimensions, data.strides)));
  return data;
}
}  // namespace

bool TensorDescriptorEqual(const TensorDescriptor& left,
                           const TensorDescriptor& right) {
  return GetTensorDescriptorData(left.get()) ==
         GetTensorDescriptorData(right.get());
}

size_t GetTensorNumElements(const TensorDescriptor& tensor) {
  auto data = GetTensorDescriptorData(tensor.get());
  size_t result = 1;
  for (int i = 0; i < data.rank; ++i) {
    result += static_cast<size_t>(data.dimensions[i] - 1) * data.strides[i];
  }
  return result;
}

size_t GetTensorSizeInBytes(const TensorDescriptor& tensor) {
  size_t result = 0;
  CHECK_OK_STATUS(GetStatus(lwdnnGetTensorSizeInBytes(tensor.get(), &result)));
  return result;
}

lwdnnDataType_t GetTensorDataType(const TensorDescriptor& tensor) {
  return GetTensorDescriptorData(tensor.get()).data_type;
}

namespace {
StatusOr<DeviceMemory> CreateDeviceDataHelper(lwdnnDataType_t data_type,
                                              size_t num_elements, double lower,
                                              double upper,
                                              const RandomGenerator& rand_gen) {
  switch (data_type) {
    case LWDNN_DATA_FLOAT:
      return CreateDeviceData<float>(num_elements, lower, upper, rand_gen);
    case LWDNN_DATA_DOUBLE:
      return CreateDeviceData<double>(num_elements, lower, upper, rand_gen);
    case LWDNN_DATA_HALF:
      return CreateDeviceData<__half>(num_elements, lower, upper, rand_gen);
    default:
      LOG(FATAL) << "Not yet supported";
  }
}
}  // namespace

StatusOr<DeviceMemory> CreateTensorData(const TensorDescriptor& tensor,
                                        double lower, double upper,
                                        const RandomGenerator& rand_gen) {
  return CreateDeviceDataHelper(GetTensorDataType(tensor),
                                GetTensorNumElements(tensor), lower, upper,
                                rand_gen);
}

namespace {
FilterDescriptor CreateFilterDescriptor() {
  lwdnnFilterDescriptor_t result;
  CHECK_OK_STATUS(GetStatus(lwdnnCreateFilterDescriptor(&result)));
  return FilterDescriptor(result);
}
}  // namespace

FilterDescriptor CreateFilterDescriptor(const proto::FilterDescriptor& proto) {
  CHECK_EQ(proto.data_type_oneof_case(), proto::FilterDescriptor::kDataType);
  CHECK_EQ(proto.format_oneof_case(), proto::FilterDescriptor::kFormat);
  int rank = proto.dimension_size();
  auto result = CreateFilterDescriptor();
  CHECK_OK_STATUS(GetStatus(lwdnnSetFilterNdDescriptor(
      result.get(), static_cast<lwdnnDataType_t>(proto.data_type()),
      static_cast<lwdnnTensorFormat_t>(proto.format()), rank,
      proto.dimension().data())));
  return result;
}

namespace {
struct FilterDescriptorData {
  lwdnnDataType_t data_type;
  lwdnnTensorFormat_t format;
  int rank;
  int dimensions[LWDNN_DIM_MAX];
};

bool operator==(const FilterDescriptorData& left,
                const FilterDescriptorData& right) {
  return left.data_type == right.data_type && left.format == right.format &&
         left.rank == right.rank &&
         std::equal(left.dimensions, left.dimensions + left.rank,
                    right.dimensions);
}

FilterDescriptorData GetFilterDescriptorData(lwdnnFilterDescriptor_t filter) {
  FilterDescriptorData data{};
  CHECK_OK_STATUS(GetStatus(
      lwdnnGetFilterNdDescriptor(filter, LWDNN_DIM_MAX, &data.data_type,
                                 &data.format, &data.rank, data.dimensions)));
  return data;
}
}  // namespace

bool FilterDescriptorEqual(const FilterDescriptor& left,
                           const FilterDescriptor& right) {
  return GetFilterDescriptorData(left.get()) ==
         GetFilterDescriptorData(right.get());
}

size_t GetFilterNumElements(const FilterDescriptor& filter) {
  auto data = GetFilterDescriptorData(filter.get());
  size_t result = 1;
  for (int i = 0; i < data.rank; ++i) {
    result *= data.dimensions[i];
  }
  return result;
}

lwdnnDataType_t GetFilterDataType(const FilterDescriptor& filter) {
  return GetFilterDescriptorData(filter.get()).data_type;
}

StatusOr<DeviceMemory> CreateFilterData(const FilterDescriptor& filter,
                                        double lower, double upper,
                                        const RandomGenerator& rand_gen) {
  return CreateDeviceDataHelper(GetFilterDataType(filter),
                                GetFilterNumElements(filter), lower, upper,
                                rand_gen);
}

namespace {
ColwolutionDescriptor CreateColwolutionDescriptor() {
  lwdnnColwolutionDescriptor_t result;
  CHECK_OK_STATUS(GetStatus(lwdnnCreateColwolutionDescriptor(&result)));
  return ColwolutionDescriptor(result);
}

#if LWDNN_MAJOR < 7
// Forward-compatibility for grouped colwolution added in lwDNN 7.
lwdnnStatus_t lwdnnSetColwolutionGroupCount(lwdnnColwolutionDescriptor_t,
                                            int group_count) {
  CHECK_EQ(proto.group_count(), 1) << "Grouped colwolution requires lwDNN 7";
  return LWDNN_STATUS_SUCCESS;
}
lwdnnStatus_t lwdnnGetColwolutionGroupCount(lwdnnColwolutionDescriptor_t,
                                            int* group_count) {
  *group_count = 1;
  return LWDNN_STATUS_SUCCESS;
}

// Forward-compatibility for tensor math added in lwDNN 7.
typedef enum {
  LWDNN_DEFAULT_MATH = 0,
} lwdnnMathType_t;

lwdnnStatus_t lwdnnSetColwolutionMathType(lwdnnColwolutionDescriptor_t,
                                          lwdnnMathType_t math_type) {
  LOG_IF(WARNING, math_type != LWDNN_DEFAULT_MATH)
      << "Math type other than LWDNN_DEFAULT_MATH requires lwDNN 7";
  return LWDNN_STATUS_SUCCESS;
}
lwdnnStatus_t lwdnnGetColwolutionMathType(lwdnnColwolutionDescriptor_t,
                                          lwdnnMathType_t* math_type) {
  *math_type = LWDNN_DEFAULT_MATH;
  return LWDNN_STATUS_SUCCESS;
}
#endif
}  // namespace

ColwolutionDescriptor CreateColwolutionDescriptor(
    proto::ColwolutionDescriptor proto) {
  CHECK_EQ(proto.compute_mode_oneof_case(),
           proto::ColwolutionDescriptor::kComputeMode);
  int rank = std::max(
      {proto.pad_size(), proto.filter_stride_size(), proto.dilation_size()});
  while (proto.pad_size() < rank) {
    proto.add_pad(0);
  }
  while (proto.filter_stride_size() < rank) {
    proto.add_filter_stride(1);
  }
  while (proto.dilation_size() < rank) {
    proto.add_dilation(1);
  }
  auto result = CreateColwolutionDescriptor();
  // Note: proto.mode() returns COLWOLUTION if not set.
  CHECK_OK_STATUS(GetStatus(lwdnnSetColwolutionNdDescriptor(
      result.get(), rank, proto.pad().data(), proto.filter_stride().data(),
      proto.dilation().data(),
      static_cast<lwdnnColwolutionMode_t>(proto.mode()),
      static_cast<lwdnnDataType_t>(proto.compute_mode()))));
  if (proto.group_count() > 0) {
    CHECK_OK_STATUS(GetStatus(
        lwdnnSetColwolutionGroupCount(result.get(), proto.group_count())));
  }
  // Note: proto.math_type() returns DEFAULT_MATH if not set.
  CHECK_OK_STATUS(GetStatus(lwdnnSetColwolutionMathType(
      result.get(), static_cast<lwdnnMathType_t>(proto.math_type()))));
  return result;
}

namespace {

struct ColwolutionDescriptorData {
  int rank;
  int pad[LWDNN_DIM_MAX];
  int stride[LWDNN_DIM_MAX];
  int dilation[LWDNN_DIM_MAX];
  lwdnnColwolutionMode_t colwolution_mode;
  lwdnnDataType_t compute_type;
  lwdnnMathType_t math_type;
  int group_count;
};

bool operator==(const ColwolutionDescriptorData& left,
                const ColwolutionDescriptorData& right) {
  return left.colwolution_mode == right.colwolution_mode &&
         left.compute_type == right.compute_type && left.rank == right.rank &&
         std::equal(left.pad, left.pad + left.rank, right.pad) &&
         std::equal(left.stride, left.stride + left.rank, right.stride) &&
         std::equal(left.dilation, left.dilation + left.rank, right.dilation);
}

ColwolutionDescriptorData GetColwolutionDescriptorData(
    lwdnnColwolutionDescriptor_t colwolution) {
  ColwolutionDescriptorData data{};
  // array_length should be no larger than LWDNN_DIM_MAX according to the
  // documentation, but at least lwDNN 7 reports LWDNN_STATUS_NOT_SUPPORTED
  // for anything larger than 6.
  int array_length = 6;
  CHECK_OK_STATUS(GetStatus(lwdnnGetColwolutionNdDescriptor(
      colwolution, array_length, &data.rank, data.pad, data.stride,
      data.dilation, &data.colwolution_mode, &data.compute_type)));
  CHECK_OK_STATUS(
      GetStatus(lwdnnGetColwolutionMathType(colwolution, &data.math_type)));
  CHECK_OK_STATUS(
      GetStatus(lwdnnGetColwolutionGroupCount(colwolution, &data.group_count)));
  return data;
}
}  // namespace

bool ColwolutionDescriptorEqual(const ColwolutionDescriptor& left,
                                const ColwolutionDescriptor& right) {
  return GetColwolutionDescriptorData(left.get()) ==
         GetColwolutionDescriptorData(right.get());
}

StatusOr<TensorDescriptor> CreateOutputDescriptor(
    const proto::TensorFormat& format, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ColwolutionDescriptor& colwolution) {
  auto input_data = GetTensorDescriptorData(input.get());
  auto output = CreateTensorDescriptor();
  if (input_data.rank == 4) {
    int n, c, h, w;
    RETURN_IF_ERROR_STATUS(GetStatus(lwdnnGetColwolution2dForwardOutputDim(
        colwolution.get(), input.get(), filter.get(), &n, &c, &h, &w)));
    RETURN_IF_ERROR_STATUS(GetStatus(lwdnnSetTensor4dDescriptor(
        output.get(), static_cast<lwdnnTensorFormat_t>(format),
        GetTensorDataType(input), n, c, h, w)));
  } else {
    // TODO: Support other formats, dilations, strides, group counts.
    if (format != proto::TENSOR_NCHW) {
      return ErrorStatus("Can only create NCHW for non-4D output descriptor.");
    }
    auto filter_data = GetFilterDescriptorData(filter.get());
    CHECK_EQ(filter_data.format, LWDNN_TENSOR_NCHW) << "not supported";

    auto colw_data = GetColwolutionDescriptorData(colwolution.get());
    auto all_ones = [&](const int* param) {
      return std::all_of(param, param + colw_data.rank,
                         [](int value) { return value == 1; });
    };
    CHECK_EQ(all_ones(colw_data.dilation), true) << "not supported";
    CHECK_EQ(all_ones(colw_data.stride), true) << "not supported";
    CHECK_EQ(colw_data.group_count, 1) << "not supported";

    int output_dimensions[LWDNN_DIM_MAX] = {input_data.dimensions[0],
                                            filter_data.dimensions[0]};
    for (int i = 2; i < input_data.rank; ++i) {
      output_dimensions[i] = input_data.dimensions[i] +
                             2 * colw_data.pad[i - 2] -
                             filter_data.dimensions[i] + 1;
    }
    auto output_strides =
        GetFullyPackedStrides(output_dimensions, input_data.rank);
    RETURN_IF_ERROR_STATUS(GetStatus(lwdnnSetTensorNdDescriptor(
        output.get(), input_data.data_type, input_data.rank, output_dimensions,
        output_strides.data())));
  }
  return {std::move(output)};
}

StatusOr<TensorDescriptor> CreateOutputDescriptor(
    const proto::ColwolutionConfig& proto, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ColwolutionDescriptor& colwolution) {
  if (proto.has_output()) {
    return CreateTensorDescriptor(proto.output());
  }
  return CreateOutputDescriptor(proto.input().format(), input, filter,
                                colwolution);
}

size_t GetAvailableDeviceMemoryBytes() {
  size_t allocated = GetAllocatedDeviceMemoryBytes();
  return std::max(device_memory_limit_bytes, allocated) - allocated;
}

StatusOr<size_t> GetWorkspaceSize(const LwdnnHandle& handle,
                                  const TensorDescriptor& input,
                                  const FilterDescriptor& filter,
                                  const ColwolutionDescriptor& colwolution,
                                  const TensorDescriptor& output,
                                  const ColwolutionAlgo& algo) {
  struct Visitor {
    lwdnnStatus_t operator()(lwdnnColwolutionFwdAlgo_t algo) {
      return lwdnnGetColwolutionForwardWorkspaceSize(
          handle, input, filter, colwolution, output, algo, &workspace_size);
    }
    lwdnnStatus_t operator()(lwdnnColwolutionBwdDataAlgo_t algo) {
      return lwdnnGetColwolutionBackwardDataWorkspaceSize(
          handle, filter, output, colwolution, input, algo, &workspace_size);
    }
    lwdnnStatus_t operator()(lwdnnColwolutionBwdFilterAlgo_t algo) {
      return lwdnnGetColwolutionBackwardFilterWorkspaceSize(
          handle, input, output, colwolution, filter, algo, &workspace_size);
    }
    lwdnnHandle_t handle;
    lwdnnTensorDescriptor_t input;
    lwdnnFilterDescriptor_t filter;
    lwdnnColwolutionDescriptor_t colwolution;
    lwdnnTensorDescriptor_t output;
    size_t workspace_size;
  };
  Visitor visitor{handle.get(), input.get(), filter.get(), colwolution.get(),
                  output.get()};
  RETURN_IF_ERROR_STATUS(GetStatus(visit(visitor, algo)));
  return visitor.workspace_size;
}

StatusOr<size_t> GetWorkspaceLimit(const proto::ColwolutionConfig& proto) {
  size_t available = GetAvailableDeviceMemoryBytes();
  if (proto.workspace_oneof_case() !=
      proto::ColwolutionConfig::kWorkspaceLimit) {
    return available;
  }
  size_t limit = proto.workspace_limit();
  if (limit > available) {
    return ErrorStatus("Workspace limit (")
           << limit << " bytes) is larger than available memory (" << available
           << " bytes)";
  }
  return limit;
}

namespace {
template <typename T>
std::vector<ColwolutionAlgo> GetSupportedColwolutionAlgosImpl(
    const LwdnnHandle& handle, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ColwolutionDescriptor& colwolution,
    const TensorDescriptor& output, size_t workspace_limit, int num_elements) {
  // See dislwssion in the ColwolutionTest.GetAlgorithm_v7 test how this
  // function differs from lwdnnGetColwolution*Algorithm_v7.
  std::vector<ColwolutionAlgo> result;
  for (int i = 0; i < num_elements; ++i) {
    auto algo = static_cast<T>(i);
    auto size_or =
        GetWorkspaceSize(handle, input, filter, colwolution, output, algo);
    if (size_or.status().ok() && size_or.ValueOrDie() <= workspace_limit) {
      result.push_back(algo);
    }
  }
  return result;
}
}  // namespace

std::vector<ColwolutionAlgo> GetSupportedColwolutionAlgos(
    const LwdnnHandle& handle, const proto::ColwolutionDirection& direction,
    const TensorDescriptor& input, const FilterDescriptor& filter,
    const ColwolutionDescriptor& colwolution, const TensorDescriptor& output,
    size_t workspace_limit) {
  switch (direction) {
    case proto::COLWOLUTION_FWD:
      return GetSupportedColwolutionAlgosImpl<lwdnnColwolutionFwdAlgo_t>(
          handle, input, filter, colwolution, output, workspace_limit,
          LWDNN_COLWOLUTION_FWD_ALGO_COUNT);
    case proto::COLWOLUTION_BWD_DATA:
      return GetSupportedColwolutionAlgosImpl<lwdnnColwolutionBwdDataAlgo_t>(
          handle, input, filter, colwolution, output, workspace_limit,
          LWDNN_COLWOLUTION_BWD_DATA_ALGO_COUNT);
    case proto::COLWOLUTION_BWD_FILTER:
      return GetSupportedColwolutionAlgosImpl<lwdnnColwolutionBwdFilterAlgo_t>(
          handle, input, filter, colwolution, output, workspace_limit,
          LWDNN_COLWOLUTION_BWD_FILTER_ALGO_COUNT);
    default:
      LOG(FATAL) << "Unsupported: " << direction;
  }
}

namespace {
template <typename T>
StatusOr<ColwolutionAlgo> ToColwolutionAlgo(const T& algo_perf,
                                            int num_algorithms) {
  if (!num_algorithms || algo_perf.status != LWDNN_STATUS_SUCCESS) {
    return ErrorStatus("No supported algorithm");
  }
  return ColwolutionAlgo(algo_perf.algo);
}
}  // namespace

StatusOr<ColwolutionAlgo> FindColwolutionAlgo(
    const LwdnnHandle& handle, const proto::ColwolutionDirection& direction,
    const TensorDescriptor& input_desc, const DeviceMemory& input_data,
    const FilterDescriptor& filter_desc, const DeviceMemory& filter_data,
    const ColwolutionDescriptor& colwolution_desc,
    const TensorDescriptor& output_desc, const DeviceMemory& output_data,
    size_t workspace_limit) {
  ASSIGN_OR_RETURN_STATUS(auto workspace,
                          AllocateDeviceMemory(workspace_limit));
  int num_algorithms = 0;
  switch (direction) {
    case proto::COLWOLUTION_FWD: {
      lwdnnColwolutionFwdAlgoPerf_t algo_perf;
      RETURN_IF_ERROR_STATUS(GetStatus(lwdnnFindColwolutionForwardAlgorithmEx(
          handle.get(), input_desc.get(), input_data.get(), filter_desc.get(),
          filter_data.get(), colwolution_desc.get(), output_desc.get(),
          output_data.get(), 1, &num_algorithms, &algo_perf, workspace.get(),
          workspace_limit)));
      return ToColwolutionAlgo(algo_perf, num_algorithms);
    }
    case proto::COLWOLUTION_BWD_DATA: {
      lwdnnColwolutionBwdDataAlgoPerf_t algo_perf;
      RETURN_IF_ERROR_STATUS(
          GetStatus(lwdnnFindColwolutionBackwardDataAlgorithmEx(
              handle.get(), filter_desc.get(), filter_data.get(),
              output_desc.get(), output_data.get(), colwolution_desc.get(),
              input_desc.get(), input_data.get(), 1, &num_algorithms,
              &algo_perf, workspace.get(), workspace_limit)));
      return ToColwolutionAlgo(algo_perf, num_algorithms);
    }
    case proto::COLWOLUTION_BWD_FILTER: {
      lwdnnColwolutionBwdFilterAlgoPerf_t algo_perf;
      RETURN_IF_ERROR_STATUS(
          GetStatus(lwdnnFindColwolutionBackwardFilterAlgorithmEx(
              handle.get(), input_desc.get(), input_data.get(),
              output_desc.get(), output_data.get(), colwolution_desc.get(),
              filter_desc.get(), filter_data.get(), 1, &num_algorithms,
              &algo_perf, workspace.get(), workspace_limit)));
      return ToColwolutionAlgo(algo_perf, num_algorithms);
    }
    default:
      return ErrorStatus("Unsupported: ") << direction;
  }
}

namespace {
// The scaling factor parameters 'alpha' and 'beta' of the lwdnnTransform* and
// lwdnnColwolution* functions are type punned pointers. The storage type is
// double for double output tensors, and float otherwise.
union ScalingFactor {
  ScalingFactor(double value, lwdnnTensorDescriptor_t descriptor)
      : ScalingFactor(value, GetTensorDescriptorData(descriptor).data_type) {}
  ScalingFactor(double value, lwdnnFilterDescriptor_t descriptor)
      : ScalingFactor(value, GetFilterDescriptorData(descriptor).data_type) {}

 private:
  ScalingFactor(double value, lwdnnDataType_t data_type) {
    if (data_type == LWDNN_DATA_DOUBLE) {
      double_value = value;
    } else {
      float_value = static_cast<float>(value);
    }
  }

  float float_value;
  double double_value;
};
}  // namespace

Status TransformTensor(const LwdnnHandle& handle, double alpha, double beta,
                       const TensorDescriptor& src_desc,
                       const DeviceMemory& src_data,
                       const TensorDescriptor& dst_desc,
                       const DeviceMemory& dst_data) {
  ScalingFactor alpha_scale(alpha, dst_desc.get());
  ScalingFactor beta_scale(beta, dst_desc.get());
  RETURN_IF_ERROR_STATUS(GetStatus(lwdnnTransformTensor(
      handle.get(), &alpha_scale, src_desc.get(), src_data.get(), &beta_scale,
      dst_desc.get(), dst_data.get())));
  return OkStatus();
}

Status TransformTensor(const LwdnnHandle& handle,
                       const TensorDescriptor& src_desc,
                       const DeviceMemory& src_data,
                       const TensorDescriptor& dst_desc,
                       const DeviceMemory& dst_data) {
  return TransformTensor(handle, 1.0, 0.0, src_desc, src_data, dst_desc,
                         dst_data);
}

namespace {
struct ColwertDeviceDataVisitor {
  template <typename DstT, typename SrcT>
  void operator()(DstT* dst, const SrcT* src) {
    ColwertDeviceData(scale, dst, src, dst_size_in_bytes / sizeof(DstT));
  }
  template <typename T>
  void operator()(T*, const T* src) {
    LOG(FATAL) << "No colwersion needed";
  }
  size_t dst_size_in_bytes;
  double scale;
};

absl::variant<float*, double*, __half*> GetPointerVariant(
    const DeviceMemory& data, lwdnnDataType_t data_type) {
  switch (data_type) {
    case LWDNN_DATA_FLOAT:
      return static_cast<float*>(data.get());
    case LWDNN_DATA_DOUBLE:
      return static_cast<double*>(data.get());
    case LWDNN_DATA_HALF:
      return static_cast<__half*>(data.get());
    default:
      LOG(FATAL) << "Not yet implemented";
  }
}
}  // namespace

Status ColwertAndTransformTensor(const LwdnnHandle& handle, double alpha,
                                 double beta, const TensorDescriptor& src_desc,
                                 const DeviceMemory& src_data,
                                 const TensorDescriptor& dst_desc,
                                 const DeviceMemory& dst_data) {
  auto src_desc_data = GetTensorDescriptorData(src_desc.get());
  auto dst_desc_data = GetTensorDescriptorData(dst_desc.get());
  if (src_desc_data.data_type == dst_desc_data.data_type) {
    return TransformTensor(handle, alpha, beta, src_desc, src_data, dst_desc,
                           dst_data);
  }
  CHECK_EQ(src_desc_data.rank, dst_desc_data.rank);
  auto temp_desc = CreateTensorDescriptor();
  RETURN_IF_ERROR_STATUS(GetStatus(lwdnnSetTensorNdDescriptor(
      temp_desc.get(), dst_desc_data.data_type, src_desc_data.rank,
      src_desc_data.dimensions, src_desc_data.strides)));

  size_t temp_size = GetTensorSizeInBytes(temp_desc);
  ASSIGN_OR_RETURN_STATUS(auto temp_data, AllocateDeviceMemory(temp_size));

  visit(ColwertDeviceDataVisitor{temp_size, alpha},
        GetPointerVariant(temp_data, dst_desc_data.data_type),
        GetPointerVariant(src_data, src_desc_data.data_type));

  RETURN_IF_ERROR_STATUS(GetStatus(lwdaDeviceSynchronize()));
  return TransformTensor(handle, 1.0, beta, temp_desc, temp_data, dst_desc,
                         dst_data);
}

namespace {
class PrintVisitor {
 public:
  PrintVisitor(std::ostringstream* oss, size_t size_in_bytes)
      : oss_(oss), size_in_bytes_(size_in_bytes) {}

  template <typename T>
  void operator()(T* device_ptr) {
    size_t num_elements = size_in_bytes_ / sizeof(T);
    std::unique_ptr<T[]> host_ptr(new T[num_elements]);
    CHECK_OK_STATUS(GetStatus(lwdaMemcpy(
        host_ptr.get(), device_ptr, size_in_bytes_, lwdaMemcpyDeviceToHost)));
    Print(host_ptr.get(), num_elements);
  }

  void operator()(__half* device_ptr) {
    size_t num_elements = size_in_bytes_ / sizeof(__half);
    auto host_memory = std::move(
        AllocateHostMemory(num_elements * sizeof(float)).ValueOrDie());
    auto host_ptr = static_cast<float*>(host_memory.get());
    ColwertDeviceData(1.0, host_ptr, device_ptr, num_elements);
    Print(host_ptr, num_elements);
  }

 private:
  template <typename T>
  void Print(const T* ptr, size_t num_elements) {
    CHECK_OK_STATUS(GetStatus(lwdaDeviceSynchronize()));
    for (size_t i = 0; i < num_elements; ++i) {
      *oss_ << " " << ptr[i];
    }
  }

  std::ostringstream* oss_;
  size_t size_in_bytes_;
};
}  // namespace

string GetTensorDebugString(const TensorDescriptor& desc,
                            const DeviceMemory& data, bool print_values) {
  std::ostringstream oss;
  auto desc_data = GetTensorDescriptorData(desc.get());
  oss << "data_type: "
      << proto::DataType_Name(
             static_cast<proto::DataType>(desc_data.data_type));
  oss << "\ndimensions:";
  for (int i = 0; i < desc_data.rank; ++i) {
    oss << " " << desc_data.dimensions[i];
  }
  oss << "\nstrides:";
  for (int i = 0; i < desc_data.rank; ++i) {
    oss << " " << desc_data.strides[i];
  }
  if (print_values) {
    oss << "\lwalues:";
    visit(PrintVisitor(&oss, GetTensorSizeInBytes(desc)),
          GetPointerVariant(data, desc_data.data_type));
  }
  return oss.str();
}

Status RunColwolution(const LwdnnHandle& handle, const ColwolutionAlgo& algo,
                      double alpha, double beta,
                      const TensorDescriptor& input_desc,
                      const DeviceMemory& input_data,
                      const FilterDescriptor& filter_desc,
                      const DeviceMemory& filter_data,
                      const ColwolutionDescriptor& colwolution_desc,
                      const TensorDescriptor& output_desc,
                      const DeviceMemory& output_data,
                      const DeviceMemory& workspace, size_t workspace_size) {
  struct Visitor {
    lwdnnStatus_t operator()(lwdnnColwolutionFwdAlgo_t algo) {
      ScalingFactor alpha_scale(alpha, output_desc);
      ScalingFactor beta_scale(beta, output_desc);
      return lwdnnColwolutionForward(
          handle, &alpha_scale, input_desc, input_data, filter_desc,
          filter_data, colwolution_desc, algo, workspace, workspace_size,
          &beta_scale, output_desc, output_data);
    }
    lwdnnStatus_t operator()(lwdnnColwolutionBwdDataAlgo_t algo) {
      ScalingFactor alpha_scale(alpha, input_desc);
      ScalingFactor beta_scale(beta, input_desc);
      return lwdnnColwolutionBackwardData(
          handle, &alpha_scale, filter_desc, filter_data, output_desc,
          output_data, colwolution_desc, algo, workspace, workspace_size,
          &beta_scale, input_desc, input_data);
    }
    lwdnnStatus_t operator()(lwdnnColwolutionBwdFilterAlgo_t algo) {
      ScalingFactor alpha_scale(alpha, filter_desc);
      ScalingFactor beta_scale(beta, filter_desc);
      return lwdnnColwolutionBackwardFilter(
          handle, &alpha_scale, input_desc, input_data, output_desc,
          output_data, colwolution_desc, algo, workspace, workspace_size,
          &beta_scale, filter_desc, filter_data);
    }
    lwdnnHandle_t handle;
    double alpha, beta;
    lwdnnTensorDescriptor_t input_desc;
    void* input_data;
    lwdnnFilterDescriptor_t filter_desc;
    void* filter_data;
    lwdnnColwolutionDescriptor_t colwolution_desc;
    lwdnnTensorDescriptor_t output_desc;
    void* output_data;
    void* workspace;
    size_t workspace_size;
  };
  Visitor visitor{handle.get(),
                  alpha,
                  beta,
                  input_desc.get(),
                  input_data.get(),
                  filter_desc.get(),
                  filter_data.get(),
                  colwolution_desc.get(),
                  output_desc.get(),
                  output_data.get(),
                  workspace.get(),
                  workspace_size};
  return GetStatus(visit(visitor, algo));
}

StatusOr<Colwolution> CreateColwolution(const proto::ColwolutionConfig& proto,
                                        double data_lower, double data_upper,
                                        const RandomGenerator& rand_gen) {
  if (LWDNN_MAJOR < 7 && proto.colwolution().group_count() > 1) {
    return ErrorStatus("Grouped colwolution requires lwDNN 7");
  }

  auto input_desc = CreateTensorDescriptor(proto.input());
  auto filter_desc = CreateFilterDescriptor(proto.filter());
  auto colw_desc = CreateColwolutionDescriptor(proto.colwolution());

  ASSIGN_OR_RETURN_STATUS(
      auto output_desc,
      CreateOutputDescriptor(proto, input_desc, filter_desc, colw_desc));

  ASSIGN_OR_RETURN_STATUS(
      auto input_data,
      CreateTensorData(input_desc, data_lower, data_upper, rand_gen));

  ASSIGN_OR_RETURN_STATUS(
      auto filter_data,
      CreateFilterData(filter_desc, data_lower, data_upper, rand_gen));

  ASSIGN_OR_RETURN_STATUS(
      auto output_data,
      CreateTensorData(output_desc, data_lower, data_upper, rand_gen));

  return Colwolution{std::move(input_desc),  std::move(filter_desc),
                     std::move(output_desc), std::move(colw_desc),
                     std::move(input_data),  std::move(filter_data),
                     std::move(output_data)};
}

namespace {
string GetAlgoName(const ColwolutionAlgo& algo) {
  struct Visitor {
    string operator()(lwdnnColwolutionFwdAlgo_t algo) const {
      return proto::ColwolutionFwdAlgo_Name(
          static_cast<proto::ColwolutionFwdAlgo>(algo));
    }
    string operator()(lwdnnColwolutionBwdDataAlgo_t algo) const {
      return proto::ColwolutionBwdDataAlgo_Name(
          static_cast<proto::ColwolutionBwdDataAlgo>(algo));
    }
    string operator()(lwdnnColwolutionBwdFilterAlgo_t algo) const {
      return proto::ColwolutionBwdFilterAlgo_Name(
          static_cast<proto::ColwolutionBwdFilterAlgo>(algo));
    }
  };
  return visit(Visitor(), algo);
}

#define CHECK_ENUMERATOR(enumerator)                      \
  static_assert(static_cast<int>(proto::enumerator) ==    \
                    static_cast<int>(LWDNN_##enumerator), \
                "enum values don't match")

#define CHECK_ENUM_SIZE(proto_enum, lwdnn_enum)                   \
  static_assert(proto::proto_enum##_ARRAYSIZE ==                  \
                    static_cast<int>(LWDNN_##lwdnn_enum##_COUNT), \
                "size does not match")

CHECK_ENUMERATOR(COLWOLUTION);
CHECK_ENUMERATOR(CROSS_CORRELATION);
CHECK_ENUMERATOR(DATA_FLOAT);
CHECK_ENUMERATOR(DATA_DOUBLE);
CHECK_ENUMERATOR(DATA_HALF);
CHECK_ENUMERATOR(DATA_INT8);
CHECK_ENUMERATOR(DATA_INT32);
CHECK_ENUMERATOR(DATA_INT8x4);
CHECK_ENUMERATOR(TENSOR_NCHW);
CHECK_ENUMERATOR(TENSOR_NHWC);
CHECK_ENUMERATOR(TENSOR_NCHW_VECT_C);
CHECK_ENUMERATOR(COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM);
CHECK_ENUMERATOR(COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
CHECK_ENUMERATOR(COLWOLUTION_FWD_ALGO_GEMM);
CHECK_ENUMERATOR(COLWOLUTION_FWD_ALGO_DIRECT);
CHECK_ENUMERATOR(COLWOLUTION_FWD_ALGO_FFT);
CHECK_ENUMERATOR(COLWOLUTION_FWD_ALGO_FFT_TILING);
CHECK_ENUMERATOR(COLWOLUTION_FWD_ALGO_WINOGRAD);
CHECK_ENUMERATOR(COLWOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
CHECK_ENUM_SIZE(ColwolutionFwdAlgo, COLWOLUTION_FWD_ALGO);
CHECK_ENUMERATOR(COLWOLUTION_BWD_DATA_ALGO_0);
CHECK_ENUMERATOR(COLWOLUTION_BWD_DATA_ALGO_1);
CHECK_ENUMERATOR(COLWOLUTION_BWD_DATA_ALGO_FFT);
CHECK_ENUMERATOR(COLWOLUTION_BWD_DATA_ALGO_FFT_TILING);
CHECK_ENUMERATOR(COLWOLUTION_BWD_DATA_ALGO_WINOGRAD);
CHECK_ENUMERATOR(COLWOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
CHECK_ENUM_SIZE(ColwolutionBwdDataAlgo, COLWOLUTION_BWD_DATA_ALGO);
CHECK_ENUMERATOR(COLWOLUTION_BWD_FILTER_ALGO_0);
CHECK_ENUMERATOR(COLWOLUTION_BWD_FILTER_ALGO_1);
CHECK_ENUMERATOR(COLWOLUTION_BWD_FILTER_ALGO_FFT);
CHECK_ENUMERATOR(COLWOLUTION_BWD_FILTER_ALGO_3);
CHECK_ENUMERATOR(COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD);
CHECK_ENUMERATOR(COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
CHECK_ENUMERATOR(COLWOLUTION_BWD_FILTER_ALGO_FFT_TILING);
CHECK_ENUM_SIZE(ColwolutionBwdFilterAlgo, COLWOLUTION_BWD_FILTER_ALGO);

#if LWDNN_MAJOR >= 7
CHECK_ENUMERATOR(DEFAULT_MATH);
CHECK_ENUMERATOR(TENSOR_OP_MATH);
#endif
}  // namespace
}  // namespace lwidia_libs_test

std::ostream& operator<<(std::ostream& str,
                         const lwidia_libs_test::ColwolutionAlgo& algo) {
  return str << lwidia_libs_test::GetAlgoName(algo);
}
