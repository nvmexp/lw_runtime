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

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "google/protobuf/repeated_field.h"
#include "gtest/gtest.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "lwdnn/include/lwdnn.h"
#include "all_pairs.h"
#include "lwda_util.h"
#include "lwdnn.pb.h"
#include "lwdnn_test.h"
#include "lwdnn_util.h"
#include "test_util.h"

// Tests that compare the outputs of all supported colwolution algorithms for
// specific configuration of input, filter, etc. The configurations are loaded
// from lwdnn_tests.textproto and generated using the all-pair testing approach.
//
// For lwDNN 7 and later, also tests that the lwdnnGetColwolution*Algorithm_v7
// returns the same set of algorithms as GetSupportedColwolutionAlgos.

namespace lwidia_libs_test {
namespace {
class ColwolutionTest
    : public ::testing::TestWithParam<proto::ColwolutionTest> {
  void TearDown() override {
    if (HasFatalFailure()) {
      ResetDevice();
    }
  }
};

// Returns expected accuracy of the given algorithm. The tolerance is a mix
// between absolute and relative error, see test_util.h.
double GetTolerance(const ColwolutionAlgo& algo) {
  struct Visitor {
    double operator()(lwdnnColwolutionFwdAlgo_t algo) {
      switch (algo) {
        case LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM:
          return 1e-4;
        case LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
          return 1e-4;
        case LWDNN_COLWOLUTION_FWD_ALGO_GEMM:
          return 1e-4;
        case LWDNN_COLWOLUTION_FWD_ALGO_DIRECT:
          return 0.0;  // Not implemented.
        case LWDNN_COLWOLUTION_FWD_ALGO_FFT:
          return 1e-5;
        case LWDNN_COLWOLUTION_FWD_ALGO_FFT_TILING:
          return 1e-5;
        case LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD:
          return 1e-5;
        case LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
          return 1e-2;
        default:
          LOG(FATAL) << "Unknown algorithm";
      }
    }
    double operator()(lwdnnColwolutionBwdDataAlgo_t algo) {
      switch (algo) {
        case LWDNN_COLWOLUTION_BWD_DATA_ALGO_0:
          return 1e-5;
        case LWDNN_COLWOLUTION_BWD_DATA_ALGO_1:
          return 1e-4;
        case LWDNN_COLWOLUTION_BWD_DATA_ALGO_FFT:
          return 1e-5;
        case LWDNN_COLWOLUTION_BWD_DATA_ALGO_FFT_TILING:
          return 1e-4;
        case LWDNN_COLWOLUTION_BWD_DATA_ALGO_WINOGRAD:
          return 1e-5;
        case LWDNN_COLWOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
          return 1e-2;
        default:
          LOG(FATAL) << "Unknown algorithm";
      }
    }
    double operator()(lwdnnColwolutionBwdFilterAlgo_t algo) {
      switch (algo) {
        case LWDNN_COLWOLUTION_BWD_FILTER_ALGO_0:
          return 1e-5;
        case LWDNN_COLWOLUTION_BWD_FILTER_ALGO_1:
          return 1e-4;
        case LWDNN_COLWOLUTION_BWD_FILTER_ALGO_FFT:
          return 1e-5;
        case LWDNN_COLWOLUTION_BWD_FILTER_ALGO_3:
          return 1e-4;
        case LWDNN_COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD:
          return 0.0;  // Not implemented.
        case LWDNN_COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
          return 1e-2;
        case LWDNN_COLWOLUTION_BWD_FILTER_ALGO_FFT_TILING:
          return 1e-5;
        default:
          LOG(FATAL) << "Unknown algorithm";
      }
    }
  };
  return visit(Visitor(), algo);
}

// Returns expected accuracy of the colwolution based on the data types
// configuration specified in the proto, relative to FLOAT_CONFIG (see lwDNN
// documentation).
double GetToleranceScale(const proto::ColwolutionConfig& proto) {
  switch (proto.filter().data_type()) {
    case proto::DATA_FLOAT:
      return 1.0;
    case proto::DATA_DOUBLE:
      // lwDNN uses different kernels for architectures before Maxwell that have
      // much lower double accuracy.
      return DeviceHasAtLeastComputeCapability(5, 0) ? 1e-8 : 1e-2;
    case proto::DATA_HALF:
      if (proto.colwolution().math_type() == proto::TENSOR_OP_MATH) {
        return 1e+4;  // Using tensor ops.
      }
      if (proto.colwolution().compute_mode() == proto::DATA_FLOAT) {
        return 1e+5;  // Pseudo half config.
      }
      return 1e+5;  // True half config.
    default:
      LOG(FATAL) << "Not yet supported";
  }
}

// Returns a tensor descriptor with the same layout as the given filter
// descriptor.
proto::TensorDescriptor GetFilterTensorDescriptor(
    const proto::FilterDescriptor& filter) {
  proto::TensorDescriptor result;
  *result.mutable_dimension() = filter.dimension();
  result.set_data_type(filter.data_type());
  if (result.dimension_size() == 4) {
    result.set_format(filter.format());
    return result;
  }
  // If filter rank is not 4, it must be fully packed.
  CHECK_EQ(filter.format(), proto::TENSOR_NCHW);
  int rank = result.dimension_size();
  auto* strides = result.mutable_stride();
  strides->Resize(rank, 0);
  for (int i = rank - 1, stride = 1; i >= 0; --i) {
    strides->Set(i, stride);
    stride *= filter.dimension(i);
  }
  return result;
}

// Returns the algorithms specified in the proto.
std::vector<ColwolutionAlgo> GetAlgorithms(
    const proto::ColwolutionConfig& proto, const LwdnnHandle& handle,
    const Colwolution& colwolution, size_t workspace_limit) {
  switch (proto.algo_oneof_case()) {
    case proto::ColwolutionConfig::kAllAlgos: {
      return GetSupportedColwolutionAlgos(
          handle, proto.all_algos(), colwolution.input_desc,
          colwolution.filter_desc, colwolution.colw_desc,
          colwolution.output_desc, workspace_limit);
    }
    case proto::ColwolutionConfig::kFwdAlgo:
      return {static_cast<lwdnnColwolutionFwdAlgo_t>(proto.fwd_algo())};
    case proto::ColwolutionConfig::kBwdDataAlgo:
      return {
          static_cast<lwdnnColwolutionBwdDataAlgo_t>(proto.bwd_data_algo())};
    case proto::ColwolutionConfig::kBwdFilterAlgo:
      return {static_cast<lwdnnColwolutionBwdFilterAlgo_t>(
          proto.bwd_filter_algo())};
    default:
      LOG(FATAL) << "Invalid algo_oneof case";
  }
}

// Runs the colwolution.
Status RunColwolution(double alpha, double beta, const LwdnnHandle& handle,
                      const Colwolution& colwolution,
                      const ColwolutionAlgo& algo) {
  ASSIGN_OR_RETURN_STATUS(
      auto workspace_size,
      GetWorkspaceSize(handle, colwolution.input_desc, colwolution.filter_desc,
                       colwolution.colw_desc, colwolution.output_desc, algo));

  ASSIGN_OR_RETURN_STATUS(auto workspace, AllocateDeviceMemory(workspace_size));
  FillWithNaNs(workspace);

  RETURN_IF_ERROR_STATUS(RunColwolution(
      handle, algo, alpha, beta, colwolution.input_desc, colwolution.input_data,
      colwolution.filter_desc, colwolution.filter_data, colwolution.colw_desc,
      colwolution.output_desc, colwolution.output_data, workspace,
      workspace_size));

  // Detect and handle LWCA errors to avoid hitting the CHECK_OK_STATUS when
  // deallocating the workspace.
  return GetStatus(lwdaDeviceSynchronize());
}

// Returns 'from' merged into 'proto', handling repeated and oneof fields.
proto::ColwolutionConfig GetMergedColwolution(
    proto::ColwolutionConfig proto, const proto::ColwolutionConfig& from) {
  auto process_tensor_desc = [](proto::TensorDescriptor* proto_tensor,
                                const proto::TensorDescriptor& from_tensor) {
    if (from_tensor.dimension_size()) {
      proto_tensor->clear_dimension();
    }
    if (from_tensor.data_type_oneof_case()) {
      proto_tensor->clear_data_type();
    }
    if (from_tensor.format_oneof_case()) {
      proto_tensor->clear_format();
    }
    if (from_tensor.stride_size()) {
      proto_tensor->clear_stride();
    }
  };

  // Clear all repeated and oneof fields in 'proto' that are set in 'from'.
  process_tensor_desc(proto.mutable_input(), from.input());
  if (from.has_output()) {
    process_tensor_desc(proto.mutable_output(), from.output());
  }

  if (from.filter().dimension_size()) {
    proto.mutable_filter()->clear_dimension();
  }
  if (from.filter().data_type_oneof_case()) {
    proto.mutable_filter()->clear_data_type();
  }
  if (from.filter().format_oneof_case()) {
    proto.mutable_filter()->clear_format();
  }

  if (from.colwolution().pad_size()) {
    proto.mutable_colwolution()->clear_pad();
  }
  if (from.colwolution().filter_stride_size()) {
    proto.mutable_colwolution()->clear_filter_stride();
  }
  if (from.colwolution().dilation_size()) {
    proto.mutable_colwolution()->clear_dilation();
  }
  if (from.colwolution().compute_mode_oneof_case()) {
    proto.mutable_colwolution()->clear_compute_mode();
  }

  if (from.algo_oneof_case()) {
    proto.set_all_algos(proto::COLWOLUTION_DIRECTION_UNSPECIFIED);
  }

  proto.MergeFrom(from);

  return proto;
}

TEST_P(ColwolutionTest, CompareResults) {
  VLOG(2) << "Running all-test for\n" << GetParam().DebugString();

  LwdnnHandle handle = CreateLwdnnHandle();
  RandomGenerator rand_gen(/*seed=*/0);

  // Run reference colwolution first, then compare others to the reference
  // result.
  double data_lower = GetParam().values_lower_bound(), data_upper = 1.;
  if (GetParam().values_bound_oneof_case() ==
      proto::ColwolutionTest::kValuesUpperBound) {
    data_upper = GetParam().values_upper_bound();
  }

  auto ref_proto = GetParam().reference();
  ASSERT_OK_AND_ASSIGN(auto reference, CreateColwolution(ref_proto, data_lower,
                                                         data_upper, rand_gen));

  ASSERT_OK_AND_ASSIGN(auto workspace_limit, GetWorkspaceLimit(ref_proto));
  auto ref_algos = GetAlgorithms(ref_proto, handle, reference, workspace_limit);
  ASSERT_FALSE(ref_algos.empty()) << "No algorithm found";
  auto ref_algo = ref_algos.front();

  // Run reference colwolution with default scaling factors.
  ASSERT_TRUE(IsOk(RunColwolution(1.0, 0.0, handle, reference, ref_algo)))
      << "algo: " << ref_algo;

  auto ref_filter_tensor_desc =
      CreateTensorDescriptor(GetFilterTensorDescriptor(ref_proto.filter()));

  for (auto proto : GetParam().test()) {
    proto = GetMergedColwolution(ref_proto, proto);
    ASSERT_OK_AND_ASSIGN(
        Colwolution test,
        CreateColwolution(proto, data_lower, data_upper, rand_gen));

    // We now have input, filter, and output buffers for the reference and the
    // test. A colwolution has two read-only argument buffers and writes the
    // result to the remaining buffer. The colwolution direction determines
    // which buffers are arguments and which buffer is the result.
    //
    // colwolution direction:    argument buffers:     result buffer:
    // --------------------------------------------------------------
    // forward                   input, filter         output
    // backward data             filter, output        input
    // backward filter           output, input         filter
    //
    // The reference buffers contain the arguments and ground truth result of
    // the colwolution. The test buffers are filled with random values at this
    // point. Tensors aren't necessarily packed and we use the random value of
    // buffer elements not referenced by the tensor to detect out-of-bounds
    // access by lwDNN.
    //
    // If beta is zero, which means the result buffer values shouldn't be read,
    // we populate result buffer with NaNs before sending it to the colwolution.

    auto direction = [&] {
      switch (proto.algo_oneof_case()) {
        case proto::ColwolutionConfig::kAllAlgos:
          return proto.all_algos();
        case proto::ColwolutionConfig::kFwdAlgo:
          return proto::COLWOLUTION_FWD;
        case proto::ColwolutionConfig::kBwdDataAlgo:
          return proto::COLWOLUTION_BWD_DATA;
        case proto::ColwolutionConfig::kBwdFilterAlgo:
          return proto::COLWOLUTION_BWD_FILTER;
        default:
          LOG(FATAL) << "Invalid algo_oneof case";
      }
    }();

    auto filter_tensor_desc =
        CreateTensorDescriptor(GetFilterTensorDescriptor(proto.filter()));
    double alpha = 1.0 - proto.one_minus_alpha();
    double beta = proto.beta();

    // Get a pointer to the test result buffer and its tensor descriptor.
    TensorDescriptor* result_desc = nullptr;
    DeviceMemory* result_data = nullptr;
    struct {
      double alpha = 1.0, beta = 0.0;
    } input_scaling, filter_scaling, output_scaling;
    switch (direction) {
      case proto::COLWOLUTION_FWD:
        result_desc = &test.output_desc;
        result_data = &test.output_data;
        output_scaling.alpha = alpha;
        output_scaling.beta = beta;
        break;
      case proto::COLWOLUTION_BWD_DATA:
        result_desc = &test.input_desc;
        result_data = &test.input_data;
        input_scaling.alpha = alpha;
        input_scaling.beta = beta;
        break;
      case proto::COLWOLUTION_BWD_FILTER:
        result_desc = &filter_tensor_desc;
        result_data = &test.filter_data;
        filter_scaling.alpha = alpha;
        filter_scaling.beta = beta;
        break;
      default:
        LOG(FATAL) << "Invalid direction: "
                   << proto::ColwolutionDirection_Name(direction);
    }
    size_t result_size = GetTensorSizeInBytes(*result_desc);

    DeviceMemory init_data(nullptr);
    if (beta > 0) {
      // Make a copy of the test result buffer so that we can reset it to the
      // same random values before running each algorithm.
      ASSERT_OK_AND_ASSIGN(init_data, AllocateDeviceMemory(result_size));
      ASSERT_TRUE(IsOk(CopyDeviceMemory(init_data, *result_data, result_size)));
    }
    // Reset the test result buffer and run the colwolution.
    const auto restore_result_buffer = [&] {
      if (beta > 0) {
        ASSERT_TRUE(
            IsOk(CopyDeviceMemory(*result_data, init_data, result_size)));
      } else {
        FillWithNaNs(*result_data);
      }
    };

    // Blend the reference buffers into the corresponding test buffers. This
    // may involve colwerting the data type (e.g. from double to float) and
    // changing the data layout (e.g. from NCHW to NHWC).
    ASSERT_TRUE(IsOk(ColwertAndTransformTensor(
        handle, input_scaling.alpha, input_scaling.beta, reference.input_desc,
        reference.input_data, test.input_desc, test.input_data)));
    ASSERT_TRUE(IsOk(ColwertAndTransformTensor(
        handle, filter_scaling.alpha, filter_scaling.beta,
        ref_filter_tensor_desc, reference.filter_data, filter_tensor_desc,
        test.filter_data)));
    ASSERT_TRUE(IsOk(ColwertAndTransformTensor(
        handle, output_scaling.alpha, output_scaling.beta,
        reference.output_desc, reference.output_data, test.output_desc,
        test.output_data)));

    // The test result buffer (pointee of result_data) now contains the copy of
    // the reference result buffer. Stash that away in ref_result_data and
    // allocate a new buffer for the test's actual result buffer.
    ASSERT_OK_AND_ASSIGN(auto ref_result_data,
                         AllocateDeviceMemory(result_size));
    std::swap(*result_data, ref_result_data);

    // Get available workspace size (after allocating all device memory).
    ASSERT_OK_AND_ASSIGN(auto workspace_limit, GetWorkspaceLimit(proto));

    // Run all supported algorithms for current test.
    double tolerance_scale = GetToleranceScale(proto);
    for (const auto& algo :
         GetAlgorithms(proto, handle, test, workspace_limit)) {
      restore_result_buffer();
      auto get_message = [&] {
        std::ostringstream oss;
        oss << "format: " << proto::TensorFormat_Name(proto.input().format())
            << "\ndata_type: "
            << proto::DataType_Name(proto.input().data_type())
            << "\ncompute_mode: "
            << proto::DataType_Name(proto.colwolution().compute_mode())
            << "\nmath_type: "
            << proto::MathType_Name(proto.colwolution().math_type())
            << "\nalgo: " << algo;
        return oss.str();
      };
      Status status = RunColwolution(alpha, beta, handle, test, algo);
      // Some colwolution configs successfully return a workspace size (or
      // return from lwdnnGetColwolution*Algorithm_v7 with success status) but
      // then fail to run, see lwbugs/2082072. Instead of triggering test
      // failures we just skip configurations that hit this bug. A targeted test
      // (for 'reference' instead of 'test') is in lwdnn_tests.textproto.
      if (!status.ok() &&
          status.message().find("LWDNN_STATUS_NOT_SUPPORTED") != string::npos) {
        continue;
      }
      ASSERT_TRUE(IsOk(status)) << get_message();

      // Check that the test result matches the reference result. This also
      // compares buffer elements not referenced by the tensor.
      double tolerance = tolerance_scale * GetTolerance(algo);
      EXPECT_TRUE(IsOk(TensorDataEqual(ref_result_data, *result_data,
                                       *result_desc, tolerance)))
          << get_message();
    }
  }
}

#if LWDNN_MAJOR >= 7
template <typename T>
std::vector<ColwolutionAlgo> ToColwolutionAlgos(std::vector<T> algo_perfs,
                                                int num_algorithms) {
  // lwdnnGetColwolution*Algorithm_v7 also returns elements with non-success
  // status. Filter them out here.
  auto end =
      std::remove_if(algo_perfs.begin(), algo_perfs.begin() + num_algorithms,
                     [](const T& algo_perf) {
                       return algo_perf.status != LWDNN_STATUS_SUCCESS;
                     });
  std::vector<ColwolutionAlgo> result;
  std::transform(algo_perfs.begin(), end, std::back_inserter(result),
                 [](const T& algo_perf) { return algo_perf.algo; });
  return result;
}

// Compare the list of algorithms returned from GetSupportedColwolutionAlgos
// with the result from lwdnnGetColwolution*Algorithm_v7.
TEST_P(ColwolutionTest, GetAlgorithm_v7) {
  LwdnnHandle handle = CreateLwdnnHandle();

  auto ref_proto = GetParam().reference();

  for (auto proto : GetParam().test()) {
    proto = GetMergedColwolution(ref_proto, proto);
    if (proto.algo_oneof_case() != proto::ColwolutionConfig::kAllAlgos) {
      continue;
    }

    auto direction = proto.all_algos();
    auto input = CreateTensorDescriptor(proto.input());
    auto filter = CreateFilterDescriptor(proto.filter());
    auto colwolution = CreateColwolutionDescriptor(proto.colwolution());

    ASSERT_OK_AND_ASSIGN(
        auto output, CreateOutputDescriptor(proto, input, filter, colwolution));

    std::vector<ColwolutionAlgo> v7_algos;
    int num_algorithms = 0;
    switch (direction) {
      case proto::COLWOLUTION_FWD: {
        ASSERT_TRUE(IsOk(GetStatus(lwdnnGetColwolutionForwardAlgorithmMaxCount(
            handle.get(), &num_algorithms))));
        std::vector<lwdnnColwolutionFwdAlgoPerf_t> algo_perfs(num_algorithms);
        ASSERT_TRUE(IsOk(GetStatus(lwdnnGetColwolutionForwardAlgorithm_v7(
            handle.get(), input.get(), filter.get(), colwolution.get(),
            output.get(), algo_perfs.size(), &num_algorithms,
            algo_perfs.data()))));
        v7_algos = ToColwolutionAlgos(algo_perfs, num_algorithms);
      } break;
      case proto::COLWOLUTION_BWD_DATA: {
        ASSERT_TRUE(
            IsOk(GetStatus(lwdnnGetColwolutionBackwardDataAlgorithmMaxCount(
                handle.get(), &num_algorithms))));
        std::vector<lwdnnColwolutionBwdDataAlgoPerf_t> algo_perfs(
            num_algorithms);
        ASSERT_TRUE(IsOk(GetStatus(lwdnnGetColwolutionBackwardDataAlgorithm_v7(
            handle.get(), filter.get(), output.get(), colwolution.get(),
            input.get(), algo_perfs.size(), &num_algorithms,
            algo_perfs.data()))));
        v7_algos = ToColwolutionAlgos(algo_perfs, num_algorithms);
      } break;
      case proto::COLWOLUTION_BWD_FILTER: {
        ASSERT_TRUE(
            IsOk(GetStatus(lwdnnGetColwolutionBackwardFilterAlgorithmMaxCount(
                handle.get(), &num_algorithms))));
        std::vector<lwdnnColwolutionBwdFilterAlgoPerf_t> algo_perfs(
            num_algorithms);
        ASSERT_TRUE(
            IsOk(GetStatus(lwdnnGetColwolutionBackwardFilterAlgorithm_v7(
                handle.get(), input.get(), output.get(), colwolution.get(),
                filter.get(), algo_perfs.size(), &num_algorithms,
                algo_perfs.data()))));
        v7_algos = ToColwolutionAlgos(algo_perfs, num_algorithms);
      } break;
      default:
        LOG(FATAL) << "Invalid direction: "
                   << proto::ColwolutionDirection_Name(direction);
    }
    std::sort(v7_algos.begin(), v7_algos.end());

    // If the colwolution specifies TENSOR_OP_MATH, GetSupportedColwolutionAlgos
    // returns algorithms that silently fall back to DEFAULT_MATH as well.
    // lwdnnGetColwolution*Algorithm_v7 on the other hand returns separate
    // algo_perfs for DEFAULT_MATH and TENSOR_OP_MATH (but the latter may still
    // fall back to DEFAULT_MATH). We can savely remove duplicates from v7_algos
    // because TensorFlow tries to run every algorithm for both math types.
    //
    // See lwbugs/2072859 for details.
    if (proto.colwolution().math_type() == proto::TENSOR_OP_MATH) {
      v7_algos.erase(std::unique(v7_algos.begin(), v7_algos.end()),
                     v7_algos.end());
    }

    auto supported_algos = GetSupportedColwolutionAlgos(
        handle, direction, input, filter, colwolution, output,
        std::numeric_limits<size_t>::max());
    std::sort(supported_algos.begin(), supported_algos.end());

    EXPECT_EQ(supported_algos, v7_algos);
  }
}
#endif  // LWDNN_MAJOR >= 7

// Creates a ColwolutionConfig from the given parameters for
// proto.ColwolutionTest.reference.
//
// Note: in_channels and out_channels are multiplied by group_count.
proto::ColwolutionConfig MakeColwolutionConfig(int batch_count, int in_channels,
                                               std::vector<int> in_sizes,
                                               int out_channels,
                                               std::vector<int> filter_sizes,
                                               int group_count,
                                               Padding padding) {
  proto::TensorDescriptor input;
  for (int dim : {batch_count, in_channels * group_count}) {
    input.add_dimension(dim);
  }
  for (int dim : in_sizes) {
    input.add_dimension(dim);
  }
  input.set_format(proto::TENSOR_NCHW);
  input.set_data_type(proto::DATA_DOUBLE);

  proto::FilterDescriptor filter;
  for (int dim : {out_channels * group_count, in_channels}) {
    filter.add_dimension(dim);
  }
  for (int dim : filter_sizes) {
    filter.add_dimension(dim);
  }
  filter.set_format(proto::TENSOR_NCHW);
  filter.set_data_type(proto::DATA_DOUBLE);

  proto::ColwolutionDescriptor colwolution;
  for (auto filter_size : filter_sizes) {
    // SAME padding is only this simple if stride and dilation are 1.
    // TODO: Add function to compute padding for generic stride/dilation.
    colwolution.add_pad(padding == Padding::SAME ? filter_size / 2 : 0);
  }
  colwolution.set_compute_mode(proto::DATA_DOUBLE);
  if (group_count > 1) {
    colwolution.set_group_count(group_count);
  }

  proto::ColwolutionConfig result;
  *result.mutable_input() = std::move(input);
  *result.mutable_filter() = std::move(filter);
  *result.mutable_colwolution() = std::move(colwolution);

  return result;
}

struct SizeRange {
  int minimum;
  int maximum;
};

// Returns range ['largest 2^k+1 not greater than size', size].
SizeRange MakePowerOfTwoRange(int size) {
  CHECK_GT(size, 0);
  int power_of_two = 1;
  while (power_of_two < size) {
    power_of_two *= 2;
  }
  return {power_of_two / 2 + 1, size};
}

// Creates a randomized ColwolutionConfig from the given parameter ranges for
// proto.ColwolutionTest.reference. For VALID padding, the ranges of input size
// and filter size are clamped so that the filter is no larger then the inputs.
template <typename RandGen>
proto::ColwolutionConfig MakeColwolutionConfig(
    RandGen&& rand_gen, SizeRange batch_count_range,
    SizeRange in_channels_range, std::vector<SizeRange> in_size_ranges,
    SizeRange out_channels_range, std::vector<SizeRange> filter_size_ranges,
    SizeRange group_count_range, Padding padding) {
  if (padding == Padding::VALID) {
    for (auto& in_size_range : in_size_ranges) {
      in_size_range.minimum =
          std::max(in_size_range.minimum, in_size_range.minimum);
    }
  }

  auto randomize = [&](const SizeRange& range) {
    CHECK_LE(range.minimum, range.maximum);
    return std::uniform_int_distribution<int>{range.minimum,
                                              range.maximum}(rand_gen);
  };
  int batch_count = randomize(batch_count_range);
  int in_channels = randomize(in_channels_range);
  std::vector<int> in_sizes;
  std::transform(in_size_ranges.begin(), in_size_ranges.end(),
                 std::back_inserter(in_sizes), randomize);
  int out_channels = randomize(out_channels_range);
  int group_count = randomize(group_count_range);

  if (padding == Padding::VALID) {
    for (size_t i = 0; i < filter_size_ranges.size(); ++i) {
      filter_size_ranges[i].maximum =
          std::min(filter_size_ranges[i].maximum, in_sizes[i]);
    }
  }
  std::vector<int> filter_sizes;
  std::transform(filter_size_ranges.begin(), filter_size_ranges.end(),
                 std::back_inserter(filter_sizes), randomize);

  auto result =
      MakeColwolutionConfig(batch_count, in_channels, in_sizes, out_channels,
                            filter_sizes, group_count, padding);

  std::ostringstream label;
  label << batch_count << 'x' << in_channels * group_count << 'x'
        << absl::StrJoin(in_sizes, "x") << '_' << out_channels * group_count
        << 'x' << in_channels << 'x' << absl::StrJoin(filter_sizes, "x") << '_'
        << (padding == Padding::SAME ? "SAME" : "VALID");
  result.set_label(label.str());

  return result;
}

// Set of format and data type configurations to run. For details, see
// http://docs.lwpu.com/deeplearning/sdk/lwdnn-developer-guide/index.html#lwdnnColwolutionForward
enum class Config {
  NCHW_FLOAT,
  NCHW_DOUBLE,
  NCHW_PSEUDO_HALF,
  NCHW_TRUE_HALF,
  NCHW_TENSOR_OP,
  NHWC_FLOAT,
  NHWC_PSEUDO_HALF,
  NHWC_TRUE_HALF
};

// Returns whether lwDNN supports the given config on the current device.
bool IsConfigSupported(Config config) {
  switch (config) {
    case Config::NCHW_PSEUDO_HALF:
      return DeviceSupportsReducedPrecision();
    case Config::NCHW_TRUE_HALF:
      return DeviceSupportsReducedPrecision();
    case Config::NCHW_TENSOR_OP:
      return LWDNN_MAJOR >= 7 && DeviceSupportsTensorOpMath();
    case Config::NHWC_FLOAT:
      // Internal error on lwdNN 6, see lwdnn_tests.textproto.
      return LWDNN_MAJOR >= 7;
    case Config::NHWC_PSEUDO_HALF:
      // Crashes on lwDNN 6, see lwdnn_tests.textproto.
      return LWDNN_MAJOR >= 7 && DeviceSupportsReducedPrecision();
    case Config::NHWC_TRUE_HALF:
      return DeviceSupportsReducedPrecision();
    default:
      return true;
  }
}

// Creates a ColwolutionConfig from the given config for
// proto.ColwolutionTest.test.
proto::ColwolutionConfig MakeColwolutionConfig(Config config) {
  proto::TensorFormat format;
  proto::DataType data_type;
  proto::DataType compute_mode;
  proto::MathType math_type;
  string label;
  switch (config) {
    case Config::NCHW_FLOAT:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_FLOAT;
      compute_mode = proto::DATA_FLOAT;
      math_type = proto::DEFAULT_MATH;
      label = "NCHW_FLOAT";
      break;
    case Config::NCHW_DOUBLE:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_DOUBLE;
      compute_mode = proto::DATA_DOUBLE;
      math_type = proto::DEFAULT_MATH;
      label = "NCHW_DOUBLE";
      break;
    case Config::NCHW_PSEUDO_HALF:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_FLOAT;
      math_type = proto::DEFAULT_MATH;
      label = "NCHW_PSEUDO_HALF";
      break;
    case Config::NCHW_TRUE_HALF:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_HALF;
      math_type = proto::DEFAULT_MATH;
      label = "NCHW_TRUE_HALF";
      break;
    case Config::NCHW_TENSOR_OP:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_HALF;
      math_type = proto::TENSOR_OP_MATH;
      label = "NCHW_TENSOR_OP";
      break;
    case Config::NHWC_FLOAT:
      format = proto::TENSOR_NHWC;
      data_type = proto::DATA_FLOAT;
      compute_mode = proto::DATA_FLOAT;
      math_type = proto::DEFAULT_MATH;
      label = "NHWC_FLOAT";
      break;
    case Config::NHWC_PSEUDO_HALF:
      format = proto::TENSOR_NHWC;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_FLOAT;
      math_type = proto::DEFAULT_MATH;
      label = "NHWC_PSEUDO_HALF";
      break;
    case Config::NHWC_TRUE_HALF:
      format = proto::TENSOR_NHWC;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_HALF;
      math_type = proto::DEFAULT_MATH;
      label = "NHWC_TRUE_HALF";
      break;
    default:
      LOG(FATAL) << "Unknown config";
  }

  proto::ColwolutionConfig result;
  result.mutable_input()->set_format(format);
  result.mutable_input()->set_data_type(data_type);
  result.mutable_filter()->set_format(format);
  result.mutable_filter()->set_data_type(data_type);
  result.mutable_colwolution()->set_compute_mode(compute_mode);
  result.mutable_colwolution()->set_math_type(math_type);
  result.set_label(label);
  return result;
}

// Returns true if lwDNN should handle the given parameters for at least one
// algorithm. Changes to this function will not just drop some tests, but
// completely alter the set of generated tests!
//
// Note: in_channels and out_channels are multiplied with group_count.
bool ValidateParams(absl::optional<SizeRange> opt_batch_count,
                    absl::optional<SizeRange> opt_in_channels,
                    std::vector<absl::optional<SizeRange>> opt_in_sizes,
                    absl::optional<SizeRange> opt_out_channels,
                    std::vector<absl::optional<SizeRange>> opt_filter_sizes,
                    absl::optional<SizeRange> opt_group_count,
                    absl::optional<Padding> opt_padding,
                    absl::optional<Config> opt_config) {
  // Default to medium sizes so we don't paint ourselves in a corner.
  SizeRange tensor_range = {32, 32};
  SizeRange filter_range = {5, 5};
  SizeRange batch_count = opt_batch_count.value_or(tensor_range);
  SizeRange out_channels = opt_out_channels.value_or(tensor_range);
  SizeRange filter_in_channels = opt_in_channels.value_or(tensor_range);
  SizeRange group_count = opt_group_count.value_or(tensor_range);
  Padding padding = opt_padding.value_or(Padding::SAME);
  Config config = opt_config.value_or(Config::NCHW_FLOAT);

  int in_max_channels = filter_in_channels.maximum * group_count.maximum;
  int out_max_channels = out_channels.maximum * group_count.maximum;

  std::vector<int> in_max_sizes;
  for (const auto& opt_in_size : opt_in_sizes) {
    in_max_sizes.push_back(opt_in_size.value_or(tensor_range).maximum);
  }

  std::vector<int> filter_min_sizes;
  std::vector<int> filter_max_sizes;
  for (const auto& opt_filter_size : opt_filter_sizes) {
    auto filter_size = opt_filter_size.value_or(filter_range);
    filter_min_sizes.push_back(filter_size.minimum);
    filter_max_sizes.push_back(filter_size.maximum);
  }

  size_t million = 1ull << 20;  // 1 million as in MiB.
  size_t max_tensor_elements = static_cast<size_t>(batch_count.maximum) *
                               std::max(in_max_channels, out_max_channels);
  for (const auto& in_max_size : in_max_sizes) {
    max_tensor_elements *= in_max_size;
  }

  if (max_tensor_elements >= 2048 * million) {
    // lwDNN has a limit of less (!) than 2G elements per tensor.
    return false;
  }

  if (padding == Padding::VALID) {
    for (size_t i = 0; i < opt_in_sizes.size(); ++i) {
      if (filter_min_sizes[i] > in_max_sizes[i]) {
        // VALID padding requires the filter to be no greater than the input.
        return false;
      }
    }
  }

  auto proto = MakeColwolutionConfig(
      batch_count.maximum, filter_in_channels.maximum, in_max_sizes,
      out_channels.maximum, filter_max_sizes, group_count.maximum, padding);

  auto input = CreateTensorDescriptor(proto.input());
  auto filter = CreateFilterDescriptor(proto.filter());
  auto colwolution = CreateColwolutionDescriptor(proto.colwolution());
  auto output_or =
      CreateOutputDescriptor(proto::TENSOR_NCHW, input, filter, colwolution);
  if (!IsOk(output_or)) {
    return false;
  }

  // Compute the size of device allocations in CompareResults and return whether
  // it fits the specified memory limit.

  size_t data_type_size;
  switch (MakeColwolutionConfig(config).filter().data_type()) {
    case proto::DATA_FLOAT:
      data_type_size = sizeof(float);
      break;
    case proto::DATA_DOUBLE:
      data_type_size = sizeof(double);
      break;
    case proto::DATA_HALF:
      data_type_size = sizeof(__half);
      break;
    default:
      LOG(FATAL) << "Not yet supported";
  }

  size_t input_num_elements = GetTensorNumElements(input);
  size_t filter_num_elements = GetFilterNumElements(filter);
  size_t output_num_elements = GetTensorNumElements(output_or.ValueOrDie());

  size_t sum_num_elements =
      input_num_elements + filter_num_elements + output_num_elements;
  size_t max_num_elements =
      std::max({input_num_elements, filter_num_elements, output_num_elements});

  // Number of buffers for CompareResults' init_data, ref_result_data and
  // (if colwersion from double is needed) ColwertAndTransformTensor.
  size_t num_extra_buffers = data_type_size == sizeof(double) ? 2 : 3;

  size_t required_bytes = (sizeof(double) + data_type_size) * sum_num_elements +
                          num_extra_buffers * data_type_size * max_num_elements;
  if (required_bytes > GetAvailableDeviceMemoryBytes()) {
    return false;
  }

  return true;
}
bool ValidateParams2d(absl::optional<SizeRange> opt_batch_count,
                      absl::optional<SizeRange> opt_in_channels,
                      absl::optional<SizeRange> opt_in_height,
                      absl::optional<SizeRange> opt_in_width,
                      absl::optional<SizeRange> opt_out_channels,
                      absl::optional<SizeRange> opt_filter_height,
                      absl::optional<SizeRange> opt_filter_width,
                      absl::optional<Padding> opt_padding,
                      absl::optional<Config> opt_config) {
  SizeRange group_count = {1, 1};
  return ValidateParams(opt_batch_count, opt_in_channels,
                        {opt_in_height, opt_in_width}, opt_out_channels,
                        {opt_filter_height, opt_filter_width}, group_count,
                        opt_padding, opt_config);
}
bool ValidateParams3d(absl::optional<SizeRange> opt_batch_count,
                      absl::optional<SizeRange> opt_in_channels,
                      absl::optional<SizeRange> opt_in_depth,
                      absl::optional<SizeRange> opt_in_height,
                      absl::optional<SizeRange> opt_in_width,
                      absl::optional<SizeRange> opt_out_channels,
                      absl::optional<SizeRange> opt_filter_depth,
                      absl::optional<SizeRange> opt_filter_height,
                      absl::optional<SizeRange> opt_filter_width,
                      absl::optional<Padding> opt_padding,
                      absl::optional<Config> opt_config) {
  SizeRange group_count = {1, 1};
  return ValidateParams(opt_batch_count, opt_in_channels,
                        {opt_in_depth, opt_in_height, opt_in_width},
                        opt_out_channels,
                        {opt_filter_depth, opt_filter_height, opt_filter_width},
                        group_count, opt_padding, opt_config);
}
bool ValidateParamsGrouped(absl::optional<SizeRange> opt_batch_count,
                           absl::optional<SizeRange> in_channels,
                           absl::optional<SizeRange> opt_in_height,
                           absl::optional<SizeRange> opt_in_width,
                           absl::optional<SizeRange> out_channels,
                           absl::optional<SizeRange> opt_filter_height,
                           absl::optional<SizeRange> opt_filter_width,
                           absl::optional<SizeRange> opt_group_count,
                           absl::optional<Padding> opt_padding,
                           absl::optional<Config> opt_config) {
  return ValidateParams(opt_batch_count, in_channels,
                        {opt_in_height, opt_in_width}, out_channels,
                        {opt_filter_height, opt_filter_width}, opt_group_count,
                        opt_padding, opt_config);
}

// Factory for proto::ColwolutionTest with one factory method per direction.
class TestFactory {
 public:
  template <typename RandGen>
  TestFactory(RandGen&& rand_gen, SizeRange batch_count, SizeRange in_channels,
              std::vector<SizeRange> in_sizes, SizeRange out_channels,
              std::vector<SizeRange> filter_sizes, SizeRange group_count,
              Padding padding, Config config) {
    reference_ =
        MakeColwolutionConfig(rand_gen, batch_count, in_channels, in_sizes,
                              out_channels, filter_sizes, group_count, padding);

    test_ = MakeColwolutionConfig(config);

    // Disable tests that are not supported on the host machine.
    //
    // Excluding unsupported configs from the parameter vector would make the
    // supported tests depend on the machine configuration, and that seems
    // worse than leaving holes in the all-pairs test space on older hardware
    // and lwDNN versions.
    disabled_label_ = IsConfigSupported(config) ? "" : "DISABLED_";

    config_label_ = "_" + test_.label() + "_" + reference_.label();

    test_.clear_label();
    reference_.clear_label();
  }

  proto::ColwolutionTest operator()(proto::ColwolutionFwdAlgo algo) const {
    // in_channels * filter_height * filter_width.
    const auto& dims = reference_.filter().dimension();
    int num_summands = std::accumulate(dims.begin() + 2, dims.end(), dims[1],
                                       std::multiplies<int>());
    auto result = MakeTest(proto::COLWOLUTION_FWD, num_summands);
    result.mutable_reference()->set_fwd_algo(algo);
    return result;
  }
  proto::ColwolutionTest operator()(proto::ColwolutionBwdDataAlgo algo) const {
    // out_channels * filter_height * filter_width.
    const auto& dims = reference_.filter().dimension();
    int num_summands = std::accumulate(dims.begin() + 2, dims.end(), dims[0],
                                       std::multiplies<int>());
    auto result = MakeTest(proto::COLWOLUTION_BWD_DATA, num_summands);
    result.mutable_reference()->set_bwd_data_algo(algo);
    return result;
  }
  proto::ColwolutionTest operator()(
      proto::ColwolutionBwdFilterAlgo algo) const {
    // batch_count * in_height * in_width.
    const auto& dims = reference_.input().dimension();
    int num_summands = std::accumulate(dims.begin() + 2, dims.end(), dims[0],
                                       std::multiplies<int>());
    auto result = MakeTest(proto::COLWOLUTION_BWD_FILTER, num_summands);
    result.mutable_reference()->set_bwd_filter_algo(algo);
    return result;
  }

 private:
  proto::ColwolutionTest MakeTest(proto::ColwolutionDirection direction,
                                  size_t num_summands) const {
    auto reference = reference_;

    string direction_label = proto::ColwolutionDirection_Name(direction);
    reference.set_label(disabled_label_ + direction_label + config_label_);

    // Scale result so that elements have an expected value of one. This
    // prevents overflow for half data.
    reference.set_one_minus_alpha(1.0 - 4.0 / num_summands);

    auto test = test_;
    test.set_all_algos(direction);

    proto::ColwolutionTest result;
    *result.mutable_reference() = std::move(reference);
    *result.add_test() = std::move(test);

    return result;
  }

  proto::ColwolutionConfig reference_;
  proto::ColwolutionConfig test_;
  string disabled_label_;
  string config_label_;
};

// Returns standard size ranges for tensor dimensions.
std::vector<SizeRange> GetTensorSizeRanges() {
  // Create conselwtive closed ranges [2^(k-1)+1, 2^k].
  std::vector<SizeRange> result;
  for (int size : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}) {
    result.push_back(MakePowerOfTwoRange(size));
  }
  return result;
}

// Returns standard size ranges for filter dimensions.
std::vector<SizeRange> GetFilterSizeRanges() {
  // Create conselwtive closed ranges [2^(k-1)+1, 2^k].
  std::vector<SizeRange> result;
  for (int size : {1, 2, 4, 8, 16}) {
    result.push_back(MakePowerOfTwoRange(size));
  }
  return result;
}

// Returns a list of 2D colwolution tests, one for every direction per generated
// parameter combination.
google::protobuf::RepeatedPtrField<proto::ColwolutionTest> MakeTests2d(
    const std::vector<SizeRange>& filter_sizes) {
  std::vector<SizeRange> batch_counts = GetTensorSizeRanges();
  std::vector<SizeRange> channel_counts = GetTensorSizeRanges();
  std::vector<SizeRange> image_sizes = GetTensorSizeRanges();
  std::vector<Padding> paddings = {Padding::SAME, Padding::VALID};
  std::vector<Config> configs = {
      Config::NCHW_FLOAT,       Config::NCHW_DOUBLE,
      Config::NCHW_PSEUDO_HALF, Config::NCHW_TRUE_HALF,
      Config::NCHW_TENSOR_OP,   Config::NHWC_FLOAT,
      Config::NHWC_PSEUDO_HALF, Config::NHWC_TRUE_HALF};

  LwdnnHandle handle = CreateLwdnnHandle();
  auto validator = MakeCallWithTuple(ValidateParams2d);

  std::minstd_rand rand_gen(GetRandomSeed());
  auto params_vec =
      MakeAllPairs(rand_gen, validator, batch_counts, channel_counts,
                   image_sizes, image_sizes, channel_counts, filter_sizes,
                   filter_sizes, paddings, configs);

  google::protobuf::RepeatedPtrField<proto::ColwolutionTest> colw_tests;
  for (const auto& params : params_vec) {
    MakeCallWithTuple(
        [&](SizeRange batch_count, SizeRange in_channels, SizeRange in_height,
            SizeRange in_width, SizeRange out_channels, SizeRange filter_height,
            SizeRange filter_width, Padding padding, Config config) {
          SizeRange group_count = {1, 1};
          TestFactory factory(rand_gen, batch_count, in_channels,
                              {in_height, in_width}, out_channels,
                              {filter_height, filter_width}, group_count,
                              padding, config);
          // Add colwolution_test for each direction.
          *colw_tests.Add() =
              factory(proto::COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
          *colw_tests.Add() = factory(proto::COLWOLUTION_BWD_DATA_ALGO_1);
          *colw_tests.Add() = factory(proto::COLWOLUTION_BWD_FILTER_ALGO_1);
        })(params);
  }

  return colw_tests;
}

// Returns a list of 3D colwolution tests.
google::protobuf::RepeatedPtrField<proto::ColwolutionTest> MakeTests3d() {
  std::vector<SizeRange> batch_counts = GetTensorSizeRanges();
  std::vector<SizeRange> channel_counts = GetTensorSizeRanges();
  std::vector<SizeRange> image_sizes = GetTensorSizeRanges();
  std::vector<SizeRange> filter_sizes = GetFilterSizeRanges();
  std::vector<Padding> paddings = {Padding::SAME, Padding::VALID};
  std::vector<Config> configs = {
      Config::NCHW_FLOAT, Config::NCHW_DOUBLE, Config::NCHW_PSEUDO_HALF,
      Config::NCHW_TRUE_HALF, Config::NCHW_TENSOR_OP};

  LwdnnHandle handle = CreateLwdnnHandle();
  auto validator = MakeCallWithTuple(ValidateParams3d);

  std::minstd_rand rand_gen(GetRandomSeed());
  auto params_vec =
      MakeAllPairs(rand_gen, validator, batch_counts, channel_counts,
                   image_sizes, image_sizes, image_sizes, channel_counts,
                   filter_sizes, filter_sizes, filter_sizes, paddings, configs);

  google::protobuf::RepeatedPtrField<proto::ColwolutionTest> colw_tests;
  for (const auto& params : params_vec) {
    MakeCallWithTuple(
        [&](SizeRange batch_count, SizeRange in_channels, SizeRange in_depth,
            SizeRange in_height, SizeRange in_width, SizeRange out_channels,
            SizeRange filter_depth, SizeRange filter_height,
            SizeRange filter_width, Padding padding, Config config) {
          SizeRange group_count = {1, 1};
          TestFactory factory(rand_gen, batch_count, in_channels,
                              {in_depth, in_height, in_width}, out_channels,
                              {filter_depth, filter_height, filter_width},
                              group_count, padding, config);
          // Add colwolution_test for each direction.
          *colw_tests.Add() =
              factory(proto::COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
          *colw_tests.Add() = factory(proto::COLWOLUTION_BWD_DATA_ALGO_0);
          *colw_tests.Add() = factory(proto::COLWOLUTION_BWD_FILTER_ALGO_0);
        })(params);
  }

  return colw_tests;
}

// Returns a list of grouped colwolution tests.
google::protobuf::RepeatedPtrField<proto::ColwolutionTest> MakeTestsGrouped() {
  std::vector<SizeRange> batch_counts = GetTensorSizeRanges();
  std::vector<SizeRange> channel_counts = GetTensorSizeRanges();
  std::vector<SizeRange> image_sizes = GetTensorSizeRanges();
  std::vector<SizeRange> filter_sizes = GetFilterSizeRanges();
  std::vector<SizeRange> group_counts = GetTensorSizeRanges();
  std::vector<Padding> paddings = {Padding::SAME, Padding::VALID};
  std::vector<Config> configs = {
      Config::NCHW_FLOAT, Config::NCHW_DOUBLE, Config::NCHW_PSEUDO_HALF,
      Config::NCHW_TRUE_HALF, Config::NCHW_TENSOR_OP};

  LwdnnHandle handle = CreateLwdnnHandle();
  auto validator = MakeCallWithTuple(ValidateParamsGrouped);

  std::minstd_rand rand_gen(GetRandomSeed());
  auto params_vec =
      MakeAllPairs(rand_gen, validator, batch_counts, channel_counts,
                   image_sizes, image_sizes, channel_counts, filter_sizes,
                   filter_sizes, group_counts, paddings, configs);

  google::protobuf::RepeatedPtrField<proto::ColwolutionTest> colw_tests;
  for (const auto& params : params_vec) {
    MakeCallWithTuple([&](SizeRange batch_count, SizeRange in_channels,
                          SizeRange in_height, SizeRange in_width,
                          SizeRange out_channels, SizeRange filter_height,
                          SizeRange filter_width, SizeRange group_count,
                          Padding padding, Config config) {
      TestFactory factory(rand_gen, batch_count, in_channels,
                          {in_height, in_width}, out_channels,
                          {filter_height, filter_width}, group_count, padding,
                          config);
      // Add colwolution_test for each direction.
      *colw_tests.Add() =
          factory(proto::COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
      *colw_tests.Add() = factory(proto::COLWOLUTION_BWD_DATA_ALGO_0);
      *colw_tests.Add() = factory(proto::COLWOLUTION_BWD_FILTER_ALGO_0);
    })(params);
  }

  return colw_tests;
}

string GetTestName(const testing::TestParamInfo<proto::ColwolutionTest>& info) {
  return info.param.reference().label();
}

INSTANTIATE_TEST_SUITE_P(
    FromFile, ColwolutionTest,
    ::testing::ValuesIn(GetLwdnnTestsFromFile().colwolution_test()),
    GetTestName);

INSTANTIATE_TEST_SUITE_P(Colw2dFilter3x3, ColwolutionTest,
                         ::testing::ValuesIn([] {
                           std::vector<SizeRange> filter_sizes = {{3, 3}};
                           return MakeTests2d(filter_sizes);
                         }()),
                         GetTestName);

INSTANTIATE_TEST_SUITE_P(Colw2dFilter5x5, ColwolutionTest,
                         ::testing::ValuesIn([] {
                           std::vector<SizeRange> filter_sizes = {{5, 5}};
                           return MakeTests2d(filter_sizes);
                         }()),
                         GetTestName);

INSTANTIATE_TEST_SUITE_P(Colw2dFilterOther, ColwolutionTest,
                         ::testing::ValuesIn([] {
                           auto filter_sizes = GetFilterSizeRanges();
                           return MakeTests2d(filter_sizes);
                         }()),
                         GetTestName);

INSTANTIATE_TEST_SUITE_P(Colw3d, ColwolutionTest,
                         ::testing::ValuesIn([] { return MakeTests3d(); }()),
                         GetTestName);

#if LWDNN_MAJOR >= 7
INSTANTIATE_TEST_SUITE_P(Colw2dGrouped, ColwolutionTest,
                         ::testing::ValuesIn([] {
                           return MakeTestsGrouped();
                         }()),
                         GetTestName);
#endif  // LWDNN_MAJOR >= 7

}  // namespace
}  // namespace lwidia_libs_test
