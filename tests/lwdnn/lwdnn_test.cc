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

#include <string>

#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "lwda_util.h"
#include "lwdnn.pb.h"
#include "lwdnn_test.h"
#include "load_textproto.h"
#include "test_util.h"

DEFINE_string(proto_path, "lwdnn_tests.textproto",
              "Path to text proto file containing tests to run.");

namespace lwidia_libs_test {
namespace {
template <typename T>
Status TensorDataEqual(const DeviceMemory& first, const DeviceMemory& second,
                       const TensorDescriptor& descriptor, double tolerance) {
  return DeviceDataEqual(static_cast<const T*>(first.get()),
                         static_cast<const T*>(second.get()),
                         GetTensorNumElements(descriptor), tolerance);
}
}  // namespace

Status TensorDataEqual(const DeviceMemory& first, const DeviceMemory& second,
                       const TensorDescriptor& descriptor, double tolerance) {
  switch (GetTensorDataType(descriptor)) {
    case LWDNN_DATA_FLOAT:
      return TensorDataEqual<float>(first, second, descriptor, tolerance);
    case LWDNN_DATA_DOUBLE:
      return TensorDataEqual<double>(first, second, descriptor, tolerance);
    case LWDNN_DATA_HALF:
      return TensorDataEqual<__half>(first, second, descriptor, tolerance);
    default:
      LOG(FATAL) << "Not yet supported";
  }
}

proto::Tests GetLwdnnTestsFromFile() {
  proto::Tests tests;
  CHECK_OK_STATUS(LoadTextProto(FLAGS_proto_path, &tests));
  return tests;
}

std::ostream& operator<<(std::ostream& ostr, Padding padding) {
  switch (padding) {
    case Padding::SAME:
      return ostr << "SAME";
    case Padding::VALID:
      return ostr << "VALID";
  }
  return ostr;
}

// Tests that lwdnnGetWorkspaceSize either returns unsupported status or a
// reasonable value.
//
// lwDNN before version 7 returns huge workspace sizes for some configurations
// that look like the internal computation overflowed.
//
// See lwbugs/1893243.
TEST(ColwolutionTest, GetWorkspaceSize_Overflow) {
  LwdnnHandle handle;
  proto::TensorDescriptor input_desc;
  input_desc.set_data_type(proto::DATA_FLOAT);
  input_desc.set_format(proto::TENSOR_NCHW);
  for (int dim : {1, 128, 300, 300}) {
    input_desc.add_dimension(dim);
  }
  auto input = CreateTensorDescriptor(input_desc);

  proto::FilterDescriptor filter_desc;
  filter_desc.set_data_type(proto::DATA_FLOAT);
  filter_desc.set_format(proto::TENSOR_NCHW);
  for (int dim : {768, 128, 3, 3}) {
    filter_desc.add_dimension(dim);
  }
  auto filter = CreateFilterDescriptor(filter_desc);

  proto::ColwolutionDescriptor colw_desc;
  colw_desc.set_compute_mode(proto::DATA_FLOAT);
  for (int pad : {1, 1}) {
    colw_desc.add_pad(pad);
  }
  auto colwolution = CreateColwolutionDescriptor(colw_desc);

  ASSERT_OK_AND_ASSIGN(
      auto output,
      CreateOutputDescriptor(proto::TENSOR_NCHW, input, filter, colwolution));

  size_t workspace_size = 0;
  auto status = lwdnnGetColwolutionForwardWorkspaceSize(
      CreateLwdnnHandle().get(), input.get(), filter.get(), colwolution.get(),
      output.get(), LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
      &workspace_size);

  if (status == LWDNN_STATUS_SUCCESS) {
    EXPECT_LE(workspace_size, 1ull << 63);
  } else {
    EXPECT_EQ(status, LWDNN_STATUS_NOT_SUPPORTED);
  }
}

// Tests the supported range of the arrayLengthRequested parameter for
// lwdnnGetColwolutionNdDescriptor, which should be [0, LWDNN_DIM_MAX]
// according to the documentation, but lwDNN reports LWDNN_STATUS_NOT_SUPPORTED
// for anything larger than 6.
//
// See lwbugs/2064417.
//
// Update: The documentation has been corrected that the valid range is
// [0, LWDNN_DIM_MAX-2].
TEST(ColwolutionTest, GetColwolutionDesciptor_ArrayLengthRequested_Range) {
  proto::ColwolutionDescriptor proto;
  proto.set_compute_mode(proto::DATA_FLOAT);
  proto.add_pad(0);
  proto.add_pad(0);

  auto colw_desc = CreateColwolutionDescriptor(proto);

  const int array_length = LWDNN_DIM_MAX - 2;

  int rank;
  int pad[array_length];
  int stride[array_length];
  int dilation[array_length];
  lwdnnColwolutionMode_t colwolution_mode;
  lwdnnDataType_t compute_type;

  for (int array_length_requested = 0; array_length_requested <= array_length;
       ++array_length_requested) {
    EXPECT_TRUE(IsOk(GetStatus(lwdnnGetColwolutionNdDescriptor(
                         colw_desc.get(), array_length_requested, &rank, pad,
                         stride, dilation, &colwolution_mode, &compute_type))
                     << " array_length_requested = "
                     << array_length_requested));
  }
}

#if LWDNN_MAJOR >= 7
// Tests that lwdnnGetColwolution2dForwardOutputDim handles grouped
// colwolutions.
//
// See lwbugs/2178340, works as intended.
TEST(ColwolutionTest, GetGroupedColwolutionForwardOutputDim) {
  LwdnnHandle handle;
  proto::TensorDescriptor input_desc;
  input_desc.set_data_type(proto::DATA_FLOAT);
  input_desc.set_format(proto::TENSOR_NCHW);
  for (int dim : {3, 88, 4, 17}) {
    input_desc.add_dimension(dim);
  }
  auto input = CreateTensorDescriptor(input_desc);

  proto::FilterDescriptor filter_desc;
  filter_desc.set_data_type(proto::DATA_FLOAT);
  filter_desc.set_format(proto::TENSOR_NCHW);
  for (int dim : {14, 44, 3, 5}) {
    filter_desc.add_dimension(dim);
  }
  auto filter = CreateFilterDescriptor(filter_desc);

  proto::ColwolutionDescriptor colw_desc;
  colw_desc.set_compute_mode(proto::DATA_FLOAT);
  for (int pad : {1, 2}) {
    colw_desc.add_pad(pad);
  }
  colw_desc.set_group_count(2);
  auto colwolution = CreateColwolutionDescriptor(colw_desc);

  int n, c, h, w;
  ASSERT_TRUE(IsOk(GetStatus(lwdnnGetColwolution2dForwardOutputDim(
      colwolution.get(), input.get(), filter.get(), &n, &c, &h, &w))));

  EXPECT_EQ(n, input_desc.dimension(0));
  EXPECT_EQ(c, filter_desc.dimension(0));
  EXPECT_EQ(h, input_desc.dimension(2));
  EXPECT_EQ(w, input_desc.dimension(3));
}
#endif

}  // namespace lwidia_libs_test

int main(int argc, char** argv) {
  // Parse and validate flags before initializing gtest.
  gflags::AllowCommandLineReparsing();
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  // Check that all non-test flags (removed in line above) are valid gflags.
  for(int i = 1; i < argc; ++i) {
    std::string str = argv[i];
    str = str.substr(std::min(str.find("--"), str.size()), str.find('='));
    if (!gflags::GetCommandLineOption(str.c_str(), &str)) {
      LOG(FATAL) << "Unrecognized flag: " << argv[i];
    }
  }
  return RUN_ALL_TESTS();
}
