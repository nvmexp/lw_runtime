// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"

using std::cout;
using std::endl;

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#else
  #define LWDA_TEST_DEVICE -1
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define CMAKE_EXT ""
#endif

int main(int argc, char** argv);

namespace caffe {

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MultiDeviceTest() {
    Caffe::set_mode(TypeParam::device);
  }
  virtual ~MultiDeviceTest() {}
};

typedef ::testing::Types<float, double> TestDtypesNoFP16;

typedef ::testing::Types<float, double
#if defined(TEST_FP16)
    , float16
#endif
    > TestDtypes;

template <typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static constexpr Caffe::Brew device = Caffe::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> > {
};

template <typename TypeParam>
struct GPUDevice {
  typedef TypeParam Dtype;
  static constexpr Caffe::Brew device = Caffe::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {
};

typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                         GPUDevice<float>, GPUDevice<double>
#if defined(TEST_FP16)
                         , GPUDevice<float16>
#endif
                         >
                         TestDtypesAndDevices;

typedef ::testing::Types<GPUDevice<float>, GPUDevice<double>
#if defined(TEST_FP16)
                         , GPUDevice<float16>
#endif
                         >
                         TestDtypesGPUOnly;

typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                         GPUDevice<float>, GPUDevice<double>>
                         TestDtypesAndDevicesNoFP16;

typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>> TestDtypesAndCPUOnly;

}  // namespace caffe

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
