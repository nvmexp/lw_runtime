#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "lwb/util_allocator.lwh"

namespace caffe {

class CommonTest : public ::testing::Test {};

TEST_F(CommonTest, TestLwblasHandlerGPU) {
  int lwda_device_id;
  LWDA_CHECK(lwdaGetDevice(&lwda_device_id));
  EXPECT_TRUE(Caffe::lwblas_handle(0));
}

TEST_F(CommonTest, TestDeviceQuery) {
  std::string dq = Caffe::DeviceQuery();
  EXPECT_TRUE(dq.find("No") == 0UL || dq.find("Dev") == 0UL);
}

TEST_F(CommonTest, TestBrewMode) {
  Caffe::Brew lwrrent_mode = Caffe::mode();
  Caffe::set_mode(Caffe::CPU);
  EXPECT_EQ(Caffe::mode(), Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  EXPECT_EQ(Caffe::mode(), Caffe::GPU);
  Caffe::set_mode(lwrrent_mode);
}

TEST_F(CommonTest, TestRandSeedCPU) {
  SyncedMemory data_a(10 * sizeof(int));
  SyncedMemory data_b(10 * sizeof(int));
  Caffe::set_random_seed(1701);
  caffe_rng_bernoulli(10, 0.5, static_cast<int*>(data_a.mutable_cpu_data()));

  Caffe::set_random_seed(1701);
  caffe_rng_bernoulli(10, 0.5, static_cast<int*>(data_b.mutable_cpu_data()));

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(static_cast<const int*>(data_a.cpu_data())[i],
        static_cast<const int*>(data_b.cpu_data())[i]);
  }
}

TEST_F(CommonTest, TestRandSeedGPU) {
  SyncedMemory data_a(10 * sizeof(unsigned int));
  SyncedMemory data_b(10 * sizeof(unsigned int));
  Caffe::set_random_seed(1701);
  LWRAND_CHECK(lwrandGenerate(Caffe::lwrand_generator(),
        static_cast<unsigned int*>(data_a.mutable_gpu_data()), 10));
  LWDA_CHECK(lwdaStreamSynchronize(Caffe::lwrand_stream()));
  Caffe::set_random_seed(1701);
  LWRAND_CHECK(lwrandGenerate(Caffe::lwrand_generator(),
        static_cast<unsigned int*>(data_b.mutable_gpu_data()), 10));
  LWDA_CHECK(lwdaStreamSynchronize(Caffe::lwrand_stream()));
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(((const unsigned int*)(data_a.cpu_data()))[i],
        ((const unsigned int*)(data_b.cpu_data()))[i]);
  }
}

inline size_t pow2(unsigned int p) {
  return 1ULL << p;
}

TEST_F(CommonTest, TestLWBNearestPowerOf2) {
  size_t rounded_bytes;
  unsigned int power;
  for (int p = 0; p < sizeof(size_t) * CHAR_BIT; ++p) {
    size_t value = pow2(p);
    ++value;
    lwb::CachingDeviceAllocator::NearestPowerOf(power, rounded_bytes, 2, value);
    EXPECT_EQ(p + 1, power);
    EXPECT_EQ(pow2(power), rounded_bytes) << p;
    --value;
    lwb::CachingDeviceAllocator::NearestPowerOf(power, rounded_bytes, 2, value);
    EXPECT_EQ(p, power);
    EXPECT_EQ(pow2(power), rounded_bytes);
    --value;
    lwb::CachingDeviceAllocator::NearestPowerOf(power, rounded_bytes, 2, value);
    // Exclusion: for zero size we return 1 as rounded bytes (per original LWB
    // design)
    EXPECT_EQ(p == 1 ? 0 : p, power);  // because 2^1 - 1 == 2^0
    EXPECT_EQ(pow2(power), rounded_bytes);
  }
}

}  // namespace caffe
