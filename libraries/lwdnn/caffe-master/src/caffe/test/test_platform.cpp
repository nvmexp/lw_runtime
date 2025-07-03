#include <cstdio>
#include <cstdlib>

#include <glog/logging.h>
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern lwdaDeviceProp CAFFE_TEST_LWDA_PROP;

class PlatformTest : public ::testing::Test {};

TEST_F(PlatformTest, TestInitialization) {
  printf("Major revision number:         %d\n",  CAFFE_TEST_LWDA_PROP.major);
  printf("Minor revision number:         %d\n",  CAFFE_TEST_LWDA_PROP.minor);
  printf("Name:                          %s\n",  CAFFE_TEST_LWDA_PROP.name);
  printf("Total global memory:           %lu\n",
         CAFFE_TEST_LWDA_PROP.totalGlobalMem);
  printf("Total shared memory per block: %lu\n",
         CAFFE_TEST_LWDA_PROP.sharedMemPerBlock);
  printf("Total registers per block:     %d\n",
         CAFFE_TEST_LWDA_PROP.regsPerBlock);
  printf("Warp size:                     %d\n",
         CAFFE_TEST_LWDA_PROP.warpSize);
  printf("Maximum memory pitch:          %lu\n",
         CAFFE_TEST_LWDA_PROP.memPitch);
  printf("Maximum threads per block:     %d\n",
         CAFFE_TEST_LWDA_PROP.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i,
           CAFFE_TEST_LWDA_PROP.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i,
           CAFFE_TEST_LWDA_PROP.maxGridSize[i]);
  printf("Clock rate:                    %d\n", CAFFE_TEST_LWDA_PROP.clockRate);
  printf("Total constant memory:         %lu\n",
         CAFFE_TEST_LWDA_PROP.totalConstMem);
  printf("Texture alignment:             %lu\n",
         CAFFE_TEST_LWDA_PROP.textureAlignment);
  printf("Conlwrrent copy and exelwtion: %s\n",
         (CAFFE_TEST_LWDA_PROP.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",
         CAFFE_TEST_LWDA_PROP.multiProcessorCount);
  printf("Kernel exelwtion timeout:      %s\n",
         (CAFFE_TEST_LWDA_PROP.kernelExecTimeoutEnabled ? "Yes" : "No"));
  printf("Unified virtual addressing:    %s\n",
         (CAFFE_TEST_LWDA_PROP.unifiedAddressing ? "Yes" : "No"));
  EXPECT_TRUE(true);
}

}  // namespace caffe
