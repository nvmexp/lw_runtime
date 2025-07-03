// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/blob.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
  lwdaDeviceProp CAFFE_TEST_LWDA_PROP;
}
using caffe::CAFFE_TEST_LWDA_PROP;

int main(int argc, char** argv) {
#if defined(DEBUG)
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = 0;
#endif
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);

  // Before starting testing, let's first print out a few lwca defice info.
  std::vector<int> devices;
  int device_count = 0;

  lwdaGetDeviceCount(&device_count);
  cout << "Lwca number of devices: " << device_count << endl;

  if (argc > 1) {
    // Use the given device
    devices.push_back(atoi(argv[1]));
    LWDA_CHECK(lwdaSetDevice(devices[0]));
  } else if (LWDA_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    devices.push_back(LWDA_TEST_DEVICE);
  }

  if (devices.size() == 1) {
    cout << "Setting to use device " << devices[0] << endl;
    LWDA_CHECK(lwdaSetDevice(devices[0]));
  } else {
    for (int i = 0; i < device_count; ++i)
      devices.push_back(i);
  }

  int device = 0;
  LWDA_CHECK(lwdaGetDevice(&device));
  cout << "Current device id: " << device << endl;
  LWDA_CHECK(lwdaGetDeviceProperties(&CAFFE_TEST_LWDA_PROP, device));

  cout << "Current device name: " << CAFFE_TEST_LWDA_PROP.name << endl;
  caffe::Caffe::SetDevice(device);
  caffe::Caffe::set_gpus(std::vector<int>(1, device));

  // ilwoke the test.
  int ret = RUN_ALL_TESTS();
  return ret;
}
