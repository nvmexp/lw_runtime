#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#include <lwblas_v2.h>
#include <lwca.h>
#include <lwda_runtime.h>
#include <lwrand.h>
#include <driver_types.h>  // lwca driver types
#ifndef NO_LWML
  #include <lwml.h>
#endif
#include <sched.h>

//
// LWCA macros
//

// LWCA: various checks for different function calls.
#define LWDA_CHECK(condition) \
  /* Code block avoids redefinition of lwdaError_t error */ \
  do { \
    lwdaError_t error = condition; \
    CHECK_EQ(error, lwdaSuccess) << " " << lwdaGetErrorString(error); \
  } while (0)

#define LWDA_CHECK_ARG(condition, arg) \
  do { \
    lwdaError_t error = condition; \
    CHECK_EQ(error, lwdaSuccess) << " " << lwdaGetErrorString(error) << \
        " (" << arg << ")"; \
  } while (0)

#define LWDA_CHECK_ARG2(condition, arg1, arg2) \
  do { \
    lwdaError_t error = condition; \
    CHECK_EQ(error, lwdaSuccess) << " " << lwdaGetErrorString(error) << \
        " (" << arg1 << ") (" << arg2 << ")"; \
  } while (0)

#define LWBLAS_CHECK(condition) \
  do { \
    lwblasStatus_t status = condition; \
    CHECK_EQ(status, LWBLAS_STATUS_SUCCESS) << " " \
      << caffe::lwblasGetErrorString(status); \
  } while (0)

#define LWRAND_CHECK(condition) \
  do { \
    lwrandStatus_t status = condition; \
    CHECK_EQ(status, LWRAND_STATUS_SUCCESS) << " " \
      << caffe::lwrandGetErrorString(status); \
  } while (0)

#define LWRAND_CHECK_ARG(condition, arg) \
  do { \
    lwrandStatus_t status = condition; \
    CHECK_EQ(status, LWRAND_STATUS_SUCCESS) << " " \
      << caffe::lwrandGetErrorString(status) << \
        " (" << arg << ")"; \
  } while (0)

// LWCA: grid stride looping
#define LWDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// LWCA: check for error after kernel exelwtion and exit loudly if there is one.
#define LWDA_POST_KERNEL_CHECK LWDA_CHECK(lwdaPeekAtLastError())

namespace caffe {

// LWCA: library error reporting.
const char* lwblasGetErrorString(lwblasStatus_t error);
const char* lwrandGetErrorString(lwrandStatus_t error);

// LWCA: use 512 threads per block
const int CAFFE_LWDA_NUM_THREADS = 512;
const int CAFFE_LWDA_NUM_THREADS_HALF = 512;

// LWCA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_LWDA_NUM_THREADS - 1) / CAFFE_LWDA_NUM_THREADS;
}
inline int CAFFE_GET_BLOCKS_HALF(const int N) {
  return (N + CAFFE_LWDA_NUM_THREADS_HALF - 1) /
      CAFFE_LWDA_NUM_THREADS_HALF;
}


#ifndef NO_LWML
namespace lwml {

// We might move this to Caffe TLS but we have to make sure that
// this one gets initialized immediately after thread start.
// Also, it's better to run this on current device (note that Caffe ctr
// might be exelwted somewhere else). So, let's keep it risk free.
struct LWMLInit {
  LWMLInit();
  ~LWMLInit();
  static std::mutex m_;
};

void setCpuAffinity(int device);

}
#endif  // NO_LWML

}  // namespace caffe

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
