#ifndef CAFFE_UTIL_LWDNN_H_
#define CAFFE_UTIL_LWDNN_H_
#ifdef USE_LWDNN

#include <lwdnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/float16.hpp"

#define LWDNN_VERSION_MIN(major, minor, patch) \
    (LWDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#if !defined(LWDNN_VERSION) || !LWDNN_VERSION_MIN(6, 0, 0)
#error "LWCaffe 0.16 and higher requires LwDNN version 6.0.0 or higher"
#endif

#define LWDNN_CHECK(condition) \
  do { \
    lwdnnStatus_t status = condition; \
    CHECK_EQ(status, LWDNN_STATUS_SUCCESS) << " "\
      << lwdnnGetErrorString(status) << ", device " << Caffe::lwrrent_device(); \
  } while (0)

#define LWDNN_CHECK2(condition, arg1, arg2) \
  do { \
    lwdnnStatus_t status = condition; \
    CHECK_EQ(status, LWDNN_STATUS_SUCCESS) << "LwDNN error " \
      << (int)status << " " << (arg1) << " " << (arg2); \
  } while (0)

namespace caffe {

namespace lwdnn {

template<typename Dtype>
class dataType;

template<>
class dataType<float> {
 public:
  static const lwdnnDataType_t type = LWDNN_DATA_FLOAT;
  static const lwdnnDataType_t colw_type = LWDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};

template<>
class dataType<double> {
 public:
  static const lwdnnDataType_t type = LWDNN_DATA_DOUBLE;
  static const lwdnnDataType_t colw_type = LWDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template<>
class dataType<float16> {
 public:
  static const lwdnnDataType_t type = LWDNN_DATA_HALF;
  static const lwdnnDataType_t colw_type = LWDNN_DATA_HALF;
  static float oneval, zeroval;
  static const void *one, *zero;
};

inline
const void* one(Type type) {
  const void* ret = nullptr;
  switch (type) {
    case FLOAT:
      ret = dataType<float>::one;
      break;
    case FLOAT16:
      ret = dataType<float16>::one;
      break;
    case DOUBLE:
      ret = dataType<double>::one;
      break;
    default:
      LOG(FATAL) << "Unknown Type " << Type_Name(type);
      break;
  }
  return ret;
}

inline
const void* zero(Type type) {
  const void* ret = nullptr;
  switch (type) {
    case FLOAT:
      ret = dataType<float>::zero;
      break;
    case FLOAT16:
      ret = dataType<float16>::zero;
      break;
    case DOUBLE:
      ret = dataType<double>::zero;
      break;
    default:
      LOG(FATAL) << "Unknown Type " << Type_Name(type);
      break;
  }
  return ret;
}

inline
lwdnnDataType_t lwdnn_data_type(Type math) {
  lwdnnDataType_t ret;
  switch (math) {
    case FLOAT:
      ret = dataType<float>::colw_type;
      break;
    case FLOAT16:
      if (caffe::Caffe::device_capability(caffe::Caffe::device()) >= 600) {
        ret = dataType<float16>::colw_type;
      } else {
        ret = dataType<float>::colw_type;
      }
      break;
    case DOUBLE:
      ret = dataType<double>::colw_type;
      break;
    default:
      LOG(FATAL) << "Unknown Math type " << Type_Name(math);
      break;
  }
  return ret;
}

template <typename Dtype>
inline void createFilterDesc(lwdnnFilterDescriptor_t* desc, int n, int c, int h, int w) {
  LWDNN_CHECK(lwdnnCreateFilterDescriptor(desc));
  LWDNN_CHECK(lwdnnSetFilter4dDescriptor(*desc, lwdnn::dataType<Dtype>::type,
      LWDNN_TENSOR_NCHW, n, c, h, w));
}

inline void setColwolutionDesc(Type math, lwdnnColwolutionDescriptor_t colw,
      int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
  int padA[2] = {pad_h, pad_w};
  int strideA[2] = {stride_h, stride_w};
  int upscaleA[2] = {dilation_h, dilation_w};
  LWDNN_CHECK(lwdnnSetColwolutionNdDescriptor(colw, 2, padA, strideA, upscaleA,
      LWDNN_CROSS_CORRELATION, lwdnn::lwdnn_data_type(math)));
}

template<typename Dtype>
inline void createTensor4dDesc(lwdnnTensorDescriptor_t *desc) {
  LWDNN_CHECK(lwdnnCreateTensorDescriptor(desc));
}

template<typename Dtype>
inline void setTensor4dDesc(lwdnnTensorDescriptor_t *desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  LWDNN_CHECK(lwdnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
      n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template<typename Dtype>
inline void setTensor4dDesc(lwdnnTensorDescriptor_t *desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
      stride_n, stride_c, stride_h, stride_w);
}

inline void setTensor4dDesc(lwdnnTensorDescriptor_t *desc, lwdnnDataType_t type,
    Packing packing, const vector<int> &shape) {
  int stride_w = 0, stride_h = 0, stride_c = 0, stride_n = 0;
  const int n = shape[0];
  const int c = shape[1];
  const int h = shape[2];
  const int w = shape[3];
  if (packing == NCHW) {
    stride_w = 1;
    stride_h = w * stride_w;
    stride_c = h * stride_h;
    stride_n = c * stride_c;
  } else if (packing == NHWC) {
    stride_c = 1;
    stride_w = c * stride_c;
    stride_h = w * stride_w;
    stride_n = h * stride_h;
  } else {
    LOG(FATAL) << "Unknown packing";
  }
  LWDNN_CHECK(lwdnnSetTensor4dDescriptorEx(*desc, type,
      n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

inline void setTensor4dDesc(lwdnnTensorDescriptor_t *desc, Type type,
    Packing packing, const vector<int> &shape) {
  setTensor4dDesc(desc, lwdnn_data_type(type), packing, shape);
}

}  // namespace lwdnn

}  // namespace caffe

#endif  // USE_LWDNN
#endif  // CAFFE_UTIL_LWDNN_H_
