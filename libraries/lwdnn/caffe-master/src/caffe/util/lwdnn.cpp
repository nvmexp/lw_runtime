#ifdef USE_LWDNN
#include "caffe/util/lwdnn.hpp"

const char* lwdnnGetErrorString(lwdnnStatus_t status) {
  switch (status) {
    case LWDNN_STATUS_SUCCESS:
      return "LWDNN_STATUS_SUCCESS";
    case LWDNN_STATUS_NOT_INITIALIZED:
      return "LWDNN_STATUS_NOT_INITIALIZED";
    case LWDNN_STATUS_ALLOC_FAILED:
      return "LWDNN_STATUS_ALLOC_FAILED";
    case LWDNN_STATUS_BAD_PARAM:
      return "LWDNN_STATUS_BAD_PARAM";
    case LWDNN_STATUS_INTERNAL_ERROR:
      return "LWDNN_STATUS_INTERNAL_ERROR";
    case LWDNN_STATUS_ILWALID_VALUE:
      return "LWDNN_STATUS_ILWALID_VALUE";
    case LWDNN_STATUS_ARCH_MISMATCH:
      return "LWDNN_STATUS_ARCH_MISMATCH";
    case LWDNN_STATUS_MAPPING_ERROR:
      return "LWDNN_STATUS_MAPPING_ERROR";
    case LWDNN_STATUS_EXELWTION_FAILED:
      return "LWDNN_STATUS_EXELWTION_FAILED";
    case LWDNN_STATUS_NOT_SUPPORTED:
      return "LWDNN_STATUS_NOT_SUPPORTED";
    case LWDNN_STATUS_LICENSE_ERROR:
      return "LWDNN_STATUS_LICENSE_ERROR";
#if LWDNN_VERSION_MIN(6, 0, 1)
    case LWDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "LWDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
    default:
      break;
  }
  return "Unknown lwdnn status";
}


namespace caffe {
namespace lwdnn {

float dataType<float>::oneval = 1.0f;
float dataType<float>::zeroval = 0.0f;
const void* dataType<float>::one =
    static_cast<void *>(&dataType<float>::oneval);
const void* dataType<float>::zero =
    static_cast<void *>(&dataType<float>::zeroval);

double dataType<double>::oneval = 1.0;
double dataType<double>::zeroval = 0.0;
const void* dataType<double>::one =
    static_cast<void *>(&dataType<double>::oneval);
const void* dataType<double>::zero =
    static_cast<void *>(&dataType<double>::zeroval);

float dataType<float16>::oneval = 1.0f;
float dataType<float16>::zeroval = 0.0f;
const void* dataType<float16>::one =
    static_cast<void *>(&dataType<float16>::oneval);
const void* dataType<float16>::zero =
    static_cast<void *>(&dataType<float16>::zeroval);

}  // namespace lwdnn
}  // namespace caffe
#endif
