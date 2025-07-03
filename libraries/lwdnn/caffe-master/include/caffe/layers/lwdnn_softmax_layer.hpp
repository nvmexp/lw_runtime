#ifndef CAFFE_LWDNN_SOFTMAX_LAYER_HPP_
#define CAFFE_LWDNN_SOFTMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

#ifdef USE_LWDNN
/**
 * @brief lwDNN implementation of SoftmaxLayer.
 *        Fallback to SoftmaxLayer for CPU mode.
 */
template <typename Ftype, typename Btype>
class LwDNNSoftmaxLayer : public SoftmaxLayer<Ftype, Btype> {
 public:
  explicit LwDNNSoftmaxLayer(const LayerParameter& param)
      : SoftmaxLayer<Ftype, Btype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual ~LwDNNSoftmaxLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
     const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  bool handles_setup_;
  lwdnnHandle_t handle_;
  lwdnnTensorDescriptor_t fwd_bottom_desc_, bwd_bottom_desc_;
  lwdnnTensorDescriptor_t fwd_top_desc_, bwd_top_desc_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_LWDNN_SOFTMAX_LAYER_HPP_
