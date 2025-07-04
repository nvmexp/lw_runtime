#ifndef CAFFE_LWDNN_TANH_LAYER_HPP_
#define CAFFE_LWDNN_TANH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"

namespace caffe {

#ifdef USE_LWDNN
/**
 * @brief LwDNN acceleration of TanHLayer.
 */
template <typename Ftype, typename Btype>
class LwDNNTanHLayer : public TanHLayer<Ftype, Btype> {
 public:
  explicit LwDNNTanHLayer(const LayerParameter& param)
      : TanHLayer<Ftype, Btype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual ~LwDNNTanHLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  bool handles_setup_;
  lwdnnTensorDescriptor_t fwd_bottom_desc_, bwd_bottom_desc_;
  lwdnnTensorDescriptor_t fwd_top_desc_, bwd_top_desc_;
  lwdnnActivationDescriptor_t activ_desc_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_LWDNN_TANH_LAYER_HPP_
