#ifdef USE_LWDNN
#include <vector>

#include "caffe/layers/lwdnn_relu_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void LwDNNReLULayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  ReLULayer<Ftype, Btype>::LayerSetUp(bottom, top);
  // initialize lwDNN
  lwdnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_top_desc_);
  handles_setup_ = true;
  lwdnnCreateActivationDescriptor(&activ_desc_);
  lwdnnSetActivationDescriptor(activ_desc_, LWDNN_ACTIVATION_RELU, LWDNN_NOT_PROPAGATE_NAN, 0.0);
}

template <typename Ftype, typename Btype>
void LwDNNReLULayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  ReLULayer<Ftype, Btype>::Reshape(bottom, top);
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  lwdnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, N, K, H, W);
  lwdnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, N, K, H, W);
  lwdnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, N, K, H, W);
  lwdnn::setTensor4dDesc<Btype>(&bwd_top_desc_, N, K, H, W);
}

template <typename Ftype, typename Btype>
LwDNNReLULayer<Ftype, Btype>::~LwDNNReLULayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  lwdnnDestroyActivationDescriptor(this->activ_desc_);
  lwdnnDestroyTensorDescriptor(fwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(fwd_top_desc_);
  lwdnnDestroyTensorDescriptor(bwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(bwd_top_desc_);
}

INSTANTIATE_CLASS_FB(LwDNNReLULayer);

}  // namespace caffe
#endif
