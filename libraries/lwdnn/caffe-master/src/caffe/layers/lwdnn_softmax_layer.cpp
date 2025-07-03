#ifdef USE_LWDNN
#include <vector>

#include "caffe/layers/lwdnn_softmax_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void LwDNNSoftmaxLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  SoftmaxLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  // Initialize LWDNN.
  lwdnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_top_desc_);
  handles_setup_ = true;
}

template <typename Ftype, typename Btype>
void LwDNNSoftmaxLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  SoftmaxLayer<Ftype, Btype>::Reshape(bottom, top);
  int N = this->outer_num_;
  int K = bottom[0]->shape(this->softmax_axis_);
  int H = this->inner_num_;
  int W = 1;
  lwdnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, N, K, H, W);
  lwdnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, N, K, H, W);
  lwdnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, N, K, H, W);
  lwdnn::setTensor4dDesc<Btype>(&bwd_top_desc_, N, K, H, W);
}

template <typename Ftype, typename Btype>
LwDNNSoftmaxLayer<Ftype, Btype>::~LwDNNSoftmaxLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  lwdnnDestroyTensorDescriptor(fwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(fwd_top_desc_);
  lwdnnDestroyTensorDescriptor(bwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(bwd_top_desc_);
}

INSTANTIATE_CLASS_FB(LwDNNSoftmaxLayer);

}  // namespace caffe
#endif
