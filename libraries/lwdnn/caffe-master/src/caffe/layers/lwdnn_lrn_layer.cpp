#ifdef USE_LWDNN
#include <vector>

#include "caffe/layers/lwdnn_lrn_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void LwDNNLRNLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  LRNLayer<Ftype, Btype>::LayerSetUp(bottom, top);

  LWDNN_CHECK(lwdnnCreateLRNDescriptor(&norm_desc_));
  lwdnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  this->size_ = this->layer_param().lrn_param().local_size();
  this->alpha_ = this->layer_param().lrn_param().alpha();
  this->beta_ = this->layer_param().lrn_param().beta();
  this->k_ = this->layer_param().lrn_param().k();
}

template <typename Ftype, typename Btype>
void LwDNNLRNLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  LRNLayer<Ftype, Btype>::Reshape(bottom, top);
  lwdnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  lwdnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  lwdnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  lwdnn::setTensor4dDesc<Btype>(&bwd_top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  LWDNN_CHECK(lwdnnSetLRNDescriptor(norm_desc_,
      this->size_, this->alpha_, this->beta_, this->k_));
}

template <typename Ftype, typename Btype>
LwDNNLRNLayer<Ftype, Btype>::~LwDNNLRNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  lwdnnDestroyTensorDescriptor(fwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(fwd_top_desc_);
  lwdnnDestroyTensorDescriptor(bwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(bwd_top_desc_);

  // destroy LRN handle
  LWDNN_CHECK(lwdnnDestroyLRNDescriptor(norm_desc_));
}

INSTANTIATE_CLASS_FB(LwDNNLRNLayer);

}   // namespace caffe

#endif
