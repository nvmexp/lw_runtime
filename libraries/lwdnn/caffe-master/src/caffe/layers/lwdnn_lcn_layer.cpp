#ifdef USE_LWDNN
#include <vector>

#include "caffe/layers/lwdnn_lcn_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void LwDNNLCNLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  LRNLayer<Ftype, Btype>::LayerSetUp(bottom, top);

  LWDNN_CHECK(lwdnnCreateLRNDescriptor(&norm_desc_));
  lwdnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

template <typename Ftype, typename Btype>
void LwDNNLCNLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
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
  LWDNN_CHECK(lwdnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));

  // size for temp buffers
  tempDataSize_ = sizeof(Ftype) * bottom[0]->num() * this->channels_ *
      this->height_ * this->width_;
}

template <typename Ftype, typename Btype>
LwDNNLCNLayer<Ftype, Btype>::~LwDNNLCNLayer() {
  temp1_.release();
  temp2_.release();

  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  LWDNN_CHECK(lwdnnDestroyTensorDescriptor(fwd_bottom_desc_));
  LWDNN_CHECK(lwdnnDestroyTensorDescriptor(fwd_top_desc_));
  LWDNN_CHECK(lwdnnDestroyTensorDescriptor(bwd_bottom_desc_));
  LWDNN_CHECK(lwdnnDestroyTensorDescriptor(bwd_top_desc_));

  // destroy LRN handle
  LWDNN_CHECK(lwdnnDestroyLRNDescriptor(norm_desc_));
}

INSTANTIATE_CLASS_FB(LwDNNLCNLayer);

}   // namespace caffe
#endif
