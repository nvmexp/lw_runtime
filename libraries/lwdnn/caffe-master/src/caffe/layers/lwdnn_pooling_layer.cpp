#ifdef USE_LWDNN
#include <vector>

#include "caffe/layers/lwdnn_pooling_layer.hpp"

namespace caffe {

void createPoolingDesc(lwdnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, lwdnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = LWDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = LWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  LWDNN_CHECK(lwdnnCreatePoolingDescriptor(pool_desc));

  int dimA[2] = {h, w};
  int padA[2] = {pad_h, pad_w};
  int strideA[2] = {stride_h, stride_w};
  LWDNN_CHECK(lwdnnSetPoolingNdDescriptor(*pool_desc, *mode,
      LWDNN_NOT_PROPAGATE_NAN, 2, dimA, padA, strideA));
}

template <typename Ftype, typename Btype>
void LwDNNPoolingLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  PoolingLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  lwdnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_top_desc_);
  createPoolingDesc(&pooling_desc_, this->layer_param_.pooling_param().pool(), &mode_,
      this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
      this->stride_h_, this->stride_w_);
  handles_setup_ = true;

  if (this->is_max_pooling_) {
    private_top_.clear();
    for (int i = 0; i < top.size(); ++i) {
      private_top_.push_back(Blob::create<Btype>());
    }
  }
}

template <typename Ftype, typename Btype>
void LwDNNPoolingLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  PoolingLayer<Ftype, Btype>::Reshape(bottom, top);
  lwdnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  lwdnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, bottom[0]->num(),
      this->channels_, this->pooled_height_, this->pooled_width_);
  lwdnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  lwdnn::setTensor4dDesc<Btype>(&bwd_top_desc_, bottom[0]->num(),
      this->channels_, this->pooled_height_, this->pooled_width_);

  if (this->is_max_pooling_) {
    private_top_.resize(top.size());
    for (int i = 0; i < top.size(); ++i) {
      if (!private_top_[i]) {
        private_top_[i] = Blob::create<Btype>(top[i]->shape());
      } else {
        private_top_[i]->ReshapeLike(top[i]);
      }
    }
  }
}

template <typename Ftype, typename Btype>
LwDNNPoolingLayer<Ftype, Btype>::~LwDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  lwdnnDestroyTensorDescriptor(fwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(fwd_top_desc_);
  lwdnnDestroyTensorDescriptor(bwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(bwd_top_desc_);
  lwdnnDestroyPoolingDescriptor(pooling_desc_);
}

INSTANTIATE_CLASS_FB(LwDNNPoolingLayer);

}   // namespace caffe
#endif
