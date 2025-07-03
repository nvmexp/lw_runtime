#ifdef USE_LWDNN

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/lwdnn_batch_norm_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void LwDNNBatchNormLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {

  BatchNormLayer<Ftype, Btype>::LayerSetUp(bottom, top);

  lwdnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  lwdnn::createTensor4dDesc<Ftype>(&fwd_scale_bias_mean_var_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_top_desc_);
  lwdnn::createTensor4dDesc<Btype>(&bwd_scale_bias_mean_var_desc_);

#if LWDNN_VERSION_MIN(7, 0, 0)
  mode_ = LWDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
  mode_ = LWDNN_BATCHNORM_SPATIAL;      // only SPATIAL mode is supported
#endif
  this->eps_ = std::max(this->eps_, LWDNN_BN_MIN_EPSILON);

  int channels = bottom[0]->channels();
  const Type btype = blobs_type();
  const vector<int> shape { 1, channels, 1, 1 };
  if (!this->scale_bias_) { // stubs for lwdnn
    scale_ones_ = Blob::create(btype, btype);
    scale_ones_->Reshape(shape);
    scale_ones_->set_data(1.F);
    bias_zeros_ = Blob::create(btype, btype);
    bias_zeros_->Reshape(shape);
    bias_zeros_->set_data(0.F);
  }
  save_mean_->Reshape(shape);
  save_ilw_var_->Reshape(shape);
  handles_setup_ = true;

  if (bottom == top) {  // LWDNN_BN does support in-place
    private_top_ = Blob::create<Ftype>(top[0]->shape());
    private_bottom_ = Blob::create<Ftype>(bottom[0]->shape());
  }
}

template<typename Ftype, typename Btype>
void
LwDNNBatchNormLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  BatchNormLayer<Ftype, Btype>::Reshape(bottom, top);

  int N = bottom[0]->num();
  int C = bottom[0]->channels();
  int H = bottom[0]->height();
  int W = bottom[0]->width();
  // set up main tensors
  lwdnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, N, C, H, W);
  lwdnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, N, C, H, W);
  lwdnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, N, C, H, W);
  lwdnn::setTensor4dDesc<Btype>(&bwd_top_desc_, N, C, H, W);
  // aux tensors for caching mean & ilwVar from fwd to bwd pass
  save_mean_->Reshape(1, C, 1, 1);
  save_ilw_var_->Reshape(1, C, 1, 1);
  if (!this->scale_bias_) {
    int C_old = scale_ones_->channels();
    if (C_old != C) {
      scale_ones_->Reshape(1, C, 1, 1);
      bias_zeros_->Reshape(1, C, 1, 1);
      scale_ones_->set_data(1.F);
      bias_zeros_->set_data(0.F);
    }
  }
  LWDNN_CHECK(
      lwdnnDeriveBNTensorDescriptor(fwd_scale_bias_mean_var_desc_, fwd_bottom_desc_, mode_));
  LWDNN_CHECK(
      lwdnnDeriveBNTensorDescriptor(bwd_scale_bias_mean_var_desc_, bwd_bottom_desc_, mode_));

  if (top[0] == bottom[0]) {
    if (!private_top_) {
      private_top_ = Blob::create<Ftype>(top[0]->shape());
    } else {
      private_top_->ReshapeLike(top[0]);
    }
    if (!private_bottom_) {
      private_bottom_ = Blob::create<Ftype>(bottom[0]->shape());
    } else {
      private_bottom_->ReshapeLike(bottom[0]);
    }
  }
}

template<typename Ftype, typename Btype>
LwDNNBatchNormLayer<Ftype, Btype>::~LwDNNBatchNormLayer() {
  if (!handles_setup_) return;
  lwdnnDestroyTensorDescriptor(fwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(bwd_bottom_desc_);
  lwdnnDestroyTensorDescriptor(fwd_top_desc_);
  lwdnnDestroyTensorDescriptor(bwd_top_desc_);
  lwdnnDestroyTensorDescriptor(fwd_scale_bias_mean_var_desc_);
  lwdnnDestroyTensorDescriptor(bwd_scale_bias_mean_var_desc_);
}

INSTANTIATE_CLASS_FB(LwDNNBatchNormLayer);

}  // namespace caffe

#endif
