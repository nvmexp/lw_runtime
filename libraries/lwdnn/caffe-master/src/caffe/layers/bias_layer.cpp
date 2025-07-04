#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void BiasLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // bias is a learned parameter; initialize it
    const BiasParameter& param = this->layer_param_.bias_param();
    const int axis = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis + num_axes)
          << "bias blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> bias_shape(shape_start, shape_end);
    this->blobs_[0] = Blob::create<Ftype>(bias_shape);
    shared_ptr<Filler<Ftype> > filler(GetFiller<Ftype>(param.filler()));
    filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Ftype, typename Btype>
void BiasLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const BiasParameter& param = this->layer_param_.bias_param();
  Blob* bias = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis == 0 in special case where bias is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis == 0 and (therefore) outer_dim_ == 1.
  const int axis = (bias->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis + bias->num_axes())
      << "bias blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis;
  for (int i = 0; i < bias->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis + i), bias->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis + i
        << ") and bias->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis);
  bias_dim_ = bias->count();
  inner_dim_ = bottom[0]->count(axis + bias->num_axes());
  dim_ = bias_dim_ * inner_dim_;
  if (bottom[0] != top[0]) {
    top[0]->ReshapeLike(*bottom[0]);
  }
  bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
  if (bias_multiplier_.cpu_data()[inner_dim_ - 1] != Ftype(1)) {
    bias_multiplier_.set_data(1.F);
  }
}

template <typename Ftype, typename Btype>
void BiasLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const Ftype* bias_data = ((bottom.size() > 1) ?
      bottom[1] : this->blobs_[0].get())->template cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  if (bottom[0] != top[0]) {
    const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  for (int n = 0; n < outer_dim_; ++n) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bias_dim_,
        inner_dim_, 1, Ftype(1), bias_data,
        bias_multiplier_.cpu_data(), Ftype(1), top_data);
    top_data += dim_;
  }
}

template <typename Ftype, typename Btype>
void BiasLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    Btype* bias_diff = (bias_param ? this->blobs_[0].get() :
        bottom[1])->template mutable_cpu_diff<Btype>();
    bool aclwm = bias_param;
    for (int n = 0; n < outer_dim_; ++n) {
      caffe_cpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Btype(1),
          top_diff, bias_multiplier_.template cpu_data<Btype>(), Btype(aclwm), bias_diff);
      top_diff += dim_;
      aclwm = true;
    }
  }
}

INSTANTIATE_CLASS_FB(BiasLayer);
REGISTER_LAYER_CLASS(Bias);

}  // namespace caffe
