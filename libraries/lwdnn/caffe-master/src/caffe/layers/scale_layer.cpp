#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void ScaleLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // scale is a learned parameter; initialize it
    axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis_;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[0] = Blob::create<Ftype>(scale_shape);
    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Ftype> > filler(GetFiller<Ftype>(filler_param));
    filler->Fill(this->blobs_[0].get());
  }
  if (param.bias_term()) {
  bias_term_ = true;
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_type("Bias");
    BiasParameter* bias_param = layer_param.mutable_bias_param();
    bias_param->set_axis(param.axis());
    if (bottom.size() > 1) {
      bias_param->set_num_axes(bottom[1]->num_axes());
    } else {
      bias_param->set_num_axes(param.num_axes());
    }
    bias_param->mutable_filler()->CopyFrom(param.bias_filler());
    bias_layer_ = LayerRegistry::CreateLayer(layer_param, this->parent_rank());
    bias_layer_->enforce_cpu();
    bias_bottom_vec_.resize(1);
    bias_bottom_vec_[0] = bottom[0];
    bias_layer_->SetUp(bias_bottom_vec_, top);
    bias_param_id_ = this->blobs_.size();
    this->blobs_.resize(bias_param_id_ + 1);
    this->blobs_[bias_param_id_] = bias_layer_->blobs()[0];
    bias_propagate_down_.resize(1, false);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Ftype, typename Btype>
void ScaleLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob* scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis_ == 0 in special case where scale is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
  // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
  // performant.)
  axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;
  for (int i = 0; i < scale->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis_);
  scale_dim_ = scale->count();
  inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
  if (bottom[0] == top[0]) {  // in-place computation
    temp_->ReshapeLike(bottom[0]);
  } else {
    top[0]->ReshapeLike(bottom[0]);
  }
  sum_result_.Reshape(vector<int>(1, outer_dim_ * scale_dim_));
  const int sum_mult_size = std::max(outer_dim_, inner_dim_);
  sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
  if (sum_multiplier_.cpu_data()[sum_mult_size - 1] != Btype(1)) {
    caffe_set<Btype>(sum_mult_size, Btype(1), sum_multiplier_.mutable_cpu_data());
  }
  if (bias_layer_) {
    bias_bottom_vec_[0] = top[0];
    bias_layer_->Reshape(bias_bottom_vec_, top);
  }
}

template <typename Ftype, typename Btype>
void ScaleLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  if (bottom[0] == top[0]) {
    // In-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we could skip this if not
    // doing Backward, but Caffe lwrrently provides no way of knowing whether
    // we'll need to do Backward at the time of the Forward call.
    caffe_copy<Ftype>(bottom[0]->count(), bottom[0]->cpu_data<Ftype>(),
               temp_->template mutable_cpu_data<Ftype>());
  }
  const Ftype* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->template cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const Ftype factor = scale_data[d];
      caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
      bottom_data += inner_dim_;
      top_data += inner_dim_;
    }
  }
  if (bias_layer_) {
    bias_layer_->Forward(bias_bottom_vec_, top);
  }
}

template <typename Ftype, typename Btype>
void ScaleLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (bias_layer_ &&
      this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
  }
  const bool scale_param = (bottom.size() == 1);
  Blob* scale = scale_param ? this->blobs_[0].get() : bottom[1];
  if ((!scale_param && propagate_down[1]) ||
      (scale_param && this->param_propagate_down_[0])) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    const bool in_place = (bottom[0] == top[0]);
    const Btype* bottom_data =
        in_place ? temp_->template cpu_data<Btype>() : bottom[0]->cpu_data<Btype>();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scale diff, and we're done.
    // If we're computing in-place (and not doing eltwise computation), this
    // hack doesn't work and we store the product in temp_.
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    Btype* product = (is_eltwise ? scale->mutable_cpu_diff<Btype>() :
        (in_place ? temp_->template mutable_cpu_data<Btype>() :
         bottom[0]->mutable_cpu_diff<Btype>()));
    caffe_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      Btype* sum_result = NULL;
      if (inner_dim_ == 1) {
        sum_result = product;
      } else if (sum_result_.count() == 1) {
        const Btype* sum_mult = sum_multiplier_.cpu_data();
        Btype* scale_diff = scale->mutable_cpu_diff<Btype>();
        if (scale_param) {
          Btype result = caffe_cpu_dot(inner_dim_, product, sum_mult);
          *scale_diff += result;
        } else {
          *scale_diff = caffe_cpu_dot(inner_dim_, product, sum_mult);
        }
      } else {
        const Btype* sum_mult = sum_multiplier_.cpu_data();
        sum_result = (outer_dim_ == 1) ?
            scale->mutable_cpu_diff<Btype>() : sum_result_.mutable_cpu_data();
        caffe_cpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       Btype(1), product, sum_mult, Btype(0), sum_result);
      }
      if (outer_dim_ != 1) {
        const Btype* sum_mult = sum_multiplier_.cpu_data();
        Btype* scale_diff = scale->mutable_cpu_diff<Btype>();
        if (scale_dim_ == 1) {
          if (scale_param) {
            Btype result = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
            *scale_diff += result;
          } else {
            *scale_diff = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
          }
        } else {
          caffe_cpu_gemv(CblasTrans, outer_dim_, scale_dim_,
                         Btype(1), sum_result, sum_mult, Btype(scale_param),
                         scale_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    const Btype* scale_data = scale->cpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    for (int n = 0; n < outer_dim_; ++n) {
      for (int d = 0; d < scale_dim_; ++d) {
        const Btype factor = scale_data[d];
        caffe_cpu_scale(inner_dim_, factor, top_diff, bottom_diff);
        bottom_diff += inner_dim_;
        top_diff += inner_dim_;
      }
    }
  }
}

INSTANTIATE_CLASS_FB(ScaleLayer);
REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe
