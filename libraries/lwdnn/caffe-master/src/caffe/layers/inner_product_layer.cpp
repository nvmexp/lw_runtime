#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void
InnerProductLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    // CPU filler always 32 bits
    this->blobs_[0] = Blob::create<Ftype>(weight_shape);
    shared_ptr<Filler<Ftype>> weight_filler(
        GetFiller<Ftype>(this->layer_param_.inner_product_param().weight_filler()));

    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1] = Blob::create<Ftype, Btype>(bias_shape);
      shared_ptr<Filler<Ftype>> bias_filler(
          GetFiller<Ftype>(this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
      bias_multiplier_ = Blob::create<Ftype>(bias_shape);
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Ftype, typename Btype>
void
InnerProductLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K) << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_->Reshape(bias_shape);
    bias_multiplier_->set_data(1.F);
  }
}

template<typename Ftype, typename Btype>
void InnerProductLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  const Ftype* weight = this->blobs_[0]->template cpu_data<Ftype>();
  caffe_cpu_gemm(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans, M_, N_, K_, (Ftype) 1.,
      bottom_data, weight, (Ftype) 0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Ftype) 1.,
        bias_multiplier_->template cpu_data<Ftype>(), this->blobs_[1]->template cpu_data<Ftype>(),
        (Ftype) 1., top_data);
  }
}

template<typename Ftype, typename Btype>
void InnerProductLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    const Btype* bottom_data = bottom[0]->cpu_data<Btype>();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Btype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Btype) 1., bottom_data, top_diff,
          (Btype) 1., this->blobs_[0]->template mutable_cpu_diff<Btype>());
    } else {
      caffe_cpu_gemm<Btype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Btype) 1., top_diff, bottom_data,
          (Btype) 1., this->blobs_[0]->template mutable_cpu_diff<Btype>());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    // Gradient with respect to bias
    caffe_cpu_gemv<Btype>(CblasTrans, M_, N_, (Btype) 1., top_diff,
        bias_multiplier_->template cpu_data<Btype>(), (Btype) 1.,
        this->blobs_[1]->template mutable_cpu_diff<Btype>());
  }
  if (propagate_down[0]) {
    const Btype* top_diff = top[0]->cpu_diff<Btype>();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Btype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Btype) 1., top_diff,
          this->blobs_[0]->template cpu_data<Btype>(), (Btype) 0.,
          bottom[0]->mutable_cpu_diff<Btype>());
    } else {
      caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Btype) 1., top_diff,
          this->blobs_[0]->template cpu_data<Btype>(), (Btype) 0.,
          bottom[0]->mutable_cpu_diff<Btype>());
    }
  }
}

INSTANTIATE_CLASS_FB(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
