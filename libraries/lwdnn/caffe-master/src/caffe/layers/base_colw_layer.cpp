#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_colw_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void BaseColwolutionLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ColwolutionParameter colw_param = this->layer_param_.colwolution_param();
  force_nd_im2col_ = colw_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(colw_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (colw_param.has_kernel_h() || colw_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2) << "kernel_h & kernel_w can only be used for 2D colwolution.";
    CHECK_EQ(0, colw_param.kernel_size_size())
      << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = colw_param.kernel_h();
    kernel_shape_data[1] = colw_param.kernel_w();
  } else {
    const int num_kernel_dims = colw_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
    << "kernel_size must be specified once, or once per spatial dimension "
    << "(kernel_size specified " << num_kernel_dims << " times; " << num_spatial_axes_
    << " spatial dims).";
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] = colw_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (colw_param.has_stride_h() || colw_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2) << "stride_h & stride_w can only be used for 2D colwolution.";
    CHECK_EQ(0, colw_param.stride_size())
      << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = colw_param.stride_h();
    stride_data[1] = colw_param.stride_w();
  } else {
    const int num_stride_dims = colw_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 || num_stride_dims == num_spatial_axes_)
    << "stride must be specified once, or once per spatial dimension " << "(stride specified "
    << num_stride_dims << " times; " << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride : colw_param.stride(
          (num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (colw_param.has_pad_h() || colw_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2) << "pad_h & pad_w can only be used for 2D colwolution.";
    CHECK_EQ(0, colw_param.pad_size()) << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = colw_param.pad_h();
    pad_data[1] = colw_param.pad_w();
  } else {
    const int num_pad_dims = colw_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 || num_pad_dims == num_spatial_axes_)
    << "pad must be specified once, or once per spatial dimension " << "(pad specified "
    << num_pad_dims << " times; " << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad : colw_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = colw_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 || num_dilation_dims == num_spatial_axes_)
  << "dilation must be specified once, or once per spatial dimension " << "(dilation specified "
  << num_dilation_dims << " times; " << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation : colw_param.dilation(
        (num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 colwolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ = is_1x1_ && kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.colwolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.colwolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0) << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    colw_out_channels_ = channels_;
    colw_in_channels_ = num_output_;
  } else {
    colw_out_channels_ = num_output_;
    colw_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = colw_out_channels_;
  weight_shape[1] = colw_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.colwolution_param().bias_term();
  int b_term = 0;
  if (bias_term_)
    b_term = 1;
  vector<int> bias_shape(b_term, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + b_term, this->blobs_.size()) << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      TBlob<Ftype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape " << weight_shaped_blob.shape_string()
                 << "; instead, shape was " << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      TBlob<Ftype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape " << bias_shaped_blob.shape_string()
                 << "; instead, shape was " << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0] = Blob::create<Ftype, Btype>(weight_shape);
    shared_ptr<Filler<Ftype>> weight_filler(
        GetFiller<Ftype>(this->layer_param_.colwolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1] = Blob::create<Ftype, Btype>(bias_shape);
      shared_ptr<Filler<Ftype>> bias_filler(
          GetFiller<Ftype>(this->layer_param_.colwolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = colw_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Ftype, typename Btype>
void
BaseColwolutionLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
    << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
    << "Input size incompatible with colwolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
    << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(), bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    colw_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    colw_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * colw_out_spatial_dim_;
  output_offset_ = colw_out_channels_ * colw_out_spatial_dim_ / group_;
  // Setup input dimensions (colw_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  colw_input_shape_.Reshape(bottom_dim_blob_shape);
  int* colw_input_shape_data = colw_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      colw_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      colw_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 colwolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = colw_in_channels_ * colw_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    bias_multiplier_.set_data(1.F);
  }
}

INSTANTIATE_CLASS_FB(BaseColwolutionLayer);

}  // namespace caffe
