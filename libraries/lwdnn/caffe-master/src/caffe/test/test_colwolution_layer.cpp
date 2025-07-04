#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/colw_layer.hpp"

#ifdef USE_LWDNN

#include "caffe/layers/lwdnn_colw_layer.hpp"

#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference colwolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template<typename Dtype>
void caffe_colw(const TBlob<Dtype>* in, ColwolutionParameter* colw_param,
    const vector<shared_ptr<Blob>>& weights, TBlob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (colw_param->has_kernel_h() || colw_param->has_kernel_w()) {
    kernel_h = colw_param->kernel_h();
    kernel_w = colw_param->kernel_w();
  } else {
    kernel_h = kernel_w = colw_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (colw_param->has_pad_h() || colw_param->has_pad_w()) {
    pad_h = colw_param->pad_h();
    pad_w = colw_param->pad_w();
  } else {
    pad_h = pad_w = colw_param->pad_size() ? colw_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (colw_param->has_stride_h() || colw_param->has_stride_w()) {
    stride_h = colw_param->stride_h();
    stride_w = colw_param->stride_w();
  } else {
    stride_h = stride_w = colw_param->stride_size() ? colw_param->stride(0) : 1;
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w = colw_param->dilation_size() ? colw_param->dilation(0) : 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = colw_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Colwolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1) && in_y >= 0 &&
                          in_y < in->shape(2 + has_depth) && in_x >= 0 &&
                          in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset) * weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (colw_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data<Dtype>();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void caffe_colw(const TBlob<float>* in, ColwolutionParameter* colw_param,
    const vector<shared_ptr<Blob>>& weights, TBlob<float>* out);
template void caffe_colw(const TBlob<double>* in, ColwolutionParameter* colw_param,
    const vector<shared_ptr<Blob>>& weights, TBlob<double>* out);

template<typename TypeParam>
class ColwolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ColwolutionLayerTest() : blob_bottom_(new TBlob<Dtype>(2, 3, 6, 4)),
                           blob_bottom_2_(new TBlob<Dtype>(2, 3, 6, 4)),
                           blob_top_(new TBlob<Dtype>()), blob_top_2_(new TBlob<Dtype>()) {}

  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ColwolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual TBlob<Dtype>* MakeReferenceTop(TBlob<Dtype>* top) {
    this->ref_blob_top_.reset(new TBlob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_bottom_2_;
  TBlob<Dtype>* const blob_top_;
  TBlob<Dtype>* const blob_top_2_;
  shared_ptr<TBlob<Dtype>> ref_blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(ColwolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(ColwolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  colwolution_param->set_num_output(3);
  colwolution_param->set_group(3);
  layer.reset(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(ColwolutionLayerTest, TestSimpleColwolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(4);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("constant");
  colwolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_colw(this->blob_bottom_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<Dtype>(1e-4, 2e-2));
  }
  caffe_colw(this->blob_bottom_2_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<Dtype>(1e-4, 2e-2));
  }
}

TYPED_TEST(ColwolutionLayerTest, TestDilatedColwolution) {
  typedef typename TypeParam::Dtype Dtype;
  vector<int> bottom_shape;
  bottom_shape.push_back(2);
  bottom_shape.push_back(3);
  bottom_shape.push_back(8);
  bottom_shape.push_back(7);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
  }
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_dilation(2);
  colwolution_param->set_num_output(4);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("constant");
  colwolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_colw(this->blob_bottom_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_colw(this->blob_bottom_2_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ColwolutionLayerTest, Test0DColwolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  const int kNumOutput = 3;
  colwolution_param->set_num_output(kNumOutput);
  colwolution_param->set_axis(3);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  vector<int> top_shape = this->blob_bottom_->shape();
  top_shape[3] = kNumOutput;
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(top_shape, this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  vector<int> weight_offset(2);
  const Blob* weight = layer->blobs()[0].get();
  const Blob* bias = layer->blobs()[1].get();
  const int num = this->blob_top_->count(3);
  const int dim = this->blob_top_->shape(3);
  const int bottom_dim = this->blob_bottom_->shape(3);
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < dim; ++d) {
      weight_offset[0] = d;
      Dtype value = bias->cpu_data<Dtype>()[d];
      for (int bottom_d = 0; bottom_d < bottom_dim; ++bottom_d) {
        weight_offset[1] = bottom_d;
        value += weight->data_at(weight_offset) *
                 this->blob_bottom_->cpu_data()[n * bottom_dim + bottom_d];
      }
      EXPECT_NEAR(value, this->blob_top_->cpu_data()[n * dim + d], tol<Dtype>(1e-4, 1e-2));
    }
  }
}

TYPED_TEST(ColwolutionLayerTest, TestSimple3DColwolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 5;
  bottom_shape[3] = this->blob_bottom_vec_[0]->shape(2);
  bottom_shape[4] = this->blob_bottom_vec_[0]->shape(3);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(4);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_colw(this->blob_bottom_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<Dtype>(1e-3, 1e-1));
  }
  caffe_colw(this->blob_bottom_2_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<Dtype>(1e-3, 1e-1));
  }
}

TYPED_TEST(ColwolutionLayerTest, TestDilated3DColwolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 6;
  bottom_shape[3] = 7;
  bottom_shape[4] = 8;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_dilation(2);
  colwolution_param->set_num_output(4);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_colw(this->blob_bottom_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    Dtype rel_err = std::max(Dtype(1.), Dtype(fabs(top_data[i]))) * tol<Dtype>(1e-4, 5e-2);
    EXPECT_NEAR(top_data[i], ref_top_data[i], rel_err);
  }
  caffe_colw(this->blob_bottom_2_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    Dtype rel_err = std::max(Dtype(1.), Dtype(fabs(top_data[i]))) * tol<Dtype>(1e-4, 3e-2);
    EXPECT_NEAR(top_data[i], ref_top_data[i], rel_err);
  }
}

TYPED_TEST(ColwolutionLayerTest, Test1x1Colwolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(1);
  colwolution_param->add_stride(1);
  colwolution_param->set_num_output(4);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("constant");
  colwolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_colw(this->blob_bottom_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<Dtype>(1e-4, 5e-3));
  }
}

TYPED_TEST(ColwolutionLayerTest, TestSimpleColwolutionGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(3);
  colwolution_param->set_group(3);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("constant");
  colwolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_colw(this->blob_bottom_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<Dtype>(1e-4, 1e-2));
  }
}

TYPED_TEST(ColwolutionLayerTest, TestSobelColwolution) {
  // Test separable colwolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the colwolution of two rectangular filters.
  typedef typename TypeParam::Dtype Dtype;
  // Fill bottoms with identical Gaussian noise.
  shared_ptr<GaussianFiller<Dtype>> filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new GaussianFiller<Dtype>(filler_param));
  filler->Fill(this->blob_bottom_);
  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
  // Compute Sobel G_x operator as 3 x 3 colwolution.
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(1);
  colwolution_param->set_bias_term(false);
  shared_ptr<Layer<Dtype, Dtype>> layer(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new TBlob<Dtype>(1, 3, 3, 3));
  Dtype* weights = layer->blobs()[0]->template mutable_cpu_data<Dtype>();
  for (int c = 0; c < 3; ++c) {
    int i = c * 9;  // 3 x 3 filter
    weights[i + 0] = -1;
    weights[i + 1] = 0;
    weights[i + 2] = 1;
    weights[i + 3] = -2;
    weights[i + 4] = 0;
    weights[i + 5] = 2;
    weights[i + 6] = -1;
    weights[i + 7] = 0;
    weights[i + 8] = 1;
  }
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 colwolutions.
  // (1) the [1 2 1] column filter
  vector<Blob*> sep_blob_bottom_vec;
  vector<Blob*> sep_blob_top_vec;
  shared_ptr<TBlob<Dtype>> blob_sep(new TBlob<Dtype>());
  sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
  sep_blob_top_vec.push_back(this->blob_top_2_);
  colwolution_param->clear_kernel_size();
  colwolution_param->clear_stride();
  colwolution_param->set_kernel_h(3);
  colwolution_param->set_kernel_w(1);
  colwolution_param->set_stride_h(2);
  colwolution_param->set_stride_w(1);
  colwolution_param->set_num_output(1);
  colwolution_param->set_bias_term(false);
  layer.reset(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new TBlob<Dtype>(1, 3, 3, 1));
  Dtype* weights_1 = layer->blobs()[0]->template mutable_cpu_data<Dtype>();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 3 x 1 filter
    weights_1[i + 0] = 1;
    weights_1[i + 1] = 2;
    weights_1[i + 2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
  // (2) the [-1 0 1] row filter
  blob_sep->CopyFrom(*this->blob_top_2_, false, true);
  sep_blob_bottom_vec.clear();
  sep_blob_bottom_vec.push_back(blob_sep.get());
  colwolution_param->set_kernel_h(1);
  colwolution_param->set_kernel_w(3);
  colwolution_param->set_stride_h(1);
  colwolution_param->set_stride_w(2);
  colwolution_param->set_num_output(1);
  colwolution_param->set_bias_term(false);
  layer.reset(new ColwolutionLayer<Dtype, Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new TBlob<Dtype>(1, 1, 1, 3));
  Dtype* weights_2 = layer->blobs()[0]->template mutable_cpu_data<Dtype>();
  weights_2[0] = -1;
  weights_2[1] = 0;
  weights_2[2] = 1;
  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
  // Test equivalence of full and separable filters.
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* sep_top_data = this->blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], tol<Dtype>(1e-4, 2e-2));
  }
}

TYPED_TEST(ColwolutionLayerTest, TestNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const int kernel_h = 11;
  const int kernel_w = 13;
  vector<int> bottom_shape(4);
  bottom_shape[0] = 15;
  bottom_shape[1] = 18;
  bottom_shape[2] = kernel_h * 2;
  bottom_shape[3] = kernel_w * 2;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->set_num_output(12);
  colwolution_param->set_bias_term(false);
  colwolution_param->set_group(6);
  colwolution_param->set_kernel_h(kernel_h);
  colwolution_param->set_kernel_w(kernel_w);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  TBlob<Dtype> weights;
  TBlob<Dtype> top_diff;
  // Shape and fill weights and top_diff.
  bool copy_diff;
  bool reshape;
  {
    ColwolutionLayer<Dtype, Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    top_diff.ReshapeLike(*this->blob_top_);
    filler.Fill(&top_diff);
    ASSERT_EQ(1, layer.blobs().size());
    copy_diff = false;
    reshape = true;
    weights.CopyFrom(*layer.blobs()[0], copy_diff, reshape);
  }
  vector<bool> propagate_down(1, true);
  TBlob<Dtype> result_2d;
  TBlob<Dtype> backward_result_2d;
  TBlob<Dtype> backward_weight_result_2d;
  // Test with 2D im2col
  {
    caffe_set<Dtype>(this->blob_top_->count(), TypedConsts<Dtype>::zero,
        this->blob_top_->mutable_cpu_data());
    caffe_set<Dtype>(this->blob_bottom_->count(), TypedConsts<Dtype>::zero,
        this->blob_bottom_->mutable_cpu_diff());
    caffe_set<Dtype>(weights.count(), TypedConsts<Dtype>::zero, weights.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result_2d.
    colwolution_param->set_force_nd_im2col(false);
    ColwolutionLayer<Dtype, Dtype> layer_2d(layer_param);
    layer_2d.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(1, layer_2d.blobs().size());
    copy_diff = false;
    reshape = false;
    layer_2d.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_2d.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false;
    reshape = true;
    result_2d.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy<Dtype>(top_diff.count(), top_diff.cpu_data(), this->blob_top_->mutable_cpu_diff());
    layer_2d.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
    copy_diff = true;
    reshape = true;
    backward_result_2d.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
  }
  TBlob<Dtype> result_nd;
  TBlob<Dtype> backward_result_nd;
  TBlob<Dtype> backward_weight_result_nd;
  // Test with ND im2col
  {
    caffe_set<Dtype>(this->blob_top_->count(), TypedConsts<Dtype>::zero,
        this->blob_top_->mutable_cpu_data());
    caffe_set<Dtype>(this->blob_bottom_->count(), TypedConsts<Dtype>::zero,
        this->blob_bottom_->mutable_cpu_diff());
    caffe_set<Dtype>(weights.count(), TypedConsts<Dtype>::zero, weights.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result_nd.
    colwolution_param->set_force_nd_im2col(true);
    ColwolutionLayer<Dtype, Dtype> layer_nd(layer_param);
    layer_nd.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(1, layer_nd.blobs().size());
    copy_diff = false;
    reshape = false;
    layer_nd.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_nd.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false;
    reshape = true;
    result_nd.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_nd.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy<Dtype>(top_diff.count(), top_diff.cpu_data(), this->blob_top_->mutable_cpu_diff());
    layer_nd.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
    copy_diff = true;
    reshape = true;
    backward_result_nd.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    backward_weight_result_nd.CopyFrom(weights, copy_diff, reshape);
  }
  ASSERT_EQ(result_nd.count(), result_2d.count());
  for (int i = 0; i < result_2d.count(); ++i) {
    EXPECT_EQ(result_2d.cpu_data()[i], result_nd.cpu_data()[i]);
  }
  ASSERT_EQ(backward_result_nd.count(), backward_result_2d.count());
  for (int i = 0; i < backward_result_2d.count(); ++i) {
    if (is_type<Dtype>(FLOAT16))
      EXPECT_NEAR(backward_result_2d.cpu_diff()[i], backward_result_nd.cpu_diff()[i], 0.2F);
    else
      EXPECT_EQ(backward_result_2d.cpu_diff()[i], backward_result_nd.cpu_diff()[i]);
  }
  ASSERT_EQ(backward_weight_result_nd.count(), backward_weight_result_2d.count());
  for (int i = 0; i < backward_weight_result_2d.count(); ++i) {
    EXPECT_EQ(backward_weight_result_2d.cpu_diff()[i], backward_weight_result_nd.cpu_diff()[i]);
  }
}

TYPED_TEST(ColwolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(2);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  ColwolutionLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 1e-1), tol<Dtype>(1e-3, 4e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ColwolutionLayerTest, TestDilatedGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  vector<int> bottom_shape;
  bottom_shape.push_back(2);
  bottom_shape.push_back(3);
  bottom_shape.push_back(5);
  bottom_shape.push_back(6);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
  }
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_dilation(2);
  colwolution_param->set_num_output(2);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  ColwolutionLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 1e-1), tol<Dtype>(1e-3, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ColwolutionLayerTest, TestGradient3D) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 5;
  bottom_shape[3] = this->blob_bottom_vec_[0]->shape(2);
  bottom_shape[4] = this->blob_bottom_vec_[0]->shape(3);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(2);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  ColwolutionLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(6e-2, 1.e-1), tol<Dtype>(1e-3, 3e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ColwolutionLayerTest, Test1x1Gradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  colwolution_param->add_kernel_size(1);
  colwolution_param->add_stride(1);
  colwolution_param->set_num_output(2);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  ColwolutionLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(5e-2, tol<Dtype>(1e-3, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ColwolutionLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<Dtype>());
  layer_param.set_backward_type(tp<Dtype>());
  layer_param.set_forward_math(tp<Dtype>());
  layer_param.set_backward_math(tp<Dtype>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(3);
  colwolution_param->set_group(3);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  ColwolutionLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(5e-2, tol<Dtype>(1e-3, 2e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

#ifdef USE_LWDNN

template<typename Dtype>
class LwDNNColwolutionLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  LwDNNColwolutionLayerTest() : blob_bottom_(new TBlob<Dtype>(2, 3, 6, 4)),
                                blob_bottom_2_(new TBlob<Dtype>(2, 3, 6, 4)),
                                blob_top_(new TBlob<Dtype>()), blob_top_2_(new TBlob<Dtype>()) {}

  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~LwDNNColwolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual TBlob<Dtype>* MakeReferenceTop(TBlob<Dtype>* top) {
    this->ref_blob_top_.reset(new TBlob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_bottom_2_;
  TBlob<Dtype>* const blob_top_;
  TBlob<Dtype>* const blob_top_2_;
  shared_ptr<TBlob<Dtype>> ref_blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(LwDNNColwolutionLayerTest, TestDtypes);

TYPED_TEST(LwDNNColwolutionLayerTest, TestSetupLwDNN) {
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<TypeParam, TypeParam>> layer(
      new LwDNNColwolutionLayer<TypeParam, TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  colwolution_param->set_num_output(3);
  colwolution_param->set_group(3);
  layer.reset(new LwDNNColwolutionLayer<TypeParam, TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(LwDNNColwolutionLayerTest, TestSimpleColwolutionLwDNN) {
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(4);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("constant");
  colwolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam, TypeParam>> layer(
      new LwDNNColwolutionLayer<TypeParam, TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  caffe_colw(this->blob_bottom_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<TypeParam>(1e-4, 2e-2));
  }
  caffe_colw(this->blob_bottom_2_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<TypeParam>(1e-4, 2e-2));
  }
}

TYPED_TEST(LwDNNColwolutionLayerTest, TestSimpleColwolutionGroupLwDNN) {
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(3);
  colwolution_param->set_group(3);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("constant");
  colwolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam, TypeParam>> layer(
      new LwDNNColwolutionLayer<TypeParam, TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference colwolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  caffe_colw(this->blob_bottom_, colwolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], tol<TypeParam>(1e-4, 1e-2))
              << i << " of " << this->blob_top_->count();
  }
}

TYPED_TEST(LwDNNColwolutionLayerTest, TestSobelColwolutionLwDNN) {
  // Test separable colwolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the colwolution of two rectangular filters.

  // Fill bottoms with identical Gaussian noise.
  shared_ptr<GaussianFiller<TypeParam>> filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new GaussianFiller<TypeParam>(filler_param));
  filler->Fill(this->blob_bottom_);
  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
  // Compute Sobel G_x operator as 3 x 3 colwolution.
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(1);
  colwolution_param->set_bias_term(false);
  shared_ptr<Layer<TypeParam, TypeParam>> layer(
      new LwDNNColwolutionLayer<TypeParam, TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new TBlob<TypeParam>(1, 3, 3, 3));
  TypeParam* weights = layer->blobs()[0]->template mutable_cpu_data<TypeParam>();
  for (int c = 0; c < 3; ++c) {
    int i = c * 9;  // 3 x 3 filter
    weights[i + 0] = -1;
    weights[i + 1] = 0;
    weights[i + 2] = 1;
    weights[i + 3] = -2;
    weights[i + 4] = 0;
    weights[i + 5] = 2;
    weights[i + 6] = -1;
    weights[i + 7] = 0;
    weights[i + 8] = 1;
  }
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 colwolutions.
  // (1) the [1 2 1] column filter
  vector<Blob*> sep_blob_bottom_vec;
  vector<Blob*> sep_blob_top_vec;
  shared_ptr<TBlob<TypeParam>> blob_sep(new TBlob<TypeParam>());
  sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
  sep_blob_top_vec.push_back(this->blob_top_2_);
  colwolution_param->clear_kernel_size();
  colwolution_param->clear_stride();
  colwolution_param->set_kernel_h(3);
  colwolution_param->set_kernel_w(1);
  colwolution_param->set_stride_h(2);
  colwolution_param->set_stride_w(1);
  colwolution_param->set_num_output(1);
  colwolution_param->set_bias_term(false);
  layer.reset(new LwDNNColwolutionLayer<TypeParam, TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new TBlob<TypeParam>(1, 3, 3, 1));
  TypeParam* weights_1 = layer->blobs()[0]->template mutable_cpu_data<TypeParam>();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 3 x 1 filter
    weights_1[i + 0] = 1;
    weights_1[i + 1] = 2;
    weights_1[i + 2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
  // (2) the [-1 0 1] row filter
  blob_sep->CopyFrom(*this->blob_top_2_, false, true);
  sep_blob_bottom_vec.clear();
  sep_blob_bottom_vec.push_back(blob_sep.get());
  colwolution_param->set_kernel_h(1);
  colwolution_param->set_kernel_w(3);
  colwolution_param->set_stride_h(1);
  colwolution_param->set_stride_w(2);
  colwolution_param->set_num_output(1);
  colwolution_param->set_bias_term(false);
  layer.reset(new LwDNNColwolutionLayer<TypeParam, TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new TBlob<TypeParam>(1, 1, 1, 3));
  TypeParam* weights_2 = layer->blobs()[0]->template mutable_cpu_data<TypeParam>();
  weights_2[0] = -1;
  weights_2[1] = 0;
  weights_2[2] = 1;
  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
  // Test equivalence of full and separable filters.
  const TypeParam* top_data = this->blob_top_->cpu_data();
  const TypeParam* sep_top_data = this->blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], tol<TypeParam>(1e-4, 2e-2));
  }
}

TYPED_TEST(LwDNNColwolutionLayerTest, TestGradientLwDNN) {
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  // colwolution_param->set_lwdnn_colwolution_algo_seeker(
  // ColwolutionParameter_LwDNNColwolutionAlgorithmSeeker_FINDEX);
  colwolution_param->set_colw_algos_override("1,1,1");
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(2);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  LwDNNColwolutionLayer<TypeParam, TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(tol<TypeParam>(4e-3, 1e-1), tol<TypeParam>(1e-3, 5e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(LwDNNColwolutionLayerTest, TestGradientGroupLwDNN) {
  LayerParameter layer_param;
  layer_param.set_forward_type(tp<TypeParam>());
  layer_param.set_backward_type(tp<TypeParam>());
  layer_param.set_forward_math(tp<TypeParam>());
  layer_param.set_backward_math(tp<TypeParam>());
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(3);
  colwolution_param->set_group(3);
  colwolution_param->set_lwdnn_colwolution_algo_seeker(
      ColwolutionParameter_LwDNNColwolutionAlgorithmSeeker_FINDEX);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  LwDNNColwolutionLayer<TypeParam, TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(tol<TypeParam>(5e-2, 1e-1), tol<TypeParam>(1e-2, 5e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

#endif

}  // namespace caffe
