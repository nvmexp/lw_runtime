#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/decolw_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Since ColwolutionLayerTest checks the shared colw/decolw code in detail,
// we'll just do a simple forward test and a gradient check.
template<typename TypeParam>
class DecolwolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DecolwolutionLayerTest() : blob_bottom_(new TBlob<Dtype>(2, 3, 6, 4)),
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

  virtual ~DecolwolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_bottom_2_;
  TBlob<Dtype>* const blob_top_;
  TBlob<Dtype>* const blob_top_2_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(DecolwolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(DecolwolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype, Dtype>> layer(new DecolwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 13);
  EXPECT_EQ(this->blob_top_->width(), 9);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 13);
  EXPECT_EQ(this->blob_top_2_->width(), 9);
  // setting group should not change the shape
  colwolution_param->set_num_output(3);
  colwolution_param->set_group(3);
  layer.reset(new DecolwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 13);
  EXPECT_EQ(this->blob_top_->width(), 9);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 13);
  EXPECT_EQ(this->blob_top_2_->width(), 9);
}

TYPED_TEST(DecolwolutionLayerTest, TestSimpleDecolwolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(3);
  colwolution_param->add_stride(2);
  colwolution_param->set_num_output(4);
  colwolution_param->mutable_weight_filler()->set_type("constant");
  colwolution_param->mutable_weight_filler()->set_value(1);
  colwolution_param->mutable_bias_filler()->set_type("constant");
  colwolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype, Dtype>> layer(new DecolwolutionLayer<Dtype, Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // constant-fill the bottom blobs
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom_2_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // simply check that aclwmulation works with overlapping filters
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          Dtype expected = 3.1;
          bool h_overlap = h % 2 == 0 && h > 0 && h < this->blob_top_->height() - 1;
          bool w_overlap = w % 2 == 0 && w > 0 && w < this->blob_top_->width() - 1;
          if (h_overlap && w_overlap) {
            expected += 9;
          } else if (h_overlap || w_overlap) {
            expected += 3;
          }
          EXPECT_NEAR(top_data[this->blob_top_->offset(n, c, h, w)], expected,
              tol<Dtype>(1e-4, 1e-2));
        }
      }
    }
  }
}

TYPED_TEST(DecolwolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  colwolution_param->add_kernel_size(2);
  colwolution_param->add_stride(1);
  colwolution_param->set_num_output(1);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  DecolwolutionLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(5e-2, 1e-1), tol<Dtype>(1e-3, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(DecolwolutionLayerTest, TestNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const int kernel_h = 11;
  const int kernel_w = 13;
  vector<int> bottom_shape(4);
  bottom_shape[0] = 15;
  bottom_shape[1] = 12;
  bottom_shape[2] = kernel_h * 2;
  bottom_shape[3] = kernel_w * 2;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->set_num_output(18);
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
    DecolwolutionLayer<Dtype, Dtype> layer(layer_param);
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
    DecolwolutionLayer<Dtype, Dtype> layer_2d(layer_param);
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
    DecolwolutionLayer<Dtype, Dtype> layer_nd(layer_param);
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
    if (is_type<Dtype>(FLOAT16))
      EXPECT_NEAR(result_2d.cpu_data()[i], result_nd.cpu_data()[i], 0.5F);
    else
      EXPECT_EQ(result_2d.cpu_data()[i], result_nd.cpu_data()[i]);
  }
  ASSERT_EQ(backward_result_nd.count(), backward_result_2d.count());
  for (int i = 0; i < backward_result_2d.count(); ++i) {
    EXPECT_EQ(backward_result_2d.cpu_diff()[i], backward_result_nd.cpu_diff()[i]);
  }
  ASSERT_EQ(backward_weight_result_nd.count(), backward_weight_result_2d.count());
  for (int i = 0; i < backward_weight_result_2d.count(); ++i) {
    EXPECT_EQ(backward_weight_result_2d.cpu_diff()[i], backward_weight_result_nd.cpu_diff()[i]);
  }
}

TYPED_TEST(DecolwolutionLayerTest, TestGradient3D) {
  typedef typename TypeParam::Dtype Dtype;
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 2;
  bottom_shape[3] = 3;
  bottom_shape[4] = 2;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ColwolutionParameter* colwolution_param = layer_param.mutable_colwolution_param();
  colwolution_param->add_kernel_size(2);
  colwolution_param->add_stride(2);
  colwolution_param->add_pad(1);
  colwolution_param->set_num_output(2);
  colwolution_param->mutable_weight_filler()->set_type("gaussian");
  colwolution_param->mutable_bias_filler()->set_type("gaussian");
  DecolwolutionLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 1e-1), tol<Dtype>(1e-3, 1e-1));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe
