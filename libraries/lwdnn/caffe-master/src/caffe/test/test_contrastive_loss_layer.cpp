#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/contrastive_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ContrastiveLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ContrastiveLossLayerTest()
      : blob_bottom_data_i_(new TBlob<Dtype>(512, 2, 1, 1)),
        blob_bottom_data_j_(new TBlob<Dtype>(512, 2, 1, 1)),
        blob_bottom_y_(new TBlob<Dtype>(512, 1, 1, 1)),
        blob_top_loss_(new TBlob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_i_);
    blob_bottom_vec_.push_back(blob_bottom_data_i_);
    filler.Fill(this->blob_bottom_data_j_);
    blob_bottom_vec_.push_back(blob_bottom_data_j_);
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
      blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;  // 0 or 1
    }
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~ContrastiveLossLayerTest() {
    delete blob_bottom_data_i_;
    delete blob_bottom_data_j_;
    delete blob_bottom_y_;
    delete blob_top_loss_;
  }

  TBlob<Dtype>* const blob_bottom_data_i_;
  TBlob<Dtype>* const blob_bottom_data_j_;
  TBlob<Dtype>* const blob_bottom_y_;
  TBlob<Dtype>* const blob_top_loss_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(ContrastiveLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ContrastiveLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ContrastiveLossLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin = layer_param.contrastive_loss_param().margin();
  const int num = this->blob_bottom_data_i_->num();
  const int channels = this->blob_bottom_data_i_->channels();
  float loss(0);
  for (int i = 0; i < num; ++i) {
    Dtype dist_sq(0);
    for (int j = 0; j < channels; ++j) {
      Dtype diff = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_j_->cpu_data()[i*channels+j];
      dist_sq += diff*diff;
    }
    if (this->blob_bottom_y_->cpu_data()[i]) {  // similar pairs
      loss += dist_sq;
    } else {
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
    }
  }
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, tol<Dtype>(1e-6, 1e-2));
}

TYPED_TEST(ContrastiveLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ContrastiveLossLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(ContrastiveLossLayerTest, TestForwardLegacy) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_contrastive_loss_param()->set_legacy_version(true);
  ContrastiveLossLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin = layer_param.contrastive_loss_param().margin();
  const int num = this->blob_bottom_data_i_->num();
  const int channels = this->blob_bottom_data_i_->channels();
  float loss(0);
  for (int i = 0; i < num; ++i) {
    Dtype dist_sq(0);
    for (int j = 0; j < channels; ++j) {
      Dtype diff = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_j_->cpu_data()[i*channels+j];
      dist_sq += diff*diff;
    }
    if (this->blob_bottom_y_->cpu_data()[i]) {  // similar pairs
      loss += dist_sq;
    } else {
      loss += std::max(Dtype(margin - dist_sq), Dtype(0.0));
    }
  }
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, tol<Dtype>(1e-6, 2e-3));
}

TYPED_TEST(ContrastiveLossLayerTest, TestGradientLegacy) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_contrastive_loss_param()->set_legacy_version(true);
  ContrastiveLossLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-1, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

}  // namespace caffe
