#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/threshold_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ThresholdLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ThresholdLayerTest()
      : blob_bottom_(new TBlob<Dtype>(2, 3, 6, 5)),
        blob_top_(new TBlob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ThresholdLayerTest() { delete blob_bottom_; delete blob_top_; }
  TBlob<Dtype>* const blob_bottom_;
  TBlob<Dtype>* const blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(ThresholdLayerTest, TestDtypesAndDevices);


TYPED_TEST(ThresholdLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ThresholdLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(ThresholdLayerTest, Test) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ThresholdLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype threshold_ = layer_param.threshold_param().threshold();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_LE(top_data[i], 1.);
    if (top_data[i] == 0) {
      EXPECT_LE(bottom_data[i], threshold_);
    }
    if (top_data[i] == 1) {
      EXPECT_GT(bottom_data[i], threshold_);
    }
  }
}

TYPED_TEST(ThresholdLayerTest, Test2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ThresholdParameter* threshold_param =
    layer_param.mutable_threshold_param();
  threshold_param->set_threshold(0.5);
  ThresholdLayer<Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype threshold_ = layer_param.threshold_param().threshold();
  EXPECT_FLOAT_EQ(threshold_, 0.5);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_LE(top_data[i], 1.);
    if (top_data[i] == 0) {
      EXPECT_LE(bottom_data[i], threshold_);
    }
    if (top_data[i] == 1) {
      EXPECT_GT(bottom_data[i], threshold_);
    }
  }
}

}  // namespace caffe
