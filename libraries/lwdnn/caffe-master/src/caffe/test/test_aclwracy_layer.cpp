#include <cfloat>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/aclwracy_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class AclwracyLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  AclwracyLayerTest()
      : blob_bottom_data_(new TBlob<Dtype>()),
        blob_bottom_label_(new TBlob<Dtype>()),
        blob_top_(new TBlob<Dtype>()),
        blob_top_per_class_(new TBlob<Dtype>()),
        top_k_(3) {
    vector<int> shape(2);
    shape[0] = 100;
    shape[1] = 10;
    blob_bottom_data_->Reshape(shape);
    shape.resize(1);
    blob_bottom_label_->Reshape(shape);
    FillBottoms();

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_per_class_vec_.push_back(blob_top_);
    blob_top_per_class_vec_.push_back(blob_top_per_class_);
  }

  virtual void FillBottoms() {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
          static_cast<caffe::rng_t*>(rng->generator());
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = (*prefetch_rng)() % 10;
    }
  }

  virtual ~AclwracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
    delete blob_top_per_class_;
  }
  TBlob<Dtype>* const blob_bottom_data_;
  TBlob<Dtype>* const blob_bottom_label_;
  TBlob<Dtype>* const blob_top_;
  TBlob<Dtype>* const blob_top_per_class_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
  vector<Blob*> blob_top_per_class_vec_;
  int top_k_;
};

TYPED_TEST_CASE(AclwracyLayerTest, TestDtypes);

TYPED_TEST(AclwracyLayerTest, TestSetup) {
  LayerParameter layer_param;
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(AclwracyLayerTest, TestSetupTopK) {
  LayerParameter layer_param;
  AclwracyParameter* aclwracy_param =
      layer_param.mutable_aclwracy_param();
  aclwracy_param->set_top_k(5);
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(AclwracyLayerTest, TestSetupOutputPerClass) {
  LayerParameter layer_param;
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_per_class_->num(), 10);
  EXPECT_EQ(this->blob_top_per_class_->channels(), 1);
  EXPECT_EQ(this->blob_top_per_class_->height(), 1);
  EXPECT_EQ(this->blob_top_per_class_->width(), 1);
}

TYPED_TEST(AclwracyLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    max_value = - max_dtype<TypeParam>();
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(this->blob_bottom_label_->data_at(i, 0, 0, 0))) {
      ++num_correct_labels;
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, tol<TypeParam>(1e-4, 2e-2));
}

TYPED_TEST(AclwracyLayerTest, TestForwardWithSpatialAxes) {
  this->blob_bottom_data_->Reshape(2, 10, 4, 5);
  vector<int> label_shape(3);
  label_shape[0] = 2; label_shape[1] = 4; label_shape[2] = 5;
  this->blob_bottom_label_->Reshape(label_shape);
  this->FillBottoms();
  LayerParameter layer_param;
  layer_param.mutable_aclwracy_param()->set_axis(1);
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  const int num_labels = this->blob_bottom_label_->count();
  int max_id;
  int num_correct_labels = 0;
  vector<int> label_offset(3);
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    for (int h = 0; h < this->blob_bottom_data_->height(); ++h) {
      for (int w = 0; w < this->blob_bottom_data_->width(); ++w) {
        max_value = - max_dtype<TypeParam>();
        max_id = 0;
        for (int c = 0; c < this->blob_bottom_data_->channels(); ++c) {
          const TypeParam pred_value =
              this->blob_bottom_data_->data_at(n, c, h, w);
          if (pred_value > max_value) {
            max_value = pred_value;
            max_id = c;
          }
        }
        label_offset[0] = n; label_offset[1] = h; label_offset[2] = w;
        const int correct_label =
            static_cast<int>(this->blob_bottom_label_->data_at(label_offset));
        if (max_id == correct_label) {
          ++num_correct_labels;
        }
      }
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / TypeParam(num_labels), tol<TypeParam>(1e-4, 1e-1));
}

TYPED_TEST(AclwracyLayerTest, TestForwardIgnoreLabel) {
  LayerParameter layer_param;
  const TypeParam kIgnoreLabelValue = -1;
  layer_param.mutable_aclwracy_param()->set_ignore_label(kIgnoreLabelValue);
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  // Manually set some labels to the ignore label value (-1).
  this->blob_bottom_label_->mutable_cpu_data()[2] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[5] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[32] = kIgnoreLabelValue;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  int count = 0;
  for (int i = 0; i < 100; ++i) {
    if (kIgnoreLabelValue == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      continue;
    }
    ++count;
    max_value = - max_dtype<TypeParam>();
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++num_correct_labels;
    }
  }
  EXPECT_EQ(count, 97);  // We set 3 out of 100 labels to kIgnoreLabelValue.
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
      static_cast<float>(num_correct_labels) / count, tol<TypeParam>(1e-4, 2e-2));
}

TYPED_TEST(AclwracyLayerTest, TestForwardCPUTopK) {
  LayerParameter layer_param;
  AclwracyParameter* aclwracy_param = layer_param.mutable_aclwracy_param();
  aclwracy_param->set_top_k(this->top_k_);
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam lwrrent_value;
  int lwrrent_rank;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 10; ++j) {
      lwrrent_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
      lwrrent_rank = 0;
      for (int k = 0; k < 10; ++k) {
        if (this->blob_bottom_data_->data_at(i, k, 0, 0) > lwrrent_value) {
          ++lwrrent_rank;
        }
      }
      if (lwrrent_rank < this->top_k_ &&
        j == static_cast<int>(this->blob_bottom_label_->data_at(i, 0, 0, 0))) {
        ++num_correct_labels;
      }
    }
  }

  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, tol<TypeParam>(1e-4, 2e-2));
}

TYPED_TEST(AclwracyLayerTest, TestForwardCPUPerClass) {
  LayerParameter layer_param;
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_per_class_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  const int num_class = this->blob_top_per_class_->num();
  vector<int> correct_per_class(num_class, 0);
  vector<int> num_per_class(num_class, 0);
  for (int i = 0; i < 100; ++i) {
    max_value = - max_dtype<TypeParam>();
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    const int id = static_cast<int>(this->blob_bottom_label_->data_at(i, 0, 0, 0));
    ++num_per_class[id];
    if (max_id == id) {
      ++num_correct_labels;
      ++correct_per_class[max_id];
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, tol<TypeParam>(1e-4, 2e-2));
  if (sizeof(TypeParam) >= 4) {
    for (int i = 0; i < num_class; ++i) {
      TypeParam aclwracy_per_class = (num_per_class[i] > 0 ?
         static_cast<TypeParam>(correct_per_class[i]) / num_per_class[i] : 0);
      EXPECT_NEAR(this->blob_top_per_class_->data_at(i, 0, 0, 0),
          aclwracy_per_class, 1e-4);
    }
  }
}


TYPED_TEST(AclwracyLayerTest, TestForwardCPUPerClassWithIgnoreLabel) {
  LayerParameter layer_param;
  const TypeParam kIgnoreLabelValue = -1;
  layer_param.mutable_aclwracy_param()->set_ignore_label(kIgnoreLabelValue);
  AclwracyLayer<TypeParam, TypeParam> layer(layer_param);
  // Manually set some labels to the ignore label value (-1).
  this->blob_bottom_label_->mutable_cpu_data()[2] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[5] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[32] = kIgnoreLabelValue;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_per_class_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  const int num_class = this->blob_top_per_class_->num();
  vector<int> correct_per_class(num_class, 0);
  vector<int> num_per_class(num_class, 0);
  int count = 0;
  for (int i = 0; i < 100; ++i) {
    if (kIgnoreLabelValue == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      continue;
    }
    ++count;
    max_value = - max_dtype<TypeParam>();
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    const int id = static_cast<int>(this->blob_bottom_label_->data_at(i, 0, 0, 0));
    ++num_per_class[id];
    if (max_id == id) {
      ++num_correct_labels;
      ++correct_per_class[max_id];
    }
  }
  EXPECT_EQ(count, 97);
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
      static_cast<float>(num_correct_labels) / count, tol<TypeParam>(1e-4, 2e-2));
  if (sizeof(TypeParam) >= 4) {
    for (int i = 0; i < 10; ++i) {
      TypeParam aclwracy_per_class = (num_per_class[i] > 0 ?
         static_cast<TypeParam>(correct_per_class[i]) / num_per_class[i] : 0);
      EXPECT_NEAR(this->blob_top_per_class_->data_at(i, 0, 0, 0),
          aclwracy_per_class, 1e-4);
    }
  }
}

}  // namespace caffe
