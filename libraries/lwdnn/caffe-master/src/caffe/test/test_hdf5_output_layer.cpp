#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename TypeParam>
class HDF5OutputLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HDF5OutputLayerTest()
      : input_file_name_(
        CMAKE_SOURCE_DIR "caffe/test/test_data/sample_data.h5"),
        blob_data_(new TBlob<Dtype>()),
        blob_label_(new TBlob<Dtype>()),
        num_(5),
        channels_(8),
        height_(5),
        width_(5) {
    output_file_name_ = MakeTempFilename();
  }

  virtual ~HDF5OutputLayerTest() {
    delete blob_data_;
    delete blob_label_;
  }

  void CheckBlobEqual(const TBlob<Dtype>& b1, const TBlob<Dtype>& b2);

  string output_file_name_;
  string input_file_name_;
  TBlob<Dtype>* const blob_data_;
  TBlob<Dtype>* const blob_label_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
  int num_;
  int channels_;
  int height_;
  int width_;
};

template<typename TypeParam>
void HDF5OutputLayerTest<TypeParam>::CheckBlobEqual(const TBlob<Dtype>& b1,
                                                    const TBlob<Dtype>& b2) {
  EXPECT_EQ(b1.num(), b2.num());
  EXPECT_EQ(b1.channels(), b2.channels());
  EXPECT_EQ(b1.height(), b2.height());
  EXPECT_EQ(b1.width(), b2.width());
  for (int n = 0; n < b1.num(); ++n) {
    for (int c = 0; c < b1.channels(); ++c) {
      for (int h = 0; h < b1.height(); ++h) {
        for (int w = 0; w < b1.width(); ++w) {
          EXPECT_EQ(b1.data_at(n, c, h, w), b2.data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST_CASE(HDF5OutputLayerTest, TestDtypesAndDevices);

TYPED_TEST(HDF5OutputLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LOG(INFO) << "Loading HDF5 file " << this->input_file_name_;
  hid_t file_id = H5Fopen(this->input_file_name_.c_str(), H5F_ACC_RDONLY,
                          H5P_DEFAULT);
  ASSERT_GE(file_id, 0)<< "Failed to open HDF5 file" <<
      this->input_file_name_;
  hdf5_load_nd_dataset(file_id, HDF5_DATA_DATASET_NAME, 0, 4,
                       this->blob_data_);
  hdf5_load_nd_dataset(file_id, HDF5_DATA_LABEL_NAME, 0, 4,
                       this->blob_label_);
  herr_t status = H5Fclose(file_id);
  EXPECT_GE(status, 0)<< "Failed to close HDF5 file " <<
      this->input_file_name_;
  this->blob_bottom_vec_.push_back(this->blob_data_);
  this->blob_bottom_vec_.push_back(this->blob_label_);

  LayerParameter param;
  param.mutable_hdf5_output_param()->set_file_name(this->output_file_name_);
  // This code block ensures that the layer is deconstructed and
  //   the output hdf5 file is closed.
  {
    HDF5OutputLayer<Dtype, Dtype> layer(param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(layer.file_name(), this->output_file_name_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }
  file_id = H5Fopen(this->output_file_name_.c_str(), H5F_ACC_RDONLY,
                          H5P_DEFAULT);
  ASSERT_GE(
    file_id, 0)<< "Failed to open HDF5 file" <<
          this->input_file_name_;

  TBlob<Dtype>* blob_data = new TBlob<Dtype>();
  hdf5_load_nd_dataset(file_id, HDF5_DATA_DATASET_NAME, 0, 4,
                       blob_data);
  this->CheckBlobEqual(*(this->blob_data_), *blob_data);

  TBlob<Dtype>* blob_label = new TBlob<Dtype>();
  hdf5_load_nd_dataset(file_id, HDF5_DATA_LABEL_NAME, 0, 4,
                       blob_label);
  this->CheckBlobEqual(*(this->blob_label_), *blob_label);

  status = H5Fclose(file_id);
  EXPECT_GE(status, 0) << "Failed to close HDF5 file " <<
      this->output_file_name_;
}

}  // namespace caffe
