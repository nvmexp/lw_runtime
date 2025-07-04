#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/crop_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Ftype, typename Btype>
void CropLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  const CropParameter& param = this->layer_param_.crop_param();
  CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_LT(start_axis, input_dim) << "crop axis bigger than input dim";
  if (param.offset_size() > 1) {
    // the number of crop values specified must be equal to the number
    // of dimensions following axis
    CHECK_EQ(start_axis + param.offset_size(), input_dim)
      << "number of offset values specified must be equal to the number of "
      << "dimensions following axis.";
  }
}

template <typename Ftype, typename Btype>
void CropLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const CropParameter& param = this->layer_param_.crop_param();
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());

  // Initialize offsets to 0 and the new shape to the current shape of the data.
  offsets = vector<int>(input_dim, 0);
  vector<int> new_shape(bottom[0]->shape());

  // Determine crop offsets and the new shape post-crop.
  for (int i = 0; i < input_dim; ++i) {
    int crop_offset = 0;
    int new_size = bottom[0]->shape(i);
    if (i >= start_axis) {
      new_size = bottom[1]->shape(i);
      if (param.offset_size() == 1) {
        // If only one offset is given, all crops have the same offset.
        crop_offset = param.offset(0);
      } else if (param.offset_size() > 1) {
        // For several offsets, the number of offsets must be equal to the
        // number of dimensions to crop, that is dimensions after the axis.
        crop_offset = param.offset(i - start_axis);
      }
      // Check that the crop and offset are within the dimension's bounds.
      CHECK_GE(bottom[0]->shape(i) - crop_offset, bottom[1]->shape(i))
          << "the crop for dimension " << i << " is out-of-bounds with "
          << "size " << bottom[1]->shape(i) << " and offset " << crop_offset;
    }
    new_shape[i] = new_size;
    offsets[i] = crop_offset;
  }
  top[0]->Reshape(new_shape);
}

template <typename Ftype, typename Btype>
template <typename Dtype>
void CropLayer<Ftype, Btype>::crop_copy(const vector<Blob*>& bottom,
             const vector<Blob*>& top,
             const vector<int>& offsets,
             vector<int> indices,
             int lwr_dim,
             const Dtype* src_data,
             Dtype* dest_data,
             bool is_forward) {
  if (lwr_dim + 1 < top[0]->num_axes()) {
    // We are not yet at the final dimension, call copy relwrsively
    for (int i = 0; i < top[0]->shape(lwr_dim); ++i) {
      indices[lwr_dim] = i;
      crop_copy(bottom, top, offsets, indices, lwr_dim+1,
                src_data, dest_data, is_forward);
    }
  } else {
    // We are at the last dimensions, which is stored continously in memory
    for (int i = 0; i < top[0]->shape(lwr_dim); ++i) {
      // prepare index vector reduced(red) and with offsets(off)
      std::vector<int> ind_red(lwr_dim, 0);
      std::vector<int> ind_off(lwr_dim+1, 0);
      for (int j = 0; j < lwr_dim; ++j) {
          ind_red[j] = indices[j];
          ind_off[j] = indices[j] + offsets[j];
      }
      ind_off[lwr_dim] = offsets[lwr_dim];
      // do the copy
      if (is_forward) {
        caffe_copy(top[0]->shape(lwr_dim),
            src_data + bottom[0]->offset(ind_off),
            dest_data + top[0]->offset(ind_red));
      } else {
        // in the backwards pass the src_data is top_diff
        // and the dest_data is bottom_diff
        caffe_copy(top[0]->shape(lwr_dim),
            src_data + top[0]->offset(ind_red),
            dest_data + bottom[0]->offset(ind_off));
      }
    }
  }
}

template <typename Ftype, typename Btype>
void CropLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  std::vector<int> indices(top[0]->num_axes(), 0);
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();
  crop_copy(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
}

template <typename Ftype, typename Btype>
void CropLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();

  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Btype>(0), bottom_diff);
    std::vector<int> indices(top[0]->num_axes(), 0);
    crop_copy(bottom, top, offsets, indices, 0, top_diff, bottom_diff, false);
  }
}

INSTANTIATE_CLASS_FB(CropLayer);
REGISTER_LAYER_CLASS(Crop);

}  // namespace caffe
