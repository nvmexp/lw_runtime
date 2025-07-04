#ifndef CAFFE_IM2COL_LAYER_HPP_
#define CAFFE_IM2COL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief A helper for image operations that rearranges image regions into
 *        column vectors.  Used by ColwolutionLayer to perform colwolution
 *        by matrix multiplication.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class Im2colLayer : public Layer<Ftype, Btype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Im2col"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  /// @brief The spatial dimensions of a filter kernel.
  TBlob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  TBlob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  TBlob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  TBlob<int> dilation_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;

  bool force_nd_im2col_;
};

}  // namespace caffe

#endif  // CAFFE_IM2COL_LAYER_HPP_
