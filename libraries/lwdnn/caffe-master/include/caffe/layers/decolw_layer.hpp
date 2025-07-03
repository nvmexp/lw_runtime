#ifndef CAFFE_DECOLW_LAYER_HPP_
#define CAFFE_DECOLW_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_colw_layer.hpp"

namespace caffe {

/**
 * @brief Colwolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and colwolution parameters in the
 *        opposite sense as ColwolutionLayer.
 *
 *   ColwolutionLayer computes each output value by dotting an input window with
 *   a filter; DecolwolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   DecolwolutionLayer is ColwolutionLayer with the forward and backward passes
 *   reversed. DecolwolutionLayer reuses ColwolutionParameter for its
 *   parameters, but they take the opposite sense as in ColwolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
template <typename Ftype, typename Btype>
class DecolwolutionLayer : public BaseColwolutionLayer<Ftype, Btype> {
 public:
  explicit DecolwolutionLayer(const LayerParameter& param)
      : BaseColwolutionLayer<Ftype, Btype>(param) {}

  virtual inline const char* type() const { return "Decolwolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual inline bool reverse_dimensions() { return true; }
  virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_DECOLW_LAYER_HPP_
