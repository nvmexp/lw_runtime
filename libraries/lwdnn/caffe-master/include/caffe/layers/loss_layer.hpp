#ifndef CAFFE_LOSS_LAYER_HPP_
#define CAFFE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */
template <typename Ftype, typename Btype>
class LossLayer : public Layer<Ftype, Btype> {
  typedef Ftype Dtype;

 public:
  explicit LossLayer(const LayerParameter& param)
     : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(
      const vector<Blob*>& bottom, const vector<Blob*>& top);

  /**
   * Read the normalization mode parameter and compute the normalizer based
   * on the blob size. If normalization_mode is VALID, the count of valid
   * outputs will be read from valid_count, unless it is -1 in which case
   * all outputs are assumed to be valid.
   */
  Ftype GetNormalizer(
      const LossParameter_NormalizationMode normalization_mode,
      const int outer_num, const int inner_num, const int valid_count);

  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * @brief For colwenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
  virtual inline bool AutoTopBlobs() const { return true; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYER_HPP_
