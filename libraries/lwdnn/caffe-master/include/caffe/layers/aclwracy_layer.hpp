#ifndef CAFFE_ACLWRACY_LAYER_HPP_
#define CAFFE_ACLWRACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Ftype, typename Btype>
class AclwracyLayer : public Layer<Ftype, Btype> {
 public:
  /**
   * @param param provides AclwracyParameter aclwracy_param,
   *     with AclwracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit AclwracyLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  inline const char* type() const override { return "Accuracy"; }
  inline int ExactNumBottomBlobs() const override { return 2; }

  // If there are two top blobs, then the second blob will contain
  // aclwracies per class.
  inline int MinTopBlobs() const override { return 1; }
  inline int MaxTopBlobs() const override { return 2; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
   *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
   *      label @f$ \hat{l}_n @f$ given by its maximal index:
   *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels @f$ l @f$, an integer-valued Blob with values
   *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
   *      indicating the correct class label among the @f$ K @f$ classes
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the computed accuracy: @f$
   *        \frac{1}{N} \sum\limits_{n=1}^N \delta\{ \hat{l}_n = l_n \}
   *      @f$, where @f$
   *      \delta\{\mathrm{condition}\} = \left\{
   *         \begin{array}{lr}
   *            1 & \mbox{if condition} \\
   *            0 & \mbox{otherwise}
   *         \end{array} \right.
   *      @f$
   */
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  /// @brief Not implemented -- AclwracyLayer cannot be used as a loss.
  void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) override {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) override;

  int label_axis_, outer_num_, inner_num_;
  int top_k_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Keeps counts of the number of samples per class.
  TBlob<float> nums_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_ACLWRACY_LAYER_HPP_
