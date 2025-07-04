#ifndef CAFFE_RELWRRENT_LAYER_HPP_
#define CAFFE_RELWRRENT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

namespace caffe {

/**
 * @brief An abstract class for implementing relwrrent behavior inside of an
 *        unrolled network.  This Layer type cannot be instantiated -- instead,
 *        you should use one of its implementations which defines the relwrrent
 *        architecture, such as RNNLayer or LSTMLayer.
 */
template<typename Ftype, typename Btype>
class RelwrrentLayer : public Layer<Ftype, Btype> {
 public:
  explicit RelwrrentLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reset();

  virtual inline const char* type() const { return "Relwrrent"; }
  virtual inline int MinBottomBlobs() const {
    int min_bottoms = 2;
    if (this->layer_param_.relwrrent_param().expose_hidden()) {
      vector<string> inputs;
      this->RelwrrentInputBlobNames(&inputs);
      min_bottoms += inputs.size();
    }
    return min_bottoms;
  }
  virtual inline int MaxBottomBlobs() const { return MinBottomBlobs() + 1; }
  virtual inline int ExactNumTopBlobs() const {
    int num_tops = 1;
    if (this->layer_param_.relwrrent_param().expose_hidden()) {
      vector<string> outputs;
      this->RelwrrentOutputBlobNames(&outputs);
      num_tops += outputs.size();
    }
    return num_tops;
  }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    // Can't propagate to sequence continuation indicators.
    return bottom_index != 1;
  }

 protected:
  /**
   * @brief Fills net_param with the relwrrent network architecture.  Subclasses
   *        should define this -- see RNNLayer and LSTMLayer for examples.
   */
  virtual void FillUnrolledNet(NetParameter* net_param) const = 0;

  /**
   * @brief Fills names with the names of the 0th timestep relwrrent input
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  virtual void RelwrrentInputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief Fills shapes with the shapes of the relwrrent input Blob&s.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  virtual void RelwrrentInputShapes(vector<BlobShape>* shapes) const = 0;

  /**
   * @brief Fills names with the names of the Tth timestep relwrrent output
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  virtual void RelwrrentOutputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief Fills names with the names of the output blobs, concatenated across
   *        all timesteps.  Should return a name for each top Blob.
   *        Subclasses should define this -- see RNNLayer and LSTMLayer for
   *        examples.
   */
  virtual void OutputBlobNames(vector<string>* names) const = 0;

  /**
   * @param bottom input Blob vector (length 2-3)
   *
   *   -# @f$ (T \times N \times ...) @f$
   *      the time-varying input @f$ x @f$.  After the first two axes, whose
   *      dimensions must correspond to the number of timesteps @f$ T @f$ and
   *      the number of independent streams @f$ N @f$, respectively, its
   *      dimensions may be arbitrary.  Note that the ordering of dimensions --
   *      @f$ (T \times N \times ...) @f$, rather than
   *      @f$ (N \times T \times ...) @f$ -- means that the @f$ N @f$
   *      independent input streams must be "interleaved".
   *
   *   -# @f$ (T \times N) @f$
   *      the sequence continuation indicators @f$ \delta @f$.
   *      These inputs should be binary (0 or 1) indicators, where
   *      @f$ \delta_{t,n} = 0 @f$ means that timestep @f$ t @f$ of stream
   *      @f$ n @f$ is the beginning of a new sequence, and hence the previous
   *      hidden state @f$ h_{t-1} @f$ is multiplied by @f$ \delta_t = 0 @f$
   *      and has no effect on the cell's output at timestep @f$ t @f$, and
   *      a value of @f$ \delta_{t,n} = 1 @f$ means that timestep @f$ t @f$ of
   *      stream @f$ n @f$ is a continuation from the previous timestep
   *      @f$ t-1 @f$, and the previous hidden state @f$ h_{t-1} @f$ affects the
   *      updated hidden state and output.
   *
   *   -# @f$ (N \times ...) @f$ (optional)
   *      the static (non-time-varying) input @f$ x_{static} @f$.
   *      After the first axis, whose dimension must be the number of
   *      independent streams, its dimensions may be arbitrary.
   *      This is mathematically equivalent to using a time-varying input of
   *      @f$ x'_t = [x_t; x_{static}] @f$ -- i.e., tiling the static input
   *      across the @f$ T @f$ timesteps and concatenating with the time-varying
   *      input.  Note that if this input is used, all timesteps in a single
   *      batch within a particular one of the @f$ N @f$ streams must share the
   *      same static input, even if the sequence continuation indicators
   *      suggest that difference sequences are ending and beginning within a
   *      single batch.  This may require padding and/or truncation for uniform
   *      length.
   *
   * @param top output Blob vector (length 1)
   *   -# @f$ (T \times N \times D) @f$
   *      the time-varying output @f$ y @f$, where @f$ D @f$ is
   *      <code>relwrrent_param.num_output()</code>.
   *      Refer to documentation for particular RelwrrentLayer implementations
   *      (such as RNNLayer and LSTMLayer) for the definition of @f$ y @f$.
   */
  virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  /// @brief A Net to implement the Relwrrent functionality.
  shared_ptr<Net> unrolled_net_;

  /// @brief The number of independent streams to process simultaneously.
  int N_;

  /**
   * @brief The number of timesteps in the layer's input, and the number of
   *        timesteps over which to backpropagate through time.
   */
  int T_;

  /// @brief Whether the layer has a "static" input copied across all timesteps.
  bool static_input_;

  /**
   * @brief The last layer to run in the network. (Any later layers are losses
   *        added to force the relwrrent net to do backprop.)
   */
  int last_layer_index_;

  /**
   * @brief Whether the layer's hidden state at the first and last timesteps
   *        are layer inputs and outputs, respectively.
   */
  bool expose_hidden_;

  vector<Blob*> relwr_input_blobs_;
  vector<Blob*> relwr_output_blobs_;
  vector<Blob*> output_blobs_;
  Blob* x_input_blob_;
  Blob* x_static_input_blob_;
  Blob* cont_input_blob_;
};

}  // namespace caffe

#endif  // CAFFE_RELWRRENT_LAYER_HPP_
