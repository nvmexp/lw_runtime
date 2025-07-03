#ifndef CAFFE_LWDNN_DECOLW_LAYER_HPP_
#define CAFFE_LWDNN_DECOLW_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/decolw_layer.hpp"

namespace caffe {

#ifdef USE_LWDNN
/*
 * @brief lwDNN implementation of DeColwolutionLayer.
 *        Fallback to DeColwolutionLayer for CPU mode.
 *
 * lwDNN accelerates decolwolution through forward kernels for filtering and
 * bias plus backward kernels for the gradient w.r.t. the filters, biases, and
 * inputs. Caffe + lwDNN further speeds up the computation through forward
 * parallelism across groups and backward parallelism across gradients.
*/
template<typename Ftype, typename Btype>
class LwDNNDecolwolutionLayer : public DecolwolutionLayer<Ftype, Btype> {
 public:
  explicit LwDNNDecolwolutionLayer(const LayerParameter& param)
    : DecolwolutionLayer<Ftype, Btype>(param),
      handles_setup_(false),
      forward_math_(tpmax<Ftype, float>()),
      backward_data_math_(tpmax<Btype, float>()),
      backward_filter_math_(tpmax<Btype, float>()) {}
  virtual ~LwDNNDecolwolutionLayer();
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

 protected:
  void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
                    const vector<Blob*>& bottom) override;

  bool handles_setup_;
  lwdnnHandle_t* handle_;
  lwdaStream_t*  stream_;

  // algorithms for forward and backwards colwolutions
  lwdnnColwolutionFwdAlgo_t *fwd_algo_;
  lwdnnColwolutionBwdFilterAlgo_t *bwd_filter_algo_;
  lwdnnColwolutionBwdDataAlgo_t *bwd_data_algo_;

  vector<lwdnnTensorDescriptor_t> bottom_descs_, top_descs_;
  lwdnnTensorDescriptor_t bias_desc_;
  lwdnnFilterDescriptor_t filter_desc_;
  vector<lwdnnColwolutionDescriptor_t> colw_descs_;
  int bottom_offset_, top_offset_, bias_offset_;
  Type forward_math_, backward_data_math_, backward_filter_math_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t *workspace_bwd_filter_sizes_;
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData
};
#endif

}  // namespace caffe

#endif  // CAFFE_LWDNN_DECOLW_LAYER_HPP_
