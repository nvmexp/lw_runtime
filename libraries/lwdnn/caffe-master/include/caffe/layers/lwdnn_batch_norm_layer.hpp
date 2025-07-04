#ifndef CAFFE_LWDNN_BATCH_NORM_LAYER_HPP_
#define CAFFE_LWDNN_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/type.hpp"


namespace caffe {

#ifdef USE_LWDNN
template <typename Ftype, typename Btype>
class LwDNNBatchNormLayer : public BatchNormLayer<Ftype, Btype> {
 public:
  explicit LwDNNBatchNormLayer(const LayerParameter& param)
      : BatchNormLayer<Ftype, Btype>(param), handles_setup_(false),
        save_mean_(Blob::create(tp<Ftype>(), tp<Ftype>())),
        save_ilw_var_(Blob::create(tp<Ftype>(), tp<Ftype>())) {
  }
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual ~LwDNNBatchNormLayer();

  bool skip_apply_update(int blob_id) const override {
    return blob_id < 3;
  }

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  Type blobs_type() const override {
    return tpmax<Ftype, float>();
  }

  // lwDNN descriptors / handles
  lwdnnTensorDescriptor_t fwd_bottom_desc_, fwd_top_desc_;
  lwdnnTensorDescriptor_t bwd_bottom_desc_, bwd_top_desc_;
  lwdnnTensorDescriptor_t fwd_scale_bias_mean_var_desc_;
  lwdnnTensorDescriptor_t bwd_scale_bias_mean_var_desc_;
  lwdnnBatchNormMode_t mode_;

  bool handles_setup_;

  shared_ptr<Blob> save_mean_;
  shared_ptr<Blob> save_ilw_var_;

  shared_ptr<Blob> scale_ones_;
  shared_ptr<Blob> bias_zeros_;

  shared_ptr<Blob> private_top_;
  shared_ptr<Blob> private_bottom_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_LWDNN_BATCH_NORM_LAYER_HPP_
