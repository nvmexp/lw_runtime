#ifdef USE_LWDNN

#include "caffe/layers/lwdnn_dropout_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void LwDNNDropoutLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  DropoutLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  // initialize lwDNN
  lwdnn::createTensor4dDesc<Ftype>(&bottom_desc_);
  lwdnn::createTensor4dDesc<Ftype>(&top_desc_);

  // initialize dropout state
  LWDNN_CHECK(lwdnnCreateDropoutDescriptor(&dropout_desc_));
  LWDNN_CHECK(lwdnnDropoutGetStatesSize(Caffe::lwdnn_handle(0), &state_size_));
  states_.reserve(state_size_, Caffe::device());

  // setup dropout descriptor
  LWDNN_CHECK(lwdnnSetDropoutDescriptor(dropout_desc_,
                                        Caffe::lwdnn_handle(0),
                                        this->threshold_,
                                        states_.data(),
                                        state_size_,
                                        seed_));
  LWDA_CHECK(lwdaStreamSynchronize(Caffe::thread_stream(0)));

  handles_setup_ = true;
}

template <typename Ftype, typename Btype>
void LwDNNDropoutLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  DropoutLayer<Ftype, Btype>::Reshape(bottom, top);
  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();
  lwdnn::setTensor4dDesc<Ftype>(&bottom_desc_, N, K, H, W);
  lwdnn::setTensor4dDesc<Ftype>(&top_desc_, N, K, H, W);

  LWDNN_CHECK(lwdnnDropoutGetReserveSpaceSize(bottom_desc_, &reserve_space_size_));
  reserve_space_.reserve(reserve_space_size_, Caffe::device());
}

template <typename Ftype, typename Btype>
LwDNNDropoutLayer<Ftype, Btype>::~LwDNNDropoutLayer() {
  states_.release();
  reserve_space_.release();
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }
  lwdnnDestroyTensorDescriptor(this->bottom_desc_);
  lwdnnDestroyTensorDescriptor(this->top_desc_);

  LWDNN_CHECK(lwdnnDestroyDropoutDescriptor(dropout_desc_));
}

INSTANTIATE_CLASS_FB(LwDNNDropoutLayer);

}  // namespace caffe
#endif
