#ifdef USE_LWDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/lwdnn_decolw_layer.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for callwlating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define LWDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain lwDNN interface
 */
template<typename Ftype, typename Btype>
void LwDNNDecolwolutionLayer<Ftype, Btype>::LayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  DecolwolutionLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  // Initialize LWCA streams and lwDNN.
  stream_         = new lwdaStream_t[this->group_ * LWDNN_STREAMS_PER_GROUP];
  handle_         = new lwdnnHandle_t[this->group_ * LWDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new lwdnnColwolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new lwdnnColwolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new lwdnnColwolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * LWDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (lwdnnColwolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (lwdnnColwolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (lwdnnColwolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * LWDNN_STREAMS_PER_GROUP; g++) {
    LWDA_CHECK(lwdaStreamCreate(&stream_[g]));
    LWDNN_CHECK(lwdnnCreate(&handle_[g]));
    LWDNN_CHECK(lwdnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  lwdnn::createFilterDesc<Btype>(&filter_desc_,    ///////// FIXME!!!!!!!!!!!!!
                                 this->channels_ / this->group_,
                                 this->num_output_ / this->group_,
                                 kernel_h,
                                 kernel_w);

  // Create tensor descriptor(s) for data and corresponding colwolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    lwdnnTensorDescriptor_t bottom_desc;
    lwdnn::createTensor4dDesc<Btype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    lwdnnTensorDescriptor_t top_desc;
    lwdnn::createTensor4dDesc<Btype>(&top_desc);
    top_descs_.push_back(top_desc);
    lwdnnColwolutionDescriptor_t colw_desc;
    lwdnnCreateColwolutionDescriptor(&colw_desc);
    colw_descs_.push_back(colw_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    lwdnn::createTensor4dDesc<Btype>(&bias_desc_);
  }

  handles_setup_ = true;
}

template<typename Ftype, typename Btype>
void LwDNNDecolwolutionLayer<Ftype, Btype>::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  DecolwolutionLayer<Ftype, Btype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "LwDNNDecolwolutionLayer input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND colwolution.";
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {
    lwdnn::setTensor4dDesc<Btype>(&bottom_descs_[i],
                                  this->num_,
                                  this->channels_ / this->group_,
                                  height,
                                  width,
                                  this->channels_ * height * width,
                                  height * width,
                                  width,
                                  1);
    lwdnn::setTensor4dDesc<Btype>(&top_descs_[i],
                                  this->num_,
                                  this->num_output_ / this->group_,
                                  height_out,
                                  width_out,
                                  this->num_output_ * height_out * width_out,
                                  height_out * width_out,
                                  width_out,
                                  1);
    lwdnn::setColwolutionDesc(forward_math_,
                              colw_descs_[i],
//                              top_descs_[i],
//                              filter_desc_,
                              pad_h,
                              pad_w,
                              stride_h,
                              stride_w, 1, 1);

    // choose forward and backward algorithms + workspace(s)
    LWDNN_CHECK(lwdnnGetColwolutionForwardAlgorithm(
        handle_[0],
        top_descs_[i],
        filter_desc_,
        colw_descs_[i],
        bottom_descs_[i],
        LWDNN_COLWOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,
        &fwd_algo_[i]));

    // We have found that LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM is
    // buggy. Thus, if this algo was chosen, choose winograd instead. If
    // winograd is not supported or workspace is larger than threshold, choose
    // implicit_gemm instead.
//    if (fwd_algo_[i] == LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
//      size_t winograd_workspace_size;
//      lwdnnStatus_t status = lwdnnGetColwolutionForwardWorkspaceSize(
//          handle_[0],
//          top_descs_[i],
//          filter_desc_,
//          colw_descs_[i],
//          bottom_descs_[i],
//          LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD,
//          &winograd_workspace_size);
//      if (status != LWDNN_STATUS_SUCCESS ||
//          winograd_workspace_size >= workspace_limit_bytes) {
//        fwd_algo_[i] = LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM;
//      } else {
//        fwd_algo_[i] = LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD;
//      }
//    }

    LWDNN_CHECK(lwdnnGetColwolutionForwardWorkspaceSize(
        handle_[0],
        top_descs_[i],
        filter_desc_,
        colw_descs_[i],
        bottom_descs_[i],
        fwd_algo_[i],
        &(workspace_fwd_sizes_[i])));

    // choose backward algorithm for filter
    LWDNN_CHECK(lwdnnGetColwolutionBackwardFilterAlgorithm(
        handle_[0],
        top_descs_[i],
        bottom_descs_[i],
        colw_descs_[i],
        filter_desc_,
        LWDNN_COLWOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,
        &bwd_filter_algo_[i]));

    // get workspace for backwards filter algorithm
    LWDNN_CHECK(lwdnnGetColwolutionBackwardFilterWorkspaceSize(
        handle_[0],
        top_descs_[i],
        bottom_descs_[i],
        colw_descs_[i],
        filter_desc_,
        bwd_filter_algo_[i],
        &workspace_bwd_filter_sizes_[i]));

    // choose backward algo for data
    LWDNN_CHECK(lwdnnGetColwolutionBackwardDataAlgorithm(
        handle_[0],
        filter_desc_,
        bottom_descs_[i],
        colw_descs_[i],
        top_descs_[i],
        LWDNN_COLWOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,
        &bwd_data_algo_[i]));

    // get workspace size
    LWDNN_CHECK(lwdnnGetColwolutionBackwardDataWorkspaceSize(
        handle_[0],
        filter_desc_,
        bottom_descs_[i],
        colw_descs_[i],
        top_descs_[i],
        bwd_data_algo_[i],
        &workspace_bwd_data_sizes_[i]));
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->group_ * LWDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    lwdaFree(this->workspaceData);

    lwdaError_t err = lwdaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != lwdaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = LWDNN_COLWOLUTION_FWD_ALGO_FFT_TILING;
        bwd_filter_algo_[i] = LWDNN_COLWOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_algo_[i] = LWDNN_COLWOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * LWDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * LWDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    lwdnn::setTensor4dDesc<Btype>(
        &bias_desc_, 1, this->num_output_ / this->group_, 1, 1);
  }
}

template<typename Ftype, typename Btype>
LwDNNDecolwolutionLayer<Ftype, Btype>::~LwDNNDecolwolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    lwdnnDestroyTensorDescriptor(bottom_descs_[i]);
    lwdnnDestroyTensorDescriptor(top_descs_[i]);
    lwdnnDestroyColwolutionDescriptor(colw_descs_[i]);
  }
  if (this->bias_term_) {
    lwdnnDestroyTensorDescriptor(bias_desc_);
  }
  lwdnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * LWDNN_STREAMS_PER_GROUP; g++) {
    lwdaStreamDestroy(stream_[g]);
    lwdnnDestroy(handle_[g]);
  }

  lwdaFree(workspaceData);
  delete [] workspace;
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS_FB(LwDNNDecolwolutionLayer);

}   // namespace caffe
#endif
