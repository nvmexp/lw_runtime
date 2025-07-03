#ifdef USE_LWDNN
#include <algorithm>
#include <vector>
#include <boost/tokenizer.hpp>
#include <lwdnn.h>

#include "caffe/parallel.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lwdnn_colw_layer.hpp"
#include "caffe/solver.hpp"

namespace caffe {

#if !LWDNN_VERSION_MIN(6, 0, 0)
#define LWDNN_COLWOLUTION_FWD_ALGO_COUNT \
    (LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD_NONFUSED + 1)
#define LWDNN_COLWOLUTION_BWD_DATA_ALGO_COUNT \
    (LWDNN_COLWOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED + 1)
#define LWDNN_COLWOLUTION_BWD_FILTER_ALGO_COUNT \
    (LWDNN_COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED + 1)
#endif

static std::mutex colw_mutex_;
static constexpr size_t MEM_REQ_READ = static_cast<size_t>(-1L);
static constexpr size_t MEM_REQ_DEFAULT = 4UL*1024UL*1024UL;  // to prevent NULL passed to lwdnn.
static constexpr size_t MEM_REQ_TMP_WEIGHTS = 2UL;

size_t mem_req(int use_case, size_t maxval = MEM_REQ_READ) {
  std::lock_guard<std::mutex> lock(colw_mutex_);
  static thread_local size_t m[MEM_REQ_TMP_WEIGHTS + 1];
  if (m[use_case] == 0UL) {
    m[use_case] = MEM_REQ_DEFAULT;
  } else if (maxval > 0UL && maxval != MEM_REQ_READ) {
    m[use_case] = std::max(maxval, m[use_case]);
  }
  CHECK_GT(m[use_case], 0UL);
  return m[use_case];
}

void setColwolutionDescMath(Type math, lwdnnColwolutionDescriptor_t colw) {
  int padA[2];
  int strideA[2];
  int upscaleA[2];
  int arrayLengthRequested = 2;
  int arrayLength;
  lwdnnColwolutionMode_t mode;
  lwdnnDataType_t dataType;

  LWDNN_CHECK(lwdnnGetColwolutionNdDescriptor(colw,
      arrayLengthRequested, &arrayLength, padA, strideA, upscaleA,
      &mode, &dataType));
  LWDNN_CHECK(lwdnnSetColwolutionNdDescriptor(colw,
      2, padA, strideA, upscaleA, mode,
      lwdnn::lwdnn_data_type(math)));
}

lwdnnDataType_t colwolutionDescDataType(lwdnnColwolutionDescriptor_t colw) {
  int padA[2];
  int strideA[2];
  int upscaleA[2];
  int arrayLengthRequested = 2;
  int arrayLength;
  lwdnnColwolutionMode_t mode;
  lwdnnDataType_t dataType;

  LWDNN_CHECK(lwdnnGetColwolutionNdDescriptor(colw,
      arrayLengthRequested, &arrayLength, padA, strideA, upscaleA,
      &mode, &dataType));
  return dataType;
}

/**
 * TODO(dox) explain lwDNN interface
 */
template <typename Ftype, typename Btype>
void LwDNNColwolutionLayer<Ftype, Btype>::LayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  ColwolutionLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  // Initialize algorithm arrays
  fwd_algo_.resize(bottom.size());
  bwd_filter_algo_.resize(bottom.size());
  bwd_data_algo_.resize(bottom.size());

#if LWDNN_VERSION_MIN(7, 0, 0)
  lwdnn_math_override_ = this->layer_param().lwdnn_math_override();
  fwd_lwdnn_math_.resize(bottom.size());
  bwd_filter_lwdnn_math_.resize(bottom.size());
  bwd_data_lwdnn_math_.resize(bottom.size());
#endif
  // initialize size arrays
  workspace_fwd_sizes_.resize(bottom.size());
  workspace_bwd_filter_sizes_.resize(bottom.size());
  workspace_bwd_data_sizes_.resize(bottom.size());

  std::string colw_algos_override = this->layer_param().colwolution_param().colw_algos_override();
  boost::char_separator<char> sep(", ");
  boost::tokenizer<boost::char_separator<char>> tokens(colw_algos_override, sep);
  for (const auto& t : tokens) {
    user_algos_override_.push_back(boost::lexical_cast<int>(t));
  }
  std::string param_err = "colw_algos_override parameter vaue '" +
      colw_algos_override + "' is ill formatted";
  CHECK_EQ(3, user_algos_override_.size()) << param_err;
  if (user_algos_override_[0] >= 0) {
    CHECK_LT(user_algos_override_[0], LWDNN_COLWOLUTION_FWD_ALGO_COUNT) << param_err;
  }
  if (user_algos_override_[1] >= 0) {
    CHECK_LT(user_algos_override_[1], LWDNN_COLWOLUTION_BWD_DATA_ALGO_COUNT) << param_err;
  }
  if (user_algos_override_[2] >= 0) {
    CHECK_LT(user_algos_override_[2], LWDNN_COLWOLUTION_BWD_FILTER_ALGO_COUNT) << param_err;
  }

  const int* dilation_data = this->dilation_.cpu_data();
  const bool use_dilation = dilation_data[0] > 1 || dilation_data[1] > 1;

  // Initializing algorithms and workspaces
  // Do not rely on initialized algorithms (Reshape will set algorithms
  // with correct values in the first iteration).
  for (size_t i = 0; i < bottom.size(); ++i) {
    fwd_algo_[i] = (lwdnnColwolutionFwdAlgo_t)
        (user_algos_override_[0] >= 0 ? user_algos_override_[0] :
         (use_dilation ? LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM :
          LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM));
    bwd_data_algo_[i] = (lwdnnColwolutionBwdDataAlgo_t)
        (user_algos_override_[1] >= 0 ? user_algos_override_[1] :
         LWDNN_COLWOLUTION_BWD_DATA_ALGO_1);
    bwd_filter_algo_[i] = (lwdnnColwolutionBwdFilterAlgo_t)
        (user_algos_override_[2] >= 0 ? user_algos_override_[2] :
         LWDNN_COLWOLUTION_BWD_FILTER_ALGO_1);

    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;

#if LWDNN_VERSION_MIN(7, 0, 0)
    fwd_lwdnn_math_[i] = LWDNN_DEFAULT_MATH;
    bwd_filter_lwdnn_math_[i] = LWDNN_DEFAULT_MATH;
    bwd_data_lwdnn_math_[i] = LWDNN_DEFAULT_MATH;
#endif
  }
  forward_math_ = this->layer_param().forward_math();
  backward_data_math_ = backward_filter_math_ = this->layer_param().backward_math();

  // Set the indexing parameters.
  bias_offset_ = this->num_output_ / groups();

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];

  if (use_v7grouping()) {
    lwdnn::createFilterDesc<Ftype>(&fwd_filter_desc_,
        this->num_output_, this->channels_ / groups(),
        kernel_h, kernel_w);
    lwdnn::createFilterDesc<Btype>(&bwd_filter_desc_,
        this->num_output_, this->channels_ / groups(),
        kernel_h, kernel_w);
    this->weight_offset_ = this->num_output_ *
                           (this->channels_ / groups()) * kernel_h * kernel_w;
  } else {
    lwdnn::createFilterDesc<Ftype>(&fwd_filter_desc_,
        this->num_output_ / groups(), this->channels_ / groups(),
        kernel_h, kernel_w);
    lwdnn::createFilterDesc<Btype>(&bwd_filter_desc_,
        this->num_output_ / groups(), this->channels_ / groups(),
        kernel_h, kernel_w);
    this->weight_offset_ = (this->num_output_ / groups()) *
                           (this->channels_ / groups()) * kernel_h * kernel_w;
  }

  // Create tensor descriptor(s) for data and corresponding colwolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    lwdnnTensorDescriptor_t fwd_bottom_desc, bwd_bottom_desc;
    lwdnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc);
    lwdnn::createTensor4dDesc<Btype>(&bwd_bottom_desc);
    fwd_bottom_descs_.push_back(fwd_bottom_desc);
    bwd_bottom_descs_.push_back(bwd_bottom_desc);
    lwdnnTensorDescriptor_t fwd_top_desc, bwd_top_desc;
    lwdnn::createTensor4dDesc<Ftype>(&fwd_top_desc);
    lwdnn::createTensor4dDesc<Btype>(&bwd_top_desc);
    fwd_top_descs_.push_back(fwd_top_desc);
    bwd_top_descs_.push_back(bwd_top_desc);
    lwdnnColwolutionDescriptor_t fwd_colw_desc, bwd_colw_data_desc, bwd_colw_filter_desc;
    LWDNN_CHECK(lwdnnCreateColwolutionDescriptor(&fwd_colw_desc));
    LWDNN_CHECK(lwdnnCreateColwolutionDescriptor(&bwd_colw_data_desc));
    LWDNN_CHECK(lwdnnCreateColwolutionDescriptor(&bwd_colw_filter_desc));
#ifdef LWDNN_GROUPING
    if (use_v7grouping()) {
      LWDNN_CHECK(lwdnnSetColwolutionGroupCount(fwd_colw_desc, groups()));
      LWDNN_CHECK(lwdnnSetColwolutionGroupCount(bwd_colw_data_desc, groups()));
      LWDNN_CHECK(lwdnnSetColwolutionGroupCount(bwd_colw_filter_desc, groups()));
    }
#endif

    fwd_colw_descs_.push_back(fwd_colw_desc);
    bwd_colw_data_descs_.push_back(bwd_colw_data_desc);
    bwd_colw_filter_descs_.push_back(bwd_colw_filter_desc);

    lwdnnTensorDescriptor_t fwd_cached_bottom_desc;
    lwdnn::createTensor4dDesc<Ftype>(&fwd_cached_bottom_desc);
    fwd_cached_bottom_descs_.push_back(fwd_cached_bottom_desc);
    lwdnnTensorDescriptor_t bwd_cached_bottom_desc;
    lwdnn::createTensor4dDesc<Btype>(&bwd_cached_bottom_desc);
    bwd_cached_bottom_descs_.push_back(bwd_cached_bottom_desc);

    lwdnnColwolutionDescriptor_t fwd_cached_colw_desc;
    lwdnnColwolutionDescriptor_t bwd_cached_colw_data_desc, bwd_cached_colw_filter_desc;
    LWDNN_CHECK(lwdnnCreateColwolutionDescriptor(&fwd_cached_colw_desc));
    LWDNN_CHECK(lwdnnCreateColwolutionDescriptor(&bwd_cached_colw_data_desc));
    LWDNN_CHECK(lwdnnCreateColwolutionDescriptor(&bwd_cached_colw_filter_desc));
#ifdef LWDNN_GROUPING
    if (use_v7grouping()) {
      LWDNN_CHECK(lwdnnSetColwolutionGroupCount(fwd_cached_colw_desc, groups()));
      LWDNN_CHECK(lwdnnSetColwolutionGroupCount(bwd_cached_colw_data_desc, groups()));
      LWDNN_CHECK(lwdnnSetColwolutionGroupCount(bwd_cached_colw_filter_desc, groups()));
    }
#endif
    fwd_cached_colw_descs_.push_back(fwd_cached_colw_desc);
    bwd_cached_colw_data_descs_.push_back(bwd_cached_colw_data_desc);
    bwd_cached_colw_filter_descs_.push_back(bwd_cached_colw_filter_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    lwdnn::createTensor4dDesc<Ftype>(&fwd_bias_desc_);
    lwdnn::createTensor4dDesc<Btype>(&bwd_bias_desc_);
  }

  handles_setup_ = true;
  // When true, Reshape asks lwDNN (either Get ot FindEx) for the best algorithm
  use_algo_seeker_ = true;
  // When true, Reshape sets descriptors, algorithms, workspaces.
  use_reshape_ = true;
  // When true, cached bottom and colw descriptors need to be set.
  initialized_cached_descs_ = false;
  // Release workspace after FindEx is done, i.e. when bwd_count_ == 2
  fwd_count_ = 0UL;
  bwd_count_ = 0UL;
}

template <typename Ftype, typename Btype>
size_t LwDNNColwolutionLayer<Ftype, Btype>::AllocateFindExWorkspace() {
  size_t bytes_available, bytes_total;
  GPUMemory::GetInfo(&bytes_available, &bytes_total, true);
  bytes_available = std::min(bytes_available + Caffe::ws(CAFFE_WS_COLW).size(), bytes_total / 2UL);

  const size_t tmp_weights_size = mem_req(MEM_REQ_TMP_WEIGHTS);
  if (bytes_available > tmp_weights_size) {
    bytes_available -= tmp_weights_size;
  } else {
    bytes_available = 0UL;
  }
  // 2+ pages => reallocate
  size_t req_bytes = align_down<8>(bytes_available > 2UL * PAGE_SIZE ?
      bytes_available - 2UL * PAGE_SIZE : 0UL);
  if (static_cast<float>(req_bytes) > PAGE_SIZE) {
    DLOG(INFO) << this->print_lwrrent_device()
        << " AllocateFindExWorkspace trying to allocate " << req_bytes << " bytes";
    int attempts = ATTEMPTS_TO_RESERVE_WS;
    while (!Caffe::ws(CAFFE_WS_COLW).try_reserve(req_bytes, Caffe::device()) && attempts > 0) {
      req_bytes = align_down<8>(req_bytes > PAGE_SIZE ? req_bytes - PAGE_SIZE : 0UL);
      --attempts;
      LOG(INFO) << this->print_lwrrent_device() << " Retrying to allocate " << req_bytes
                << " bytes, attempts left: " << attempts;
    }
  }
  return Caffe::ws(CAFFE_WS_COLW).size();
}

template <typename Ftype, typename Btype>
size_t LwDNNColwolutionLayer<Ftype, Btype>::AllocateWorkspace(size_t bottom_size) {
  const int dev = Caffe::device();
  lwdnnHandle_t handle = Caffe::lwdnn_handle(0);
  for (int i = 0; i < bottom_size; ++i) {
    if (this->phase_ == TRAIN) {
      LWDNN_CHECK(lwdnnGetColwolutionBackwardDataWorkspaceSize(handle,
          bwd_filter_desc_, bwd_top_descs_[i], bwd_colw_data_descs_[i], bwd_bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]));
      LWDNN_CHECK(lwdnnGetColwolutionBackwardFilterWorkspaceSize(handle,
          bwd_bottom_descs_[i], bwd_top_descs_[i], bwd_colw_filter_descs_[i], bwd_filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));
    }
    LWDNN_CHECK(lwdnnGetColwolutionForwardWorkspaceSize(handle,
        fwd_bottom_descs_[i], fwd_filter_desc_, fwd_colw_descs_[i], fwd_top_descs_[i],
        fwd_algo_[i], &(workspace_fwd_sizes_[i])));
    LWDA_CHECK(lwdaStreamSynchronize(Caffe::thread_stream(0)));
  }

  for (int i = 0; i < bottom_size; ++i) {
    if (this->phase_ == TRAIN) {
      mem_req(this->phase_, align_up<8>(workspace_bwd_data_sizes_[i]) * ws_groups());
      mem_req(this->phase_, align_up<8>(workspace_bwd_filter_sizes_[i]) * ws_groups());
    }
    mem_req(this->phase_, align_up<8>(std::max<size_t>(1UL,
        (workspace_fwd_sizes_[i]) * ws_groups())));
  }
  size_t req_bytes = mem_req(this->phase_);
  DLOG(INFO) << this->print_lwrrent_device()
      << " AllocateWorkspace trying to allocate " << req_bytes << " bytes for layer "
      << this->name();
  Caffe::ws(CAFFE_WS_COLW).safe_reserve(req_bytes, dev);
  return Caffe::ws(CAFFE_WS_COLW).size();
}

template <typename Ftype, typename Btype>
void LwDNNColwolutionLayer<Ftype, Btype>::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  // Check whether cached descriptors have been initialized.
  if (initialized_cached_descs_) {
    // Check whether bottom and colw descriptors have changed,
    // which then requires a new reshape and set algo.
    if (IsBottomDescChanged(bottom, true) ||
        (this->phase_ == TRAIN && IsBottomDescChanged(bottom, false)) ||
        IsColwDescChanged(bottom, true) ||
        (this->phase_ == TRAIN && IsColwDescChanged(bottom, false))) {
      use_reshape_ = true;
    } else {
      // When no reshape is needed, setting algo may be still needed
      // (for example, if we are at iteration 1).
      // If we want to set algos, we have to use reshape in
      // current implementation.
      use_reshape_ = use_algo_seeker_;
    }
  } else {
    // If cached descriptors are not initialized yet, need to
    // do reshape which also initializes cached descriptors.
    use_reshape_ = true;
  }
  if (!use_reshape_) {
    return;
  }

  ColwolutionLayer<Ftype, Btype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "LwDNNColwolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND colwolution.";

  bottom_offset_ = this->bottom_dim_ / groups();
  top_offset_ = this->top_dim_ / groups();

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
  const int* dilation_data = this->dilation_.cpu_data();
  const int dilation_h = dilation_data[0];
  const int dilation_w = dilation_data[1];

  // Set lwDNN tensor and colwolution descriptors
  for (int i = 0; i < bottom.size(); i++) {
    lwdnn::setTensor4dDesc<Ftype>(&fwd_bottom_descs_[i],
        this->num_,
        use_v7grouping() ? this->channels_ : this->channels_ / groups(),
        height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    lwdnn::setTensor4dDesc<Btype>(&bwd_bottom_descs_[i],
        this->num_,
        use_v7grouping() ? this->channels_ : this->channels_ / groups(),
        height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    lwdnn::setTensor4dDesc<Ftype>(&fwd_top_descs_[i],
        this->num_,
        use_v7grouping() ? this->num_output_ : this->num_output_ / groups(),
        height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    lwdnn::setTensor4dDesc<Btype>(&bwd_top_descs_[i],
        this->num_,
        use_v7grouping() ? this->num_output_ : this->num_output_ / groups(),
        height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);

    lwdnn::setColwolutionDesc(forward_math_, fwd_colw_descs_[i],
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    lwdnn::setColwolutionDesc(forward_math_, fwd_cached_colw_descs_[i],
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    lwdnn::setColwolutionDesc(backward_data_math_, bwd_colw_data_descs_[i],
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    lwdnn::setColwolutionDesc(backward_filter_math_, bwd_colw_filter_descs_[i],
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    lwdnn::setColwolutionDesc(backward_data_math_, bwd_cached_colw_data_descs_[i],
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    lwdnn::setColwolutionDesc(backward_filter_math_, bwd_cached_colw_filter_descs_[i],
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

    // Set cached descriptors
    lwdnn::setTensor4dDesc<Ftype>(&fwd_cached_bottom_descs_[i],
        this->num_,
        use_v7grouping() ? this->channels_ : this->channels_ / groups(),
        height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    lwdnn::setTensor4dDesc<Btype>(&bwd_cached_bottom_descs_[i],
        this->num_,
        use_v7grouping() ? this->channels_ : this->channels_ / groups(),
        height, width,
        this->channels_ * height * width,
        height * width, width, 1);
  }
  initialized_cached_descs_ = true;

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    lwdnn::setTensor4dDesc<Ftype>(&fwd_bias_desc_, 1,
        use_v7grouping() ? this->num_output_ : this->num_output_ / groups(),
        1, 1);
    lwdnn::setTensor4dDesc<Btype>(&bwd_bias_desc_, 1,
        use_v7grouping() ? this->num_output_ : this->num_output_ / groups(),
        1, 1);
  }

  // per device
  if (this->phase_ == TRAIN) {
    mem_req(MEM_REQ_TMP_WEIGHTS,
        align_up<8>(this->weight_offset_ * tsize(tpmax<Btype, float>())));
  }

  if (fwd_count_ == 0UL) {
    AllocateWorkspace(bottom.size());
  }
  // Ask lwDNN to find the best algorithm
  // When batch is small and every image is different we don't want to call Find* over and over
  if (use_algo_seeker_) {
    // FindEx: A workspace of size workspace_bytes is allocated for FindEx.
    //         Besides, workspace, a buffer is allocated for the output of
    //         FindEx-backward-filter. The size of buffer is as big as weights.
    // Get: workspace_bytes is only used as a workspace limit by Get.
    //      (no allocation happens before Get or by Get).
    size_t workspace_bytes;
    switch (this->layer_param_.colwolution_param().lwdnn_colwolution_algo_seeker()) {
      case ColwolutionParameter_LwDNNColwolutionAlgorithmSeeker_GET:
        workspace_bytes = AllocateFindExWorkspace();
        GetColwAlgo(bottom, top, workspace_bytes, pad_h, pad_w, stride_h, stride_w);
        AllocateWorkspace(bottom.size());
        break;
      case ColwolutionParameter_LwDNNColwolutionAlgorithmSeeker_FINDEX:
        if (!use_modest_workspace()) {
          if (this->phase_ == TRAIN) {
            // Now taking the rest for running FindEx calls
            // We'll release what's possible in BW pass
            AllocateFindExWorkspace();
            // Also used by Test Net but based on shared space taken by Train:
            FindExColwAlgo(bottom, top);
          }
          use_algo_seeker_ = false;
        }
        break;
      default:
        LOG(FATAL) << "Wrong value for lwdnn_colwolution_algo_seeker";
    }
  }
}

template <typename Ftype, typename Btype>
void LwDNNColwolutionLayer<Ftype, Btype>::GetColwAlgo(const vector<Blob*>& bottom,
    const vector<Blob*>& top, const size_t workspace_bytes, int pad_h, int pad_w,
    int stride_h, int stride_w) {
  for (int i = 0; i < bottom.size(); ++i) {
    // Get backward data algorithm (if not set by user)
    if (user_algos_override_[1] < 0) {
      LWDNN_CHECK(lwdnnGetColwolutionBackwardDataAlgorithm(Caffe::lwdnn_handle(0),
          bwd_filter_desc_, bwd_top_descs_[i], bwd_colw_data_descs_[i], bwd_bottom_descs_[i],
          LWDNN_COLWOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          align_down<8>(workspace_bytes / ws_groups()), &bwd_data_algo_[i]));
    }
    // Get forward algorithm (if not set by user)
    if (user_algos_override_[0] < 0) {
      LWDNN_CHECK(lwdnnGetColwolutionForwardAlgorithm(Caffe::lwdnn_handle(0),
          fwd_bottom_descs_[i], fwd_filter_desc_, fwd_colw_descs_[i], fwd_top_descs_[i],
          LWDNN_COLWOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          align_down<8>(workspace_bytes / ws_groups()), &fwd_algo_[i]));
      LWDA_CHECK(lwdaStreamSynchronize(Caffe::thread_stream(0)));
    }
    // Get backward filter algorithm (if not set by user)
    if (user_algos_override_[2] < 0) {
      LWDNN_CHECK(lwdnnGetColwolutionBackwardFilterAlgorithm(Caffe::lwdnn_handle(0),
          bwd_bottom_descs_[i], bwd_top_descs_[i], bwd_colw_filter_descs_[i], bwd_filter_desc_,
          LWDNN_COLWOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          align_down<8>(workspace_bytes / ws_groups()), &bwd_filter_algo_[i]));
      LWDA_CHECK(lwdaStreamSynchronize(Caffe::thread_stream(0)));
    }
    LOG(INFO) << Phase_Name(this->phase_)
        << " Colw Algos by Get* (F,BD,BF) for layer '" << this->name()
        << "' with space " << workspace_bytes << "/" << ws_groups() <<  " "
        << fwd_algo_[i] << " " << bwd_data_algo_[i] << " " << bwd_filter_algo_[i];
  }
}

template<typename Ftype, typename Btype>
void LwDNNColwolutionLayer<Ftype, Btype>::FindExColwAlgo(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  int fwd_algo_count = 0;
  int filter_algo_count = 0;
  int data_algo_count = 0;
  lwdnnColwolutionFwdAlgoPerf_t fwd_results[REQUEST_ALGO_COUNT];
  lwdnnColwolutionBwdFilterAlgoPerf_t bwd_filter_results[REQUEST_ALGO_COUNT];
  lwdnnColwolutionBwdDataAlgoPerf_t bwd_data_results[REQUEST_ALGO_COUNT];
  bool fwd_pseudo = false;
  bool bwd_filter_pseudo = false;
  bool bwd_data_pseudo = false;
  float ftime = 0.F, bdtime = 0.F, bftime = 0.F;

#if LWDNN_VERSION_MIN(7, 0, 0)
  // does it support TENSOR_OP?
  const bool top_device = Caffe::device_capability(Caffe::device()) >= 700;
  bool try_top = top_device;
  if (lwdnn_math_override_ < 0 && (is_precise<Ftype>() || is_precise<Btype>())) {
    // 32/64 mode, user doesn't override => default math only
    try_top = false;
  }
#endif
  lwdnnHandle_t handle = Caffe::lwdnn_handle(0);
  lwdaStream_t stream = Caffe::thread_stream(0);
  const size_t gsize = Caffe::ws(CAFFE_WS_COLW).size() / ws_groups();
  CHECK(is_even(gsize)) << Caffe::ws(CAFFE_WS_COLW).size() << " / "
      << ws_groups() << " -> " << gsize;

  for (int i = 0; i < bottom.size(); ++i) {
#if LWDNN_VERSION_MIN(7, 0, 0)
    lwdnnMathType_t fwd_lwdnn_math_0 = LWDNN_DEFAULT_MATH;
    if (try_top) {
      fwd_lwdnn_math_[i] = fwd_lwdnn_math_0 =
          lwdnn_math_override_ == 0 ? LWDNN_DEFAULT_MATH : LWDNN_TENSOR_OP_MATH;
      LWDNN_CHECK(lwdnnSetColwolutionMathType(fwd_colw_descs_[i], fwd_lwdnn_math_[i]));
    }
#endif
    // Find forward algorithm
    if (user_algos_override_[0] < 0) {
      float algo_time = 0.F;
      for (int m = 0; m < 2; ++m) {
        if (m > 0 &&
            // if user wants specific math type, no need to check anything else
            (this->is_fm_by_user() ||
             // also, we skip this in fp32/64 modes
             !is_type<Ftype>(FLOAT16) ||
             // and sanity check for current descriptor type
             colwolutionDescDataType(fwd_colw_descs_[i]) != LWDNN_DATA_HALF)) {
          break;
        }
        if (m == 1) {
          // second run in pseudo fp32 mode
          setColwolutionDescMath(FLOAT, fwd_colw_descs_[i]);
        }

        int prev_algo = -1;
        for (int t = 0; t < 5; ++t) {
          LWDNN_CHECK(lwdnnFindColwolutionForwardAlgorithmEx(handle,
              fwd_bottom_descs_[i],
              bottom[i]->gpu_data<Ftype>(),
              fwd_filter_desc_,
              this->blobs_[0]->template gpu_data<Ftype>(),
              fwd_colw_descs_[i],
              fwd_top_descs_[i],
              top[i]->mutable_gpu_data<Ftype>(),  // overwritten
              REQUEST_ALGO_COUNT,
              &fwd_algo_count,
              fwd_results,
              Caffe::ws(CAFFE_WS_COLW).data(),
              gsize));
          LWDA_CHECK(lwdaStreamSynchronize(stream));
          CHECK_GT(fwd_algo_count, 0);
          // Waiting for two identical decisions in a row
          if (prev_algo == (int)fwd_results[0].algo) {
            break;
          }
          prev_algo = (int)fwd_results[0].algo;
        }

        for (int k = 0; k < fwd_algo_count; ++k) {
          if (fwd_results[k].status == LWDNN_STATUS_SUCCESS) {
            if (m == 0) {
              algo_time = fwd_results[k].time;
            } else {
              // here we compare pseudo fp32 against native fp16
              if (fwd_results[k].time >= algo_time) {
                // pseudo fp32 lost, switching back to native fp16
                setColwolutionDescMath(FLOAT16, fwd_colw_descs_[i]);
                break;
              }
              // pseudo fp32 won
              forward_math_ = tpm(tp<Ftype>(), FLOAT);
            }
            fwd_algo_[i] = fwd_results[k].algo;
#if LWDNN_VERSION_MIN(7, 0, 0)
            if (lwdnn_math_override_ < 0) {
              // Winning Math for either native or pseudo mode:
              fwd_lwdnn_math_0 = fwd_results[k].mathType;
            } else {
              fwd_lwdnn_math_0 =
                  lwdnn_math_override_ == 0 ? LWDNN_DEFAULT_MATH : LWDNN_TENSOR_OP_MATH;
            }
#endif
            workspace_fwd_sizes_[i] = fwd_results[k].memory;
            mem_req(this->phase_, align_up<8>(workspace_fwd_sizes_[i]) * ws_groups());
            fwd_pseudo = is_precise(forward_math_) && !is_precise(tp<Ftype>());
            break;
          }
        }
      }
    }
#if LWDNN_VERSION_MIN(7, 0, 0)
    if (top_device) {
      fwd_lwdnn_math_[i] = fwd_lwdnn_math_0;
      LWDNN_CHECK(lwdnnSetColwolutionMathType(fwd_colw_descs_[i], fwd_lwdnn_math_[i]));
    }
#endif

    // Only set backward-filter/data algorithms in training phase
    if (this->phase_ == TRAIN) {
#if LWDNN_VERSION_MIN(7, 0, 0)
      lwdnnMathType_t bwd_filter_lwdnn_math_0 = LWDNN_DEFAULT_MATH;
      if (try_top) {
        bwd_filter_lwdnn_math_[i] = bwd_filter_lwdnn_math_0 =
            lwdnn_math_override_ == 0 ? LWDNN_DEFAULT_MATH : LWDNN_TENSOR_OP_MATH;
        LWDNN_CHECK(lwdnnSetColwolutionMathType(bwd_colw_filter_descs_[i],
            bwd_filter_lwdnn_math_[i]));
      }
#endif
      if (user_algos_override_[2] < 0) {
        const size_t tmp_weights_size = mem_req(MEM_REQ_TMP_WEIGHTS);
        Caffe::ws(CAFFE_WS_COLW_WEIGHTS).safe_reserve(tmp_weights_size, Caffe::device());
        float algo_time = 0.F;
        for (int m = 0; m < 2; ++m) {
          if (m > 0 &&
              // if user wants specific math type, no need to check anything else
              (this->is_bm_by_user() ||
               // also, we skip this in fp32/64 modes
               !is_type<Ftype>(FLOAT16) ||
               // and sanity check for current descriptor type
               colwolutionDescDataType(bwd_colw_filter_descs_[i])
               != LWDNN_DATA_HALF)) {
            break;
          }
          if (m == 1) {
            // second run in pseudo fp32 mode
            setColwolutionDescMath(FLOAT, bwd_colw_filter_descs_[i]);
          }

          int prev_algo = -1;
          for (int t = 0; t < 5; ++t) {
            // Find backward filter algorithm
            LWDNN_CHECK(lwdnnFindColwolutionBackwardFilterAlgorithmEx(handle,
                bwd_bottom_descs_[i],
                bottom[i]->gpu_data<Btype>(),
                bwd_top_descs_[i],
                top[i]->gpu_diff<Btype>(),
                bwd_colw_filter_descs_[i],
                bwd_filter_desc_,
                Caffe::ws(CAFFE_WS_COLW_WEIGHTS).data(),  // overwritten
                REQUEST_ALGO_COUNT,
                &filter_algo_count,
                bwd_filter_results,
                Caffe::ws(CAFFE_WS_COLW).data(),
                gsize));
            LWDA_CHECK(lwdaStreamSynchronize(stream));
            CHECK_GT(filter_algo_count, 0);
            // Waiting for two identical decisions in a row
            if (prev_algo == (int)bwd_filter_results[0].algo) {
              break;
            }
            prev_algo = (int)bwd_filter_results[0].algo;
          }

          for (int k = 0; k < filter_algo_count; ++k) {
            if (bwd_filter_results[k].status == LWDNN_STATUS_SUCCESS) {
              if (m == 0) {
                algo_time = bwd_filter_results[k].time;
              } else {
                // here we compare pseudo fp32 against native fp16
                if (bwd_filter_results[k].time >= algo_time) {
                  // pseudo fp32 lost, switching back to native fp16
                  setColwolutionDescMath(FLOAT16, bwd_colw_filter_descs_[i]);
                  break;
                }
                // pseudo fp32 won
                backward_filter_math_ = tpm(tp<Btype>(), FLOAT);
              }
              bwd_filter_algo_[i] = bwd_filter_results[k].algo;
#if LWDNN_VERSION_MIN(7, 0, 0)
              if (lwdnn_math_override_ < 0) {
                // Winning Math for either native or pseudo mode:
                bwd_filter_lwdnn_math_0 = bwd_filter_results[k].mathType;
              } else {
                bwd_filter_lwdnn_math_0 =
                    lwdnn_math_override_ == 0 ? LWDNN_DEFAULT_MATH : LWDNN_TENSOR_OP_MATH;
              }
#endif
              workspace_bwd_filter_sizes_[i] = bwd_filter_results[k].memory;
              mem_req(TRAIN, align_up<8>(workspace_bwd_filter_sizes_[i]) * ws_groups());
              bwd_filter_pseudo = is_precise(backward_filter_math_) && !is_precise(tp<Btype>());
              bftime = bwd_filter_results[k].time;
              break;
            }
          }
        }
      }
#if LWDNN_VERSION_MIN(7, 0, 0)
      if (top_device && !use_modest_workspace()) {
        bwd_filter_lwdnn_math_[i] = bwd_filter_lwdnn_math_0;
        LWDNN_CHECK(lwdnnSetColwolutionMathType(bwd_colw_filter_descs_[i],
            bwd_filter_lwdnn_math_[i]));
      }
#endif
      if (propagate_down_.size() > i && propagate_down_[i]) {
#if LWDNN_VERSION_MIN(7, 0, 0)
        lwdnnMathType_t bwd_data_lwdnn_math_0 = LWDNN_DEFAULT_MATH;
        if (try_top && lwdnn_math_override_ != 0) {
          bwd_data_lwdnn_math_[i] = bwd_data_lwdnn_math_0 =
              lwdnn_math_override_ == 0 ? LWDNN_DEFAULT_MATH : LWDNN_TENSOR_OP_MATH;
          LWDNN_CHECK(lwdnnSetColwolutionMathType(bwd_colw_data_descs_[i],
              bwd_data_lwdnn_math_[i]));
        }
#endif
        if (user_algos_override_[1] < 0) {
          float algo_time = 0.F;
          for (int m = 0; m < 2; ++m) {
            if (m > 0 &&
                // if user wants specific math type, no need to check anything else
                (this->is_bm_by_user() ||
                 // also, we skip this in fp32/64 modes
                 !is_type<Ftype>(FLOAT16) ||
                 // and sanity check for current descriptor type
                 colwolutionDescDataType(bwd_colw_data_descs_[i])
                 != LWDNN_DATA_HALF)) {
              break;
            }
            if (m == 1) {
              // second run in pseudo fp32 mode
              setColwolutionDescMath(FLOAT, bwd_colw_data_descs_[i]);
            }

            int prev_algo = -1;
            for (int t = 0; t < 5; ++t) {
              // Find backward data algorithm
              LWDNN_CHECK(lwdnnFindColwolutionBackwardDataAlgorithmEx(handle,
                  bwd_filter_desc_,
                  this->blobs_[0]->template gpu_data<Btype>(),
                  bwd_top_descs_[i],
                  top[i]->gpu_diff<Btype>(),
                  bwd_colw_data_descs_[i],
                  bwd_bottom_descs_[i],
                  bottom[i]->mutable_gpu_diff<Btype>(),  // overwritten
                  REQUEST_ALGO_COUNT,
                  &data_algo_count,
                  bwd_data_results,
                  Caffe::ws(CAFFE_WS_COLW).data(),
                  gsize));
              LWDA_CHECK(lwdaStreamSynchronize(stream));
              CHECK_GT(data_algo_count, 0);
              // Waiting for two identical decisions in a row
              if (prev_algo == (int) bwd_data_results[0].algo) {
                break;
              }
              prev_algo = (int) bwd_data_results[0].algo;
            }

            for (int k = 0; k < data_algo_count; ++k) {
              if (bwd_data_results[k].status == LWDNN_STATUS_SUCCESS) {
                if (m == 0) {
                  algo_time = bwd_data_results[k].time;
                } else {
                  // here we compare pseudo fp32 against native fp16
                  if (bwd_data_results[k].time >= algo_time) {
                    // pseudo fp32 lost, switching back to native fp16
                    setColwolutionDescMath(FLOAT16, bwd_colw_data_descs_[i]);
                    break;
                  }
                  // pseudo fp32 won
                  backward_data_math_ = tpm(tp<Btype>(), FLOAT);
                }
                bwd_data_algo_[i] = bwd_data_results[k].algo;
#if LWDNN_VERSION_MIN(7, 0, 0)
                if (lwdnn_math_override_ < 0) {
                  // Winning Math for either native or pseudo mode:
                  bwd_data_lwdnn_math_0 = bwd_data_results[k].mathType;
                } else {
                  bwd_data_lwdnn_math_0 =
                      lwdnn_math_override_ == 0 ? LWDNN_DEFAULT_MATH : LWDNN_TENSOR_OP_MATH;
                }
#endif
                workspace_bwd_data_sizes_[i] = bwd_data_results[k].memory;
                mem_req(TRAIN, align_up<8>(workspace_bwd_data_sizes_[i]) * ws_groups());
                bwd_data_pseudo = is_precise(backward_data_math_) && !is_precise(tp<Btype>());
                bdtime = bwd_data_results[k].time;
                break;
              }
            }
          }
        }
#if LWDNN_VERSION_MIN(7, 0, 0)
        if (top_device) {
          bwd_data_lwdnn_math_[i] = bwd_data_lwdnn_math_0;
          LWDNN_CHECK(lwdnnSetColwolutionMathType(bwd_colw_data_descs_[i],
              bwd_data_lwdnn_math_[i]));
        }
#endif
      }
    }
    LWDA_CHECK(lwdaStreamSynchronize(Caffe::thread_stream()));
    DLOG(INFO) << this->print_lwrrent_device() << " FindEx to release "
        << Caffe::ws(CAFFE_WS_COLW).size() << " bytes";
    Caffe::ws(CAFFE_WS_COLW).release();
    AllocateWorkspace(bottom.size());  // if user overrides
    Caffe::ws(CAFFE_WS_COLW_WEIGHTS).release();

    size_t available_memory, total_memory;
    GPUMemory::GetInfo(&available_memory, &total_memory, true);
    std::ostringstream os;
    os << this->print_lwrrent_device()
        << (this->phase_ == TRAIN ? " Colw Algos (F,BD,BF): '" : " Colw Algo (F): '")
        << this->name() << "' with space "
        << mem_fmt(Caffe::ws(CAFFE_WS_COLW).size()) << " "
        << this->channels_ << "/" << this->group_ << (use_v7grouping() ? "." : "")
#ifdef DEBUG
        << " -> [" << workspace_fwd_sizes_[i]
        << " " << workspace_bwd_data_sizes_[i]
        << " " << workspace_bwd_filter_sizes_[i] << "]"
#endif
        << " " << fwd_algo_[i]
#if LWDNN_VERSION_MIN(7, 0, 0)
        << (fwd_lwdnn_math_[i] == LWDNN_TENSOR_OP_MATH ? "T" : "")
#endif
        << (user_algos_override_[0] >= 0 ? "u " : (fwd_pseudo ? "p " : " "));

    if (this->phase_ == TRAIN) {
      os << bwd_data_algo_[i]
#if LWDNN_VERSION_MIN(7, 0, 0)
          << (bwd_data_lwdnn_math_[i] == LWDNN_TENSOR_OP_MATH ? "T" : "")
#endif
          << (user_algos_override_[1] >= 0 ? "u " : (bwd_data_pseudo ? "p " : " "))
          << bwd_filter_algo_[i]
#if LWDNN_VERSION_MIN(7, 0, 0)
          << (bwd_filter_lwdnn_math_[i] == LWDNN_TENSOR_OP_MATH ? "T" : "")
#endif
          << (user_algos_override_[2] >= 0 ? "u " : (bwd_filter_pseudo ? "p " : " "));
    }

    os << "\t(avail " << mem_fmt(available_memory) << ", req "
        << mem_fmt(mem_req(this->phase_)) << ")\tt: " << f_round2(ftime);

    if (this->phase_ == TRAIN) {
      os << " " << f_round2(bdtime) << " " << f_round2(bftime);
    }

    LOG_IF(INFO, P2PManager::global_rank() == 0) << os.str();
  }
}

// Checked if there is a difference between the corresponding descriptors in
// cached_bottom_descs_ and bottom_descs_.
// No need to compare all parameters: batchsize, height, and width are enough.
template <typename Ftype, typename Btype>
bool LwDNNColwolutionLayer<Ftype, Btype>::IsBottomDescChanged(
  const vector<Blob*>& bottom, bool fwd_mode) {
  int cached_n; int cached_c; int cached_h; int cached_w;
  int cached_stride_n; int cached_stride_c;
  int cached_stride_h; int cached_stride_w;
  int n; int c; int h; int w;
  lwdnnDataType_t type;

  for (int i = 0; i < bottom.size(); i++) {
    LWDNN_CHECK(lwdnnGetTensor4dDescriptor(
        fwd_mode ? fwd_cached_bottom_descs_[i] : bwd_cached_bottom_descs_[i],
        &type,
        &cached_n, &cached_c, &cached_h, &cached_w,
        &cached_stride_n, &cached_stride_c,
        &cached_stride_h, &cached_stride_w));
    const vector<int>& shape = bottom[i]->shape();
    n = shape[0];
    c = shape[1] / ws_groups();
    h = shape[2];
    w = shape[3];

    if ((cached_n != n) || (cached_c != c) || (cached_h != h) || (cached_w != w)) {
      return true;
    }
  }
  return false;
}

// Checked if there is a difference between the corresponding descriptors in
// cached_colw_descs_ and colw_descs_.
// No need to compare all parameters; pads, strides, and upscales are enough.
template <typename Ftype, typename Btype>
bool LwDNNColwolutionLayer<Ftype, Btype>::IsColwDescChanged(
  const vector<Blob*>& bottom, bool fwd_mode) {
  int cached_padA[2];
  int padA[2];
  int cached_strideA[2];
  int strideA[2];
  int cached_upscaleA[2];
  int upscaleA[2];
  int arrayLength;
  lwdnnColwolutionMode_t mode;
  lwdnnDataType_t type;

  for (int i = 0; i < bottom.size(); i++) {
    LWDNN_CHECK(lwdnnGetColwolutionNdDescriptor(
        fwd_mode ? fwd_cached_colw_descs_[i] : bwd_cached_colw_data_descs_[i],
        2, &arrayLength, cached_padA, cached_strideA, cached_upscaleA,
        &mode, &type));
    LWDNN_CHECK(lwdnnGetColwolutionNdDescriptor(
        fwd_mode ? fwd_colw_descs_[i] : bwd_colw_data_descs_[i],
        2, &arrayLength, padA, strideA, upscaleA, &mode, &type));
    if ((cached_padA[0] != padA[0]) ||
        (cached_padA[1] != padA[1]) ||
        (cached_strideA[0]  != strideA[0])  ||
        (cached_strideA[1]  != strideA[1])  ||
        (cached_upscaleA[0] != upscaleA[0]) ||
        (cached_upscaleA[1] != upscaleA[1])) {
      return true;
    }
    if (!fwd_mode) {
      LWDNN_CHECK(lwdnnGetColwolutionNdDescriptor(
        bwd_cached_colw_filter_descs_[i],
        2, &arrayLength, cached_padA, cached_strideA, cached_upscaleA,
        &mode, &type));
      LWDNN_CHECK(lwdnnGetColwolutionNdDescriptor(
        bwd_colw_filter_descs_[i],
        2, &arrayLength, padA, strideA, upscaleA, &mode, &type));
      if ((cached_padA[0] != padA[0]) ||
          (cached_padA[1] != padA[1]) ||
          (cached_strideA[0]  != strideA[0])  ||
          (cached_strideA[1]  != strideA[1])  ||
          (cached_upscaleA[0] != upscaleA[0]) ||
          (cached_upscaleA[1] != upscaleA[1])) {
        return true;
      }
    }
  }
  return false;
}

template <typename Ftype, typename Btype>
LwDNNColwolutionLayer<Ftype, Btype>::~LwDNNColwolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < fwd_bottom_descs_.size(); ++i) {
    LWDNN_CHECK(lwdnnDestroyTensorDescriptor(fwd_bottom_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyTensorDescriptor(bwd_bottom_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyTensorDescriptor(fwd_top_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyTensorDescriptor(bwd_top_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyColwolutionDescriptor(fwd_colw_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyColwolutionDescriptor(bwd_colw_data_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyColwolutionDescriptor(bwd_colw_filter_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyTensorDescriptor(fwd_cached_bottom_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyTensorDescriptor(bwd_cached_bottom_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyColwolutionDescriptor(fwd_cached_colw_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyColwolutionDescriptor(bwd_cached_colw_data_descs_[i]));
    LWDNN_CHECK(lwdnnDestroyColwolutionDescriptor(bwd_cached_colw_filter_descs_[i]));
  }
  if (this->bias_term_) {
    LWDNN_CHECK(lwdnnDestroyTensorDescriptor(fwd_bias_desc_));
    LWDNN_CHECK(lwdnnDestroyTensorDescriptor(bwd_bias_desc_));
  }
  LWDNN_CHECK(lwdnnDestroyFilterDescriptor(fwd_filter_desc_));
  LWDNN_CHECK(lwdnnDestroyFilterDescriptor(bwd_filter_desc_));
}

INSTANTIATE_CLASS_FB(LwDNNColwolutionLayer);

}   // namespace caffe
#endif
