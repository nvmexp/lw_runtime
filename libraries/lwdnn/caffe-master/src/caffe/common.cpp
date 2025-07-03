#include <glog/logging.h>
#include <syscall.h>
#include <cmath>
#include <ctime>
#include <memory>

#include "caffe/common.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/rng.hpp"
#if defined(USE_LWDNN)
#include "caffe/util/lwdnn.hpp"
#endif

namespace caffe {

// Must be set before brewing
Caffe::Brew Caffe::mode_ = Caffe::GPU;
int Caffe::solver_count_ = 1;
std::vector<int> Caffe::gpus_;
int Caffe::root_device_ = -1;
int Caffe::thread_count_ = 0;
int Caffe::restored_iter_ = -1;
std::atomic<uint64_t> Caffe::root_seed_(Caffe::SEED_NOT_SET);
// NOLINT_NEXT_LINE(runtime/int)
std::atomic<size_t> Caffe::epoch_count_(static_cast<size_t>(-1L));

std::mutex Caffe::cd_mutex_;
std::mutex Caffe::caffe_mutex_;
std::mutex Caffe::pstream_mutex_;
std::mutex Caffe::lwblas_mutex_;
std::mutex Caffe::lwdnn_mutex_;
std::mutex Caffe::seed_mutex_;

std::uint32_t lwp_id() {
#if defined(APPLE)
  return static_cast<std::uint32_t>(std::this_thread::get_id());
#else
  return static_cast<std::uint32_t>(syscall(SYS_gettid));
#endif
}

std::uint64_t lwp_dev_id(int dev) {
  std::uint64_t dev64 = static_cast<std::uint64_t>(dev < 0 ? Caffe::lwrrent_device() : dev);
  return lwp_id() + (dev64 << 32LL);
}

Caffe& Caffe::Get() {
  // Make sure each thread can have different values.
  std::lock_guard<std::mutex> lock(caffe_mutex_);
  static thread_local Caffe caffe;
  DCHECK_EQ(caffe._device(), lwrrent_device()) << " thread " << lwp_id();
  return caffe;
}

// random seeding
uint64_t cluster_seedgen(void) {
  uint64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = static_cast<uint64_t>(getpid());
  s = static_cast<uint64_t>(time(NULL));
  seed = static_cast<uint64_t>(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

void Caffe::set_root_seed(uint64_t random_seed) {
  if (random_seed != Caffe::SEED_NOT_SET) {
    root_seed_.store(random_seed);
    set_random_seed(random_seed);
  }
}

void Caffe::set_random_seed_int(uint64_t random_seed) {
  std::lock_guard<std::mutex> lock(seed_mutex_);
  if (root_seed_.load() == Caffe::SEED_NOT_SET) {
    root_seed_.store(random_seed);
  } else if (random_seed == Caffe::SEED_NOT_SET) {
    return;  // i.e. root solver was previously set to 0+ and there is no need to re-generate
  }
  // Lwrand seed
  if (random_seed == Caffe::SEED_NOT_SET) {
    random_seed = cluster_seedgen();
  }
  init();
  LWRAND_CHECK(lwrandSetPseudoRandomGeneratorSeed(lwrand_generator_, random_seed));
  LWRAND_CHECK(lwrandSetGeneratorOffset(lwrand_generator_, 0));
  // RNG seed
  random_generator_.reset(new RNG(random_seed + P2PManager::global_rank()));
}

uint64_t Caffe::next_seed() {
  return (*caffe_rng())();
}

void Caffe::set_restored_iter(int val) {
  std::lock_guard<std::mutex> lock(caffe_mutex_);
  restored_iter_ = val;
}

void GlobalInit(int* pargc, char*** pargv) {
  P2PManager::Init(pargc, pargv);
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

int Caffe::device_count() {
  int count = 0;
  lwdaGetDeviceCount(&count);
  return count;
}

Caffe::Caffe()
    : lwrand_generator_(nullptr),
      random_generator_(),
      is_root_solver_(true),
      device_(lwrrent_device()),
      gpu_memory_scope_(Caffe::gpus_) {
  ++thread_count_;
  DLOG(INFO) << "[" << _device()
             << "] New Caffe instance " << this
             << ", count " << thread_count_ << ", thread " << lwp_id();
  init();
}

void Caffe::init() {
  if (lwrand_generator_ == nullptr) {
    lwrand_stream_ = LwdaStream::create();
    LWRAND_CHECK(lwrandCreateGenerator(&lwrand_generator_, LWRAND_RNG_PSEUDO_DEFAULT));
    LWRAND_CHECK(lwrandSetPseudoRandomGeneratorSeed(lwrand_generator_, cluster_seedgen()));
    LWRAND_CHECK(lwrandSetStream(lwrand_generator_, lwrand_stream_->get()));
  }
}

Caffe::~Caffe() {
  std::lock_guard<std::mutex> lock(caffe_mutex_);
  int lwrrent_device;  // Just to check LWCA status:
  lwdaError_t status = lwdaGetDevice(&lwrrent_device);
  // Preventing crash while Caffe shutting down.
  if (status != lwdaErrorLwdartUnloading && lwrand_generator_ != nullptr) {
    LWRAND_CHECK(lwrandDestroyGenerator(lwrand_generator_));
  }
  --thread_count_;
  DLOG(INFO) << "[" << lwrrent_device
             << "] Caffe instance " << this
             << " deleted, count " << thread_count_ << ", thread "
             << lwp_id();
}

size_t Caffe::min_avail_device_memory() {
  std::lock_guard<std::mutex> lock(caffe_mutex_);
  size_t ret = 0UL;
  const std::vector<int>& lwr_gpus = gpus();
  int lwr_device;
  size_t gpu_bytes, total_memory;
  LWDA_CHECK(lwdaGetDevice(&lwr_device));
  GPUMemory::GetInfo(&ret, &total_memory, true);
  for (int gpu : lwr_gpus) {
    if (gpu != lwr_device) {
      LWDA_CHECK(lwdaSetDevice(gpu));
      GPUMemory::GetInfo(&gpu_bytes, &total_memory, true);
      if (gpu_bytes < ret) {
        ret = gpu_bytes;
      }
    }
  }
  LWDA_CHECK(lwdaSetDevice(lwr_device));
  return ret;
}

LwdaStream::LwdaStream(bool high_priority) {
  if (high_priority) {
    int leastPriority, greatestPriority;
    LWDA_CHECK(lwdaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    LWDA_CHECK(lwdaStreamCreateWithPriority(&stream_, lwdaStreamDefault, greatestPriority));
  } else {
    LWDA_CHECK(lwdaStreamCreate(&stream_));
  }
  DLOG(INFO) << "New " << (high_priority ? "high priority " : "") << "stream "
      << stream_ << ", device " << Caffe::lwrrent_device() << ", thread "
      << lwp_id();
}

LwdaStream::~LwdaStream() {
  int lwrrent_device;  // Just to check LWCA status:
  lwdaError_t status = lwdaGetDevice(&lwrrent_device);
  // Preventing dead lock while Caffe shutting down.
  if (status != lwdaErrorLwdartUnloading) {
    LWDA_CHECK(lwdaStreamDestroy(stream_));
  }
}

shared_ptr<LwdaStream> Caffe::pstream(int group) {
  CHECK_GE(group, 0);
  std::lock_guard<std::mutex> lock(pstream_mutex_);
  if (group < streams_.size() && streams_[group]) {
    return streams_[group];
  }
  if (group >= streams_.size()) {
    streams_.resize(group + 1UL);
  }
  if (!streams_[group]) {
    streams_[group] = LwdaStream::create();
  }
  return streams_[group];
}

shared_ptr<LwBLASHandle> Caffe::th_lwblas_handle(int group) {
  CHECK_GE(group, 0);
  std::lock_guard<std::mutex> lock(lwblas_mutex_);
  if (group < lwblas_handles_.size() && lwblas_handles_[group]) {
    return lwblas_handles_[group];
  }
  if (group >= lwblas_handles_.size()) {
    lwblas_handles_.resize(group + 1UL);
  }
  if (!lwblas_handles_[group]) {
    lwblas_handles_[group] = make_shared<LwBLASHandle>(pstream(group));
  }
  return lwblas_handles_[group];
}

#ifdef USE_LWDNN
lwdnnHandle_t Caffe::th_lwdnn_handle(int group) {
  CHECK_GE(group, 0);
  std::lock_guard<std::mutex> lock(lwdnn_mutex_);
  if (group < lwdnn_handles_.size() && lwdnn_handles_[group]) {
    return lwdnn_handles_[group]->get();
  }
  if (group >= lwdnn_handles_.size()) {
    lwdnn_handles_.resize(group + 1UL);
  }
  if (!lwdnn_handles_[group]) {
    lwdnn_handles_[group] = make_shared<LwDNNHandle>(pstream(group));
  }
  return lwdnn_handles_[group]->get();
}
#endif

void Caffe::SetDevice(const int device_id) {
  root_device_ = device_id;
  LWDA_CHECK(lwdaSetDevice(root_device_));
}

std::string Caffe::DeviceQuery() {
  lwdaDeviceProp prop;
  int device;
  std::ostringstream os;
  if (lwdaSuccess != lwdaGetDevice(&device)) {
    os << "No lwca device present." << std::endl;
  } else {
    LWDA_CHECK(lwdaGetDeviceProperties(&prop, device));
    os << "Device id:                     " << device << std::endl;
    os << "Major revision number:         " << prop.major << std::endl;
    os << "Minor revision number:         " << prop.minor << std::endl;
    os << "Name:                          " << prop.name << std::endl;
    os << "Total global memory:           " << prop.totalGlobalMem << std::endl;
    os << "Total shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    os << "Total registers per block:     " << prop.regsPerBlock << std::endl;
    os << "Warp size:                     " << prop.warpSize << std::endl;
    os << "Maximum memory pitch:          " << prop.memPitch << std::endl;
    os << "Maximum threads per block:     " << prop.maxThreadsPerBlock << std::endl;
    os << "Maximum dimension of block:    "
        << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
        << prop.maxThreadsDim[2] << std::endl;
    os << "Maximum dimension of grid:     "
        << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
        << prop.maxGridSize[2] << std::endl;
    os << "Clock rate:                    " << prop.clockRate << std::endl;
    os << "Total constant memory:         " << prop.totalConstMem << std::endl;
    os << "Texture alignment:             " << prop.textureAlignment << std::endl;
    os << "Conlwrrent copy and exelwtion: "
        << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
    os << "Number of multiprocessors:     " << prop.multiProcessorCount << std::endl;
    os << "Kernel exelwtion timeout:      "
        << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
  }
  return os.str();
}

bool Caffe::CheckDevice(const int device_id) {
  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling lwdaFree(0).
  // lwdaSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, lwdaSetDevice() returns lwdaSuccess
  // even if the device is exclusively oclwpied by another process or thread.
  // Lwca operations that initialize the context are needed to check
  // the permission. lwdaFree(0) is one of those with no side effect,
  // except the context initialization.
  bool r = ((lwdaSuccess == lwdaSetDevice(device_id)) &&
            (lwdaSuccess == lwdaFree(0)));
  // reset any error that may have oclwrred.
  lwdaGetLastError();
  return r;
}

int Caffe::FindDevice(const int start_id) {
  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
  int count = 0;
  LWDA_CHECK(lwdaGetDeviceCount(&count));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(uint64_t seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG()
    : generator_(new Generator()) {}

Caffe::RNG::RNG(uint64_t seed)
    : generator_(new Generator(seed)) {}

Caffe::RNG::RNG(const RNG& other)
    : generator_(other.generator_) {}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

const char* lwblasGetErrorString(lwblasStatus_t error) {
  switch (error) {
  case LWBLAS_STATUS_SUCCESS:
    return "LWBLAS_STATUS_SUCCESS";
  case LWBLAS_STATUS_NOT_INITIALIZED:
    return "LWBLAS_STATUS_NOT_INITIALIZED";
  case LWBLAS_STATUS_ALLOC_FAILED:
    return "LWBLAS_STATUS_ALLOC_FAILED";
  case LWBLAS_STATUS_ILWALID_VALUE:
    return "LWBLAS_STATUS_ILWALID_VALUE";
  case LWBLAS_STATUS_ARCH_MISMATCH:
    return "LWBLAS_STATUS_ARCH_MISMATCH";
  case LWBLAS_STATUS_MAPPING_ERROR:
    return "LWBLAS_STATUS_MAPPING_ERROR";
  case LWBLAS_STATUS_EXELWTION_FAILED:
    return "LWBLAS_STATUS_EXELWTION_FAILED";
  case LWBLAS_STATUS_INTERNAL_ERROR:
    return "LWBLAS_STATUS_INTERNAL_ERROR";
#if LWDA_VERSION >= 6000
  case LWBLAS_STATUS_NOT_SUPPORTED:
    return "LWBLAS_STATUS_NOT_SUPPORTED";
#endif
#if LWDA_VERSION >= 6050
  case LWBLAS_STATUS_LICENSE_ERROR:
    return "LWBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown lwblas status";
}

const char* lwrandGetErrorString(lwrandStatus_t error) {
  switch (error) {
  case LWRAND_STATUS_SUCCESS:
    return "LWRAND_STATUS_SUCCESS";
  case LWRAND_STATUS_VERSION_MISMATCH:
    return "LWRAND_STATUS_VERSION_MISMATCH";
  case LWRAND_STATUS_NOT_INITIALIZED:
    return "LWRAND_STATUS_NOT_INITIALIZED";
  case LWRAND_STATUS_ALLOCATION_FAILED:
    return "LWRAND_STATUS_ALLOCATION_FAILED";
  case LWRAND_STATUS_TYPE_ERROR:
    return "LWRAND_STATUS_TYPE_ERROR";
  case LWRAND_STATUS_OUT_OF_RANGE:
    return "LWRAND_STATUS_OUT_OF_RANGE";
  case LWRAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "LWRAND_STATUS_LENGTH_NOT_MULTIPLE";
  case LWRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "LWRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case LWRAND_STATUS_LAUNCH_FAILURE:
    return "LWRAND_STATUS_LAUNCH_FAILURE";
  case LWRAND_STATUS_PREEXISTING_FAILURE:
    return "LWRAND_STATUS_PREEXISTING_FAILURE";
  case LWRAND_STATUS_INITIALIZATION_FAILED:
    return "LWRAND_STATUS_INITIALIZATION_FAILED";
  case LWRAND_STATUS_ARCH_MISMATCH:
    return "LWRAND_STATUS_ARCH_MISMATCH";
  case LWRAND_STATUS_INTERNAL_ERROR:
    return "LWRAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown lwrand status";
}

const double  TypedConsts<double>::zero = 0.0;
const double  TypedConsts<double>::one = 1.0;
const float   TypedConsts<float>::zero = 0.0f;
const float   TypedConsts<float>::one = 1.0f;
const float16 TypedConsts<float16>::zero = 0.0f;
const float16 TypedConsts<float16>::one = 1.0f;
const int     TypedConsts<int>::zero = 0;
const int     TypedConsts<int>::one = 1;

LwBLASHandle::LwBLASHandle()
  : handle_(nullptr), stream_(Caffe::thread_pstream()) {
  LWBLAS_CHECK(lwblasCreate(&handle_));
  LWBLAS_CHECK(lwblasSetStream(handle_, stream_->get()));
}
LwBLASHandle::LwBLASHandle(shared_ptr<LwdaStream> stream)
    : handle_(nullptr), stream_(std::move(stream)) {
  LWBLAS_CHECK(lwblasCreate(&handle_));
  LWBLAS_CHECK(lwblasSetStream(handle_, stream_->get()));
}
LwBLASHandle::~LwBLASHandle() {
  LWBLAS_CHECK(lwblasDestroy(handle_));
}
#ifdef USE_LWDNN
LwDNNHandle::LwDNNHandle(shared_ptr<LwdaStream> stream)
  : handle_(nullptr), stream_(std::move(stream)) {
  LWDNN_CHECK(lwdnnCreate(&handle_));
  LWDNN_CHECK(lwdnnSetStream(handle_, stream_->get()));
}
LwDNNHandle::~LwDNNHandle() {
  LWDNN_CHECK(lwdnnDestroy(handle_));
}
#endif

Caffe::Properties& Caffe::props() {
  static Caffe::Properties props_;
  return props_;
}

Caffe::Properties::Properties() :
      init_time_(std::time(nullptr)),
      caffe_version_(AS_STRING(CAFFE_VERSION)) {
  const std::vector<int>& gpus = Caffe::gpus();
  const int count = gpus.size();
  if (count == 0) {
    return;
  }
  compute_capabilities_.resize(count);
  lwdaDeviceProp device_prop;
  for (int gpu = 0; gpu < compute_capabilities_.size(); ++gpu) {
    LWDA_CHECK(lwdaGetDeviceProperties(&device_prop, gpus[gpu]));
    compute_capabilities_[gpu] = device_prop.major * 100 + device_prop.minor;
    DLOG(INFO) << "GPU " << gpus[gpu] << " '" << device_prop.name
               << "' has compute capability " << device_prop.major << "." << device_prop.minor;
  }
#ifdef USE_LWDNN
  lwdnn_version_ = std::to_string(lwdnnGetVersion());
#else
  lwdnn_version_ = "USE_LWDNN is not defined";
#endif
  shared_ptr<LwBLASHandle> phandle = Caffe::short_term_lwblas_phandle();
  int lwblas_version = 0;
  LWBLAS_CHECK(lwblasGetVersion(phandle->get(), &lwblas_version));
  lwblas_version_ = std::to_string(lwblas_version);

  int lwda_version = 0;
  LWDA_CHECK(lwdaRuntimeGetVersion(&lwda_version));
  lwda_version_ = std::to_string(lwda_version);

  int lwda_driver_version = 0;
  LWDA_CHECK(lwdaDriverGetVersion(&lwda_driver_version));
  lwda_driver_version_ = std::to_string(lwda_driver_version);
}

std::string Caffe::time_from_init() {
  std::ostringstream os;
  os.unsetf(std::ios_base::floatfield);
  os.precision(4);
  double span = std::difftime(std::time(NULL), init_time());
  const double mn = 60.;
  const double hr = 3600.;
  if (span < mn) {
    os << span << "s";
  } else if (span < hr) {
    int m = static_cast<int>(span / mn);
    double s = span - m * mn;
    os << m << "m " << s << "s";
  } else {
    int h = static_cast<int>(span / hr);
    int m = static_cast<int>((span - h * hr) / mn);
    double s = span - h * hr - m * mn;
    os << h << "h " << m << "m " << s << "s";
  }
  return os.str();
}

#ifndef NO_LWML
namespace lwml {

std::mutex LWMLInit::m_;

LWMLInit::LWMLInit() {
  if (lwmlInit() != LWML_SUCCESS) {
    LOG(ERROR) << "LWML failed to initialize";
  } else {
    LOG(INFO) << "LWML initialized, thread " << lwp_id();
  }
}

LWMLInit::~LWMLInit() {
  lwmlShutdown();
}

// set the CPU affinity for this thread
void setCpuAffinity(int device) {
  std::lock_guard<std::mutex> lock(LWMLInit::m_);
  static LWMLInit lwml_init_;

  char pciBusId[16];
  LWDA_CHECK(lwdaDeviceGetPCIBusId(pciBusId, 15, device));
  lwmlDevice_t lwml_device;

  if (lwmlDeviceGetHandleByPciBusId(pciBusId, &lwml_device) != LWML_SUCCESS ||
      lwmlDeviceSetCpuAffinity(lwml_device) != LWML_SUCCESS) {
    LOG(ERROR) << "LWML failed to set CPU affinity on device " << device
               << ", thread " << lwp_id();
  } else {
    LOG(INFO) << "{"
#ifdef USE_MPI
       << P2PManager::global_rank() << "."
#endif
       << device << "} LWML succeeded to set CPU affinity";
  }
}

}  // namespace lwml
#endif  // NO_LWML

}  // namespace caffe
