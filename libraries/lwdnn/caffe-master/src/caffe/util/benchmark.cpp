#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

Timer::Timer(bool enforce_cpu)
    : initted_(false),
      running_(false),
      has_run_at_least_once_(false),
      start_gpu_(nullptr),
      stop_gpu_(nullptr),
      device_(-1),
      use_gpu_(!enforce_cpu && Caffe::mode() == Caffe::GPU) {
  Init();
}

Timer::~Timer() {
  if (use_gpu_) {
    int lwrrent_device;  // Just to check LWCA status:
    lwdaError_t status = lwdaGetDevice(&lwrrent_device);
    // Preventing crash while Caffe shutting down.
    if (status != lwdaErrorLwdartUnloading) {
      if (start_gpu_ != nullptr) {
        LWDA_CHECK(lwdaEventDestroy(start_gpu_));
      }
      if (stop_gpu_ != nullptr) {
        LWDA_CHECK(lwdaEventDestroy(stop_gpu_));
      }
    }
  }
}

void Timer::Start() {
  if (!running()) {
    if (use_gpu_) {
      CHECK_EQ(device_, Caffe::lwrrent_device());
      LWDA_CHECK(lwdaEventRecord(start_gpu_, 0));
    } else {
      start_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
    if (use_gpu_) {
      CHECK_EQ(device_, Caffe::lwrrent_device());
      LWDA_CHECK(lwdaEventRecord(stop_gpu_, 0));
      LWDA_CHECK(lwdaEventSynchronize(stop_gpu_));
    } else {
      stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = false;
  }
}

float Timer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (use_gpu_) {
    CHECK_EQ(device_, Caffe::lwrrent_device());
    LWDA_CHECK(lwdaEventElapsedTime(&elapsed_milliseconds_, start_gpu_, stop_gpu_));
    // Lwca only measure milliseconds
    elapsed_microseconds_ = elapsed_milliseconds_ * 1000;
  } else {
    elapsed_microseconds_ = (stop_cpu_ - start_cpu_).total_microseconds();
  }
  return elapsed_microseconds_;
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (use_gpu_) {
    CHECK_EQ(device_, Caffe::lwrrent_device());
    LWDA_CHECK(lwdaEventElapsedTime(&elapsed_milliseconds_, start_gpu_, stop_gpu_));
  } else {
    elapsed_milliseconds_ = (stop_cpu_ - start_cpu_).total_milliseconds();
  }
  return elapsed_milliseconds_;
}

float Timer::Seconds() {
  return MilliSeconds() / 1000.;
}

void Timer::Init() {
  if (!initted()) {
    if (use_gpu_) {
      int lwrrent_device = Caffe::lwrrent_device();
      if (device_ < 0) {
        device_ = lwrrent_device;
      } else {
        CHECK_EQ(device_, lwrrent_device);
      }
      LWDA_CHECK(lwdaEventCreate(&start_gpu_));
      LWDA_CHECK(lwdaEventCreate(&stop_gpu_));
    } else {
      start_gpu_ = nullptr;
      stop_gpu_ = nullptr;
    }
    initted_ = true;
  }
}

CPUTimer::CPUTimer() {
  this->initted_ = true;
  this->running_ = false;
  this->has_run_at_least_once_ = false;
}

void CPUTimer::Start() {
  if (!running()) {
    this->start_cpu_ = boost::posix_time::microsec_clock::local_time();
    this->running_ = true;
    this->has_run_at_least_once_ = true;
  }
}

void CPUTimer::Stop() {
  if (running()) {
    this->stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    this->running_ = false;
  }
}

float CPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  this->elapsed_milliseconds_ = (this->stop_cpu_ -
                                this->start_cpu_).total_milliseconds();
  return this->elapsed_milliseconds_;
}

float CPUTimer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  this->elapsed_microseconds_ = (this->stop_cpu_ -
                                this->start_cpu_).total_microseconds();
  return this->elapsed_microseconds_;
}

}  // namespace caffe
