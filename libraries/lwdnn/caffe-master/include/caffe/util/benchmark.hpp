#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/util/device_alternate.hpp"

namespace caffe {

class Timer {
 public:
  explicit Timer(bool enforce_cpu = false);
  virtual ~Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  bool initted() { return initted_; }
  bool running() { return running_; }
  bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
  lwdaEvent_t start_gpu_;
  lwdaEvent_t stop_gpu_;
  boost::posix_time::ptime start_cpu_;
  boost::posix_time::ptime stop_cpu_;
  float elapsed_milliseconds_;
  float elapsed_microseconds_;
  int device_;
  const bool use_gpu_;
};

class CPUTimer : public Timer {
 public:
  explicit CPUTimer();
  ~CPUTimer() override {}
  void Start() override;
  void Stop() override;
  float MilliSeconds() override;
  float MicroSeconds() override;
};

}  // namespace caffe

#endif   // CAFFE_UTIL_BENCHMARK_H_
