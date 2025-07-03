/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_timer.h"

#include <atomic>
#include <numeric>

#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "lwca/extras/LWPTI/include/lwpti.h"
#include "lwda_util.h"

namespace lwidia_libs_test {
namespace {

class KernelNopTimer : public KernelTimer {
 public:
  KernelNopTimer() = default;
  void StartTiming() override {}
  void StopTiming() override {}
  double GetTime() override { return 0.0; }
  void ResetTime() override {}
};

// Measures GPU kernel runtime using the LWPTI activity module.
class KernelDurationTimer : public KernelTimer {
 public:
  KernelDurationTimer() {
    LWcontext context = nullptr;
    CHECK_OK_STATUS(GetStatus(lwCtxGetLwrrent(&context)));
    CHECK_NE(context, nullptr) << "No LWCA context bound.";
    uint32_t context_id = 0;
    CHECK_OK_STATUS(GetStatus(lwptiGetContextId(context, &context_id)));
    CHECK_GT(context_id, 0) << "Unexpected zero context id.";
    if (std::atomic_exchange(&context_id_, context_id)) {
      LOG(FATAL) << "Only one KernelDurationTimer can be instanced at a time.";
    }
    CHECK_OK_STATUS(GetStatus(lwptiActivityRegisterCallbacks(
        LwptiBufferRequested, LwptiBufferCompleted)));
    ResetTime();
  }

  ~KernelDurationTimer() override {
    StopTiming();
    lwptiFinalize();
    context_id_ = 0;
  }

  void StartTiming() override {
    CHECK_OK_STATUS(GetStatus(lwptiActivityEnable(LWPTI_ACTIVITY_KIND_KERNEL)));
  }

  void StopTiming() override {
    CHECK_OK_STATUS(GetStatus(lwCtxSynchronize()));
    CHECK_OK_STATUS(
        GetStatus(lwptiActivityDisable(LWPTI_ACTIVITY_KIND_KERNEL)));
  }

  double GetTime() override {
    CHECK_OK_STATUS(GetStatus(lwptiActivityFlushAll(/*flag=*/0)));
    return kernel_durations_ns_ * 1e-9;
  }

  void ResetTime() override { kernel_durations_ns_ = 0; }

 private:
  static void LWPTIAPI LwptiBufferRequested(uint8_t** buffer, size_t* size,
                                            size_t* max_num_records) {
    *size = 1024 * 1024;
    *buffer = reinterpret_cast<uint8_t*>(malloc(*size));
    *max_num_records = 0;
  }

  static void LWPTIAPI LwptiBufferCompleted(LWcontext, uint32_t,
                                            uint8_t* buffer, size_t /*size*/,
                                            size_t valid_size) {
    uint32_t context_id = context_id_;
    uint64_t kernel_duration_ns = 0;
    LWpti_Activity* record = nullptr;
    while (LWPTI_SUCCESS ==
           lwptiActivityGetNextRecord(buffer, valid_size, &record)) {
      if (record->kind != LWPTI_ACTIVITY_KIND_KERNEL) continue;
      LWpti_ActivityKernel3* kernel =
          reinterpret_cast<LWpti_ActivityKernel3*>(record);
      if (kernel->contextId == context_id) {
        kernel_duration_ns += kernel->end - kernel->start;
      }
    }
    kernel_durations_ns_ += kernel_duration_ns;

    size_t dropped;
    CHECK_OK_STATUS(
        GetStatus(lwptiActivityGetNumDroppedRecords(/*context=*/nullptr,
                                                    /*streamId=*/0, &dropped)));
    CHECK(!dropped) << "Dropped " << dropped << " activity records";

    free(buffer);
  }

  static std::atomic<uint32_t> context_id_;
  static std::atomic<uint64_t> kernel_durations_ns_;
};

std::atomic<uint32_t> KernelDurationTimer::context_id_{};
std::atomic<uint64_t> KernelDurationTimer::kernel_durations_ns_{};

// Measures GPU kernel cycles using the LWPTI callback module.
class KernelCyclesTimer : public KernelTimer {
 public:
  KernelCyclesTimer() : active_cycles_(0) {
    LWdevice device = 0;
    CHECK_OK_STATUS(GetStatus(lwCtxGetDevice(&device)));
    LWcontext context = nullptr;
    CHECK_OK_STATUS(GetStatus(lwCtxGetLwrrent(&context)));
    CHECK_OK_STATUS(GetStatus(lwptiEventGetIdFromName(
        device, "elapsed_cycles_sm", &lwpti_event_id_)));
    CHECK_OK_STATUS(GetStatus(
        lwptiEventGroupCreate(context, &lwpti_event_group_, /*flags=*/0)));
    CHECK_OK_STATUS(GetStatus(
        lwptiEventGroupAddEvent(lwpti_event_group_, lwpti_event_id_)));
    CHECK_OK_STATUS(GetStatus(
        lwptiSubscribe(&lwpti_subscriber_, LwptiEventCallback, this)));
    size_t domain_count_size = sizeof(domain_count_);
    CHECK_OK_STATUS(GetStatus(lwptiEventGroupGetAttribute(
        lwpti_event_group_, LWPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
        &domain_count_size, &domain_count_)));
    int clock_rate_khz;
    CHECK_OK_STATUS(GetStatus(lwDeviceGetAttribute(
        &clock_rate_khz, LW_DEVICE_ATTRIBUTE_CLOCK_RATE, device)));
    normalized_clock_period_sec_ = 1e-3 / (clock_rate_khz * domain_count_);
  }

  ~KernelCyclesTimer() override {
    StopTiming();
    CHECK_OK_STATUS(GetStatus(lwptiUnsubscribe(lwpti_subscriber_)));
    CHECK_OK_STATUS(GetStatus(
        lwptiEventGroupRemoveEvent(lwpti_event_group_, lwpti_event_id_)));
    CHECK_OK_STATUS(GetStatus(lwptiEventGroupDestroy(lwpti_event_group_)));
  }

  void StartTiming() override { EnableCallback(true); }

  void StopTiming() override { EnableCallback(false); }

  double GetTime() override {
    return active_cycles_ * normalized_clock_period_sec_;
  }

  void ResetTime() override { active_cycles_ = 0; }

 private:
  static void LWPTIAPI LwptiEventCallback(void* userdata,
                                          LWpti_CallbackDomain domain,
                                          LWpti_CallbackId cbid,
                                          const void* cb_data) {
    CHECK_EQ(domain, LWPTI_CB_DOMAIN_DRIVER_API);
    CHECK_EQ(cbid, LWPTI_DRIVER_TRACE_CBID_lwLaunchKernel);
    static_cast<KernelCyclesTimer*>(userdata)->LwptiEventCallback(
        *static_cast<const LWpti_CallbackData*>(cb_data));
  }

  void LwptiEventCallback(const LWpti_CallbackData& cb_data) {
    // On entry, enable the event group.
    if (cb_data.callbackSite == LWPTI_API_ENTER) {
      CHECK_OK_STATUS(GetStatus(lwCtxSynchronize()));
      CHECK_OK_STATUS(GetStatus(lwptiSetEventCollectionMode(
          cb_data.context, LWPTI_EVENT_COLLECTION_MODE_KERNEL)));
      CHECK_OK_STATUS(GetStatus(lwptiEventGroupEnable(lwpti_event_group_)));
    }

    // On exit, read and accumulate event values.
    if (cb_data.callbackSite == LWPTI_API_EXIT) {
      CHECK_OK_STATUS(GetStatus(lwCtxSynchronize()));
      std::vector<uint64_t> domain_cycles(domain_count_);
      size_t domain_cycles_size = domain_count_ * sizeof(uint64_t);
      CHECK_OK_STATUS(GetStatus(lwptiEventGroupReadEvent(
          lwpti_event_group_, LWPTI_EVENT_READ_FLAG_NONE, lwpti_event_id_,
          &domain_cycles_size, domain_cycles.data())));
      uint64_t total_cycles = std::accumulate(domain_cycles.begin(),
                                              domain_cycles.end(), uint64_t{0});
      CHECK(total_cycles > 0) << "LWPTI failed to report active cycles.";
      active_cycles_ += total_cycles;
      CHECK_OK_STATUS(GetStatus(lwptiEventGroupDisable(lwpti_event_group_)));
    }
  }

  void EnableCallback(bool enable) {
    CHECK_OK_STATUS(GetStatus(lwptiEnableCallback(
        static_cast<uint32_t>(enable), lwpti_subscriber_,
        LWPTI_CB_DOMAIN_DRIVER_API, LWPTI_DRIVER_TRACE_CBID_lwLaunchKernel)));
  }

  LWpti_SubscriberHandle lwpti_subscriber_;
  LWpti_EventGroup lwpti_event_group_;
  LWpti_EventID lwpti_event_id_;
  // How many event domains report 'elapsed_cycles_sm'.
  uint32_t domain_count_;
  // Duration of an 'elapsed_cycle_sm' event, normalized by domain_count_.
  double normalized_clock_period_sec_;
  // Aclwmulated active cycles across all domains.
  uint64_t active_cycles_;
};

}  // namespace

std::unique_ptr<KernelTimer> KernelTimer::CreateNopTimer() {
  return std::unique_ptr<KernelTimer>(new KernelNopTimer());
}

std::unique_ptr<KernelTimer> KernelTimer::CreateDurationTimer() {
  return std::unique_ptr<KernelTimer>(new KernelDurationTimer());
}

std::unique_ptr<KernelTimer> KernelTimer::CreateCyclesTimer() {
  return std::unique_ptr<KernelTimer>(new KernelCyclesTimer());
}
}  // namespace lwidia_libs_test
