#pragma once

#include <cstdint>
#include <mutex>
#include <atomic>
#include "Config.h"

#if ENABLE_NETWORK_HOOKS == 1
class NetworkActivityDetector
{
public:
    static std::atomic<uint64_t> m_bytesTransferred;

    virtual ~NetworkActivityDetector();
    // installs all the network hooks and enables them
    void initialize(bool checkIfSocketIsLocal);
    // uninstalls all the network hooks and deinitializes MinHook library
    // only call this function once
    void deinitialize();
    void tick();
    void setEnabled(bool state);
    void resetActivity();

    bool isActivityDetected() const;
    uint64_t bytesTransferred() const;
private:
    static std::mutex m_initializationLock, m_deinitializationLock;
    static bool m_initialized;
    static bool m_deinitialized;

    bool m_active = false;
    bool m_enabled = false;
    uint32_t m_cooldown = 0u;
    uint32_t m_traffic = 0u;
};
#endif
