#pragma once

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11

#include <lwblasLt.h>

#include <lwtensor/internal/context.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{

/**
 * \brief global lwtblasLt handles, one per device.
 * Initialized once with handle creation, immutable and hence thread-safe afterwards.
 */
extern lwblasLtHandle_t globalLwblasLtHandles[kMaxNumDevices];

/**
 * \brief global lwtblasLt flags, one per device.
 * Initialized once and note whether a given device has an initialized lwblasLt handle.
 */
extern bool globalLwblasLtHandleIsInitialized[kMaxNumDevices];

}
#endif
