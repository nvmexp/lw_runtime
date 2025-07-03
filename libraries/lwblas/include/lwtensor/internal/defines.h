#pragma once

#include <cstdint>
// TODO Fix until LWCA supplies this dATA TYPE
#include <lwda_runtime.h>
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
#include <lwda_bf16.h>
#endif
#include <lwda_runtime_api.h>

#define LWTENSOR_NAMESPACE lwtensor_internal_namespace

const lwdaDataType_t LWDA_R_TF32 = (lwdaDataType_t) 100;
const lwdaDataType_t LWDA_C_TF32 = (lwdaDataType_t) 101;
#if LWTENSOR_LWDA_VERSION_MAJOR < 11
const lwdaDataType_t LWDA_R_16BF = (lwdaDataType_t) 102;
#endif

namespace LWTENSOR_NAMESPACE
{
    static const uint32_t kMaxNumDevices = 16U;
    static const uint32_t kMaxNumModes = 28U;
    static const uint32_t kMaxNumModesExternal = 40U;
    static_assert(kMaxNumModes <= kMaxNumModesExternal, "kMaxNumModes <= LWTENSOR_MAX_MODES_EXTERNAL");

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
#if (LWDART_VERSION < 11000) || \
    (LWDART_VERSION == 11000 && defined(LWDA_VERSION_REVISION) && LWDA_VERSION_REVISION <= 126)
    typedef __bfloat16 BFloat16;
#else
    typedef __lw_bfloat16 BFloat16;
#endif
#endif

}  // namespace LWTENSOR_NAMESPACE
