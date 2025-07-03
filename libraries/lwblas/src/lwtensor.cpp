#include <mutex>

#include <lwda_runtime.h>

extern "C"
{
#include <lwtensor.h>
}

#include <lwtensor/internal/export.h>
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/typesEx.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/utilEx.h>
#include <lwtensor/internal/context.h>
#include <lwtensor/internal/lwblasLtHandles.h>
#include <lwtensor/internal/defines.h>

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
static std::once_flag globalLwblasLtInitControl[LWTENSOR_NAMESPACE::kMaxNumDevices];

lwblasLtHandle_t LWTENSOR_NAMESPACE::globalLwblasLtHandles[LWTENSOR_NAMESPACE::kMaxNumDevices];
bool LWTENSOR_NAMESPACE::globalLwblasLtHandleIsInitialized[LWTENSOR_NAMESPACE::kMaxNumDevices] = {0};

/**
 * \brief Initialize a lwblasLt handle for the lwrrentDevice.
 */
static void initLwblasLtHandle(int deviceId)
{
    using namespace LWTENSOR_NAMESPACE;
    // This function may not return anything, so we
    // have to work around to get error codes out.

    auto lwblasRet = lwblasLtCreate(&globalLwblasLtHandles[deviceId]);
    if (lwblasRet != LWBLAS_STATUS_SUCCESS)
    {
        return;
    }

    globalLwblasLtHandleIsInitialized[deviceId] = 1;
}
#endif

extern "C" EXPORT_SYMBOL
lwtensorStatus_t lwtensorInit(lwtensorHandle_t* const handle)
{
   using namespace LWTENSOR_NAMESPACE;
   if (handle == nullptr)
   {
       RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
   }

   auto ctx = reinterpret_cast<Context*>(handle);
   if (ctx == nullptr)
   {
       RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
   }
   ctx->unsetInitialized();

   /* Initialize LWCA context */
   auto lwdaErr = lwdaFree(nullptr);
   if (lwdaErr != lwdaSuccess)
   {
       const auto errorMsg = lwdaGetErrorString(lwdaErr);
       ctx->logError(LWTENSOR_STATUS_LWDA_ERROR, errorMsg);
   }

   int lwrrentDevice = 0;
   lwdaErr = lwdaGetDevice(&lwrrentDevice);
   if (lwdaErr != lwdaSuccess)
   {
       const auto errorMsg = lwdaGetErrorString(lwdaErr);
       ctx->logError(LWTENSOR_STATUS_LWDA_ERROR, errorMsg);
   }

   if (lwrrentDevice >= static_cast<int>(kMaxNumDevices))
   {
       RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_LWDA_ERROR, "current GPU id exceeds the supported maximum (please let us know if you run into this)."));
   }

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
   std::call_once(globalLwblasLtInitControl[lwrrentDevice], &initLwblasLtHandle, lwrrentDevice);
   if (globalLwblasLtHandleIsInitialized[lwrrentDevice] <= 0)
   {
       RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_INTERNAL_ERROR, "Internal error"));
   }
#endif

   // Do something with the context
   ctx->init(lwrrentDevice);

   return LWTENSOR_STATUS_SUCCESS;
}

lwtensorStatus_t lwtensorInitTensorDescriptor(const lwtensorHandle_t* handle,
                                              lwtensorTensorDescriptor_t* desc_,
                                              const uint32_t numModes,
                                              const int64_t * const extent,
                                              const int64_t * const stride,
                                              const lwdaDataType_t dataType,
                                              const lwtensorOperator_t op,
                                              const uint32_t vectorWidth,
                                              const uint32_t vectorModeIndex)
{
    using namespace LWTENSOR_NAMESPACE;

    auto ctx = reinterpret_cast<const Context*>(handle);
    if (ctx == nullptr) 
    {
        RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
    }
    if (!ctx->isInitialized())
    {
        RETURN_STATUS(LWTENSOR_STATUS_NOT_INITIALIZED);
    }

    if (desc_ == nullptr)
    {
        RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Descriptor must not be nullptr."));
    }
    auto descInternal = reinterpret_cast<TensorDescriptor*>(desc_);
    descInternal->unsetInitialized();

    if ((numModes > 0U) && (extent == nullptr))
    {
        RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Extent must not be nullptr."));
    }

    if( !isValidLwdaDataType(dataType) )
    {
        RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Data type is invalid."));
    }

    if( (! ctx->supportsBF16andTF32()) && (dataType == LWDA_R_16BF))
    {
        RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Data type is invalid."));
    }

    for (uint32_t i = 0U; i < numModes; ++ i)
    {
        if ((extent[i] <= 0) ||
                (extent[i] > static_cast<int64_t>(std::numeric_limits<stride_type>::max())))
        {
            RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Extent is too large to fit into int32_t; please request 64bit version."));
        }
        if (stride != nullptr)
        {
            if( (stride[i] <= 0) ||
                    (stride[i] > static_cast<int64_t>(std::numeric_limits<stride_type>::max())))
            {
                RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Stride is too large to fit into int32_t; please request 64bit version."));
            }
        }
    }

    std::vector<extent_type> extentInternal_(numModes);
    std::vector<stride_type> strideInternal_(numModes);
    int64_t maxDisplacement = 0;

    if (stride != nullptr)
    {
        for (uint32_t i = 0U; i < numModes; ++ i)
        {
            extentInternal_[i] = extent[i];
            strideInternal_[i] = stride[i];
            maxDisplacement += static_cast<int64_t>(extentInternal_[i] - 1) * strideInternal_[i];
        }
    }
    else
    {
        int64_t totalStride = static_cast<int64_t>(vectorWidth);
        for (uint32_t i = 0U; i < numModes; ++ i)
        {
            extentInternal_[i] = extent[i];
            strideInternal_[i] = totalStride;
            totalStride *= static_cast<int64_t>(extent[i]);
            maxDisplacement += static_cast<int64_t>(extentInternal_[i] - 1) * strideInternal_[i];
        }
    }

    if (maxDisplacement > static_cast<int64_t>(std::numeric_limits<stride_type>::max()))
    {
        RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Stride is too large to fit into int32_t; please request 64bit version."));
    }

    HANDLE_ERROR(descInternal->init( ctx,
                                                   numModes,
                                                   extentInternal_.data(),
                                                   strideInternal_.data(),
                                                   dataType, op, 1U, 0U));

    /* Set the vector width and mode. */
    HANDLE_ERROR(descInternal->setVectorization( vectorWidth, vectorModeIndex ));

    bool ilwalidStride = false;

    if (descInternal->isVectorized())
    {
        /* Check if all strides are multiple of the vector width? */
        for (uint32_t i = 0U; i < descInternal->getNumModes(); i ++)
        {
            if ((static_cast<uint32_t>(descInternal->getStride(i)) % descInternal->getVectorWidth()) != 0U)
            {
                ilwalidStride = true;
            }
        }
    }

    if (ilwalidStride)
    {
        RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
    }
    return LWTENSOR_STATUS_SUCCESS;
}

extern "C" EXPORT_SYMBOL
lwtensorStatus_t lwtensorInitTensorDescriptor(
                                              const lwtensorHandle_t* handle,
                                              lwtensorTensorDescriptor_t* desc,
                                              const uint32_t numModes,
                                              const int64_t * const extent,
                                              const int64_t * const stride,
                                              const lwdaDataType_t dataType,
                                              const lwtensorOperator_t op)
{
    return lwtensorInitTensorDescriptor( handle, desc, numModes, extent, stride, dataType, op, 1U, 0U);
}

extern "C" EXPORT_INTERNAL_SYMBOL
lwtensorStatus_t lwtensorContractionDescriptorInfo(const lwtensorHandle_t* handle,
                                                   const lwtensorContractionDescriptor_t* desc_,
                                                   char* dst, int sz)
{
    using namespace LWTENSOR_NAMESPACE;
    auto ctx = reinterpret_cast<const Context*>(handle);
    if (ctx == nullptr)
    {
        RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
    }

    const ContractionDescriptor * const descInternal = reinterpret_cast<const ContractionDescriptor*>(desc_);
    if(descInternal == nullptr || !descInternal->isInitialized() )
        return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Descriptor invalid");
    descInternal->info(dst, sz);
    return LWTENSOR_STATUS_SUCCESS;
}

extern "C" EXPORT_SYMBOL
const char *lwtensorGetErrorString(const lwtensorStatus_t error)
{
    if (error == LWTENSOR_STATUS_SUCCESS)
    {
        return "LWTENSOR_STATUS_SUCCESS";
    }
    else if (error == LWTENSOR_STATUS_NOT_INITIALIZED)
    {
        return "LWTENSOR_STATUS_NOT_INITIALIZED";
    }
    else if (error == LWTENSOR_STATUS_ALLOC_FAILED)
    {
        return "LWTENSOR_STATUS_ALLOC_FAILED";
    }
    else if (error == LWTENSOR_STATUS_ILWALID_VALUE)
    {
        return "LWTENSOR_STATUS_ILWALID_VALUE";
    }
    else if (error == LWTENSOR_STATUS_ARCH_MISMATCH)
    {
        return "LWTENSOR_STATUS_ARCH_MISMATCH";
    }
    else if (error == LWTENSOR_STATUS_MAPPING_ERROR)
    {
        return "LWTENSOR_STATUS_MAPPING_ERROR";
    }
    else if (error == LWTENSOR_STATUS_EXELWTION_FAILED)
    {
        return "LWTENSOR_STATUS_EXELWTION_FAILED";
    }
    else if (error == LWTENSOR_STATUS_NOT_SUPPORTED)
    {
        return "LWTENSOR_STATUS_NOT_SUPPORTED";
    }
    else if (error == LWTENSOR_STATUS_LICENSE_ERROR)
    {
        return "LWTENSOR_STATUS_LICENSE_ERROR";
    }
    else if (error == LWTENSOR_STATUS_LWBLAS_ERROR)
    {
        return "LWTENSOR_STATUS_LWBLAS_ERROR";
    }
    else if (error == LWTENSOR_STATUS_LWDA_ERROR)
    {
        return "LWTENSOR_STATUS_LWDA_ERROR";
    }
    else if (error == LWTENSOR_STATUS_INTERNAL_ERROR)
    {
        return "LWTENSOR_STATUS_INTERNAL_ERROR";
    }
    else if (error == LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
    {
        return "LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    }
    else if (error == LWTENSOR_STATUS_INSUFFICIENT_DRIVER)
    {
        return "LWTENSOR_STATUS_INSUFFICIENT_DRIVER";
    }
    else if (error == LWTENSOR_STATUS_IO_ERROR)
    {
        return "LWTENSOR_STATUS_IO_ERROR";
    }
    else
    {
        return "<unknown>";
    }
}

extern "C" EXPORT_SYMBOL
size_t lwtensorGetVersion()
{
    return static_cast<size_t>(LWTENSOR_MAJOR * 10000 + LWTENSOR_MINOR * 100 + LWTENSOR_PATCH);
}

extern "C" EXPORT_SYMBOL
size_t lwtensorGetLwdartVersion()
{
    return LWDART_VERSION;
}
