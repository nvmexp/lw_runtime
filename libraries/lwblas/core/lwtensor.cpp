#include <lwda_runtime.h>
extern "C"
{
#include <lwtensor.h>
}
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/util.h>

extern "C"
lwtensorStatus_t lwtensorInit(lwtensorHandle_t* const handle)
{
   if (handle == nullptr)
   {
       return lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE;
   }

   /* Initialize LWCA context */
   if (lwdaFree(nullptr) != lwdaSuccess)
   {
       return lwtensorStatus_t::LWTENSOR_STATUS_LWDA_ERROR;
   }
   auto ctx = reinterpret_cast<LWTENSOR_NAMESPACE::lwtensorContext_t* const>(handle);
   if (ctx == nullptr)
   {
       return lwtensorStatus_t::LWTENSOR_STATUS_ALLOC_FAILED;
   }

   // Do something with the context
   ctx->initContext();

   return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
}

extern "C"
lwtensorStatus_t lwtensorInitTensorDescriptor( const lwtensorHandle_t* handle,
          lwtensorTensorDescriptor_t* const desc_,
          const uint32_t numModes,
          const int64_t * const extent,
          const int64_t * const stride,
          const lwdaDataType_t dataType,
          const lwtensorOperator_t op,
          const uint32_t vectorWidth,
          const uint32_t vectorModeIndex )
{
    using LWTENSOR_NAMESPACE::lwtensorContext_t;
    using LWTENSOR_NAMESPACE::lwtensorTensorDescriptor;
    using LWTENSOR_NAMESPACE::stride_type;
    using LWTENSOR_NAMESPACE::extent_type;

    auto ctx = reinterpret_cast<const lwtensorContext_t*>(handle);
    if (ctx == nullptr || !ctx->isInitialized())
        return LWTENSOR_NAMESPACE::handleError(lwtensorStatus_t::LWTENSOR_STATUS_NOT_INITIALIZED, "Handle must be initialized.");

    if ((desc_ == nullptr) || ((numModes > 0U) && (extent == nullptr)))
    {
        return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE, "Descriptor and extent must be allocated.");
    }

    for (uint32_t i = 0U; i < numModes; ++ i)
    {
        if ((extent[i] <= 0) ||
                (extent[i] > static_cast<const int64_t>(std::numeric_limits<stride_type>::max())))
        {
            return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED, "Extent is too large to fit into int32_t; please request 64bit version.");
        }
        if (stride != nullptr)
        {
            if( (stride[i] <= 0) ||
                    (stride[i] > static_cast<const int64_t>(std::numeric_limits<stride_type>::max())))
            {
                return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED, "Stride is too large to fit into int32_t; please request 64bit version.");
            }
        }
    }

    auto descInternal = reinterpret_cast<lwtensorTensorDescriptor*>(desc_);


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

    if (maxDisplacement > static_cast<const int64_t>(std::numeric_limits<stride_type>::max()))
    {
        return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED, "Stride is too large to fit into int32_t; please request 64bit version.");
    }

    auto err = descInternal->initTensorDescriptor( ctx,
                                                   numModes,
                                                   extentInternal_.data(),
                                                   strideInternal_.data(),
                                                   dataType, op, 1U, 0U);
    if (err != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS)
    {
       return err;
    }        

    /* Set the vector width and mode. */
    err = descInternal->setVectorization( vectorWidth, vectorModeIndex );
    if ( err != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
    {
        return err;
    }

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
        return lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE;
    }
    else
    {
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }
}

extern "C"
lwtensorStatus_t lwtensorInitTensorDescriptorSimple(
                                              const lwtensorHandle_t* handle,
                                              lwtensorTensorDescriptor_t* const desc,
                                              const uint32_t numModes,
                                              const int64_t * const extent,
                                              const int64_t * const stride,
                                              const lwdaDataType_t dataType,
                                              const lwtensorOperator_t op)
{
    return lwtensorInitTensorDescriptor( handle, desc, numModes, extent, stride, dataType, op, 1U, 0U);
}


extern "C"
lwtensorStatus_t lwtensorSetTensorDescriptorVectorization(
    lwtensorTensorDescriptor_t* desc_,
    const uint32_t vectorWidth,
    const uint32_t vectorMode,
    const uint32_t vectorOffset,
    const lwtensorPaddingType_t zeroPadding)
{
    using LWTENSOR_NAMESPACE::lwtensorTensorDescriptor;
    try
    {
        lwtensorTensorDescriptor * const descInternal = reinterpret_cast<lwtensorTensorDescriptor* const>(desc_);
        if ( descInternal == nullptr )
        {
            return lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE;
        }
        /* Set the vector width and mode. */
        auto err = descInternal->setVectorization( vectorWidth, vectorMode );
        if ( err != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
        {
            return err;
        }
        //TODO check test of desc & add unit test Chenhan
        /* If vectorized, then set the offset and padding and check for the strides. */
        if( descInternal->isVectorized() )
        {
            /* Check if all strides are multiple of the vector width? */
            for (uint32_t i = 0U; i < descInternal->getNumModes(); i ++)
            {
                if ((static_cast<uint32_t>(descInternal->getStride( i )) % descInternal->getVectorWidth()) != 0U)
                {
                    return lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE;
                }
            }
            /* Set the vector offset. */
            err = descInternal->setVectorOffset( vectorOffset );
            if (err != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS)
            {
                return err;
            }
            /* Whether to pad the tensor? */
            return descInternal->setZeroPadding(zeroPadding == LWTENSOR_PADDING_ZERO);
        }
        else
        {
            /* Return with no error. */
            return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
        }
    }
    catch (const std::exception & e)
    {
        return LWTENSOR_NAMESPACE::handleException(e);
    }
}

extern "C"
const char *lwtensorGetErrorString(const lwtensorStatus_t error)
{
    if (error == lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_NOT_INITIALIZED)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_NOT_INITIALIZED";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_ALLOC_FAILED)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_ALLOC_FAILED";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_ARCH_MISMATCH)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_ARCH_MISMATCH";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_MAPPING_ERROR)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_MAPPING_ERROR";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_EXELWTION_FAILED)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_EXELWTION_FAILED";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_LICENSE_ERROR)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_LICENSE_ERROR";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_LWBLAS_ERROR)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_LWBLAS_ERROR";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_LWDA_ERROR)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_LWDA_ERROR";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    }
    else if (error == lwtensorStatus_t::LWTENSOR_STATUS_INSUFFICIENT_DRIVER)
    {
        return "lwtensorStatus_t::LWTENSOR_STATUS_INSUFFICIENT_DRIVER";
    }
    else
    {
        return "<unknown>";
    }
}

extern "C"
size_t lwtensorGetVersion()
{
    return static_cast<size_t>(LWTENSOR_MAJOR * 10000 + LWTENSOR_MINOR * 100 + LWTENSOR_PATCH);
}

extern "C"
size_t lwtensorGetLwdartVersion()
{
    return LWDART_VERSION;
}
