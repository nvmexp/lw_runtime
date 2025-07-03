#include <unordered_map>
#include <list>
#include <limits>
#include <cassert>

#include <lwtensor/internal/exceptions.h>
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
    lwtensorContext_t::lwtensorContext_t() : numSMs_(80U), isInitialized_(false)
    {
        int32_t lwrrentDeviceId;
        int32_t numSM;
        if (lwdaGetDevice(&lwrrentDeviceId) != lwdaSuccess)
        {
            throw InternalError("lwdaGetDevice() failed.\n");
        }
        if (lwdaDeviceGetAttribute(&numSM, lwdaDevAttrMultiProcessorCount, lwrrentDeviceId) == lwdaSuccess)
        {
            numSMs_ = numSM;
        }
        this->setInitialized();
    }

    lwtensorContext_t::~lwtensorContext_t()
    {
        this->setNotInitialized();
    }

    lwtensorStatus_t lwtensorContext_t::initContext() noexcept
    {
        this->numSMs_ = 80U;
        this->setNotInitialized();
        int32_t lwrrentDeviceId;
        int32_t numSM;

        if (lwdaGetDevice(&lwrrentDeviceId) != lwdaSuccess)
        {
            return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "lwdaGetDevice() failed.");
        }

        if (lwdaDeviceGetAttribute(&numSM, lwdaDevAttrMultiProcessorCount, lwrrentDeviceId) == lwdaSuccess)
        {
            numSMs_ = numSM;
        }

        if(getelw("LWTENSOR_LOGINFO_DBG") != nullptr && atoi(getelw("LWTENSOR_LOGINFO_DBG")) == 1)
            this->handleError_   = static_cast<lwtensorStatus_t (*)(const lwtensorStatus_t, const std::string &&)>(&handleError_log);
        else
            this->handleError_   = static_cast<lwtensorStatus_t (*)(const lwtensorStatus_t, const std::string &&)>(&handleError);

        this->setInitialized();

        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t initElementwiseParameters(
            ElementwiseParameters & param,
            const std::list<mode_type> &mode_order,
            const std::unordered_map<mode_type, extent_type> &extent,
            const std::list<mode_type> &modeA,
            const std::list<mode_type> &modeB,
            const std::list<mode_type> &modeC,
            const std::unordered_map<mode_type, stride_type> &strideA_tmp,
            const std::unordered_map<mode_type, stride_type> &strideB_tmp,
            const std::unordered_map<mode_type, stride_type> &strideC_tmp,
            const VectorInfo &info )
    {
        param.is_vectorized = false;
        param.padding_size = 0;
        param.nmodeC = static_cast<uint32_t>(mode_order.size());

        if( param.nmodeC > ElementwiseParameters::LWTENSOR_MAX_MODES )
        {
            return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "too many modes.");
        }

        for (uint32_t i = 0U; i < param.nmodeC; ++ i)
        {
            param.layoutA.vectorWidthLog2[i] = 0U;
            param.layoutB.vectorWidthLog2[i] = 0U;
            param.layoutC.vectorWidthLog2[i] = 0U;
            param.layoutA.vectorOffset[i] = 0U;
            param.layoutB.vectorOffset[i] = 0U;
            param.layoutC.vectorOffset[i] = 0U;
        }

        const std::unordered_map<mode_type, stride_type> * strideAptr = &strideA_tmp;
        const std::unordered_map<mode_type, stride_type> * strideBptr = &strideB_tmp;
        const std::unordered_map<mode_type, stride_type> * strideCptr = &strideC_tmp;
        std::unordered_map<mode_type, stride_type> strideApacked;
        std::unordered_map<mode_type, stride_type> strideBpacked;
        std::unordered_map<mode_type, stride_type> strideCpacked;
        if ( strideA_tmp.size() == 0U )
        {
            if ( initStride(extent, modeA, strideApacked) != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
            {
                return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Strides of A are invalid." );
            }
            strideAptr = &strideApacked;
        }
        if ( strideB_tmp.size() == 0U )
        {
            if ( initStride(extent, modeB, strideBpacked) != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
            {
                return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Strides of B are invalid." );
            }
            strideBptr = &strideBpacked;
        }
        if ( strideC_tmp.size() == 0U )
        {
            if ( initStride(extent, modeC, strideCpacked) != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
            {
                return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Strides of C are invalid." );
            }
            strideCptr = &strideCpacked;
        }

        /* A3-9-1: do not use basic type int. */
        uint32_t idx = 0U;
        /* Loop over all modes in the order of C's leading, A' leading, then the rest. */
        for ( auto itMode = mode_order.cbegin(); itMode != mode_order.cend(); itMode ++ )
        {
            mode_type mode = *itMode;
            if ( extent.find(mode) == extent.end() )
            {
                return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Extent not found." );
            }
            if ( extent.at( mode ) <= 0 )
            {
                return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Invalid extent." );
            }
            param.extent[ idx ] = extent.at( mode );
            const auto ita = strideAptr->find(mode);
            const auto itb = strideBptr->find(mode);
            const auto itc = strideCptr->find(mode);
            param.layoutA.stride[idx] = (ita != strideAptr->end() ) ? ita->second : 0;
            param.layoutB.stride[idx] = (itb != strideBptr->end() ) ? itb->second : 0;
            param.layoutC.stride[idx] = (itc != strideCptr->end() ) ? itc->second : 0;

            /* Initialize vector widths and offsets for each tensor. */
            param.layoutA.vectorWidthLog2[idx] = 0U;
            param.layoutB.vectorWidthLog2[idx] = 0U;
            param.layoutC.vectorWidthLog2[idx] = 0U;
            param.layoutA.vectorOffset[idx] = 0U;
            param.layoutB.vectorOffset[idx] = 0U;
            param.layoutC.vectorOffset[idx] = 0U;
            /* If the target mode is vectorized in tensorA, then set the width and offset. */
            if (mode == info.getVectorModeA())
            {
                param.is_vectorized = true;
                param.layoutA.vectorWidthLog2[idx] = static_cast<uint8_t>(std::log2(info.getVectorWidthA()));
                param.layoutA.vectorOffset[idx] = info.getVectorOffsetA();
            }
            /* If the target mode is vectorized in tensorB, then set the width and offset. */
            if (mode == info.getVectorModeB())
            {
                param.is_vectorized = true;
                param.layoutB.vectorWidthLog2[idx] = static_cast<uint8_t>(std::log2(info.getVectorWidthB()));
                param.layoutB.vectorOffset[idx] = info.getVectorOffsetB();
            }
            /* If the target mode is vectorized in tensorC, then set the width and offset. */
            if ( mode == info.getVectorModeC() )
            {
                auto vectorWidth = info.getVectorWidthC(); 
                param.is_vectorized = true;
                param.layoutC.vectorWidthLog2[idx] = static_cast<uint8_t>(std::log2(vectorWidth));
                param.layoutC.vectorOffset[idx] = info.getVectorOffsetC();
                /* Compute padding size for tensor C. */
                if (info.useZeroPaddingC() == true)
                {
                    param.padding_size = (param.extent[idx] + static_cast<extent_type>(param.layoutC.vectorOffset[idx])) % vectorWidth;
                    if (param.padding_size != 0)
                    {
                        param.padding_size = vectorWidth - param.padding_size;
                    }
                }
            }
            /* Advance mode index. */
            idx++;
        }

        param.isStrideOne = false;
        const auto strideOneMode = mode_order.front();
        if (((modeA.size() == 0U) || (strideOneMode == modeA.front())) &&
            ((modeB.size() == 0U) || (strideOneMode == modeB.front())) &&
            ((modeC.size() == 0U) || (strideOneMode == modeC.front())))
        {
            param.isStrideOne = true;
        }
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    VectorInfo::VectorInfo(
            const mode_type* const modeA, const lwtensorTensorDescriptor* const descA,
            const mode_type* const modeB, const lwtensorTensorDescriptor* const descB,
            const mode_type* const modeC, const lwtensorTensorDescriptor* const descC ) :
        vectorModeA_(LWTENSOR_ILWALID_MODE),
        vectorModeB_(LWTENSOR_ILWALID_MODE),
        vectorModeC_(LWTENSOR_ILWALID_MODE),
        /* The vector width can be 1, 2, 4, 8, 16, and 32. */
        vectorWidthA_(1U),
        vectorWidthB_(1U),
        vectorWidthC_(1U),
        /** The vector offset should be smaller than the vector width. */
        vectorOffsetA_(0U),
        vectorOffsetB_(0U),
        vectorOffsetC_(0U),
        /** Whether the output tensor is padded with zero. */
        useZeroPaddingC_(false),
        /** Whether lwTensor should take the vectorized path? */
        isVectorized_(false),
        isAaligned_(false),
        isBaligned_(false),
        isCaligned_(false)
    {
        /* If tensor A is vectorized. */
        if ( (descA != nullptr) && (descA->isVectorized()) )
        {
            vectorModeA_ = descA->getVectorMode(modeA);
            vectorWidthA_ = descA->getVectorWidth();
            vectorOffsetA_ = descA->getVectorOffset();
            isVectorized_ = true;
        }
        /* If tensor B is vectorized. */
        if ( (descB != nullptr) && (descB->isVectorized()) )
        {
            vectorModeB_ = descB->getVectorMode(modeB);
            vectorWidthB_ = descB->getVectorWidth();
            vectorOffsetB_ = descB->getVectorOffset();
            isVectorized_ = true;
        }
        /* If tensor C is vectorized. */
        if ( (descC != nullptr) && (descC->isVectorized()) )
        {
            vectorModeC_ = descC->getVectorMode(modeC);
            vectorWidthC_ = descC->getVectorWidth();
            vectorOffsetC_ = descC->getVectorOffset();
            /* Whether the output tensor is padded? */
            if ( descC->getZeroPadding() )
            {
                useZeroPaddingC_ = true;
            }
            isVectorized_ = true;
        }
    }

    lwtensorStatus_t lwtensorTensorDescriptor::setVectorization(
            const uint8_t vectorWidth,
            const uint32_t vectorModeIndex) noexcept
    {
        if ((((((vectorWidth == 1U ) ||
             (vectorWidth == 2U )) ||
            (vectorWidth == 4U )) ||
            (vectorWidth == 8U )) ||
            (vectorWidth == 16U)) ||
            (vectorWidth == 32U))
        {
            /* Do nothing. */
        }
        else
        {
            return lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE;
        }

        if ((vectorWidth > 1U) && (vectorModeIndex >= numModes_))
        {
            return lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE;
        }

        this->vectorWidth_ = vectorWidth;

        this->vectorModeIndex_ = (vectorWidth > 1U) ? vectorModeIndex : 999U;

        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorTensorDescriptor::lwtensorTensorDescriptor(
            const uint32_t numModes,
            const extent_type * const extent,
            const stride_type * const stride,
            const lwdaDataType_t dataType,
            const lwtensorOperator_t op,
            const uint8_t vectorWidth,
            const uint32_t vectorModeIndex) : 
        numModes_(numModes), 
        dataType_(dataType), 
        op_(op), 
        vectorWidth_(1U),
        vectorModeIndex_(0U),
        vectorOffset_(0U), 
        zeroPadding_(true)
    {
        if ( (numModes > static_cast<uint32_t>(LWTENSOR_MAX_MODES_EXTERNAL)) )
        {
            throw NotSupported( "Number of modes is invalid.\n" );
        }
        if( this->setVectorization(vectorWidth, vectorModeIndex) != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
        {
            throw NotSupported( "Vectorization is invalid.\n" );
        }

        stride_type totalStride = 1;
        for ( uint32_t i = 0U; i < numModes_; ++ i )
        {
            extent_[ i ] = extent[ i ];
            if ( extent[ i ] <= 0 )
            {
                throw NotSupported( "extent must be > 0.\n" );
            }
            if ( stride != nullptr )
            {
                /* Check if provided strides are valid. */
                stride_[ i ] = stride[ i ];
                if ( stride_[ i ] <= 0 )
                {
                    throw NotSupported( "stride must be > 0.\n" );
                }
                /* Check if provided stides are multiples of the vector width. */
                if ( (vectorWidth_ > 0U) && ( (stride_[ i ] % static_cast<stride_type>(vectorWidth_)) != 0 ) )
                {
                    throw NotSupported( "all strides must be the multiple of the vector width.\n" );
                }
            }
            else
            {
                /* If strides are not provided, then ...  */
                stride_[ i ] = totalStride;
                totalStride *= extent_[ i ];
            }
        }

        if( (dataType_ != LWDA_R_8I)  &&
            (dataType_ != LWDA_R_8U)  &&
            (dataType_ != LWDA_R_32U) &&
            (dataType_ != LWDA_R_32I) &&
            (dataType_ != LWDA_R_16F) &&
            (dataType_ != LWDA_R_32F) &&
            (dataType_ != LWDA_C_16F) &&
            (dataType_ != LWDA_R_64F) &&
            (dataType_ != LWDA_C_32F) &&
            (dataType_ != LWDA_C_64F) )
        {
            throw NotSupported("Datatype is not yet supported.\n");
        }
    }

    lwtensorTensorDescriptor::lwtensorTensorDescriptor(
            const uint32_t numModes,
            const extent_type * const extent,
            const stride_type * const stride,
            const lwdaDataType_t dataType,
            const lwtensorOperator_t op,
            const uint8_t vectorWidth,
            const uint32_t vectorModeIndex,
            const uint8_t  vectorOffset,
            const bool zeroPadding) :
    lwtensorTensorDescriptor(numModes, extent, stride, dataType, op, vectorWidth, vectorModeIndex)
    {
        if (vectorOffset >= vectorWidth)
        {
            throw NotSupported("Vector offset is invalid.\n");
        }
        vectorOffset_ = vectorOffset;
        zeroPadding_ = zeroPadding;
    }

    lwtensorTensorDescriptor::lwtensorTensorDescriptor(const lwtensorTensorDescriptor &other) :
        numModes_(other.numModes_),
        dataType_(other.dataType_),
        extent_(other.extent_),
        stride_(other.stride_),
        op_(other.op_),
        vectorWidth_(other.vectorWidth_),
        vectorModeIndex_(other.vectorModeIndex_),
        vectorOffset_(other.vectorOffset_),
        zeroPadding_(other.zeroPadding_)
    {
        for(uint32_t i=0U; i < numModes_; ++i)
        {
            extent_[i] = other.extent_[i];
            stride_[i] = other.stride_[i];
        }
    }

    lwtensorTensorDescriptor& lwtensorTensorDescriptor::operator=(const lwtensorTensorDescriptor &other)
    {
        numModes_       = other.numModes_;
        dataType_       = other.dataType_;
        extent_         = other.extent_;
        stride_         = other.stride_;
        op_             = other.op_;
        vectorWidth_    = other.vectorWidth_;
        vectorModeIndex_= other.vectorModeIndex_;
        vectorOffset_   = other.vectorOffset_;
        zeroPadding_    = other.zeroPadding_;

        for(uint32_t i=0U; i < numModes_; ++i)
        {
            extent_[i] = other.extent_[i];
            stride_[i] = other.stride_[i];
        }

        return *this;
    }

    lwtensorStatus_t lwtensorTensorDescriptor::initTensorDescriptor(
                                        const lwtensorContext_t* ctx,
                                        const uint32_t numModes,
                                        const extent_type * const extent,
                                        const stride_type * const stride,
                                        const lwdaDataType_t dataType,
                                        const lwtensorOperator_t op,
                                        const uint8_t vectorWidth,
                                        const uint32_t vectorModeIndex) noexcept 
    {
        numModes_        = numModes;
        dataType_        = dataType;
        op_              = op;
        vectorWidth_     = 1U;
        vectorModeIndex_ = 0U;
        vectorOffset_    = 0U;
        zeroPadding_     = true;

        if ( (numModes > static_cast<uint32_t>(LWTENSOR_MAX_MODES_EXTERNAL)) )
        {
            return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED, "Too many modes.");
        }

        auto err = this->setVectorization(vectorWidth, vectorModeIndex);
        if( err != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
        {
            return ctx->logError(err, "Vectorization is invalid.");
        }

        stride_type totalStride = 1;
        for ( uint32_t i = 0U; i < numModes_; ++ i )
        {
            extent_[ i ] = extent[ i ];
            if ( extent[ i ] <= 0 )
            {
                return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE, "Extent must be > 0.");
            }
            if ( stride != nullptr )
            {
                /* Check if provided strides are valid. */
                stride_[ i ] = stride[ i ];
                if ( stride_[ i ] <= 0 )
                {  
                    return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE, "Stride must be > 0.");
                }
                /* Check if provided stides are multiples of the vector width. */
                if ( (vectorWidth_ > 0U) && ( (stride_[ i ] % static_cast<stride_type>(vectorWidth_)) != 0 ) )
                {
                    return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE, "All strides must be the multiple of the vector width.");
                }
            }
            else
            {
                /* If strides are not provided, then ...  */
                stride_[ i ] = totalStride;
                totalStride *= extent_[ i ];
            }
        }

        if( (dataType_ != LWDA_R_8I)  &&
            (dataType_ != LWDA_R_8U)  &&
            (dataType_ != LWDA_R_32U) &&
            (dataType_ != LWDA_R_32I) &&
            (dataType_ != LWDA_R_16F) &&
            (dataType_ != LWDA_R_32F) &&
            (dataType_ != LWDA_C_16F) &&
            (dataType_ != LWDA_R_64F) &&
            (dataType_ != LWDA_C_32F) &&
            (dataType_ != LWDA_C_64F) )
        {
            return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED, "Data type not yet supported.");
        }

        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t lwtensorTensorDescriptor::setVectorOffset(const uint8_t offset) noexcept
    {
        if (offset >= vectorWidth_)
        {
            return lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE;
        }
        vectorOffset_ = offset;
        /* Return with no error. */
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t lwtensorTensorDescriptor::setZeroPadding(const bool padding) noexcept
    {
        zeroPadding_ = padding;
        /* Return with no error. */
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }
}
