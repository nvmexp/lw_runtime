#include <lwda_runtime.h>

#include <list>
#include <limits>
#include <cassert>

#include <lwtensor/internal/exceptions.h>
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/context.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{

    std::string ElementwisePlan::toString() const noexcept
    {
        auto numModes = params_.nmodeC_;
        std::string desc = "-extent";
        for (int i=0; i < numModes; ++i)
        {
            desc += std::to_string(i) + "=" + std::to_string(params_.extent_[i]) + ",";
        }
        desc += " -modeC";
        for (int i=0; i < numModes; ++i)
        {
            desc += std::to_string(i);
            if (i != numModes - 1)
            {
                desc += ",";
            }
        }
        desc += " -strideC";
        for (int i=0; i < numModes; ++i)
        {
            desc += std::to_string(params_.strideC_[i]);
            if (i!= numModes - 1)
            {
                desc += ",";
            }
        }
        desc += " -Pc" + lwdaDataTypeToString(params_.typePack_.typeC_);

        if (params_.useA_)
        {
            std::array<std::pair<stride_type, mode_type>, TensorDescriptor::LWTENSOR_MAX_MODES> toSort;
            for ( uint32_t i = 0; i < numModes; i ++ )
            {
                toSort[i] = std::pair<stride_type, mode_type>(params_.strideA_[i], i);
            }

            // sort w.r.t. stride
            std::sort(toSort.begin(), toSort.begin() + numModes, [](
                        const std::pair<stride_type, mode_type>&a,
                        const std::pair<stride_type, mode_type>&b) { return a.first < b.first; });

            desc += " -modeA";
            for (int i=0; i < numModes; ++i)
            {
                desc += std::to_string(toSort[i].second);
                if (i != numModes - 1)
                {
                    desc += ",";
                }
            }
            desc += " -strideA";
            for (int i=0; i < numModes; ++i)
            {
                desc += std::to_string(toSort[i].first);
                if (i!= numModes - 1)
                {
                    desc += ",";
                }
            }
            desc += " -Pa" + lwdaDataTypeToString(params_.typePack_.typeA_);
        }

        if (params_.useB_)
        {
            desc += " -modeB";
            for (int i=0; i < numModes; ++i)
            {
                desc += std::to_string(i);
                if (i != numModes - 1)
                {
                    desc += ",";
                }
            }
            desc += " -strideB";
            for (int i=0; i < numModes; ++i)
            {
                desc += std::to_string(params_.strideB_[i]);
                if (i!= numModes - 1)
                {
                    desc +=",";
                }
            }
            desc += " -Pb" + lwdaDataTypeToString(params_.typePack_.typeB_);
        }
        desc += " -Relementwise";
        return desc + "\n";
    }

    stride_type getTotalTiles(const ElementwiseParameters & param, const uint32_t nmodes_blocked, 
            const uint32_t * const blocking)
    {
        uint64_t total = 1U;

        for ( uint32_t i = 0U; i < param.nmodeC_; ++i )
        {
            extent_type numCTAs = 1;
            if (i < nmodes_blocked)
            {
                /* M5-0-2: requires parenthesis. */
                uint32_t num_blocks = (param.extent_[i] + static_cast<extent_type>(blocking[i])) - 1U;
                /* M5-2-12: cannot derefer an array in assert(). */
                if (blocking[i] == 0U)
                {
                    throw std::ilwalid_argument("Block size must not be zero.\n");
                }

                num_blocks /= static_cast<extent_type>(blocking[ i ]);
                numCTAs = num_blocks;
            }
            else
            {
                /* If not blocked. */
                numCTAs = param.extent_[ i ];
            }
            total *= static_cast<uint64_t>(numCTAs);
        }

        //TODO This case must be handled
        if( total > static_cast<uint64_t>(std::numeric_limits<stride_type>::max()) )
        {
            throw std::overflow_error( "Extent is too large; this feature needs to be implemented.\n" );
        }

        return static_cast<stride_type>(total);
    }

    lwtensorStatus_t ElementwiseParameters::init(
                    const Context* ctx,
                    const ModeList &mode_order,
                    const ExtentMap &extent,
                    const bool useA, lwdaDataType_t typeA, const ModeList &modeA, lwtensorOperator_t opA,
                    const bool useB, lwdaDataType_t typeB, const ModeList &modeB, lwtensorOperator_t opB,
                    const bool useC, lwdaDataType_t typeC, const ModeList &modeC, lwtensorOperator_t opC, lwtensorOperator_t opAB, lwtensorOperator_t opABC, lwdaDataType_t typeCompute,
                    const uint32_t alignmentRequirementA,
                    const uint32_t alignmentRequirementB,
                    const uint32_t alignmentRequirementC,
                    const uint32_t alignmentRequirementD,
                    const StrideMap &strideA_tmp,
                    const StrideMap &strideB_tmp,
                    const StrideMap &strideC_tmp)
    {
        this->useA_ = useA;
        this->useB_ = useB;
        this->useC_ = useC;
        this->alignmentRequirementA_ = alignmentRequirementA;
        this->alignmentRequirementB_ = alignmentRequirementB;
        this->alignmentRequirementC_ = alignmentRequirementC;
        this->alignmentRequirementD_ = alignmentRequirementD;
        this->opPack_.opA_ = opA;
        this->opPack_.opB_ = opB;
        this->opPack_.opC_ = opC;
        this->opPack_.opAB_ = opAB;
        this->opPack_.opUnaryAfterBinary_ = lwtensorOperator_t::LWTENSOR_OP_IDENTITY; // we don't expose this
        this->opPack_.opABC_ = opABC;

        this->typePack_.typeA_ = typeA;
        this->typePack_.typeB_ = typeB;
        this->typePack_.typeC_ = typeC;
        this->typePack_.typeCompute_ = typeCompute;

        this->nmodeC_ = static_cast<uint32_t>(mode_order.size());

        if( this->nmodeC_ > ElementwiseParameters::LWTENSOR_MAX_MODES )
        {
            return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED, "Too many (non-fusable) modes.");
        }

        const StrideMap * strideAptr = &strideA_tmp;
        StrideMap strideApacked;
        if ( strideA_tmp.size() == 0U )
        {
            if ( initStride(extent, modeA, strideApacked) != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
            {
                RETURN_STATUS(handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Strides of A are invalid." ));
            }
            strideAptr = &strideApacked;
        }
        const StrideMap * strideBptr = &strideB_tmp;
        StrideMap strideBpacked;
        if ( strideB_tmp.size() == 0U )
        {
            if ( initStride(extent, modeB, strideBpacked) != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
            {
                RETURN_STATUS(handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Strides of B are invalid." ));
            }
            strideBptr = &strideBpacked;
        }
        const StrideMap * strideCptr = &strideC_tmp;
        StrideMap strideCpacked;
        if ( strideC_tmp.size() == 0U )
        {
            if ( initStride(extent, modeC, strideCpacked) != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS )
            {
                RETURN_STATUS(handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Strides of C are invalid." ));
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
                RETURN_STATUS(handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Extent not found." ));
            }
            if ( extent.at( mode ) <= 0 )
            {
                RETURN_STATUS(handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Invalid extent." ));
            }
            this->extent_[ idx ] = extent.at( mode );
            const auto ita = strideAptr->find(mode);
            const auto itb = strideBptr->find(mode);
            const auto itc = strideCptr->find(mode);
            this->strideA_[idx] = (ita != strideAptr->end() ) ? ita->second : 0;
            this->strideB_[idx] = (itb != strideBptr->end() ) ? itb->second : 0;
            this->strideC_[idx] = (itc != strideCptr->end() ) ? itc->second : 0;

            /* If the target mode is vectorized in tensorA, then set the width and offset. */
            /* Advance mode index. */
            idx++;
        }

        this->isStrideOne_ = false;
        const auto strideOneMode = mode_order.front();
        if (((modeA.size() == 0U) || (strideOneMode == modeA.front())) &&
            ((modeB.size() == 0U) || (strideOneMode == modeB.front())) &&
            ((modeC.size() == 0U) || (strideOneMode == modeC.front())))
        {
            isStrideOne_ = true;
        }

        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t TensorDescriptor::setVectorization(
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
            RETURN_STATUS(lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE);
        }

        if ((vectorWidth > 1U) && (vectorModeIndex >= numModes_))
        {
            RETURN_STATUS(lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE);
        }

        this->vectorWidth_ = vectorWidth;

        this->vectorModeIndex_ = (vectorWidth > 1U) ? vectorModeIndex : 999U;

        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    TensorDescriptor::TensorDescriptor(
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
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
            (dataType_ != LWDA_R_16BF) &&
#endif
            (dataType_ != LWDA_C_16F) &&
            (dataType_ != LWDA_R_64F) &&
            (dataType_ != LWDA_C_32F) &&
            (dataType_ != LWDA_C_64F) )
        {
            throw NotSupported("Datatype is not yet supported.\n");
        }
        this->setInitialized();
    }

    TensorDescriptor::TensorDescriptor(
            const uint32_t numModes,
            const extent_type * const extent,
            const stride_type * const stride,
            const lwdaDataType_t dataType,
            const lwtensorOperator_t op,
            const uint8_t vectorWidth,
            const uint32_t vectorModeIndex,
            const uint8_t  vectorOffset,
            const bool zeroPadding) :
    TensorDescriptor(numModes, extent, stride, dataType, op, vectorWidth, vectorModeIndex)
    {
        if (vectorOffset >= vectorWidth)
        {
            throw NotSupported("Vector offset is invalid.\n");
        }
        vectorOffset_ = vectorOffset;
        zeroPadding_ = zeroPadding;
        this->setInitialized();
    }

    TensorDescriptor::TensorDescriptor(const TensorDescriptor &other) :
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
        this->setInitialized(other.isInitialized());
    }

    TensorDescriptor& TensorDescriptor::operator=(const TensorDescriptor &other)
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

        this->setInitialized(other.isInitialized());

        return *this;
    }

    lwtensorStatus_t TensorDescriptor::init(
                                        const Context* ctx,
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

        if ( !isValidUnaryOperator(op, dataType) ) {
            return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE, "Invalid operator.");
        }

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
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
            (dataType_ != LWDA_R_16BF) &&
#endif
            (dataType_ != LWDA_R_32F) &&
            (dataType_ != LWDA_C_16F) &&
            (dataType_ != LWDA_R_64F) &&
            (dataType_ != LWDA_C_32F) &&
            (dataType_ != LWDA_C_64F) )
        {
            return ctx->logError(lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED, "Data type not yet supported.");
        }

        this->setInitialized();
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t TensorDescriptor::setVectorOffset(const uint8_t offset) noexcept
    {
        if (offset >= vectorWidth_)
        {
            RETURN_STATUS(lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE);
        }
        vectorOffset_ = offset;
        /* Return with no error. */
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t TensorDescriptor::setZeroPadding(const bool padding) noexcept
    {
        zeroPadding_ = padding;
        /* Return with no error. */
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t ContractionFind::setAutotuneMode(const lwtensorAutotuneMode_t &autotuneMode)
    { 
        if( isValidAutotuneMode(autotuneMode) )
        {
            autotuneMode_ = autotuneMode;
            return LWTENSOR_STATUS_SUCCESS;
        } else {
            RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
        }
    }
    lwtensorStatus_t ContractionFind::setCacheMode(const lwtensorCacheMode_t &cacheMode)
    {
        if( isValidCacheMode(cacheMode) )
        {
            cacheMode_ = cacheMode;
            return LWTENSOR_STATUS_SUCCESS;
        }else{
            RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
        }
    }

    lwtensorStatus_t ContractionFind::setIncrementalCount(const uint32_t incCount)
    {
        if( incCount > 0 )
        {
            incrementalCount_ = incCount;
            return LWTENSOR_STATUS_SUCCESS;
        }
        else
        {
            RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
        }
    }

    lwtensorStatus_t ContractionFind::setPartitionsK(const int32_t numPartitionsK)
    {
        if( numPartitionsK != -1 && numPartitionsK > 0 )
        {
            numPartitionsK_ = numPartitionsK;
            return LWTENSOR_STATUS_SUCCESS;
        }
        else
        {
            RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
        }
    }

}
