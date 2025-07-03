#include <cassert>
#include <cmath>
#include <limits>
#include <array>
#include <iostream>
#include <unordered_set>

#include <lwda_runtime.h>
#include <lwComplex.h>

#include <lwtensor/internal/types.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/exceptions.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
    bool isValidUnaryOperator( const lwtensorOperator_t op, const lwdaDataType_t computeType ) noexcept
    {
        if (((computeType == LWDA_R_32F) || (computeType == LWDA_R_64F)) || (computeType == LWDA_R_16F))
        {
            switch (op)
            {
            case lwtensorOperator_t::LWTENSOR_OP_IDENTITY:
            case lwtensorOperator_t::LWTENSOR_OP_SQRT:
            case lwtensorOperator_t::LWTENSOR_OP_RCP:
            case lwtensorOperator_t::LWTENSOR_OP_RELU:
            // New operators for activations
            case lwtensorOperator_t::LWTENSOR_OP_CLIP:
            case lwtensorOperator_t::LWTENSOR_OP_THRESHOLDED_RELU:
            case lwtensorOperator_t::LWTENSOR_OP_SIGMOID:
            case lwtensorOperator_t::LWTENSOR_OP_TANH:
            case lwtensorOperator_t::LWTENSOR_OP_ELU:
            case lwtensorOperator_t::LWTENSOR_OP_LEAKY_RELU:
            case lwtensorOperator_t::LWTENSOR_OP_SOFT_PLUS:
            case lwtensorOperator_t::LWTENSOR_OP_SOFT_SIGN:
            case lwtensorOperator_t::LWTENSOR_OP_SELU:
            case lwtensorOperator_t::LWTENSOR_OP_HARD_SIGMOID:
            case lwtensorOperator_t::LWTENSOR_OP_SCALED_TANH:
            // New operators for TRT unary runner.
            case lwtensorOperator_t::LWTENSOR_OP_EXP:
            case lwtensorOperator_t::LWTENSOR_OP_LOG:
            case lwtensorOperator_t::LWTENSOR_OP_ABS:
            case lwtensorOperator_t::LWTENSOR_OP_NEG:
            case lwtensorOperator_t::LWTENSOR_OP_SIN:
            case lwtensorOperator_t::LWTENSOR_OP_COS:
            case lwtensorOperator_t::LWTENSOR_OP_TAN:
            case lwtensorOperator_t::LWTENSOR_OP_SINH:
            case lwtensorOperator_t::LWTENSOR_OP_COSH:
            case lwtensorOperator_t::LWTENSOR_OP_ASIN:
            case lwtensorOperator_t::LWTENSOR_OP_ACOS:
            case lwtensorOperator_t::LWTENSOR_OP_ATAN:
            case lwtensorOperator_t::LWTENSOR_OP_ASINH:
            case lwtensorOperator_t::LWTENSOR_OP_ACOSH:
            case lwtensorOperator_t::LWTENSOR_OP_ATANH:
            case lwtensorOperator_t::LWTENSOR_OP_CEIL:
            case lwtensorOperator_t::LWTENSOR_OP_FLOOR: return true;
            default: return false;
            }
        }
        else if ( ((computeType == LWDA_C_32F) || (computeType == LWDA_C_64F)) || (computeType == LWDA_C_16F) )
        {
            return (( op == lwtensorOperator_t::LWTENSOR_OP_IDENTITY ) ||
                    ( op == lwtensorOperator_t::LWTENSOR_OP_CONJ ) );
        }
        else if ( (((computeType == LWDA_R_8I) || (computeType == LWDA_R_8U)) || (computeType == LWDA_R_32I)) ||
                   (computeType == LWDA_R_32U) )
        {
            return (( op == lwtensorOperator_t::LWTENSOR_OP_IDENTITY ) ||
                    ( op == lwtensorOperator_t::LWTENSOR_OP_RELU ));
        }
        else
        {
            /* This op is not valid. */
            return false;
        }
    }

    bool isValidBinaryOperator( const lwtensorOperator_t op, const lwdaDataType_t computeType) noexcept
    {

        if (computeType == LWDA_C_32F) 
        {
            return (op == lwtensorOperator_t::LWTENSOR_OP_ACTIVATION_WITH_QUANTIZATION) ||
                    ( op == lwtensorOperator_t::LWTENSOR_OP_ADD ) ||
                    ( op == lwtensorOperator_t::LWTENSOR_OP_MUL );
        }
        else if ((computeType == LWDA_C_64F) || (computeType == LWDA_C_16F) )
        {
            return ( ( op == lwtensorOperator_t::LWTENSOR_OP_ADD ) ||
                     ( op == lwtensorOperator_t::LWTENSOR_OP_MUL ) );
        }
        else
        {
            return ( ((( op == lwtensorOperator_t::LWTENSOR_OP_ADD ) ||
                     ( op == lwtensorOperator_t::LWTENSOR_OP_MUL )) ||
                     ( op == lwtensorOperator_t::LWTENSOR_OP_MAX )) ||
                     ( op == lwtensorOperator_t::LWTENSOR_OP_MIN ) );
        }
    }

    bool isZero(const void* const ptr, const lwdaDataType_t type)
    {
        bool returlwalue;
        switch(type)
        {
            case LWDA_R_8I:
            {
                returlwalue = *(static_cast<const int8_t*>(ptr)) == 0;
                break;
            }
            case LWDA_R_8U:
            {
                returlwalue = *(static_cast<const uint8_t*>(ptr)) == 0U;
                break;
            }
            case LWDA_R_32U:
            {
                returlwalue = *(static_cast<const uint32_t*>(ptr)) == 0U;
                break;
            }
            case LWDA_R_32I:
            {
                returlwalue = *(static_cast<const int32_t*>(ptr)) == 0;
                break;
            }
            case LWDA_R_16F:
            {
                returlwalue = *(static_cast<const uint16_t*>(ptr)) == 0U;
                break;
            }
            case LWDA_R_32F:
            {
                returlwalue = *(static_cast<const float*>(ptr)) == 0.F;
                break;
            }
            case LWDA_R_64F:
            {
                returlwalue = *(static_cast<const double*>(ptr)) == 0.;
                break;
            }
            case LWDA_C_32F:
            {
                const lwComplex x = *( static_cast<const lwComplex *>(ptr) );
                returlwalue = (lwCrealf(x) == 0.F) && (lwCimagf(x) == 0.F);
                break;
            }
            case LWDA_C_64F:
            {
                const lwDoubleComplex x = *(static_cast<const lwDoubleComplex*>(ptr) );
                returlwalue = (lwCreal(x) == 0.) && (lwCimag(x) == 0.);
                break;
            }
            default:
            {
                throw NotSupported("Datatype is not yet supported (isZero).\n");
            }
        }
        return returlwalue;
    }

    lwtensorStatus_t  initStride(
            const std::unordered_map<mode_type, extent_type> &extent,
            const std::list<mode_type> &modeA,
            std::unordered_map<mode_type, stride_type> &strideA) noexcept
    {
        strideA.clear();
        stride_type lwrrentStride = 1;
        for ( const auto m : modeA )
        {
            strideA[m] = lwrrentStride;
            if ( extent.find(m) == extent.end() )
            {
                /* This should never happen other than the unit test. */
                return lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR;
            }
            lwrrentStride *= extent.at(m);
        }
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    bool hasDuplicates(const mode_type *const modes, const uint32_t numModes)
    {
        std::unordered_set<mode_type> duplicates;
        for(uint32_t i=0U; i < numModes; ++i)
        {
            if(duplicates.find(modes[i]) != duplicates.end() )
            {
                return true;
            }
            duplicates.insert(modes[i]); // return not checked because we don't care if the value was newly inserted
                                         // or if it was already present; we only care
                                         // that it was inserted and this cannot be
                                         // checked at all.
        }
        return false;
    }

    lwtensorStatus_t handleError( const lwdaError_t err ) noexcept
    {
        if (err != lwdaSuccess)
        {
            std::cerr << "LWCA ERROR: " << (int)err << std::endl;
            if ((err == lwdaErrorNoDevice) || (err == lwdaErrorDevicesUnavailable) ||
                (err == lwdaErrorDeviceAlreadyInUse) || (err == lwdaErrorIlwalidDeviceFunction))
            {
                return lwtensorStatus_t::LWTENSOR_STATUS_ARCH_MISMATCH;
            } else if (err == lwdaErrorInsufficientDriver)
            {
                return lwtensorStatus_t::LWTENSOR_STATUS_INSUFFICIENT_DRIVER;
            }
            return lwtensorStatus_t::LWTENSOR_STATUS_LWDA_ERROR;
        }
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t handleError( const lwtensorStatus_t err ) noexcept
    {
        return err;
    }

    /* No user logging */
    lwtensorStatus_t  handleError( const lwtensorStatus_t err, const std::string &&desc ) noexcept
    {
        #ifdef DEVELOP
            std::cerr << "LWTENSOR ERROR: " << desc << std::endl;
        #endif
        return err;
    }

    /* User logging */
    lwtensorStatus_t  handleError_log( const lwtensorStatus_t err, const std::string &&desc ) noexcept
    {
        std::cerr << "LWTENSOR ERROR: " << desc << std::endl;
        return err;
    }

    lwtensorStatus_t validateStride(const std::unordered_map<mode_type,stride_type> &strides,
                                    const std::list<mode_type> &modes) noexcept
    {
        if(modes.size() != strides.size() ) {
            return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "modes and strides do not match.");
        }
        if (modes.size() == 0U) {
            return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
        }
        if(strides.find(modes.front()) == strides.end()) {
            return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "No stride for mode found.");
        }
        auto lastStride = strides.at(modes.front());
        for(const auto mode : modes)
        {
            if(strides.find(mode) == strides.end()) {
                return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "No stride for mode found.");
            }
            if(lastStride > strides.at(mode) ) {
                return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Modes are not sorted.");
            }
            lastStride = strides.at(mode);
        }
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t initStrideExtentModesSorted(
            const lwtensorTensorDescriptor * const desc,
            const mode_type *const modesUnsorted,
            std::unordered_map<mode_type, stride_type>& strides,
            std::list<mode_type>& modesSorted,
            std::unordered_map<mode_type, extent_type> &extent )
    {
        assert( (strides.size() == 0U) && (modesSorted.size() == 0U) );
        //TODO remove extents of size 1

        if ( (desc == nullptr) || (modesUnsorted == nullptr))
        {
            return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
        }

        std::array<uint32_t, lwtensorTensorDescriptor::LWTENSOR_MAX_MODES> perm;
        desc->getStridePermutationAscending(perm);

        /* Loop over all modes in the descriptor. */
        for ( uint32_t i = 0U; i < desc->getNumModes(); ++ i )
        {
            const auto lwrrentMode = modesUnsorted[perm.at(static_cast<size_t>(i))];
            const auto lwrrentExtent = desc->getExtent(perm.at(static_cast<size_t>(i)));

            if (extent.find(lwrrentMode) != extent.end())
            {
                if (extent.at(lwrrentMode) != lwrrentExtent)
                {
                    std::cerr<< "LWTENSOR ERROR: extent of mode " << lwrrentMode <<" does not match.\n";
                    return lwtensorStatus_t::LWTENSOR_STATUS_ILWALID_VALUE;
                }
                else
                {
                    /* Map the current mode to its extent. */
                    extent[lwrrentMode] = lwrrentExtent;
                }
            }
            else
            {
                /* Map the current mode to its extent. */
                extent[lwrrentMode] = lwrrentExtent;
            }
            /* Map the current mode to its stride. */
            strides[lwrrentMode] = desc->getStride(perm.at(static_cast<size_t>(i)));
            /* Push the current mode to the output. */
            modesSorted.push_back(lwrrentMode);
        }
        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t fuseModes(
            std::list<mode_type> &modeA, std::unordered_map<mode_type, stride_type> &strideA, const mode_type vectorModeA,
            std::list<mode_type> &modeB, std::unordered_map<mode_type, stride_type> &strideB, const mode_type vectorModeB,
            std::list<mode_type> &modeC, std::unordered_map<mode_type, stride_type> &strideC, const mode_type vectorModeC,
            std::unordered_map<mode_type, extent_type> &extent )
    {
#ifdef DEBUG
        std::cout << "Modes before merge:\n";
        std::cout << "A: ";
        for(const auto mode : modeA)
            printf("%d ",mode);
        std::cout << "\nB: ";
        for(const auto mode : modeB)
            std::cout << mode;
        std::cout << "\nC: ";
        for(const auto mode : modeC)
            std::cout << mode;
        std::cout << "\n";
#endif
        /* Collect all the "batched" modes (i.e., the modes that appear in every tensor) */
        std::unordered_set<mode_type> batchModes;
        for(const auto mode : modeC)
        {
            if ((std::find(modeA.begin(), modeA.end(), mode) != modeA.end()) &&
                (std::find(modeB.begin(), modeB.end(), mode) != modeB.end()))
            {
                batchModes.insert(mode); // return not checked because we don't care if the value was newly inserted
                                         // or if it was already present; we only care
                                         // that it was inserted and this cannot be
                                         // checked at all.
            }
        }

        /* Precondition: Assuming that all modes are either in A or C */
        std::list<mode_type> allModes;
        for(const auto mode : modeC)
        {
            if (((mode != vectorModeA) &&
                (mode != vectorModeB)) &&
                (mode != vectorModeC))
            {
                allModes.push_back(mode);
            }
        }
        for(const auto mode : modeA)
        {
            // only push if not found before
            if( (std::find(allModes.begin(), allModes.end(), mode) == allModes.end()) &&
                (mode != vectorModeA) && (mode != vectorModeB) && (mode != vectorModeC))
            {
                allModes.push_back(mode);
            }
        }

        for( const auto mode : allModes )
        {
            auto itA = std::find(modeA.begin(), modeA.end(), mode);
            auto itB = std::find(modeB.begin(), modeB.end(), mode);
            auto itC = std::find(modeC.begin(), modeC.end(), mode);

            const uint32_t foundA = static_cast<uint32_t>(itA != modeA.end());
            const uint32_t foundB = static_cast<uint32_t>(itB != modeB.end());
            const uint32_t foundC = static_cast<uint32_t>(itC != modeC.end());

                  auto it  = (foundA == 1U) ? itA : itC;
            //const auto end = (foundA == 1U) ? modeA.cend() : modeC.cend();
            const auto fuseStart = it;

            /* fix A0-1-1, Value is stored in "foundX" but is not subsequently used on this path. */
            const uint32_t numModesFound = foundA + foundB + foundC;

            if( (foundA == 0U) && (foundC == 0U) )
            {
                continue; // no fusion possible
            }

            if (numModesFound > 1U)
            {
                /* fix A0-1-1, Value is stored in "end" but is not subsequently used on this path. */
                const auto end = (foundA == 1U) ? modeA.end() : modeC.end();
                mode_type prevMode = mode;
                extent_type prevExtent = extent.at(prevMode);
                it++;
                while ( it != end )
                {
                    // Advance iterator to next mode
                    if (itA != modeA.end()){ itA++; }
                    if (itB != modeB.end()){ itB++; }
                    if (itC != modeC.end()){ itC++; }
                    const mode_type lwrrentMode = *it; // select from itA, itB, itC
                    if (((lwrrentMode == vectorModeA) || (lwrrentMode == vectorModeB)) || (lwrrentMode == vectorModeC))
                    {
                        break; // do not fuse vectorized modes
                    }

                    const bool isBatchedMode = batchModes.find(lwrrentMode) != batchModes.end();
                    if( isBatchedMode && (numModesFound != 3U) )
                    {
                        break; // prevents cases where the lwrrentMode should be fused but it actually appears in
                               // a third tensor (while the initial ones don't)
                    }

                    const mode_type lwrrentModeA = (itA != modeA.end()) ? *itA : LWTENSOR_ILWALID_MODE;
                    const bool fuseA = ((foundA != 1U) || // index must be found
                            ((lwrrentMode == lwrrentModeA) &&
                             (strideA.at(lwrrentModeA) == (strideA.at(prevMode) * prevExtent)))); // dense

                    const mode_type lwrrentModeB = (itB != modeB.end()) ? *itB : LWTENSOR_ILWALID_MODE;
                    const bool fuseB = ((foundB != 1U) ||
                            ((lwrrentMode == lwrrentModeB) &&
                             (strideB.at(lwrrentModeB) == (strideB.at(prevMode) * prevExtent))));

                    const mode_type lwrrentModeC = (itC != modeC.end()) ? *itC : LWTENSOR_ILWALID_MODE;
                    const bool fuseC = ((foundC != 1U) ||
                            ((lwrrentMode == lwrrentModeC) &&
                             (strideC.at(lwrrentModeC) == (strideC.at(prevMode) * prevExtent))));

                    if( ((!fuseA) || (!fuseB)) || (!fuseC) )
                    {
                        break;
                    }
                    prevMode = lwrrentMode;
                    prevExtent = extent.at(prevMode);
                    it++;
                }

                if( it != std::next(fuseStart) )
                {
                    const auto fuseEnd = it;
                    // callwlate merged extent
                    extent_type totalExtent = 1;
                    for(auto iter = fuseStart; iter != fuseEnd; iter++)
                    {
                        totalExtent *= extent.at(*iter);
                    }
                    extent[mode] = totalExtent;

                    // remove merged modes
                    for(auto iter = std::next(fuseStart); iter != fuseEnd; iter++)
                    {
                        const auto modeFuse = *iter;
                        if (extent.erase(modeFuse) != 1U)
                        {
                            return lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR;
                        }
                        if (foundA == 1U)
                        {
                            iter = modeA.erase(iter);
                            iter--; // correct for erase (due to subsequent iter++ of for loop)
                            if (strideA.erase(modeFuse) != 1U)
                            {
                                return lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR;
                            }

                            if (foundC == 1U)
                            {
                                modeC.remove(modeFuse);
                                if (strideC.erase(modeFuse) != 1U)
                                {
                                    return lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR;
                                }
                            }
                        }
                        else
                        {
                            // This logic is important since we assume that iter either refers to modeA or modeC
                            // in this case iter refers to modeC !
                            iter = modeC.erase(iter);
                            iter--; // correct for erase (due to subsequent iter++ of for loop)
                            if (strideC.erase(modeFuse) != 1U)
                            {
                                    return lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR;
                            }
                        }

                        if (foundB == 1U)
                        {
                            modeB.remove(modeFuse);
                            if (strideB.erase(modeFuse) != 1U)
                            {
                                    return lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR;
                            }
                        }
                    }
                }
            }
        }

#ifdef DEBUG
        for( const auto mode : modeC )
        {
            if(extent.find(mode) == extent.end() ) {
                return handleError(lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR, "Extent for mode not found.");
            }
        }

        printf("modes after merge:\n");
        printf("A: ");
        for(auto m : modeA)
            printf("%c ",m);
        printf("\nB: ");
        for(auto m : modeB)
            printf("%c ",m);
        printf("\nC: ");
        for(auto m : modeC)
            printf("%c ",m);
        printf("\n");
#endif

        return lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS;
    }

    size_t getDataTypeSize( const lwdaDataType_t type )
    {
        if ( (type == LWDA_R_8I) || (type == LWDA_R_8U) )
        {
            return 1U;
        }
        else if( type == LWDA_R_16F )
        {
            return 2U;
        }
        else if( (type == LWDA_R_32I) || (type == LWDA_R_32U) )
        {
            return 4U;
        }
        else if( (type == LWDA_R_32F) || (type == LWDA_C_16F) )
        {
            return 4U;
        }
        else if( (type == LWDA_R_64F) || (type == LWDA_C_32F)  )
        {
            return 8U;
        }
        else if( type == LWDA_C_64F )
        {
            return 16U;
        }
        else
        {
            throw NotSupported( "Datatype is not yet supported.\n" );
        }
    }

    lwtensorStatus_t handleException(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        if( dynamic_cast<const LWTENSOR_NAMESPACE::NotSupported*>(&e) != nullptr )
        {
            return lwtensorStatus_t::LWTENSOR_STATUS_NOT_SUPPORTED;
        }
        return lwtensorStatus_t::LWTENSOR_STATUS_INTERNAL_ERROR;
    }
}
