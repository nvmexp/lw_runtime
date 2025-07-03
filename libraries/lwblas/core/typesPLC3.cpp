#include <unordered_map>
#include <list>
#include <limits>
#include <cassert>

#include <lwtensor/internal/typesPLC3.h>
#include <lwtensor/internal/utilPLC3.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
    stride_type getTotalTiles(ElementwiseParameters const & param, uint32_t const nmodes_blocked, 
            uint32_t const * const blocking)
    {
        uint64_t total {1U};

        for ( uint32_t i {0U}; i < param.nmodeC; ++i )
        {
            extent_type numCTAs {1};
            if (i < nmodes_blocked)
            {
                int32_t num_blocks {(param.extent[i] + static_cast<extent_type>(blocking[i])) - 1};
                /* Take the padding size into account. */
                if ( i == 0U )
                {
                    num_blocks += static_cast<extent_type>(param.padding_size);
                }

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
                numCTAs = param.extent[ i ];
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

    lwtensorStatus_t validateElementwiseLayout(
            uint32_t const numModes,
            ElementwiseParameters::ElementwiseLayout const & layout)
    {
        constexpr uint32_t unreachableMode {ElementwiseParameters::LWTENSOR_MAX_MODES + 1U};
        uint32_t vectorMode {unreachableMode};
        uint32_t vectorWidth {1U};
        for (uint32_t i {0U}; i < numModes; ++ i)
        {
            /* 0 <= log2(vectorWidth) <= 5 */
            if (layout.vectorWidthLog2[i] > 5U)
            {
                return LWTENSOR_STATUS_ILWALID_VALUE;
            }
            /* Check whether this is the vectorized mode? */
            if (layout.vectorWidthLog2[i] > 0U)
            {
                /* There can only be one vectorized mode per tensor. */
                if (vectorMode != unreachableMode)
                {
                    return LWTENSOR_STATUS_ILWALID_VALUE;
                }
                /* Set mode i to be the vectorized mode. */
                vectorMode = i;
                /* Compute the vectorWidth by shifting. */
                vectorWidth = static_cast<uint32_t>(static_cast<uint8_t>(1U << layout.vectorWidthLog2[i]));
                /* vector offset < vector width */  
                if (layout.vectorOffset[i] >= vectorWidth)
                {
                    return LWTENSOR_STATUS_ILWALID_VALUE;
                }
            }
            else
            {
                /* Only the vectorized mode can have an non-zero vector offset. */
                if (layout.vectorOffset[i] > 0U)
                {
                    return LWTENSOR_STATUS_ILWALID_VALUE;
                }
            }
        }

        for (uint32_t i {0U}; i < numModes; ++ i)
        {
            int64_t const stride {layout.stride[i]};
            /* Negative strides are not allowed */
            if (stride < 0)
            {
                return LWTENSOR_STATUS_ILWALID_VALUE;
            }
            /* All strides must be multiple of the vector width. */
            if ((static_cast<int64_t>(stride) % static_cast<int64_t>(vectorWidth)) != 0)
            {
                return LWTENSOR_STATUS_ILWALID_VALUE;
            }
        }
        return LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t validatePaddingSize(
            uint32_t const numModes,
            extent_type const extents[ElementwiseParameters::LWTENSOR_MAX_MODES],
            ElementwiseParameters::ElementwiseLayout const & layout,
            extent_type padding_size)
    {
        int64_t goal {0};
        if (padding_size == 0)
        {
            return LWTENSOR_STATUS_SUCCESS;
        }
        for (uint32_t i {0U}; i < numModes; ++ i)
        {
            if (layout.vectorWidthLog2[i] > 0U)
            {
                /* Compute the vectorWidth by shifting. */
                int64_t const vectorWidth {static_cast<int64_t>(static_cast<uint8_t>(1U << layout.vectorWidthLog2[i]))}; 
                int64_t const vectorOffset {static_cast<int64_t>(layout.vectorOffset[i])};
                goal = (extents[i] + vectorOffset) % vectorWidth;
                goal = (goal != 0) ? (vectorWidth - goal) : 0;
            }
        }
        if (padding_size != goal)
        {
            return LWTENSOR_STATUS_ILWALID_VALUE;
        }
        return LWTENSOR_STATUS_SUCCESS;
    }
} /* end namespace LWTENSOR_NAMESPACE */
