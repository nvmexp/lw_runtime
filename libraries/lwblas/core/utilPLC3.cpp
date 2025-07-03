#include <lwtensor/internal/typesPLC3.h>
#include <lwtensor/internal/defines.h>
#include <lwtensor/internal/utilPLC3.h>

namespace LWTENSOR_NAMESPACE
{

    bool isRangeOverlapping(uint64_t const min1,
            uint64_t const max1,
            uint64_t const min2,
            uint64_t const max2) noexcept
    {
        return (min1 <= max2) && (min2 <= max1);
    }

    //int64_t getMaximalOffset(uint32_t const nmode,
    //        extent_type const *const extents,
    //        stride_type const *const strides,
    //        uint8_t const *const vectorWidthLog2,
    //        uint8_t const *const vectorOffset,
    //        extent_type const padding_size) noexcept
    //{
    //    int64_t maximalOffset {0L};
    //    for(uint32_t i {0U}; i < nmode; ++i)
    //    {
    //        int64_t const myStride {static_cast<int64_t>(strides[i])};
    //        uint8_t const vectorWidth {static_cast<uint8_t>(1U << vectorWidthLog2[i])};
    //        if (vectorWidth > 1U)
    //        {
    //            const int64_t maxOffsetLwrrentMode {static_cast<int64_t>(extents[i] - 1) + static_cast<int8_t>(vectorOffset[i]) + padding_size};
    //            maximalOffset += (static_cast<int64_t>((maxOffsetLwrrentMode / static_cast<int8_t>(vectorWidth))) * myStride) +
    //                static_cast<int64_t>(maxOffsetLwrrentMode % static_cast<int8_t>(vectorWidth));
    //        }
    //        else
    //        {
    //            maximalOffset += (static_cast<int64_t>(extents[i]) - 1) * myStride;
    //        }
    //    }
    //    return maximalOffset;
    //}
    
    int64_t getMaximalOffset(
            uint32_t const numModes,
            extent_type const extents[ElementwiseParameters::LWTENSOR_MAX_MODES],
            ElementwiseParameters::ElementwiseLayout const & layout,
            extent_type const padding_size) noexcept
    {
        int64_t offset {0L};
        for (uint32_t i {0U}; i < numModes; ++ i)
        {
            int64_t const stride {static_cast<int64_t>(layout.stride[i])};
            int64_t extent {static_cast<int64_t>(extents[i] - 1)};
            if (layout.vectorWidthLog2[i] != 0U)
            {
                int64_t const vectorWidth {static_cast<int64_t>(static_cast<uint8_t>(1U << layout.vectorWidthLog2[i]))};
                int64_t const vectorOffset {static_cast<int64_t>(layout.vectorOffset[i])};
                /* The non-vectorized mode has vectorOffset = 0 and padding = 0. */
                extent += static_cast<int64_t>(vectorOffset) + static_cast<int64_t>(padding_size);
                /* Using the module formula to compute the offset contributed from the vector mode. */
                offset += ((extent / vectorWidth) * stride) + (extent % vectorWidth);
            }
            else
            {
                offset += extent * stride;
            }
        }
        return offset;
    }
}
