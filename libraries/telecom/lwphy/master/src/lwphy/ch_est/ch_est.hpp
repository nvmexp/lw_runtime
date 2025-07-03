/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(CH_EST_HPP_INCLUDED_)
#define CH_EST_HPP_INCLUDED_

#include "tensor_desc.hpp"

namespace ch_est
{

enum class dmrsCfg_t : uint32_t
{
   DMRS_CFG0 = LWPHY_DMRS_CFG0, // 1 layer : DMRS grid 0   ; fOCC = [+1, +1]          ; 1 DMRS symbol
   DMRS_CFG1 = LWPHY_DMRS_CFG1, // 2 layers: DMRS grids 0,1; fOCC = [+1, +1]          ; 1 DMRS symbol
   DMRS_CFG2 = LWPHY_DMRS_CFG2  // 4 layers: DMRS grids 0,1; fOCC = [+1, +1], [+1, -1]; 1 DMRS symbol
};

void estimate_channel(uint32_t           cellId,
                      uint32_t           slotNum,
                      uint32_t           nBSAnts,
                      uint32_t           nLayers,
                      uint32_t           nDMRSSyms,
                      uint32_t           nDMRSGridsPerPRB,
                      uint32_t           nTotalDMRSPRB,
                      uint32_t           nTotalDataPRB,
                      uint32_t           Nh,
                      uint32_t           activeDMRSGridBmsk,
                      const_tensor_pair& tDataRx,
                      const_tensor_pair& tWFreq,
                      const_tensor_pair& tShiftSeq,
                      const_tensor_pair& tUnShiftSeq,
                      tensor_pair&       tH,
                      tensor_pair&       tDbg,
                      lwdaStream_t       strm);
} // namespace ch_est

#endif // !defined(CH_EST_HPP_INCLUDED_)
