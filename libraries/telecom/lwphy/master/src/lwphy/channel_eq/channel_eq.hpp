/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(CHANNEL_EQ_HPP_INCLUDED_)
#define CHANNEL_EQ_HPP_INCLUDED_

#include "tensor_desc.hpp"

namespace channel_eq
{
// QAM levels
enum class QAM_t : uint32_t
{
    QAM_4   = LWPHY_QAM_4,
    QAM_16  = LWPHY_QAM_16,
    QAM_64  = LWPHY_QAM_64,
    QAM_256 = LWPHY_QAM_256
};

void eqCoefCompute(uint32_t           nBSAnts,
                   uint32_t           nLayers,
                   uint32_t           Nh,
                   uint32_t           Nprb,
                   const_tensor_pair& tH,
                   const_tensor_pair& tRwwIlw,
                   tensor_pair&       tCoef,
                   tensor_pair&       tReeDiag,
                   tensor_pair&       tDbg,
                   lwdaStream_t       strm);

void eqSoftDemap(uint32_t           nBSAnts,
                 uint32_t           nLayers,
                 uint32_t           Nh,
                 uint32_t           Nd,
                 uint32_t           Nprb,
                 const_tensor_pair& tDataSymbLoc,
                 const_tensor_pair& tQam,
                 const_tensor_pair& tCoef,
                 const_tensor_pair& tReeDiag,
                 const_tensor_pair& tDataRx,
                 tensor_pair&       tDataEq,
                 tensor_pair&       tLlr,
                 tensor_pair&       tDbg,
                 lwdaStream_t       strm);

void equalize(uint32_t           nBSAnts,
              uint32_t           nLayers,
              uint32_t           Nh,
              uint32_t           Nf,
              uint32_t           Nd,
              QAM_t              qam,
              const_tensor_pair& tData_sym_loc,
              const_tensor_pair& tData_rx,
              const_tensor_pair& tH,
              const_tensor_pair& tNoise_pwr,
              tensor_pair&       tData_eq,
              tensor_pair&       tRee_diag,
              tensor_pair&       tLLR,
              lwdaStream_t       strm);

} // namespace channel_eq

#endif // !defined(CHANNEL_EQ_HPP_INCLUDED_)
