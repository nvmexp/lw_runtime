/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LWPHY_POLAR_ENCODER_HPP_INCLUDED_)
#define LWPHY_POLAR_ENCODER_HPP_INCLUDED_

#include "tensor_desc.hpp"

namespace polar_encoder
{
static constexpr uint32_t N_MAX_INFO_BITS   = LWPHY_POLAR_ENC_MAX_INFO_BITS;
static constexpr uint32_t N_MIN_CODED_BITS  = 32;  // smallest polar code word length
static constexpr uint32_t N_MAX_CODED_BITS  = 512; // biggest polar code word length
static constexpr uint32_t N_MAX_TX_BITS     = LWPHY_POLAR_ENC_MAX_TX_BITS;
static constexpr uint32_t MIN_CODE_RATE_ILW = 8; // min code rate = (1/8) per 3GPP

void encodeRateMatch(uint32_t       nInfoBits, // # of information bits
                     uint32_t       nTxBits,   // # of bits to transmit
                     uint8_t const* pInfoBits,
                     uint32_t*      pNCodedBits, // # of generated encoded bits
                     uint8_t*       pCodedBits,  // Storage in bytes to hold N_MAX_CODED_BITS
                     uint8_t*       pTxBits,
                     lwdaStream_t   strm);

} // namespace polar_encoder

#endif // !defined(LWPHY_POLAR_ENCODER_HPP_INCLUDED_)
