/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef LW_ALTRAN_STACK_HPP_INCLUDED_
#define LW_ALTRAN_STACK_HPP_INCLUDED_

#include "altran_fapi_includes.hpp"
#include "lw_phy_factory.hpp"
#include "lw_altran_phy.hpp"
#include <cstring>
#include "lwphytools.hpp"

namespace lw_altran_stack
{
    static const char* LW_ALTRAN_PHY = "lw_altran_phy";
    void init(struct phytools_ctx * _ptctx);
} // namespace lw_altran_stack

#endif //LW_ALTRAN_STACK_HPP_INCLUDED_
