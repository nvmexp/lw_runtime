/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <cstring>
#include "lw_altran_stack.hpp"
#include "lw_altran_phy.hpp"

static struct phytools_ctx * ptctx;

namespace
{
    lw::PHY_instance* create_lw_altran_phy(lw::PHY_module& module,
                                    yaml::node      config)
    {
        return new lw_altran_stack::lw_altran_phy(module, config, ptctx);
    }

    lw::phy_creator lw_altran_phy_creator = {
        lw_altran_stack::LW_ALTRAN_PHY,
        &create_lw_altran_phy
    };
} // namespace

namespace lw_altran_stack
{
    void init(struct phytools_ctx * _ptctx)
    {
        ptctx = _ptctx;
        lw::phy_factory::register_type(lw_altran_phy_creator);
    }
} // namespace lw_altran_stack
