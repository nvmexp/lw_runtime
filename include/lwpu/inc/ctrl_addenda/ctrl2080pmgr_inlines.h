/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef LW_SUBDEVICE_PMGR_INLINES_H_
#define LW_SUBDEVICE_PMGR_INLINES_H_

#include "ctrl/ctrl2080/ctrl2080pmgr.h"

/*!
 * @brief   Determines if a given PWR_POLICY type is a sub-class of the DOMGRP
 *          class.
 *
 * @note    This interface does not do error checking and simply returns
 *          @ref LW_FALSE for invalid/unknown types.
 *
 * @param[in]   type        LW2080_CTRL_PMGR_PWR_POLICY_TYPE_* value
 *
 * @return  @ref LW_TRUE    type is a sub-class implementing DOMGRP interfaces
 * @return  @ref LW_FALSE   type is a not sub-class of DOMGRP interfaces
 */
static LW_FORCEINLINE LwBool
LW2080_CTRL_PMGR_PWR_POLICY_TYPE_IMPLEMENTS_DOMGRP
(
   LwU32 type
)
{
    LwBool bImplements;

    switch (type)
    {
        case LW2080_CTRL_PMGR_PWR_POLICY_TYPE_WORKLOAD:
        case LW2080_CTRL_PMGR_PWR_POLICY_TYPE_BANG_BANG_VF:
        case LW2080_CTRL_PMGR_PWR_POLICY_TYPE_MARCH_VF:
        case LW2080_CTRL_PMGR_PWR_POLICY_TYPE_WORKLOAD_MULTIRAIL:
        case LW2080_CTRL_PMGR_PWR_POLICY_TYPE_WORKLOAD_COMBINED_1X:
            bImplements = LW_TRUE;
            break;
        default:
            bImplements = LW_FALSE;
            break;
    }

    return bImplements;
}

#endif // LW_SUBDEVICE_PMGR_INLINES_H_
