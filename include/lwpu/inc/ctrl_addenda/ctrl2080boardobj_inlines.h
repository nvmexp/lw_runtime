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

#ifndef LW_SUBDEVICE_BOARDOBJ_INLINES_H_
#define LW_SUBDEVICE_BOARDOBJ_INLINES_H_

#include "ctrl/ctrl2080/ctrl2080boardobj.h"

//
// This file is for any helper functions that are REQUIRED to live in the
// SDK. Because FINN cannot represent helper functions, these need to live
// outside of the ctrl/ directory (will be FINN-generated) and this file
// will not ported until FINN adds support for generating function definitons.
//

/*!
 * Helper function which to colwert a LwBoardObjIdx to LwU8, handling
 * any necessary truncation above @ref LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT.
 *
 * @param[in]  grpIdx   LwBoardObjIdx value to to colwert to 8bit.
 *
 * @return grpIdx as 8-bit value, ceilinged to
 * LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT.
 */
static LW_FORCEINLINE LwU8
LW2080_CTRL_BOARDOBJ_IDX_TO_8BIT
(
   LwBoardObjIdx grpIdx
)
{
    //
    // Handle cases where grpIdx is larger than 8bits and cast to
    // LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT, as any value above 8-bits
    // is equally invalid as an 8-bit value.
    //
    if (grpIdx > LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT)
    {
        return LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT;
    }

    return (LwU8)grpIdx;
}

/*!
 * Helper function which to provide a LwBoardObjIdx index from an LwU8
 * index input, handling any necessary colwersion
 * LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT -> LW2080_CTRL_BOARDOBJ_IDX_ILWALID.
 *
 * @param[in]  grpIdx   8bit index value to to colwert to LwBoardObjIdx.
 *
 * @return grpIdx as an LwBoardObjIdx value, including necessary colwersion
 * LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT -> LW2080_CTRL_BOARDOBJ_IDX_ILWALID.
 */
static LW_FORCEINLINE LwBoardObjIdx
LW2080_CTRL_BOARDOBJ_IDX_FROM_8BIT
(
   LwU8 grpIdx
)
{
    return (grpIdx == LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT) ?
        LW2080_CTRL_BOARDOBJ_IDX_ILWALID :
        (LwBoardObjIdx)grpIdx;
}

#endif // LW_SUBDEVICE_BOARDOBJ_INLINES_H_
