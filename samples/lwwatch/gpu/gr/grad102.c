/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grga102.c
//
//*****************************************************

//
// includes
//
#include "ada/ad102/hwproject.h"
#include "gr.h"


/*!
 * @brief   Function to get the max number of Gpcs
 *
 * @return  Returns LwU32      The max number of Gpcs
 *
 */
LwU32 grGetMaxGpc_AD102(void)
{
    return LW_SCAL_LITTER_NUM_GPCS;
}
