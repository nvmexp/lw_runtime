/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include "fm_log.h"
#include "LocalFMInbandEventHndlr.h"

/*****************************************************************************/

LocalFMInbandEventHndlr::LocalFMInbandEventHndlr(LocalFabricManagerControl *pLfm)
{
    mpLfm = pLfm;
}

LocalFMInbandEventHndlr::~LocalFMInbandEventHndlr()
{
    // nothing as of now
}

void
LocalFMInbandEventHndlr::processLWSwitchInbandEvent(FMUuid_t &switchUuid)
{
    // TODO: process LWSwitch inband events
}
