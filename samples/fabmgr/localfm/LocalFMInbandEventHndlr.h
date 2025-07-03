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

#pragma once

#include "FMCommonTypes.h"

/*******************************************************************************/
/* Local FM LWSwitch LWLink In-band message event handler to read and process  */
/* received in-band messages                                                   */
/*******************************************************************************/

class LocalFabricManagerControl;

class LocalFMInbandEventHndlr
{

public:
    LocalFMInbandEventHndlr(LocalFabricManagerControl *pLfm);

    ~LocalFMInbandEventHndlr();

    void processLWSwitchInbandEvent(FMUuid_t &switchUuid);

private:
    LocalFabricManagerControl *mpLfm;
};
