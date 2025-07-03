/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include <string>

#include "FMCommonTypes.h"

/*****************************************************************************/
/*  Fabric Manager LWLink main and sublink states                            */
/*****************************************************************************/

/*
 * This class will translate the link state to human readable format.
 * All the members are defined as static as there is no specific member 
 * state to hold and no other class should create an instance for this class.
 */

// TODO - Mask/Translate any state which we don't want to show to user

class GlobalFMLWLinkState
{
public:
    static std::string getMainLinkState(uint32 linkMode);
    static std::string getTxSubLinkState(uint32 txSubLinkMode);
    static std::string getRxSubLinkState(uint32 rxSubLinkMode);
};
