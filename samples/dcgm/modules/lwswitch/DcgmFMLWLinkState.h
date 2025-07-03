#pragma once

#include <string>

#include "DcgmFMCommon.h"

/*****************************************************************************/
/*  Fabric Manager LWLink main and sublink states                            */
/*****************************************************************************/

/*
 * This class will translate the link state to human readable format.
 * All the members are defined as static as there is no specific member 
 * state to hold and no other class should create an instance for this class.
 */

// TODO - Mask/Translate any state which we don't want to show to user

class DcgmFMLWLinkState
{
public:
    static std::string getMainLinkState(uint32 linkMode);
    static std::string getTxSubLinkState(uint32 txSubLinkMode);
    static std::string getRxSubLinkState(uint32 rxSubLinkMode);
};
