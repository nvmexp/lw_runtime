/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

// DO NOT EDIT
// See https://wiki.lwpu.com/engwiki/index.php/MODS/sim_linkage#How_to_change_ifspec

#ifndef _IINTERRUPT2_H
#define _IINTERRUPT2_H

#include "ITypes.h"
#include "IIface.h"

class IInterrupt2 : public IIfaceObject {
public:
    virtual void HandleInterrupt() = 0; 
    virtual void DeassertInterrupt() = 0; 

    virtual void AddRef() = 0;
    virtual void Release() = 0;
    virtual IIfaceObject* QueryIface(IID_TYPE id) = 0;
};

#endif // _IINTERRUPT2_H_
