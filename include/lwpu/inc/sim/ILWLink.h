/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2006-2008,2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

// DO NOT EDIT
// See https://wiki.lwpu.com/engwiki/index.php/MODS/sim_linkage#How_to_change_ifspec

#ifndef _ILWLINK_H_
#define _ILWLINK_H_

#include "ITypes.h"
#include "IIface.h"
#include "LWLink_AN1_pkt.h"

class ILWLink : public IIfaceObject {
public:
    // Data from SIMICS to GPU
    virtual void LWLink_CPU2GPU(LWLink_AN1_pkt *req_pkt) = 0;

    // IIfaceObject Interface
    virtual void AddRef() = 0;
    virtual void Release() = 0;
    virtual IIfaceObject* QueryIface(IID_TYPE id) = 0;
};

class ILWLinkCallbacks : public IIfaceObject {
public:
    // Data from GPU to SIMICS
    virtual void LWLinkPktGPU2CPU(LWLink_AN1_pkt *req_pkt, LWLink_AN1_pkt *resp_pkt) = 0;
    virtual void LWLinkPktGPU2CPU_ASYNC(LWLink_AN1_pkt *req_pkt) = 0;

    // IIfaceObject Interface
    virtual void AddRef() = 0;
    virtual void Release() = 0;
    virtual IIfaceObject* QueryIface(IID_TYPE id) = 0;
};

#endif // _ILWLINK_H_
