/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - July 2004
// OutputCb.h - Output Callback COM class
// 
//*****************************************************

#ifndef _OUTPUTCB_H
#define _OUTPUTCB_H

#include <stdio.h>
#include "lwwatch.h"

class OutputCb: public IDebugOutputCallbacks{

public:

    OutputCb();

    ~OutputCb();

    STDMETHOD(Output)
    (
        IN ULONG Mask,
        IN PCSTR Text
    );

    STDMETHOD(QueryInterface)
    (
        THIS_
        __in REFIID InterfaceId,
        __out PVOID* Interface
    ); 

    STDMETHOD_(ULONG, AddRef)
    (
        THIS
    );    

    STDMETHOD_(ULONG, Release)
    (
        THIS
    );

private:
    //
    // Stores reference count
    //
    ULONG ref;
    
};

typedef OutputCb * PLWWATCH_DEBUG_OUTPUT_CALLBACK;

#endif // _OUTPUTCB_H
