/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2010-2014 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl83de_h_
#define _cl83de_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GT200_DEBUGGER                                            (0x000083de)

/*
 * Creating the GT200_DEBUGGER object:
 * - The debug object is instantiated as a child of either the compute or the
 *   3D-class object.
 * - The Lwca/GR debugger uses the LW83DE_ALLOC_PARAMETERS to fill in the Client
 *   and 3D-Class handles of the debuggee and passes this to the LwRmAlloc.
 *   e.g:
        LW83DE_ALLOC_PARAMETERS params;
 *      memset (&params, 0, sizeof (LW83DE_ALLOC_PARAMETERS));
 *      params.hAppClient = DebuggeeClient;
 *      params.hClass3dObject = 3DClassHandle;
 *      LwRmAlloc(hDebuggerClient, hDebuggerClient, hDebugger, GT200_DEBUGGER, &params);
 */

typedef struct {
    LwHandle    hDebuggerClient_Obsolete;  // No longer supported (must be zero)
    LwHandle    hAppClient;
    LwHandle    hClass3dObject;
} LW83DE_ALLOC_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl83de_h_ */

