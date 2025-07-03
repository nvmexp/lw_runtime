/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the writte
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef _cla0bd_h_
#define _cla0bd_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define LWFBC_SW_SESSION                    (0x0000a0bd)

/*
 * LWA0BD_ALLOC_PARAMETERS
 *
  *   displayOrdinal
 *     This parameter specifies the display identifier.
 *   sessionType
 *     This parameter LWFBC session type. Possible values are specified 
 *     by LWA0BD_LWFBC_SESSION_TYPE_* macros.
 *   sessionFlags
 *     This parameter returns various flags values of the LWFBC session.
 *     Valid flag values include:
 *       LWA0BD_LWFBC_SESSION_FLAG_DIFFMAP_ENABLED
 *         When true this flag indicates there are user has enabled 
 *         diff map feature for the session.
 *       LWA0BD_LWFBC_SESSION_FLAG_CLASSIFICATIONMAP_ENABLED
 *         When true this flag indicates there are user has enabled 
 *         diff map feature for the session.
 *   hMaxResolution
 *     This parameter returns the max horizontal resolution supported by
 *     the LwFBC session.
 *   vMaxResolution
 *     This parameter returns the max vertical resolution supported by
 *     the LwFBC session.
 */

typedef struct
{
    LwU32    displayOrdinal;
    LwU32    sessionType;
    LwU32    sessionFlags;
    LwU32    hMaxResolution;
    LwU32    vMaxResolution;
} LWA0BD_ALLOC_PARAMETERS;

#define LWA0BD_LWFBC_SESSION_TYPE_UNKNOWN                                 0x000000
#define LWA0BD_LWFBC_SESSION_TYPE_TOSYS                                   0x000001
#define LWA0BD_LWFBC_SESSION_TYPE_LWDA                                    0x000002
#define LWA0BD_LWFBC_SESSION_TYPE_VID                                     0x000003
#define LWA0BD_LWFBC_SESSION_TYPE_HWENC                                   0x000004

#define LWA0BD_LWFBC_SESSION_FLAG_DIFFMAP_ENABLED                         0:0
#define LWA0BD_LWFBC_SESSION_FLAG_DIFFMAP_ENABLED_FALSE                   (0x00000000)
#define LWA0BD_LWFBC_SESSION_FLAG_DIFFMAP_ENABLED_TRUE                    (0x00000001)
#define LWA0BD_LWFBC_SESSION_FLAG_CLASSIFICATIONMAP_ENABLED               1:1
#define LWA0BD_LWFBC_SESSION_FLAG_CLASSIFICATIONMAP_ENABLED_FALSE         (0x00000000)
#define LWA0BD_LWFBC_SESSION_FLAG_CLASSIFICATIONMAP_ENABLED_TRUE          (0x00000001)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cla0bd_h
