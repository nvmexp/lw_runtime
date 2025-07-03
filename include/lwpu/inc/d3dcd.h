/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2002 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _D3DCD_H_
#define _D3DCD_H_

//******************************************************************************
//
// Module Name: D3DCD.H
//
// This file contains structures and constants that define the D3D Driver
// specific data for the crash dump file. The record definitions defined here
// are always stored after the crash dump file header. Each record defined here
// is preceded by the LWCD_RECORD structure.
//
//******************************************************************************
#include "lwtypes.h"
#include "lwcd.h"
typedef enum _D3DCD_RECORD_TYPE
{
    D3DHWData        = 0,
    D3DPusherData    = 1,
    D3DTextureData   = 2,
    D3DVBData        = 3,
    D3DContextData   = 4,
    D3DPusherData_V2 = 5
} D3DCD_RECORD_TYPE;

typedef struct _D3DHWData_RECORD
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwpDriverData;
    LwU32       dwClassFields1;
    LwU32       dwClassFields2;
    LwU32       dwClassFields3;
    LwU32       dwClassFields4;
    LwU32       dwGPUFeature;
} D3DHWData_RECORD, *PD3DHWData_RECORD;


typedef struct _D3DPusherData_RECORD
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwGet;         
    LwU32       dwPut;         
    LwU32       dwLastIssuedRef;
    LwU32       dwLastRecievedRef;
} D3DPusherData_RECORD, *PD3DPusherData_RECORD;

typedef struct _D3DPusherData_V2_RECORD
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwGet;         
    LwU32       dwPut;         
    LwU64       qwLastIssuedRef;
    LwU64       qwLastRecievedRef;
    LwU32       pLastRefPut;
} D3DPusherData_V2_RECORD, *PD3DPusherData_V2_RECORD;

typedef struct _D3DtextureData_RECORD
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwNumAllocated;         
    LwU32       dwNumFreed;         
    LwU32       dwNumFailed;
} D3DTextureData_RECORD, *PD3DTextureData_RECORD;

typedef struct _D3DVBData_RECORD
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwNumAllocated;         
    LwU32       dwNumFreed;         
    LwU32       dwNumFailed;
} D3DVBData_RECORD, *PD3DVBData_RECORD;

typedef struct _D3DContextData_RECORD
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwNumAllocated;         
    LwU32       dwNumFreed;         
    LwU32       dwNumFailed;
} D3DContextData_RECORD, *PD3DContextData_RECORD;

#endif
