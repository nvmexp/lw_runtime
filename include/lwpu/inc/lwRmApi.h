 /***************************************************************************\
|*                                                                           *|
|*       Copyright 1993-2019 LWPU, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     NOTICE TO USER:   The source code  is copyrighted under  U.S. and     *|
|*     international laws.  LWPU, Corp. of Sunnyvale,  California owns     *|
|*     copyrights, patents, and has design patents pending on the design     *|
|*     and  interface  of the LW chips.   Users and  possessors  of this     *|
|*     source code are hereby granted a nonexclusive, royalty-free copy-     *|
|*     right  and design patent license  to use this code  in individual     *|
|*     and commercial software.                                              *|
|*                                                                           *|
|*     Any use of this source code must include,  in the user dolwmenta-     *|
|*     tion and  internal comments to the code,  notices to the end user     *|
|*     as follows:                                                           *|
|*                                                                           *|
|*     Copyright  1993-2014  LWPU,  Corporation.   LWPU  has  design     *|
|*     patents and patents pending in the U.S. and foreign countries.        *|
|*                                                                           *|
|*     LWPU, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY     *|
|*     OF THIS SOURCE CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITH-     *|
|*     OUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORPORATION     *|
|*     DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOURCE CODE, INCLUD-     *|
|*     ING ALL IMPLIED WARRANTIES  OF MERCHANTABILITY  AND FITNESS FOR A     *|
|*     PARTICULAR  PURPOSE.  IN NO EVENT  SHALL LWPU,  CORPORATION  BE     *|
|*     LIABLE FOR ANY SPECIAL,  INDIRECT,  INCIDENTAL,  OR CONSEQUENTIAL     *|
|*     DAMAGES, OR ANY DAMAGES  WHATSOEVER  RESULTING  FROM LOSS OF USE,     *|
|*     DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR     *|
|*     OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION  WITH THE     *|
|*     USE OR PERFORMANCE OF THIS SOURCE CODE.                               *|
|*                                                                           *|
|*     RESTRICTED RIGHTS LEGEND:  Use, duplication, or disclosure by the     *|
|*     Government is subject  to restrictions  as set forth  in subpara-     *|
|*     graph (c) (1) (ii) of the Rights  in Technical Data  and Computer     *|
|*     Software  clause  at DFARS  52.227-7013 and in similar clauses in     *|
|*     the FAR and NASA FAR Supplement.                                      *|
|*                                                                           *|
 \***************************************************************************/

/*
 * lwRmApi.h
 *
 * LWpu resource manager API header file exported to drivers.
 *
 */

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

#include "lwos.h"

#ifndef WINNT
LwU32  LW_APIENTRY LwRmAlloc(LwHandle hClient, LwHandle hParent, LwHandle hObject, LwU32 hClass, void *pAllocParms);
LwU32  LW_APIENTRY LwRmAllocWithAccess(LwHandle hClient, LwHandle hParent, LwHandle *phObject, LwU32 hClass, void *pAllocParms, const RS_ACCESS_MASK *pRightsRequired);
LwU32  LW_APIENTRY LwRmAllocRoot(LwHandle *phClient);
LwU32  LW_APIENTRY LwRmAllocRootWithAccess(LwHandle *phClient, const RS_ACCESS_MASK *pRightsRequired);
LwU32  LW_APIENTRY LwRmGetDeviceReference(LwU32 *dwRetIsLW, LwU32 size);
LwU32  LW_APIENTRY LwRmAllocMemory(LwHandle hClient, LwHandle hParent, LwHandle hMemory, LwU32 hClass, LwU32 flags, void **ppAddress, LwU32 *pLimit);
LwU32  LW_APIENTRY LwRmAllocMemory64(LwHandle hClient, LwHandle hParent, LwHandle hMemory, LwU32 hClass, LwU32 flags, void **ppAddress, LwU64 *pLimit);
LwU32  LW_APIENTRY LwRmAllocObject(LwHandle hClient, LwHandle hChannel, LwHandle hObject, LwU32 hClass);
LwU32  LW_APIENTRY LwRmFree(LwHandle hClient, LwHandle hParent, LwHandle hObject);
LwU32  LW_APIENTRY LwRmControl(LwHandle hClient, LwHandle hObject, LwU32 cmd, void *pParams, LwU32 paramsSize);
LwU32  LW_APIENTRY LwRmControlWithGpuId(LwHandle hClient, LwHandle hObject, LwU32 cmd, void *pParams, LwU32 paramsSize, LwU32 *pRmGpuId);
LwU32  LW_APIENTRY LwRmAllocEvent(LwHandle hClient, LwHandle hParent, LwHandle hObject, LwU32 hClass, LwU32 index, void *hEvent);
LwU32  LW_APIENTRY LwRmVidHeapControl(void *pVidHeapControlParms);
LwU32  LW_APIENTRY LwRmConfigGet(LwHandle hClient, LwHandle hDevice, LwU32 index, LwU32 *pValue);
LwU32  LW_APIENTRY LwRmConfigSet(LwHandle hClient, LwHandle hDevice, LwU32 index, LwU32 newValue, LwU32 *pOldValue);
LwU32  LW_APIENTRY LwRmConfigSetEx(LwHandle hClient, LwHandle hDevice, LwU32 index, void * paramStructPtr, LwU32 ParamSize);
LwU32  LW_APIENTRY LwRmConfigGetEx(LwHandle hClient, LwHandle hDevice, LwU32 index, void * paramStructPtr, LwU32 ParamSize);
LwU32  LW_APIENTRY LwRmI2CAccess(LwHandle hClient, LwHandle hDevice, void * ctrlStructPtr);
LwU32  LW_APIENTRY LwRmIdleChannels(LwHandle hClient, LwHandle hDevice, LwHandle hChannel, LwU32 numChannels, LwHandle *phClients, LwHandle *phDevices, LwHandle *phChannels, LwU32 flags, LwU32 timeout);
LwU32  LW_APIENTRY LwRmMapMemory(LwHandle hClient, LwHandle hDevice, LwHandle hMemory, LwU64 offset, LwU64 length, void **ppLinearAddress, LwU32 flags);
LwU32  LW_APIENTRY LwRmUnmapMemory(LwHandle hClient, LwHandle hDevice, LwHandle hMemory, void *pLinearAddress, LwU32 flags);
LwU32  LW_APIENTRY LwRmAddVblankCallback(LwHandle hClient, LwHandle hDevice, LwHandle hVblank, void *pProc, LwU32 logicalHead, void *pParm1, void *pParm2, LwBool bAdd);
#if LWOS_IS_UNIX || LWOS_IS_QNX || LWOS_IS_INTEGRITY || LWOS_IS_HOS
LwU32  LW_APIENTRY LwRmReadRegistryDword(LwHandle hClient, LwHandle hObject, void *DevNode, const void *ParmStr, LwU32 *Data);
LwU32  LW_APIENTRY LwRmWriteRegistryDword(LwHandle hClient, LwHandle hObject, void *DevNode, const void *ParmStr, LwU32 Data);
LwU32  LW_APIENTRY LwRmReadRegistryBinary(LwHandle hClient, LwHandle hObject, void *DevNode, const void *ParmStr, void **Data, LwU32 *Length);
LwU32  LW_APIENTRY LwRmWriteRegistryBinary(LwHandle hClient, LwHandle hObject, void *DevNode, const void *ParmStr, void *Data, LwU32 Length);
LwU32  LW_APIENTRY LwRmAllocOsEvent(LwHandle hClient, LwHandle hDevice, LwHandle *hOsEvent, void *fd);
LwU32  LW_APIENTRY LwRmFreeOsEvent(LwHandle hClient, LwHandle hDevice, LwU32 fd);
LwU32  LW_APIENTRY LwRmGetEventData(LwHandle hClient, LwU32 fd, void *pEventData, LwU32 *pMoreEvents);
#endif
LwU32  LW_APIENTRY LwRmAllocContextDma2(LwHandle hClient, LwHandle hDma, LwU32 hClass, LwU32 flags, LwHandle hMemory, LwU64 offset, LwU64 limit);
LwU32  LW_APIENTRY LwRmBindContextDma(LwHandle hClient, LwHandle hChannel, LwHandle hCtxDma);
LwU32  LW_APIENTRY LwRmMapMemoryDma(LwHandle hClient, LwHandle hDevice, LwHandle hDma, LwHandle hMemory, LwU64 offset, LwU64 length, LwU32 flags, LwU64 *pDmaOffset);
LwU32  LW_APIENTRY LwRmUnmapMemoryDma(LwHandle hClient, LwHandle hDevice, LwHandle hDma, LwHandle hMemory, LwU32 flags, LwU64 dmaOffset);
LwU32  LW_APIENTRY LwRmDupObject(LwHandle hClient, LwHandle hParent, LwHandle hObjectDest, LwHandle hClientSrc, LwHandle hObjectSrc, LwU32 flags);
LwU32  LW_APIENTRY LwRmDupObject2(LwHandle hClient, LwHandle hParent, LwHandle *phObjectDest, LwHandle hClientSrc, LwHandle hObjectSrc, LwU32 flags);
LwU32  LW_APIENTRY LwRmShare(LwHandle hClient, LwHandle hObjectDest, const RS_SHARE_POLICY *pSharePolicy);
#if LWOS_IS_UNIX
LwU32  LW_APIENTRY LwRmCheckVersion(LwHandle hClient);
#endif
#endif // #ifndef WINNT

#if LWOS_IS_WINDOWS
LwU32 LW_APIENTRY LwRmEnumerateGpusByAdapterType(LwU32 adapterType, LwU32 *adapterCount, LwU32 *rmGpuIds, LwU32 maxRmGpuIdCount);
#endif

/*
** Export the RM API Initialize function for binaries that want control over 
** initialization time.  The OpenGL ICD needs to execute this at process attach
** time, to be thread safe.  If not, it will have a lockup issue with Cylthrd.
*/
void LwRmApiInitialize(void);

#define RMGPU_ILWALID_ID (0xffffffff)

#ifdef __cplusplus
}
#endif //__cplusplus
