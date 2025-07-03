/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 1999-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

// DO NOT EDIT
// See https://wiki.lwpu.com/engwiki/index.php/MODS/sim_linkage#How_to_change_ifspec

#ifndef IFSPEC_2_H
#define IFSPEC_2_H   

#include "IChip.h"
#include "ILWLink.h"
#include "ICPUModel.h"
#include "IInterrupt4.h"

extern "C" {
IIfaceObject* call_client_QueryIface(IIfaceObject* system, IID_TYPE id);
void      call_client_Release(IIfaceObject* system);
BusMemRet call_client_BusMemWrBlk(IBusMem*, LwU064 address, const void *appdata, LwU032 count);
BusMemRet call_client_BusMemRdBlk(IBusMem*, LwU064 address, void *appdata, LwU032 count);
BusMemRet call_client_BusMemCpBlk(IBusMem*, LwU064 dest, LwU064 source, LwU032 count);
BusMemRet call_client_BusMemSetBlk(IBusMem*, LwU064 address, LwU032 size, void* data, LwU032 data_size);
void      call_client_HandleInterrupt(IInterrupt* system);
void      call_client_HandleInterrupt2(IInterrupt2* system);
void      call_client_HandleSpecificInterrupt3(IInterrupt3* system, LwU032 irqNumber);
void      call_client_HandleInterruptVectorChange(IInterrupt4* system, const LwU032 *v, LwU032 n);
void      call_client_LWLinkPktGPU2CPU(ILWLinkCallbacks*, LWLink_AN1_pkt *req_pkt, LWLink_AN1_pkt *resp_pkt);
void      call_client_LWLinkPktGPU2CPU_ASYNC(ILWLinkCallbacks*, LWLink_AN1_pkt *req_pkt);
void      call_client_CPUModelAtsRequest2(ICPUModelCallbacks2*, struct LwPciDev bdf, LwU032 pasid, LwU064 address,
                bool isGpa, LwU032 numPages, bool numPageAlign, LwU032 *pageSize, CPUModelAtsResult *results);
}

typedef void(*void_call_void)(IIfaceObject*);
typedef IIfaceObject*(*call_IIfaceObject)(IIfaceObject*, IID_TYPE);
typedef int(*call_Startup)(IIfaceObject*, IIfaceObject* system, char** argv, int argc);
typedef int (*call_AllocSysMem)(IIfaceObject*, int numBytes, LwU032* physAddr);
typedef void (*call_FreeSysMem)(IIfaceObject*, LwU032 physAddr);
typedef void (*call_ClockSimulator)(IIfaceObject*, LwS032 numClocks);
typedef void (*call_Delay)(IIfaceObject*, LwU032 numMicroSeconds);
typedef int (*call_EscapeWrite)(IIfaceObject*, char* path, LwU032 index, LwU032 size, LwU032 value);
typedef int (*call_EscapeRead)(IIfaceObject*, char* path, LwU032 index, LwU032 size, LwU032* value);
typedef int (*call_FindPCIDevice)(IIfaceObject*, LwU016 vendorId, LwU016 deviceId, int index, LwU032* address);
typedef int (*call_FindPCIClassCode)(IIfaceObject*, LwU032 classCode, int index, LwU032* address);
typedef int (*call_GetSimulatorTime)(IIfaceObject*, LwU064* simTime);
typedef double (*call_GetSimulatorTimeUnitsNS)(IIfaceObject*);
typedef IChip::ELEVEL (*call_GetChipLevel)(IIfaceObject*);

typedef BusMemRet (*call_BusMemWrBlk)(IBusMem*, LwU064, const void *, LwU032);
typedef BusMemRet (*call_BusMemRdBlk)(IBusMem*, LwU064, void *, LwU032);
typedef BusMemRet (*call_BusMemCpBlk)(IBusMem*, LwU064, LwU064, LwU032);
typedef BusMemRet (*call_BusMemSetBlk)(IBusMem*, LwU064, LwU032, void*, LwU032);

typedef void (*call_LWLink_CPU2GPU)(ILWLink*, LWLink_AN1_pkt *req_pkt);
typedef void (*call_LWLinkPktGPU2CPU)(ILWLinkCallbacks*, LWLink_AN1_pkt *req_pkt, LWLink_AN1_pkt *resp_pkt);
typedef void (*call_LWLinkPktGPU2CPU_ASYNC)(ILWLinkCallbacks*, LWLink_AN1_pkt *req_pkt);

typedef void (*call_AllocContextDma)(IIfaceObject*, LwU032, LwU032, LwU032, LwU032, LwU032, LwU032, 
                                                    LwU032, LwU032 *);
typedef void (*call_AllocChannelDma)(IIfaceObject*, LwU032, LwU032, LwU032, LwU032);
typedef void (*call_FreeChannel)(IIfaceObject*, LwU032);
typedef void (*call_AllocObject)(IIfaceObject*, LwU032, LwU032, LwU032);
typedef void (*call_FreeObject)(IIfaceObject*, LwU032, LwU032);
typedef void (*call_ProcessMethod)(IIfaceObject*, LwU032, LwU032, LwU032, LwU032);
typedef void (*call_AllocChannelGpFifo)(IIfaceObject*, LwU032, LwU032, LwU032, LwU064, LwU032, LwU032);
typedef bool (*call_PassAdditionalVerification)(IIfaceObject*, const char *traceFileName);
typedef const char* (*call_GetModelIdentifierString)(IIfaceObject*);

typedef LwU008 (*call_IoRd08)(IIfaceObject*,  LwU016 address);
typedef LwU016 (*call_IoRd16)(IIfaceObject*,  LwU016 address);
typedef LwU032 (*call_IoRd32)(IIfaceObject*,  LwU016 address);
typedef void (*call_IoWr08)(IIfaceObject*,  LwU016 address, LwU008 data);
typedef void (*call_IoWr16)(IIfaceObject*,  LwU016 address, LwU016 data);
typedef void (*call_IoWr32)(IIfaceObject*,  LwU016 address, LwU032 data);
typedef LwU008 (*call_CfgRd08)(IIfaceObject*,  LwU032 address);
typedef LwU016 (*call_CfgRd16)(IIfaceObject*,  LwU032 address);
typedef LwU032 (*call_CfgRd32)(IIfaceObject*,  LwU032 address);
typedef void (*call_CfgWr08)(IIfaceObject*,  LwU032 address, LwU008 data);
typedef void (*call_CfgWr16)(IIfaceObject*,  LwU032 address, LwU016 data);
typedef void (*call_CfgWr32)(IIfaceObject*,  LwU032 address, LwU032 data);

typedef void (*call_HandleInterrupt)(IInterrupt*);
typedef void (*call_HandleInterrupt2)(IInterrupt2*);
typedef void (*call_DeassertInterrupt)(IInterrupt2*);
typedef void (*call_HandleSpecificInterrupt3)(IInterrupt3*, LwU032);
typedef void (*call_HandleInterruptVectorChange)(IInterrupt4*, const LwU032*, LwU032);

typedef void (*call_CPUModelInitialize2)(ICPUModel2*);
typedef void (*call_CPUModelEnableResponse2)(ICPUModel2*, bool enable);
typedef void (*call_CPUModelEnableResponseSpecific2)(ICPUModel2*, bool enable, CPUModelEvent event);
typedef bool (*call_CPUModelHasResponse2)(ICPUModel2*);
typedef void (*call_CPUModelGetResponse2)(ICPUModel2*, CPUModelResponse *response);
typedef CPUModelRet (*call_CPUModelRead2)(ICPUModel2*, LwU032 uniqueId, LwU064 address, LwU032 sizeBytes, bool isCoherentlyCaching, bool isProbing);
typedef CPUModelRet (*call_CPUModelWrite2)(ICPUModel2*, LwU032 uniqueId, LwU064 address, LwU032 offset, LwU064 data, LwU032 sizeBytes, bool isCoherentlyCaching, bool isProbing, bool isPosted);
typedef CPUModelRet (*call_CPUModelGetCacheData2)(ICPUModel2*, LwU064 gpa, LwU064 *data);
typedef void (*call_CPUModelAtsShootDown2)(ICPUModel2*, LwU032 uniqueId, struct LwPciDev bdf, LwU032 pasid,
                LwU064 address, LwU032 atSize, bool isGpa, bool flush, bool isGlobal);

typedef void (*call_CPUModelAtsRequest2)(ICPUModelCallbacks2*, struct LwPciDev bdf, LwU032 pasid, LwU064 address,
                bool isGpa, LwU032 numPages, bool numPageAlign, LwU032 *pageSize, CPUModelAtsResult *results);

struct BusMemVTable
{
    void_call_void Release;

    call_BusMemWrBlk BusMemWrBlk;
    call_BusMemRdBlk BusMemRdBlk;
    call_BusMemCpBlk BusMemCpBlk;
    call_BusMemSetBlk BusMemSetBlk;

    BusMemVTable() : Release(0), BusMemWrBlk(0), BusMemRdBlk(0), BusMemCpBlk(0), BusMemSetBlk(0) {}

    bool SanityCheck()
    {
        return Release != 0 && BusMemWrBlk != 0 && BusMemRdBlk != 0 && BusMemCpBlk != 0 && BusMemSetBlk != 0;
    }
};

struct InterruptVTable
{
    void_call_void Release;
    call_HandleInterrupt HandleInterrupt;

    bool SanityCheck()
    {
        return Release != 0 && HandleInterrupt != 0;
    }
};

struct Interrupt2VTable
{
    void_call_void Release;
    call_HandleInterrupt2 HandleInterrupt;
    call_DeassertInterrupt DeassertInterrupt;

    bool SanityCheck()
    {
        return Release != 0 && HandleInterrupt != 0 && DeassertInterrupt != 0;
    }
};

struct Interrupt3VTable
{
    void_call_void Release;
    call_HandleSpecificInterrupt3 HandleSpecificInterrupt;

    bool SanityCheck()
    {
        return Release != 0 && HandleSpecificInterrupt != 0;
    }
};

struct Interrupt4VTable
{
    void_call_void Release;
    call_HandleInterruptVectorChange HandleInterruptVectorChange;

    bool SanityCheck()
    {
        return Release != 0 && HandleInterruptVectorChange != 0;
    }
};

struct LWLinkCallbacksVTable
{
    void_call_void Release;

    call_LWLinkPktGPU2CPU LWLinkPktGPU2CPU;
    call_LWLinkPktGPU2CPU_ASYNC LWLinkPktGPU2CPU_ASYNC;

    LWLinkCallbacksVTable() : Release(0), LWLinkPktGPU2CPU(0), LWLinkPktGPU2CPU_ASYNC(0) {}

    bool SanityCheck()
    {
        return Release != 0 && LWLinkPktGPU2CPU != 0 && LWLinkPktGPU2CPU_ASYNC != 0;
    }

};

struct CPUModelCallbacks2VTable
{
    void_call_void Release;
    call_CPUModelAtsRequest2 CPUModelAtsRequest;

    CPUModelCallbacks2VTable() : Release(0), CPUModelAtsRequest(0) {}

    bool SanityCheck()
    {
        return Release != 0 && CPUModelAtsRequest != 0;
    }
};

#endif /* IFSPEC_2_H */

