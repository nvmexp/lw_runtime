/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 1999-2007 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef INCLUDED_SHIM_H
#define INCLUDED_SHIM_H

#include "IIface.h"
#include "IChip.h"
#include "IBusMem.h"
#include "IInterrupt.h"
#include "IInterrupt2.h"
#include "IInterrupt2a.h"
#include "IInterrupt3.h"
#include "IAModelBackdoor.h"
#include "IMemory.h"
#include "IMemAlloc64.h"
#include "IIo.h"
#include "ICPUModel.h"
#include "ifspec-2.h"

#if defined(MODS)
#include "platform.h"
#endif

#if !defined(MASSERT)
#include <assert.h>
#define MASSERT(cond) assert(cond)
#endif

class ChipProxy : public IChip
{
public:
    ChipProxy(void* module, IChip* chip);
    ~ChipProxy(){};

    // IIfaceObject Interface
    void AddRef();
    void Release();
    IIfaceObject* QueryIface(IID_TYPE id);

    // IChip Interface
    int Startup(IIfaceObject* system, char** argv, int argc);
    void Shutdown();
    int AllocSysMem(int numBytes, LwU032* physAddr);
    void FreeSysMem(LwU032 physAddr);
    void ClockSimulator(LwS032 numClocks);
    void Delay(LwU032 numMicroSeconds);
    int EscapeWrite(char* path, LwU032 index, LwU032 size, LwU032 value);
    int EscapeRead(char* path, LwU032 index, LwU032 size, LwU032* value);
    int FindPCIDevice(LwU016 vendorId, LwU016 deviceId, int index, LwU032* address);
    int FindPCIClassCode(LwU032 classCode, int index, LwU032* address);
    int GetSimulatorTime(LwU064* simTime);
    double GetSimulatorTimeUnitsNS();
    IChip::ELEVEL GetChipLevel();

protected:
    int Init(char** argv, int argc){MASSERT(0); return 0;};    // depricated.  use Startup method

    void*  m_Module;
    IChip* m_Chip;

private:
    ChipProxy(const ChipProxy&);
    ChipProxy& operator=(const ChipProxy&);

    struct VTable
    {
        void_call_void AddRef;
        void_call_void Release;
        call_IIfaceObject QueryIface;

        call_Startup      Startup;
        void_call_void    Shutdown;
        call_AllocSysMem AllocSysMem;
        call_FreeSysMem FreeSysMem;
        call_ClockSimulator ClockSimulator;
        call_Delay Delay;
        call_EscapeWrite EscapeWrite;
        call_EscapeRead EscapeRead;
        call_FindPCIDevice FindPCIDevice;
        call_FindPCIClassCode FindPCIClassCode;
        call_GetSimulatorTime GetSimulatorTime;
        call_GetSimulatorTimeUnitsNS GetSimulatorTimeUnitsNS;
        call_GetChipLevel GetChipLevel;

        VTable() : AddRef(0), Release(0), QueryIface(0), 
                   Startup(0), Shutdown(0), AllocSysMem(0), FreeSysMem(0),
                   ClockSimulator(0), Delay(0), EscapeWrite(0), EscapeRead(0),
                   FindPCIDevice(0), FindPCIClassCode(0), GetSimulatorTime(0),
                   GetSimulatorTimeUnitsNS(0), GetChipLevel(0) {}

        bool SanityCheck()
        {
            return AddRef != 0 && Release != 0 && QueryIface != 0 &&
                   Startup != 0 && Shutdown != 0 && AllocSysMem != 0 && FreeSysMem != 0 &&
                   ClockSimulator != 0 && Delay != 0 && EscapeWrite != 0 && EscapeRead != 0 &&
                   FindPCIDevice != 0 && FindPCIClassCode != 0 && GetSimulatorTime != 0 &&
                   GetSimulatorTimeUnitsNS != 0 && GetChipLevel != 0;
        }
    };

    VTable m_VTable;
    unsigned long m_RefCount;
};

class CPUModel2Proxy : public ICPUModel2
{
public:
    CPUModel2Proxy(void* module, IIfaceObject* cpu);
    ~CPUModel2Proxy(){}

    void CPUModelInitialize(void);
    void CPUModelEnableResponse(bool enable);
    void CPUModelEnableResponseSpecific(bool enable, CPUModelEvent event);
    bool CPUModelHasResponse(void);
    void CPUModelGetResponse(CPUModelResponse *response);
    CPUModelRet CPUModelRead(LwU032 uniqueId, LwU064 address, LwU032 sizeBytes,
                bool isCoherentlyCaching, bool isProbing);
    CPUModelRet CPUModelGetCacheData(LwU064 gpa, LwU064 *data);
    CPUModelRet CPUModelWrite(LwU032 uniqueId, LwU064 address, LwU032 offset, LwU064 data,
                LwU032 sizeBytes, bool isCoherentlyCaching, bool isProbing, bool isPosted);
    void CPUModelAtsShootDown(LwU032 uniqueId, struct LwPciDev bdf, LwU032 pasid, LwU064 address,
                LwU032 sizeBytes, bool isGpa, bool flush, bool isGlobal);

    void AddRef() {MASSERT(!"Not implemented");}
    void Release();
    IIfaceObject* QueryIface(IID_TYPE id) {MASSERT(!"Not implemented");return 0;}

private:
    CPUModel2Proxy(const CPUModel2Proxy&);
    CPUModel2Proxy& operator=(const CPUModel2Proxy&);

    ICPUModel2 *m_cpuModel;

    struct VTable
    {
        void_call_void Release;
        call_CPUModelInitialize2 CPUModelInitialize;
        call_CPUModelEnableResponse2 CPUModelEnableResponse;
        call_CPUModelEnableResponseSpecific2 CPUModelEnableResponseSpecific;
        call_CPUModelHasResponse2 CPUModelHasResponse;
        call_CPUModelGetResponse2 CPUModelGetResponse;
        call_CPUModelRead2 CPUModelRead;
        call_CPUModelGetCacheData2 CPUModelGetCacheData;
        call_CPUModelWrite2 CPUModelWrite;
        call_CPUModelAtsShootDown2 CPUModelAtsShootDown;

        VTable() : Release(0),
                   CPUModelInitialize(0),
                   CPUModelEnableResponse(0),
                   CPUModelEnableResponseSpecific(0),
                   CPUModelHasResponse(0),
                   CPUModelGetResponse(0),
                   CPUModelRead(0),
                   CPUModelGetCacheData(0),
                   CPUModelWrite(0),
                   CPUModelAtsShootDown(0) {}

        bool SanityCheck()
        {
            return Release != 0 &&
                   CPUModelInitialize != 0 &&
                   CPUModelEnableResponse != 0 &&
                   CPUModelEnableResponseSpecific != 0 &&
                   CPUModelHasResponse != 0 &&
                   CPUModelGetResponse != 0 &&
                   CPUModelRead != 0 &&
                   CPUModelGetCacheData != 0 &&
                   CPUModelWrite != 0 &&
                   CPUModelAtsShootDown != 0;
        }
    };

    VTable m_VTable;
};

class BusMemProxy : public IBusMem 
{
public:
    BusMemProxy(void* module, IIfaceObject* bus);
    ~BusMemProxy(){}

    BusMemRet BusMemWrBlk(LwU064 address, const void *appdata, LwU032 count);
    BusMemRet BusMemRdBlk(LwU064 address, void *appdata, LwU032 count);
    BusMemRet BusMemCpBlk(LwU064 dest, LwU064 source, LwU032 count);
    BusMemRet BusMemSetBlk(LwU064 address, LwU032 size, void* data, LwU032 data_size);

    void AddRef() {MASSERT(!"Not implemented");}
    void Release();
    IIfaceObject* QueryIface(IID_TYPE id) {MASSERT(!"Not implemented");return 0;}

private:
    BusMemProxy(const BusMemProxy&);
    BusMemProxy& operator=(const BusMemProxy&);

    IBusMem* m_Bus;
    BusMemVTable m_VTable;
};

class AModelBackdoorProxy : public IAModelBackdoor
{
public:
    AModelBackdoorProxy(void* module, IIfaceObject* bus);
    ~AModelBackdoorProxy(){}

    // IIfaceObject Interface
    void AddRef(){MASSERT(!"Not implemented");}
    void Release();
    IIfaceObject* QueryIface(IID_TYPE id){MASSERT(!"Not implemented");return 0;}

    void AllocContextDma(LwU032 ChID, LwU032 Handle, LwU032 Class, LwU032 target, LwU032 Limit, LwU032 Base, LwU032 Protect, LwU032 *PageTable);
    void AllocChannelDma(LwU032 ChID, LwU032 Class, LwU032 CtxDma, LwU032 ErrorNotifierCtxDma);
    void FreeChannel(LwU032 ChID);
    void AllocObject(LwU032 ChID, LwU032 Handle, LwU032 Class);
    void FreeObject(LwU032 ChID, LwU032 Handle);

    // !!!
    void ProcessMethod(LwU032 ChID, LwU032 Subch, LwU032 MethodAddr, LwU032 MethodData ){MASSERT(!"Not implemented");}

    void AllocChannelGpFifo(LwU032 ChID, LwU032 Class, LwU032 CtxDma, LwU064 GpFifoOffset, 
        LwU032 GpFifoEntries, LwU032 ErrorNotifierCtxDma );
    bool PassAdditionalVerification( const char *traceFileName );
    const char* GetModelIdentifierString();

    // deprecated
    char GetModelIdentifier() { return 'a'; }

private:
    AModelBackdoorProxy(const AModelBackdoorProxy&);
    AModelBackdoorProxy& operator=(const AModelBackdoorProxy&);

    IIfaceObject* m_AModelBackdoor;

    struct VTable
    {
        void_call_void Release;

        call_AllocContextDma AllocContextDma;
        call_AllocChannelDma AllocChannelDma;
        call_FreeChannel  FreeChannel;
        call_AllocObject AllocObject;
        call_FreeObject FreeObject;
        call_AllocChannelGpFifo AllocChannelGpFifo;
        call_PassAdditionalVerification PassAdditionalVerification;
        call_GetModelIdentifierString GetModelIdentifierString;

        VTable() : Release(0),
                   AllocContextDma(0),
                   AllocChannelDma(0),
                   FreeChannel(0),
                   AllocObject(0),
                   FreeObject(0),
                   AllocChannelGpFifo(0),
                   PassAdditionalVerification(0),
                   GetModelIdentifierString(0) {}

        bool SanityCheck()
        {
            return Release != 0 &&
                   AllocContextDma != 0 &&
                   AllocChannelDma != 0 && 
                   FreeChannel != 0 &&
                   AllocObject != 0 && 
                   FreeObject  != 0 &&
                   AllocChannelGpFifo != 0 &&
                   PassAdditionalVerification != 0 &&
#if defined(MODS)
                   Platform::ChipLibVersion() >= 3 ? GetModelIdentifierString != 0 :
#endif
                   true;
        }
    };

    VTable m_VTable;
};

class MemoryProxy : public IMemory 
{
public:
    MemoryProxy(void* module, IIfaceObject* bus);
    ~MemoryProxy(){}

    LwU008 MemRd08(LwU032 address) { MASSERT(!"Not implemented"); return 0; }
    LwU016 MemRd16(LwU032 address) { MASSERT(!"Not implemented"); return 0; }
    LwU032 MemRd32(LwU032 address) { MASSERT(!"Not implemented"); return 0; }
    LwU064 MemRd64(LwU032 address) { MASSERT(!"Not implemented"); return 0; }
    void MemWr08(LwU032 address, LwU008 data)  { MASSERT(!"Not implemented"); }
    void MemWr16(LwU032 address, LwU016 data)  { MASSERT(!"Not implemented"); }
    void MemWr32(LwU032 address, LwU032 data)  { MASSERT(!"Not implemented"); }
    void MemWr64(LwU032 address, LwU064 data)  { MASSERT(!"Not implemented"); }

    void MemSet08(LwU032 address, LwU032 size, LwU008 data)  { MASSERT(!"Not implemented"); }
    void MemSet16(LwU032 address, LwU032 size, LwU016 data)  { MASSERT(!"Not implemented"); }
    void MemSet32(LwU032 address, LwU032 size, LwU032 data)  { MASSERT(!"Not implemented"); }
    void MemSet64(LwU032 address, LwU032 size, LwU064 data)  { MASSERT(!"Not implemented"); }
    void MemSetBlk(LwU032 address, LwU032 size, void* data, LwU032 data_size) { MASSERT(!"Not implemented"); }

    void MemWrBlk(LwU032 address, const void *appdata, LwU032 count)  { MASSERT(!"Not implemented"); }
    void MemWrBlk32(LwU032 address, const void *appdata, LwU032 count)  { MASSERT(!"Not implemented"); }
    void MemRdBlk(LwU032 address, void *appdata, LwU032 count)  { MASSERT(!"Not implemented"); }
    void MemRdBlk32(LwU032 address, void *appdata, LwU032 count)  { MASSERT(!"Not implemented"); }
    void MemCpBlk(LwU032 address, LwU032 appdata, LwU032 count)  { MASSERT(!"Not implemented"); }
    void MemCpBlk32(LwU032 address, LwU032 appdata, LwU032 count) { MASSERT(!"Not implemented"); }

    // IIfaceObject Interface
    void AddRef() { MASSERT(!"Not implemented"); }
    void Release();
    IIfaceObject* QueryIface(IID_TYPE id)  { MASSERT(!"Not implemented"); return 0; }

private:
    MemoryProxy(const MemoryProxy&);
    MemoryProxy& operator=(const MemoryProxy&);

    IIfaceObject* m_Memory;

    struct VTable
    {
        void_call_void Release;

        VTable() : Release(0) {}

        bool SanityCheck()
        {
            return Release       != 0;
        }
    };

    VTable m_VTable;
};

class IoProxy : public IIo 
{
public:
    IoProxy(void* module, IIfaceObject* bus);
    ~IoProxy(){}

    LwU008 IoRd08(LwU016 address);
    LwU016 IoRd16(LwU016 address);
    LwU032 IoRd32(LwU016 address);
    void IoWr08(LwU016 address, LwU008 data);
    void IoWr16(LwU016 address, LwU016 data);
    void IoWr32(LwU016 address, LwU032 data);

    LwU008 CfgRd08(LwU032 address);
    LwU016 CfgRd16(LwU032 address);
    LwU032 CfgRd32(LwU032 address);
    void CfgWr08(LwU032 address, LwU008 data);
    void CfgWr16(LwU032 address, LwU016 data);
    void CfgWr32(LwU032 address, LwU032 data);

    // IIfaceObject Interface
    void AddRef()  { MASSERT(!"Not implemented"); }
    void Release();
    IIfaceObject* QueryIface(IID_TYPE id)  { MASSERT(!"Not implemented"); return 0;}

private:
    IoProxy(const IoProxy&);
    IoProxy& operator=(const IoProxy&);

    IIfaceObject* m_Io;

    struct VTable
    {
        void_call_void Release;
        call_IoRd08 IoRd08;
        call_IoRd16 IoRd16;
        call_IoRd32 IoRd32;
        call_IoWr08 IoWr08;
        call_IoWr16 IoWr16;
        call_IoWr32 IoWr32;
        call_CfgRd08 CfgRd08;
        call_CfgRd16 CfgRd16;
        call_CfgRd32 CfgRd32;
        call_CfgWr08 CfgWr08;
        call_CfgWr16 CfgWr16;
        call_CfgWr32 CfgWr32;

        VTable() : Release(0) {}

        bool SanityCheck()
        {
            return Release != 0 &&
                   IoRd08 != 0 &&
                   IoRd16 != 0 &&
                   IoRd32 != 0 &&
                   IoWr08 != 0 &&
                   IoWr16 != 0 &&
                   IoWr32 != 0 &&
                   CfgRd08 != 0 &&
                   CfgRd16 != 0 &&
                   CfgRd32 != 0 &&
                   CfgWr08 != 0 &&
                   CfgWr16 != 0 &&
                   CfgWr32 != 0;
        }
    };

    VTable m_VTable;
};

class LWLinkProxy : public ILWLink
{
public:
    LWLinkProxy(void* module, IIfaceObject* bus);
    ~LWLinkProxy(){}

    void LWLink_CPU2GPU(LWLink_AN1_pkt *req_pkt);


    // IIfaceObject Interface
    void AddRef()  { MASSERT(!"Not implemented"); }
    void Release();
    IIfaceObject* QueryIface(IID_TYPE id)  { MASSERT(!"Not implemented"); return 0;}

private:
    LWLinkProxy(const LWLinkProxy&);
    LWLinkProxy& operator=(const LWLinkProxy&);

    ILWLink *m_LWLink;

    struct VTable
    {
        void_call_void Release;
        call_LWLink_CPU2GPU LWLink_CPU2GPU;

        VTable() : Release(0) {}

        bool SanityCheck()
        {
            return Release != 0 &&
                   LWLink_CPU2GPU != 0;
        }
    };

    VTable m_VTable;
};

extern "C" {
IIfaceObject* call_mods_ifspec2_QueryIface(IIfaceObject* system, IID_TYPE id);
void      call_mods_ifspec2_Release(IIfaceObject* system);
BusMemRet call_mods_ifspec2_BusMemWrBlk(IBusMem*, LwU064 address, const void *appdata, LwU032 count);
BusMemRet call_mods_ifspec2_BusMemRdBlk(IBusMem*, LwU064 address, void *appdata, LwU032 count);
BusMemRet call_mods_ifspec2_BusMemCpBlk(IBusMem*, LwU064 dest, LwU064 source, LwU032 count);
BusMemRet call_mods_ifspec2_BusMemSetBlk(IBusMem*, LwU064 address, LwU032 size, void* data, LwU032 data_size);
void      call_mods_ifspec2_HandleInterrupt(IInterrupt* system);
void      call_mods_ifspec2_HandleSpecificInterrupt(IInterrupt3* system, LwU032 irqNumber);
void      call_mods_ifspec2_LWLinkPktGPU2CPU(ILWLinkCallbacks*, LWLink_AN1_pkt *req_pkt, LWLink_AN1_pkt *resp_pkt);
void      call_mods_ifspec2_LWLinkPktGPU2CPU_ASYNC(ILWLinkCallbacks*, LWLink_AN1_pkt *req_pkt);
void      call_mods_ifspec2_CPUModelAtsRequest2(ICPUModelCallbacks2*, struct LwPciDev bdf, LwU032 pasid, LwU064 address, bool isGpa, LwU032 numPages, bool numPageAlign, LwU032 *pageSize, CPUModelAtsResult *results);
}

#endif  // INCLUDED_SHIM_H
