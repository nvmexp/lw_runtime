/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 1999-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
#include "sim/shim.h" 

/*
 * Note: it's intended users of this file will #include this inline file
 * replacing GET_DLL_PROC and DBG_PRINT_STR with their own routines.
 */

ChipProxy::ChipProxy(void *module, IChip* chip)
: m_Module(module), m_Chip(chip), m_RefCount(1)
{
    m_VTable.AddRef = (void_call_void)GET_DLL_PROC(module, "call_server_AddRef");
    m_VTable.Release = (void_call_void)GET_DLL_PROC(module, "call_server_Release");
    m_VTable.QueryIface = (call_IIfaceObject)GET_DLL_PROC(module, "call_server_QueryIface");

    m_VTable.Startup = (call_Startup)GET_DLL_PROC(module, "call_server_Startup");
    m_VTable.Shutdown = (void_call_void)GET_DLL_PROC(module, "call_server_Shutdown");
    m_VTable.AllocSysMem = (call_AllocSysMem)GET_DLL_PROC(module, "call_server_AllocSysMem");
    m_VTable.FreeSysMem = (call_FreeSysMem)GET_DLL_PROC(module, "call_server_FreeSysMem");
    m_VTable.ClockSimulator = (call_ClockSimulator)GET_DLL_PROC(module, "call_server_ClockSimulator");
    m_VTable.Delay = (call_Delay)GET_DLL_PROC(module, "call_server_Delay");
    m_VTable.EscapeWrite = (call_EscapeWrite)GET_DLL_PROC(module, "call_server_EscapeWrite");
    m_VTable.EscapeRead = (call_EscapeRead)GET_DLL_PROC(module, "call_server_EscapeRead");
    m_VTable.FindPCIDevice = (call_FindPCIDevice)GET_DLL_PROC(module, "call_server_FindPCIDevice");
    m_VTable.FindPCIClassCode = (call_FindPCIClassCode)GET_DLL_PROC(module, "call_server_FindPCIClassCode");
    m_VTable.GetSimulatorTime = (call_GetSimulatorTime)GET_DLL_PROC(module, "call_server_GetSimulatorTime");
    m_VTable.GetSimulatorTimeUnitsNS = (call_GetSimulatorTimeUnitsNS)GET_DLL_PROC(module, "call_server_GetSimulatorTimeUnitsNS");
    m_VTable.GetSimulatorTimeUnitsNS = (call_GetSimulatorTimeUnitsNS)GET_DLL_PROC(module, "call_server_GetSimulatorTimeUnitsNS");
    m_VTable.GetChipLevel = (call_GetChipLevel)GET_DLL_PROC(module, "call_server_GetChipLevel");

    MASSERT(m_VTable.SanityCheck());
}

void ChipProxy::AddRef()
{
    m_VTable.AddRef(m_Chip);
    ++m_RefCount;
}

void ChipProxy::Release()
{
    MASSERT(m_RefCount > 0);

    m_VTable.Release(m_Chip);
    if (--m_RefCount == 0)
        delete this;
}

IIfaceObject* ChipProxy::QueryIface(IID_TYPE id)
{
    IIfaceObject* interface = m_VTable.QueryIface(m_Chip, id);

    if (interface == 0)
    {
        DBG_PRINT_STR("IFSPEC2: id=%d not supported in chiplib.\n", id);
        return 0;
    }

    switch (id)
    {
        case IID_BUSMEM_IFACE:
            DBG_PRINT_STR("IFSPEC2: sim supplies IBusMem to mods.\n");
            return new BusMemProxy(m_Module, interface);
        case IID_AMODEL_BACKDOOR_IFACE:
            DBG_PRINT_STR("IFSPEC2: sim supplies IAModelBackDoor to mods.\n");
            return new AModelBackdoorProxy(m_Module, interface);
        case IID_MEMORY_IFACE:
            DBG_PRINT_STR("IFSPEC2: sim supplies IMemory to mods.\n");
            return new MemoryProxy(m_Module, interface);
        case IID_MULTIHEAP_IFACE:
            return 0;
        case IID_MEMALLOC64_IFACE:
            return 0;
        case IID_GPUESCAPE_IFACE:
            return 0;
        case IID_GPUESCAPE2_IFACE:
            return 0;
        case IID_IO_IFACE:
            DBG_PRINT_STR("IFSPEC2: sim supplies IIo to mods.\n");
            return new IoProxy(m_Module, interface);
        case IID_LWLINK_IFACE:
            DBG_PRINT_STR("IFSPEC2: sim supplies ILWLink to mods.\n");
            return new LWLinkProxy(m_Module, interface);
        case IID_CPUMODEL2_IFACE:
            return new CPUModel2Proxy(m_Module, interface);
        default:
            MASSERT(!"Unknown interface!");
            return 0;
    }
}

int ChipProxy::Startup(IIfaceObject* system, char** argv, int argc)
{
    return m_VTable.Startup(m_Chip, system, argv, argc);
}

void ChipProxy::Shutdown()
{
    return m_VTable.Shutdown(m_Chip);
}

int ChipProxy::AllocSysMem(int numBytes, LwU032* physAddr)
{
    return m_VTable.AllocSysMem(m_Chip, numBytes, physAddr);
}

void ChipProxy::FreeSysMem(LwU032 physAddr)
{
    return m_VTable.FreeSysMem(m_Chip, physAddr);
}

void ChipProxy::ClockSimulator(LwS032 numClocks)
{
    return m_VTable.ClockSimulator(m_Chip, numClocks);
}

void ChipProxy::Delay(LwU032 numMicroSeconds)
{
    return m_VTable.Delay(m_Chip, numMicroSeconds);
}

int ChipProxy::EscapeWrite(char* path, LwU032 index, LwU032 size, LwU032 value)
{
    return m_VTable.EscapeWrite(m_Chip, path, index, size, value);
}

int ChipProxy::EscapeRead(char* path, LwU032 index, LwU032 size, LwU032* value)
{
    return m_VTable.EscapeRead(m_Chip, path, index, size, value);
}

int ChipProxy::FindPCIDevice(LwU016 vendorId, LwU016 deviceId, int index, LwU032* address)
{
    return m_VTable.FindPCIDevice(m_Chip, vendorId, deviceId, index, address);
}

int ChipProxy::FindPCIClassCode(LwU032 classCode, int index, LwU032* address)
{
    return m_VTable.FindPCIClassCode(m_Chip, classCode, index, address);
}

int ChipProxy::GetSimulatorTime(LwU064* simTime)
{
    return m_VTable.GetSimulatorTime(m_Chip, simTime);
}

double ChipProxy::GetSimulatorTimeUnitsNS()
{
    return m_VTable.GetSimulatorTimeUnitsNS(m_Chip);
}

IChip::ELEVEL ChipProxy::GetChipLevel()
{
    return m_VTable.GetChipLevel(m_Chip);
}

//----------------------------------------------------------------------
// Proxy for IBusMem interface
//----------------------------------------------------------------------
BusMemProxy::BusMemProxy(void *module, IIfaceObject* chip) : m_Bus((IBusMem*)chip)
{
    m_VTable.Release = (void_call_void)GET_DLL_PROC(module, "call_server_Release");

    m_VTable.BusMemWrBlk = (call_BusMemWrBlk)GET_DLL_PROC(module, "call_server_BusMemWrBlk");
    m_VTable.BusMemRdBlk = (call_BusMemRdBlk)GET_DLL_PROC(module, "call_server_BusMemRdBlk");
    m_VTable.BusMemCpBlk = (call_BusMemCpBlk)GET_DLL_PROC(module, "call_server_BusMemCpBlk");
    m_VTable.BusMemSetBlk = (call_BusMemSetBlk)GET_DLL_PROC(module, "call_server_BusMemSetBlk");

    MASSERT(m_VTable.SanityCheck());
}

BusMemRet BusMemProxy::BusMemWrBlk(LwU064 address, const void *appdata, LwU032 count)
{
    return m_VTable.BusMemWrBlk(m_Bus, address, appdata, count);
}

BusMemRet BusMemProxy::BusMemRdBlk(LwU064 address, void *appdata, LwU032 count)
{
    return m_VTable.BusMemRdBlk(m_Bus, address, appdata, count);
}

BusMemRet BusMemProxy::BusMemCpBlk(LwU064 dest, LwU064 source, LwU032 count)
{
    return m_VTable.BusMemCpBlk(m_Bus, dest, source, count);
}

BusMemRet BusMemProxy::BusMemSetBlk(LwU064 address, LwU032 size, void* data, LwU032 data_size)
{
    return m_VTable.BusMemSetBlk(m_Bus, address, size, data, data_size);
}

void BusMemProxy::Release()
{
    m_VTable.Release(m_Bus);

    // No need to keep ref count since AddRef & QueryIface aren't implemented
    delete this;
}

//----------------------------------------------------------------------
// Proxy for ILWLink interface
//----------------------------------------------------------------------
LWLinkProxy::LWLinkProxy(void *module, IIfaceObject* bus) : m_LWLink((ILWLink*)bus)
{
    m_VTable.Release = (void_call_void)GET_DLL_PROC(module, "call_server_Release");

    m_VTable.LWLink_CPU2GPU = (call_LWLink_CPU2GPU)GET_DLL_PROC(module, "call_server_LWLink_CPU2GPU");

    MASSERT(m_VTable.SanityCheck());
}

void LWLinkProxy::LWLink_CPU2GPU(LWLink_AN1_pkt *req_pkt)
{
    return m_VTable.LWLink_CPU2GPU(m_LWLink, req_pkt);
}

void LWLinkProxy::Release()
{
    m_VTable.Release(m_LWLink);

    // No need to keep ref count since AddRef & QueryIface aren't implemented
    delete this;
}

//----------------------------------------------------------------------
// Proxy for IAModelBackdoor interface
//----------------------------------------------------------------------
AModelBackdoorProxy::AModelBackdoorProxy(void* module, IIfaceObject* backdoor) : m_AModelBackdoor(backdoor)
{
    m_VTable.Release = (void_call_void)GET_DLL_PROC(module, "call_server_Release");

    m_VTable.AllocContextDma = (call_AllocContextDma)GET_DLL_PROC(module, "call_server_AllocContextDma");
    m_VTable.AllocChannelDma = (call_AllocChannelDma)GET_DLL_PROC(module, "call_server_AllocChannelDma");
    m_VTable.FreeChannel = (call_FreeChannel)GET_DLL_PROC(module, "call_server_FreeChannel");
    m_VTable.AllocObject = (call_AllocObject)GET_DLL_PROC(module, "call_server_AllocObject");
    m_VTable.FreeObject = (call_FreeObject)GET_DLL_PROC(module, "call_server_FreeObject");
    m_VTable.AllocChannelGpFifo = (call_AllocChannelGpFifo)GET_DLL_PROC(module, "call_server_AllocChannelGpFifo");
    m_VTable.PassAdditionalVerification = (call_PassAdditionalVerification)GET_DLL_PROC(module, "call_server_PassAdditionalVerification");
    m_VTable.GetModelIdentifierString = (call_GetModelIdentifierString)GET_DLL_PROC(module, "call_server_GetModelIdentifierString");

    MASSERT(m_VTable.SanityCheck());
}

void AModelBackdoorProxy::Release()
{
    m_VTable.Release(m_AModelBackdoor);

    // No need to keep ref count since AddRef & QueryIface aren't implemented
    delete this;
}

void AModelBackdoorProxy::AllocContextDma(LwU032 ChID, LwU032 Handle, LwU032 Class, LwU032 target, 
        LwU032 Limit, LwU032 Base, LwU032 Protect, LwU032 *PageTable)
{
    return m_VTable.AllocContextDma(m_AModelBackdoor, ChID, Handle, Class, target, Limit, Base, 
                                                   Protect, PageTable);
}

void AModelBackdoorProxy::AllocChannelDma(LwU032 ChID, LwU032 Class, LwU032 CtxDma, LwU032 ErrorNotifierCtxDma)
{
    return m_VTable.AllocChannelDma(m_AModelBackdoor, ChID, Class, CtxDma, ErrorNotifierCtxDma);
}

void AModelBackdoorProxy::FreeChannel(LwU032 ChID)
{
    return m_VTable.FreeChannel(m_AModelBackdoor, ChID);
}

void AModelBackdoorProxy::AllocObject(LwU032 ChID, LwU032 Handle, LwU032 Class)
{
    return m_VTable.AllocObject(m_AModelBackdoor, ChID, Handle, Class);
}

void AModelBackdoorProxy::FreeObject(LwU032 ChID, LwU032 Handle)
{
    return m_VTable.FreeObject(m_AModelBackdoor, ChID, Handle);
}

void AModelBackdoorProxy::AllocChannelGpFifo(LwU032 ChID, LwU032 Class, LwU032 CtxDma, 
                 LwU064 GpFifoOffset, LwU032 GpFifoEntries, LwU032 ErrorNotifierCtxDma)
{
    return m_VTable.AllocChannelGpFifo(m_AModelBackdoor, ChID, Class, CtxDma, GpFifoOffset,
            GpFifoEntries, ErrorNotifierCtxDma);
}

bool AModelBackdoorProxy::PassAdditionalVerification(const char *traceFileName)
{
    return m_VTable.PassAdditionalVerification(m_AModelBackdoor, traceFileName);
}

const char* AModelBackdoorProxy::GetModelIdentifierString()
{
    return m_VTable.GetModelIdentifierString ? m_VTable.GetModelIdentifierString(m_AModelBackdoor) : 0;
}

//----------------------------------------------------------------------
// Proxy for IMemory interface
//----------------------------------------------------------------------
MemoryProxy::MemoryProxy(void* module, IIfaceObject* memory) : m_Memory(memory)
{
    m_VTable.Release = (void_call_void)GET_DLL_PROC(module, "call_server_Release");

    MASSERT(m_VTable.SanityCheck());
}

void MemoryProxy::Release()
{
    m_VTable.Release(m_Memory);

    // No need to keep ref count since AddRef & QueryIface aren't implemented
    delete this;
}

//----------------------------------------------------------------------
// Proxy for IIo interface
//----------------------------------------------------------------------
IoProxy::IoProxy(void* module, IIfaceObject* io) : m_Io(io)
{
    m_VTable.IoRd08 = (call_IoRd08)GET_DLL_PROC(module, "call_server_IoRd08");
    m_VTable.IoRd16 = (call_IoRd16)GET_DLL_PROC(module, "call_server_IoRd16");
    m_VTable.IoRd32 = (call_IoRd32)GET_DLL_PROC(module, "call_server_IoRd32");
    m_VTable.IoWr08 = (call_IoWr08)GET_DLL_PROC(module, "call_server_IoWr08");
    m_VTable.IoWr16 = (call_IoWr16)GET_DLL_PROC(module, "call_server_IoWr16");
    m_VTable.IoWr32 = (call_IoWr32)GET_DLL_PROC(module, "call_server_IoWr32");
    m_VTable.CfgRd08 = (call_CfgRd08)GET_DLL_PROC(module, "call_server_CfgRd08");
    m_VTable.CfgRd16 = (call_CfgRd16)GET_DLL_PROC(module, "call_server_CfgRd16");
    m_VTable.CfgRd32 = (call_CfgRd32)GET_DLL_PROC(module, "call_server_CfgRd32");
    m_VTable.CfgWr08 = (call_CfgWr08)GET_DLL_PROC(module, "call_server_CfgWr08");
    m_VTable.CfgWr16 = (call_CfgWr16)GET_DLL_PROC(module, "call_server_CfgWr16");
    m_VTable.CfgWr32 = (call_CfgWr32)GET_DLL_PROC(module, "call_server_CfgWr32");
    m_VTable.Release = (void_call_void)GET_DLL_PROC(module, "call_server_Release");

    MASSERT(m_VTable.SanityCheck());
}

void IoProxy::Release()
{
    m_VTable.Release(m_Io);

    // No need to keep ref count since AddRef & QueryIface aren't implemented
    delete this;
}

void IoProxy::IoWr08(LwU016 address, LwU008 data)
{
    return m_VTable.IoWr08(m_Io, address, data);
}

void IoProxy::IoWr16(LwU016 address, LwU016 data)
{
    return m_VTable.IoWr16(m_Io, address, data);
}

void IoProxy::IoWr32(LwU016 address, LwU032 data)
{
    return m_VTable.IoWr32(m_Io, address, data);
}

LwU008 IoProxy::IoRd08(LwU016 address)
{
    return m_VTable.IoRd08(m_Io, address);
}

LwU016 IoProxy::IoRd16(LwU016 address)
{
    return m_VTable.IoRd16(m_Io, address);
}

LwU032 IoProxy::IoRd32(LwU016 address)
{
    return m_VTable.IoRd32(m_Io, address);
}

void IoProxy::CfgWr08(LwU032 address, LwU008 data)
{
    return m_VTable.CfgWr08(m_Io, address, data);
}

void IoProxy::CfgWr16(LwU032 address, LwU016 data)
{
    return m_VTable.CfgWr16(m_Io, address, data);
}

void IoProxy::CfgWr32(LwU032 address, LwU032 data)
{
    return m_VTable.CfgWr32(m_Io, address, data);
}

LwU008 IoProxy::CfgRd08(LwU032 address)
{
    return m_VTable.CfgRd08(m_Io, address);
}

LwU016 IoProxy::CfgRd16(LwU032 address)
{
    return m_VTable.CfgRd16(m_Io, address);
}

LwU032 IoProxy::CfgRd32(LwU032 address)
{
    return m_VTable.CfgRd32(m_Io, address);
}

//----------------------------------------------------------------------
// Proxy for ICPUModel2 interface
//----------------------------------------------------------------------

CPUModel2Proxy::CPUModel2Proxy(void* module, IIfaceObject* cpu) : m_cpuModel((ICPUModel2*)cpu)
{
    m_VTable.CPUModelInitialize = (call_CPUModelInitialize2)GET_DLL_PROC(module, "call_server_CPUModelInitialize2");
    m_VTable.CPUModelEnableResponse = (call_CPUModelEnableResponse2)GET_DLL_PROC(module, "call_server_CPUModelEnableResponse2");
    m_VTable.CPUModelEnableResponseSpecific = (call_CPUModelEnableResponseSpecific2)GET_DLL_PROC(module, "call_server_CPUModelEnableResponseSpecific2");
    m_VTable.CPUModelHasResponse = (call_CPUModelHasResponse2)GET_DLL_PROC(module, "call_server_CPUModelHasResponse2");
    m_VTable.CPUModelGetResponse = (call_CPUModelGetResponse2)GET_DLL_PROC(module, "call_server_CPUModelGetResponse2");
    m_VTable.CPUModelRead = (call_CPUModelRead2)GET_DLL_PROC(module, "call_server_CPUModelRead2");
    m_VTable.CPUModelGetCacheData = (call_CPUModelGetCacheData2)GET_DLL_PROC(module, "call_server_CPUModelGetCacheData2");
    m_VTable.CPUModelWrite = (call_CPUModelWrite2)GET_DLL_PROC(module, "call_server_CPUModelWrite2");
    m_VTable.CPUModelAtsShootDown = (call_CPUModelAtsShootDown2)GET_DLL_PROC(module, "call_server_CPUModelAtsShootDown2");
    m_VTable.Release = (void_call_void)GET_DLL_PROC(module, "call_server_Release");

    MASSERT(m_VTable.SanityCheck());
}

void CPUModel2Proxy::Release()
{
    m_VTable.Release(m_cpuModel);

    // No need to keep ref count since AddRef & QueryIface aren't implemented
    delete this;
}

void CPUModel2Proxy::CPUModelInitialize(void)
{
    m_VTable.CPUModelInitialize(m_cpuModel);
}

void CPUModel2Proxy::CPUModelEnableResponse(bool enable)
{
    m_VTable.CPUModelEnableResponse(m_cpuModel, enable);
}

void CPUModel2Proxy::CPUModelEnableResponseSpecific(bool enable, CPUModelEvent event)
{
    m_VTable.CPUModelEnableResponseSpecific(m_cpuModel, enable, event);
}

bool CPUModel2Proxy::CPUModelHasResponse(void)
{
    return m_VTable.CPUModelHasResponse(m_cpuModel);
}

void CPUModel2Proxy::CPUModelGetResponse(CPUModelResponse *response)
{
    m_VTable.CPUModelGetResponse(m_cpuModel, response);
}

CPUModelRet CPUModel2Proxy::CPUModelRead(LwU032 uniqueId, LwU064 address, LwU032 sizeBytes,
                bool isCoherentlyCaching, bool isProbing)
{
    return m_VTable.CPUModelRead(m_cpuModel, uniqueId, address, sizeBytes, isCoherentlyCaching, isProbing);
}

CPUModelRet CPUModel2Proxy::CPUModelGetCacheData(LwU064 gpa, LwU064 *data)
{
    return m_VTable.CPUModelGetCacheData(m_cpuModel, gpa, data);
}

CPUModelRet CPUModel2Proxy::CPUModelWrite(LwU032 uniqueId, LwU064 address, LwU032 offset, LwU064 data,
                LwU032 sizeBytes, bool isCoherentlyCaching, bool isProbing, bool isPosted)
{
    return m_VTable.CPUModelWrite(m_cpuModel, uniqueId, address, offset, data, sizeBytes, isCoherentlyCaching, isProbing, isPosted);
}

void CPUModel2Proxy::CPUModelAtsShootDown(LwU032 uniqueId, struct LwPciDev bdf, LwU032 pasid, LwU064 address, LwU032 sizeBytes, bool isGpa, bool flush, bool isGlobal)
{
    return m_VTable.CPUModelAtsShootDown(m_cpuModel, uniqueId, bdf, pasid, address, sizeBytes, isGpa, flush, isGlobal);
}

// ----------------------------------------------------------------------------

extern "C" {
IIfaceObject* call_mods_ifspec2_QueryIface(IIfaceObject* system, IID_TYPE id)
{
    return system->QueryIface(id);
}

void call_mods_ifspec2_Release(IIfaceObject* system)
{
    system->Release();
}

BusMemRet call_mods_ifspec2_BusMemWrBlk(IBusMem* bus, LwU064 address, const void *appdata, LwU032 count)
{
    return bus->BusMemWrBlk(address, appdata, count);
}

BusMemRet call_mods_ifspec2_BusMemRdBlk(IBusMem* bus, LwU064 address, void *appdata, LwU032 count)
{
    return bus->BusMemRdBlk(address, appdata, count);
}

BusMemRet call_mods_ifspec2_BusMemCpBlk(IBusMem* bus, LwU064 dest, LwU064 source, LwU032 count)
{
    return bus->BusMemCpBlk(dest, source, count);
}

BusMemRet call_mods_ifspec2_BusMemSetBlk(IBusMem* bus, LwU064 address, LwU032 size, void* data, LwU032 data_size)
{
    return bus->BusMemSetBlk(address, size, data, data_size);
}

void call_mods_ifspec2_HandleInterrupt(IInterrupt* system)
{
    system->HandleInterrupt();
}

void call_mods_ifspec2_CPUModelAtsRequest2(ICPUModelCallbacks2 *cpuCallbacks, struct LwPciDev bdf, LwU032 pasid, LwU064 address, bool isGpa, LwU032 numPages, bool numPageAlign, LwU032 *pageSize, CPUModelAtsResult *results)
{
    cpuCallbacks->CPUModelAtsRequest(bdf, pasid, address, isGpa, numPages, numPageAlign, pageSize, results);
}

void call_mods_ifspec2_LWLinkPktGPU2CPU(ILWLinkCallbacks* lwlinkCallbacks, LWLink_AN1_pkt *req_pkt, LWLink_AN1_pkt *resp_pkt)
{
    lwlinkCallbacks->LWLinkPktGPU2CPU(req_pkt, resp_pkt);
}

void call_mods_ifspec2_LWLinkPktGPU2CPU_ASYNC(ILWLinkCallbacks* lwlinkCallbacks, LWLink_AN1_pkt *req_pkt)
{
    lwlinkCallbacks->LWLinkPktGPU2CPU_ASYNC(req_pkt);
}

}

