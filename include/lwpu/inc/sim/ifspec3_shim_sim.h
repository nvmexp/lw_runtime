/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2008-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

// Generated with //sw/mods/chiplib/shim/generate/generate_shim.py#26
//   by lbangerter on 2016-02-19.

// DO NOT HAND-EDIT.
// See https://wiki.lwpu.com/engwiki/index.php/MODS/sim_linkage#How_to_change_ifspec

#if !defined(INCLUDED_IFSPEC3_SHIM_SIM_H)
#define INCLUDED_IFSPEC3_SHIM_SIM_H

#include "ifspec3.h"

namespace IFSPEC3
{


class IIfaceObjectSimProxy : public IIfaceObject
{
private:
    // Pointer to wrapped mods object.
    void* m_pvthis;

public:
    static IIfaceObjectFuncs  s_IIfaceObjectFuncs;

    IIfaceObjectSimProxy(void* pvIIfaceObject);

    virtual ~IIfaceObjectSimProxy();

    virtual void           AddRef();
    virtual void           Release();
    virtual IIfaceObject*  QueryIface(IID_TYPE id);
};

class IBusMemSimProxy : public IBusMem
{
private:
    // Pointer to wrapped mods object.
    void* m_pvthis;

public:
    static IBusMemFuncs  s_IBusMemFuncs;

    IBusMemSimProxy(void* pvIBusMem);

    virtual ~IBusMemSimProxy();

    virtual void           AddRef();
    virtual void           Release();
    virtual IIfaceObject*  QueryIface(IID_TYPE id);
    virtual BusMemRet      BusMemWrBlk(LwU064 address, const void* appdata, LwU032 count);
    virtual BusMemRet      BusMemRdBlk(LwU064 address, void* appdata, LwU032 count);
    virtual BusMemRet      BusMemCpBlk(LwU064 dest, LwU064 source, LwU032 count);
    virtual BusMemRet      BusMemSetBlk(LwU064 address, LwU032 size, void* data, LwU032 data_size);
};

class IInterrupt4SimProxy : public IInterrupt4
{
private:
    // Pointer to wrapped mods object.
    void* m_pvthis;

public:
    static IInterrupt4Funcs  s_IInterrupt4Funcs;

    IInterrupt4SimProxy(void* pvIInterrupt4);

    virtual ~IInterrupt4SimProxy();

    virtual void           AddRef();
    virtual void           Release();
    virtual IIfaceObject*  QueryIface(IID_TYPE id);
    virtual void           HandleInterruptVectorChange(const LwU032* pVector, LwU032 numWords);
};

class IInterrupt3SimProxy : public IInterrupt3
{
private:
    // Pointer to wrapped mods object.
    void* m_pvthis;

public:
    static IInterrupt3Funcs  s_IInterrupt3Funcs;

    IInterrupt3SimProxy(void* pvIInterrupt3);

    virtual ~IInterrupt3SimProxy();

    virtual void           AddRef();
    virtual void           Release();
    virtual IIfaceObject*  QueryIface(IID_TYPE id);
    virtual void           HandleSpecificInterrupt(LwU032 irqNumber);
};

class IInterruptSimProxy : public IInterrupt
{
private:
    // Pointer to wrapped mods object.
    void* m_pvthis;

public:
    static IInterruptFuncs  s_IInterruptFuncs;

    IInterruptSimProxy(void* pvIInterrupt);

    virtual ~IInterruptSimProxy();

    virtual void           AddRef();
    virtual void           Release();
    virtual IIfaceObject*  QueryIface(IID_TYPE id);
    virtual void           HandleInterrupt();
};

class ICPUModelCallbacks2SimProxy : public ICPUModelCallbacks2
{
private:
    // Pointer to wrapped mods object.
    void* m_pvthis;

public:
    static ICPUModelCallbacks2Funcs  s_ICPUModelCallbacks2Funcs;

    ICPUModelCallbacks2SimProxy(void* pvICPUModelCallbacks2);

    virtual ~ICPUModelCallbacks2SimProxy();

    virtual void           AddRef();
    virtual void           Release();
    virtual IIfaceObject*  QueryIface(IID_TYPE id);
    virtual void           CPUModelAtsRequest(LwPciDev bdf, LwU032 pasid, LwU064 address, bool isGpa, LwU032 numPages, bool numPageAlign, LwU032 * pageSize, CPUModelAtsResult * results);
};

class IMapMemoryExtSimProxy : public IMapMemoryExt
{
private:
    // Pointer to wrapped mods object.
    void* m_pvthis;

public:
    static IMapMemoryExtFuncs  s_IMapMemoryExtFuncs;

    IMapMemoryExtSimProxy(void* pvIMapMemoryExt);

    virtual ~IMapMemoryExtSimProxy();

    virtual void           AddRef();
    virtual void           Release();
    virtual IIfaceObject*  QueryIface(IID_TYPE id);

    virtual int             MapMemoryRegion(void ** pReturnedVirtualAddress, LwU064 PhysicalAddress, size_t NumBytes, int Attrib, int Protect);
    virtual void           UnMapMemoryRegion(void * VirtualAddress);
    virtual int             RemapPages(void * VirtualAddress, LwU064 PhysicalAddress, size_t NumBytes, int Protect);

};
}; // namespace IFSPEC3
#endif // !defined(INCLUDED_IFSPEC3_SHIM_SIM_H)
