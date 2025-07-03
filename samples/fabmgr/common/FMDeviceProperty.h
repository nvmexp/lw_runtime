/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include "FMCommonTypes.h"
#include "topology.pb.h"

/*****************************************************************************/
/*  Fabric Manager FM Device property abstractions                           */
/*****************************************************************************/

/*
 * This class abstracts FM LWSwitch and GPU device properties
 *
 * Lwrrently, GFM and LFM uses LWSwitch arch to identify a system, because
 * all systems built so far have the specific LWSwitch and GPU pair, hence
 * LWSwitch arch alone is enough to identify the platform.
 *
 * In the future, LWSwitch might be able to pair with different GPUs. The GPU
 * properties needs to identified by GPU arch independently.
 *
 * These properties are defined statically in header files, not queried from
 * the hardware.
 *
 * To add a new property
 * 1. Add a new device in lwSwitchArchType in topology.proto.precomp
 * 2. Add the device properties in a header file, similar to FMWillowProperty.h
 *    FMLimerockProperty.h and FMLagunaProperty.h
 * 3. Add LWSwtich and GPU entry in mSwitchSpec and mGpuSpec
 * 4. Implement the member methods to get the properties in FMDeviceProperty.h
 *    and FMDeviceProperty.cpp
 *
 */

class FMDeviceProperty
{
public:
    static uint32_t getLWLinksPerSwitch(lwSwitchArchType arch);

    static uint32_t getIngressReqTableSize(lwSwitchArchType arch);
    static uint32_t getIngressRespTableSize(lwSwitchArchType arch);

    static uint32_t getRemapTableSize(lwSwitchArchType arch);
    static uint32_t getRidTableSize(lwSwitchArchType arch);
    static uint32_t getRlanTableSize(lwSwitchArchType arch);
    static uint32_t getGangedLinkTableSize(lwSwitchArchType arch);

    static uint32_t getExtARemapTableSize(lwSwitchArchType arch);
    static uint32_t getExtBRemapTableSize(lwSwitchArchType arch);
    static uint32_t getMulticastRemapTableSize(lwSwitchArchType arch);

    static uint32_t getNumIngressReqEntriesPerGpu(lwSwitchArchType arch);
    static uint32_t getNumIngressRespEntriesPerGpu(lwSwitchArchType arch);

    static RemapTable getFlaRemapTbl(lwSwitchArchType arch);
    static uint32_t getFirstFlaRemapSlot(lwSwitchArchType arch);
    static uint32_t getNumFlaRemapEntriesPerGpu(lwSwitchArchType arch);
    static uint32_t getTargetIdFromFla(lwSwitchArchType arch, uint64_t fla);
    static uint64_t getFlaFromTargetId(lwSwitchArchType arch, uint32_t targetId);
    static uint32_t getFlaRemapIndexFromTargetId(lwSwitchArchType arch, uint32_t targetId);
    static uint32_t getFlaEgmRemapIndexFromTargetId(lwSwitchArchType arch, uint32_t targetId);
    static uint32_t getTargetIdFromFlaEgm(lwSwitchArchType arch, uint64_t Egm);
    static uint64_t getFlaEgmFromTargetId(lwSwitchArchType arch, uint32_t targetId);
    static uint64_t getMulticastBaseAddrFromGroupId(lwSwitchArchType arch, uint32_t groupId);

    static RemapTable getGpaRemapTbl(lwSwitchArchType arch);
    static uint32_t getFirstGpaRemapSlot(lwSwitchArchType arch);
    static uint32_t getNumGpaRemapEntriesPerGpu(lwSwitchArchType arch);
    static uint32_t getTargetIdFromGpa(lwSwitchArchType arch, uint64_t gpa);
    static uint64_t getGpaFromTargetId(lwSwitchArchType arch, uint32_t targetId);
    static uint32_t getGpaRemapIndexFromTargetId(lwSwitchArchType arch, uint32_t targetId);
    static uint32_t getTargetIdFromGpaEgm(lwSwitchArchType arch, uint64_t Egm);
    static uint64_t getGpaEgmFromTargetId(lwSwitchArchType arch, uint32_t targetId);

    static RemapTable getSpaRemapTbl(lwSwitchArchType arch);
    static uint32_t getFirstSpaRemapSlot(lwSwitchArchType arch);
    static uint32_t getNumSpaRemapEntriesPerGpu(lwSwitchArchType arch);
    static uint32_t getSpaRemapIndexFromSpaAddress(lwSwitchArchType arch, uint64_t spaAddr);

    static uint32_t getLWLinksPerGpu(lwSwitchArchType arch);
    static uint64_t getAddressRangePerGpu(lwSwitchArchType arch);
    static uint64_t getEgmAddressRangePerGpu(lwSwitchArchType arch);

private:
    typedef struct {
         uint32_t numLWLinksPerSwitch;

         uint32_t ingressReqTableSize;
         uint32_t ingressRespTableSize;

         uint32_t remapTableSize;
         uint32_t ridTableSize;
         uint32_t rlanTableSize;
         uint32_t gangedLinkTableSize;

         uint32_t extARemapTableSize;
         uint32_t extBRemapTableSize;
         uint32_t multicastRemapTableSize;

         uint32_t numIngressReqEntriesPerGpu;
         uint32_t numIngressRespEntriesPerGpu;

         RemapTable flaRemapTbl;
         uint32_t firstFlaRemapSlot;
         uint32_t numFlaRemapEntriesPerGpu;

         RemapTable gpaRemapTbl;
         uint32_t firstGpaRemapSlot;
         uint32_t numGpaRemapEntriesPerGpu;

         RemapTable spaRemapTbl;
         uint32_t firstSpaRemapSlot;
         uint32_t spaRemapEntriesPerGpu;

     } LWSwitchSpec_t;


     typedef struct {
         uint32_t numLWLinksPerGpu;
         uint64_t fabricAddressRange;
         uint64_t egmAddressRange;
     } GPUSpec_t;

    static LWSwitchSpec_t mSwitchSpec[];
    static GPUSpec_t mGpuSpec[];
};
