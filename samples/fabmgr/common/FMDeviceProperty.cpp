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
#include <sstream>
#include <g_lwconfig.h>
#include "FMDeviceProperty.h"
#include "FMWillowProperty.h"
#include "FMLimerockProperty.h"
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#include "FMLagunaProperty.h"
#endif

FMDeviceProperty::LWSwitchSpec_t FMDeviceProperty::mSwitchSpec[LWSWITCH_ARCH_TYPE_MAX] =
{
        // LWSWITCH_ARCH_TYPE_ILWALID
        { 0 },

        // LWSWITCH_ARCH_TYPE_SV10
        {
            WILLOW_NUM_LWLINKS_PER_SWITCH,

            WILLOW_INGRESS_REQUEST_TABLE_SIZE,
            WILLOW_INGRESS_RESPONSETABLE_SIZE,

            WILLOW_REMAP_TABLE_SIZE,
            WILLOW_RLAN_TABLE_SIZE,
            WILLOW_RID_TABLE_SIZE,
            WILLOW_GANGED_LINK_TABLE_SIZE,

            WILLOW_REMAP_EXT_RANGE_A_TABLE_SIZE,
            WILLOW_REMAP_EXT_RANGE_B_TABLE_SIZE,
            WILLOW_MULTICAST_REMAP_TABLE_SIZE,

            WILLOW_NUM_INGRESS_REQUEST_ENTRIES_PER_GPU,
            WILLOW_NUM_INGRESS_RESPONSE_ENTRIES_PER_GPU,

            NORMAL_RANGE,
            WILLOW_FIRST_FLA_REMAP_SLOT,
            WILLOW_NUM_FLA_REMAP_ENTRIES_PER_GPU,

            NORMAL_RANGE,
            WILLOW_FIRST_GPA_REMAP_SLOT,
            WILLOW_NUM_GPA_REMAP_ENTRIES_PER_GPU,

            NORMAL_RANGE,
            WILLOW_FIRST_SPA_REMAP_SLOT,
            WILLOW_NUM_SPA_REMAP_ENTRIES_PER_GPU,
        },

        // LWSWITCH_ARCH_TYPE_LR10
        {
            LIMEROCK_NUM_LWLINKS_PER_SWITCH,

            LIMEROCK_INGRESS_REQUEST_TABLE_SIZE,
            LIMEROCK_INGRESS_RESPONSETABLE_SIZE,

            LIMEROCK_REMAP_TABLE_SIZE,
            LIMEROCK_RLAN_TABLE_SIZE,
            LIMEROCK_RID_TABLE_SIZE,
            LIMEROCK_GANGED_LINK_TABLE_SIZE,

            LIMEROCK_REMAP_EXT_RANGE_A_TABLE_SIZE,
            LIMEROCK_REMAP_EXT_RANGE_B_TABLE_SIZE,
            LIMEROCK_MULTICAST_REMAP_TABLE_SIZE,

            LIMEROCK_NUM_INGRESS_REQUEST_ENTRIES_PER_GPU,
            LIMEROCK_NUM_INGRESS_RESPONSE_ENTRIES_PER_GPU,

            NORMAL_RANGE,
            LIMEROCK_FIRST_FLA_REMAP_SLOT,
            LIMEROCK_NUM_FLA_REMAP_ENTRIES_PER_GPU,

            NORMAL_RANGE,
            LIMEROCK_FIRST_GPA_REMAP_SLOT,
            LIMEROCK_NUM_GPA_REMAP_ENTRIES_PER_GPU,

            NORMAL_RANGE,
            LIMEROCK_FIRST_SPA_REMAP_SLOT,
            LIMEROCK_NUM_SPA_REMAP_ENTRIES_PER_GPU,
        },

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        {
            LAGUNA_NUM_LWLINKS_PER_SWITCH,

            LAGUNA_INGRESS_REQUEST_TABLE_SIZE,
            LAGUNA_INGRESS_RESPONSETABLE_SIZE,

            LAGUNA_REMAP_TABLE_SIZE,
            LAGUNA_RLAN_TABLE_SIZE,
            LAGUNA_RID_TABLE_SIZE,
            LAGUNA_GANGED_LINK_TABLE_SIZE,

            LAGUNA_REMAP_EXT_RANGE_A_TABLE_SIZE,
            LAGUNA_REMAP_EXT_RANGE_B_TABLE_SIZE,
            LAGUNA_MULTICAST_REMAP_TABLE_SIZE,

            LAGUNA_NUM_INGRESS_REQUEST_ENTRIES_PER_GPU,
            LAGUNA_NUM_INGRESS_RESPONSE_ENTRIES_PER_GPU,

            NORMAL_RANGE,
            LAGUNA_FIRST_FLA_REMAP_SLOT,
            LAGUNA_NUM_FLA_REMAP_ENTRIES_PER_GPU,

            EXTENDED_RANGE_B,
            LAGUNA_FIRST_GPA_REMAP_SLOT,
            LAGUNA_NUM_GPA_REMAP_ENTRIES_PER_GPU,

            NORMAL_RANGE,
            LAGUNA_FIRST_SPA_REMAP_SLOT,
            LAGUNA_NUM_SPA_REMAP_ENTRIES_PER_GPU,
        },
#endif
};

FMDeviceProperty::GPUSpec_t FMDeviceProperty::mGpuSpec[LWSWITCH_ARCH_TYPE_MAX] =
{
    { 0 },

    // GV100 Volta
    {
        NUM_LWLINKS_PER_VOLTA,
        VOLTA_FABRIC_ADDRESS_RANGE,
        VOLTA_EGM_ADDRESS_RANGE
    },

    // GA100 Ampere
    {
        NUM_LWLINKS_PER_AMPERE,
        AMPERE_FABRIC_ADDRESS_RANGE,
        AMPERE_EGM_ADDRESS_RANGE
    },

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    // GH100 Hopper
    {
        NUM_LWLINKS_PER_HOPPER,
        HOPPER_FABRIC_ADDRESS_RANGE,
        HOPPER_EGM_ADDRESS_RANGE
    },
#endif
};

uint32_t
FMDeviceProperty::getLWLinksPerSwitch(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].numLWLinksPerSwitch;
}

uint32_t
FMDeviceProperty::getIngressReqTableSize(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].ingressReqTableSize;
}

uint32_t
FMDeviceProperty::getIngressRespTableSize(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].ingressRespTableSize;
}

uint32_t
FMDeviceProperty::getRemapTableSize(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].remapTableSize;
}

uint32_t
FMDeviceProperty::getRidTableSize(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].ridTableSize;
}

uint32_t
FMDeviceProperty::getRlanTableSize(lwSwitchArchType arch)

{
    return mSwitchSpec[arch].rlanTableSize;
}

uint32_t
FMDeviceProperty::getGangedLinkTableSize(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].gangedLinkTableSize;
}

uint32_t
FMDeviceProperty::getExtARemapTableSize(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].extARemapTableSize;
}

uint32_t
FMDeviceProperty::getExtBRemapTableSize(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].extBRemapTableSize;
}

uint32_t
FMDeviceProperty::getMulticastRemapTableSize(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].multicastRemapTableSize;
}

uint32_t
FMDeviceProperty::getNumIngressReqEntriesPerGpu(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].numIngressReqEntriesPerGpu;
}

uint32_t
FMDeviceProperty::getNumIngressRespEntriesPerGpu(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].numIngressRespEntriesPerGpu;
}

RemapTable
FMDeviceProperty::getFlaRemapTbl(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].flaRemapTbl;
}

uint32_t
FMDeviceProperty::getFirstFlaRemapSlot(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].firstFlaRemapSlot;
}

uint32_t
FMDeviceProperty::getNumFlaRemapEntriesPerGpu(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].numFlaRemapEntriesPerGpu;
}

uint32_t
FMDeviceProperty::getTargetIdFromFla(lwSwitchArchType arch, uint64_t fla)
{
    uint32_t targetId = 0;

    switch (arch) {
    case LWSWITCH_ARCH_TYPE_SV10:
        targetId = WILLOW_FLA_TO_TARGET_ID(fla);
        break;

    case LWSWITCH_ARCH_TYPE_LR10:
        targetId = LIMEROCK_FLA_TO_TARGET_ID(fla);
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        targetId = LAGUNA_FLA_TO_TARGET_ID(fla);
        break;
#endif

    default:
        break;
    }

    return targetId;
}

uint64_t
FMDeviceProperty::getFlaFromTargetId(lwSwitchArchType arch, uint32_t targetId)
{
    uint64_t fla = 0;

    switch (arch) {
    case LWSWITCH_ARCH_TYPE_SV10:
        fla = WILLOW_TARGET_ID_TO_FLA(targetId);
        break;

    case LWSWITCH_ARCH_TYPE_LR10:
        fla = LIMEROCK_TARGET_ID_TO_FLA(targetId);
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        fla = LAGUNA_TARGET_ID_TO_FLA(targetId);
        break;
#endif

    default:
        break;
    }

    return fla;
}

uint32_t
FMDeviceProperty::getFlaRemapIndexFromTargetId(lwSwitchArchType arch, uint32_t targetId)
{
    uint32_t remapIndex = 0;

    switch (arch) {
    case LWSWITCH_ARCH_TYPE_SV10:
        remapIndex = WILLOW_TARGET_ID_TO_FLA_INDEX(targetId);
        break;

    case LWSWITCH_ARCH_TYPE_LR10:
        remapIndex = LIMEROCK_TARGET_ID_TO_FLA_INDEX(targetId);
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        remapIndex = LAGUNA_TARGET_ID_TO_FLA_INDEX(targetId);
        break;
#endif

    default:
        break;
    }

    return remapIndex;
}

uint32_t
FMDeviceProperty::getFlaEgmRemapIndexFromTargetId(lwSwitchArchType arch, uint32_t targetId)
{
    uint32_t remapIndex = 0;

    switch (arch) {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        remapIndex = LAGUNA_TARGET_ID_TO_FLA_EGM_INDEX(targetId);
        break;
#endif

    default:
        break;
    }

    return remapIndex;
}

uint32_t
FMDeviceProperty::getTargetIdFromFlaEgm(lwSwitchArchType arch, uint64_t Egm)
{
    return getTargetIdFromFla(arch, Egm);
}

uint64_t
FMDeviceProperty::getFlaEgmFromTargetId(lwSwitchArchType arch, uint32_t targetId)
{
    uint64_t egm = 0;

    switch (arch) {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        egm = LAGUNA_TARGET_ID_TO_FLA_EGM(targetId);
        break;
#endif

    default:
        break;
    }

    return egm;
}

uint64_t
FMDeviceProperty::getMulticastBaseAddrFromGroupId(lwSwitchArchType arch, uint32_t groupId)
{
    uint64_t addressBase = 0;

    switch (arch) {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        // Addr[51:50] = 2’b11: Multicast
        // groupId = Addr[45:39]
        addressBase = ((uint64_t)LAGUNA_MULTICAST_FABRIC_ADDR_VALUE << LAGUNA_MULTICAST_FABRIC_ADDR_SHIFT) |
                            ((uint64_t)groupId << LAGUNA_MULTICAST_GROUP_ID_SHIFT);
        break;
#endif

    default:
        break;
    }

    return addressBase;
}

RemapTable
FMDeviceProperty::getGpaRemapTbl(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].gpaRemapTbl;
}

uint32_t
FMDeviceProperty::getFirstGpaRemapSlot(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].firstGpaRemapSlot;
}

uint32_t
FMDeviceProperty::getNumGpaRemapEntriesPerGpu(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].numGpaRemapEntriesPerGpu;
}

uint32_t
FMDeviceProperty::getTargetIdFromGpa(lwSwitchArchType arch, uint64_t gpa)
{
    uint32_t targetId = 0;

    switch (arch) {
    case LWSWITCH_ARCH_TYPE_SV10:
        targetId = WILLOW_GPA_TO_TARGET_ID(gpa);
        break;

    case LWSWITCH_ARCH_TYPE_LR10:
        targetId = LIMEROCK_GPA_TO_TARGET_ID(gpa);
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        targetId = LAGUNA_GPA_TO_TARGET_ID(gpa);
        break;
#endif

    default:
        break;
    }

    return targetId;
}

uint64_t
FMDeviceProperty::getGpaFromTargetId(lwSwitchArchType arch, uint32_t targetId)
{
    uint64_t gpa = 0;

    switch (arch) {
    case LWSWITCH_ARCH_TYPE_SV10:
        gpa = WILLOW_TARGET_ID_TO_GPA(targetId);
        break;

    case LWSWITCH_ARCH_TYPE_LR10:
        gpa = LIMEROCK_TARGET_ID_TO_GPA(targetId);
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        gpa = LAGUNA_TARGET_ID_TO_GPA(targetId);
        break;
#endif

    default:
        break;
    }

    return gpa;
}

uint32_t
FMDeviceProperty::getGpaRemapIndexFromTargetId(lwSwitchArchType arch, uint32_t targetId)
{
    uint64_t gpa = 0;

    switch (arch) {
    case LWSWITCH_ARCH_TYPE_SV10:
        gpa = WILLOW_TARGET_ID_TO_GPA_INDEX(targetId);
        break;

    case LWSWITCH_ARCH_TYPE_LR10:
        gpa = LIMEROCK_TARGET_ID_TO_GPA_INDEX(targetId);
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        gpa = LAGUNA_TARGET_ID_TO_GPA_INDEX(targetId);
        break;
#endif

    default:
        break;
    }

    return gpa;
}

uint32_t
FMDeviceProperty::getTargetIdFromGpaEgm(lwSwitchArchType arch, uint64_t Egm)
{
    return getTargetIdFromGpa(arch, Egm);
}

uint64_t
FMDeviceProperty::getGpaEgmFromTargetId(lwSwitchArchType arch, uint32_t targetId)
{
    uint64_t egm = 0;

    switch (arch) {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        egm = LAGUNA_TARGET_ID_TO_GPA_EGM(targetId);
        break;
#endif

    default:
        break;
    }

    return egm;
}

RemapTable
FMDeviceProperty::getSpaRemapTbl(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].spaRemapTbl;
}

uint32_t
FMDeviceProperty::getFirstSpaRemapSlot(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].firstSpaRemapSlot;
}

uint32_t
FMDeviceProperty::getNumSpaRemapEntriesPerGpu(lwSwitchArchType arch)
{
    return mSwitchSpec[arch].spaRemapEntriesPerGpu;
}

uint32_t
FMDeviceProperty::getSpaRemapIndexFromSpaAddress(lwSwitchArchType arch, uint64_t spaAddr)
{
    uint64_t spaIndex = 0;

    switch (arch) {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_ARCH_TYPE_LS10:
        spaIndex = LAGUNA_SPA_TO_SPA_REMAP_INDEX(spaAddr);
        break;
#endif

    default:
        break;
    }

    return spaIndex;
}

uint32_t
FMDeviceProperty::getLWLinksPerGpu(lwSwitchArchType arch)
{
    return mGpuSpec[arch].numLWLinksPerGpu;
}

uint64_t
FMDeviceProperty::getAddressRangePerGpu(lwSwitchArchType arch)
{
    return mGpuSpec[arch].fabricAddressRange;
}

uint64_t
FMDeviceProperty::getEgmAddressRangePerGpu(lwSwitchArchType arch)
{
    return mGpuSpec[arch].egmAddressRange;
}

