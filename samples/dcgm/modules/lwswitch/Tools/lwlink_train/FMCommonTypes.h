/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include <stdint.h>
#include <string.h>

/**
 * Buffer size guaranteed to be large enough for pci bus id
 */
#define FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE  32

typedef struct {
    unsigned int domain;             // PCI domain on which the device's bus resides, 0 to 0xffffffff
    unsigned int bus;                // bus on which the device resides, 0 to 0xff
    unsigned int device;             // device's id on the bus, 0 to 31
    unsigned int function;           // PCI function information
    char busId[FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //the tuple domain:bus:device PCI identifier (&amp; NULL terminator)
} FMPciInfo_t;

/**
 * PCI format string for ::busId
 */
#define FM_DEVICE_PCI_BUS_ID_FMT                  "%08X:%02X:%02X.0"

/**
 * Utility macro for filling the pci bus id format from a lwmlPciInfo_t
 */
#define FM_DEVICE_PCI_BUS_ID_FMT_ARGS(pciInfo)    (pciInfo)->domain, \
                                                  (pciInfo)->bus,    \
                                                  (pciInfo)->device

#define FM_UUID_BUFFER_SIZE 80 // ref LWML

typedef struct FMUuid_st{
    char bytes[FM_UUID_BUFFER_SIZE];
    bool operator==(const FMUuid_st& rhs)
    {
        if (0 == memcmp(bytes, rhs.bytes, sizeof(FMUuid_st))) {
            return true;
        }
        // no match
        return false;
    }    
} FMUuid_t;

typedef struct {
    unsigned int gpuIndex;      // GPU Index
    FMPciInfo_t pciInfo;        // The PCI BDF information
    FMUuid_t uuid;              // UUID (ASCII string format)
} FMGpuInfo_t;


typedef struct {
    FMPciInfo_t pciInfo;        // The PCI information for the blacklisted GPU
    FMUuid_t uuid;              // The ASCII string UUID for the blacklisted GPU
} FMBlacklistGpuInfo_t;


