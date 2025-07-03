/*++

Copyright (C) LWPU Corporation, 2020

Module Name:

    driver.h

Abstract:

    This file contains declarations for lwmetf.sys --
    test kernel mode filter driver for queue reservation code in storlwme.sys.

--*/

#pragma once

#ifndef __DRIVER__H__
#define __DRIVER__H__

#include <wdf.h>
#include <initguid.h>
#include <wdmguid.h>
#include <ntddscsi.h>
#include <lwme.h>
#include <limits.h>
#include "trace.h"
#include "LWMePrivateHeader.h"
#include <windef.h>
#include <ntifs.h>
#include <ntddstor.h>
#include "lwMemoryWindows.h"

class LWMeCQueue;
class LWMeSQueue;

#if DBG
// Enable ETW logging on debug driver by default.
#define USE_ETW_LOGGING
#endif

// LWMe caps (hardcoded for now.)
#define LW_SECTOR_SIZE          512
#define LW_SECTOR_IN_CLUSTER    8
#define LW_CLUSTER_SIZE         (LW_SECTOR_IN_CLUSTER * LW_SECTOR_SIZE) // 4096

// TODO: Need to use sector/cluster dynamically.
//#define LW_SECTOR_SIZE          (DeviceContext->m_bytesPerSector)
//#define LW_SECTOR_IN_CLUSTER    (DeviceContext->m_sectorsPerCluster)
//#define LW_CLUSTER_SIZE         (DeviceContext->m_bytesPerCluster)

#define LW_MAX_READ_PER_COMMAND (DeviceContext->m_ReadSizePerCmd)
#define LW_MIN_READ_PER_COMMAND (LW_SECTOR_SIZE) // 512 byts = 1 sector
#define LW_MAX_PRP_PER_COMMAND  (LW_MAX_READ_PER_COMMAND / LW_CLUSTER_SIZE)

#define LW_DEFAULT_QUEUE_DEPTH  (256) // this will be used only on fallback


#define POLLING_RETRY_INTERVAL_MS  1  // 1 milisec

#define DPFLTR_VERBOSE_LEVEL 4

#define TRACE_DRIVER 0
#define TRACE_QUEUE 1

#define WPP_INIT_TRACING(_DRIVER, _REGISTRY)
#define WPP_CLEANUP(_DRIVER)

#ifndef LWBIT
#define LWBIT(n)  (1<<n)
#endif

#if DBG
#define CHECK_IRQL(irql)                                                                                      \
        if (KeGetLwrrentIrql() > irql)                                                                            \
        {                                                                                                         \
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER,"IRQL mismatch in %s (e:%d c:%d)\n",                     \
                __FUNCTION__, irql, KeGetLwrrentIrql());                                                          \
            __debugbreak();                                                                                       \
        }
#define LW_LOG_ENABLE
#else
#define CHECK_IRQL(irql)
// #define LW_LOG_ENABLE   // Uncomment to enable logging in release driver.
#endif // DEBUG

#define INIT_CODE               "INIT"
#define NONPAGE_CODE            ".text"  
#define PAGE_CODE               "PAGE"

#define PUSH_SEGMENTS           __pragma(code_seg(push)) __pragma(data_seg(push)) __pragma(bss_seg(push)) __pragma(const_seg(push))
#define POP_SEGMENTS            __pragma(code_seg(pop))  __pragma(data_seg(pop))  __pragma(bss_seg(pop))  __pragma(const_seg(pop))

#define CODE_SEGMENT(__seg)     __pragma(code_seg(__seg))
#define DATA_SEGMENT(__seg)     __pragma(data_seg(__seg))
#define BSS_SEGMENT(__seg)      __pragma(bss_seg(__seg))
#define CONS_SEGMENT(__seg)     __pragma(const_seg(__seg))

#define LW_VENDOR_ID                0x10DE

typedef struct _DRIVER_GLOBAL
{
    // Driver information
    DRIVER_OBJECT* pDriverObject;
}DRIVER_GLOBAL;

typedef struct _MEMORY_REQUEST_DATA
{
    ULONG Length;
    PVOID MapIo;
    PUCHAR Buffer;
    PMDL Mdl;
} MEMORY_REQUEST_DATA, *PMEMORY_REQUEST_DATA;

typedef struct _LOCKED_MDL
{
    LIST_ENTRY  ListEntry;
    HANDLE      ProcessId;
    PMDL        pMdl;
} LOCKED_MDL, *PLOCKED_MDL;

typedef struct _CREATE_QUEUES_BUFFER
{
    SRB_IO_CONTROL                          SrbIoCtl;
    LWME_RESERVED_QUEUES_CREATE_REQUEST     CreateQueueRequest;
    LWME_RESERVED_QUEUES_CREATE_RESPONSE    CreateQueueResponse;
} CREATE_QUEUES_BUFFER, *PCREATE_QUEUES_BUFFER;

typedef struct _RESERVE_QUEUES_BUFFER
{
    SRB_IO_CONTROL                          SrbIoCtl;
    LWME_RESERVED_QUEUES_PROPERTIES         QueueProperties;
} RESERVE_QUEUES_BUFFER, * PRESERVE_QUEUES_BUFFER;

typedef struct _MAP_ALLOC_BUFFER
{
    SRB_IO_CONTROL                          SrbIoCtl;
    LWME_MAP_ALLOC                          MapAlloc;
} MAP_ALLOC_BUFFER, * PMAP_ALLOC_BUFFER;

#ifdef __cplusplus
extern "C"
{
#endif

    void TraceEvents(int DebugPrintLevel, int driverOrQueue, PCCHAR DebugMessage, ...);
    NTSTATUS  MappingMemory(__in PVOID Va, __in ULONG size, __out PMEMORY_REQUEST_DATA memoryRequest);

#ifdef __cplusplus
}
#endif


// Any remaining pinned mdl with be unpinned and removed on process exits. This is in case app crashed or missed clean up. 
// Not cleaning up pinned allocation results in BSOD.

static LIST_ENTRY  g_PinnedAllocLockedList;
static KSPIN_LOCK  g_PinnedAllocListLock;

enum POLLING_MODE
{
    POLLING_MODE_IDLE,
    POLLING_MODE_ENABLE,
};
enum _IOCTL_STORAGE_SET_PROPERTY
{
    IOCTL_STORAGE_SET_PROPERTY_VIBRANIUM = CTL_CODE(IOCTL_STORAGE_BASE, 0x0503, METHOD_BUFFERED, FILE_WRITE_ACCESS),
    IOCTL_STORAGE_SET_PROPERTY_COLBAT = CTL_CODE(IOCTL_STORAGE_BASE, 0x04FF, METHOD_BUFFERED, FILE_WRITE_ACCESS)
};

typedef struct _DEVICE_CONTEXT {
    ULONG                               NumberOfSqs;                // Number of SQs

    KDPC                                PollingDpc;                 // Lwstomtimer DPC object
    
    KTIMER                              PollingTimer;               // Polling timer object
    ULONG                               PollingIntervalMS;          // Polling interval when active waiter (retry interval)
    volatile LONG                       pollingStatustEntryCounter; // This is used to emulate critical section with automic increment & decrement in UpdatePollingDPCState function.
    POLLING_MODE                        pollingMode;

    LWMeCQueue*                         CompletionQueue;
    LWMeSQueue*                         SubmissionQueues[LW_MAX_SQS];
    BOOLEAN                             IsQueuesAvailable;
    USHORT                              m_queueDepth;

    MEMORY_REQUEST_DATA                 SQCompleteRefCountMappingData[LW_MAX_SQS];

    UNICODE_STRING                      m_volumeName;               // To be used to ignore if different file name is used from the same volume
    ULONG                               m_bytesPerSector;
    ULONG                               m_sectorsPerCluster;
    ULONG                               m_bytesPerCluster;

    ULONG                               m_ReadSizePerCmd;

    PLWME_IDENTIFY_CONTROLLER_DATA      m_pIdentifyControllerData;
    BOOLEAN                             m_assertOnFailure;

} DEVICE_CONTEXT, *PDEVICE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DEVICE_CONTEXT,
                                   GetDeviceContext)

EXTERN_C_START

DRIVER_INITIALIZE DriverEntry;
EVT_WDF_DRIVER_DEVICE_ADD lwmetfEvtDeviceAdd;
EVT_WDF_OBJECT_CONTEXT_CLEANUP lwmetfEvtDriverContextCleanup;
EVT_WDF_IO_QUEUE_IO_DEVICE_CONTROL lwmetfEvtIoDeviceControl;

EXTERN_C_END

void TraceEvents(int DebugPrintLevel, int driverOrQueue, PCCHAR DebugMessage, ...);
LONG MyExceptionFilter(_In_ PEXCEPTION_POINTERS ExceptionPointer);
VOID ProcessCallback(HANDLE hParentId, HANDLE hProcessId, BOOLEAN bCreate);
PDEVICE_OBJECT* LwEnumeratePciDeviceObjectList(PULONG ActualNumberDeviceObjects);
PBUS_INTERFACE_STANDARD getPciInterface(PDEVICE_OBJECT PhysicalDeviceObject);
void freePciInterface(PBUS_INTERFACE_STANDARD PciInterface);

VOID UpdatePollingDPCState(_In_ PDEVICE_CONTEXT DeviceContext, _In_ BOOL calledFromPolling);


BOOLEAN AllocateMemoryForQueues(_In_ WDFDEVICE Device);
BOOLEAN AllocateMemoryForQueuesAndTransfers(_In_ PDEVICE_CONTEXT DeviceContext, _In_ ULONG NumberOfSQs, _In_ USHORT QueueDepth);

NTSTATUS CreateReservedQueue(_In_ WDFDEVICE Device, _In_ PSRB_IO_CONTROL SrbIoControl);
NTSTATUS ReservedQueueQuery(_In_ WDFDEVICE Device, _In_ PSRB_IO_CONTROL SrbIoControl);
NTSTATUS CreateMemoryMapping(_In_ WDFDEVICE Device, _In_ PSRB_IO_CONTROL SrbIoControl);
VOID HandleLWMetfDriverHandShake(_In_ WDFDEVICE Device, _In_ WDFREQUEST Request, _In_ PSRB_IO_CONTROL SrbIoControl, _In_ size_t OutputBufferLength, _In_ size_t InputBufferLength);

void logSystemEvent(NTSTATUS errorCode, const char* pFormat, ...);
void logSystemEventV(NTSTATUS errorCode, const char* pFormat, va_list arglist);
#endif // __DRIVER__H__