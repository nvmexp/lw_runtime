/*++

Copyright (C) LWPU Corporation, 2020

Module Name:

    CQueue.cpp

Abstract:

    This file implements lwmetf.sys --
    kernel mode filter driver for using LWMe queue reservation code in storlwme.sys for LWAPI direct storage implementation.

--*/
#define ENABLE_WPP_RECORDER 1

#define TRACE_QUEUE 1

#if defined(LW_LDDM) && (LW_LDDM >= 22499)
// Disable warning C4996 due to ExAllocatePoolWithTag deprecation
#pragma warning(push)
#pragma warning( disable : 4996 )
#endif
#include <Ntifs.h>
#if defined(LW_LDDM) && (LW_LDDM >= 22499)
#pragma warning(pop)
#endif
#include <storswtr.h>
#include <wdf.h>
#include <wdm.h>
#include <stdarg.h>
#include <ntddk.h>
#include <Ntddvol.h>
#include "driver.h"
#include "CQueue.h"

#ifdef USE_ETW_LOGGING
#include "etw\lwmetfETW.h"
#endif

PUSH_SEGMENTS

CODE_SEGMENT(PAGE_CODE)
LWMeCQueue::LWMeCQueue() :
    m_pPhaseTag(NULL),
    m_IsAllocated(FALSE),
    m_IsCreated(FALSE),
    m_CqVas(NULL),
    m_CqPas(NULL),
    m_CqHeadPointer(0)
{
    CHECK_IRQL(PASSIVE_LEVEL);
    RtlZeroMemory(m_SubmissionQueues, sizeof(m_SubmissionQueues));
    m_pollingEntryCounter = 0;
}

CODE_SEGMENT(PAGE_CODE)
LWMeCQueue::~LWMeCQueue()
{
    CHECK_IRQL(PASSIVE_LEVEL);
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
LWMeCQueue::DoFreeMemory()
/*++

Routine Description:

    Free allocated queues and reset physical addresses of the queues in
    the IOCTL request buffer.

--*/
{
    CHECK_IRQL(DISPATCH_LEVEL);
    m_IsAllocated = FALSE;

    if (m_CqPas)
    {
        m_CqPas = 0;
    }

    if (m_CqVas)
    {
        MmFreeContiguousMemory(m_CqVas);
        m_CqVas = 0;
    }

    if (m_pPhaseTag)
    {
        ExFreePoolWithTag(m_pPhaseTag, LW_LWME_TAG);
        m_pPhaseTag = 0;
    }

    return TRUE;
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
LWMeCQueue::DoAlloc(
    USHORT QueueDepth
)
/*++

Routine Description:

    Allocate non-paged memory for the specified number of queues,
    update the IOCTL input buffer with physical addresses of the queues,
    save the queue informations into the device context, so we could free the queues later.
--*/
{
    CHECK_IRQL(DISPATCH_LEVEL);
    ASSERT(m_IsAllocated == FALSE);

    BOOLEAN success = FALSE;
    SIZE_T allocationSize;
    PHYSICAL_ADDRESS lowPa, highPa;
    lowPa.QuadPart = 0;
    highPa.QuadPart = ULLONG_MAX;

    allocationSize = sizeof(LWME_COMPLETION_ENTRY) * QueueDepth;
    m_CqVas = (PLWME_COMPLETION_ENTRY) MmAllocateContiguousMemorySpecifyCache(allocationSize, lowPa, highPa, lowPa, MmNonCached);
    if (!m_CqVas)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to allocate CQ %d!\n", allocationSize);

        logSystemEvent(STATUS_NO_MEMORY, "Fail to allocate contiguous memory");
        goto Cleanup;
    }
    RtlZeroMemory(m_CqVas, allocationSize);

    allocationSize = QueueDepth * sizeof(CHAR);

    m_pPhaseTag = (PCHAR)lwAllocatePoolWithTag(NonPagedPool, allocationSize, LW_LWME_TAG);
    if (!m_pPhaseTag)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to allocate m_pPhaseTag size %d!\n", allocationSize);

        logSystemEvent(STATUS_NO_MEMORY, "Fail to allocate non page memory");
        goto Cleanup;
    }
    RtlZeroMemory(m_pPhaseTag, allocationSize);
    
    m_CqPas = MmGetPhysicalAddress(m_CqVas).QuadPart;
    m_IsAllocated = TRUE;

    success = TRUE;

Exit:
    return success;

Cleanup:
    DoFreeMemory();
    goto Exit;
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
LWMeCQueue::DoDestroy()
{
    CHECK_IRQL(DISPATCH_LEVEL);

    ASSERT(m_IsCreated);
    ASSERT(m_IsAllocated);
    ASSERT(m_IsAllocated == FALSE);
    m_IsCreated = FALSE;
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
LWMeCQueue::DoCreate(
    PLWME_RESERVED_CQ_INFO pCQInfo
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    ASSERT(m_IsAllocated);
    ASSERT(!m_IsCreated);

    //
    // Save the details about created queues into the device context,
    // so we could ring the doorbells, etc.
    //
    m_Info = *pCQInfo;
    m_CqHeadPointer = 0;

    m_IsCreated = TRUE;

    KeInitializeSpinLock(&m_ProcessComplectionLock);

    return TRUE;
}

CODE_SEGMENT(NONPAGE_CODE)
NTSTATUS
LWMeCQueue::CheckForCompletion(BOOL isCalledFromDPC)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    NTSTATUS status = STATUS_TIMEOUT;
    volatile ULONG* cqDoorbell = (volatile ULONG*)(ULONG_PTR)m_Info.DoorbellRegisterAddress;

    LONG entryCounter = InterlockedIncrement(&m_pollingEntryCounter);
    if (entryCounter == 1)
    {
        ULONG completedCount = 0;
        USHORT cqIndex = m_CqHeadPointer;

        while (true) // If we find consulwtive unprocessed completion entries we will try to process in this call.
        {
            KDPC_WATCHDOG_INFORMATION WatchdogInfo;

            if (isCalledFromDPC)
            {
                NTSTATUS watchdogStatus = KeQueryDpcWatchdogInformation(&WatchdogInfo);
                if (NT_SUCCESS(watchdogStatus)
                    && WatchdogInfo.DpcTimeLimit != 0 // Verify the watchdog is enabled
                    && WatchdogInfo.DpcTimeCount < WatchdogInfo.DpcTimeLimit / 4)
                {
                    // Once we go below 25% of the limit we exit this DPC and allow to DPC reset it. Not breaking will lead to BSOD 0x133.
                    // Processing completion will resume in next Timer DPC interval call.
                    break;
                }
            }
            else
            {
                // Not breaking loop can lead to extra holdup in usermode thread in case other thead is still submitting & disk completing work, which can be postponed to DPC timer polling.
                // For now putting, hard limit of max 1000 processing per single call.
                if (completedCount >= 1000)
                {
                    // Exit loop
                    break;
                }
            }

            volatile PLWME_COMPLETION_ENTRY ce = &m_CqVas[cqIndex];
            if (ce->DW3.Status.P == m_pPhaseTag[cqIndex])
            {
                // There are no completion entries available to process right now.
                break;
            }

            m_pPhaseTag[cqIndex] = (UCHAR)ce->DW3.Status.P;
            completedCount++;

            if (ce->DW3.CID <= LW_MAX_SQS * m_Info.QueueDepth)
            {
                USHORT queuePriority = CID_TO_QUEUE_PRIORITY(ce->DW3.CID);
                USHORT sqEntryIndex = CID_TO_IDX(ce->DW3.CID);

                ASSERT(m_SubmissionQueues[queuePriority]->IsValid());

                m_SubmissionQueues[queuePriority]->DoMarkCompletion(sqEntryIndex);

                TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "Read completed CID 0x%x, cq index 0x%x, lwme status 0x%x, sq priority %d, sq Index %d, \n",
                    ce->DW3.CID, cqIndex, ce->DW3.Status.SC, queuePriority, sqEntryIndex);

                if (ce->DW3.Status.SC == LWME_STATUS_SUCCESS_COMPLETION)
                {
                    status = STATUS_SUCCESS;
                }
                else
                {
                    ASSERT(0);
                    status = STATUS_UNEXPECTED_IO_ERROR;
                    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Read completed with LWMe StatusCode 0x%x, StatusCodeType 0x%x.\n", ce->DW3.Status.SC, ce->DW3.Status.SCT);
                    logSystemEvent(status, "Read completed with LWMe StatusCode 0x%x, StatusCodeType 0x%x.\n", ce->DW3.Status.SC, ce->DW3.Status.SCT);
                    break;
                }
            }
            else
            {
                ASSERT(0);
                TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, " ERROR invalid CID found in CQ entry - CID 0x%x, CQ index 0x%x, lwme status 0x%x.\n", ce->DW3.CID, cqIndex, ce->DW3.Status.SC);
                logSystemEvent(status, "ERROR invalid CID found in CQ entry - CID 0x % x, CQ index 0x % x, lwme status 0x % x.\n", ce->DW3.CID, cqIndex, ce->DW3.Status.SC);
            }

            cqIndex = (cqIndex + 1) % m_Info.QueueDepth;
            WRITE_REGISTER_ULONG(cqDoorbell, (ULONG)cqIndex);
            m_CqHeadPointer = cqIndex;
        }
    }

    InterlockedDecrement(&m_pollingEntryCounter);

    return status;
}

CODE_SEGMENT(PAGE_CODE)
void LWMeCQueue::operator delete(void* ptr)
{
    CHECK_IRQL(PASSIVE_LEVEL);
    ExFreePool(ptr);
}

CODE_SEGMENT(PAGE_CODE)
void* LWMeCQueue::operator new(size_t size)
{
    CHECK_IRQL(PASSIVE_LEVEL);
    return lwAllocatePoolWithTag(PagedPool, size, LW_LWME_TAG);
}

POP_SEGMENTS