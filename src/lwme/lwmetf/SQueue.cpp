/*++

Copyright (C) LWPU Corporation, 2020

Module Name:

    SQueue.cpp

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
#include "SQueue.h"
#include "CQueue.h"

#ifdef USE_ETW_LOGGING
#include "etw\lwmetfETW.h"
#endif

PUSH_SEGMENTS

CODE_SEGMENT(PAGE_CODE)
LWMeSQueue::LWMeSQueue(PDEVICE_CONTEXT pDeviceContext) :
    m_SubmissionRefCount(0),
    m_CompletionRefCount(0),
    m_pEntryStatus(NULL),
    m_IsAllocated(FALSE),
    m_IsCreated(FALSE),
    DeviceContext(pDeviceContext),
    m_notificationProcessEntryCounter(0),
    m_SqVas(NULL),
    m_SqPas(NULL),
    m_prpVa(NULL)
{
    CHECK_IRQL(PASSIVE_LEVEL);
}

CODE_SEGMENT(PAGE_CODE)
LWMeSQueue::~LWMeSQueue()
{
    CHECK_IRQL(PASSIVE_LEVEL);
}

CODE_SEGMENT(NONPAGE_CODE)
ULONG LWMeSQueue::GetPendingCommandCount()
{
    CHECK_IRQL(HIGH_LEVEL);
    ASSERT(m_SubmissionRefCount >= m_CompletionRefCount);

    return (ULONG) (m_SubmissionRefCount - m_CompletionRefCount);
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
LWMeSQueue::DoFreeMemory()
/*++

Routine Description:

    Free allocated queues and reset physical addresses of the queues in
    the IOCTL request buffer.

--*/
{
    CHECK_IRQL(DISPATCH_LEVEL);
    
    ASSERT(m_IsAllocated);
    m_IsAllocated = FALSE;

    // lwmestor driver is freeing Queue command allocations when called to remove queues.
    // And freeing it again would result in bugcheck.

    if(m_SqPas)
    {
        m_SqPas = 0;
    }

    if (m_SqVas)
    {
        MmFreeContiguousMemory(m_SqVas);
        m_SqVas = 0;
    }
     
    if (m_prpVa)
    {
        MmFreeContiguousMemory(m_prpVa);
        m_prpVa = 0;
    }

    if (m_pEntryStatus)
    {
        ExFreePoolWithTag(m_pEntryStatus, LW_LWME_TAG);
        m_pEntryStatus = 0;
    }

    return TRUE;
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
LWMeSQueue::DoAlloc(
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

    allocationSize = sizeof(LWME_COMMAND) * QueueDepth;
    m_SqVas = MmAllocateContiguousMemorySpecifyCache(allocationSize, lowPa, highPa, lowPa, /*MmCached*/MmWriteCombined);
    if (!m_SqVas)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to allocate SQ [%d] size %d! Type:MmWriteCombined\n", QueueDepth, allocationSize);
        logSystemEvent(STATUS_NO_MEMORY, "%s()::%d Fail to allocate contiguous memory", __FUNCDNAME__, __LINE__, QueueDepth, allocationSize);
        m_SqVas = MmAllocateContiguousMemorySpecifyCache(allocationSize, lowPa, highPa, lowPa, MmNonCached);
        if (!m_SqVas)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL,
                TRACE_QUEUE,
                "Retry Failed to allocate SQ size %d! Type:MmNonCached\n", allocationSize);
            logSystemEvent(STATUS_NO_MEMORY, "%s()::%d Retry Failed to allocate SQ size %d! Type:MmNonCached\n", __FUNCDNAME__, __LINE__, allocationSize);
            goto Cleanup;
        }
    }
    RtlZeroMemory(m_SqVas, allocationSize);

    allocationSize = QueueDepth * sizeof(BOOLEAN);
    m_pEntryStatus = (PBOOLEAN)lwAllocatePoolWithTag(NonPagedPool, allocationSize, LW_LWME_TAG);
    if (!m_pEntryStatus)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to allocate SqCmdStatus, size %d !\n", allocationSize);
        logSystemEvent(STATUS_NO_MEMORY, "%s()::%d Failed to allocate SqCmdStatus, size %d !\n", __FUNCDNAME__, __LINE__, allocationSize);
        goto Cleanup;
    }
    RtlZeroMemory(m_pEntryStatus, allocationSize);

    
    allocationSize = QueueDepth * sizeof(LWME_PRP_ENTRY) * LW_MAX_PRP_PER_COMMAND;
    m_prpVa = MmAllocateContiguousMemorySpecifyCache(allocationSize, lowPa, highPa, lowPa, MmWriteCombined);
    if (!m_prpVa)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to allocate prpVa! size %d Type:MmWriteCombined\n", allocationSize);

        m_prpVa = MmAllocateContiguousMemorySpecifyCache(allocationSize, lowPa, highPa, lowPa, MmNonCached);
        if (!m_prpVa)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL,
                TRACE_QUEUE,
                "Retry Failed to allocate prpVa size %d! Type:MmNonCached\n", allocationSize);

            goto Cleanup;
        }
    }

    // As per LWMe specs PRP tobe qword aligned which would be served for new allocation 4KB Aligned.
    // MmAllocateContiguousMemorySpecifyCache : "The routine aligns the starting address of a contiguous memory allocation to a memory page boundary."
    ASSERT((ULONGLONG)m_prpVa % 16 == 0);

    RtlZeroMemory(m_prpVa, allocationSize);
    m_SqPas = MmGetPhysicalAddress(m_SqVas).QuadPart;
    m_IsAllocated = TRUE;

    success = TRUE;

Exit:
    return success;

Cleanup:
    DoFreeMemory();
    goto Exit;
}

CODE_SEGMENT(NONPAGE_CODE)
inline
VOID
LWMeSQueue::DoCreateReadCommand(
    _In_ volatile PLWME_COMMAND  c,
    _In_ PUCHAR                  DataVA,
    _In_ ULONG                   DataWriteSize,
    _In_ UINT16                  CID,
    _In_ PLWME_PRP_ENTRY         pPRP,
    _In_ ULONG                   PRPAvailablecount,
    _In_ ULONGLONG               LBA
    )
{
    CHECK_IRQL(HIGH_LEVEL);

    ASSERT(IsValid());
    ASSERT(CID < m_Info.QueueDepth * (m_Info.QueuePriority + 1));

    // CLEAR command
    RtlZeroMemory(c, sizeof(LWME_COMMAND));

    ULONG NLB = DataWriteSize / LW_SECTOR_SIZE;

    if (DataWriteSize <= LW_SECTOR_SIZE)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "LWME read size 0x%x should not be less than 1 block = 0x%x.\n", DataWriteSize, LW_SECTOR_SIZE);
        logSystemEvent(STATUS_NO_MEMORY, "%s()::%d LWME read size 0x%x should not be less than 1 block = 0x%x.\n", __FUNCDNAME__, __LINE__, DataWriteSize, LW_SECTOR_SIZE);
        NLB = 0;
    }
    else if (DataWriteSize % LW_SECTOR_SIZE == 0)
    {
        ASSERT(NLB != 0);
        NLB--;
    }

    ASSERT( ((NLB+1) * LW_SECTOR_SIZE) >= DataWriteSize);

    c->CDW0.OPC = LWME_LWM_COMMAND_READ;
    c->CDW0.CID = CID;

    c->NSID = 1; // Use first namespace
    c->CDW0.PSDT = 0; // PRP (Physical region page)
    c->u.READWRITE.CDW12.NLB = NLB;

    c->u.READWRITE.LBALOW  = (ULONG)(LBA & 0xffffffff);
    c->u.READWRITE.LBAHIGH = (ULONG)(LBA >> 32);

    c->PRP1 = MmGetPhysicalAddress(DataVA).QuadPart;

    if (c->u.READWRITE.CDW12.NLB < (LW_SECTOR_IN_CLUSTER * 2) )
    {
        c->PRP2 = MmGetPhysicalAddress(DataVA + LW_CLUSTER_SIZE).QuadPart;
    }
    else
    {
        ULONG entriesRequired = (DataWriteSize / LW_CLUSTER_SIZE) - 1; // -1 since One page is read in PRP1.

        if ((DataWriteSize % LW_CLUSTER_SIZE) != 0)
        {
            entriesRequired++;
        }

        ASSERT(entriesRequired <= PRPAvailablecount);
        entriesRequired = min(entriesRequired, PRPAvailablecount);

        RtlZeroMemory(pPRP, sizeof(LWME_PRP_ENTRY) * entriesRequired);

        for (ULONG i = 0; i < entriesRequired; i++)
        {
            ULONG offset = (i + 1) * LW_CLUSTER_SIZE;
            pPRP[i].AsUlonglong = MmGetPhysicalAddress(DataVA + offset).QuadPart;
        }

        c->PRP2 = MmGetPhysicalAddress(pPRP).QuadPart;
    }

    c->CDW0.CID = CID;

#if defined(SUBMISSION_DEBUG_TRACE)
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->CDW0.OPC                  = 0x%x.\n", c->CDW0.OPC);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->CDW0.FUSE                 = 0x%x.\n", c->CDW0.FUSE);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->CDW0.PSDT                 = 0x%x.\n", c->CDW0.PSDT);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->CDW0.CID                  = 0x%x.\n", c->CDW0.CID);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->NSID                      = 0x%x.\n", c->NSID);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->MPTR                      = 0x%llx.\n", c->MPTR);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->PRP1                      = 0x%llx.\n", c->PRP1);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->PRP2                      = 0x%llx.\n", c->PRP2);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->u.READWRITE.LBALOW        = 0x%x.\n", c->u.READWRITE.LBALOW);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->u.READWRITE.LBAHIGH       = 0x%x.\n", c->u.READWRITE.LBAHIGH);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->u.READWRITE.CDW12.NLB     = 0x%x.\n", c->u.READWRITE.CDW12.NLB);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->u.READWRITE.CDW13.AsUlong = 0x%x.\n", c->u.READWRITE.CDW13.AsUlong);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->u.READWRITE.CDW14         = 0x%x.\n", c->u.READWRITE.CDW14);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "c->u.READWRITE.CDW14.AsUlong = 0x%x.\n", c->u.READWRITE.CDW15.AsUlong);
#endif // SUBMISSION_DEBUG_TRACE
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
LWMeSQueue::DoDestroy()
{
    CHECK_IRQL(DISPATCH_LEVEL);

    ASSERT(m_IsCreated);
    ASSERT(m_IsAllocated);
    ASSERT(m_IsAllocated == FALSE);
    m_IsCreated = FALSE;
}

CODE_SEGMENT(PAGE_CODE)
NTSTATUS 
LWMeSQueue::RegisterNotification(_In_ HANDLE hUserEvent, _In_ ULONGLONG refCount)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    PKEVENT pKrnlEvent = NULL;

    // Can't wait for work which is not submitted yet..
    ASSERT(refCount <= (ULONGLONG)m_SubmissionRefCount);
    if (refCount > (ULONGLONG)m_SubmissionRefCount)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER,
            "RegisterNotification: Invalid register notify on SQ[%d] for LwrrentSubRefCout = 0x%llx is NOT <= wait for ref 0x%llx ",
            m_Info.QueuePriority, m_SubmissionRefCount, refCount);
        logSystemEvent(STATUS_ILWALID_PARAMETER_2,
            "%s()::%d RegisterNotification: Invalid register notify on SQ[%d] for LwrrentSubRefCout = 0x%llx is NOT <= wait for ref 0x%llx ",
            __FUNCDNAME__, __LINE__, m_Info.QueuePriority, m_SubmissionRefCount, refCount);

        return STATUS_ILWALID_PARAMETER_2;
    }

    NTSTATUS status = ObReferenceObjectByHandle(hUserEvent,
        SYNCHRONIZE,
        *ExEventObjectType,
        UserMode,
        reinterpret_cast<PVOID*>(&pKrnlEvent),
        NULL);
    if (!NT_SUCCESS(status))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER,
            "RegisterNotification: ObReferenceObjectByHandle failed, handle is not valid (%p), status = 0x%x ", hUserEvent, status);
        logSystemEvent(status,
            "%s()::%d RegisterNotification: ObReferenceObjectByHandle failed, handle is not valid (%p)", __FUNCDNAME__, __LINE__, hUserEvent);
        switch (status)
        {
        case STATUS_OBJECT_TYPE_MISMATCH:
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "STATUS_OBJECT_TYPE_MISMATCH\n");
            break;
        case STATUS_ACCESS_DENIED:
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "STATUS_ACCESS_DENIED\n");
            break;
        case STATUS_ILWALID_HANDLE:
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "STATUS_ILWALID_HANDLE\n");
            break;
        default:
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "STATUS_UNKNOWN\n");
        };

        return status;
    }

#ifdef USE_ETW_LOGGING
    EventWriteRTXIORegisterWaitEvent_AssumeEnabled((UINT32)((UINT64)pKrnlEvent), GetPriority(), refCount);
#endif

    if(!pKrnlEvent)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "RegisterNotification: ObReferenceObjectByHandle returned NULL (%p), ststus = 0x%x\n", hUserEvent, status);
        logSystemEvent(STATUS_DRIVER_INTERNAL_ERROR,
            "%s()::%d RegisterNotification: ObReferenceObjectByHandle failed, handle is not valid (%p)", __FUNCDNAME__, __LINE__, hUserEvent);
        return STATUS_DRIVER_INTERNAL_ERROR;
    }

    KeMemoryBarrier();

    if ((ULONGLONG)m_CompletionRefCount >= refCount)
    {
        TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "RegisterNotification: Event ready to be signaled refCount \n", hUserEvent);

        KeSetEvent(pKrnlEvent, 0, FALSE);
        ObDereferenceObject(pKrnlEvent);

        return STATUS_SUCCESS;
    }

    NotificationListEntry* nleNew = (NotificationListEntry*)lwAllocatePoolWithTag(
        NonPagedPoolNx, sizeof(NotificationListEntry), LW_LWME_TAG);

    if (NULL == nleNew)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "RegisterNotification: lwAllocatePoolWithTag failed (%p)\n", nleNew);
        logSystemEvent(STATUS_INSUFFICIENT_RESOURCES,
            "%s()::%d RegisterNotification: lwAllocatePoolWithTag failed (%p)\n", __FUNCDNAME__, __LINE__, nleNew);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    nleNew->kKernelEvent = pKrnlEvent;
    nleNew->RefCount = refCount;

    KIRQL  oldIrql;
    KeAcquireSpinLock(&m_NotificationListLock, &oldIrql);

    // Fail if the notification event already exists in our list.
    // This is purly as per contract with UMD driver, I am planning to make UMD code to make sure not to reuse event until it is satisfied from earlier call.
    if (!IsListEmpty(&m_NotificationList))
    {
        PLIST_ENTRY leLwrrent = m_NotificationList.Flink;
        while (leLwrrent != &m_NotificationList)
        {
            NotificationListEntry* nleLwrrent = CONTAINING_RECORD(leLwrrent, NotificationListEntry, ListEntry);
            if (nleLwrrent->kKernelEvent == pKrnlEvent)
            {
                TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "RegisterNotification: Event already present in Queue failed (%p)\n", hUserEvent);
                logSystemEvent(STATUS_UNSUCCESSFUL,
                    "%s()::%d RegisterNotification: Event already present in Queue failed (%p)\n", __FUNCDNAME__, __LINE__, hUserEvent);
                ASSERT(nleLwrrent->RefCount < refCount);
                ExFreePoolWithTag(nleNew, LW_LWME_TAG);
                return STATUS_UNSUCCESSFUL;
            }

            leLwrrent = leLwrrent->Flink;
        }
    }
    
    InsertTailList(&m_NotificationList, &(nleNew->ListEntry));
    KeReleaseSpinLock(&m_NotificationListLock, oldIrql);

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "RegisterNotification: Inserted event for refCount (%lld), current refCount %lld\n", refCount, m_CompletionRefCount);
   
    if ((ULONGLONG)m_CompletionRefCount >= refCount)
    {
        ProcessNotifications();
    }

    return STATUS_SUCCESS;
}

CODE_SEGMENT(NONPAGE_CODE)
NTSTATUS
LWMeSQueue::ProcessNotifications()
{
    CHECK_IRQL(DISPATCH_LEVEL);

    if (IsListEmpty(&m_NotificationList))
    {
        return STATUS_SUCCESS;
    }

    // If we are already running loop we can ignore additional call.
    LONG entryCounter = InterlockedIncrement(&m_notificationProcessEntryCounter);
    if (entryCounter == 1)
    {
        KIRQL  oldIrql;
        KeAcquireSpinLock(&m_NotificationListLock, &oldIrql);

        if (!IsListEmpty(&m_NotificationList))
        {
            PLIST_ENTRY leLwrrent = m_NotificationList.Flink;
            while (leLwrrent != &m_NotificationList)
            {
                NotificationListEntry* nleLwrrent = CONTAINING_RECORD(leLwrrent, NotificationListEntry, ListEntry);
                if (nleLwrrent->RefCount <= (ULONGLONG)m_CompletionRefCount)
                {
                    TraceEvents(DPFLTR_INFO_LEVEL,
                        TRACE_QUEUE,
                        "Signaled kEvent (%p) priority %d signal on ref count = 0x%llx.\n", nleLwrrent->kKernelEvent, GetPriority(), nleLwrrent->RefCount);

                    PLIST_ENTRY previous = leLwrrent->Blink;

                    // Signal event & remove it from the list
                    {
                        KeSetEvent(nleLwrrent->kKernelEvent, 0, FALSE);
#ifdef USE_ETW_LOGGING
                        EventWriteRTXIORegisterEventComplete_AssumeEnabled((UINT32)((UINT64)nleLwrrent->kKernelEvent), GetPriority(), nleLwrrent->RefCount); // TODO: add QueuePriority & RefCount 
#endif
                        ObDereferenceObject(nleLwrrent->kKernelEvent);
                    }

                    RemoveEntryList(leLwrrent);
                    ExFreePoolWithTag(nleLwrrent, LW_LWME_TAG);

                    leLwrrent = previous;
                }

                leLwrrent = leLwrrent->Flink;
            }
        }

        KeReleaseSpinLock(&m_NotificationListLock, oldIrql);
    }

    InterlockedDecrement(&m_notificationProcessEntryCounter);

    return STATUS_SUCCESS;
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
LWMeSQueue::DoCreate(
    PLWME_RESERVED_SQ_INFO pSQInfo
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    ASSERT(m_IsAllocated);
    ASSERT(!m_IsCreated);

    //
    // Save the details about created queues into the device context,
    // so we could ring the doorbells, etc.
    //
    m_Info = *pSQInfo;
    m_SqSubmitPos = 0;

    m_IsCreated = TRUE;

    KeInitializeSpinLock(&m_UpdateCompletionRefLock);
    KeInitializeSpinLock(&m_SubmissionLock);
    KeInitializeSpinLock(&m_NotificationListLock);

    InitializeListHead(&m_NotificationList);

    return TRUE;
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
LWMeSQueue::UpdateCompletionRefCount()
{
    CHECK_IRQL(DISPATCH_LEVEL);

    ASSERT(m_Info.QueueDepth > 0);
    USHORT completionIndex = m_CompletionRefCount % m_Info.QueueDepth;

    ULONGLONG refCountBefore = m_CompletionRefCount;

#ifdef DBG
    ULONG countIncreased = 0;
#endif

    // Return early and dont take critical section when not required.
    if (IsEmpty() || m_pEntryStatus[completionIndex] == TRUE)
    {
        return;
    }

    KIRQL  oldIrql;
    KeAcquireSpinLock(&m_UpdateCompletionRefLock, &oldIrql);

    while (m_pEntryStatus[completionIndex] == FALSE && !IsEmpty()) // index starts with 1 and not 0
    {
#ifdef DBG
        countIncreased++;
#endif
        InterlockedIncrement64(&m_CompletionRefCount);
        completionIndex = m_CompletionRefCount % m_Info.QueueDepth;
        if (IsEmpty())
        {
            break;
        }
    }

    KeReleaseSpinLock(&m_UpdateCompletionRefLock, oldIrql);

#ifdef USE_ETW_LOGGING
    if (countIncreased)
    {
        EventWriteCompletionRef_AssumeEnabled(GetPriority(), m_CompletionRefCount);
    }
#endif

    if (refCountBefore < (ULONGLONG)m_CompletionRefCount)
    {
        // Update notifications if we any
        ProcessNotifications();
    }

    ASSERT(m_CompletionRefCount <= m_SubmissionRefCount);

#ifdef DBG
    ASSERT(refCountBefore <= (ULONGLONG)m_CompletionRefCount);
    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "CompletionRefCount = 0x%llx, increased by 0x%x, PendingCount 0x%x\n", m_CompletionRefCount, countIncreased, GetPendingCommandCount());
#endif
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
LWMeSQueue::DoMarkCompletion(
    USHORT CID
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    USHORT entryIndex = CID_TO_IDX(CID);
    ASSERT(!IsEmpty());
    ASSERT(m_pEntryStatus[entryIndex] == 1); // It should be masked as busy...
    m_pEntryStatus[entryIndex] = 0;

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "Read completed SQ priority 0x%x, entry index 0x%x, CID 0x%x, pending 0x%x \n",
        GetPriority(), entryIndex, CID, GetPendingCommandCount());

    // Update ref count if needed.
    UpdateCompletionRefCount();
}

CODE_SEGMENT(NONPAGE_CODE)
NTSTATUS
LWMeSQueue::DoSubmitRead(
    PLWME_READ readCmd
)
{
    CHECK_IRQL(DISPATCH_LEVEL);
    CONST ULONG maxReadSize = min(max(readCmd->inDebugPerCmdReadSize, LW_MIN_READ_PER_COMMAND), LW_MAX_READ_PER_COMMAND);

    PMDL mdl = NULL; // will be used in case VA provided is userVA and not kernelVA
    PVOID userSpaceAllocKernelVA = NULL;

    NTSTATUS status = STATUS_SUCCESS;

    if (readCmd->addr.inAllocOffset >= readCmd->addr.inAllocSize)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Allocation offset 0x%llx >= Allocation size = 0x%llx, not enough space for read.\n", readCmd->inReadSize, (readCmd->addr.inAllocSize - readCmd->addr.inAllocOffset));
        status = STATUS_ILWALID_PARAMETER_4;
        logSystemEvent(status,
            "%s()::%d Allocation offset 0x%llx >= Allocation size = 0x%llx, not enough space for read.\n",
            __FUNCDNAME__, __LINE__, readCmd->inReadSize, (readCmd->addr.inAllocSize - readCmd->addr.inAllocOffset));
        goto Exit;
    }

    if (readCmd->inReadSize > (readCmd->addr.inAllocSize - readCmd->addr.inAllocOffset))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE,
            "Not enough space in VA space for read to finish. ReadSize = 0x%llx, allocation size 0x%llx \n",
            readCmd->inReadSize, (readCmd->addr.inAllocSize - readCmd->addr.inAllocOffset));
        status = STATUS_ILWALID_PARAMETER_3;
        logSystemEvent(status,
            "%s()::%d Not enough space in VA space for read to finish. ReadSize = 0x%llx, allocation size 0x%llx \n",
            __FUNCDNAME__, __LINE__, readCmd->inReadSize, (readCmd->addr.inAllocSize - readCmd->addr.inAllocOffset));
        goto Exit;
    }

    if (readCmd->inReadSize >= (ULONG_MAX - PAGE_SIZE)) // single MDL limit 4GB - PAGE_SIZE
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE,
            "Not supported read size. ReadSize = 0x%llx MaxRead supported ULONG_MAX (0x%x)-1 \n", readCmd->inReadSize, (ULONG_MAX - PAGE_SIZE));
        status = STATUS_ILWALID_PARAMETER_3;
        logSystemEvent(status,
            "%s()::%d Not supported read size. ReadSize = 0x%llx MaxRead supported ULONG_MAX (0x%x)-1 \n",
            __FUNCDNAME__, __LINE__, readCmd->inReadSize, (ULONG_MAX - PAGE_SIZE));
        goto Exit;
    }

    ASSERT(readCmd->addr.inAllocBaseVA);

    if (readCmd->addr.IsUserVA == FALSE) // So, passed va is kernel VA
    {
        userSpaceAllocKernelVA = readCmd->addr.inAllocBaseVA;
    }
    else 
    {
        ASSERT(readCmd->addr.inAllocBaseVA);
        ASSERT(readCmd->addr.inAllocSize <= (ULONG_MAX- PAGE_SIZE));

        mdl = IoAllocateMdl(readCmd->addr.inAllocBaseVA, (ULONG)readCmd->addr.inAllocSize, FALSE, FALSE, NULL);
        if (!mdl)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Failed to IoAllocateMdl, userVA ox%llx, size 0x%llx\n", readCmd->addr.inAllocBaseVA, readCmd->inReadSize);
            status = STATUS_INSUFFICIENT_RESOURCES;
            logSystemEvent(status,
                "%s()::%d Failed to IoAllocateMdl, userVA ox%llx, size 0x%llx\n",
                __FUNCDNAME__, __LINE__, readCmd->addr.inAllocBaseVA, readCmd->inReadSize);
            goto Exit;
        }

        _try
        {
            MmProbeAndLockPages(mdl, UserMode, IoWriteAccess);
        }
        _except(EXCEPTION_EXELWTE_HANDLER)
        {
            status = GetExceptionCode();
            IoFreeMdl(mdl);
            mdl = NULL;
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Failed to MmProbeAndLockPages(mdl, UserMode, IoWriteAccess), status 0x%x\n", status);
            logSystemEvent(status,
                "%s()::%d Failed to MmProbeAndLockPages(mdl, UserMode, IoWriteAccess)\n", __FUNCDNAME__, __LINE__);
            goto Exit;
        }

        userSpaceAllocKernelVA = (PVOID)MmGetSystemAddressForMdlSafe(mdl, NormalPagePriority | MdlMappingNoExelwte);
        if (!userSpaceAllocKernelVA) {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Failed to MmGetSystemAddressForMdlSafe(mdl, NormalPagePriority | MdlMappingNoExelwte)\n");
            MmUnlockPages(mdl);
            IoFreeMdl(mdl);
            mdl = NULL;
            status = STATUS_INSUFFICIENT_RESOURCES;
            logSystemEvent(status, "%s()::%d Failed to MmGetSystemAddressForMdlSafe(mdl, NormalPagePriority | MdlMappingNoExelwte)\n", __FUNCDNAME__, __LINE__);
            goto Exit;
        }
        TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "userSpaceAllocKernelVA = 0x%llx, Physical = 0x%llx\n", userSpaceAllocKernelVA, MmGetPhysicalAddress(userSpaceAllocKernelVA));

        ASSERT(userSpaceAllocKernelVA);
        ASSERT(mdl);

        if (readCmd->inReadSize > MmGetMdlByteCount(mdl))
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE,
                "Size in Allocation MmGetMdlByteCount(mdl) 0x%llx not enough for read data size 0x%llx \n",
                MmGetMdlByteCount(mdl), readCmd->inReadSize);
            logSystemEvent(status, "%s()::%d Size in Allocation MmGetMdlByteCount(mdl) 0x%llx not enough for read data size 0x%llx \n",
                __FUNCDNAME__, __LINE__,
                MmGetMdlByteCount(mdl), readCmd->inReadSize);

            MmUnlockPages(mdl);
            IoFreeMdl(mdl);
            mdl = NULL;

            goto Exit;
        }

        ASSERT(MmGetMdlByteCount(mdl) >= readCmd->addr.inAllocSize);
    }

    KIRQL  oldIrql;
    KeAcquireSpinLock(&m_SubmissionLock, &oldIrql);

    readCmd->outSubmissionRefCount = 0;

    ULONGLONG readLwrrentOffset = 0;
    
    ULONG submitCounter = 0;
    ULONG waitCallCounter = 0;

    ASSERT(m_Info.QueueDepth != 0);

    while (readLwrrentOffset < readCmd->inReadSize)
    {
        ULONG cmdReadSize = (ULONG) min( (readCmd->inReadSize - readLwrrentOffset), maxReadSize );

        volatile PLWME_COMMAND commands = (PLWME_COMMAND)m_SqVas;
        USHORT entryIndex = m_SqSubmitPos; // 0 to queueDepth-1

        volatile PBOOLEAN pSqCmdStatus = (PBOOLEAN)m_pEntryStatus;

        if (pSqCmdStatus[entryIndex] == TRUE)
        {
            ASSERT(!IsEmpty());

            ULONGLONG startWaitTickCount = KeQueryInterruptTime();
            while (pSqCmdStatus[entryIndex] == TRUE)
            {
                // Need to busy poll as we dont have the space to write cmd in SQ.
                // In case timeout, due to some error lets fail this call. (Just added as safety fallback and not tested)
                DeviceContext->CompletionQueue->CheckForCompletion(FALSE);

                ULONGLONG waitedDuration = KeQueryInterruptTime() - startWaitTickCount;
                if (pSqCmdStatus[entryIndex] == TRUE && waitedDuration >= (ULONGLONG)WDF_ABS_TIMEOUT_IN_SEC(1))
                {
                    ULONG totalPendingCount = GetPendingCommandCount();
                    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE,
                        "Failed at SQ[%d] index = 0x%x is marked busy (Duration %lld ms) and could not free it,"
                        " readOffset 0x%llx, total pending cmd in all the SQs = %d,  submitCounter = 0x%x, waitCallCounter= 0x%x\n",
                        GetPriority(), entryIndex, waitedDuration / WDF_ABS_TIMEOUT_IN_MS(1), readLwrrentOffset, totalPendingCount, submitCounter, waitCallCounter);

                    status = STATUS_UNEXPECTED_IO_ERROR;
                    logSystemEvent(status,
                        "%s()::%d Failed at SQ[%d] index = 0x%x is marked busy (Duration %lld ms) and could not free it,"
                        " readOffset 0x%llx, total pending cmd in all the SQs = %d,  submitCounter = 0x%x, waitCallCounter= 0x%x\n",
                        __FUNCDNAME__, __LINE__,
                        GetPriority(), entryIndex, waitedDuration / WDF_ABS_TIMEOUT_IN_MS(1), readLwrrentOffset, totalPendingCount, submitCounter, waitCallCounter);

                    KeReleaseSpinLock(&m_SubmissionLock, oldIrql);

                    goto CleanUpAndExit;
                }

                waitCallCounter++;
            }
        }

        ASSERT(pSqCmdStatus[entryIndex] == FALSE);

        USHORT lwrrentCID = IDX_TO_CID(entryIndex);

        pSqCmdStatus[entryIndex] = TRUE;
        m_SqSubmitPos = (entryIndex + 1) % m_Info.QueueDepth;
        TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "-- READ offset 0x%llx, lwrrentCID = 0x%x\n", readLwrrentOffset, lwrrentCID);

        volatile PLWME_COMMAND c = &commands[entryIndex];
        PVOID pQPRP = (PLWME_PRP_ENTRY)m_prpVa;
        ULONG prpOffset = (LW_MAX_PRP_PER_COMMAND * entryIndex);
        PLWME_PRP_ENTRY pPRP2 = (PLWME_PRP_ENTRY)((PUCHAR)pQPRP + prpOffset * sizeof(LWME_PRP_ENTRY));
        TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "prpOffset start = 0x%x, upto 0x%x, start VA 0x%llx\n", prpOffset, prpOffset + LW_MAX_PRP_PER_COMMAND, pPRP2);


        ASSERT((readLwrrentOffset + readCmd->inFileOffestInBytes) % LW_SECTOR_SIZE == 0);
        ULONGLONG lwrrentLBA = ( (readLwrrentOffset + readCmd->inFileOffestInBytes) / LW_SECTOR_SIZE) + readCmd->inLBA;
        PUCHAR readAtVA = (PUCHAR)userSpaceAllocKernelVA + readCmd->addr.inAllocOffset + readLwrrentOffset;
        DoCreateReadCommand(c, readAtVA, cmdReadSize, lwrrentCID, pPRP2, LW_MAX_PRP_PER_COMMAND, lwrrentLBA);
        
        submitCounter++;

        readCmd->outSubmissionRefCount = InterlockedIncrement64(&m_SubmissionRefCount);
        ASSERT(readCmd->outSubmissionRefCount > 0);

        entryIndex = (entryIndex + 1) % m_Info.QueueDepth;

        TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "SQ sqDoorbell at = 0x%x.\n", entryIndex);

        readLwrrentOffset += cmdReadSize;

        //if (entryIndex % 10 || readLwrrentOffset >= readCmd->inReadSize)
        {
            KeMemoryBarrier();

            volatile ULONG *sqDoorbell = (volatile ULONG *)(ULONG_PTR)m_Info.DoorbellRegisterAddress;
            WRITE_REGISTER_ULONG(sqDoorbell, entryIndex);
        }
    }

    KeReleaseSpinLock(&m_SubmissionLock, oldIrql);
    ASSERT(readCmd->outSubmissionRefCount > 0);

CleanUpAndExit:

    if (mdl)
    {
        MmUnlockPages(mdl);
        IoFreeMdl(mdl);
        mdl = NULL;
    }

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "SQ[%d] Total Submitted commands %d,  polling fun called %d times. \n \n >> Pending entries 0x%x,  Completion ref 0x%llx, Submission ref: 0x%llx\n",
        GetPriority(),
        submitCounter,
        waitCallCounter,
         GetPendingCommandCount(), m_CompletionRefCount, m_SubmissionRefCount);

    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_QUEUE, "SQ [%d], SQ depth %d, STATUS 0x%x\n",
        GetPriority(), m_Info.QueueDepth, status);

Exit:
    return status;
}

CODE_SEGMENT(PAGE_CODE)
void LWMeSQueue::operator delete(void* ptr)
{
    CHECK_IRQL(PASSIVE_LEVEL);
    ExFreePool(ptr);
}

CODE_SEGMENT(PAGE_CODE)
void* LWMeSQueue::operator new(size_t size)
{
    CHECK_IRQL(PASSIVE_LEVEL);
    return lwAllocatePoolWithTag(NonPagedPool, size, LW_LWME_TAG);
}

POP_SEGMENTS