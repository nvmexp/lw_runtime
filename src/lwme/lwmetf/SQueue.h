/*++

Copyright (C) LWPU Corporation, 2020

Module Name:

    SQueue.h

Abstract:

    This file implements lwmetf.sys --
    kernel mode filter driver for using LWMe queue reservation code in storlwme.sys for LWAPI direct storage implementation.

--*/

#pragma once

#include "driver.h"
#include <limits.h>

#define IDX_TO_CID(IDX)             (IDX + (m_Info.QueueDepth * m_Info.QueuePriority))
#define CID_TO_IDX(CID)             ((CID % m_Info.QueueDepth))
#define CID_TO_QUEUE_PRIORITY(CID)  (CID / m_Info.QueueDepth)

class LWMeSQueue
{
    //
    // Structure to store notifications events in a protected list
    //
    typedef struct _NotificationListEntry
    {
        LIST_ENTRY  ListEntry;
        PKEVENT     kKernelEvent;
        ULONGLONG   RefCount;
       
    } NotificationListEntry;

public:
    LWMeSQueue(PDEVICE_CONTEXT pDeviceContext);
    ~LWMeSQueue();

    BOOLEAN IsEmpty() { return (m_SubmissionRefCount == m_CompletionRefCount) ? true : false; }
    USHORT  GetPriority() { return m_Info.QueuePriority; }
    ULONG   GetPendingCommandCount();

    USHORT  GetQueueID() { return m_Info.QueueID; }
    USHORT  GetCompletionQueueID() { return m_Info.CompletionQueueID; }
    LWME_RESERVED_SQ_INFO GetReserveSQueueInfo() { return m_Info; }

    ULONG64 GetPhysicalAddress() { return m_SqPas; }

    NTSTATUS RegisterNotification(_In_ HANDLE hUserEvent, _In_ ULONGLONG refCount);
    NTSTATUS ProcessNotifications();

    BOOLEAN HasPendingNotification() { return !IsListEmpty(&m_NotificationList); }

#ifdef DBG
    BOOLEAN IsValid() { return m_IsCreated && m_IsAllocated; }
    BOOLEAN IsAllocated() { return m_IsAllocated; }
#endif

    volatile ULONGLONG GetSubmissionCounter() { return m_SubmissionRefCount;  }
    volatile ULONGLONG GetCompletionCounter() { return m_CompletionRefCount; }
    PULONGLONG GetCompletionCounterPointer() { return (PULONGLONG)&m_CompletionRefCount; }

    void* operator new(size_t size);
    void operator delete(void*);

    VOID DoMarkCompletion(USHORT CID);

    BOOLEAN DoAlloc(USHORT QueueDepth);
    BOOLEAN DoFreeMemory();

    BOOLEAN DoCreate(PLWME_RESERVED_SQ_INFO pSQInfo);
    VOID    DoDestroy();

    NTSTATUS DoSubmitRead(PLWME_READ readCmd);

private:

    inline VOID DoCreateReadCommand(
        _In_ volatile PLWME_COMMAND  c,
        _In_ PUCHAR                  DataVA,
        _In_ ULONG                   DataWriteSize,
        _In_ UINT16                  CID,
        _In_ PLWME_PRP_ENTRY         pPRP,
        _In_ ULONG                   PRPAvailablecount,
        _In_ ULONGLONG               LBA);

    VOID UpdateCompletionRefCount();

    LWME_RESERVED_SQ_INFO m_Info;
    volatile LONGLONG     m_SubmissionRefCount;
    volatile LONGLONG     m_CompletionRefCount;
    
    USHORT                m_SqSubmitPos;

    // Clearing after create causes crash in storlwme driver.
    PVOID                 m_SqVas;    // SQ virtual addresses
    ULONG64               m_SqPas;    // SQ physical address

    PVOID                 m_prpVa;    // Allocation for PRP entries 
    // TODO: can be moved to bit per entry
    PBOOLEAN              m_pEntryStatus;

    BOOLEAN               m_IsAllocated;
    BOOLEAN               m_IsCreated;

    KSPIN_LOCK            m_UpdateCompletionRefLock;
    KSPIN_LOCK            m_SubmissionLock;
    KSPIN_LOCK            m_NotificationListLock;

    LIST_ENTRY            m_NotificationList;
    volatile LONG         m_notificationProcessEntryCounter;
    PDEVICE_CONTEXT       DeviceContext;
};