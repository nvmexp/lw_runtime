/*++

Copyright (C) LWPU Corporation, 2020

Module Name:

    CQueue.h

Abstract:

    This file implements lwmetf.sys --
    kernel mode filter driver for using LWMe queue reservation code in storlwme.sys for LWAPI direct storage implementation.

--*/

#pragma once

#include "driver.h"
#include "SQueue.h"

class LWMeCQueue
{
public:
    LWMeCQueue();
    ~LWMeCQueue();

    USHORT  GetQueueID() { return m_Info.QueueID; }

    LWME_RESERVED_CQ_INFO GetReserveCQueueInfo() { return m_Info; }

    ULONG64 GetPhysicalAddress() { return m_CqPas; }

    VOID    SetSubmissionQueue(LWMeSQueue* SQueue, ULONG index) {
        ASSERT(index < LW_MAX_SQS);
        m_SubmissionQueues[index] = SQueue;
    }

#ifdef DBG
    BOOLEAN IsValid() { return m_IsCreated && m_IsAllocated; }
    BOOLEAN IsAllocated() { return m_IsAllocated; }
#endif 

    NTSTATUS CheckForCompletion(BOOL isCalledFromDPC);

    BOOLEAN DoAlloc(USHORT QueueDepth);
    BOOLEAN DoFreeMemory();

    BOOLEAN DoCreate(PLWME_RESERVED_CQ_INFO pSQInfo);
    VOID    DoDestroy();

    void* operator new(size_t size);
    void operator delete(void*);

private:

    LWME_RESERVED_CQ_INFO m_Info;

    LWMeSQueue*           m_SubmissionQueues[LW_MAX_SQS];

    // Clearing after create causes crash in storlwme driver.
    volatile PLWME_COMPLETION_ENTRY m_CqVas;    // CQ virtual addresses
    ULONG64               m_CqPas;              // CQ physical address

    USHORT                m_CqHeadPointer;

    // TODO: can be moved to bit per entry
    PCHAR                 m_pPhaseTag;

    BOOLEAN               m_IsAllocated;
    BOOLEAN               m_IsCreated;

    KSPIN_LOCK            m_ProcessComplectionLock;
    volatile LONG         m_pollingEntryCounter;
};