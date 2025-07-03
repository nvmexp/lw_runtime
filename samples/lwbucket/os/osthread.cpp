 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: osthread.cpp                                                      *|
|*                                                                            *|
 \****************************************************************************/
#include "osprecomp.h"

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
// Locals
//
//******************************************************************************
// Kernel Thread Type Helpers
CMemberType     CKThread::m_KThreadType                     (&osKernel(), "KTHREAD", "_KTHREAD");

// Kernel Thread Field Helpers
CMemberField    CKThread::m_headerField                     (&KThreadType(), false, NULL, "Header");
CMemberField    CKThread::m_initialStackField               (&KThreadType(), false, NULL, "InitialStack");
CMemberField    CKThread::m_stackLimitField                 (&KThreadType(), false, NULL, "StackLimit");
CMemberField    CKThread::m_stackBaseField                  (&KThreadType(), false, NULL, "StackBase");
CMemberField    CKThread::m_kernelStackField                (&KThreadType(), false, NULL, "KernelStack");
CMemberField    CKThread::m_runningField                    (&KThreadType(), false, NULL, "Running");
CMemberField    CKThread::m_alertedField                    (&KThreadType(), false, NULL, "Alerted");
CMemberField    CKThread::m_priorityField                   (&KThreadType(), false, NULL, "Priority");
CMemberField    CKThread::m_contextSwitchesField            (&KThreadType(), false, NULL, "ContextSwitches");
CMemberField    CKThread::m_stateField                      (&KThreadType(), false, NULL, "State");
CMemberField    CKThread::m_nextProcessorField              (&KThreadType(), false, NULL, "NextProcessor");
CMemberField    CKThread::m_idealProcessorField             (&KThreadType(), false, NULL, "IdealProcessor");
CMemberField    CKThread::m_processField                    (&KThreadType(), false, NULL, "Process");
CMemberField    CKThread::m_kernelTimeField                 (&KThreadType(), false, NULL, "KernelTime");
CMemberField    CKThread::m_userTimeField                   (&KThreadType(), false, NULL, "UserTime");
CMemberField    CKThread::m_threadListEntryField            (&KThreadType(), false, NULL, "ThreadListEntry");

// Exelwtive Thread Type Helpers
CMemberType     CEThread::m_EThreadType                     (&osKernel(), "ETHREAD", "_ETHREAD");

// Exelwtive Thread Field Helpers
CMemberField    CEThread::m_tcbField                        (&EThreadType(), false, NULL, "Tcb");
CMemberField    CEThread::m_createTimeField                 (&EThreadType(), false, NULL, "CreateTime");
CMemberField    CEThread::m_exitTimeField                   (&EThreadType(), false, NULL, "ExitTime");
CMemberField    CEThread::m_exitStatusField                 (&EThreadType(), false, NULL, "ExitStatus");
CMemberField    CEThread::m_irpListField                    (&EThreadType(), false, NULL, "IrpList");
CMemberField    CEThread::m_threadListEntryField            (&EThreadType(), false, NULL, "ThreadListEntry");

// CKThread object tracking
CKThreadList    CKThread::m_KThreadList;

// CEThread object tracking
CEThreadList    CEThread::m_EThreadList;

//******************************************************************************

CKThread::CKThread
(
    CKThreadList       *pKThreadList,
    THREAD              ptrKThread
)
:   LWnqObj(pKThreadList, ptrKThread),
    m_ptrKThread(ptrKThread),
    INIT(initialStack),
    INIT(stackLimit),
    INIT(stackBase),
    INIT(kernelStack),
    INIT(running),
    INIT(alerted),
    INIT(priority),
    INIT(contextSwitches),
    INIT(state),
    INIT(nextProcessor),
    INIT(idealProcessor),
    INIT(process),
    INIT(kernelTime),
    INIT(userTime),
    m_pDispatcherHeader(NULL)
{
    assert(pKThreadList != NULL);

    // Get the kernel thread information
    READ(initialStack,    ptrKThread);
    READ(stackLimit,      ptrKThread);
    READ(stackBase,       ptrKThread);
    READ(kernelStack,     ptrKThread);
    READ(running,         ptrKThread);
    READ(alerted,         ptrKThread);
    READ(priority,        ptrKThread);
    READ(contextSwitches, ptrKThread);
    READ(state,           ptrKThread);
    READ(nextProcessor,   ptrKThread);
    READ(idealProcessor,  ptrKThread);
    READ(process,         ptrKThread);
    READ(kernelTime,      ptrKThread);
    READ(userTime,        ptrKThread);

} // CKThread

//******************************************************************************

CKThread::~CKThread()
{

} // ~CKThread

//******************************************************************************

CKThreadPtr
CKThread::createKThread
(
    THREAD              ptrKThread
)
{
    CKThreadPtr         pKThread;

    // Check for valid kernel thread address given
    if (isKernelModeAddress(ptrKThread))
    {
        // Check to see if this kernel thread already exists
        pKThread = findObject(KThreadList(), ptrKThread);
        if (pKThread == NULL)
        {
            // Try to create the new kernel thread object
            pKThread = new CKThread(KThreadList(), ptrKThread);
        }
    }
    return pKThread;

} // createKThread

//******************************************************************************

const CDispatcherHeaderPtr
CKThread::dispatcherHeader() const
{
    POINTER             ptrDispatcherHeader;

    // Check for dispatcher header already created
    if (m_pDispatcherHeader == NULL)
    {
        // Check to see if dispatcher header is present
        if (headerField().isPresent())
        {
            // Compute the dispatcher header address
            ptrDispatcherHeader = ptrKThread() + headerField().offset();

            // Try to create the kernel thread dispatcher header
            m_pDispatcherHeader = createDispatcherHeader(ptrDispatcherHeader);
        }
    }
    return m_pDispatcherHeader;

} // dispatcherHeader

//******************************************************************************

CEThreadPtr
CKThread::EThread() const
{
    CEThreadPtr         pEThread;

    // Try to create the exelwtive thread (At the same address)
    pEThread = createEThread(ptrKThread());

    return pEThread;

} // EThread

//******************************************************************************

CKProcessPtr
CKThread::KProcess() const
{
    PROCESS             ptrProcess;
    CKProcessPtr        pKProcess;

    // Check for thread process present
    if (processMember().isPresent())
    {
        // Get the thread process address
        ptrProcess = process();
        if (ptrProcess != NULL)
        {
            // Try to create the thread process
            pKProcess = createKProcess(process());
        }
    }
    return pKProcess;

} // KProcess

//******************************************************************************

CEProcessPtr
CKThread::EProcess() const
{
    PROCESS             ptrProcess;
    CEProcessPtr        pEProcess;

    // Check for thread process present
    if (processMember().isPresent())
    {
        // Get the thread process address
        ptrProcess = process();
        if (ptrProcess != NULL)
        {
            // Try to create the thread process
            pEProcess = createEProcess(process());
        }
    }
    return pEProcess;

} // EProcess

//******************************************************************************

CKThreads::CKThreads
(
    const CListEntry&   KList
)
:   m_ulKThreadCount(0),
    m_ptrKThreads(NULL),
    m_aKThreads(NULL)
{
    PROCESS             ptrProcess;
    THREAD              ptrThread;
    ULONG               ulThread = 0;

    // Check for a process list (no list head) or thread list
    if (KList.listField() == NULL)
    {
        // Count the number of threads for all processes (Process list)
        ptrProcess = KList.ptrHeadEntry();
        while (ptrProcess != NULL)
        {
            POINTER         ptrThreadList = ptrProcess + CKProcess::threadListHeadField().offset();
            CListEntry      KThreadList(&CKThread::threadListEntryField(), ptrThreadList);

            // Count the number of threads for this process
            ptrThread = KThreadList.ptrHeadEntry();
            while (ptrThread != NULL)
            {
                // Increment thread count and move to next thread
                m_ulKThreadCount++;
                ptrThread = KThreadList.ptrNextEntry();
            }
            // Move to next process
            ptrProcess = KList.ptrNextEntry();
        }
        // Check for threads present
        if (m_ulKThreadCount != 0)
        {
            // Allocate the thread base addresses and thread array
            m_ptrKThreads = new THREAD[m_ulKThreadCount];
            m_aKThreads   = new CKThreadPtr[m_ulKThreadCount];

            // Loop filling in the thread base addresses
            ptrProcess = KList.ptrHeadEntry();
            while (ptrProcess != NULL)
            {
                POINTER         ptrThreadList = ptrProcess + CKProcess::threadListHeadField().offset();
                CListEntry      KThreadList(&CKThread::threadListEntryField(), ptrThreadList);

                // Loop filling in thread base addresses
                ptrThread = KThreadList.ptrHeadEntry();
                while (ptrThread != NULL)
                {
                    // Fill in thread base address (Only if enough room)
                    if (ulThread < m_ulKThreadCount)
                    {
                        m_ptrKThreads[ulThread] = ptrThread;
                    }
                    // Increment thread count and move to next thread
                    ulThread++;
                    ptrThread = KThreadList.ptrNextEntry();
                }
                // Move to next process
                ptrProcess = KList.ptrNextEntry();
            }
        }
    }
    else    // Not a process list (Thread list)
    {
        // Count the number of threads for this process (Thread list)
        ptrThread = KList.ptrHeadEntry();
        while (ptrThread != NULL)
        {
            // Increment thread count and move to next thread
            m_ulKThreadCount++;
            ptrThread = KList.ptrNextEntry();
        }
        // Check for threads present
        if (m_ulKThreadCount != 0)
        {
            // Allocate the thread base addresses and thread array
            m_ptrKThreads = new THREAD[m_ulKThreadCount];
            m_aKThreads   = new CKThreadPtr[m_ulKThreadCount];

            // Loop filling in the thread base addresses
            ptrThread = KList.ptrHeadEntry();
            while (ptrThread != NULL)
            {
                // Fill in thread base address (Only if enough room)
                if (ulThread < m_ulKThreadCount)
                {
                    m_ptrKThreads[ulThread] = ptrThread;
                }
                // Increment thread count and move to next thread
                ulThread++;
                ptrThread = KList.ptrNextEntry();
            }
        }
    }
    // Make sure the thread count is correct
    if (ulThread != m_ulKThreadCount)
    {
        throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                         ": Kernel thread count and list don't agree (%d != %d)",
                         ulThread, m_ulKThreadCount);
    }

} // CKThreads

//******************************************************************************

CKThreads::CKThreads
(
    const CEThreads    *pEThreads
)
:   m_ulKThreadCount((pEThreads != NULL) ? pEThreads->EThreadCount() : 0),
    m_ptrKThreads(NULL),
    m_aKThreads(NULL)
{
    THREAD              ptrThread;
    ULONG               ulThread;

    assert(pEThreads != NULL);

    // Check for threads present
    if (m_ulKThreadCount != 0)
    {
        // Allocate the thread base addresses and thread array
        m_ptrKThreads = new THREAD[m_ulKThreadCount];
        m_aKThreads   = new CKThreadPtr[m_ulKThreadCount];

        // Loop filling in the thread base addresses
        for (ulThread = 0; ulThread < m_ulKThreadCount; ulThread++)
        {
            // Fill in the next thread base address
            m_ptrKThreads[ulThread] = pEThreads->ptrEThread(ulThread);
        }
    }

} // CKThreads

//******************************************************************************

CKThreads::~CKThreads()
{

} // ~CKThreads

//******************************************************************************

const CKThreadPtr
CKThreads::KThread
(
    ULONG               ulKThread
) const
{
    // Check for invalid thread index
    if (ulKThread >= KThreadCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid kernel thread index %d (>= %d)",
                         ulKThread, KThreadCount());
    }
    // Check to see if thread needs to be loaded
    if (m_aKThreads[ulKThread] == NULL)
    {
        // Check for non-zero thread address
        if (m_ptrKThreads[ulKThread] != NULL)
        {
            // Try to create the requested kernel thread
            m_aKThreads[ulKThread] = createKThread(m_ptrKThreads[ulKThread]);
        }
    }
    return m_aKThreads[ulKThread];

} // KThread

//******************************************************************************

THREAD
CKThreads::ptrKThread
(
    ULONG               ulKThread
) const
{
    // Check for invalid thread index
    if (ulKThread >= KThreadCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid kernel thread index %d (>= %d)",
                         ulKThread, KThreadCount());
    }
    // Return the requested thread address
    return m_ptrKThreads[ulKThread];

} // ptrKThread

//******************************************************************************

CKThreadPtr*
CKThreads::KThreads()
{
    ULONG               ulKThread;

    // Loop making sure all the threads are loaded
    for (ulKThread = 0; ulKThread < KThreadCount(); ulKThread++)
    {
        // Check to see if next thread needs to be loaded
        if (m_aKThreads[ulKThread] == NULL)
        {
            // Check for non-zero thread address
            if (m_ptrKThreads[ulKThread] != NULL)
            {
                // Try to create the next kernel thread
                m_aKThreads[ulKThread] = createKThread(m_ptrKThreads[ulKThread]);
            }
        }
    }
    return &m_aKThreads[0];

} // KThreads

//******************************************************************************

const CKThreadPtr
CKThreads::findKThread
(
    THREAD              ptrKThread
) const
{
    ULONG               ulKThread;
    CKThreadPtr         pKThread;

    // Loop searching for the requested thread
    for (ulKThread = 0; ulKThread < KThreadCount(); ulKThread++)
    {
        // Check to see if next thread is the requested thread
        if (CKThreads::ptrKThread(ulKThread) == ptrKThread)
        {
            // Found the requested kernel thread, get it and stop search
            pKThread = KThread(ulKThread);
            break;
        }
    }
    return pKThread;

} // findKThread

//******************************************************************************

CEThread::CEThread
(
    CEThreadList       *pEThreadList,
    THREAD              ptrEThread
)
:   LWnqObj(pEThreadList, ptrEThread),
    m_ptrEThread(ptrEThread),
    INIT(createTime),
    INIT(exitTime),
    INIT(exitStatus),
    m_pKThread(NULL)
{
    assert(pEThreadList != NULL);

    // Get the exelwtive thread information
    READ(createTime, ptrEThread);
    READ(exitTime,   ptrEThread);
    READ(exitStatus, ptrEThread);

} // CEThread

//******************************************************************************

CEThread::~CEThread()
{

} // ~CEThread

//******************************************************************************

CEThreadPtr
CEThread::createEThread
(
    THREAD              ptrEThread
)
{
    CEThreadPtr         pEThread;

    // Check for valid exelwtive thread address given
    if (isKernelModeAddress(ptrEThread))
    {
        // Check to see if this exelwtive thread already exists
        pEThread = findObject(EThreadList(), ptrEThread);
        if (pEThread == NULL)
        {
            // Try to create the new exelwtive thread object
            pEThread = new CEThread(EThreadList(), ptrEThread);
        }
    }
    return pEThread;

} // createEThread

//******************************************************************************

const CKThreadPtr
CEThread::KThread() const
{
    THREAD              ptrKThread;

    // Check for kernel thread already created
    if (m_pKThread == NULL)
    {
        // Check to see if kernel thread is present
        if (tcbField().isPresent())
        {
            // Compute the kernel thread address
            ptrKThread = ptrEThread() + tcbField().offset();

            // Try to create the exelwtive thread kernel thread
            m_pKThread = createKThread(ptrKThread);
        }
    }
    return m_pKThread;

} // KThread

//******************************************************************************

CKProcessPtr
CEThread::KProcess() const
{
    CKThreadPtr         pKThread = KThread();
    CKProcessPtr        pKProcess;

    // Check for kernel thread present
    if (pKThread != NULL)
    {
        // Try to get the exelwtive thread kernel process
        pKProcess = pKThread->KProcess();
    }
    return pKProcess;

} // KProcess

//******************************************************************************

CEProcessPtr
CEThread::EProcess() const
{
    CKThreadPtr         pKThread = KThread();
    CEProcessPtr        pEProcess;

    // Check for kernel thread present
    if (pKThread != NULL)
    {
        // Try to get the exelwtive thread exelwtive process
        pEProcess = pKThread->EProcess();
    }
    return pEProcess;

} // EProcess

//******************************************************************************

CEThreads::CEThreads
(
    const CListEntry&   EList
)
:   m_ulEThreadCount(0),
    m_ptrEThreads(NULL),
    m_aEThreads(NULL)
{
    PROCESS             ptrProcess;
    THREAD              ptrThread;
    ULONG               ulThread = 0;

    // Check for a process list (no list head) or thread list
    if (EList.listField() == NULL)
    {
        // Count the number of threads for all processes (Process list)
        ptrProcess = EList.ptrHeadEntry();
        while (ptrProcess != NULL)
        {
            POINTER         ptrThreadList = ptrProcess + CEProcess::threadListHeadField().offset();
            CListEntry      EThreadList(&CEThread::threadListEntryField(), ptrThreadList);

            // Count the number of threads for this process
            ptrThread = EThreadList.ptrHeadEntry();
            while (ptrThread != NULL)
            {
                // Increment thread count and move to next thread
                m_ulEThreadCount++;
                ptrThread = EThreadList.ptrNextEntry();
            }
            // Move to next process
            ptrProcess = EList.ptrNextEntry();
        }
        // Check for threads present
        if (m_ulEThreadCount != 0)
        {
            // Allocate the thread base addresses and thread array
            m_ptrEThreads = new THREAD[m_ulEThreadCount];
            m_aEThreads   = new CEThreadPtr[m_ulEThreadCount];

            // Loop filling in the thread base addresses
            ptrProcess = EList.ptrHeadEntry();
            while (ptrProcess != NULL)
            {
                POINTER         ptrThreadList = ptrProcess + CEProcess::threadListHeadField().offset();
                CListEntry      EThreadList(&CEThread::threadListEntryField(), ptrThreadList);

                // Loop filling in thread base addresses
                ptrThread = EThreadList.ptrHeadEntry();
                while (ptrThread != NULL)
                {
                    // Fill in thread base address (Only if enough room)
                    if (ulThread < m_ulEThreadCount)
                    {
                        m_ptrEThreads[ulThread] = ptrThread;
                    }
                    // Increment thread count and move to next thread
                    ulThread++;
                    ptrThread = EThreadList.ptrNextEntry();
                }
                // Move to next process
                ptrProcess = EList.ptrNextEntry();
            }
        }
    }
    else    // Not a process list (Thread list)
    {
        // Count the number of threads for this process (Thread list)
        ptrThread = EList.ptrHeadEntry();
        while (ptrThread != NULL)
        {
            // Increment thread count and move to next thread
            m_ulEThreadCount++;
            ptrThread = EList.ptrNextEntry();
        }
        // Check for threads present
        if (m_ulEThreadCount != 0)
        {
            // Allocate the thread base addresses and thread array
            m_ptrEThreads = new THREAD[m_ulEThreadCount];
            m_aEThreads   = new CEThreadPtr[m_ulEThreadCount];

            // Loop filling in the thread base addresses
            ptrThread = EList.ptrHeadEntry();
            while (ptrThread != NULL)
            {
                // Fill in thread base address (Only if enough room)
                if (ulThread < m_ulEThreadCount)
                {
                    m_ptrEThreads[ulThread] = ptrThread;
                }
                // Increment thread count and move to next thread
                ulThread++;
                ptrThread = EList.ptrNextEntry();
            }
        }
    }
    // Make sure the thread count is correct
    if (ulThread != m_ulEThreadCount)
    {
        throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                         ": Exelwtive thread count and list don't agree (%d != %d)",
                         ulThread, m_ulEThreadCount);
    }

} // CEThreads

//******************************************************************************

CEThreads::~CEThreads()
{

} // ~CEThreads

//******************************************************************************

const CEThreadPtr
CEThreads::EThread
(
    ULONG               ulEThread
) const
{
    // Check for invalid thread index
    if (ulEThread >= EThreadCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid exelwtive thread index %d (>= %d)",
                         ulEThread, EThreadCount());
    }
    // Check to see if thread needs to be loaded
    if (m_aEThreads[ulEThread] == NULL)
    {
        // Check for non-zero thread address
        if (m_ptrEThreads[ulEThread] != NULL)
        {
            // Try to create the requested thread
            m_aEThreads[ulEThread] = createEThread(m_ptrEThreads[ulEThread]);
        }
    }
    return m_aEThreads[ulEThread];

} // EThread

//******************************************************************************

THREAD
CEThreads::ptrEThread
(
    ULONG               ulEThread
) const
{
    // Check for invalid thread index
    if (ulEThread >= EThreadCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid exelwtive thread index %d (>= %d)",
                         ulEThread, EThreadCount());
    }
    // Return the requested thread address
    return m_ptrEThreads[ulEThread];

} // ptrEThread

//******************************************************************************

CEThreadPtr*
CEThreads::EThreads()
{
    ULONG               ulEThread;

    // Loop making sure all the threads are loaded
    for (ulEThread = 0; ulEThread < EThreadCount(); ulEThread++)
    {
        // Check to see if next thread needs to be loaded
        if (m_aEThreads[ulEThread] == NULL)
        {
            // Check for non-zero thread address
            if (m_ptrEThreads[ulEThread] != NULL)
            {
                // Try to create the next exelwtive thread
                m_aEThreads[ulEThread] = createEThread(m_ptrEThreads[ulEThread]);
            }
        }
    }
    return &m_aEThreads[0];

} // EThreads

//******************************************************************************

const CEThreadPtr
CEThreads::findEThread
(
    THREAD              ptrEThread
) const
{
    ULONG               ulEThread;
    CEThreadPtr         pEThread;

    // Loop searching for the requested thread
    for (ulEThread = 0; ulEThread < EThreadCount(); ulEThread++)
    {
        // Check to see if next thread is the requested thread
        if (CEThreads::ptrEThread(ulEThread) == ptrEThread)
        {
            // Found the requested exelwtive thread, get it and stop search
            pEThread = EThread(ulEThread);
            break;
        }
    }
    return pEThread;

} // findEThread

//******************************************************************************

const CKThreadsPtr
CEThreads::KThreads() const
{
    // Check for threads not already created
    if (m_pKThreads == NULL)
    {
        // Try to create the threads
        m_pKThreads = new CKThreads(this);
    }
    return m_pKThreads;

} // KThreads

//******************************************************************************

const CKThreadsPtr
getKThreads
(
    bool                bThrow
)
{
    CKThreadsPtr        pKThreads;

    // Check to see if we can get the kernel threads
    if (mmProcessList().isPresent() && CEProcess::mmProcessLinksField().isPresent() && CEProcess::threadListHeadField().isPresent() && CEThread::threadListEntryField().isPresent())
    {
        // Declare kernel process list (No list field to indicate process list)
        CListEntry  processList(&CEProcess::mmProcessLinksField(), mmProcessList().offset());

        // Create the kernel threads (All processes)
        try
        {
            pKThreads = new CKThreads(processList);
        }
        catch (CMemoryException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            if (bThrow)
            {
                throw;
            }
        }
    }
    else    // No kernel threads present
    {
        if (bThrow)
        {
            if (!mmProcessList().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 mmProcessList().name());
            }
            else if (!CEProcess::mmProcessLinksField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEProcess::mmProcessLinksField().name());
            }
            else if (!CEProcess::threadListHeadField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEProcess::threadListHeadField().name());
            }
            else if (!CEThread::threadListEntryField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEThread::threadListEntryField().name());
            }
        }
    }
    return pKThreads;

} // getKThreads

//******************************************************************************

const CKThreadsPtr
getKernelThreads
(
    bool                bThrow
)
{
    CKThreadsPtr        pKernelThreads;

    // Check to see if we can get the active kernel threads
    if (psActiveProcessHead().isPresent() && CEProcess::activeProcessLinksField().isPresent() && CEProcess::threadListHeadField().isPresent() && CEThread::threadListEntryField().isPresent())
    {
        // Declare active process list (No list field to indicate process list)
        CListEntry  processList(&CEProcess::activeProcessLinksField(), psActiveProcessHead().offset());

        // Create the active kernel threads (All active processes)
        try
        {
            pKernelThreads = new CKThreads(processList);
        }
        catch (CMemoryException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            if (bThrow)
            {
                throw;
            }
        }
    }
    else    // No active kernel threads present
    {
        if (bThrow)
        {
            if (!psActiveProcessHead().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 psActiveProcessHead().name());
            }
            else if (!CEProcess::activeProcessLinksField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEProcess::activeProcessLinksField().name());
            }
            else if (!CEProcess::threadListHeadField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEProcess::threadListHeadField().name());
            }
            else if (!CEThread::threadListEntryField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEThread::threadListEntryField().name());
            }
        }
    }
    return pKernelThreads;

} // getKernelThreads

//******************************************************************************

const CEThreadsPtr
getEThreads
(
    bool                bThrow
)
{
    CEThreadsPtr        pEThreads;

    // Check to see if we can get the exelwtive threads
    if (mmProcessList().isPresent() && CEProcess::mmProcessLinksField().isPresent() && CEProcess::threadListHeadField().isPresent() && CEThread::threadListEntryField().isPresent())
    {
        // Declare exelwtive process list (No list field to indicate process list)
        CListEntry  processList(&CEProcess::mmProcessLinksField(), mmProcessList().offset());

        // Create the exelwtive threads (All processes)
        try
        {
            pEThreads = new CEThreads(processList);
        }
        catch (CMemoryException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            if (bThrow)
            {
                throw;
            }
        }
    }
    else    // No exelwtive threads present
    {
        if (bThrow)
        {
            if (!mmProcessList().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 mmProcessList().name());
            }
            else if (!CEProcess::mmProcessLinksField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEProcess::mmProcessLinksField().name());
            }
            else if (!CEProcess::threadListHeadField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEProcess::threadListHeadField().name());
            }
            else if (!CEThread::threadListEntryField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEThread::threadListEntryField().name());
            }
        }
    }
    return pEThreads;

} // getEThreads

//******************************************************************************

const CEThreadsPtr
getExelwtiveThreads
(
    bool                bThrow
)
{
    CEThreadsPtr        pExelwtiveThreads;

    // Check to see if we can get the active exelwtive threads
    if (psActiveProcessHead().isPresent() && CEProcess::activeProcessLinksField().isPresent() && CEProcess::threadListHeadField().isPresent() && CEThread::threadListEntryField().isPresent())
    {
        // Declare active process list (No list field to indicate process list)
        CListEntry  processList(&CEProcess::activeProcessLinksField(), psActiveProcessHead().offset());

        // Create the active exelwtive threads (All active processes)
        try
        {
            pExelwtiveThreads = new CEThreads(processList);
        }
        catch (CMemoryException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            if (bThrow)
            {
                throw;
            }
        }
    }
    else    // No active exelwtive threads present
    {
        if (bThrow)
        {
            if (!psActiveProcessHead().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 psActiveProcessHead().name());
            }
            else if (!CEProcess::activeProcessLinksField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEProcess::activeProcessLinksField().name());
            }
            else if (!CEProcess::threadListHeadField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEProcess::threadListHeadField().name());
            }
            else if (!CEThread::threadListEntryField().isPresent())
            {
                throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                 ": Couldn't get offset of %s",
                                 CEThread::threadListEntryField().name());
            }
        }
    }
    return pExelwtiveThreads;

} // getExelwtiveThreads

//******************************************************************************

const CKThreadPtr
findKThread
(
    THREAD              ptrKThread
)
{
    CKThreadsPtr        pKThreads;
    CKThreadPtr         pKThread;

    // Get the kernel threads to search for the given thread
    pKThreads = getKThreads();
    if (pKThreads != NULL)
    {
        // Try to find the requested kernel thread
        pKThread = pKThreads->findKThread(ptrKThread);
        if (pKThread == NULL)
        {
            // Get active kernel threads to search for the given thread
            pKThreads = getKernelThreads();
            if (pKThreads != NULL)
            {
                // Try to find the requested kernel thread
                pKThread = pKThreads->findKThread(ptrKThread);
            }
        }
    }
    else    // Can't get kernel threads, try active kernel threads
    {
        // Get active kernel threads to search for the given thread
        pKThreads = getKernelThreads();
        if (pKThreads != NULL)
        {
            // Try to find the requested kernel thread
            pKThread = pKThreads->findKThread(ptrKThread);
        }
    }
    return pKThread;

} // findKThread

//******************************************************************************

const CEThreadPtr
findEThread
(
    THREAD              ptrEThread
)
{
    CEThreadsPtr        pEThreads;
    CEThreadPtr         pEThread;

    // Get the exelwtive threads to search for the given thread
    pEThreads = getEThreads();
    if (pEThreads != NULL)
    {
        // Try to find the requested exelwtive thread
        pEThread = pEThreads->findEThread(ptrEThread);
        if (pEThread == NULL)
        {
            // Get active exelwtive threads to search for the given thread
            pEThreads = getExelwtiveThreads();
            if (pEThreads != NULL)
            {
                // Try to find the requested exelwtive thread
                pEThread = pEThreads->findEThread(ptrEThread);
            }
        }
    }
    else    // Can't get exelwtive threads, try active exelwtive threads
    {
        // Get active exelwtive threads to search for the given thread
        pEThreads = getExelwtiveThreads();
        if (pEThreads != NULL)
        {
            // Try to find the requested exelwtive thread
            pEThread = pEThreads->findEThread(ptrEThread);
        }
    }
    return pEThread;

} // findEThread

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
