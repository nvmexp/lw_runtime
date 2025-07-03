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
|*  Module: osprocess.cpp                                                     *|
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
// Kernel Process Type Helpers
CMemberType     CKProcess::m_KProcessType                   (&osKernel(), "KPROCESS", "_KPROCESS");

// Kernel Process Field Helpers
CMemberField    CKProcess::m_headerField                    (&KProcessType(), false, NULL, "Header");
CMemberField    CKProcess::m_threadListHeadField            (&KProcessType(), false, NULL, "ThreadListHead");
CMemberField    CKProcess::m_processLockField               (&KProcessType(), false, NULL, "ProcessLock");
CMemberField    CKProcess::m_basePriorityField              (&KProcessType(), false, NULL, "BasePriority");
CMemberField    CKProcess::m_quantumResetField              (&KProcessType(), false, NULL, "QuantumReset");
CMemberField    CKProcess::m_processListEntryField          (&KProcessType(), false, NULL, "ProcessListEntry");
CMemberField    CKProcess::m_cycleTimeField                 (&KProcessType(), false, NULL, "CycleTime");
CMemberField    CKProcess::m_kernelTimeField                (&KProcessType(), false, NULL, "KernelTime");
CMemberField    CKProcess::m_userTimeField                  (&KProcessType(), false, NULL, "UserTime");

// Exelwtive Process Type Helpers
CMemberType     CEProcess::m_EProcessType                   (&osKernel(), "EPROCESS", "_EPROCESS");

// Exelwtive Process Field Helpers
CMemberField    CEProcess::m_pcbField                       (&EProcessType(), false, NULL, "Pcb");
CMemberField    CEProcess::m_createTimeField                (&EProcessType(), false, NULL, "CreateTime");
CMemberField    CEProcess::m_exitTimeField                  (&EProcessType(), false, NULL, "ExitTime");
CMemberField    CEProcess::m_commitChargeField              (&EProcessType(), false, NULL, "CommitCharge");
CMemberField    CEProcess::m_uniqueProcessIdField           (&EProcessType(), false, NULL, "UniqueProcessId");
CMemberField    CEProcess::m_activeProcessLinksField        (&EProcessType(), false, NULL, "ActiveProcessLinks");
CMemberField    CEProcess::m_peakVirtualSizeField           (&EProcessType(), false, NULL, "PeakVirtualSize");
CMemberField    CEProcess::m_virtualSizeField               (&EProcessType(), false, NULL, "VirtualSize");
CMemberField    CEProcess::m_sessionProcessLinksField       (&EProcessType(), false, NULL, "SessionProcessLinks");
CMemberField    CEProcess::m_sessionField                   (&EProcessType(), false, NULL, "Session");
CMemberField    CEProcess::m_imageFileNameField             (&EProcessType(), false, NULL, "ImageFileName");
CMemberField    CEProcess::m_threadListHeadField            (&EProcessType(), false, NULL, "ThreadListHead");
CMemberField    CEProcess::m_activeThreadsField             (&EProcessType(), false, NULL, "ActiveThreads");
CMemberField    CEProcess::m_lastThreadExitStatusField      (&EProcessType(), false, NULL, "LastThreadExitStatus");
CMemberField    CEProcess::m_pebField                       (&EProcessType(), false, NULL, "Peb");
CMemberField    CEProcess::m_readOperationCountField        (&EProcessType(), false, NULL, "ReadOperationCount");
CMemberField    CEProcess::m_writeOperationCountField       (&EProcessType(), false, NULL, "WriteOperationCount");
CMemberField    CEProcess::m_otherOperationCountField       (&EProcessType(), false, NULL, "OtherOperationCount");
CMemberField    CEProcess::m_readTransferCountField         (&EProcessType(), false, NULL, "ReadTransferCount");
CMemberField    CEProcess::m_writeTransferCountField        (&EProcessType(), false, NULL, "WriteTransferCount");
CMemberField    CEProcess::m_otherTransferCountField        (&EProcessType(), false, NULL, "OtherTransferCount");
CMemberField    CEProcess::m_commitChargeLimitField         (&EProcessType(), false, NULL, "CommitChargeLimit");
CMemberField    CEProcess::m_commitChargePeakField          (&EProcessType(), false, NULL, "CommitChargePeak");
CMemberField    CEProcess::m_mmProcessLinksField            (&EProcessType(), false, NULL, "MmProcessLinks");

// CKProcess object tracking
CKProcessList   CKProcess::m_KProcessList;

// CEProcess object tracking
CEProcessList   CEProcess::m_EProcessList;

//******************************************************************************

CKProcess::CKProcess
(
    CKProcessList      *pKProcessList,
    PROCESS             ptrKProcess
)
:   LWnqObj(pKProcessList, ptrKProcess),
    m_ptrKProcess(ptrKProcess),
    INIT(processLock),
    INIT(basePriority),
    INIT(quantumReset),
    INIT(cycleTime),
    INIT(kernelTime),
    INIT(userTime),
    m_ThreadList(&m_threadListHeadField, &CKThread::threadListEntryField(), ptrKProcess + (m_threadListHeadField.isPresent() ? m_threadListHeadField.offset() : 0)),
    m_pDispatcherHeader(NULL),
    m_pKThreads(NULL)
{
    assert(pKProcessList != NULL);

    // Get the kernel process information
    READ(processLock,  ptrKProcess);
    READ(basePriority, ptrKProcess);
    READ(quantumReset, ptrKProcess);
    READ(cycleTime,    ptrKProcess);
    READ(kernelTime,   ptrKProcess);
    READ(userTime,     ptrKProcess);

} // CKProcess

//******************************************************************************

CKProcess::~CKProcess()
{

} // ~CKProcess

//******************************************************************************

CKProcessPtr
CKProcess::createKProcess
(
    PROCESS             ptrKProcess
)
{
    CKProcessPtr        pKProcess;

    // Check for valid kernel process address given
    if (isKernelModeAddress(ptrKProcess))
    {
        // Check to see if this kernel process already exists
        pKProcess = findObject(KProcessList(), ptrKProcess);
        if (pKProcess == NULL)
        {
            // Try to create the new kernel process object
            pKProcess = new CKProcess(KProcessList(), ptrKProcess);
        }
    }
    return pKProcess;

} // createKProcess

//******************************************************************************

const CDispatcherHeaderPtr
CKProcess::dispatcherHeader() const
{
    POINTER             ptrDispatcherHeader;

    // Check for dispatcher header already created
    if (m_pDispatcherHeader == NULL)
    {
        // Check to see if dispatcher header is present
        if (headerField().isPresent())
        {
            // Compute the dispatcher header address
            ptrDispatcherHeader = ptrKProcess() + headerField().offset();

            // Try to create the kernel process dispatcher header
            m_pDispatcherHeader = createDispatcherHeader(ptrDispatcherHeader);
        }
    }
    return m_pDispatcherHeader;

} // dispatcherHeader

//******************************************************************************

const CKThreadsPtr
CKProcess::KThreads() const
{
    // Check for kernel threads already created
    if (m_pKThreads == NULL)
    {
        // Check to see if kernel threads are present
        if (threadList().isPresent())
        {
            // Try to create the kernel process threads
            m_pKThreads = new CKThreads(threadList());
        }
    }
    return m_pKThreads;

} // KThreads

//******************************************************************************

CEProcessPtr
CKProcess::EProcess() const
{
    CEProcessPtr        pEProcess;

    // Try to create the exelwtive process (at the same address)
    pEProcess = createEProcess(ptrKProcess());

    return pEProcess;

} // EProcess

//******************************************************************************

CKProcesses::CKProcesses
(
    const CListEntry&   KProcessList
)
:   m_ulKProcessCount(0),
    m_ptrKProcesses(NULL),
    m_aKProcesses(NULL)
{
    PROCESS             ptrProcess;
    ULONG               ulProcess;

    // Count the number of processes
    ptrProcess = KProcessList.ptrHeadEntry();
    while (ptrProcess != NULL)
    {
        // Increment process count and move to next process
        m_ulKProcessCount++;
        ptrProcess = KProcessList.ptrNextEntry();
    }
    // Check for processes present
    if (m_ulKProcessCount != 0)
    {
        // Allocate the process base addresses and process array
        m_ptrKProcesses = new PROCESS[m_ulKProcessCount];
        m_aKProcesses   = new CKProcessPtr[m_ulKProcessCount];

        // Loop filling in the process base addresses
        ulProcess  = 0;
        ptrProcess = KProcessList.ptrHeadEntry();
        while (ptrProcess != NULL)
        {
            // Fill in process base address (Only if enough room)
            if (ulProcess < m_ulKProcessCount)
            {
                m_ptrKProcesses[ulProcess] = ptrProcess;
            }
            // Increment process count and move to next process
            ulProcess++;
            ptrProcess = KProcessList.ptrNextEntry();
        }
        // Make sure the process count is correct
        if (ulProcess != m_ulKProcessCount)
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": Kernel process count and list don't agree (%d != %d)",
                             ulProcess, m_ulKProcessCount);
        }
    }

} // CKProcesses

//******************************************************************************

CKProcesses::CKProcesses
(
    const CEProcesses  *pEProcesses
)
:   m_ulKProcessCount((pEProcesses != NULL) ? pEProcesses->EProcessCount() : 0),
    m_ptrKProcesses(NULL),
    m_aKProcesses(NULL)
{
    PROCESS             ptrProcess;
    ULONG               ulProcess;

    assert(pEProcesses != NULL);

    // Check for processes present
    if (m_ulKProcessCount != 0)
    {
        // Allocate the process base addresses and process array
        m_ptrKProcesses = new PROCESS[m_ulKProcessCount];
        m_aKProcesses   = new CKProcessPtr[m_ulKProcessCount];

        // Loop filling in the process base addresses
        for (ulProcess = 0; ulProcess < m_ulKProcessCount; ulProcess++)
        {
            // Fill in the next process base address
            m_ptrKProcesses[ulProcess] = pEProcesses->ptrEProcess(ulProcess);
        }
    }

} // CKProcesses

//******************************************************************************

CKProcesses::~CKProcesses()
{

} // ~CKProcesses

//******************************************************************************

const CKProcessPtr
CKProcesses::KProcess
(
    ULONG               ulKProcess
) const
{
    // Check for invalid process index
    if (ulKProcess >= KProcessCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid kernel process index %d (>= %d)",
                         ulKProcess, KProcessCount());
    }
    // Check to see if process needs to be loaded
    if (m_aKProcesses[ulKProcess] == NULL)
    {
        // Check for non-zero process address
        if (m_ptrKProcesses[ulKProcess] != NULL)
        {
            // Try to create the requested kernel process
            m_aKProcesses[ulKProcess] = createKProcess(m_ptrKProcesses[ulKProcess]);
        }
    }
    return m_aKProcesses[ulKProcess];

} // KProcess

//******************************************************************************

PROCESS
CKProcesses::ptrKProcess
(
    ULONG               ulKProcess
) const
{
    // Check for invalid process index
    if (ulKProcess >= KProcessCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid kernel process index %d (>= %d)",
                         ulKProcess, KProcessCount());
    }
    // Return the requested process address
    return m_ptrKProcesses[ulKProcess];

} // ptrKProcess

//******************************************************************************

CKProcessPtr*
CKProcesses::KProcesses()
{
    ULONG               ulKProcess;

    // Loop making sure all the processes are loaded
    for (ulKProcess = 0; ulKProcess < KProcessCount(); ulKProcess++)
    {
        // Check to see if next process needs to be loaded
        if (m_aKProcesses[ulKProcess] == NULL)
        {
            // Check for non-zero process address
            if (m_ptrKProcesses[ulKProcess] != NULL)
            {
                // Try to create the next kernel process
                m_aKProcesses[ulKProcess] = createKProcess(m_ptrKProcesses[ulKProcess]);
            }
        }
    }
    return &m_aKProcesses[0];

} // KProcesses

//******************************************************************************

const CKProcessPtr
CKProcesses::findKProcess
(
    PROCESS             ptrKProcess
) const
{
    ULONG               ulKProcess;
    CKProcessPtr        pKProcess;

    // Loop searching for the requested process
    for (ulKProcess = 0; ulKProcess < KProcessCount(); ulKProcess++)
    {
        // Check to see if next process is the requested process
        if (CKProcesses::ptrKProcess(ulKProcess) == ptrKProcess)
        {
            // Found the requested kernel process, get it and stop search
            pKProcess = KProcess(ulKProcess);
            break;
        }
    }
    return pKProcess;

} // findKProcess

//******************************************************************************

CEProcess::CEProcess
(
    CEProcessList      *pEProcessList,
    PROCESS             ptrEProcess
)
:   LWnqObj(pEProcessList, ptrEProcess),
    m_ptrEProcess(ptrEProcess),
    INIT(createTime),
    INIT(exitTime),
    INIT(uniqueProcessId),
    INIT(commitCharge),
    INIT(peakVirtualSize),
    INIT(virtualSize),
    INIT(session),
    INIT(imageFileName),
    INIT(activeThreads),
    INIT(lastThreadExitStatus),
    INIT(peb),
    INIT(readOperationCount),
    INIT(writeOperationCount),
    INIT(otherOperationCount),
    INIT(readTransferCount),
    INIT(writeTransferCount),
    INIT(otherTransferCount),
    INIT(commitChargeLimit),
    INIT(commitChargePeak),
    m_ThreadList(&m_threadListHeadField, &CEThread::threadListEntryField(), ptrEProcess + (m_threadListHeadField.isPresent() ? m_threadListHeadField.offset() : 0)),
    m_pKProcess(NULL),
    m_pEThreads(NULL)
{
    assert(pEProcessList != NULL);

    // Get the exelwtive process information
    READ(createTime,           ptrEProcess);
    READ(exitTime,             ptrEProcess);
    READ(uniqueProcessId,      ptrEProcess);
    READ(commitCharge,         ptrEProcess);
    READ(peakVirtualSize,      ptrEProcess);
    READ(virtualSize,          ptrEProcess);
    READ(session,              ptrEProcess);
    READ(imageFileName,        ptrEProcess);
    READ(activeThreads,        ptrEProcess);
    READ(lastThreadExitStatus, ptrEProcess);
    READ(peb,                  ptrEProcess);
    READ(readOperationCount,   ptrEProcess);
    READ(writeOperationCount,  ptrEProcess);
    READ(otherOperationCount,  ptrEProcess);
    READ(readTransferCount,    ptrEProcess);
    READ(writeTransferCount,   ptrEProcess);
    READ(otherTransferCount,   ptrEProcess);
    READ(commitChargeLimit,    ptrEProcess);
    READ(commitChargePeak,     ptrEProcess);

} // CEProcess

//******************************************************************************

CEProcess::~CEProcess()
{

} // ~CEProcess

//******************************************************************************

CEProcessPtr
CEProcess::createEProcess
(
    PROCESS             ptrEProcess
)
{
    CEProcessPtr        pEProcess;

    // Check for valid exelwtive process address given
    if (isKernelModeAddress(ptrEProcess))
    {
        // Check to see if this exelwtive process already exists
        pEProcess = findObject(EProcessList(), ptrEProcess);
        if (pEProcess == NULL)
        {
            // Try to create the new exelwtive process object
            pEProcess = new CEProcess(EProcessList(), ptrEProcess);
        }
    }
    return pEProcess;

} // createEProcess

//******************************************************************************

const CKProcessPtr
CEProcess::KProcess() const
{
    PROCESS             ptrKProcess;

    // Check for kernel process already created
    if (m_pKProcess == NULL)
    {
        // Check to see if kernel process is present
        if (pcbField().isPresent())
        {
            // Compute the kernel process address
            ptrKProcess = ptrEProcess() + pcbField().offset();

            // Try to create the exelwtive process kernel process
            m_pKProcess = createKProcess(ptrKProcess);
        }
    }
    return m_pKProcess;

} // KProcess

//******************************************************************************

const CEThreadsPtr
CEProcess::EThreads() const
{
    // Check for exelwtive threads already created
    if (m_pEThreads == NULL)
    {
        // Check to see if exelwtive threads are present
        if (threadList().isPresent())
        {
            // Try to create the exelwtive process threads
            m_pEThreads = new CEThreads(threadList());
        }
    }
    return m_pEThreads;

} // EThreads

//******************************************************************************

CString
CEProcess::processName() const
{
    ULONG               ulChar;
    ULONG               ulNameLength;
    CString             sProcessName;

    // Check for process name available
    if (imageFileNameField().isPresent())
    {
        // Check for a valid process
        if (isValid())
        {
            // get the process name length (Array size)
            ulNameLength = imageFileNameField().dimension(0);

            // Reserve enough string space for the process name
            sProcessName.reserve(ulNameLength);

            // Copy the image filename as the process name
            memcpy(sProcessName.data(), imageFileNameMember().getStruct(), ulNameLength);

            // Loop checking for a file extension (Terminate)
            for (ulChar = 0; ulChar < ulNameLength; ulChar++)
            {
                // Check for end of process name
                if (sProcessName[ulChar] == '\0')
                {
                    break;
                }
                // Check for start of file extension (Period)
                if (sProcessName[ulChar] == '.')
                {
                    // Terminate the process name and exit
                    sProcessName[ulChar] = '\0';
                    break;
                }
            }
            // Make sure process name is terminated
            sProcessName[ulNameLength - 1] = '\0';
        }
        else    // Process is not longer valid (Don't try to use process name)
        {
            // Reserve enough string space for process address as name
            sProcessName.reserve(2 + pointerWidth());

            // Generate the process name as the process address
            sProcessName.sprintf("0x%0*I64x", PTR(ptrEProcess()));
        }
    }
    else    // Process name is not available
    {
        // Reserve enough string space for process address as name
        sProcessName.reserve(2 + pointerWidth());

        // Generate the process name as the process address
        sProcessName.sprintf("0x%0*I64x", PTR(ptrEProcess()));
    }
    return sProcessName;

} // processName

//******************************************************************************

CEProcesses::CEProcesses
(
    const CListEntry&   EProcessList
)
:   m_ulEProcessCount(0),
    m_ptrEProcesses(NULL),
    m_aEProcesses(NULL),
    m_pKProcesses(NULL)
{
    PROCESS             ptrProcess;
    ULONG               ulProcess;

    // Count the number of processes
    ptrProcess = EProcessList.ptrHeadEntry();
    while (ptrProcess != NULL)
    {
        // Increment process count and move to next process
        m_ulEProcessCount++;
        ptrProcess = EProcessList.ptrNextEntry();
    }
    // Check for processes present
    if (m_ulEProcessCount != 0)
    {
        // Allocate the process base addresses and process array
        m_ptrEProcesses = new PROCESS[m_ulEProcessCount];
        m_aEProcesses   = new CEProcessPtr[m_ulEProcessCount];

        // Loop filling in the process base addresses
        ulProcess  = 0;
        ptrProcess = EProcessList.ptrHeadEntry();
        while (ptrProcess != NULL)
        {
            // Fill in process base address (Only if enough room)
            if (ulProcess < m_ulEProcessCount)
            {
                m_ptrEProcesses[ulProcess] = ptrProcess;
            }
            // Increment process count and move to next process
            ulProcess++;
            ptrProcess = EProcessList.ptrNextEntry();
        }
        // Make sure the process count is correct
        if (ulProcess != m_ulEProcessCount)
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": Exelwtive process count and list don't agree (%d != %d)",
                             ulProcess, m_ulEProcessCount);
        }
    }

} // CEProcesses

//******************************************************************************

CEProcesses::~CEProcesses()
{

} // ~CEProcesses

//******************************************************************************

const CEProcessPtr
CEProcesses::EProcess
(
    ULONG               ulEProcess
) const
{
    // Check for invalid process index
    if (ulEProcess >= EProcessCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid exelwtive process index %d (>= %d)",
                         ulEProcess, EProcessCount());
    }
    // Check to see if process needs to be loaded
    if (m_aEProcesses[ulEProcess] == NULL)
    {
        // Check for non-zero process address
        if (m_ptrEProcesses[ulEProcess] != NULL)
        {
            // Try to create the requested process
            m_aEProcesses[ulEProcess] = createEProcess(m_ptrEProcesses[ulEProcess]);
        }
    }
    return m_aEProcesses[ulEProcess];

} // EProcess

//******************************************************************************

PROCESS
CEProcesses::ptrEProcess
(
    ULONG               ulEProcess
) const
{
    // Check for invalid process index
    if (ulEProcess >= EProcessCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid exelwtive process index %d (>= %d)",
                         ulEProcess, EProcessCount());
    }
    // Return the requested process address
    return m_ptrEProcesses[ulEProcess];

} // ptrEProcess

//******************************************************************************

CEProcessPtr*
CEProcesses::EProcesses()
{
    ULONG               ulEProcess;

    // Loop making sure all the processes are loaded
    for (ulEProcess = 0; ulEProcess < EProcessCount(); ulEProcess++)
    {
        // Check to see if next process needs to be loaded
        if (m_aEProcesses[ulEProcess] == NULL)
        {
            // Check for non-zero process address
            if (m_ptrEProcesses[ulEProcess] != NULL)
            {
                // Try to create the next exelwtive process
                m_aEProcesses[ulEProcess] = createEProcess(m_ptrEProcesses[ulEProcess]);
            }
        }
    }
    return &m_aEProcesses[0];

} // EProcesses

//******************************************************************************

const CEProcessPtr
CEProcesses::findEProcess
(
    PROCESS             ptrEProcess
) const
{
    ULONG               ulEProcess;
    CEProcessPtr        pEProcess;

    // Loop searching for the requested process
    for (ulEProcess = 0; ulEProcess < EProcessCount(); ulEProcess++)
    {
        // Check to see if next process is the requested process
        if (CEProcesses::ptrEProcess(ulEProcess) == ptrEProcess)
        {
            // Found the requested exelwtive process, get it and stop search
            pEProcess = EProcess(ulEProcess);
            break;
        }
    }
    return pEProcess;

} // findEProcess

//******************************************************************************

const CEProcessPtr
CEProcesses::findEProcess
(
    ULONG64             ulEProcessId
) const
{
    ULONG               ulEProcess;
    CEProcessPtr        pEProcess;

    // Loop searching for the requested process
    for (ulEProcess = 0; ulEProcess < EProcessCount(); ulEProcess++)
    {
        // Get the next exelwtive process to check
        pEProcess = EProcess(ulEProcess);
        if (pEProcess != NULL)
        {
            // Check to see if next process is the requested process
            if (pEProcess->uniqueProcessId() == ulEProcessId)
            {
                // Found the requested process, stop search
                break;
            }
            else    // Not the requested process
            {
                // Clear the current process
                pEProcess = NULL;
            }
        }
    }
    return pEProcess;

} // findEProcess

//******************************************************************************

const CKProcessesPtr
CEProcesses::KProcesses() const
{
    // Check for kernel processes not already created
    if (m_pKProcesses == NULL)
    {
        // Try to create the kernel processes
        m_pKProcesses = new CKProcesses(this);
    }
    return m_pKProcesses;

} // KProcesses

//******************************************************************************

const CKProcessesPtr
getKProcesses
(
    bool                bThrow
)
{
    CKProcessesPtr      pKProcesses;

    // Check to see if we can get the kernel processes
    if (mmProcessList().isPresent() && CEProcess::mmProcessLinksField().isPresent())
    {
        CListEntry  processList(&CEProcess::mmProcessLinksField(), mmProcessList().offset());

        // Create the kernel processes
        try
        {
            pKProcesses = new CKProcesses(processList);
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
    else    // No kernel processes present
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
        }
    }
    return pKProcesses;

} // getKProcesses

//******************************************************************************

const CKProcessesPtr
getKernelProcesses
(
    bool                bThrow
)
{
    CKProcessesPtr      pKernelProcesses;

    // Check to see if we can get the active kernel processes
    if (psActiveProcessHead().isPresent() && CEProcess::activeProcessLinksField().isPresent())
    {
        CListEntry  processList(&CEProcess::activeProcessLinksField(), psActiveProcessHead().offset());

        // Get the active kernel processes
        try
        {
            pKernelProcesses = new CKProcesses(processList);
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
    else    // No active kernel processes present
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
        }
    }
    return pKernelProcesses;

} // getKernelProcesses

//******************************************************************************

const CEProcessesPtr
getEProcesses
(
    bool                bThrow
)
{
    CEProcessesPtr      pEProcesses;

    // Check to see if we can get the exelwtive processes
    if (mmProcessList().isPresent() && CEProcess::mmProcessLinksField().isPresent())
    {
        CListEntry  processList(&CEProcess::mmProcessLinksField(), mmProcessList().offset());

        // Create the exelwtive processes
        try
        {
            pEProcesses = new CEProcesses(processList);
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
    else    // No exelwtive processes present
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
        }
    }
    return pEProcesses;

} // getEProcesses

//******************************************************************************

const CEProcessesPtr
getExelwtiveProcesses
(
    bool                bThrow
)
{
    CEProcessesPtr      pExelwtiveProcesses;

    // Check to see if we can get the active exelwtive processes
    if (psActiveProcessHead().isPresent() && CEProcess::activeProcessLinksField().isPresent())
    {
        CListEntry  processList(&CEProcess::activeProcessLinksField(), psActiveProcessHead().offset());

        // Get the active exelwtive processes
        try
        {
            pExelwtiveProcesses = new CEProcesses(processList);
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
    else    // No active exelwtive processes present
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
        }
    }
    return pExelwtiveProcesses;

} // getExelwtiveProcesses

//******************************************************************************

const CKProcessPtr
findKProcess
(
    PROCESS             ptrKProcess
)
{
    CKProcessesPtr      pKProcesses;
    CKProcessPtr        pKProcess;

    // Get the kernel processes to search for the given process
    pKProcesses = getKProcesses();
    if (pKProcesses != NULL)
    {
        // Try to find the requested kernel process
        pKProcess = pKProcesses->findKProcess(ptrKProcess);
        if (pKProcess == NULL)
        {
            // Get active kernel processes to search for the given process
            pKProcesses = getKernelProcesses();
            if (pKProcesses != NULL)
            {
                // Try to find the requested kernel process
                pKProcess = pKProcesses->findKProcess(ptrKProcess);
            }
        }
    }
    else    // Can't get kernel processes, try active kernel processes
    {
        // Get active kernel processes to search for the given process
        pKProcesses = getKernelProcesses();
        if (pKProcesses != NULL)
        {
            // Try to find the requested kernel process
            pKProcess = pKProcesses->findKProcess(ptrKProcess);
        }
    }
    return pKProcess;

} // findKProcess

//******************************************************************************

const CEProcessPtr
findEProcess
(
    PROCESS             ptrEProcess
)
{
    CEProcessesPtr      pEProcesses;
    CEProcessPtr        pEProcess;

    // Get the exelwtive processes to search for the given process
    pEProcesses = getEProcesses();
    if (pEProcesses != NULL)
    {
        // Try to find the requested exelwtive process
        pEProcess = pEProcesses->findEProcess(ptrEProcess);
        if (pEProcess == NULL)
        {
            // Get active exelwtive processes to search for the given process
            pEProcesses = getExelwtiveProcesses();
            if (pEProcess != NULL)
            {
                // Try to find the requested exelwtive process
                pEProcess = pEProcesses->findEProcess(ptrEProcess);
            }
        }
    }
    else    // Can't get exelwtive processes, try active exelwtive processes
    {
        // Get active exelwtive processes to search for the given process
        pEProcesses = getExelwtiveProcesses();
        if (pEProcesses != NULL)
        {
            // Try to find the requested exelwtive process
            pEProcess = pEProcesses->findEProcess(ptrEProcess);
        }
    }
    return pEProcess;

} // findEProcess

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
