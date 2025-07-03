/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "msdec.h"
#include "falctrace.h"
#include "vmem.h"
#include <stdio.h>

/*!
 * Defines the default number of lines/trace buffers to dump when the user does
 * not make an explicit request.
 */
#define FALCTRACE_NUM_LINES_DEFAULT 16;


/*!
 * Structure containing per-engine specific falcon-trace context.  This
 * information will be populated the user initializes an engine for falctrace
 * usage.
 */
struct FALCTRACE_CTX
{
    //<! Address of the start of the falcon-trace buffer
    LwU64      addr;

    //<! The size (in bytes) of the falcon-trace buffer
    LwU32      size;

    //<! Is the address in physical or virtual space
    LwBool     bIsPhyAddr;

    //<! Is the address in system ram or video ram
    LwBool     bIsSysMem;

    //<! The vmem-structure used to access the falcon-trace buffer
    VMemSpace  vmemSpace;

    //<! Used to keep track of whether or not this context is valid/initialized
    BOOL       bInitialized;

    //<! Used when printing sequential message that do no all contain newlines
    BOOL       bSkipNextPrefix;
};
typedef struct FALCTRACE_CTX FALCTRACE_CTX;


/*!
 * Structure used to represent an individual entry in the falcon-trace buffer.
 */
struct FALCTRACE_ENTRY
{
    //<! Offset of this entry (in bytes) from the start of the trace buffer
    LwU32   offset;

    //<! Trace-count value of this entry
    LwU32   count;

    //<! Array containing the four falcon-trace parameters
    LwU32   param[4];

    //<! Pointer to the start of this entry's format string
    char   *fmtString;
};
typedef struct FALCTRACE_ENTRY FALCTRACE_ENTRY;


/*!
 * Array of falcon-trace engine context's
 */
static FALCTRACE_CTX Engines[MSDEC_ENGINE_COUNT];


//
// Local function prototypes
//
static void  _falctraceGetBufferByIndex    (FALCTRACE_CTX *pTraceInfo, LwU8 *pData, LwU32 index, FALCTRACE_ENTRY *pTraceBuffer);
static void  _falctraceGetLastBufferEntry  (FALCTRACE_CTX *pTraceInfo, LwU8 *pData, LwU32 *pIndex);
static LW_STATUS _falctraceGetVmemFromEngineId (LwU32 engineId, VMemSpace *pVmemSpace);
static void  _falctracePrintTraceBuffer    (FALCTRACE_CTX *pTraceInfo, FALCTRACE_ENTRY *pTraceBuffer);


/*!
 * One-time initialization function for the falctrace dump facility.  Simply
 * used to keep track of the engines which were initialized by the user for
 * falctrace.  This function should be called once when lwwatch is first
 * initialized.
 */
void
falctraceInit(void)
{
    LwU32 e;
    for (e = 0; e < MSDEC_ENGINE_COUNT; e++)
    {
        Engines[e].bInitialized = FALSE;
    }
}


/*!
 * Initializes an engine for falctrace use by declaring the location (e.g.
 * address space, aperture, address, etc.) and size of the engine's trace
 * buffer area. This function must be called prior to any attempt to dump
 * information in the trace buffer.  This function need only be called once
 * for each engine.
 *
 * @param[in]  engineId    An ID representing the specific msdec/falcon engine
 *                         being initialized. See msdec.h for a list of all
 *                         valid falctrace engine identifiers.
 * @param[in]  bIsPhyAddr  The address space of the trace buffer.
 * @param[in]  bIsSysMem   The aperture of the trace buffer.
 * @param[in]  addr        The address of the engine's trace buffer.
 * @param[in]  size        The size (in bytes) of the trace buffer.
 *
 * @return  LW_OK     if the engine was successfully initialized for falctrace
 * @return  LW_ERR_GENERIC  if the engine identifier is invalid or does not
 *                    otherwise support falctrace.
 */
LW_STATUS
falctraceInitEngine
(
    LwU32    engineId,
    LwBool   bIsPhyAddr,
    LwBool   bIsSysMem,
    LwU64    addr,
    LwU32    size
)
{
    LW_STATUS status = LW_OK;

    Engines[engineId].bInitialized = FALSE;
    Engines[engineId].bIsPhyAddr = bIsPhyAddr;

    if (!bIsPhyAddr)
    {
        // find the virtual-memory information for the engine
        status = _falctraceGetVmemFromEngineId(
                     engineId,
                     &Engines[engineId].vmemSpace);
    }

    if (status == LW_OK)
    {
        Engines[engineId].bIsSysMem       = bIsSysMem;
        Engines[engineId].addr            = addr;
        Engines[engineId].size            = size;
        Engines[engineId].bInitialized    = TRUE;
        Engines[engineId].bSkipNextPrefix = FALSE;
    }
    else
    {
        dprintf("%s: error retrieving virtual-memory information for engine=" \
                "%d\n", __FUNCTION__, engineId);
    }
    return status;
}


/*!
 * Dump the last 'numEntries' entries in the engine's falctrace buffer. If
 * 'numEntries' is zero, the default number of entries will be dumped.
 *
 * @param[in]  engineId    The identifier for the engine to dump falctrace info
 *                         for. A previous call to @ref falctraceInitEngine
 *                         must have been made to for this specific engine
 *                         before calling this function.
 * @param[in]  numEntries  The number of entries to dump.  Zero to print the
 *                         default number of entries.
 *
 * @return  LW_OK     if the dump operation was successful.
 * @return  LW_ERR_GENERIC  if the engine was not previously initialized for
 *                    falctrace
 * @return  LW_ERR_GENERIC  for unexpected errors (like malloc failures)
 */
LW_STATUS
falctraceDump
(
    LwU32  engineId,
    LwU32  numEntries
)
{
    FALCTRACE_ENTRY  traceBuffer;
    LwU8            *pData;
    LwU32            numBuffers;
    LwU32            i;
    LwU32            index;

    // throw an error if the falctrace was not initialized for the engine
    if (!Engines[engineId].bInitialized)
    {
        dprintf("Error: falctrace not initialized for engine=%d. See !help " \
                "for falctrace usage.\n", engineId);
        return LW_ERR_GENERIC;
    }

    // dump the default when zero or less lines are requested
    if (numEntries == 0)
    {
        numEntries = FALCTRACE_NUM_LINES_DEFAULT;
    }

    // truncate the requested line count if necessary
    numBuffers = Engines[engineId].size / FALCTRACE_BUFFER_SIZE;
    if (numEntries > numBuffers)
    {
        numEntries = numBuffers;
    }

    // allocate a buffer to store the trace
    pData = (LwU8*)malloc(Engines[engineId].size * sizeof(LwU8));
    if (pData == NULL)
    {
        dprintf("%s: malloc failure\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    // now read-in the trace
    dprintf("falctrace: reading %d bytes from %s addr=0x%llx at %s ... ",
                Engines[engineId].size,
                Engines[engineId].bIsPhyAddr ? "physical" : "virtual",
                Engines[engineId].addr,
                Engines[engineId].bIsSysMem ? "sysmem" : "vidmem");

    if (Engines[engineId].bIsPhyAddr)
    {
        if (Engines[engineId].bIsSysMem)
        {
            readSystem(Engines[engineId].addr, (void*)pData, Engines[engineId].size);
        }
        else
        {
            pFb[indexGpu].fbRead(Engines[engineId].addr, (void*)pData, Engines[engineId].size);
        }
    }
    else
    {
        pVmem[indexGpu].vmemRead(
            &Engines[engineId].vmemSpace,  // pVMemSpace
             Engines[engineId].addr,       // addr
             Engines[engineId].size,       // length
             pData);                       // pData
    }

    dprintf("done.\n");

    //
    // Find the last buffer logged in the trace walk back from that
    // index to print the requested number of entries.
    //
    _falctraceGetLastBufferEntry(&Engines[engineId], pData, &index);
    for (i = 0; i < numEntries; i++)
    {
        _falctraceGetBufferByIndex(
            &Engines[engineId],
             pData,
             (i + index - numEntries) % numBuffers,
            &traceBuffer);

        _falctracePrintTraceBuffer(&Engines[engineId], &traceBuffer);
    }
    free(pData);
    return LW_OK;
}


/*!
 * Extract the information for a specific trace buffer entry using the entry's
 * index in the trace buffer area.  The returned information will include the
 * the trace count value for the entry, all 4 32-bit parameters, and the format
 * string. This function expects each entry to be formatted as follows:
 *
 *    4-bytes       4-bytes  4-bytes  4-bytes  4-bytes  44-bytes
 *   <trace_count> <param0> <param1> <param2> <param3> <format_string ....>
 *
 * @param[in]   pTraceInfo    Engine-specific falcon-trace context
 * @param[in]   pData         Buffer containing the entire falcon-trace data
 *                            read from memory.
 * @param[in]   index         The desired index in the falcon-trace buffer.
 *                            This value is modded by the max entry count to
 *                            prevent out-of-bounds access.
 * @param[out]  pTraceBuffer  Pointer to trace-buffer structure to populate
 *                            for entry at the given index.
 *
 */
static void
_falctraceGetBufferByIndex
(
    FALCTRACE_CTX   *pTraceInfo,
    LwU8            *pData,
    LwU32            index,
    FALCTRACE_ENTRY *pTraceBuffer
)
{
    LwU32  numBuffers;
    LwU32 *pData32;
    LwU32  offset;

    numBuffers = pTraceInfo->size / FALCTRACE_BUFFER_SIZE;
    index      = (index % numBuffers);
    offset     = (index * FALCTRACE_BUFFER_SIZE);
    pData     += offset;
    pData32    = (LwU32*)pData;

    pTraceBuffer->offset    = offset;
    pTraceBuffer->count     = pData32[0];
    pTraceBuffer->param[0]  = pData32[1];
    pTraceBuffer->param[1]  = pData32[2];
    pTraceBuffer->param[2]  = pData32[3];
    pTraceBuffer->param[3]  = pData32[4];
    pTraceBuffer->fmtString = (char*)&pData32[5];
}


/*!
 * Walk over the falcon-trace buffer and find the last entry that was logged.
 *
 * @param[in]   pTraceInfo  Engine-specific falcon-trace context
 * @param[in]   pData       Buffer containing the entire falcon-trace data read
 *                          from memory.
 * @param[out]  pIndex      Pointer to write with the index of the last entry
 */
static void
_falctraceGetLastBufferEntry
(
    FALCTRACE_CTX *pTraceInfo,
    LwU8          *pData,
    LwU32         *pIndex
)
{
    FALCTRACE_ENTRY  traceBuffer;
    LwU32            numBuffers;
    LwU32            prevCount = 0;
    LwU32            b;
    //
    // We don't need to go through every buffer in most cases (though that is
    // the worst-case).  We just need to find the buffer that is followed by a
    // buffer with a lesser count value.
    //
    numBuffers = pTraceInfo->size / FALCTRACE_BUFFER_SIZE;
    for (b = 0; b < numBuffers; b++)
    {
        _falctraceGetBufferByIndex(pTraceInfo, pData, b, &traceBuffer);
        if (traceBuffer.count < prevCount)
        {
            *pIndex = b;
            break;
        }
        prevCount = traceBuffer.count;
    }
    //
    // If we reached the end, the end is the last buffer.  An optization here
    // would be keep track of the previous value returned and start the search
    // at that point.
    //
    if (b == numBuffers)
    {
        *pIndex = b - 1;
    }
}


/*!
 * Retieves the vmem-structure associated with the given engine-id if the
 * structure exists.
 *
 * @param[in]   engineId    The identifier for the engine to retrieve the vmem
 *                          information for.
 * @param[out]  pVmemSpace  Pointer to the vmem-structure to populate.
 *
 * @return  LW_OK     if the vmem-structure was successfully retrieved
 * @return  LW_ERR_GENERIC  if the engine does not have a vmem-structure associated
 *                    with it.
 */
static LW_STATUS
_falctraceGetVmemFromEngineId
(
    LwU32      engineId,
    VMemSpace *pVmemSpace
)
{
    LW_STATUS  status = LW_ERR_GENERIC;
    switch (engineId)
    {
        case MSDEC_PMU:
        {
            status = vmemGet(pVmemSpace, VMEM_TYPE_PMU, NULL);
            break;
        }
    }
    return status;
}


/*!
 * Print the specified falcon-trace buffer entry.
 *
 * @param[in]  pTraceInfo    Engine-specific falcon-trace context.
 * @param[in]  pTraceBuffer  The buffer entry to print.
 */
static void
_falctracePrintTraceBuffer
(
    FALCTRACE_CTX   *pTraceInfo,
    FALCTRACE_ENTRY *pTraceBuffer
)
{
    char   tempBuffer[FALCTRACE_BUFFER_SIZE];
    char  *pFirstNewline;
    size_t length;
    LwU32  i;

    // bail if there is nothing to print
    if (*pTraceBuffer->fmtString == '\0')
    {
        return;
    }

    //
    // Each trace buffer/entry will be printed on its on line. For debugging
    // purposes, it is useful to prefix each trace entry with the address of
    // the entry in the trace buffer as well as the trace-count value of the
    // entry (provides a notion of ordering). Some entries will contain more
    // than one newline character.  These newline characters should still be
    // honored.  However, they too should be prefixed with the same debug
    // information. The general strategy for doing this will be to write the
    // formatted string into a temporary buffer using sprintf, and then pick
    // out any newline characters that do not appear at the end of the string.
    //

    // print the initial prefix information
    if (!pTraceInfo->bSkipNextPrefix)
    {
        dprintf("[0x%08x] ", pTraceBuffer->count);
    }

    // format the string in a temporary buffer
    snprintf(tempBuffer, FALCTRACE_BUFFER_SIZE,
             pTraceBuffer->fmtString,
             pTraceBuffer->param[0],
             pTraceBuffer->param[1],
             pTraceBuffer->param[2],
             pTraceBuffer->param[3]);

    //
    // Find the oclwrrance of the first newline character as well as the length
    // of the string.
    //
    pFirstNewline = strchr(tempBuffer, '\n');
    length        = strlen(tempBuffer);
    pTraceInfo->bSkipNextPrefix = FALSE;

    //
    // Simply print the entire string in one shot if the first newline is at
    // the end of the string (normal case).
    //
    if (pFirstNewline == (&tempBuffer[0] + length))
    {
        dprintf("%s", tempBuffer);
    }
    //
    // Otherwise, detect each newline and print the prefix information before
    // further characters are printed.
    //
    else
    {
        // ignore the last character (most likely a newline)
        for (i = 0; i < length - 1; i++)
        {
            dprintf("%c", tempBuffer[i]);
            if (tempBuffer[i] == '\n')
            {
                dprintf("[0x%08x] ", pTraceBuffer->count);
            }
        }
        //
        // Deal with the last character now.  Start by just printing it out.
        // Then look at to see if it is a newline.  If it's not, then we
        // should NOT print the prefix the next time around. It's likely
        // that the log contains a dump of bytes that are getting printed
        // in a loop (like in a memory dump).
        //
        dprintf("%c", tempBuffer[i]);
        if (tempBuffer[i] != '\n')
        {
            pTraceInfo->bSkipNextPrefix = TRUE;
        }
    }
}

