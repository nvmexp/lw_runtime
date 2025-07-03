/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1993-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/***************************************************************************\
*                                                                           *
* Module: dumpSession.c                                                     *
*   Supports hooking I/O routines to an LwDebug dump file.                  *
*                                                                           *
\***************************************************************************/

//
// includes
//
#include <stdio.h>
#include <string.h>

#include <lwdump.h>
#include "os.h"
#include "hal.h"
#include "print.h"
#include "priv.h"
#include "fermi/gf100/dev_master.h"

// Macro used to get the "count" of the number array elements
#define countof(array)  (sizeof(array) / sizeof(array[0]))

#if defined(LW_WINDOWS)  &&  !defined(LW_MODS)
#include "lwoca.h"
#endif // LW_WINDOWS

#ifdef LWDEBUG_SUPPORTED
#include "lwdzip.h"
#include "prbdec.h"
#include "g_lwdebug_pb.h"
#include "g_all_dcl_pb.h"
#include "g_regs_pb.h"
#include "g_lwlog_pb.h"

typedef struct
{
    LwU64 offset;
    LwU32 length;
    LwU32 capacity;
    LwU8 *buffer;
} MEMORY_RANGE;

typedef struct
{
    MEMORY_RANGE *ranges;
    LwU32 numRanges;
    MEMORY_RANGE *needed;
    LwU32 numNeeded;
} MEMORY_SPACE;

#define MEMORY_SPACE_UNKNOWN 0
#define MEMORY_SPACE_REGS    1
#define MEMORY_SPACE_FB      2
#define MEMORY_SPACE_COUNT   3

static MEMORY_SPACE memSpaces[MEMORY_SPACE_COUNT];
static PRB_MSG dumpMsg;

LwBool isInitialized = FALSE;

//
// Boolean for tracking if command run on a dump is missing any registers.
//  For now a Boolean and this simple logic is good enough.  If lwwatch commands
//  wind up calling lwwatch commands, then this will require a more elaborate
//  "book keeping".
//
LwBool bRegisterMissing = LW_FALSE;


// This is the same key used by LwDebugDump
static LWD_CRYPT_KEY cryptKey = {
    0x4f, 0x77, 0x44, 0x65,
    0x75, 0x67, 0x44, 0x75,
    0x70, 0x20, 0x21, 0x19,
    0x4e, 0x56, 0x49, 0x44,
};

static void
clearMemRange(MEMORY_RANGE *pRange)
{
    free(pRange->buffer);
    pRange->offset = 0;
    pRange->length = 0;
    pRange->capacity = 0;
    pRange->buffer = NULL;
}

//
// Expand this range to hold the new value passed in.
//   Add new value to the end of the range.
//
static void
expandMemRange(MEMORY_RANGE *pRange, LwU8 value)
{
    if (pRange->length >= pRange->capacity)
    {
        if (pRange->length > 0)
            pRange->capacity = pRange->length << 1;
        else
            pRange->capacity = 1;
        pRange->buffer = (LwU8*)realloc(pRange->buffer, pRange->capacity);
    }
    pRange->buffer[pRange->length] = value;
    ++pRange->length;
}

static LwU32
mergeMemRanges(MEMORY_RANGE *ranges, LwU32 numRanges)
{
    LwU32 i;
    LwU32 j;
    LwU32 k;
    MEMORY_RANGE temp;

    // Loop through Range Table Entries
    for (i = 0; i < numRanges; ++i)
    {
        //
        // Loop through all ranges after current (i)
        //    until one is found that is NOT contiguous  with this one.
        //
        temp.length = ranges[i].length;
        for (j = i; j < (numRanges - 1) ; ++j)
        {
            // If Next Range is NOT adjacent to current, then break.
            if ((ranges[j].offset + ranges[j].length) < ranges[j+1].offset)
                break;
            temp.length += ranges[j+1].length;
        }

        // If adjacent Ranges found, combine them.
        if (i != j)
        {
            temp.offset = ranges[i].offset;
            if (ranges[i].buffer != NULL)
            {
                temp.capacity = temp.length;
                temp.buffer = (LwU8*)malloc(temp.capacity);
                for (k = i; k <= j; ++k)
                {
                    // Copy data from the old buffer, then free old buffer.
                    memcpy(&temp.buffer[ranges[k].offset - temp.offset],
                           ranges[k].buffer, ranges[k].length);
                    free(ranges[k].buffer);
                }
            }
            else
            {
                // Current Range has NULL buffer (no Entries)
                temp.capacity = 0;
                temp.buffer = NULL;
            }
            // Update current Range Table Entry
            ranges[i] = temp;
            // If there is a next Range, then move "down" the rest of the entries.
            if (j <(numRanges-1))
            {
                memmove(&ranges[i+1], &ranges[j+1],
                        (numRanges - (j + 1)) * sizeof(MEMORY_RANGE));
            }
            numRanges -= j - i;
        }
    }
    return numRanges;
}

static LwBool
findMemRange
(
    MEMORY_RANGE *ranges,
    LwU32 numRanges,
    LwU64 offset,
    LwU32 *pIndex
)
{
    LwU32 min = 0;
    LwU32 max = numRanges;
    LwU32 mid = max / 2;

    // range based binary search
    while (min != max)
    {
        if (ranges[mid].offset > offset)
            max = mid;
        else if ((offset - ranges[mid].offset) >= ranges[mid].length)
            min = mid + 1;
        else
            break;
        mid = (min + max) / 2;
    }

    if (pIndex != NULL)
        *pIndex = mid;

    return (mid < numRanges) && ((ranges[mid].offset <= offset) &&
                ((offset - ranges[mid].offset) < ranges[mid].length));
}

static void
addValueToMemSpace(MEMORY_SPACE *pSpace, LwU64 offset, LwU8 value)
{
    LwU32 i;
    LwU32 relative;

    // Find this element in an existing range
    //    If found returns index to the range
    //    otherwise returns i = next range
    if (!findMemRange(pSpace->ranges, pSpace->numRanges, offset, &i))
    {
        // Does it go on the end of the previous range?  That is adjacent to
        //    an existing range, tack it on.
        if ((i > 0) && ((offset - pSpace->ranges[i-1].offset) ==
                pSpace->ranges[i-1].length))
        {
            // if so move index back to previous range
            --i;
        }
        else
        {
            // Create an new Range Table Entry.
            pSpace->ranges = (MEMORY_RANGE*)realloc(pSpace->ranges,
                (pSpace->numRanges + 1) * sizeof(MEMORY_RANGE));
            memmove(&pSpace->ranges[i+1], &pSpace->ranges[i],
                    (pSpace->numRanges - i) * sizeof(MEMORY_RANGE));
            ++pSpace->numRanges;
            pSpace->ranges[i].offset = offset;
            pSpace->ranges[i].length = 0;
            pSpace->ranges[i].capacity = 0;
            pSpace->ranges[i].buffer = NULL;
        }
        // Add new entry
        expandMemRange(&pSpace->ranges[i], value);
    }
    else
    { // Was found
        // verify value
        relative = (LwU32)(offset - pSpace->ranges[i].offset);
        if (pSpace->ranges[i].buffer[relative] != value)
        {
            dprintf("lw: Warning! Inconsistent dump memory at offset "
                    "%08X of memory space %i! 0x%02X != 0x%02X\n",
                    (LwU32)offset,
                    (int)(pSpace - memSpaces),
                    (LwU32)pSpace->ranges[i].buffer[relative],
                    (LwU32)value);
        }
    } // Was found
}

static void
addNeededToMemSpace(MEMORY_SPACE *pSpace, LwU64 offset)
{
    LwU32 i = 0;
    if (!findMemRange(pSpace->needed, pSpace->numNeeded, offset, &i))
    {
        if ((i > 0) && ((offset - pSpace->needed[i-1].offset) ==
                pSpace->needed[i-1].length))
        {
            ++pSpace->needed[i-1].length;
        }
        else if (((i + 1) < pSpace->numNeeded) &&
                 ((offset + 1) == pSpace->needed[i+1].offset))
        {
            --pSpace->needed[i+1].offset;
        }
        else
        {
            pSpace->needed = (MEMORY_RANGE*)realloc(pSpace->needed,
                (pSpace->numNeeded + 1) * sizeof(MEMORY_RANGE));
            memmove(&pSpace->needed[i+1], &pSpace->needed[i],
                    (pSpace->numNeeded - i) * sizeof(MEMORY_RANGE));
            ++pSpace->numNeeded;
            pSpace->needed[i].offset = offset;
            pSpace->needed[i].length = 1;
            pSpace->needed[i].capacity = 0;
            pSpace->needed[i].buffer = NULL;
        }
    }
}

static void
addRangesToMemSpace(MEMORY_SPACE *pSpace, const PRB_MSG *pMemMsg)
{
    LwU32 i;
    LwU32 value;
    const PRB_FIELD *pOffsetField = prbGetField(pMemMsg, REGS_REGSANDMEM_OFFSET);
    const PRB_FIELD *pStrideField = prbGetField(pMemMsg, REGS_REGSANDMEM_STRIDE);
    const PRB_FIELD *pValuesField = prbGetField(pMemMsg, REGS_REGSANDMEM_VAL);
    LwU64 offset;
    LwU32 stride;

    if (pOffsetField->values != NULL)
    {
        offset = pOffsetField->values->uint64;
    }
    else
    {
        return;
    }

    if (pStrideField->values != NULL)
    {
        stride = pStrideField->values->uint32;
    }
    else
    {
        return;
    }

    for (i = 0; i < pValuesField->count; ++i)
    {
        value = pValuesField->values[i].uint32;
        addValueToMemSpace(pSpace, offset+0, ((LwU8*)&value)[0]);
        addValueToMemSpace(pSpace, offset+1, ((LwU8*)&value)[1]);
        addValueToMemSpace(pSpace, offset+2, ((LwU8*)&value)[2]);
        addValueToMemSpace(pSpace, offset+3, ((LwU8*)&value)[3]);
        offset += stride;
    }
}

static LwBool
memSpaceRead08(MEMORY_SPACE *pSpace, LwU64 offset, LwU8 *pValue)
{
    LwU32 i;

    if (findMemRange(pSpace->ranges, pSpace->numRanges, offset, &i))
    {
        *pValue = pSpace->ranges[i].buffer[
                    (LwU32)(offset - pSpace->ranges[i].offset)];
        return TRUE;
    }
    else
    {
#if !defined(LW_WINDOWS) // lwdump
        addNeededToMemSpace(pSpace, offset);
        dprintf("lw: Warning! Memory space %i offset "LwU64_FMT" not available!\n",
                (int)(pSpace - memSpaces), offset);
#else // Win OCA
        addNeededToMemSpace(pSpace, offset);
        if (!bRegisterMissing)   // ONLY Print for the first one missing on each command.
        {
            if (lwMode == MODE_LIVE)
            {
                dprintf("lw: Warning! Required registers missing from dump file.  Use !lw.dumpfeedback to get complete list.\n");
            }
            bRegisterMissing = LW_TRUE;
        }
#endif  // Win OCA
        return FALSE;
    }
}

#if defined(LW_WINDOWS)
// static data areas for lwlog extracted froma Proto Buffer.
static  LwBool        bLwLogFound = LW_FALSE;
static  LWLD_Decoder  lwld;
static  char         *pLwLogPtr   = NULL;

/*!
 * @brief This function is called from DebugExtensionUninitialize
 *      when the lw extension is unloaded.   It releases everything allocated
 *      for lwlog from a Proto Buffer.
 *
 * @param[in] none.
 *
 * @return none
 */
void dumpModeReleaseLwlog(void)
{
    unsigned int    i;

    if (bLwLogFound)
    {
        // if allocated a lwlog from ProtoBuffer, free it here.
        bLwLogFound = LW_FALSE;

        // Loop freeing the LwLog data
        for (i = 0; i < countof(lwld.pBuffers); i++)
        {
            // Free next LwLog buffer
            free(lwld.pBuffers[i]);
        }
    }
}

/*!
 * @brief This function is called from the normal lwlogInit when we have a mini
 *      dump (OCA data only).  It copies the lwld block that was created in extractLwLog
 *      over the normal lwld that would be used for lwlog from memory.
 *
 * @param[in] pointer to lwld (lwlog decoder).
 *
 * @return LW_TRUE when a lwlog from Proto Buffer is available
 *         LW_FALSE when no lwlog was found in the Proto Buffer, or we failed to
 *               load the lwlog data from the Proto Buffer.
 */
LwBool lwlogInitFromOca(LWLD_Decoder *pLwld)
{
    if (bLwLogFound)
    {
        memcpy(pLwld, &lwld, sizeof(LWLD_Decoder));
        return LW_TRUE;
    }
    else
    {
        dprintf("Error unable to find lwLog in the Proto Buffer.\n");
        return LW_FALSE;
    }
}

/*!
 * @brief This function is called from extractRegsAndMem when it finds a Proto Buffer
 *         field of type LWLOG_LOGGERINFO.  It reads all fields from the Proto Buffer and builds a lwld
 *         structure and buffers to be used by the normal lwlog code.
 *
 * @param[in] pointer to the LWLOG_LOGGERINFO Proto Buffer message.
 *
 * @return LW_TRUE when lwlog is successfully extracted from the Proto Buffer.
 *         LW_FALSE if any error is encountered extracting the lwlog from the Proto Buffer.
 */
static LwBool
extractLwLog(const PRB_MSG *pMsg)
{
    const PRB_FIELD *pField        = NULL;
    const PRB_FIELD *pBufferField  = NULL;
    LwU8            *pTags         = NULL;
    unsigned int    i              = 0;

    if (bLwLogFound)
    {
        dprintf("Warning extractLwLog called twice to extract lwlog.  Possible duplicated data in Proto Buffer.\n");
        return LW_TRUE;
    }
    // Init the lwlog data from the Proto Buffer
    memset(&lwld, 0, sizeof(LWLD_Decoder));
    bLwLogFound = LW_FALSE;

    // Get the lwlog version.
    pField = prbGetField(pMsg, LWLOG_LOGGERINFO_VERSION);
    if ((pField == NULL) || (pField->count != 1))
        goto error_exit;
    lwld.version = pField->values->uint32;

    pField = prbGetField(pMsg, LWLOG_LOGGERINFO_PRINTFLAGS);
    if ((pField == NULL) || (pField->count != 1))
        goto error_exit;
    lwld.printFlags = pField->values->uint32;

    pField = prbGetField(pMsg, LWLOG_LOGGERINFO_SIGNATURE);
    if ((pField == NULL) || (pField->count != 1) || sizeof(LWLOG_DB_SIGNATURE) != pField->values->bytes.len)
        goto error_exit;
    memcpy(&lwld.signature, pField->values->bytes.data, sizeof(LWLOG_DB_SIGNATURE));

    pField = prbGetField(pMsg, LWLOG_LOGGERINFO_PRINTBUFFERS);
    if ((pField == NULL) || (pField->count != 1) || sizeof(lwld.printBuffers) != pField->values->bytes.len)
        goto error_exit;
    memcpy(&lwld.printBuffers, pField->values->bytes.data, sizeof(lwld.printBuffers));

    pField = prbGetField(pMsg, LWLOG_LOGGERINFO_RUNTIMESIZES);
    if ((pField == NULL) || (pField->count != 1) || sizeof(lwld.runtimeSizes) != pField->values->bytes.len)
        goto error_exit;
    memcpy(&lwld.runtimeSizes, pField->values->bytes.data, sizeof(lwld.runtimeSizes));

    pField = prbGetField(pMsg, LWLOG_LOGGERINFO_TAGS);
    if ((pField == NULL) || (pField->count != 1) || pField->values->bytes.len != LWLOG_MAX_BUFFERS*sizeof(LwU32))
        goto error_exit;
    pTags = pField->values->bytes.data;

    pBufferField = prbGetField(pMsg, LWLOG_LOGGERINFO_PBUFFERS);
    if ((pBufferField == NULL))
        goto error_exit;

    // Loop allocating and copying the LwLog buffers
    for (i = 0; i < pBufferField->count; i++)
    {
        // Allocate and copy the next LwLog buffer
        lwld.pBuffers[i] = (LWLOG_BUFFER *)malloc(pBufferField->values[i].bytes.len);
        if (lwld.pBuffers[i] == NULL)
        {
            dprintf("Error allocating lwLog.\n");
            return LW_FALSE;
        }
        memcpy(lwld.pBuffers[i], pBufferField->values[i].bytes.data, pBufferField->values[i].bytes.len);
    }
    bLwLogFound = LW_TRUE;
    return LW_TRUE;

error_exit:
    dprintf("Error reading lwlog fields from the ProtoBuffer.\n");
    return LW_FALSE;
}

#endif // LW_WINDOWS

static void
extractRegsAndMem(const PRB_MSG *pMsg)
{
    LwU32 i = 0;
    LwU32 j = 0;
    LwU32 memType;
    const PRB_FIELD *pField = NULL;

    if (pMsg->desc == REGS_REGSANDMEM)
    {
        pField = prbGetField(pMsg, REGS_REGSANDMEM_TYPE);
        memType = pField->values->enum_;

        pField = prbGetField(pMsg, REGS_REGSANDMEM_OFFSET);
        if (pField->count == 0)
            dprintf("lw: Missing RegsAndMem.offset field. Ignoring.\n");
        else
        {
            switch (memType)
            {
                case REGS_REGSANDMEM_GPU_REGS:
                    addRangesToMemSpace(&memSpaces[MEMORY_SPACE_REGS], pMsg);
                    break;
                case REGS_REGSANDMEM_INSTANCE:
                    addRangesToMemSpace(&memSpaces[MEMORY_SPACE_FB], pMsg);
                    break;
                default:
                    addRangesToMemSpace(&memSpaces[MEMORY_SPACE_UNKNOWN], pMsg);
                    break;
            }
        }
    }
#if defined(LW_WINDOWS)
    else if (pMsg->desc == LWLOG_LOGGERINFO)
    {
        if (!extractLwLog(pMsg))
        {
            dprintf("Error: Unable to extract lwLog from the Proto Buffer\n");
            dprintf("       lwlog commands will not be available.\n");
        }
    }
#endif // LW_WINDOWS
    else
    {
        for (i = 0; i < pMsg->desc->num_fields; ++i)
        {
            if (pMsg->fields[i].desc->opts.typ == PRB_MESSAGE)
            {
                for (j = 0; j < pMsg->fields[i].count; ++j)
                {
                    extractRegsAndMem((PRB_MSG *)pMsg->fields[i].
                                      values[j].message.data);
                }
            }
        }
    }
}

static void
destroyMemSpace(MEMORY_SPACE *pSpace)
{
    LwU32 i;

    for (i = 0; i < pSpace->numRanges; ++i)
    {
        clearMemRange(&pSpace->ranges[i]);
    }
    free(pSpace->ranges);
    pSpace->ranges = NULL;
    pSpace->numRanges = 0;

    free(pSpace->needed);
    pSpace->needed = NULL;
    pSpace->numNeeded = 0;
}

static void
printZipContents(const char *filename, LWD_ZIP_HANDLE hZip)
{
    LWD_STATUS status;
    LwU32 numInnerFiles = 0;
    char **innerFiles;
    LwU32 i;

    dprintf("lw: Use one of the following for the 'inner filename' parameter:\n");
    status = lwdZip_ListFiles(hZip, &innerFiles, &numInnerFiles);
    if (status != LWD_OK)
    {
        dprintf("lw: Could not list inner files in zip file %s.\n", filename);
        return;
    }

    for (i = 0; i < numInnerFiles; i++)
    {
        dprintf("lw: \t %s\n", innerFiles[i]);
    }

    lwdZip_ReleaseFileList(innerFiles, numInnerFiles);
}

static LW_STATUS
openDumpFile(const char *zipName, const char *innerName)
{
    LW_STATUS status = LW_OK;
    LWD_STATUS lwdStatus;
    PRB_STATUS prbStatus;
    LWD_ZIP_HANDLE hZip;
    LwU32 size = 0;
    void *buffer = NULL;
    LWD_CRYPT_KEY *pKey = &cryptKey;

    // Open up the dump file
    lwdStatus = lwdZip_Open((char *)zipName,
                            NULL,
                            pKey,
                            LWD_ZIP_MODE_READ,
                            &hZip);
    if (lwdStatus != LWD_OK)
    {
        dprintf("lw: Could not open zip file '%s'\n", zipName);
        status = LW_ERR_GENERIC;
        goto done;
    }

    lwdStatus = lwdZip_ExtractFile(hZip, innerName, &buffer, &size);
    if (lwdStatus != LWD_OK)
    {
        dprintf("lw: Could not extract inner file '%s'.\n", innerName);
        printZipContents(zipName, hZip);
        status = LW_ERR_GENERIC;
        goto abort;
    }

    // Decode the message
    prbStatus = prbDecodeMsg(&dumpMsg, buffer, size);
    if (prbStatus != PRB_OK)
    {
        dprintf("lw: Could not decode file '%s' in zip file %s.\n", innerName,
                zipName);
        status = LW_ERR_GENERIC;
        goto abort;
    }

    // Print file info
    dprintf("lw: Loaded file: %s \t Size: %d\n", innerName, size);

abort:
    //
    // Close zip file (this will also close the inner file first, if it's open)
    // Even if this doesn't succeed, try to free the buffer if it exists
    //
    lwdStatus = lwdZip_Close(hZip);
    if (lwdStatus != LWD_OK)
    {
        dprintf("lw: Could not close zip file %s.\n", zipName);
        status = LW_ERR_GENERIC;
    }

    free(buffer);

done:
    return status;
}

#if defined(LW_WINDOWS)  &&  !defined(LW_MODS)

// Redirect protobuf output to lwwatch output.
int lwdPrintfWrapper(const char *szFmt, ...)
{
    char szBuf[1024];
    va_list list;
    int nRet = 0;

    va_start(list, szFmt);
    nRet = vsnprintf(szBuf, LW_ARRAY_ELEMENTS(szBuf), szFmt, list);
    va_end(list);

    if (nRet > 0)
        dprintf("lw: prbdec: %s", szBuf);

    return nRet;
}

static LW_STATUS
openProtoBuffer()
{
    LW_STATUS status = LW_OK;
    PRB_STATUS prbStatus;
    LwU32      size = 0;
    void      *pBuffer = NULL;

    // Set the printf funtion pointer that lwDump uses.
    lwdPrintf = lwdPrintfWrapper;

    // Init from Win OCA Proto Buffer
    pBuffer = findAndLoadOcaProtoBuffer(&size);
    if (pBuffer == NULL)
    {
        dprintf("lw: Could not find OCA Proto Buffer in lwrrently loaded dump\n");
        status = LW_ERR_GENERIC;
        goto done;
    }

    // Decode the message
    prbStatus = prbDecodeMsg(&dumpMsg, pBuffer, size);
    if (prbStatus != PRB_OK)
    {
        dprintf("lw: Could not decode Proto Buffer.\n");
        status = LW_ERR_GENERIC;
        goto abort;
    }

    // Print file info
    dprintf("lw: Loaded diag registers from Proto Buffer.\n");

abort:
    free(pBuffer);

done:
    return status;
}
#endif // LW_WINDOWS

LW_STATUS
getRiscvCoreDumpFromProtobuf(LwU8* buffer, LwU32 bufferSize, LwU32 *actualSize)
{
    const PRB_FIELD *pField;
    const PRB_MSG *pMsg;
    LwU32 i = 0;

    if (dumpMsg.fields == NULL)
    {
        dprintf("lw: Proto Buffer extraction failed.\n");
        return LW_ERR_GENERIC;
    }

    pField = prbGetField(&dumpMsg, LWDEBUG_LWDUMP_GPU_INFO);
    if (pField == NULL || pField->count == 0 || pField->values == NULL)
    {
        dprintf("lw: Dump message missing GpuInfo field.\n");
        return LW_ERR_GENERIC;
    }

    pMsg = (const PRB_MSG*) pField->values->message.data;
    pField = prbGetField(pMsg, LWDEBUG_GPUINFO_ENG_RTOS_FLCN);
    if (pField == NULL || pField->count == 0 || pField->values == NULL)
    {
        dprintf("lw: Dump message missing GpuInfo.RtosFlcn field.\n");
        return LW_ERR_GENERIC;
    }

    pMsg = (const PRB_MSG*) pField->values->message.data;
    pField = prbGetField(pMsg, LWDEBUG_ENG_RTOSFLCN_CORE_DUMP);
    if (pField == NULL || pField->count == 0 || pField->values == NULL)
    {
        dprintf("lw: Dump message missing RtosFlcn.CoreDump field.\n");
        return LW_ERR_GENERIC;
    }

    if (pField->values->bytes.len > bufferSize)
    {
        dprintf("Error: Can't read core dump, size is too big (%d bytes).\n", pField->values->bytes.len);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    memcpy(buffer, pField->values->bytes.data, pField->values->bytes.len);
    *actualSize = pField->values->bytes.len;

    return LW_OK;
}

LW_STATUS
dumpModeInit(const char *zipName, const char *innerName)
{
    LW_STATUS status = LW_OK;
    LwU32  i = 0;
    LwU32  j = 0;
    LwU32  k = 0;
    LwU32  l = 0;
    LwU32  tmpValue = 0;
    LwU32  pmcBoot0 = 0;
    LwBool readPmcBoot0 = LW_TRUE;
    LwBool bEngMcFound = LW_FALSE;
    const PRB_FIELD *pFieldGpuInfo;
    const PRB_FIELD *pField;
    const PRB_MSG *pMsg;
    const PRB_MSG *pMsg2;
    const char *innerFile;

    if ((innerName == NULL) || (*innerName == '\0'))
    {
        innerFile = "rm_00.pb";
    }
    else
    {
        innerFile = innerName;
    }

    // cleanup previously loaded dump
    if (isInitialized)
    {
        prbDestroyMsg(&dumpMsg);
        for (i = 0; i < MEMORY_SPACE_COUNT; ++i)
            destroyMemSpace(&memSpaces[i]);
    }
    else
    {
        memset((void *)memSpaces, 0, sizeof(memSpaces));
        isInitialized = TRUE;
    }

    // create a new dump message
    prbCreateMsg(&dumpMsg, LWDEBUG_LWDUMP);

#if !defined(LW_WINDOWS) // lwdump
    // try to read in the dump file
    dprintf("lw: Attempting to extract inner file ...\n");

    status = openDumpFile(zipName, innerFile);
    if (status != LW_OK)
    {
        dprintf("lw: Inner file extraction failed.\n");
        return status;
    }
#else // Win OCA
    // try to read in the dump file
    dprintf("lw: Attempting to extract Proto Buffer ...\n");

    status = openProtoBuffer();
    if (status != LW_OK)
    {
        dprintf("lw: Proto Buffer extraction failed.  Returned %d\n", status);
        return status;
    }
#endif  // Win OCA

    // extract general GPU info and BAR offsets
    pFieldGpuInfo = prbGetField(&dumpMsg, LWDEBUG_LWDUMP_GPU_INFO);
    if (pFieldGpuInfo->count == 0)
    {
        dprintf("lw: Dump message missing GpuInfo field.\n");
        return LW_ERR_GENERIC;
    }

    for (j = 0; !bEngMcFound && j < pFieldGpuInfo->count; j++)
    {
        pMsg = (const PRB_MSG*)pFieldGpuInfo->values[j].message.data;
        pField = prbGetField(pMsg, LWDEBUG_GPUINFO_ENG_MC);
        if(pField->count > 0)
        {
            // The first EngMc message contains the bar info
            pMsg = (const PRB_MSG*)pField->values->message.data;
            pField = prbGetField(pMsg, LWDEBUG_ENG_MC_RM_DATA);
            if (pField->count == 1)
            {
                pMsg2 = (const PRB_MSG*)pField->values->message.data;
                pField = prbGetField(pMsg2, LWDEBUG_ENG_MC_RMDATA_PMCBOOT0);
                if (pField->count == 1)
                {
                    pmcBoot0 = pField->values->uint32;
                    readPmcBoot0 = LW_FALSE;
                }
            }
            pField = prbGetField(pMsg, LWDEBUG_ENG_MC_PCI_BARS);
            for (i = 0; i < pField->count; ++i)
            {
                pMsg2 = (const PRB_MSG*)pField->values[i].message.data;
                pField = prbGetField(pMsg2, LWDEBUG_ENG_MC_PCIBARINFO_OFFSET);
                if (pField->count == 1)
                {
                    if (i == 0)
                        lwBar0 = pField->values->uint32;
                    else if (i == 1)
                        lwBar1 = pField->values->uint32;
                }
            }

            bEngMcFound = LW_TRUE;
        }
    }

    if (!bEngMcFound)
    {
        dprintf("lw: Dump message missing GpuInfo.EngMc field.\n");
        return LW_ERR_GENERIC;
    }

    // verifify at least BAR0 was included
    if (lwBar0 == 0)
    {
        dprintf("lw: Dump message does not include BAR0 offset.\n");
        return LW_ERR_GENERIC;
    }

    // extract raw memory spaces from dump messages
    dprintf("lw: Extracting memory ranges...\n");
    extractRegsAndMem(&dumpMsg);
    for (i = 0; i < MEMORY_SPACE_COUNT; ++i)
    {
        if (memSpaces[i].numRanges > 0)
        {
            memSpaces[i].numRanges = mergeMemRanges(memSpaces[i].ranges,
                                                    memSpaces[i].numRanges);
            if (verboseLevel > 1)
            {
                dprintf("lw: Dump ranges from memory space %i:\n", i);
                for (j = 0; j < memSpaces[i].numRanges; ++j)
                {
                    dprintf("      Offset: 0x%08X     Length: 0x%08X\n",
                            (LwU32)memSpaces[i].ranges[j].offset,
                            (LwU32)memSpaces[i].ranges[j].length);
                    for (k = 0; k < memSpaces[i].ranges[j].length; k+=4)
                    {
                        tmpValue = 0;
                        for (l = 0; l < 4 && (k + l) < memSpaces[i].ranges[j].length; l++)
                        {
                            tmpValue |= memSpaces[i].ranges[j].buffer[k+l] << (l * 8);
                        }
                        dprintf("            + 0x%08X  :  0x%08X\n", k, tmpValue);
                    }
                    dprintf("\n");
                }
            }
        }
    }

    if (readPmcBoot0)
    {
        pmcBoot0 = GPU_REG_RD32(LW_PMC_BOOT_0);
    }

    // hook up HAL routines
    verboseLevel = 1;
    status = initLwWatchHal(pmcBoot0);

    return status;
}

LW_STATUS
dumpModePrint(char *prbFieldName)
{
    const PRB_MSG* pMsg;
    const PRB_FIELD *pField;
    char *next;
    LwU32 index;

    if (*prbFieldName == '\0')
    {
        dprintf("LwDump *\n");
        prbPrintMsgOutline(&dumpMsg, 0);
    }
    else if (strcmp(prbFieldName, "*") == 0)
    {
        dprintf("LwDump\n");
        prbPrintMsg(&dumpMsg, 0);
    }
    else
    {
        pMsg = &dumpMsg;

        do
        {
            if (pMsg == NULL)
                return LW_ERR_GENERIC;
            next = prbFieldName;
            index = 0;
            while ((*next != '.') && (*next != '\0'))
            {
                if (*next == '[')
                {
                    *next = '\0';
                    ++next;
                    index = strtoul(next, &next, 0);
                    if (*next != ']')
                        return LW_ERR_GENERIC;
                    ++next;
                    break;
                }
                ++next;
            }

            if (*next == '.')
            {
                *next = '\0';
                ++next;
                if (*next == '\0')
                    return LW_ERR_GENERIC;
            }

            pField = prbGetFieldByName(pMsg, prbFieldName);
            if (pField == NULL)
                return LW_ERR_GENERIC;
            if (index >= pField->count)
                return LW_ERR_GENERIC;

            if (pField->desc->opts.typ == PRB_MESSAGE)
                pMsg = (const PRB_MSG *)pField->values[index].message.data;
            else
                pMsg = NULL;

            prbFieldName = next;

        } while (*prbFieldName != '\0');

        if (*(prbFieldName - 1) == ']')
            prbPrintField(pField, &index, 0);
        else
            prbPrintField(pField, NULL, 0);
    }
    return LW_OK;
}

LW_STATUS
dumpModeFeedback(const char *filename)
{
    LwU32 i;
    LwU32 j;
    FILE *pFile;
    LwU32 count = 0;
    char  szRegisterName[128];

    pFile = fopen(filename, "a");
    if (pFile == NULL)
    {
        dprintf("lw: Could not open dump feedback file '%s'\n", filename);
        return LW_ERR_GENERIC;
    }

    for (i = 0; i < MEMORY_SPACE_COUNT; ++i)
    {
        if (memSpaces[i].numNeeded > 0)
        {
            ++count;
            memSpaces[i].numNeeded = mergeMemRanges(memSpaces[i].needed,
                                                    memSpaces[i].numNeeded);

            for (j = 0; j < memSpaces[i].numNeeded; ++j)
            {
                szRegisterName[0] = 0;

                if (i == MEMORY_SPACE_REGS)
                {
                    getManualRegName((LwU32)memSpaces[i].needed[j].offset,
                        szRegisterName,
                        LW_ARRAY_ELEMENTS(szRegisterName));
                }

                if (szRegisterName[0] != 0)
                {
                    fprintf(pFile, "%i %08X %08X # %s\n", i,
                            (LwU32)memSpaces[i].needed[j].offset,
                            memSpaces[i].needed[j].length,
                            szRegisterName);
                }
                else
                {
                    fprintf(pFile, "%i %08X %08X #\n", i,
                            (LwU32)memSpaces[i].needed[j].offset,
                            memSpaces[i].needed[j].length);
                }
            }
        }
    }

    if (count == 0)
    {
        dprintf("lw: All memory ranges have been available since the last "
                "!lw.dumpinit\n");
    }

    dprintf("lw: Dump feedback appended to file '%s'\n", filename);
    fclose(pFile);
    return LW_OK;
}

LW_STATUS
dumpModeReadFb(LwU64 offset, LwU32 length, LwU8 size)
{
    char *tbuffer;
    LwU32 i = 0;

    /* Make sure that this is on a 32-bit boundary */
    offset &= ~(LwU64)3;
    length = (length + 3) & ~(LwU32)3;

    tbuffer = (char*)malloc(length);
    if (tbuffer == NULL)
        return LW_ERR_GENERIC;

    for (i = 0; i < length; i += 4)
        *(LwU32*)(tbuffer+i) = FB_RD32((LwU32)(offset + i));

    printBuffer(tbuffer, length, offset, 4);
    free(tbuffer);

    return LW_OK;
}

LwU32
REG_RD32_DUMP(PhysAddr offset)
{
    LwU32 value = 0;
    memSpaceRead08(&memSpaces[MEMORY_SPACE_REGS],
                   offset+0, ((LwU8*)&value)+0);
    memSpaceRead08(&memSpaces[MEMORY_SPACE_REGS],
                   offset+1, ((LwU8*)&value)+1);
    memSpaceRead08(&memSpaces[MEMORY_SPACE_REGS],
                   offset+2, ((LwU8*)&value)+2);
    memSpaceRead08(&memSpaces[MEMORY_SPACE_REGS],
                   offset+3, ((LwU8*)&value)+3);
    return value;
}

LwU32
FB_RD32_DUMP(LwU32 offset)
{
    LwU32 value = 0;
    memSpaceRead08(&memSpaces[MEMORY_SPACE_FB],
                   offset+0, ((LwU8*)&value)+0);
    memSpaceRead08(&memSpaces[MEMORY_SPACE_FB],
                   offset+1, ((LwU8*)&value)+1);
    memSpaceRead08(&memSpaces[MEMORY_SPACE_FB],
                   offset+2, ((LwU8*)&value)+2);
    memSpaceRead08(&memSpaces[MEMORY_SPACE_FB],
                   offset+3, ((LwU8*)&value)+3);
    return value;
}

/*!
 * @brief Read/write from/to FB memory.
 *
 * If is_write == 0, reads from FB memory at the given address for the given
 * number of bytes and stores it in the buffer.
 * If is_write != 0, writes to FB memory at the given address for the given
 * number of bytes with the contents of buffer.
 * Note that function doesn't align offset by DWORD.
 *
 * @param[in] offset        LwU64 address in FB memory to read/write.
 * @param[in, out] buffer   void * pointer to buffer,
 *                          function reads/writes into/from this buffer.
 * @param[in] length        LwU32 number of bytes to read/write.
 * @param[in] is_write      LwU32 0 for read, otherwise, write
 *
 * @return LW_OK on success, LW_ERR_GENERIC on failure.
 */
LW_STATUS
fbReadWrite_DUMP(LwU64 offset, void* buffer, LwU32 length, LwU32 is_write)
{
    LwU32  i;

    if (buffer == NULL)
    {
        return LW_ERR_GENERIC;
    }

    if (is_write)
    {
        dprintf("lw: %s: Writing FB is not supported for dump files.\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    for (i = 0; i < length; i++)
    {
        memSpaceRead08(&memSpaces[MEMORY_SPACE_FB],
                       offset + i, ((LwU8*)buffer) + i);
    }

    return LW_OK;
}

#else // LWDEBUG_SUPPORTED

LW_STATUS
dumpModeReadFb(LwU64 offset, LwU32 length, LwU8 size)
{
    dprintf("lw: Dump mode not supported - missing LwDebug support\n");
    return LW_OK;
}

LwU32
REG_RD32_DUMP(PhysAddr offset)
{
    dprintf("lw: Dump mode not supported - missing LwDebug support\n");
    return 0;
}

LwU32
FB_RD32_DUMP(LwU32 offset)
{
    dprintf("lw: Dump mode not supported - missing LwDebug support\n");
    return 0;
}

LW_STATUS
fbReadWrite_DUMP(LwU64 offset, void* buffer, LwU32 length, LwU32 is_write)
{
    dprintf("lw: Dump mode not supported - missing LwDebug support\n");
    return 0;
}

#endif
