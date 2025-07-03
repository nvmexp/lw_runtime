/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "os.h"
#include "hal.h"
#include "pmu.h"
#include "elpg.h"
#include "mmu.h"
#include "g_pmu_hal.h"
#include "pgLog.h"

#include "g_elpg_private.h"         // (rmconfig) implementation prototypes


// Function Prototypes
static LwU32 _getLogDmemBase (const char *symName);

//*****************************************************************************
// Global functions
//*****************************************************************************

/**
 * @brief Lwwatch Extension to dump the PG log that resides in the DMEM.
 *
 * Read the content of the PG log and print in a parsed format
 *
 * @param   void
 * @return  void
 **/
void elpgDumpPgLog(void)
{
    LwU32                   logDmemBaseAddr = 0;
    LwU32                   logEntryAddr    = 0;
    LwU32                  *buffer          = NULL;
    LwU32                   headerSize      = 0;
    LwU32                   entrySize       = 0;
    LwU32                   numEntry        = 0;
    LwU32                   length          = 0;
    LwU32                   i               = 0;
    RM_PMU_PG_LOG_HEADER   *pLogHeader      = NULL;
    RM_PMU_PG_LOG_ENTRY    *pLogEntry       = NULL;
    const char             *symName         = "PgLog";

    logDmemBaseAddr = _getLogDmemBase(symName);
    if (logDmemBaseAddr == 0)
    {
        dprintf("lw: Error: Failed to read symbol \"%s\" from DMEM.\n",symName);
        return;
    }

    headerSize =  sizeof(RM_PMU_PG_LOG_HEADER);
    entrySize  =  sizeof(RM_PMU_PG_LOG_ENTRY);

    //
    //Although header and entry are the same size now, this might change in the
    //future. Therefore allocate a buffer big enough for either of them.
    //
    buffer = (LwU32 *)malloc((headerSize > entrySize) ? headerSize : entrySize);
    if (buffer == NULL)
    {
        dprintf("lw: Error: Failed to malloc memory.\n");
        return;
    }

    //Read the header of the log
    length = pPmu[indexGpu].pmuDmemRead(logDmemBaseAddr,
                                        LW_TRUE,
                                        headerSize / sizeof(LwU32),
                                        3,
                                        buffer);
    if (length == 0)
    {
        dprintf("lw: Error: Failed reading log header in DMEM.\n");
        goto exit;
    }

    pLogHeader = (RM_PMU_PG_LOG_HEADER *)buffer;
    numEntry   = pLogHeader->numEvents;
    dprintf("lw: Dumping PG log from 0x%x.\n",logDmemBaseAddr);
    dprintf("PMU_HEADER: <Record Id> <Number of Events> <Put Pointer> <Reserved>\n");
    dprintf("PMU_HEADER:     %-11d %-15d 0x%-11.8x 0x%x\n",pLogHeader->recordId,
                                                           pLogHeader->numEvents,
                                                           pLogHeader->pPut,
                                                           pLogHeader->rsvd);


    //Read all the entries of the log in DMEM
    dprintf("PMU_EVENT: <Event ID> <Engine ID> <Status>   <TimeStamp>\n");
    logEntryAddr = logDmemBaseAddr + entrySize;
    for (i=0; i < numEntry; i++)
    {
        length = pPmu[indexGpu].pmuDmemRead(logEntryAddr,
                                            LW_TRUE,
                                            entrySize / sizeof(LwU32),
                                            3,
                                            buffer);
        if (length == 0)
        {
            dprintf("lw: Error: Failed reading log entry in DMEM.\n");
            goto exit;
        }
        pLogEntry = (RM_PMU_PG_LOG_ENTRY *)buffer;
        dprintf("PMU_EVENT:    %-11d %-7d 0x%.8x 0x%x\n",pLogEntry->eventType,
                                                         pLogEntry->engineId,
                                                         pLogEntry->status,
                                                         pLogEntry->timeStamp);
        logEntryAddr += entrySize;
    }

exit:
    free(buffer);
    return;
}

//*****************************************************************************
// Private functions
//*****************************************************************************

/**
 * @brief A private function used for finding the PG log base address in DMEM
 *
 * @param[in] symName Name of the symbol that contains PG log information
 *
 * @return Address/Offset of the PG log in DMEM.
 **/
static LwU32
_getLogDmemBase
(
    const char *symName
)
{
    PMU_SYM *pSym            = NULL;
    LwU32    count           = 0;
    LwU32    length          = 0;
    LwU32    offset          = 0;
    LwU32    logDmemBaseAddr = 0;
    LwBool   bIgnoreCase     = FALSE;
    LwBool   bExactFound     = FALSE;

    // Find the symbol for the PG log
    pSym = pmuSymFind(symName, bIgnoreCase, (BOOL *)&bExactFound, &count);
    if (pSym != NULL && bExactFound != FALSE && count == 1)
    {
        //Find the PG log DMEM base address in the symbol "PgLog"
        if (pSym->bData)
        {
            offset = (LwU32)(LwUPtr)&(((PG_LOG *)0)->pOffset);
            length = pPmu[indexGpu].pmuDmemRead((pSym->addr) + offset,
                                                LW_TRUE,
                                                sizeof(LwU32) / sizeof(LwU32),
                                                1,
                                                &logDmemBaseAddr);
            if (length == 0)
            {
                dprintf("lw: Error: Failed to read DMEM with the given address 0x%x.\n",
                        (pSym->addr) + offset);
                return 0;
            }
        }
        else
        {
            // code-symbol dumping not supported
            dprintf("lw: Dumping symbols in CODE section not supported. "
                    "Cannot dump <%s>\n", pSym->name);
            return 0;
        }
    }
    else if (count > 1)
    {
        dprintf("lw: Error: More than one symbol found for \"%s\".\n",symName);
        return 0;
    }
    else
    {
        dprintf("lw: Error: Cannot find symbol \"%s\". "
                "Please load symbols from a nm-file using \"pmusym command\".\n",
                symName);
        return 0;
    }

    return logDmemBaseAddr;
}

