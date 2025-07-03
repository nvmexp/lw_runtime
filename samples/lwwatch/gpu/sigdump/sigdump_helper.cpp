#if defined(SIGDUMP_ENABLE)

#include <iostream>
#include <string>
#include <typeinfo>
#include <map>

extern "C"
{
#include "gr.h"                     // To access the LwU32 and LwBool macros
#include "sigdump.h"
#include "PmlsplitterCApi.h"
#include "sigdump_helper.h"
}

// checkRegWrite: Checks if the value written to the register is the same as
// what is read back from it (POLL_TIMEOUT_COUNT number of reads are made).
//
// ** For GM10x only **
// This function is used as a WAR for the race between selectmux
// serialization write completion and perfmon PRI read to get
// the signal value. See http://lwbugs/1260703.
extern "C" LwBool checkRegWrite (pmcsSigdump *dump, LwU32 i)
{
    LwU32 numLastWriteChecks;
    LwU32 lastRegWriteReadback;

    assert (dump);

    //
    // See if SMC is enabled and then set syspipe index for debug window
    // The function handles duplicate setting fine
    //
    if(pGr[indexGpu].grGetSmcState() && dump->writes[i].syspipeIdx != ILWALID_GR_IDX)
    {
        pGr[indexGpu].grConfigBar0Window(dump->writes[i].syspipeIdx, LW_TRUE);
    }

    for (numLastWriteChecks = 0 ; numLastWriteChecks < POLL_TIMEOUT_COUNT ; numLastWriteChecks++)
    {
        lastRegWriteReadback = GPU_REG_RD32 (dump -> writes[i].address) & dump -> writes[i].mask;
        if ( lastRegWriteReadback == (dump -> writes[i].value & dump -> writes[i].mask) )
        {
            break;
        }
    }

    // Unset BAR0 window if SMC is was enabled
    if(pGr[indexGpu].grGetSmcState() && dump->writes[i].syspipeIdx != ILWALID_GR_IDX)
    {
        pGr[indexGpu].grConfigBar0Window(dump->writes[i].syspipeIdx, LW_FALSE);
    }

    if (numLastWriteChecks == POLL_TIMEOUT_COUNT)
    {
        dprintf ( "WARNING: Invalid write to address 0x%.8x with mask 0x%.8x. "
        "Attempted to write 0x%.8x, but readback 0x%.8x.\n", dump -> writes[i].address,
                                                             dump -> writes[i].mask,
                                                             (dump -> writes[i].value & dump -> writes[i].mask),
                                                             lastRegWriteReadback );
        return FALSE;
    }
    return TRUE;
}

// The map to cache writes.
static std::map<LwU32,LwU32> cachedWrites;

// RegWrite function with write caching logic. Register writes that have already been made won't be repeated.
extern "C" int optimizedRegWriteWrapper(pmcsSigdump *dump, LwU32 i, LwBool optimization, LwBool checkWrites)
{
    LwU32 addr       = dump -> writes[i].address;
    LwU32 value      = dump -> writes[i].value;
    LwU32 mask       = dump -> writes[i].mask;
    LwU32 finalVal   = value & mask;
    int status = REG_WRITE_SUCCESSFUL;

    assert (dump);

    // See if SMC is enabled and then set syspipe index for debug window
    if(pGr[indexGpu].grGetSmcState() && dump->writes[i].syspipeIdx != ILWALID_GR_IDX)
    {
        pGr[indexGpu].grConfigBar0Window(dump->writes[i].syspipeIdx, LW_TRUE);
    }

    if (optimization)
    {
        std::map<LwU32,LwU32>::const_iterator itr = cachedWrites.find(addr);
        if((itr != cachedWrites.end()) && ((itr->second & mask) == finalVal))
        {
            status = REG_WRITE_SKIPPED;
            goto exit;
        }
        RegWrite(addr, value, mask);

        cachedWrites[addr] = (cachedWrites[addr] & ~mask) | finalVal;
    }
    else
    {
       RegWrite(addr, value, mask);
    }
    // If "-checkregwrites" argument is used and the register write was performed
    // in this iteration, then check the status of the write
    // (compare read value with the value that was written).
    if (checkWrites)
    {
        if (checkRegWrite (dump, i) == LW_FALSE)
        {
            status = REG_WRITE_CHECK_FAILED;
            goto exit;
        }
     }

exit:
    // Unset BAR0 window if SMC is was enabled
    if(pGr[indexGpu].grGetSmcState() && dump->writes[i].syspipeIdx != ILWALID_GR_IDX)
    {
        pGr[indexGpu].grConfigBar0Window(dump->writes[i].syspipeIdx, LW_FALSE);
    }

    return status;
}

// Function to clear the cache.
extern "C" void clearRegWriteCache (void)
{
    cachedWrites.clear ();
    return;
}

#endif // defined(SIGDUMP_ENABLE)
