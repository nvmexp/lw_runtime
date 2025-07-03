/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//

#include "volta/gv100/dev_bus.h"
#include "volta/gv100/dev_mmu.h"
#include "volta/gv100/dev_fb.h"
#include "volta/gv100/dev_ram.h"
#include "volta/gv100/dev_fifo.h"
#include "os.h"
#include "chip.h"
#include "hal.h"
#include "mmu.h"
#include "inst.h"
#include "fb.h"
#include "fifo.h"
#include "print.h"
#include "vmem.h"

#include "hwref/lwutil.h"
#include "class/cl906f.h"

#include "g_instmem_private.h"

LW_STATUS fillVMemSpace_GK104(VMemSpace *pVMemSpace, LwU64 instMemAddr, readFn_t instMemReadFn,
                                writeFn_t writeMemReadFn, MEM_TYPE memType);


/*
 * Returns a formattedMemoryEntry list of RAMFC memory. This list can be
 * printed using PrintFormattedMemory.
 *
 * @param ppOutFormattedMemoryEntry Output variable corresponding to the formatted list.
 * @param pOutNumEntries Output variable corresponding to the number of entries in the list.
 */
void instmemGetRamfcFormattedMemory_GV100(
        formattedMemoryEntry** ppOutFormattedMemoryEntry,
        LwU32* pOutNumEntries
        )
{
    static formattedMemoryEntry ramfc[] =
    {
        {"  LW_RAMFC_GP_PUT                     ",DEVICE_BASE(LW_RAMFC_GP_PUT                   )},
        {"  LW_RAMFC_MEM_OP_A                   ",DEVICE_BASE(LW_RAMFC_MEM_OP_A                 )},
        {"  LW_RAMFC_USERD                      ",DEVICE_BASE(LW_RAMFC_USERD                    )},
        {"  LW_RAMFC_USERD_HI                   ",DEVICE_BASE(LW_RAMFC_USERD_HI                 )},
        {"  LW_RAMFC_SIGNATURE                  ",DEVICE_BASE(LW_RAMFC_SIGNATURE                )},
        {"  LW_RAMFC_GP_GET                     ",DEVICE_BASE(LW_RAMFC_GP_GET                   )},
        {"  LW_RAMFC_PB_GET                     ",DEVICE_BASE(LW_RAMFC_PB_GET                   )},
        {"  LW_RAMFC_PB_GET_HI                  ",DEVICE_BASE(LW_RAMFC_PB_GET_HI                )},
        {"  LW_RAMFC_PB_TOP_LEVEL_GET           ",DEVICE_BASE(LW_RAMFC_PB_TOP_LEVEL_GET         )},
        {"  LW_RAMFC_PB_TOP_LEVEL_GET_HI        ",DEVICE_BASE(LW_RAMFC_PB_TOP_LEVEL_GET_HI      )},
        {"  LW_RAMFC_REF                        ",DEVICE_BASE(LW_RAMFC_REF                      )},
        {"  LW_RAMFC_RUNTIME                    ",DEVICE_BASE(LW_RAMFC_RUNTIME                  )},
        {"  LW_RAMFC_ACQUIRE                    ",DEVICE_BASE(LW_RAMFC_ACQUIRE                  )},
        {"  LW_RAMFC_ACQUIRE_DEADLINE           ",DEVICE_BASE(LW_RAMFC_ACQUIRE_DEADLINE         )},
        {"  LW_RAMFC_SEM_ADDR_HI                ",DEVICE_BASE(LW_RAMFC_SEM_ADDR_HI              )},
        {"  LW_RAMFC_SEM_ADDR_LO                ",DEVICE_BASE(LW_RAMFC_SEM_ADDR_LO              )},
        {"  LW_RAMFC_SEM_PAYLOAD_HI             ",DEVICE_BASE(LW_RAMFC_SEM_PAYLOAD_HI           )},
        {"  LW_RAMFC_SEM_PAYLOAD_LO             ",DEVICE_BASE(LW_RAMFC_SEM_PAYLOAD_LO           )},
        {"  LW_RAMFC_SEM_EXELWTE                ",DEVICE_BASE(LW_RAMFC_SEM_EXELWTE              )},
        {"  LW_RAMFC_GP_BASE                    ",DEVICE_BASE(LW_RAMFC_GP_BASE                  )},
        {"  LW_RAMFC_GP_BASE_HI                 ",DEVICE_BASE(LW_RAMFC_GP_BASE_HI               )},
        {"  LW_RAMFC_GP_FETCH                   ",DEVICE_BASE(LW_RAMFC_GP_FETCH                 )},
        {"  LW_RAMFC_PB_FETCH                   ",DEVICE_BASE(LW_RAMFC_PB_FETCH                 )},
        {"  LW_RAMFC_PB_FETCH_HI                ",DEVICE_BASE(LW_RAMFC_PB_FETCH_HI              )},
        {"  LW_RAMFC_PB_PUT                     ",DEVICE_BASE(LW_RAMFC_PB_PUT                   )},
        {"  LW_RAMFC_PB_PUT_HI                  ",DEVICE_BASE(LW_RAMFC_PB_PUT_HI                )},
        {"  LW_RAMFC_MEM_OP_B                   ",DEVICE_BASE(LW_RAMFC_MEM_OP_B                 )},
        {"  LW_RAMFC_GP_CRC                     ",DEVICE_BASE(LW_RAMFC_GP_CRC                   )},
        {"  LW_RAMFC_SPLITTER_REM_LINES         ",DEVICE_BASE(LW_RAMFC_SPLITTER_REM_LINES       )},
        {"  LW_RAMFC_SPLITTER_OFFSET_IN_LOWER   ",DEVICE_BASE(LW_RAMFC_SPLITTER_OFFSET_IN_LOWER )},
        {"  LW_RAMFC_SPLITTER_OFFSET_IN_UPPER   ",DEVICE_BASE(LW_RAMFC_SPLITTER_OFFSET_IN_UPPER )},
        {"  LW_RAMFC_PB_HEADER                  ",DEVICE_BASE(LW_RAMFC_PB_HEADER                )},
        {"  LW_RAMFC_PB_COUNT                   ",DEVICE_BASE(LW_RAMFC_PB_COUNT                 )},
        {"  LW_RAMFC_PB_DATA0                   ",DEVICE_BASE(LW_RAMFC_PB_DATA0                 )},
        {"  LW_RAMFC_PB_DATA1                   ",DEVICE_BASE(LW_RAMFC_PB_DATA1                 )},
        {"  LW_RAMFC_SUBDEVICE                  ",DEVICE_BASE(LW_RAMFC_SUBDEVICE                )},
        {"  LW_RAMFC_PB_CRC                     ",DEVICE_BASE(LW_RAMFC_PB_CRC                   )},
        {"  LW_RAMFC_MEM_OP_C                   ",DEVICE_BASE(LW_RAMFC_MEM_OP_C                 )},
        {"  LW_RAMFC_RESERVED20                 ",DEVICE_BASE(LW_RAMFC_RESERVED20               )},
        {"  LW_RAMFC_RESERVED21                 ",DEVICE_BASE(LW_RAMFC_RESERVED21               )},
        {"  LW_RAMFC_TARGET                     ",DEVICE_BASE(LW_RAMFC_TARGET                   )},
        {"  LW_RAMFC_METHOD_CRC                 ",DEVICE_BASE(LW_RAMFC_METHOD_CRC               )},
        {"  LW_RAMFC_SPLITTER_REM_PIXELS        ",DEVICE_BASE(LW_RAMFC_SPLITTER_REM_PIXELS      )},
        {"  LW_RAMFC_SPLITTER_OFFSET_OUT_LOWER  ",DEVICE_BASE(LW_RAMFC_SPLITTER_OFFSET_OUT_LOWER )},
        {"  LW_RAMFC_SPLITTER_OFFSET_OUT_UPPER  ",DEVICE_BASE(LW_RAMFC_SPLITTER_OFFSET_OUT_UPPER )},
        {"  LW_RAMFC_METHOD0                    ",DEVICE_BASE(LW_RAMFC_METHOD0                  )},
        {"  LW_RAMFC_DATA0                      ",DEVICE_BASE(LW_RAMFC_DATA0                    )},
        {"  LW_RAMFC_METHOD1                    ",DEVICE_BASE(LW_RAMFC_METHOD1                  )},
        {"  LW_RAMFC_DATA1                      ",DEVICE_BASE(LW_RAMFC_DATA1                    )},
        {"  LW_RAMFC_METHOD2                    ",DEVICE_BASE(LW_RAMFC_METHOD2                  )},
        {"  LW_RAMFC_DATA2                      ",DEVICE_BASE(LW_RAMFC_DATA2                    )},
        {"  LW_RAMFC_METHOD3                    ",DEVICE_BASE(LW_RAMFC_METHOD3                  )},
        {"  LW_RAMFC_DATA3                      ",DEVICE_BASE(LW_RAMFC_DATA3                    )},
        {"  LW_RAMFC_SPARE56                    ",DEVICE_BASE(LW_RAMFC_SPARE56                  )},
        {"  LW_RAMFC_HCE_CTRL                   ",DEVICE_BASE(LW_RAMFC_HCE_CTRL                 )},
        {"  LW_RAMFC_ALLOWED_SYNCPOINTS         ",DEVICE_BASE(LW_RAMFC_ALLOWED_SYNCPOINTS       )},
        {"  LW_RAMFC_GP_PEEK                    ",DEVICE_BASE(LW_RAMFC_GP_PEEK                  )},
        {"  LW_RAMFC_PB_DATA2                   ",DEVICE_BASE(LW_RAMFC_PB_DATA2                 )},
        {"  LW_RAMFC_CONFIG                     ",DEVICE_BASE(LW_RAMFC_CONFIG                   )},
        {"  LW_RAMFC_RUNLIST_TIMESLICE          ",DEVICE_BASE(LW_RAMFC_RUNLIST_TIMESLICE        )},
        {"  LW_RAMFC_SET_CHANNEL_INFO           ",DEVICE_BASE(LW_RAMFC_SET_CHANNEL_INFO         )}
    };

    if ( ppOutFormattedMemoryEntry )
    {
        *ppOutFormattedMemoryEntry = ramfc;
    }

    if ( pOutNumEntries )
    {
        *pOutNumEntries = sizeof(ramfc)/sizeof(formattedMemoryEntry);
    }
};

/*!
 *  Dumps the subctx PDB mem mgmt entries
 *
 *  @param[in] chid - channel idinstance ptr
 *  @param[in] veid - subctx id
 *
 *  @return   void
 */
void instmemGetSubctxPDB_GV100(ChannelId *pChannelId, LwU32 veid)
{
    LwU64       instMemAddr=0;
    readFn_t    instMemReadFn;
    writeFn_t   instMemWriteFn;
    MEM_TYPE    memType;
    VMemSpace   vMemSpace;
    LwU32       buf = 0, val = 0;
    LwU64       pPDE = 0, temp = 0, buffer = 0;

    dprintf("ChId = %d, veid = %d\n", pChannelId->id, veid);

    // Get instance memory address for the channel
    instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId,
                                                                   &instMemReadFn,
                                                                   &instMemWriteFn,
                                                                   &memType);
    // Populate vMemSpace
    if (fillVMemSpace_GK104(&vMemSpace, instMemAddr, instMemReadFn, instMemWriteFn, memType) != LW_OK)
    {
        dprintf("\tFailed to populate vMemSpace\n");
        return;
    }

    // Check if the PDB is valid/filled and subsequently bound by FECS
    instMemReadFn(instMemAddr + SF_OFFSET(LW_RAMIN_SC_PDB_VALID(0)), &buffer, 8);
    dprintf("LW_RAMIN_SC_PDB_VALID = 0x%llx\n", buffer);

    instMemReadFn(instMemAddr + SF_OFFSET(LW_RAMIN_SC_PDB_VALID(veid)), &buffer, 8);
    if (SF_VAL(_RAMIN, _SC_PDB_VALID(veid), buffer) == LW_RAMIN_SC_PDB_VALID_TRUE)
    {
        dprintf("\tRequested PDB is valid\n\n");
    }
    else
    {
        dprintf("\tRequested PDB is invalid.\n\tReturning early...\n");
        return;
    }

    //
    // Memory-Management VEID array
    //
    // The LW_RAMIN_SC_PAGE_DIR_BASE_* entries are an array of page table settings
    // for each subcontext. When a context supports subcontexts, the page table
    // information for a given VEID/Subcontext needs to be filled in or else page
    // faults will result on access.
    //

    // Read the 4 bytes of LW_RAMIN_SC_PAGE_DIR_BASE_*(veid)
    instMemReadFn(instMemAddr + SF_OFFSET(LW_RAMIN_SC_PAGE_DIR_BASE_TARGET(veid)), &buf, 4);
    dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_*(veid) = 0x%08x\n\n", buf);

    // Check if top level page table is in vidmem or sysmem
    val = SF_VAL(_RAMIN, _SC_PAGE_DIR_BASE_TARGET(veid), buf);
    dprintf("\tPDB location : ");
    switch (val)
    {
        case LW_RAMIN_SC_PAGE_DIR_BASE_TARGET_VID_MEM:
                dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_TARGET_VID_MEM\n");
                break;
        case LW_RAMIN_SC_PAGE_DIR_BASE_TARGET_SYS_MEM_COHERENT:
                dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_TARGET_SYS_MEM_COHERENT\n");
                break;
        case LW_RAMIN_SC_PAGE_DIR_BASE_TARGET_SYS_MEM_NONCOHERENT:
                dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_TARGET_SYS_MEM_NONCOHERENT\n");
                break;
        case LW_RAMIN_SC_PAGE_DIR_BASE_TARGET_ILWALID:
                dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_TARGET_ILWALID\n");
                break;
    }

    // Check volatile behavior of top level page table
    val = SF_VAL(_RAMIN, _SC_PAGE_DIR_BASE_VOL(veid), buf);
    dprintf("\tVolatile behavior : ");
    if (val == LW_RAMIN_SC_PAGE_DIR_BASE_VOL_TRUE)
        dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_VOL_TRUE\n");
    else
        dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_VOL_FALSE\n");

    // Check if TEX faults are replayable or not
    val = SF_VAL(_RAMIN, _SC_PAGE_DIR_BASE_FAULT_REPLAY_TEX(veid), buf);
    dprintf("\tTEX faults : ");
    if (val == LW_RAMIN_SC_PAGE_DIR_BASE_FAULT_REPLAY_TEX_ENABLED)
        dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_FAULT_REPLAY_TEX_ENABLED\n");
    else
        dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_FAULT_REPLAY_TEX_DISABLED\n");

    // Check if GCC faults are replayable or not
    val = SF_VAL(_RAMIN, _SC_PAGE_DIR_BASE_FAULT_REPLAY_GCC(veid), buf);
    dprintf("\tGCC faults : ");
    if (val == LW_RAMIN_SC_PAGE_DIR_BASE_FAULT_REPLAY_GCC_ENABLED)
        dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_FAULT_REPLAY_GCC_ENABLED\n");
    else
        dprintf("LW_RAMIN_SC_PAGE_DIR_BASE_FAULT_REPLAY_GCC_DISABLED\n");

    // Check which page table format is in use
    val = SF_VAL(_RAMIN, _SC_USE_VER2_PT_FORMAT(veid), buf);
    dprintf("\tPage table format : ");
    if (val == LW_RAMIN_SC_USE_VER2_PT_FORMAT_TRUE)
        dprintf("LW_RAMIN_SC_USE_VER2_PT_FORMAT_TRUE => Using new format (5-level 49-bit VA format)\n");
    else
        dprintf("LW_RAMIN_SC_USE_VER2_PT_FORMAT_FALSE => Using old format (2-level page table)\n");

    //
    // Check the big page size for all subcontexts
    // Volta supports only 64KB for big pages, selecting 128KB causes UNBOUND_INSTANCE faluts
    //
    val = SF_VAL(_RAMIN, _SC_BIG_PAGE_SIZE(veid), buf);
    dprintf("\tBig page size : ");
    if (val == LW_RAMIN_SC_BIG_PAGE_SIZE_64KB)
        dprintf("LW_RAMIN_SC_BIG_PAGE_SIZE_64KB\n");
    else
        dprintf("Error! %s supports only 64KB for big pages\n", GpuArchitecture());

    // Check the page directory base address for this subcontext
    buf = 0;
    instMemReadFn(instMemAddr + SF_OFFSET(LW_RAMIN_SC_PAGE_DIR_BASE_LO(veid)), &buf, 4);
    pPDE = SF_VAL(_RAMIN, _SC_PAGE_DIR_BASE_LO(veid), buf) << 12; // lower 32 addr bits
    instMemReadFn(instMemAddr + SF_OFFSET(LW_RAMIN_SC_PAGE_DIR_BASE_HI(veid)), &buf, 4);
    temp = SF_VAL(_RAMIN, _SC_PAGE_DIR_BASE_HI(veid), buf);       // higher 32 addr bits
    temp <<= 32;      // these are higher bits of address, so shift accordingly
    pPDE += temp;     // 64bit 4k alligned page dir base address
    dprintf("\tPDB address for subcontext %d = 0x%llx\n", veid, pPDE);

    return;
}
