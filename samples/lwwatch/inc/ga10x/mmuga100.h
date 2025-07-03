/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// rypanwar@lwpu.com - June 28 2018
// mmuga100.c - page table routines for Ampere
//
//*****************************************************

#ifndef __MMUGA100_H__
#define __MMUGA100_H__


// Includes

#include "ampere/ga100/dev_bus.h"
#include "ampere/ga100/dev_mmu.h"
#include "ampere/ga100/dev_fb.h"
#include "ampere/ga100/dev_ram.h"
#include "ampere/ga100/dev_fifo.h"
#include "ampere/ga100/dev_pwr_pri.h"
#include "os.h"
#include "chip.h"
#include "hal.h"
#include "mmu.h"
#include "inst.h"
#include "fb.h"

#define MMU_PDE_PAGE_SIZE_IS_4KB_BIG_128KB          0
#define MMU_PDE_PAGE_SIZE_IS_4KB_BIG_64KB           1
#define MMU_PDE_PAGE_SIZE_IS_64KB                   2
#define MMU_PDE_PAGE_SIZE_IS_128KB                  3

#define MMU_PDE_INDEX_128KB                         39:27
#define MMU_PTE_INDEX_128KB                         26:17
#define MMU_OFFSET_PAGE_128KB                       16:0

#define MMU_PDE_INDEX_64KB                          39:26
#define MMU_PTE_INDEX_64KB                          25:16
#define MMU_OFFSET_PAGE_64KB                        15:0
#define MMU_PDE_MAP_128KB                           26:0

#define MMU_PDE_INDEX_4KB_BIG_128KB                 39:27
#define MMU_PTE_INDEX_4KB_BIG_128KB                 26:12
#define MMU_PDE_INDEX_4KB_BIG_64KB                  39:26
#define MMU_PTE_INDEX_4KB_BIG_64KB                  25:12
#define MMU_PDE_MAP_64KB                            25:0

#define MMU_OFFSET_PAGE_4KB                         11:0


#define MMU_SIZEOF_BIG_PAGE                 (64 * 1024)
#define MMU_SIZEOF_SMALL_PAGE               (4 * 1024)
#define MMU_SIZEOF_VA_SPACE                 (1 << 40)

//Used for compacting various memory flags into a single word
#define LW_WATCH_MEMORY_DESCRIPTION_APERTURE            1:0
#define LW_WATCH_MEMORY_DESCRIPTION_PEER                3:2

//#define PDE_MAP(v, size)      ((size == LW_PFB_PRI_MMU_CTRL_VM_PG_SIZE_128KB) ? S_VAL64(MMU_PDE_MAP_128KB, v) : S_VAL64(MMU_PDE_MAP_64KB, v))
#define PDE_MAP(v, size)      ((size ==  FERMI_BIG_PAGESIZE_128K)? S_VAL64(MMU_PDE_MAP_128KB, v) : S_VAL64(MMU_PDE_MAP_64KB, v))

#define SIZEOF_VA_SPACE_PER_PDE_MAP(bigpagesz)  (bigpagesz * 1024)
#define PDE_INDEX(bigpagesz, v)                 ((v) / SIZEOF_VA_SPACE_PER_PDE_MAP(bigpagesz))
#define PTE_INDEX(bigpagesz, v, pagesz)         ((v) % SIZEOF_VA_SPACE_PER_PDE_MAP(bigpagesz) / (pagesz))
#define PAGE_OFFSET(v, pagesz)                  ((v) % (pagesz))
#define PAGETABLE_INDEX_SMALL   0
#define PAGETABLE_INDEX_BIG     1
#define PAGETABLE_COUNT         2

#endif  // __MMUGA100_H__
