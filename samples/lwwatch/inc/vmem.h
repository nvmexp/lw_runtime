/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2011-2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _VMEM_H_
#define _VMEM_H_

#include "os.h"
#include "hal.h"
#include "fifo.h"

#include "g_vmem_hal.h"                    // (rmconfig) public interface

LW_STATUS   vmemPdeGetByIndex_T30(VMemSpace *pVMemSpace, LwU32 index, GMMU_ENTRY_VALUE *pPde);
LW_STATUS   vmemPdeGetByVa_T30(VMemSpace *pVMemSpace, LwU64 va, PdeEntry *pPde);
LW_STATUS   vmemPteGetByVa_T30(VMemSpace *pVMemSpace, LwU64 va, PdeEntry *pPde, PteEntry *pPte);
LW_STATUS   vmemPteGetByIndex_T30(VMemSpace *pVMemSpace, LwU32 pteIndex, PdeEntry *pPde, PteEntry *pPte);
LW_STATUS   vmemVToP_T30(VMemSpace *pVMemSpace, LwU64 va, LwU64 *pPa, GMMU_APERTURE *pMemDesc);
void        vmemDoVToPDump_T30(VMemSpace *pVMemSpace, LwU64 va);
void        vmemPToV_T30(VMemTypes vMemType, VMEM_INPUT_TYPE *pId, LwU64 physAddr, BOOL vidMem);
LwBool      vmemIsGvpteDeprecated();

#endif // _VMEM_H_
