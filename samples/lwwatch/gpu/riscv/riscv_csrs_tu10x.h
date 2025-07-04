/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RISCV_CSRS_H
#define _RISCV_CSRS_H

#include "riscv_prv.h"

#include "turing/tu102/dev_gsp_riscv_csr_64.h"

#define MK_CSR(X) { #X, LW_RISCV_CSR_##X }

static const struct NamedCsr _csrs[] =
{
    MK_CSR(MISA),
    MK_CSR(MVENDORID),
    MK_CSR(MARCHID),
    MK_CSR(MIMPID),
    MK_CSR(MHARTID),
    MK_CSR(MSTATUS),
    MK_CSR(MTVEC),
    MK_CSR(MIP),
    MK_CSR(MIE),
    MK_CSR(MTIME),
    MK_CSR(MCYCLE),
    MK_CSR(MINSTRET),
    MK_CSR(MHPMCOUNTER3),
    MK_CSR(MHPMCOUNTER4),
    MK_CSR(MHPMCOUNTER5),
    MK_CSR(MHPMCOUNTER6),
    MK_CSR(MHPMCOUNTER7),
    MK_CSR(MHPMCOUNTER8),
    MK_CSR(MHPMCOUNTER9),
    MK_CSR(MHPMCOUNTER10),
    MK_CSR(MHPMCOUNTER11),
    MK_CSR(MHPMCOUNTER12),
    MK_CSR(MHPMCOUNTER13),
    MK_CSR(MHPMCOUNTER14),
    MK_CSR(MHPMCOUNTER15),
    MK_CSR(MHPMCOUNTER16),
    MK_CSR(MHPMCOUNTER17),
    MK_CSR(MHPMCOUNTER18),
    MK_CSR(MHPMCOUNTER19),
    MK_CSR(MHPMCOUNTER20),
    MK_CSR(MHPMCOUNTER21),
    MK_CSR(MHPMCOUNTER22),
    MK_CSR(MHPMCOUNTER23),
    MK_CSR(MHPMCOUNTER24),
    MK_CSR(MHPMCOUNTER25),
    MK_CSR(MHPMCOUNTER26),
    MK_CSR(MHPMCOUNTER27),
    MK_CSR(MHPMCOUNTER28),
    MK_CSR(MHPMCOUNTER29),
    MK_CSR(MHPMCOUNTER30),
    MK_CSR(MHPMCOUNTER31),
    MK_CSR(MUCOUNTEREN),
    MK_CSR(MHPMEVENT3),
    MK_CSR(MHPMEVENT4),
    MK_CSR(MHPMEVENT5),
    MK_CSR(MHPMEVENT6),
    MK_CSR(MHPMEVENT7),
    MK_CSR(MHPMEVENT8),
    MK_CSR(MHPMEVENT9),
    MK_CSR(MHPMEVENT10),
    MK_CSR(MHPMEVENT11),
    MK_CSR(MHPMEVENT12),
    MK_CSR(MHPMEVENT13),
    MK_CSR(MHPMEVENT14),
    MK_CSR(MHPMEVENT15),
    MK_CSR(MHPMEVENT16),
    MK_CSR(MHPMEVENT17),
    MK_CSR(MHPMEVENT18),
    MK_CSR(MHPMEVENT19),
    MK_CSR(MHPMEVENT20),
    MK_CSR(MHPMEVENT21),
    MK_CSR(MHPMEVENT22),
    MK_CSR(MHPMEVENT23),
    MK_CSR(MHPMEVENT24),
    MK_CSR(MHPMEVENT25),
    MK_CSR(MHPMEVENT26),
    MK_CSR(MHPMEVENT27),
    MK_CSR(MHPMEVENT28),
    MK_CSR(MHPMEVENT29),
    MK_CSR(MHPMEVENT30),
    MK_CSR(MHPMEVENT31),
    MK_CSR(MSCRATCH),
    MK_CSR(MEPC),
    MK_CSR(MCAUSE),
    MK_CSR(MCAUSE2),
    MK_CSR(MBADADDR),
    MK_CSR(UDCACHEOP),
    MK_CSR(UACCESSATTR),
    MK_CSR(MTOHOST),
    MK_CSR(MFROMHOST),
    MK_CSR(MMPUIDX),
    MK_CSR(MMPUVA),
    MK_CSR(MMPUPA),
    MK_CSR(MMPURNG),
    MK_CSR(MMPUATTR),
    MK_CSR(MTIMECMP),
    MK_CSR(MWFE),
    MK_CSR(MCFG),
    MK_CSR(MDCACHEOP),
    MK_CSR(USTATUS),
    MK_CSR(CYCLE),
    MK_CSR(TIME),
    MK_CSR(INSTRET),
    MK_CSR(HPMCOUNTER3),
    MK_CSR(HPMCOUNTER4),
    MK_CSR(HPMCOUNTER5),
    MK_CSR(HPMCOUNTER6),
    MK_CSR(HPMCOUNTER7),
    MK_CSR(HPMCOUNTER8),
    MK_CSR(HPMCOUNTER9),
    MK_CSR(HPMCOUNTER10),
    MK_CSR(HPMCOUNTER11),
    MK_CSR(HPMCOUNTER12),
    MK_CSR(HPMCOUNTER13),
    MK_CSR(HPMCOUNTER14),
    MK_CSR(HPMCOUNTER15),
    MK_CSR(HPMCOUNTER16),
    MK_CSR(HPMCOUNTER17),
    MK_CSR(HPMCOUNTER18),
    MK_CSR(HPMCOUNTER19),
    MK_CSR(HPMCOUNTER20),
    MK_CSR(HPMCOUNTER21),
    MK_CSR(HPMCOUNTER22),
    MK_CSR(HPMCOUNTER23),
    MK_CSR(HPMCOUNTER24),
    MK_CSR(HPMCOUNTER25),
    MK_CSR(HPMCOUNTER26),
    MK_CSR(HPMCOUNTER27),
    MK_CSR(HPMCOUNTER28),
    MK_CSR(HPMCOUNTER29),
    MK_CSR(HPMCOUNTER30),
    MK_CSR(HPMCOUNTER31),
    MK_CSR(MSPM),
    MK_CSR(MRSP),
    MK_CSR(USPM),
    MK_CSR(URSP),
    MK_CSR(TSELECT),
    MK_CSR(TDATA1),
    MK_CSR(TDATA2),
    MK_CSR(TDATA3),
    MK_CSR(DCSR),
    MK_CSR(MBIND),
    MK_CSR(MTLBILWDATA1),
    MK_CSR(MTLBILWOP),
    MK_CSR(MFLUSH),
    MK_CSR(ML2SYSILW),
    MK_CSR(ML2PEERILW),
    MK_CSR(ML2CLNCOMP),
    MK_CSR(ML2FLHDTY),
    MK_CSR(ML2WFSPR),
    MK_CSR(MUSYSOPEN),
    MK_CSR(BIND),
    MK_CSR(TLBILWDATA1),
    MK_CSR(TLBILWOP),
    MK_CSR(FLUSH),
    MK_CSR(L2SYSILW),
    MK_CSR(L2PEERILW),
    MK_CSR(L2CLNCOMP),
    MK_CSR(L2FLHDTY),
    MK_CSR(L2WFSPR),
    MK_CSR(LWFENCEIO),
    MK_CSR(LWFENCEMEM),
    MK_CSR(LWFENCEALL),
    { NULL, 0 }
};

#endif // _RISCV_CSRS_H
