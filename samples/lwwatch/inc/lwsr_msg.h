
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef LWSR_MSG_H
#define LWSR_MSG_H

#include "lwsr_parsereg.h"

#define LW_ECHO dprintf

void echoLWSRInfoRegs(lwsr_inforeg_fields *);
void echoLWSRCapabilityRegs(lwsr_capreg_fields *);
void echoLWSRControlRegs(lwsr_controlreg_fields *);
void echoLWSRStatusRegs(lwsr_statusreg_fields *, lwsr_controlreg_fields *);
void echoLWSRTimingRegs(lwsr_timingreg_fields *);
void echoLWSRLinkRegs(lwsr_linkreg_fields *);
void echoLWSRBacklightRegs(lwsr_backlightreg_fields *);
void echoLWSRDiagnosticRegs(lwsr_diagnosticreg_fields *);

#endif
