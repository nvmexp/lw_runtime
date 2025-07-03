//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// diags.h
//
//*****************************************************

#ifndef _DIAG_H_
#define _DIAG_H_

#include "os.h"

//
// data stucts
//
typedef struct def_diagstruct_struct
{
    LwU32 lw_pfifo_intr_0;
    LwU32 lw_pfifo_intr_en_0;
    LwU32 lw_pmc_intr_0;
    LwU32 lw_pmc_intr_en_0;
    LwU32 lw_pmc_enable;
} LWWATCHDIAGSTRUCT,*PLWWATCHDIAGSTRUCT;

//
// diag routines - diag.c
//
void    diagFillStruct(PLWWATCHDIAGSTRUCT pdiagstruct);
void    diagMaster(PLWWATCHDIAGSTRUCT pdiagstruct);
void    diagFifo(PLWWATCHDIAGSTRUCT pdiagstruct);
void    diagGraphics(PLWWATCHDIAGSTRUCT pdiagstruct, LwU32 grIdx);
void    diagDisplay(PLWWATCHDIAGSTRUCT pdiagstruct);

#endif // _DIAG_H_
