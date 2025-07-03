// LWIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2016-2020, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
// LWIDIA_COPYRIGHT_END

// Definitions of code information for elf generation

#ifndef _CODEINFO_H_
#define _CODEINFO_H_

#include <stddef.h>
#include "lwtypes.h"
#include "copi_ucode.h"
#include "stdTypes.h"
#include "stdList.h"
#include "SymInfo.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  LWuCode *UCode;
  int Arch;
  Bool IsMerlwry;
  SymInfo *SymHandle;
  // because generators each have own memory management,
  // keep list of objects we allocate internally and free them at end.
  stdList_t AllocedMemory;
} CodeInfo;

/************* Initialization and Deletion **/
// call once to initialize
extern CodeInfo* beginCodeInfo(LWuCode *UCode, int Arch, Bool IsMerlwry, SymInfo *SI);
extern void endCodeInfo(CodeInfo *CI); // call once to cleanup

#ifdef __cplusplus
}
#endif
#endif
