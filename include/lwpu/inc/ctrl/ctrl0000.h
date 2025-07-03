/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2005-2015 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0000.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl0000/ctrl0000system.h"
#include "ctrl0000/ctrl0000gpu.h"
#include "ctrl0000/ctrl0000gpuacct.h"
#include "ctrl0000/ctrl0000gsync.h"
#include "ctrl0000/ctrl0000diag.h"
#include "ctrl0000/ctrl0000event.h"
#include "ctrl0000/ctrl0000lwd.h"
#include "ctrl0000/ctrl0000swinstr.h"
#include "ctrl0000/ctrl0000proc.h"
#include "ctrl0000/ctrl0000syncgpuboost.h"
#include "ctrl0000/ctrl0000gspc.h"
#include "ctrl0000/ctrl0000vgpu.h"
#include "ctrl0000/ctrl0000client.h"

/* include appropriate os-specific command header */
#include "ctrl0000/ctrl0000windows.h"
#include "ctrl0000/ctrl0000unix.h"
