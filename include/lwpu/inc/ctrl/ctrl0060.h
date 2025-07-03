/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0060.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW0060_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0x0060, LW0060_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LW0060_CTRL_RESERVED       (0x00)
#define LW0060_CTRL_SYNC_GPU_BOOST (0x01)
