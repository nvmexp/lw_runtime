/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
#ifndef _cl_e44e_h_
#define _cl_e44e_h_

#include "lwtypes.h"
#define T114_CAPTURE_SW                   (0xE44E)

/*
 * This is the SW class for T114 Camera engine that provides rmcontrol
 * APIs to clients
 */

/*
 * Max sub clients of camera for this chip/class.
 */
#define LWE44E_MAX_SUB_CLIENTS    (2)

#endif // ifndef _cl_e44e_h_
