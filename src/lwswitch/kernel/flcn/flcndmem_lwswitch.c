/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   flcndmem_lwswitch.c
 * @brief  FLCN Data-Memory Manager
 *
 * This module is intended to serve as the primary interface between all upper-
 * level Falcon-object layers and the HAL-layer. It provides APIs for accessing
 * the Falcon DMEM (read and write) as well as managing all allocations in the
 * RM-managed region of the Falcon DMEM.
 *
 * DMEM allocations are satisfied out of a carved-out portion of the Falcon
 * DMEM. The location of this region is determined when the Falcon image is
 * built and is communicated to the RM from the Falcon via the INIT message
 * that the Falcon sends upon initialization. Therefore, allocations cannot be
 * satisfied until this message arrives (oclwrs immediately after STATE_LOAD).
 */

/* ------------------------ Includes --------------------------------------- */
#include "flcn/flcn_lwswitch.h"
#include "common_lwswitch.h"

/* ------------------------ Static Function Prototypes --------------------- */

/* ------------------------ Globals ---------------------------------------- */
/* ------------------------ Public Functions  ------------------------------ */
/* ------------------------ Private Static Functions ----------------------- */
