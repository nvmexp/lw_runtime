/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _BIF_H_
#define _BIF_H_

#include "os.h"
#include "hal.h"


#include "g_bif_hal.h"                    // (rmconfig) public interface
typedef LwU32 RM_PMU_BIF_LINK_SPEED;
#define RM_PMU_BIF_LINK_SPEED_ILWALID  (0x00u)
#define RM_PMU_BIF_LINK_SPEED_GEN1PCIE (0x01u)
#define RM_PMU_BIF_LINK_SPEED_GEN2PCIE (0x02u)
#define RM_PMU_BIF_LINK_SPEED_GEN3PCIE (0x03u)
#define RM_PMU_BIF_LINK_SPEED_GEN4PCIE (0x04u)
#define RM_PMU_BIF_LINK_SPEED_GEN5PCIE (0x05u)


#endif // _BIF_H_
