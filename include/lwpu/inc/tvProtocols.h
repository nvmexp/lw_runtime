/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2002 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _TVPROTOCOLS_H_
#define _TVPROTOCOLS_H_
//******************************************************************************
//
// Module Name: tvProtocols.h
//
// This file contains #defines used by ctrl calls, methods and also internally
// in RM. .
//
//******************************************************************************
#include "lwtypes.h"

#define TV_STANDARD_RESERVED0                               (0x00000006)
#define TV_STANDARD_RESERVED1                               (0x00000007)
#define TV_STANDARD_576i                                    (0x00000008)
#define TV_STANDARD_480i                                    (0x00000009)
#define TV_STANDARD_480p                                    (0x0000000A)
#define TV_STANDARD_576p                                    (0x0000000B)
#define TV_STANDARD_720p                                    (0x0000000C)
#define TV_STANDARD_1080i                                   (0x0000000D)
#define TV_STANDARD_1080p                                   (0x0000000E)
#define TV_STANDARD_720p50                                  (0x0000000F)
#define TV_STANDARD_1080p24                                 (0x00000010)
#define TV_STANDARD_1080i50                                 (0x00000011)
#define TV_STANDARD_1080p50                                 (0x00000012)


#define TV_STANDARD_HDFIRSTSTANDARD                         TV_STANDARD_576i

// XXXDISP: Note that TV_STANDARD_LASTSTANDARD is still set to
// TV_STANDARD_1080p not to TV_STANDARD_1080i50. This is for backward
// compatibility. We should get rid of all references to
// TV_STANDARD_LASTSTANDARD as soon as possible.
#define TV_STANDARD_LASTSTANDARD                            TV_STANDARD_1080i50

#define TV_STANDARD_525x60                                  29 /* verboten! internal use only */
#define TV_STANDARD_625x50                                  30 /* verboten! internal use only */
#define TV_STANDARD_HDTV                                    31 /* verboten! internal use only */

#define TV_STANDARD_NTSC_VGA                                0x12
#define TV_STANDARD_480p_861C                               0x13

#endif  // _TVPROTOCOLS_H_
