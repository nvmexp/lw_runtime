 /***************************************************************************\
|*                                                                           *|
|*       Copyright 1993-2015 LWPU, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     NOTICE TO USER:   The source code  is copyrighted under  U.S. and     *|
|*     international laws.  Users and possessors of this source code are     *|
|*     hereby granted a nonexclusive,  royalty-free copyright license to     *|
|*     use this code in individual and commercial software.                  *|
|*                                                                           *|
|*     Any use of this source code must include,  in the user dolwmenta-     *|
|*     tion and  internal comments to the code,  notices to the end user     *|
|*     as follows:                                                           *|
|*                                                                           *|
|*       Copyright 1993-2015 LWPU, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     LWPU, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY     *|
|*     OF  THIS SOURCE  CODE  FOR ANY PURPOSE.  IT IS  PROVIDED  "AS IS"     *|
|*     WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORPOR-     *|
|*     ATION DISCLAIMS ALL WARRANTIES  WITH REGARD  TO THIS SOURCE CODE,     *|
|*     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGE-     *|
|*     MENT,  AND FITNESS  FOR A PARTICULAR PURPOSE.   IN NO EVENT SHALL     *|
|*     LWPU, CORPORATION  BE LIABLE FOR ANY SPECIAL,  INDIRECT,  INCI-     *|
|*     DENTAL, OR CONSEQUENTIAL DAMAGES,  OR ANY DAMAGES  WHATSOEVER RE-     *|
|*     SULTING FROM LOSS OF USE,  DATA OR PROFITS,  WHETHER IN AN ACTION     *|
|*     OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF     *|
|*     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.     *|
|*                                                                           *|
|*     U.S. Government  End  Users.   This source code  is a "commercial     *|
|*     item,"  as that  term is  defined at  48 C.F.R. 2.101 (OCT 1995),     *|
|*     consisting  of "commercial  computer  software"  and  "commercial     *|
|*     computer  software  documentation,"  as such  terms  are  used in     *|
|*     48 C.F.R. 12.212 (SEPT 1995)  and is provided to the U.S. Govern-     *|
|*     ment only as  a commercial end item.   Consistent with  48 C.F.R.     *|
|*     12.212 and  48 C.F.R. 227.7202-1 through  227.7202-4 (JUNE 1995),     *|
|*     all U.S. Government End Users  acquire the source code  with only     *|
|*     those rights set forth herein.                                        *|
|*                                                                           *|
 \***************************************************************************/



 /***************************************************************************\
|*                                                                           *|
|*                         LW Architecture Interface                         *|
|*                                                                           *|
|*  <lw32.h> defines a 32-bit wide naming convention  for the functionality  *|
|*  of LWPU's Unified Media Architecture (TM).                             *|
|*                                                                           *|
 \***************************************************************************/


#ifndef LW32_INCLUDED
#define LW32_INCLUDED
#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"
#include "lwgputypes.h"


 /***************************************************************************\
|*                                LW Classes                                 *|
 \***************************************************************************/

/*
 * The structures indicate the offsets of hardware registers corresponding to
 * the methods of each class.  Since the first 256 bytes of each subchannel is
 * the LwControlPio registers, the hexadecimal offsets in comments start at
 * 0x100.
 */

#include "class/cl0000.h"
#include "class/cl0002.h"
#include "class/cl0004.h"
#include "class/cl0005.h"
#include "class/cl003e.h"
#include "class/cl003f.h"
#include "class/cl0040.h"
#include "class/cl0041.h"

#include "class/cl0070.h"
#include "class/cl0073.h"

#include "class/cl0080.h"

#include "class/cl2080.h"
#include "class/cl5070.h"

 /***************************************************************************\
|*                                 Channels                                  *|
 \***************************************************************************/

#include "class/cl506f.h"
#include "class/cl507a.h"
#include "class/cl507b.h"
#include "class/cl507c.h"
#include "class/cl507d.h"
#include "class/cl507e.h"

#include "class/cl826f.h"
#include "class/cl827c.h"
#include "class/cl827e.h"

#include "class/cl837c.h"
#include "class/cl837e.h"

#include "class/cl857c.h"
#include "class/cl857e.h"

#include "class/cl866f.h"

#include "class/cl906f.h"
#include "class/cla06f.h"
#include "class/clb06f.h"
#include "class/clc36f.h"

#include "ctrl/ctrl0000.h"
#include "ctrl/ctrl0041.h"
#include "ctrl/ctrl0073.h"
#include "ctrl/ctrl0080.h"
#include "ctrl/ctrl2080.h"
#include "ctrl/ctrl5070.h"
#include "ctrl/ctrlc36f.h"

#ifdef __cplusplus
};          // extern "C" {
#endif

#endif /* LW32_INCLUDED */

