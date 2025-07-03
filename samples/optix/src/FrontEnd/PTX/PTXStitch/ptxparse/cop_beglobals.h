/****************************************************************************\
Copyright (c) 2017, LWPU CORPORATION.

LWPU Corporation("LWPU") supplies this software to you in
consideration of your agreement to the following terms, and your use,
installation, modification or redistribution of this LWPU software
constitutes acceptance of these terms.  If you do not agree with these
terms, please do not use, install, modify or redistribute this LWPU
software.

In consideration of your agreement to abide by the following terms, and
subject to these terms, LWPU grants you a personal, non-exclusive
license, under LWPU's copyrights in this original LWPU software (the
"LWPU Software"), to use, reproduce, modify and redistribute the
LWPU Software, with or without modifications, in source and/or binary
forms; provided that if you redistribute the LWPU Software, you must
retain the copyright notice of LWPU, this notice and the following
text and disclaimers in all such redistributions of the LWPU Software.
Neither the name, trademarks, service marks nor logos of LWPU
Corporation may be used to endorse or promote products derived from the
LWPU Software without specific prior written permission from LWPU.
Except as expressly stated in this notice, no other rights or licenses
express or implied, are granted by LWPU herein, including but not
limited to any patent rights that may be infringed by your derivative
works or by other works in which the LWPU Software may be
incorporated. No hardware is licensed hereunder.

THE LWPU SOFTWARE IS BEING PROVIDED ON AN "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING WITHOUT LIMITATION, WARRANTIES OR CONDITIONS OF TITLE,
NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
ITS USE AND OPERATION EITHER ALONE OR IN COMBINATION WITH OTHER
PRODUCTS.

IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT,
INCIDENTAL, EXEMPLARY, CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, LOST PROFITS; PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) OR ARISING IN ANY WAY
OUT OF THE USE, REPRODUCTION, MODIFICATION AND/OR DISTRIBUTION OF THE
LWPU SOFTWARE, HOWEVER CAUSED AND WHETHER UNDER THEORY OF CONTRACT,
TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY OR OTHERWISE, EVEN IF
LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\****************************************************************************/

//
// cop_beglobals.h - Back-end globals.  Global include for code generators.
//

#if !defined(__BEGLOBALS_H)
#define __BEGLOBALS_H 1

#if defined(LWCFG_ENABLED) || (defined(LW_PARSEASM) && !defined(LW_IN_CGC))
#include "g_lwconfig.h"
#endif

#ifndef NULL
#ifdef __cplusplus
#define NULL    0
#else
#define NULL    ((void *) 0)
#endif
#endif

#include "lwctassert.h"
#define CT_ASSERT(b) ct_assert(b)

// Fix for bug http://lwbugs/514627 :
// 1. Use special offsetof() for Android (off base 256)
// 2. On Android, CT_ASSERT does not support offsetof. Use CT_ASSERT_OFFOF() to those CT_ASSERTS
//    that have offsetof() in the condition. These will be ignored for Android
#if defined(LW_ANDROID)
#define LWOFFSETOF(s,m) (((size_t)(&(((s*)256)->m))) - 256)
#define CT_ASSERT_OFFOF(b)
#else
#define LWOFFSETOF(s,m) offsetof(s,m)
#define CT_ASSERT_OFFOF(b) CT_ASSERT(b)
#endif

// parseasm/cgc/GL/D3D differences

#include "lwString.h"

#if !defined(LW_IN_CGC) && !defined(LW_PARSEASM)
#include "lwogld3d.h"
#endif

#if defined(LW_MACOSX_OPENGL)
#include <string.h>
#endif

#undef COP_PARSEASM_ONLY
#if (defined(LW_PARSEASM) && !(defined(LW_IN_CGC)))
#define COP_PARSEASM_ONLY 1
#endif

#if defined(BINARY_PROFILES_ONLY)
#define DEBUG_STRING(s) ""
#else
#define DEBUG_STRING(s) s
#endif

// The following can be understood as follows:
// First line: For standalone cop_parseasm, enable tracing only for DEBUG version and ENABLE_TRACE_CODE.
// Following lines: If we are not compiling standalone cop_parseasm, 
// turn on tracing if (DEBUG or ENABLE_TRACE_CODE) and one or more of the following
// symbols are defined: LW_PARSEASM, LW_IN_CGC, IS_OPENGL, LW_LDDM_UMODE.
#if ((defined(COP_PARSEASM_ONLY) && defined(ENABLE_TRACE_CODE) && defined(DEBUG)) || \
        ((defined(DEBUG) || defined(ENABLE_TRACE_CODE)) && !defined(COP_PARSEASM_ONLY) && \
        (defined(LW_PARSEASM) || defined(LW_IN_CGC) || defined(IS_OPENGL) || defined(LW_LDDM_UMODE))))
#define SHOW_SCHEDULING 1
#define TRACE_PRINT_DAG 1
#define TRACE_SCHEDULING 1*1
#define TRACE_REG_ALLOCATION 1*1
#define SHOW_STATISTICS 1
#define ADD_DAG_NUMBER_TO_OUTPUT 1*1
#define ADD_COLORS_TO_OUTPUT 1*1
#define ADD_SCHEDULE_TO_OUTPUT 1
#define ADD_LIVE_DEAD_TO_OUTPUT 1*1
#define PRINT_SHADERPERF_CONSTS 1
#define ENABLE_LWIR_VISUALIZATION 1

#else /*((defined(COP_PARSEASM_ONLY) && defined(ENABLE_TRACE_CODE) && defined(DEBUG)) || \
        ((defined(DEBUG) || defined(ENABLE_TRACE_CODE)) && !defined(COP_PARSEASM_ONLY) && \
         (defined(LW_PARSEASM) || defined(LW_IN_CGC) || defined(IS_OPENGL) || defined(LW_LDDM_UMODE)))) */
#define SHOW_SCHEDULING 0
#define TRACE_PRINT_DAG 0
#define TRACE_SCHEDULING 1*0
#define TRACE_REG_ALLOCATION 2*0
#if defined(DEVELOP) // keep LWUC_SECTION_OCG_COMMENTs around
#define SHOW_STATISTICS 1
#else
#define SHOW_STATISTICS 0
#endif
#define ADD_DAG_NUMBER_TO_OUTPUT 0
#define ADD_COLORS_TO_OUTPUT 0
#define ADD_SCHEDULE_TO_OUTPUT 0
#if defined(LW_IN_CGC)
#define ADD_LIVE_DEAD_TO_OUTPUT 0
#define PRINT_SHADERPERF_CONSTS 0
#else
#define ADD_LIVE_DEAD_TO_OUTPUT 1
#define PRINT_SHADERPERF_CONSTS 1
#endif

#endif /*((defined(COP_PARSEASM_ONLY) && defined(ENABLE_TRACE_CODE) && defined(DEBUG)) || \
        (!defined(COP_PARSEASM_ONLY) && (defined(DEBUG) || defined(ENABLE_TRACE_CODE)) && \
         (defined(LW_PARSEASM) || defined(LW_IN_CGC) || defined(IS_OPENGL) || defined(LW_LDDM_UMODE)))) */

#if TRACE_PRINT_DAG || TRACE_REG_ALLOCATION || TRACE_SCHEDULING
#define TRACE_DAGUTIL 1
#else
#define TRACE_DAGUTIL 0
#endif

#define ADD_ANYTHING_TO_OUTPUT       \
    (ADD_DAG_NUMBER_TO_OUTPUT > 0 || \
     ADD_LIVE_DEAD_TO_OUTPUT > 0 ||  \
     ADD_SCHEDULE_TO_OUTPUT > 0 ||   \
     ADD_COLORS_TO_OUTPUT > 0)

#define PRELOAD_INTERLOPANTS 1
#define USE_CMP_OPTIMIZATION 1
#define USE_SCC_OPTIMIZATION 1
#define USE_SCC_OPTIMIZATION_AGGRESSIVE 0
#define USE_PCC_OPTIMIZATION 0
#define USE_EYE_OPTIMIZATION 0

// Temp definitions used during development:

#define ADD_SYNC_INSTRUCTIONS 1

#if !defined(COP_DX9) && !defined(COP_DX1x) && !defined(COP_OGL)
#define COP_DX9 1
#define COP_DX1x 1
#define COP_OGL 1
#else // !defined(COP_DX9) && !defined(COP_DX1x) && !defined(COP_OGL)
#if !defined(COP_DX9)
#define COP_DX9 0
#endif
#if !defined(COP_DX1x)
#define COP_DX1x 0
#endif
#if !defined(COP_OGL)
#define COP_OGL 0
#endif
#endif // !defined(COP_DX9) && !defined(COP_DX1x) && !defined(COP_OGL)

#define COP_INT64_ENABLE                   (COP_DX1x | COP_OGL)
#define COP_INT32_ENABLE                   (COP_DX1x | COP_OGL)
#define COP_DOUBLE_ENABLE                  (COP_DX1x | COP_OGL)
#define COP_COMPUTE_PROFILE_ENABLE         (COP_DX1x)
#define COP_TESSELLATION_PROFILE_ENABLE    (COP_DX1x | COP_OGL)
#define COP_HULL_PROFILE_ENABLE            (COP_DX1x | COP_OGL)
#define COP_OPENGL_ENABLE                  (COP_OGL)
#define COP_SURF_MEMBAR_ENABLE             (COP_DX1x | COP_OGL)
#define COP_SURF_ENABLE                    (COP_DX1x | COP_OGL)

// Disable TRACE_LWIR for DX9 release builds
#if defined(DEBUG) || defined(ENABLE_TRACE_CODE)
#define TRACE_LWIR 1
#define TRACE_STR(s) s
#else
#define TRACE_LWIR 0
#define TRACE_STR(s) ""
#endif

// COP uses this profile specific BIND flag

typedef enum CopSpecificBindingProperties_Enum {
    BIND_SCRATCH  = 0x80000000,
} CopSpecificBindingProperties;

// Code generation headers:

#include "copi_inglobals.h"
#include "cop_types.h"
#include "cop_knobs.h"
#include "cop_parseasm.h"
#include "cop_dag.h"
#include "cop_block.h"
#include "cop_dagutils.h"
#include "cop_cfgutils.h"
#include "cop_set_utils.h"
#include "cop_sdag_utils.h"
#include "cop_transforms.h"
#include "cop_base_schedule.h"
#include "cop_base_peephole.h"
#include "cop_base_flow.h"
#include "cop_base_codegen.h"
#include "cop_livedead.h"
#include "cop_cse.h"
#include "cop_validate.h"
#include "cop_function.h"

#include "utils/cop_temp_dependency.h"
#include "utils/cop_bitvector.h"
#include "utils/cop_bitutils.h"
#include "utils/cop_sparse_set.h"

#if defined(__STDC99__) || defined(__APPLE__) || defined(__GNUC__)
# if defined(DJGPP)
// do nothing, Dos GL Mods neither requires nor has either file
# elif !defined(LW_BSD) && !defined(LW_SUNOS)
#  include <stdint.h>
# else
#  include <inttypes.h>
# endif
#else
# if !(defined(_MSC_VER) && _MSC_VER >= 1300) || defined(LW_WINCE) || defined(LW_RVDS) // defined with newer compilers
#  if defined(LWCPU_IA64) || defined(LWCPU_X86_64)
typedef unsigned long uintptr_t;
typedef long intptr_t;
#  else
typedef unsigned int uintptr_t;
typedef int intptr_t;
#  endif
# endif
#endif

#endif // !defined(__BEGLOBALS_H)
