#include "g_lwconfig.h"
/* ---------------------------------------------------------
 * Instruction set classes, plus basic GPU feature definitions:
 */
ISA
#if LWCFG(GLOBAL_ARCH_TESLA) && !STRIP_TESLA_SUPPORT
   SM_1.0  CLASS Tesla
   SM_1.1  > SM_1.0
   SM_1.2  > SM_1.1
   SM_1.3  > SM_1.2
#endif
   
#if LWCFG(GLOBAL_ARCH_FERMI)
   SM_2.0  CLASS Fermi
#endif

#if LWCFG(GLOBAL_ARCH_KEPLER)
   SM_3.0  CLASS Kepler
   SM_3.2  CLASS Kepler
   SM_3.5  > SM_3.0
#if LWCFG(GLOBAL_GPU_IMPL_GK110C)
   SM_3.7  > SM_3.5
#endif
#endif

#if LWCFG(GLOBAL_ARCH_MAXWELL)
   SM_5.0  CLASS Maxwell
#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
   SM_5.2  > SM_5.0
   SM_5.3  > SM_5.2
#endif
#endif

FEATURES
   FLOAT
   DOUBLE > FLOAT

   MATH_FUNCTIONS

   SM_11_ATOMIC_INTRINSICS
   SM_12_ATOMIC_INTRINSICS > SM_11_ATOMIC_INTRINSICS   // shorthand, SM_12 implies SM_11
   SM_13_DOUBLE_INTRINSICS
   
#if LWCFG(GLOBAL_ARCH_FERMI)
   SM_20_INTRINSICS        > SM_13_DOUBLE_INTRINSICS   // shorthand, SM_20 implies SM_13
#endif

#if LWCFG(GLOBAL_ARCH_KEPLER)
   SM_30_INTRINSICS        > SM_20_INTRINSICS
   #if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
      SM_32_INTRINSICS        > SM_30_INTRINSICS
   #endif
   #if LWCFG(GLOBAL_GPU_FAMILY_GK11X)
      #if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
         SM_35_INTRINSICS        > SM_32_INTRINSICS
      #else
         SM_35_INTRINSICS        > SM_30_INTRINSICS
      #endif
      #if LWCFG(GLOBAL_GPU_IMPL_GK110C)
         SM_37_INTRINSICS           > SM_35_INTRINSICS
      #endif
   #endif
#endif

#if LWCFG(GLOBAL_ARCH_MAXWELL)
   #if LWCFG(GLOBAL_GPU_IMPL_GK110C)
      SM_50_INTRINSICS        > SM_37_INTRINSICS
   #else
      SM_50_INTRINSICS        > SM_35_INTRINSICS
   #endif
#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
   SM_52_INTRINSICS        > SM_50_INTRINSICS
   SM_53_INTRINSICS        > SM_52_INTRINSICS
#endif
#endif


/* ---------------------------------------------------------
 * Emulation capabilities of static compiler phases :
 */
FE_EMULATES
   MATH_FUNCTIONS -> {DOUBLE SM_13_DOUBLE_INTRINSICS}  (FUNCTIONALITY) : "-DLWDA_DOUBLE_MATH_FUNCTIONS"
   MATH_FUNCTIONS ->  FLOAT                            (PERFORMANCE  ) : "-DLWDA_FLOAT_MATH_FUNCTIONS"  IFNOT {DOUBLE SM_13_DOUBLE_INTRINSICS}

   SM_11_ATOMIC_INTRINSICS  : "-DLWDA_NO_SM_11_ATOMIC_INTRINSICS"   // 'not implemented'
   SM_12_ATOMIC_INTRINSICS  : "-DLWDA_NO_SM_12_ATOMIC_INTRINSICS"   // 'not implemented'
   SM_13_DOUBLE_INTRINSICS  : "-DLWDA_NO_SM_13_DOUBLE_INTRINSICS"   // 'not implemented'
#if LWCFG(GLOBAL_ARCH_FERMI)
   SM_20_INTRINSICS         : ""
#endif

#if LWCFG(GLOBAL_ARCH_KEPLER)
   SM_30_INTRINSICS         : ""
#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
   SM_32_INTRINSICS         : ""
#endif
#if LWCFG(GLOBAL_GPU_FAMILY_GK11X)
   SM_35_INTRINSICS         : ""
#endif
#if LWCFG(GLOBAL_GPU_IMPL_GK110C)
   SM_37_INTRINSICS         : ""
#endif
#endif

#if LWCFG(GLOBAL_ARCH_MAXWELL)
   SM_50_INTRINSICS         : ""
#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
   SM_52_INTRINSICS         : ""
   SM_53_INTRINSICS         : ""
#endif
#endif


//LWOPENCC_EMULATES
   // -- nothing --

/* ---------------------------------------------------------
 * Definitions of LWPU compute architectures:
 * Note: REG_FILE_SIZE is in bytes
 */

#if LWCFG(GLOBAL_ARCH_TESLA) && !STRIP_TESLA_SUPPORT
//// g80
PROFILE  sm_10
   INTERNAL_NAME sm_10
   ISA           SM_1.0
   CCOPTS        "-D__LWDA_ARCH__=100"
   IMPLEMENTS    FLOAT
   PTX_EMULATES  DOUBLE   -> FLOAT
   REG_FILE_SIZE          32768
   REG_FILE_SIZE_PER_CTA  32768 
   REG_ALLOC_UNIT         256
   REG_ALIGNMENT          1
   MAX_CTA                8
   MAX_WARPS              24
   WARP_SZ                32
   WARP_ALIGN             1   
   MAX_REG_PER_THREAD     124
   MIN_REG_PER_THREAD     0

//// g84 g86
PROFILE  sm_11
   INTERNAL_NAME sm_11
   ISA           SM_1.1
   CCOPTS        "-D__LWDA_ARCH__=110"
   IMPLEMENTS    FLOAT  SM_11_ATOMIC_INTRINSICS
   PTX_EMULATES  DOUBLE   -> FLOAT
   REG_FILE_SIZE          32768
   REG_FILE_SIZE_PER_CTA  32768 
   REG_ALLOC_UNIT         256
   REG_ALIGNMENT          1
   MAX_CTA                8
   MAX_WARPS              24
   WARP_SZ                32
   WARP_ALIGN             1   
   MAX_REG_PER_THREAD     124
   MIN_REG_PER_THREAD     0

////
PROFILE  sm_12
   INTERNAL_NAME sm_12
   ISA           SM_1.2
   CCOPTS        "-D__LWDA_ARCH__=120"
   IMPLEMENTS    FLOAT  SM_12_ATOMIC_INTRINSICS
   PTX_EMULATES  DOUBLE   -> FLOAT
   REG_FILE_SIZE          65536
   REG_FILE_SIZE_PER_CTA  65536 
   REG_ALLOC_UNIT         512
   REG_ALIGNMENT          1
   MAX_CTA                8
   MAX_WARPS              32
   WARP_SZ                32
   WARP_ALIGN             1   
   MAX_REG_PER_THREAD     124
   MIN_REG_PER_THREAD     0

//// gt200
PROFILE  sm_13
   INTERNAL_NAME sm_13
   ISA           SM_1.3
   CCOPTS        "-D__LWDA_ARCH__=130"
   IMPLEMENTS    DOUBLE SM_13_DOUBLE_INTRINSICS SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          65536
   REG_FILE_SIZE_PER_CTA  65536 
   REG_ALLOC_UNIT         512
   REG_ALIGNMENT          1
   MAX_CTA                8
   MAX_WARPS              32
   WARP_SZ                32
   WARP_ALIGN             1
   MAX_REG_PER_THREAD     124
   MIN_REG_PER_THREAD     0

#endif // LWCFG(GLOBAL_ARCH_TESLA)

#if LWCFG(GLOBAL_ARCH_FERMI)
//// Fermi
PROFILE  sm_20
   INTERNAL_NAME sm_20
   ISA           SM_2.0
   CCOPTS        "-D__LWDA_ARCH__=200"
   IMPLEMENTS    DOUBLE SM_20_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          131072
   REG_FILE_SIZE_PER_CTA  131072 
   REG_ALLOC_UNIT         64  
   REG_ALIGNMENT          2
   MAX_CTA                8
   MAX_WARPS              48
   WARP_SZ                32
   WARP_ALIGN             2   
   MAX_REG_PER_THREAD     63
   // NOTE -> MIN_REG_PER_THREAD is actually 20 but use 16, see bugs 1008386 and 1049036
   MIN_REG_PER_THREAD     16


PROFILE  sm_21
   INTERNAL_NAME sm_21
   ISA           SM_2.0
   CCOPTS        "-D__LWDA_ARCH__=210"
   IMPLEMENTS    DOUBLE SM_20_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          131072
   REG_FILE_SIZE_PER_CTA  131072 
   REG_ALLOC_UNIT         64  
   REG_ALIGNMENT          2
   MAX_CTA                8
   MAX_WARPS              48
   WARP_SZ                32
   WARP_ALIGN             2   
   MAX_REG_PER_THREAD     63
   // NOTE -> MIN_REG_PER_THREAD is actually 20 but use 16, see bugs 1008386 and 1049036
   MIN_REG_PER_THREAD     16
#endif // LWCFG(GLOBAL_ARCH_FERMI)

#if LWCFG(GLOBAL_GPU_FAMILY_GK10X)
PROFILE  sm_30
   INTERNAL_NAME sm_30
   ISA           SM_3.0
   CCOPTS        "-D__LWDA_ARCH__=300"
   IMPLEMENTS    DOUBLE SM_30_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          262144
   REG_FILE_SIZE_PER_CTA  262144 
   REG_ALLOC_UNIT         256 
   REG_ALIGNMENT          8
   MAX_CTA                16
   MAX_WARPS              64
   WARP_SZ                32
   WARP_ALIGN             4
   MAX_REG_PER_THREAD     63
   MIN_REG_PER_THREAD     32   
#endif // LWCFG(GLOBAL_GPU_FAMILY_GK10X)

#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
PROFILE  sm_32
   INTERNAL_NAME sm_32
   ISA           SM_3.2
   CCOPTS        "-D__LWDA_ARCH__=320"
   IMPLEMENTS    DOUBLE SM_32_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          262144
   REG_FILE_SIZE_PER_CTA  262144 
   REG_ALLOC_UNIT         256 
   REG_ALIGNMENT          8
   MAX_CTA                16
   MAX_WARPS              64
   WARP_SZ                32
   WARP_ALIGN             4
   MAX_REG_PER_THREAD     255
   MIN_REG_PER_THREAD     32   
#endif // LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A)

#if LWCFG(GLOBAL_GPU_FAMILY_GK11X)
PROFILE  sm_35
   INTERNAL_NAME sm_35
   ISA           SM_3.5
   CCOPTS        "-D__LWDA_ARCH__=350"
   IMPLEMENTS    DOUBLE SM_35_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          262144
   REG_FILE_SIZE_PER_CTA  262144 
   REG_ALLOC_UNIT         256       
   REG_ALIGNMENT          8         
   MAX_CTA                16        
   MAX_WARPS              64        
   WARP_SZ                32        
   WARP_ALIGN             4         
   MAX_REG_PER_THREAD     255       
   MIN_REG_PER_THREAD     32        
#endif

#if LWCFG(GLOBAL_GPU_IMPL_GK110C)
PROFILE  sm_37
   INTERNAL_NAME sm_37
   ISA           SM_3.7
   CCOPTS        "-D__LWDA_ARCH__=370"
   IMPLEMENTS    DOUBLE SM_37_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          524288    // sm_37 RF size  is 2x that to sm_35
   REG_FILE_SIZE_PER_CTA  262144    // sm_37 RF size per CTA limit is same as that of sm_35
   REG_ALLOC_UNIT         256
   REG_ALIGNMENT          8
   MAX_CTA                16
   MAX_WARPS              64
   WARP_SZ                32
   WARP_ALIGN             4
   MAX_REG_PER_THREAD     255
   MIN_REG_PER_THREAD     32
#endif

#if LWCFG(GLOBAL_ARCH_MAXWELL)
//// Maxwell
PROFILE  sm_50
   INTERNAL_NAME sm_50
   ISA           SM_5.0
   CCOPTS        "-D__LWDA_ARCH__=500"
   IMPLEMENTS    DOUBLE SM_50_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          262144
   REG_FILE_SIZE_PER_CTA  262144 
   REG_ALLOC_UNIT         256
   REG_ALIGNMENT          8
   MAX_CTA                16
   MAX_WARPS              64
   WARP_SZ                32
   WARP_ALIGN             4
   MAX_REG_PER_THREAD     255
   MIN_REG_PER_THREAD     32

#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
PROFILE  sm_52
   INTERNAL_NAME sm_52
   ISA           SM_5.2
   CCOPTS        "-D__LWDA_ARCH__=520"
   IMPLEMENTS    DOUBLE SM_52_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          262144
   REG_FILE_SIZE_PER_CTA  262144 
   REG_ALLOC_UNIT         256
   REG_ALIGNMENT          8
   MAX_CTA                16
   MAX_WARPS              64
   WARP_SZ                32
   WARP_ALIGN             4
   MAX_REG_PER_THREAD     255
   MIN_REG_PER_THREAD     32

PROFILE  sm_53
   INTERNAL_NAME sm_53
   ISA           SM_5.3
   CCOPTS        "-D__LWDA_ARCH__=530"
   IMPLEMENTS    DOUBLE SM_53_INTRINSICS        SM_12_ATOMIC_INTRINSICS
   REG_FILE_SIZE          262144     // FIXME: Update correct values for sm_53
   REG_FILE_SIZE_PER_CTA  262144     // FIXME: Update correct values for sm_53
   REG_ALLOC_UNIT         256        // FIXME: Update correct values for sm_53
   REG_ALIGNMENT          8          // FIXME: Update correct values for sm_53
   MAX_CTA                16         // FIXME: Update correct values for sm_53
   MAX_WARPS              64         // FIXME: Update correct values for sm_53
   WARP_SZ                32         // FIXME: Update correct values for sm_53
   WARP_ALIGN             4          // FIXME: Update correct values for sm_53
   MAX_REG_PER_THREAD     255        // FIXME: Update correct values for sm_53
   MIN_REG_PER_THREAD     32         // FIXME: Update correct values for sm_53   
#endif // LWCFG(GLOBAL_GPU_FAMILY_GM20X)
#endif // LWCFG(GLOBAL_ARCH_MAXWELL)


/* ---------------------------------------------------------
 * Definitions of LWPU virtual compute interfaces.
 * These interfaces correspond with 'ptx versions':
 */
#if LWCFG(GLOBAL_ARCH_TESLA) && !STRIP_TESLA_SUPPORT
PTX PROFILE compute_10
   ISA           SM_1.0
   CCOPTS        "-D__LWDA_ARCH__=100"
   IMPLEMENTS    FLOAT
   PTX_EMULATES  DOUBLE -> FLOAT

PTX PROFILE compute_11
   ISA           SM_1.0
   CCOPTS        "-D__LWDA_ARCH__=110"
   IMPLEMENTS    FLOAT  SM_11_ATOMIC_INTRINSICS
   PTX_EMULATES  DOUBLE -> FLOAT

PTX PROFILE compute_12
   ISA           SM_1.0
   CCOPTS        "-D__LWDA_ARCH__=120"
   IMPLEMENTS    FLOAT  SM_12_ATOMIC_INTRINSICS
   PTX_EMULATES  DOUBLE -> FLOAT

PTX PROFILE compute_13
   ISA           SM_1.0
   CCOPTS        "-D__LWDA_ARCH__=130"
   IMPLEMENTS    DOUBLE SM_13_DOUBLE_INTRINSICS SM_12_ATOMIC_INTRINSICS
#endif

#if LWCFG(GLOBAL_ARCH_FERMI)
PTX PROFILE compute_20
   ISA           SM_2.0
   CCOPTS        "-D__LWDA_ARCH__=200"
   IMPLEMENTS    DOUBLE SM_20_INTRINSICS        SM_12_ATOMIC_INTRINSICS
#endif

#if LWCFG(GLOBAL_GPU_FAMILY_GK10X)
PTX PROFILE compute_30
   ISA           SM_3.0
   CCOPTS        "-D__LWDA_ARCH__=300"
   IMPLEMENTS    DOUBLE SM_30_INTRINSICS        SM_12_ATOMIC_INTRINSICS
#endif // LWCFG(GLOBAL_GPU_FAMILY_GK10X)

#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A) 
PTX PROFILE compute_32
   ISA           SM_3.2
   CCOPTS        "-D__LWDA_ARCH__=320"
   IMPLEMENTS    DOUBLE SM_32_INTRINSICS        SM_12_ATOMIC_INTRINSICS
#endif // LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A)
   
#if LWCFG(GLOBAL_GPU_FAMILY_GK11X)
PTX PROFILE compute_35
   ISA           SM_3.5
   CCOPTS        "-D__LWDA_ARCH__=350"
   IMPLEMENTS    DOUBLE SM_35_INTRINSICS        SM_12_ATOMIC_INTRINSICS
#endif // LWCFG(GLOBAL_GPU_FAMILY_GK11X)

#if LWCFG(GLOBAL_GPU_IMPL_GK110C)
PTX PROFILE compute_37
   ISA           SM_3.7
   CCOPTS        "-D__LWDA_ARCH__=370"
   IMPLEMENTS    DOUBLE SM_37_INTRINSICS        SM_12_ATOMIC_INTRINSICS
#endif

#if LWCFG(GLOBAL_ARCH_MAXWELL)
PTX PROFILE compute_50
   ISA           SM_5.0
   CCOPTS        "-D__LWDA_ARCH__=500"
   IMPLEMENTS    DOUBLE SM_50_INTRINSICS        SM_12_ATOMIC_INTRINSICS
#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
PTX PROFILE compute_52
   ISA           SM_5.2
   CCOPTS        "-D__LWDA_ARCH__=520"
   IMPLEMENTS    DOUBLE SM_52_INTRINSICS        SM_12_ATOMIC_INTRINSICS

PTX PROFILE compute_53
   ISA           SM_5.3
   CCOPTS        "-D__LWDA_ARCH__=530"
   IMPLEMENTS    DOUBLE SM_53_INTRINSICS        SM_12_ATOMIC_INTRINSICS
#endif // LWCFG(GLOBAL_GPU_FAMILY_GM20X)
#endif // LWCFG(GLOBAL_ARCH_MAXWELL)


/* ---------------------------------------------------------
 * Default profile, to be used by ptxas and lwcc:
 */
#if LWCFG(GLOBAL_ARCH_TESLA) && !STRIP_TESLA_SUPPORT
DEFAULT_PROFILE sm_10
#else
DEFAULT_PROFILE sm_20
#endif
