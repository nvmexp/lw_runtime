/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2017-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
#ifndef gpuSPEC_INCLUDED
#define gpuSPEC_INCLUDED

#ifndef PROVIDE_OWN_GPUSPEC_DEFINES
#include "gpuSPEC.defs"
#endif

#if !LWCFG(GLOBAL_SAFETY_BUILD)
#if LWCFG(GLOBAL_ARCH_KEPLER)
#if LWCFG(GLOBAL_GPU_FAMILY_GK10X)
#if defined(GPGPUCOMP_DRV_BUILD) || defined(ENABLE_SM30_32)
  DefineProfile     (30, Kepler);

  RegFileSize       (30, 262144);
  RegFileSizePerCta (30, 262144);
  RegAllolwnit      (30, 256);
  RegAlignment      (30, 8);
  MaxCtaPerSM       (30, 16);
  MaxWarps          (30, 64);
  WarpSize          (30, 32);
  WarpAlign         (30, 4);
  MaxRegsPerThread  (30, 63);
  MinRegsPerThread  (30, 32);

#if LWCFG(GLOBAL_CHIP_T124) || LWCFG(GLOBAL_GPU_IMPL_GK20A)
  DefineProfile     (32, Kepler);

  RegFileSize       (32, 262144);
  RegFileSizePerCta (32, 131072);
  RegAllolwnit      (32, 256);
  RegAlignment      (32, 8);
  MaxCtaPerSM       (32, 16);
  MaxWarps          (32, 64);
  WarpSize          (32, 32);
  WarpAlign         (32, 4);
  MaxRegsPerThread  (32, 255);
  MinRegsPerThread  (32, 32);
#endif
#endif // defined(GPGPUCOMP_DRV_BUILD) || defined(ENABLE_SM30_32)

#if LWCFG(GLOBAL_GPU_FAMILY_GK11X)
  DefineProfile     (35, Kepler);

  RegFileSize       (35, 262144);
  RegFileSizePerCta (35, 262144);
  RegAllolwnit      (35, 256);
  RegAlignment      (35, 8);
  MaxCtaPerSM       (35, 16);
  MaxWarps          (35, 64);
  WarpSize          (35, 32);
  WarpAlign         (35, 4);
  MaxRegsPerThread  (35, 255);
  MinRegsPerThread  (35, 32);

#if defined(GPGPUCOMP_DRV_BUILD) || defined(ENABLE_SM30_32)
  CanRun            (35, 30);
#endif
#endif

#if LWCFG(GLOBAL_GPU_IMPL_GK110C)
  DefineProfile     (37, Kepler);

  RegFileSize       (37, 524288);   // sm_37 RF size is 2x sm_35
  RegFileSizePerCta (37, 262144);   // sm_37 RF size per CTA limit is same as sm_35
  RegAllolwnit      (37, 256);
  RegAlignment      (37, 8);
  MaxCtaPerSM       (37, 16);
  MaxWarps          (37, 64);
  WarpSize          (37, 32);
  WarpAlign         (37, 4);
  MaxRegsPerThread  (37, 255);
  MinRegsPerThread  (37, 32);

#if defined(GPGPUCOMP_DRV_BUILD) || defined(ENABLE_SM30_32)
  CanRun            (37, 30);
#endif
  CanRun            (37, 35);
  CanLink           (37, 35);
#endif
#endif
#endif // KEPLER

#if LWCFG(GLOBAL_ARCH_MAXWELL)
  DefineProfile     (50, Maxwell);

  RegFileSize       (50, 262144);
  RegFileSizePerCta (50, 262144);
  RegAllolwnit      (50, 256);
  RegAlignment      (50, 8);
  MaxCtaPerSM       (50, 32);
  MaxWarps          (50, 64);
  WarpSize          (50, 32);
  WarpAlign         (50, 4);
  MaxRegsPerThread  (50, 255);
  MinRegsPerThread  (50, 32);

#if LWCFG(GLOBAL_GPU_FAMILY_GM20X)
  DefineProfile     (52, Maxwell);

  RegFileSize       (52, 262144);
  RegFileSizePerCta (52, 131072); // minimum when CTA-Steering/Caching is ON
  RegAllolwnit      (52, 256);
  RegAlignment      (52, 8);
  MaxCtaPerSM       (52, 32);
  MaxWarps          (52, 64);
  WarpSize          (52, 32);
  WarpAlign         (52, 4);
  MaxRegsPerThread  (52, 255);
  MinRegsPerThread  (52, 32);

  CanRun            (52, 50);
  CanLink           (52, 50);
#endif

#if LWCFG(GLOBAL_CHIP_T210)
  DefineProfile     (53, Maxwell);
  SameRegInfo       (53, 52); // 53 has same register values as 52
#endif
#endif // MAXWELL

#if LWCFG(GLOBAL_ARCH_PASCAL)
  DefineProfile     (60, Pascal);
  SameRegInfo       (60, 50);

#if LWCFG(GLOBAL_GPU_FAMILY_GP10X)
  DefineProfile     (61, Pascal);
  SameRegInfo       (61, 52);

  CanRun            (61, 60);
  CanLink           (61, 60);
#endif

#if LWCFG(GLOBAL_CHIP_T186)
  DefineProfile     (62, Pascal);
  SameRegInfo       (62, 52);
#endif
#endif // PASCAL
#endif // GLOBAL_SAFETY_BUILD

#if LWCFG(GLOBAL_ARCH_VOLTA)
#if !LWCFG(GLOBAL_SAFETY_BUILD)
  DefineProfile     (70, Volta);
  RegFileSize       (70, 262144);
  RegFileSizePerCta (70, 262144);
  RegAllolwnit      (70, 256);
  RegAlignment      (70, 8);
  MaxCtaPerSM       (70, 32);
  MaxWarps          (70, 64);
  WarpSize          (70, 32);
  WarpAlign         (70, 4);
  MaxRegsPerThread  (70, 255);
  MinRegsPerThread  (70, 32);
#endif // GLOBAL_SAFETY_BUILD

#if LWCFG(GLOBAL_CHIP_T194) || LWCFG(GLOBAL_GPU_IMPL_GV11B)
  DefineProfile     (72, Volta);
  RegFileSize       (72, 262144);
  RegFileSizePerCta (72, 262144);
  RegAllolwnit      (72, 256);
  RegAlignment      (72, 8);
  MaxCtaPerSM       (72, 32);
  MaxWarps          (72, 64);
  WarpSize          (72, 32);
  WarpAlign         (72, 4);
  MaxRegsPerThread  (72, 255);
  MinRegsPerThread  (72, 32);
#endif

#endif // VOLTA

#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_GPU_FAMILY_TU10X)
#if LWCFG(GLOBAL_GPU_IMPL_SM73)
#if !LWCFG(GLOBAL_SAFETY_BUILD)
// SM_73
  DefineProfile     (73, Turing);

  RegFileSize       (73, 262144);
  RegFileSizePerCta (73, 262144);
  RegAllolwnit      (73, 256);
  RegAlignment      (73, 8);
  MaxCtaPerSM       (73, 16);
  MaxWarps          (73, 32);
  WarpSize          (73, 32);
  WarpAlign         (73, 4);
  MaxRegsPerThread  (73, 255);
  MinRegsPerThread  (73, 32);

  CanRun            (73, 70);
  CanLink           (73, 70);
#endif // GLOBAL_SAFETY_BUILD
#endif // GLOBAL_GPU_IMPL_SM73

// SM_75
  DefineProfile     (75, Turing);
  // Same info as 73, but safety build doesn't include 73 so repeat values
  // SameRegInfo       (75, 73);
  RegFileSize       (75, 262144);
  RegFileSizePerCta (75, 262144);
  RegAllolwnit      (75, 256);
  RegAlignment      (75, 8);
  MaxCtaPerSM       (75, 16);
  MaxWarps          (75, 32);
  WarpSize          (75, 32);
  WarpAlign         (75, 4);
  MaxRegsPerThread  (75, 255);
  MinRegsPerThread  (75, 32);

#if !LWCFG(GLOBAL_SAFETY_BUILD)
  CanRun            (75, 70);
  CanLink           (75, 70);
#endif
#if LWCFG(GLOBAL_GPU_IMPL_SM73)
#if !LWCFG(GLOBAL_SAFETY_BUILD)
  CanRun            (75, 73);
  CanLink           (75, 73);
#endif // GLOBAL_SAFETY_BUILD
#endif // GLOBAL_GPU_IMPL_SM73
#endif // GLOBAL_GPU_FAMILY_TU10X && TURING

#if !LWCFG(GLOBAL_SAFETY_BUILD)
#if LWCFG(GLOBAL_ARCH_AMPERE)
// SM_80
  DefineProfile     (80, Ampere);

  RegFileSize       (80, 262144);
  RegFileSizePerCta (80, 262144);
  RegAllolwnit      (80, 256);
  RegAlignment      (80, 8);
  MaxCtaPerSM       (80, 32);
  MaxWarps          (80, 64);
  WarpSize          (80, 32);
  WarpAlign         (80, 4);
  MaxRegsPerThread  (80, 255);
  MinRegsPerThread  (80, 32);

  SetFirstInternalMerlwry (80);

#if LWCFG(GLOBAL_GPU_FAMILY_GA10X)
// SM_86
  DefineProfile     (86, Ampere);

  RegFileSize       (86, 262144);
  RegFileSizePerCta (86, 262144);
  RegAllolwnit      (86, 256);
  RegAlignment      (86, 8);
  MaxCtaPerSM       (86, 16);
  MaxWarps          (86, 48);
  WarpSize          (86, 32);
  WarpAlign         (86, 4);
  MaxRegsPerThread  (86, 255);
  MinRegsPerThread  (86, 32);

  CanRun            (86, 80);
  CanLink           (86, 80);

#if LWCFG(GLOBAL_GPU_IMPL_GA10B) || LWCFG(GLOBAL_FAMILY_T23X) 
// SM_87
  DefineProfile     (87, Ampere);

  RegFileSize       (87, 262144);
  RegFileSizePerCta (87, 262144);
  RegAllolwnit      (87, 256);
  RegAlignment      (87, 8);
  MaxCtaPerSM       (87, 16);
  MaxWarps          (87, 48);
  WarpSize          (87, 32);
  WarpAlign         (87, 4);
  MaxRegsPerThread  (87, 255);
  MinRegsPerThread  (87, 32);

#endif // GA10B || T23X

#if LWCFG(GLOBAL_GPU_IMPL_GA10F) || LWCFG(GLOBAL_CHIP_T239) 
// SM_88
  DefineProfile     (88, Ampere);

  RegFileSize       (88, 262144);
  RegFileSizePerCta (88, 262144);
  RegAllolwnit      (88, 256);
  RegAlignment      (88, 8);
  MaxCtaPerSM       (88, 16);
  MaxWarps          (88, 48);
  WarpSize          (88, 32);
  WarpAlign         (88, 4);
  MaxRegsPerThread  (88, 255);
  MinRegsPerThread  (88, 32);

#endif // GA10F || T239
#endif // GA10X
#endif // AMPERE
#endif // GLOBAL_SAFETY_BUILD

#if !LWCFG(GLOBAL_SAFETY_BUILD)
#if LWCFG(GLOBAL_ARCH_ADA)
#if LWCFG(GLOBAL_GPU_FAMILY_AD10X)
// SM_89
// TODO: FIXME: Assigning the same values as sm86 for now.
//              Need to update this with ada specific values
  DefineProfile     (89, Ada);

  RegFileSize       (89, 262144);
  RegFileSizePerCta (89, 262144);
  RegAllolwnit      (89, 256);
  RegAlignment      (89, 8);
  MaxCtaPerSM       (89, 24);
  MaxWarps          (89, 48);
  WarpSize          (89, 32);
  WarpAlign         (89, 4);
  MaxRegsPerThread  (89, 255);
  MinRegsPerThread  (89, 32);

  CanRun            (89, 80);
  CanLink           (89, 80);
  
  CanRun            (89, 86);
  CanLink           (89, 86);

  SetNoMerlwrySupport(89);

  #endif // AD10X 
  #endif // ADA
  #endif // GLOBAL_SAFETY_BUILD

#if !LWCFG(GLOBAL_SAFETY_BUILD)
#if LWCFG(GLOBAL_ARCH_HOPPER)
// SM_90
  DefineProfile     (90, Hopper);

  RegFileSize       (90, 262144);
  RegFileSizePerCta (90, 262144);
  RegAllolwnit      (90, 256);
  RegAlignment      (90, 8);
  MaxCtaPerSM       (90, 32);
  MaxWarps          (90, 64);
  WarpSize          (90, 32);
  WarpAlign         (90, 4);
  MaxRegsPerThread  (90, 255);
  MinRegsPerThread  (90, 32);

  DefineSASSProfile (90);  /* add sass_90 profile */
  SetFirstMerlwry   (90);
  #endif // HOPPER
  #endif // GLOBAL_SAFETY_BUILD

 #define DEFAULT_PROFILE  "sm_52"

#endif
