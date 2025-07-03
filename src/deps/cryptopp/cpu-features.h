/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

// Fake cpu-features.h for Android, our toolchain doesn't have all NDK stuff

#define ANDROID_CPU_FAMILY_ARM 1
#define ANDROID_CPU_FAMILY_ARM64 1

#define ANDROID_CPU_ARM_FEATURE_AES 1
#define ANDROID_CPU_ARM_FEATURE_ARMv7 1
#define ANDROID_CPU_ARM_FEATURE_CRC32 1
#define ANDROID_CPU_ARM_FEATURE_NEON 1
#define ANDROID_CPU_ARM_FEATURE_PMULL 0
#define ANDROID_CPU_ARM_FEATURE_SHA1 0
#define ANDROID_CPU_ARM_FEATURE_SHA2 0
#define ANDROID_CPU_ARM_FEATURE_SHA3 0
#define ANDROID_CPU_ARM_FEATURE_SHA512 0
#define ANDROID_CPU_ARM_FEATURE_SM3 0
#define ANDROID_CPU_ARM_FEATURE_SM4 0

#define ANDROID_CPU_ARM64_FEATURE_AES 1
#define ANDROID_CPU_ARM64_FEATURE_ASIMD 1
#define ANDROID_CPU_ARM64_FEATURE_CRC32 1
#define ANDROID_CPU_ARM64_FEATURE_PMULL 0
#define ANDROID_CPU_ARM64_FEATURE_SHA1 0
#define ANDROID_CPU_ARM64_FEATURE_SHA2 0
#define ANDROID_CPU_ARM64_FEATURE_SHA3 0
#define ANDROID_CPU_ARM64_FEATURE_SHA512 0
#define ANDROID_CPU_ARM64_FEATURE_SM3 0
#define ANDROID_CPU_ARM64_FEATURE_SM4 0

#define android_getCpuFamily() 1
#define android_getCpuFeatures() 1
