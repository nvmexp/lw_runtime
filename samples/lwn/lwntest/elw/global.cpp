/*
 * Copyright (c) 2006 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"
#include "cmdline.h"

// Global variables and their default values.
//
// Values for many of these can be set via commandline options, which are
// parsed in cmdline.c

int lwnCompileAsFp16Mask = 0;
int lwnDebugEnabled = 0;
int lwnDebugLevel = 4;      // default to using the maximum debug level if enabled
int lwnDebugErrorMessageCount = 0;
int lwnDebugWarningMessageCount = 0;
int lwnDebugInitErrorMessageCount = 0;
int lwnTestVerbose = 0;

int lwnGlslcBinaryMissCount = 0;
int lwnGlslcBinaryCacheApiMismatch = 0;
int lwnGlslcBinaryCacheGpuMismatch = 0;
#if defined(SPIRV_ENABLED)
int lwnGlslcSpirvErrorCount = 0;
int useSpirv = 0;
int logSpirvErrors = 0;
#endif

int noUnloadICD = 0;
int memDebug = 0;
int memDebugBreakAllocNum = -1;
int useDebugTests = 0;
int useSkippedTests = 0;
int useWorkerThreads = 0;
int reportTestTime = 0;
int useGL = 0;
int useGLPresent = 0;
int enableTiledCache = 0;
int lwdaEnabled = 1;
int interactiveMode = 0;
InteractiveTestData interactiveTestData;
int describe = 0;
int makedocs = 0;
int loopCount = 1;
int repeatCount = 1;
int repeatsUseDifferentSeeds = 0;
int crcPrint = 0;
int newlineDelim = 0;
int test_seed = 1;
FILE *md5File = NULL;
char *cmdlineFileBuffer = NULL;
char **cmdlineFileBufferPtrs = NULL;
const char *nameSingleTestRun = NULL;

#if defined(LW_WINDOWS_64)
const char *lwnGlslcDLL = "LwnGlslc.dll";       // Default library built in development environment
#else
const char *lwnGlslcDLL = "LwnGlslc32.dll";     // Default library built in development environment
#endif

int glslang = 0;
int glslangFallbackOnError = 0;
int glslangFallbackOnAbsolute = 0;
int lwnGlasmCacheNumEntries = 0;
int lwnGlslcDebugLevel = -1; // -1 indicates no debug info needed from GLSLC
int lwnGlslcOptLevel = 0;
const char *lwnGlslcInputFile = NULL;
const char *lwnGlslcOutputFile = NULL;
const char *goldDir = NULL;
const char *outputDir = NULL;
const char *missDir = NULL;

const char *resultsFilename = NULL;
const char *genlistFilename = NULL;
const char *includeFilename = NULL;
const char *excludeFilename = NULL;

int lwrrentWindowWidth  = 640;
int lwrrentWindowHeight = 480;

// Global results output file
FILE *results = NULL;

// Default memory allocation sizes (HOS-only).
int hosMallocHeapMB = 512;
int hosGraphicsHeapMB = 32;
int hosDevtoolsHeapMB = 32;
int hosCompilerHeapMB = 64;
int hosFirmwareMemMB = 16;

int queueCommandMemKB = 128;
int queueComputeMemKB = 256;
int queueFlushThresholdKB = 0;

int noZlwll = 0;
