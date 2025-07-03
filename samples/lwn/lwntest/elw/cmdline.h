/*
 * Copyright (c) 2006 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __cmdline_h__
#define __cmdline_h__

#include <stdio.h>

extern int lwnCompileAsFp16Mask;
extern int lwnDebugEnabled;
extern int lwnDebugLevel;
extern int lwnDebugErrorMessageCount;
extern int lwnDebugWarningMessageCount;
extern int lwnDebugInitErrorMessageCount;
extern int lwnTestVerbose;

// If using lwnGlslcInputFile, indicates if a hashed shader in the test could
// not be found in the GLSLC cache binary.
extern int lwnGlslcBinaryMissCount;

// If using lwnGlslcInputFile, indicates if the GLSLC cache binary was generated with
// an incompatible API w.r.t. what LWNtest is compiling with.
extern int lwnGlslcBinaryCacheApiMismatch;

// If using lwnGlslcInputFile, indicates if the GLSLC cache binary was generated with an
// incompatible GPU version w.r.t. what the LWN driver reports.
extern int lwnGlslcBinaryCacheGpuMismatch;

#if defined(SPIRV_ENABLED)
// If using the SPIRV path, indicates if a test's GLSL shaders can't be compiled by glslang to
// SPIR-V.
extern int lwnGlslcSpirvErrorCount;
#endif

extern int noUnloadICD;
extern int memDebug;
extern int memDebugBreakAllocNum;
extern int useDebugTests;
extern int useSkippedTests;
extern int describe; // 1: we output description and do not run tests 
extern int makedocs;
extern int loopCount;
extern int repeatCount;
extern int repeatsUseDifferentSeeds;
extern int crcPrint;
extern int newlineDelim;
extern int test_seed;
extern int useWorkerThreads;
extern int reportTestTime;
extern int useGL;
extern int useGLPresent;
extern int enableTiledCache;
extern int lwdaEnabled;
extern int interactiveMode;
extern FILE * md5File;
extern FILE * crcFile;
extern char * cmdlineFileBuffer;
extern char ** cmdlineFileBufferPtrs;
extern const char * nameSingleTestRun;
extern const char * lwnGlslcDLL;
extern int lwnGlasmCacheNumEntries;
extern int lwnGlslcDebugLevel;
extern int lwnGlslcOptLevel;
extern const char * lwnGlslcOutputFile;
extern const char * lwnGlslcInputFile;
extern const char * goldDir;
extern const char * outputDir;
extern const char * missDir;
extern const char * resultsFilename;
extern const char * genlistFilename;
extern const char * includeFilename;
extern const char * excludeFilename;
extern int hosMallocHeapMB;
extern int hosGraphicsHeapMB;
extern int hosDevtoolsHeapMB;
extern int hosCompilerHeapMB;
extern int hosFirmwareMemMB;
extern int queueCommandMemKB;
extern int queueComputeMemKB;
extern int queueFlushThresholdKB;

extern int noZlwll;

#if defined(SPIRV_ENABLED)
extern int useSpirv;
extern int logSpirvErrors;
#endif

extern int glslang;
extern int glslangFallbackOnError;
extern int glslangFallbackOnAbsolute;

typedef struct InteractiveTestData {
    int     inputPending;       // is there input pending?
    int     testIndex;          // current test being run
    size_t  nameLen;            // number of characters in test name selection
    char    name[1024];         // buffer holding interactive test name selection
    int     testDirection;      // selected test direction (-1 = rewind, +1 = advance, 0 = none)
} InteractiveTestData;
extern InteractiveTestData interactiveTestData;

void ParseCmdLine(int argc, char** argv);

#endif // #ifndef __cmdline_h__
