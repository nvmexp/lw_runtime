/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/******************************************************************************
*
*   Module: threadMigration.cpp
*
*   Description:
*     Simple sample demonstrating multi-GPU/multithread functionality using
*     the LWCA Context Management API.  This API allows the a LWCA context to
*     be associated with a CPU process. A host thread may have only one device
*     context current at a time.
*
*    Refer to the LWCA programming guide 4.5.3.3 on Context Management
*
******************************************************************************/

#define MAXTHREADS 256
#define NUM_INTS 32

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Windows threads use different data structures
#include <windows.h>
DWORD rgdwThreadIds[MAXTHREADS];
HANDLE rghThreads[MAXTHREADS];
CRITICAL_SECTION g_cs;

#define ENTERCRITICALSECTION EnterCriticalSection(&g_cs);
#define LEAVECRITICALSECTION LeaveCriticalSection(&g_cs);
#define STRICMP stricmp
#else

// Includes POSIX thread headers for Linux thread support
#include <pthread.h>
#include <stdint.h>
pthread_t rghThreads[MAXTHREADS];
pthread_mutex_t g_mutex;

#define ENTERCRITICALSECTION pthread_mutex_lock(&g_mutex);
#define LEAVECRITICALSECTION pthread_mutex_unlock(&g_mutex);
#define STRICMP strcasecmp
#endif

#include <stdlib.h>
#include <stdio.h>
#include <lwca.h>
#include <lwda_runtime_api.h>
#include <helper_lwda_drvapi.h>

#include <iostream>
#include <cstring>

using namespace std;

int NumThreads;
int ThreadLaunchCount;

typedef struct _LWDAContext_st {
  LWcontext hlwContext;
  LWmodule hlwModule;
  LWfunction hlwFunction;
  LWdeviceptr dptr;
  int deviceID;
  int threadNum;
} LWDAContext;

LWDAContext g_ThreadParams[MAXTHREADS];

// define input fatbin file
#ifndef FATBIN_FILE
#define FATBIN_FILE "threadMigration_kernel64.fatbin"
#endif

bool gbAutoQuit = false;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);

#define CLEANUP_ON_ERROR(dptr, hlwModule, hlwContext, status) \
  if (dptr) lwMemFree(dptr);                                  \
  if (hlwModule) lwModuleUnload(hlwModule);                   \
  if (hlwContext) lwCtxDestroy(hlwContext);                   \
  return status;

#define THREAD_QUIT  \
  printf("Error\n"); \
  return 0;

// This sample uses the Driver API interface.  The LWCA context needs
// to be setup and the LWCA module (LWBIN) is built by LWCC
static LWresult InitLWDAContext(LWDAContext *pContext, LWdevice hlwDevice,
                                int deviceID, char **argv) {
  LWcontext hlwContext = 0;
  LWmodule hlwModule = 0;
  LWfunction hlwFunction = 0;
  LWdeviceptr dptr = 0;

  // lwCtxCreate: Function works on floating contexts and current context
  LWresult status = lwCtxCreate(&hlwContext, 0, hlwDevice);

  if (LWDA_SUCCESS != status) {
    fprintf(stderr, "lwCtxCreate for <deviceID=%d> failed %d\n", deviceID,
            status);
    CLEANUP_ON_ERROR(dptr, hlwModule, hlwContext, status);
  }

  status = LWDA_ERROR_ILWALID_IMAGE;

  string module_path, ptx_source;
  std::ostringstream fatbin;

  if (!findFatbinPath(FATBIN_FILE, module_path, argv, fatbin)) {
    exit(EXIT_FAILURE);
  } else {
    printf("> initLWDA loading module: <%s>\n", module_path.c_str());
  }

  if (!fatbin.str().size()) {
    printf("fatbin file empty. exiting..\n");
    exit(EXIT_FAILURE);
  }

  // Create module from binary file (FATBIN)
  checkLwdaErrors(lwModuleLoadData(&hlwModule, fatbin.str().c_str()));

  status = lwModuleGetFunction(&hlwFunction, hlwModule, "kernelFunction");

  if (LWDA_SUCCESS != status) {
    fprintf(stderr, "lwModuleGetFunction failed %d\n", status);
    CLEANUP_ON_ERROR(dptr, hlwModule, hlwContext, status);
  }

  // Here we must release the LWCA context from the thread context
  status = lwCtxPopLwrrent(NULL);

  if (LWDA_SUCCESS != status) {
    fprintf(stderr, "lwCtxPopLwrrent failed %d\n", status);
    CLEANUP_ON_ERROR(dptr, hlwModule, hlwContext, status);
  }

  pContext->hlwContext = hlwContext;
  pContext->hlwModule = hlwModule;
  pContext->hlwFunction = hlwFunction;
  pContext->deviceID = deviceID;

  return LWDA_SUCCESS;
}

// ThreadProc launches the LWCA kernel on a LWCA context.
// We have more than one thread that talks to a LWCA context
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
DWORD WINAPI ThreadProc(LWDAContext *pParams)
#else
void *ThreadProc(LWDAContext *pParams)
#endif
{
  int wrong = 0;
  int *pInt = 0;

  printf("<LWCA Device=%d, Context=%p, Thread=%d> - ThreadProc() Launched...\n",
         pParams->deviceID, pParams->hlwContext, pParams->threadNum);

  // lwCtxPushLwrrent: Attach the caller LWCA context to the thread context.
  LWresult status = lwCtxPushLwrrent(pParams->hlwContext);

  if (LWDA_SUCCESS != status) {
    THREAD_QUIT;
  }
  checkLwdaErrors(lwMemAlloc(&pParams->dptr, NUM_INTS * sizeof(int)));

  // There are two ways to launch LWCA kernels via the Driver API.
  // In this LWCA Sample, we illustrate both ways to pass parameters
  // and specify parameters.  By default we use the simpler method.

  if (1) {
    // This is the new LWCA 4.0 API for Kernel Parameter passing and Kernel
    // Launching (simpler method)
    void *args[5] = {&pParams->dptr};

    // new LWCA 4.0 Driver API Kernel launch call
    status = lwLaunchKernel(pParams->hlwFunction, 1, 1, 1, 32, 1, 1, 0, NULL,
                            args, NULL);

    if (LWDA_SUCCESS != status) {
      fprintf(stderr, "lwLaunch failed %d\n", status);
      THREAD_QUIT;
    }
  } else {
    // This is the new LWCA 4.0 API for Kernel Parameter passing and Kernel
    // Launching (advanced method)
    int offset = 0;
    char argBuffer[256];

    // pass in launch parameters (not actually de-referencing LWdeviceptr).
    // LWdeviceptr is storing the value of the parameters
    *((LWdeviceptr *)&argBuffer[offset]) = pParams->dptr;
    offset += sizeof(LWdeviceptr);

    void *kernel_launch_config[5] = {LW_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                                     LW_LAUNCH_PARAM_BUFFER_SIZE, &offset,
                                     LW_LAUNCH_PARAM_END};

    // new LWCA 4.0 Driver API Kernel launch call
    status = lwLaunchKernel(pParams->hlwFunction, 1, 1, 1, 32, 1, 1, 0, 0, NULL,
                            (void **)&kernel_launch_config);

    if (LWDA_SUCCESS != status) {
      fprintf(stderr, "lwLaunch failed %d\n", status);
      THREAD_QUIT;
    }
  }

  pInt = (int *)malloc(NUM_INTS * sizeof(int));

  if (!pInt) return 0;

  if (LWDA_SUCCESS ==
      lwMemcpyDtoH(pInt, pParams->dptr, NUM_INTS * sizeof(int))) {
    for (int i = 0; i < NUM_INTS; i++) {
      if (pInt[i] != 32 - i) {
        printf("<LWCA Device=%d, Context=%p, Thread=%d> error [%d]=%d!\n",
               pParams->deviceID, pParams->hlwContext, pParams->threadNum, i,
               pInt[i]);
        wrong++;
      }
    }

    ENTERCRITICALSECTION

    if (!wrong) ThreadLaunchCount += 1;

    LEAVECRITICALSECTION
  }

  free(pInt);
  fflush(stdout);
  checkLwdaErrors(lwMemFree(pParams->dptr));

  // lwCtxPopLwrrent: Detach the current LWCA context from the calling thread.
  checkLwdaErrors(lwCtxPopLwrrent(NULL));

  printf("<LWCA Device=%d, Context=%p, Thread=%d> - ThreadProc() Finished!\n\n",
         pParams->deviceID, pParams->hlwContext, pParams->threadNum);

  return 0;
}

bool FinalErrorCheck(LWDAContext *pContext, int NumThreads, int deviceCount) {
  if (ThreadLaunchCount != NumThreads * deviceCount) {
    printf("<Expected=%d, Actual=%d> ThreadLaunchCounts(s)\n",
           NumThreads * deviceCount, ThreadLaunchCount);
    return false;
  } else {
    for (int iDevice = 0; iDevice < deviceCount; iDevice++) {
      // lwCtxDestroy called on current context or a floating context
      if (LWDA_SUCCESS != lwCtxDestroy(pContext[iDevice].hlwContext))
        return false;
    }

    return true;
  }
}

int main(int argc, char **argv) {
  printf("Starting threadMigration\n");

  bool bTestResult = runTest(argc, argv);

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

bool runTest(int argc, char **argv) {
  printf("[ threadMigration ] API test...\n");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  InitializeCriticalSection(&g_cs);
#else
  pthread_mutex_init(&g_mutex, NULL);
#endif
  // By default, we will launch 2 LWCA threads for each device
  NumThreads = 2;

  if (argc > 1) {
    // If we are doing the QAtest or automated testing, we quit without
    // prompting
    if (checkCmdLineFlag(argc, (const char **)argv, "qatest") ||
        checkCmdLineFlag(argc, (const char **)argv, "noprompt")) {
      gbAutoQuit = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "numthreads")) {
      NumThreads =
          getCmdLineArgumentInt(argc, (const char **)argv, "numthreads");

      if (NumThreads < 1 || NumThreads > 15) {
        printf(
            "Usage: \"threadMigration -n=<threads>\", <threads> ranges 1-15\n");
        return 1;
      }
    }
  }

  int deviceCount;
  int hlwDevice = 0;
  LWresult status;
  status = lwInit(0);

  if (LWDA_SUCCESS != status) return false;

  status = lwDeviceGetCount(&deviceCount);

  if (LWDA_SUCCESS != status) return false;

  printf("> %d LWCA device(s), %d Thread(s)/device to launched\n\n",
         deviceCount, NumThreads);

  if (deviceCount == 0) {
    return false;
  }

  int ihThread = 0;
  int ThreadIndex = 0;

  LWDAContext *pContext =
      (LWDAContext *)malloc(sizeof(LWDAContext) * deviceCount);

  for (int iDevice = 0; iDevice < deviceCount; iDevice++) {
    char szName[256];
    status = lwDeviceGet(&hlwDevice, iDevice);

    if (LWDA_SUCCESS != status) return false;

    status = lwDeviceGetName(szName, 256, hlwDevice);

    if (LWDA_SUCCESS != status) return false;

    {
      int major = 0, minor = 0;
      checkLwdaErrors(lwDeviceGetAttribute(
          &major, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hlwDevice));
      checkLwdaErrors(lwDeviceGetAttribute(
          &minor, LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hlwDevice));
      int sharedMemPerBlock;
      checkLwdaErrors(lwDeviceGetAttribute(
          &sharedMemPerBlock, LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
          hlwDevice));
      int totalConstantMemory;
      checkLwdaErrors(lwDeviceGetAttribute(
          &totalConstantMemory, LW_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
          hlwDevice));
      int regsPerBlock;
      checkLwdaErrors(lwDeviceGetAttribute(
          &regsPerBlock, LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
          hlwDevice));
      int clockRate;
      checkLwdaErrors(lwDeviceGetAttribute(
          &clockRate, LW_DEVICE_ATTRIBUTE_CLOCK_RATE, hlwDevice));
      printf("Device %d: \"%s\" (Compute %d.%d)\n", iDevice, szName, major,
             minor);
      printf("\tsharedMemPerBlock: %d\n", sharedMemPerBlock);
      printf("\tconstantMemory   : %d\n", totalConstantMemory);
      printf("\tregsPerBlock     : %d\n", regsPerBlock);
      printf("\tclockRate        : %d\n", clockRate);
      printf("\n");
    }

    if (LWDA_SUCCESS !=
        InitLWDAContext(&pContext[iDevice], hlwDevice, iDevice, argv)) {
      return FinalErrorCheck(pContext, NumThreads, deviceCount);
    } else {
      for (int iThread = 0; iThread < NumThreads; iThread++, ihThread++) {
        g_ThreadParams[ThreadIndex].hlwContext = pContext[iDevice].hlwContext;
        g_ThreadParams[ThreadIndex].hlwModule = pContext[iDevice].hlwModule;
        g_ThreadParams[ThreadIndex].hlwFunction = pContext[iDevice].hlwFunction;
        g_ThreadParams[ThreadIndex].deviceID = pContext[iDevice].deviceID;
        g_ThreadParams[ThreadIndex].threadNum = iThread;
        // Launch (NumThreads) for each LWCA context
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        rghThreads[ThreadIndex] = CreateThread(
            NULL, 0, (LPTHREAD_START_ROUTINE)ThreadProc,
            &g_ThreadParams[ThreadIndex], 0, &rgdwThreadIds[ThreadIndex]);
#else  // Assume we are running linux
        pthread_create(&rghThreads[ThreadIndex], NULL,
                       (void *(*)(void *))ThreadProc,
                       &g_ThreadParams[ThreadIndex]);
#endif
        ThreadIndex += 1;
      }
    }
  }

  // Wait until all workers are done
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  WaitForMultipleObjects(ThreadIndex, rghThreads, TRUE, INFINITE);
#else

  for (int i = 0; i < ThreadIndex; i++) {
    pthread_join(rghThreads[i], NULL);
  }

#endif

  bool ret_status = FinalErrorCheck(pContext, NumThreads, deviceCount);
  free(pContext);
  return ret_status;
}
