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


/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample revisits matrix multiplication with LWCA task. The code of matrix
 * multiplication is exactly the same as in matrixMulDrv sample of this SDK.
 * This sample, however, demonstrates how to link LWCA driver at runtime and
 * how to perform JIT (just-in-time) compilation of LWCA kernel from PTX image,
 * stored in memory.
 *
 * For more details on acquiring auto-generated sources refer README.TXT file
 * in "extras" directory.
 *
 * Unlike LWBLAS, the sample doesn't address high-performance matrix
 * multiplication.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, LWCA
#include "lwda_drvapi_dynlink.h"
#include "helper_lwda_drvapi.h"

// includes, project
#include "matrixMul.h"
#include "matrixMul_kernel_32_ptxdump.h"
#include "matrixMul_kernel_64_ptxdump.h"

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

#if defined _MSC_VER
#pragma warning (disable : 4312)
#endif


////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
LWcontext g_lwContext;
bool noprompt = false;

static const char *sSDKsample = "matrixMulDynlinkJIT (LWCA dynamic linking)";


////////////////////////////////////////////////////////////////////////////////
// Allocates a matrix with random float entries
////////////////////////////////////////////////////////////////////////////////
void randomInit(float *data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = rand() / (float)RAND_MAX;
    }
}

////////////////////////////////////////////////////////////////////////////////
// LWCA driver runtime linking and initialization
////////////////////////////////////////////////////////////////////////////////
LWresult initLWDA(int argc, char **argv, LWfunction *pMatrixMul, int *block_size_out)
{
    LWresult status;
    LWdevice lwDevice;
    LWmodule lwModule;
    LWfunction lwFunction;
    int major, minor, block_size, devID = 0;
    char deviceName[256];

    // link to lwca driver dynamically
    checkLwdaErrors(lwInit(0, __LWDA_API_VERSION));

    // This assumes that the user is attempting to specify a explicit device -device=n
    if (argc > 1)
    {
        bool bFound = false;

        for (int param=0; param < argc; param++)
        {
            if (!strncmp(argv[param], "-device", 7))
            {
                int i=(int)strlen(argv[1]);

                while (argv[1][i] != '=')
                {
                    i--;
                }

                devID = atoi(&argv[1][++i]);
                bFound = true;
            }

            if (bFound)
                break;
        }
    }

    // get lwca-capable device count
    int deviceCount = 0;
    checkLwdaErrors(lwDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "No devices supporting LWCA detected, exiting...\n");
        exit(EXIT_SUCCESS);
    }

    if (devID < 0) devID = 0;

    if (devID > deviceCount -1)
    {
        fprintf(stderr, "initLWDA (Device=%d) invalid GPU device.  %d GPU device(s) detected.\n\n", devID, deviceCount);
        status = LWDA_ERROR_NOT_FOUND;

        lwCtxDestroy(g_lwContext);
        exit(EXIT_FAILURE);
    }

    // pick up device with zero ordinal (default, or devID)
    checkLwdaErrors(lwDeviceGet(&lwDevice, devID));

    // get compute capabilities and the devicename
    checkLwdaErrors(lwDeviceComputeCapability(&major, &minor, lwDevice));
    checkLwdaErrors(lwDeviceGetName(deviceName, 256, lwDevice));
    printf("> Device %d: \"%s\" with Compute %d.%d capability\n", lwDevice, deviceName, major, minor);

    block_size = 32;
    *block_size_out = block_size;

    // create context for picked device
    status = lwCtxCreate(&g_lwContext, 0, lwDevice);

    if (LWDA_SUCCESS != status)
    {
        lwCtxDestroy(g_lwContext);
        exit(EXIT_SUCCESS);
    }

    // setup JIT compilation options and perform compilation
    {
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 3;
        LWjit_option *jitOptions = new LWjit_option[jitNumOptions];
        void **jitOptVals = new void *[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = LW_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = LW_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2] = LW_JIT_MAX_REGISTERS;
        int jitRegCount = 32;
        jitOptVals[2] = (void *)(size_t)jitRegCount;

        // compile with set parameters
        printf("> Compiling LWCA module\n");

#if defined(_WIN64) || defined(__LP64__)
        status = lwModuleLoadDataEx(&lwModule, matrixMul_kernel_64_ptxdump, jitNumOptions, jitOptions, (void **)jitOptVals);
#else
        status = lwModuleLoadDataEx(&lwModule, matrixMul_kernel_32_ptxdump, jitNumOptions, jitOptions, (void **)jitOptVals);
#endif

        printf("> PTX JIT log:\n%s\n", jitLogBuffer);

        delete [] jitOptions;
        delete [] jitOptVals;
        delete [] jitLogBuffer;
    }

    if (LWDA_SUCCESS != status)
    {
        printf("Error while compiling PTX\n");
        lwCtxDestroy(g_lwContext);
        exit(EXIT_FAILURE);
    }

    // retrieve LWCA function from the compiled module
    status = lwModuleGetFunction(&lwFunction, lwModule,
                                 (block_size == 16) ? "matrixMul_bs16_32bit" : "matrixMul_bs32_32bit");

    if (LWDA_SUCCESS != status)
    {
        lwCtxDestroy(g_lwContext);
        exit(EXIT_FAILURE);
    }

    *pMatrixMul = lwFunction;
    return LWDA_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// Entry point
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[ %s ]\n", sSDKsample);

    // initialize LWCA
    LWfunction matrixMul = NULL;
    int block_size = 0;
    checkLwdaErrors(initLWDA(argc, argv, &matrixMul, &block_size));

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    size_t       size_A = WA * HA;
    size_t       mem_size_A = sizeof(float) * size_A;
    size_t       size_B = WB * HB;
    size_t       mem_size_B = sizeof(float) * size_B;

    float *h_A = (float *) malloc(mem_size_A);
    float *h_B = (float *) malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    LWdeviceptr d_A;
    checkLwdaErrors(lwMemAlloc(&d_A, mem_size_A));
    LWdeviceptr d_B;
    checkLwdaErrors(lwMemAlloc(&d_B, mem_size_B));

    // copy host memory to device
    checkLwdaErrors(lwMemcpyHtoD(d_A, h_A, mem_size_A));
    checkLwdaErrors(lwMemcpyHtoD(d_B, h_B, mem_size_B));

    // allocate device memory for result
    size_t       size_C = WC * HC;
    size_t       mem_size_C = sizeof(float) * size_C;

    LWdeviceptr d_C;
    checkLwdaErrors(lwMemAlloc(&d_C, mem_size_C));

    // allocate mem for the result on host side
    float *h_C = (float *) malloc(mem_size_C);

#if __LWDA_API_VERSION >= 4000
    {
        // This is the new LWCA 4.0 API for Kernel Parameter passing and Kernel Launching (simpler method)
        int Matrix_Width_A = WA;
        int Matrix_Width_B = WB;
        void *args[5] = { &d_C, &d_A, &d_B, &Matrix_Width_A, &Matrix_Width_B };

        checkLwdaErrors(lwLaunchKernel(matrixMul, (WC/block_size), (HC/block_size), 1,
                                       block_size     , block_size     , 1,
                                       0,
                                       NULL, args, NULL));
    }
#else // __LWDA_API_VERSION <= 3020
    {
        // This is the older LWCA Driver API for Kernel Parameter passing and Kernel Launching
        int offset = 0;
        {
            // setup exelwtion parameters
            checkLwdaErrors(lwParamSetv(matrixMul, offset, &d_C, sizeof(d_C)));
            offset += sizeof(d_C);

            checkLwdaErrors(lwParamSetv(matrixMul, offset, &d_A, sizeof(d_A)));
            offset += sizeof(d_A);

            checkLwdaErrors(lwParamSetv(matrixMul, offset, &d_B, sizeof(d_B)));
            offset += sizeof(d_B);
        }

        int Matrix_Width_A = WA;
        int Matrix_Width_B = WB;

        checkLwdaErrors(lwParamSeti(matrixMul, offset, Matrix_Width_A));
        offset += sizeof(Matrix_Width_A);

        checkLwdaErrors(lwParamSeti(matrixMul, offset, Matrix_Width_B));
        offset += sizeof(Matrix_Width_B);

        checkLwdaErrors(lwParamSetSize(matrixMul, offset));
        checkLwdaErrors(lwFuncSetBlockShape(matrixMul, block_size, block_size, 1));
        checkLwdaErrors(lwFuncSetSharedSize(matrixMul, 2*block_size*block_size*sizeof(float)));

        // set exelwtion configuration for the LWCA kernel
        checkLwdaErrors(lwLaunchGrid(matrixMul, WC / block_size, HC / block_size));
    }
#endif

    checkLwdaErrors(lwCtxSynchronize());

    // copy result from device to host
    checkLwdaErrors(lwMemcpyDtoH((void *) h_C, d_C, mem_size_C));

    // compute reference solution
    float *reference = (float *) malloc(mem_size_C);
    computeGold(reference, h_A, h_B, HA, WA, WB);

    // check result
    float diff=0.0f;

    for (unsigned int i=0; i<size_C; i++)
    {
        float tmp = reference[i] - h_C[i];
        diff += tmp*tmp;
    }

    int res = (diff / (float)size_C < 1e-6f);

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    checkLwdaErrors(lwMemFree(d_A));
    checkLwdaErrors(lwMemFree(d_B));
    checkLwdaErrors(lwMemFree(d_C));
    checkLwdaErrors(lwCtxDestroy(g_lwContext));

    printf("Test run %s\n", (1==res) ? "success!" : "failed!");

    exit((1 == res) ? EXIT_SUCCESS : EXIT_FAILURE);
}
