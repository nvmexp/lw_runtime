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

#include "lwdla.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscisync.h"

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <unistd.h>

#define DPRINTF(...) printf(__VA_ARGS__)

static void printTensorDesc(lwdlaModuleTensorDescriptor* tensorDesc) {
  DPRINTF("\tTENSOR NAME : %s\n", tensorDesc->name);
  DPRINTF("\tsize: %lu\n", tensorDesc->size);

  DPRINTF("\tdims: [%lu, %lu, %lu, %lu]\n", tensorDesc->n, tensorDesc->c,
          tensorDesc->h, tensorDesc->w);

  DPRINTF("\tdata fmt: %d\n", tensorDesc->dataFormat);
  DPRINTF("\tdata type: %d\n", tensorDesc->dataType);
  DPRINTF("\tdata category: %d\n", tensorDesc->dataCategory);
  DPRINTF("\tpixel fmt: %d\n", tensorDesc->pixelFormat);
  DPRINTF("\tpixel mapping: %d\n", tensorDesc->pixelMapping);
  DPRINTF("\tstride[0]: %d\n", tensorDesc->stride[0]);
  DPRINTF("\tstride[1]: %d\n", tensorDesc->stride[1]);
  DPRINTF("\tstride[2]: %d\n", tensorDesc->stride[2]);
  DPRINTF("\tstride[3]: %d\n", tensorDesc->stride[3]);
}

static int initializeInputBuffers(char* filePath,
                                  lwdlaModuleTensorDescriptor* tensorDesc,
                                  unsigned char* buf) {
  // Read the file in filePath and fill up 'buf' according to format
  // specified by the user.

  return 0;
}

typedef struct {
  lwdlaDevHandle devHandle;
  lwdlaModule moduleHandle;
  unsigned char* loadableData;
  unsigned char* inputBuffer;
  unsigned char* outputBuffer;
  LwSciBufObj inputBufObj;
  LwSciBufObj outputBufObj;
  LwSciBufModule bufModule;
  LwSciBufAttrList inputAttrList;
  LwSciBufAttrList reconciledInputAttrList;
  LwSciBufAttrList inputConflictList;
  LwSciBufAttrList outputAttrList;
  LwSciBufAttrList reconciledOutputAttrList;
  LwSciBufAttrList outputConflictList;
  LwSciSyncObj syncObj1;
  LwSciSyncObj syncObj2;
  LwSciSyncModule syncModule;
  LwSciSyncFence preFence;
  LwSciSyncFence eofFence;
  LwSciSyncCpuWaitContext lwSciCtx;
  LwSciSyncAttrList waiterAttrListObj1;
  LwSciSyncAttrList signalerAttrListObj1;
  LwSciSyncAttrList waiterAttrListObj2;
  LwSciSyncAttrList signalerAttrListObj2;
  LwSciSyncAttrList lwSciSyncConflictListObj1;
  LwSciSyncAttrList lwSciSyncReconciledListObj1;
  LwSciSyncAttrList lwSciSyncConflictListObj2;
  LwSciSyncAttrList lwSciSyncReconciledListObj2;
  lwdlaModuleTensorDescriptor* inputTensorDesc;
  lwdlaModuleTensorDescriptor* outputTensorDesc;
  LwdlaFence* preFences;
  uint64_t** devPtrs;
  lwdlaWaitEvents* waitEvents;
  lwdlaSignalEvents* signalEvents;
} ResourceList;

void cleanUp(ResourceList* resourceList);

void cleanUp(ResourceList* resourceList) {
  if (resourceList->inputBufObj != NULL) {
    LwSciBufObjFree(resourceList->inputBufObj);
    resourceList->inputBufObj = NULL;
  }

  if (resourceList->outputBufObj != NULL) {
    LwSciBufObjFree(resourceList->outputBufObj);
    resourceList->outputBufObj = NULL;
  }

  if (resourceList->reconciledInputAttrList != NULL) {
    LwSciBufAttrListFree(resourceList->reconciledInputAttrList);
    resourceList->reconciledInputAttrList = NULL;
  }

  if (resourceList->inputConflictList != NULL) {
    LwSciBufAttrListFree(resourceList->inputConflictList);
    resourceList->inputConflictList = NULL;
  }

  if (resourceList->inputAttrList != NULL) {
    LwSciBufAttrListFree(resourceList->inputAttrList);
    resourceList->inputAttrList = NULL;
  }

  if (resourceList->reconciledOutputAttrList != NULL) {
    LwSciBufAttrListFree(resourceList->reconciledOutputAttrList);
    resourceList->reconciledOutputAttrList = NULL;
  }

  if (resourceList->outputConflictList != NULL) {
    LwSciBufAttrListFree(resourceList->outputConflictList);
    resourceList->outputConflictList = NULL;
  }

  if (resourceList->outputAttrList != NULL) {
    LwSciBufAttrListFree(resourceList->outputAttrList);
    resourceList->outputAttrList = NULL;
  }

  if (resourceList->bufModule != NULL) {
    LwSciBufModuleClose(resourceList->bufModule);
    resourceList->bufModule = NULL;
  }

  LwSciSyncFenceClear(&(resourceList->preFence));
  LwSciSyncFenceClear(&(resourceList->eofFence));

  if (resourceList->syncObj1 != NULL) {
    LwSciSyncObjFree(resourceList->syncObj1);
    resourceList->syncObj1 = NULL;
  }

  if (resourceList->syncObj2 != NULL) {
    LwSciSyncObjFree(resourceList->syncObj2);
    resourceList->syncObj2 = NULL;
  }

  if (resourceList->lwSciSyncConflictListObj1 != NULL) {
    LwSciSyncAttrListFree(resourceList->lwSciSyncConflictListObj1);
    resourceList->lwSciSyncConflictListObj1 = NULL;
  }

  if (resourceList->lwSciSyncReconciledListObj1 != NULL) {
    LwSciSyncAttrListFree(resourceList->lwSciSyncReconciledListObj1);
    resourceList->lwSciSyncReconciledListObj1 = NULL;
  }

  if (resourceList->lwSciSyncConflictListObj2 != NULL) {
    LwSciSyncAttrListFree(resourceList->lwSciSyncConflictListObj2);
    resourceList->lwSciSyncConflictListObj2 = NULL;
  }

  if (resourceList->lwSciSyncReconciledListObj2 != NULL) {
    LwSciSyncAttrListFree(resourceList->lwSciSyncReconciledListObj2);
    resourceList->lwSciSyncReconciledListObj2 = NULL;
  }

  if (resourceList->signalerAttrListObj1 != NULL) {
    LwSciSyncAttrListFree(resourceList->signalerAttrListObj1);
    resourceList->signalerAttrListObj1 = NULL;
  }

  if (resourceList->waiterAttrListObj1 != NULL) {
    LwSciSyncAttrListFree(resourceList->waiterAttrListObj1);
    resourceList->waiterAttrListObj1 = NULL;
  }

  if (resourceList->signalerAttrListObj2 != NULL) {
    LwSciSyncAttrListFree(resourceList->signalerAttrListObj2);
    resourceList->signalerAttrListObj2 = NULL;
  }

  if (resourceList->waiterAttrListObj2 != NULL) {
    LwSciSyncAttrListFree(resourceList->waiterAttrListObj2);
    resourceList->waiterAttrListObj2 = NULL;
  }

  if (resourceList->lwSciCtx != NULL) {
    LwSciSyncCpuWaitContextFree(resourceList->lwSciCtx);
    resourceList->lwSciCtx = NULL;
  }

  if (resourceList->syncModule != NULL) {
    LwSciSyncModuleClose(resourceList->syncModule);
    resourceList->syncModule = NULL;
  }

  if (resourceList->waitEvents != NULL) {
    free(resourceList->waitEvents);
    resourceList->waitEvents = NULL;
  }

  if (resourceList->preFences != NULL) {
    free(resourceList->preFences);
    resourceList->preFences = NULL;
  }

  if (resourceList->signalEvents != NULL) {
    if (resourceList->signalEvents->eofFences != NULL) {
      free(resourceList->signalEvents->eofFences);
      resourceList->signalEvents->eofFences = NULL;
    }

    free(resourceList->signalEvents);
    resourceList->signalEvents = NULL;
  }

  if (resourceList->devPtrs != NULL) {
    free(resourceList->devPtrs);
    resourceList->devPtrs = NULL;
  }

  if (resourceList->inputTensorDesc != NULL) {
    free(resourceList->inputTensorDesc);
    resourceList->inputTensorDesc = NULL;
  }
  if (resourceList->outputTensorDesc != NULL) {
    free(resourceList->outputTensorDesc);
    resourceList->outputTensorDesc = NULL;
  }

  if (resourceList->loadableData != NULL) {
    free(resourceList->loadableData);
    resourceList->loadableData = NULL;
  }

  if (resourceList->moduleHandle != NULL) {
    lwdlaModuleUnload(resourceList->moduleHandle, 0);
    resourceList->moduleHandle = NULL;
  }

  if (resourceList->devHandle != NULL) {
    lwdlaDestroyDevice(resourceList->devHandle);
    resourceList->devHandle = NULL;
  }

  if (resourceList->inputBuffer != NULL) {
    free(resourceList->inputBuffer);
    resourceList->inputBuffer = NULL;
  }
  if (resourceList->outputBuffer != NULL) {
    free(resourceList->outputBuffer);
    resourceList->outputBuffer = NULL;
  }
}

lwdlaStatus createAndSetAttrList(LwSciBufModule module, uint64_t bufSize,
                                 LwSciBufAttrList* attrList);

lwdlaStatus createAndSetAttrList(LwSciBufModule module, uint64_t bufSize,
                                 LwSciBufAttrList* attrList) {
  lwdlaStatus status = lwdlaSuccess;
  LwSciError sciStatus = LwSciError_Success;

  sciStatus = LwSciBufAttrListCreate(module, attrList);
  if (sciStatus != LwSciError_Success) {
    status = lwdlaErrorLwSci;
    DPRINTF("Error in creating LwSciBuf attribute list\n");
    return status;
  }

  bool needCpuAccess = true;
  LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
  uint32_t dimcount = 1;
  uint64_t sizes[] = {bufSize};
  uint32_t alignment[] = {1};
  uint32_t dataType = LwSciDataType_Int8;
  LwSciBufType type = LwSciBufType_Tensor;
  uint64_t baseAddrAlign = 512;

  LwSciBufAttrKeyValuePair setAttrs[] = {
      {.key = LwSciBufGeneralAttrKey_Types,
       .value = &type,
       .len = sizeof(type)},
      {.key = LwSciBufTensorAttrKey_DataType,
       .value = &dataType,
       .len = sizeof(dataType)},
      {.key = LwSciBufTensorAttrKey_NumDims,
       .value = &dimcount,
       .len = sizeof(dimcount)},
      {.key = LwSciBufTensorAttrKey_SizePerDim,
       .value = &sizes,
       .len = sizeof(sizes)},
      {.key = LwSciBufTensorAttrKey_AlignmentPerDim,
       .value = &alignment,
       .len = sizeof(alignment)},
      {.key = LwSciBufTensorAttrKey_BaseAddrAlign,
       .value = &baseAddrAlign,
       .len = sizeof(baseAddrAlign)},
      {.key = LwSciBufGeneralAttrKey_RequiredPerm,
       .value = &perm,
       .len = sizeof(perm)},
      {.key = LwSciBufGeneralAttrKey_NeedCpuAccess,
       .value = &needCpuAccess,
       .len = sizeof(needCpuAccess)}};
  size_t length = sizeof(setAttrs) / sizeof(LwSciBufAttrKeyValuePair);

  sciStatus = LwSciBufAttrListSetAttrs(*attrList, setAttrs, length);
  if (sciStatus != LwSciError_Success) {
    status = lwdlaErrorLwSci;
    DPRINTF("Error in setting LwSciBuf attribute list\n");
    return status;
  }

  return status;
}

LwSciError fillCpuSignalerAttrList(LwSciSyncAttrList list);

LwSciError fillCpuSignalerAttrList(LwSciSyncAttrList list) {
  bool cpuSignaler = true;
  LwSciSyncAttrKeyValuePair keyValue[2];
  memset(keyValue, 0, sizeof(keyValue));
  keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
  keyValue[0].value = (void*)&cpuSignaler;
  keyValue[0].len = sizeof(cpuSignaler);

  LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_SignalOnly;
  keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
  keyValue[1].value = (void*)&cpuPerm;
  keyValue[1].len = sizeof(cpuPerm);

  return LwSciSyncAttrListSetAttrs(list, keyValue, 2);
}

LwSciError fillCpuWaiterAttrList(LwSciSyncAttrList list);

LwSciError fillCpuWaiterAttrList(LwSciSyncAttrList list) {
  bool cpuWaiter = true;
  LwSciSyncAttrKeyValuePair keyValue[2];
  memset(keyValue, 0, sizeof(keyValue));
  keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
  keyValue[0].value = (void*)&cpuWaiter;
  keyValue[0].len = sizeof(cpuWaiter);

  LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_WaitOnly;
  keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
  keyValue[1].value = (void*)&cpuPerm;
  keyValue[1].len = sizeof(cpuPerm);

  return LwSciSyncAttrListSetAttrs(list, keyValue, 2);
}

int main(int argc, char** argv) {
  lwdlaDevHandle devHandle;
  lwdlaModule moduleHandle;
  lwdlaStatus err;
  FILE* fp = NULL;
  struct stat st;
  size_t file_size;
  size_t actually_read = 0;
  unsigned char* loadableData = NULL;

  ResourceList resourceList;

  memset(&resourceList, 0x00, sizeof(ResourceList));

  if (argc != 3) {
    DPRINTF("Usage : ./lwDLAStandaloneMode <loadable> <imageFile>\n");
    return 1;
  }

  // Read loadable into buffer.
  fp = fopen(argv[1], "rb");
  if (fp == NULL) {
    DPRINTF("Cannot open file %s\n", argv[1]);
    return 1;
  }

  if (stat(argv[1], &st) != 0) {
    DPRINTF("Cannot stat file\n");
    return 1;
  }

  file_size = st.st_size;
  DPRINTF("The file size = %ld\n", file_size);

  loadableData = (unsigned char*)malloc(file_size);
  if (loadableData == NULL) {
    DPRINTF("Cannot Allocate memory for loadable\n");
    return 1;
  }

  actually_read = fread(loadableData, 1, file_size, fp);
  if (actually_read != file_size) {
    free(loadableData);
    DPRINTF("Read wrong size\n");
    return 1;
  }
  fclose(fp);

  resourceList.loadableData = loadableData;

  err = lwdlaCreateDevice(0, &devHandle, LWDLA_STANDALONE);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in lwDLA create device = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  DPRINTF("Device created successfully\n");
  resourceList.devHandle = devHandle;

  err = lwdlaModuleLoadFromMemory(devHandle, loadableData, file_size,
                                  &moduleHandle, 0);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in lwdlaModuleLoadFromMemory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  } else {
    DPRINTF("Successfully loaded module\n");
  }

  resourceList.moduleHandle = moduleHandle;
  // Get tensor attributes.
  uint32_t numInputTensors = 0;
  uint32_t numOutputTensors = 0;
  lwdlaModuleAttribute attribute;

  err = lwdlaModuleGetAttributes(moduleHandle, LWDLA_NUM_INPUT_TENSORS,
                                 &attribute);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in getting numInputTensors = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  numInputTensors = attribute.numInputTensors;
  DPRINTF("numInputTensors = %d\n", numInputTensors);

  err = lwdlaModuleGetAttributes(moduleHandle, LWDLA_NUM_OUTPUT_TENSORS,
                                 &attribute);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in getting numOutputTensors = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  numOutputTensors = attribute.numOutputTensors;
  DPRINTF("numOutputTensors = %d\n", numOutputTensors);

  lwdlaModuleTensorDescriptor* inputTensorDesc =
      (lwdlaModuleTensorDescriptor*)malloc(sizeof(lwdlaModuleTensorDescriptor) *
                                           numInputTensors);
  lwdlaModuleTensorDescriptor* outputTensorDesc =
      (lwdlaModuleTensorDescriptor*)malloc(sizeof(lwdlaModuleTensorDescriptor) *
                                           numOutputTensors);

  if ((inputTensorDesc == NULL) || (outputTensorDesc == NULL)) {
    if (inputTensorDesc != NULL) {
      free(inputTensorDesc);
      inputTensorDesc = NULL;
    }

    if (outputTensorDesc != NULL) {
      free(outputTensorDesc);
      outputTensorDesc = NULL;
    }

    cleanUp(&resourceList);
    return 1;
  }

  resourceList.inputTensorDesc = inputTensorDesc;
  resourceList.outputTensorDesc = outputTensorDesc;

  attribute.inputTensorDesc = inputTensorDesc;
  err = lwdlaModuleGetAttributes(moduleHandle, LWDLA_INPUT_TENSOR_DESCRIPTORS,
                                 &attribute);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in getting input tensor descriptor = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  DPRINTF("Printing input tensor descriptor\n");
  printTensorDesc(inputTensorDesc);

  attribute.outputTensorDesc = outputTensorDesc;
  err = lwdlaModuleGetAttributes(moduleHandle, LWDLA_OUTPUT_TENSOR_DESCRIPTORS,
                                 &attribute);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in getting output tensor descriptor = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  DPRINTF("Printing output tensor descriptor\n");
  printTensorDesc(outputTensorDesc);

  // Setup the input and output buffers which will be used as an input to LWCA.
  unsigned char* inputBuffer = (unsigned char*)malloc(inputTensorDesc[0].size);
  if (inputBuffer == NULL) {
    DPRINTF("Error in allocating input memory\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.inputBuffer = inputBuffer;

  unsigned char* outputBuffer =
      (unsigned char*)malloc(outputTensorDesc[0].size);
  if (outputBuffer == NULL) {
    DPRINTF("Error in allocating output memory\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.outputBuffer = outputBuffer;

  memset(inputBuffer, 0x00, inputTensorDesc[0].size);
  memset(outputBuffer, 0x00, outputTensorDesc[0].size);

  // Fill up the buffers with data.
  if (initializeInputBuffers(argv[2], inputTensorDesc, inputBuffer) != 0) {
    DPRINTF("Error in initializing input buffer from PGM image\n");
    cleanUp(&resourceList);
    return 1;
  }

  LwSciBufModule bufModule = NULL;
  LwSciBufAttrList inputAttrList = NULL;
  LwSciBufAttrList outputAttrList = NULL;
  LwSciBufAttrList reconciledInputAttrList = NULL;
  LwSciBufAttrList reconciledOutputAttrList = NULL;
  LwSciBufAttrList inputConflictList = NULL;
  LwSciBufAttrList outputConflictList = NULL;
  LwSciError sciError = LwSciError_Success;

  sciError = LwSciBufModuleOpen(&bufModule);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in initializing LwSciBufModule\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.bufModule = bufModule;

  // creating and setting input attribute list
  err =
      createAndSetAttrList(bufModule, inputTensorDesc[0].size, &inputAttrList);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in creating LwSciBuf attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.inputAttrList = inputAttrList;

  sciError = LwSciBufAttrListReconcile(
      &inputAttrList, 1, &reconciledInputAttrList, &inputConflictList);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in reconciling LwSciBuf attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.reconciledInputAttrList = reconciledInputAttrList;
  resourceList.inputConflictList = inputConflictList;

  // creating and setting output attribute list
  err = createAndSetAttrList(bufModule, outputTensorDesc[0].size,
                             &outputAttrList);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in creating LwSciBuf attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.outputAttrList = outputAttrList;

  sciError = LwSciBufAttrListReconcile(
      &outputAttrList, 1, &reconciledOutputAttrList, &outputConflictList);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in reconciling LwSciBuf attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.reconciledOutputAttrList = reconciledOutputAttrList;
  resourceList.outputConflictList = outputConflictList;

  LwSciBufObj inputBufObj, outputBufObj;
  sciError = LwSciBufObjAlloc(reconciledInputAttrList, &inputBufObj);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in allocating LwSciBuf object\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.inputBufObj = inputBufObj;

  sciError = LwSciBufObjAlloc(reconciledOutputAttrList, &outputBufObj);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in allocating LwSciBuf object\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.outputBufObj = outputBufObj;

  uint64_t* inputBufObjRegPtr = NULL;
  uint64_t* outputBufObjRegPtr = NULL;
  void* inputBufObjBuffer;
  void* outputBufObjBuffer;

  // importing external memory
  lwdlaExternalMemoryHandleDesc memDesc = {0};
  memset(&memDesc, 0, sizeof(memDesc));
  memDesc.extBufObject = (void*)inputBufObj;
  memDesc.size = inputTensorDesc[0].size;
  err = lwdlaImportExternalMemory(devHandle, &memDesc, &inputBufObjRegPtr, 0);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in importing external memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  sciError = LwSciBufObjGetCpuPtr(inputBufObj, &inputBufObjBuffer);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in getting LwSciBuf CPU pointer\n");
    cleanUp(&resourceList);
    return 1;
  }
  memcpy(inputBufObjBuffer, inputBuffer, inputTensorDesc[0].size);

  memset(&memDesc, 0, sizeof(memDesc));
  memDesc.extBufObject = (void*)outputBufObj;
  memDesc.size = outputTensorDesc[0].size;
  err = lwdlaImportExternalMemory(devHandle, &memDesc, &outputBufObjRegPtr, 0);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in importing external memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  sciError = LwSciBufObjGetCpuPtr(outputBufObj, &outputBufObjBuffer);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in getting LwSciBuf CPU pointer\n");
    cleanUp(&resourceList);
    return 1;
  }
  memset(outputBufObjBuffer, 0, outputTensorDesc[0].size);

  LwSciSyncObj syncObj1, syncObj2;
  LwSciSyncModule syncModule;
  LwSciSyncAttrList syncAttrListObj1[2];
  LwSciSyncAttrList syncAttrListObj2[2];
  LwSciSyncCpuWaitContext lwSciCtx;
  LwSciSyncAttrList waiterAttrListObj1 = NULL;
  LwSciSyncAttrList signalerAttrListObj1 = NULL;
  LwSciSyncAttrList waiterAttrListObj2 = NULL;
  LwSciSyncAttrList signalerAttrListObj2 = NULL;
  LwSciSyncAttrList lwSciSyncConflictListObj1;
  LwSciSyncAttrList lwSciSyncReconciledListObj1;
  LwSciSyncAttrList lwSciSyncConflictListObj2;
  LwSciSyncAttrList lwSciSyncReconciledListObj2;

  sciError = LwSciSyncModuleOpen(&syncModule);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in initializing LwSciSyncModuleOpen\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.syncModule = syncModule;

  sciError = LwSciSyncAttrListCreate(syncModule, &signalerAttrListObj1);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in creating LwSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.signalerAttrListObj1 = signalerAttrListObj1;

  sciError = LwSciSyncAttrListCreate(syncModule, &waiterAttrListObj1);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in creating LwSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.waiterAttrListObj1 = waiterAttrListObj1;

  err = lwdlaGetLwSciSyncAttributes(
      reinterpret_cast<uint64_t*>(waiterAttrListObj1),
      LWDLA_LWSCISYNC_ATTR_WAIT);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in getting lwDLA's LwSciSync attributes\n");
    cleanUp(&resourceList);
    return 1;
  }

  sciError = fillCpuSignalerAttrList(signalerAttrListObj1);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in setting LwSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }

  syncAttrListObj1[0] = signalerAttrListObj1;
  syncAttrListObj1[1] = waiterAttrListObj1;
  sciError = LwSciSyncAttrListReconcile(syncAttrListObj1, 2,
                                        &lwSciSyncReconciledListObj1,
                                        &lwSciSyncConflictListObj1);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in reconciling LwSciSync's attribute lists\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.lwSciSyncConflictListObj1 = lwSciSyncConflictListObj1;
  resourceList.lwSciSyncReconciledListObj1 = lwSciSyncReconciledListObj1;

  sciError = LwSciSyncObjAlloc(lwSciSyncReconciledListObj1, &syncObj1);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in allocating LwSciSync object\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.syncObj1 = syncObj1;

  sciError = LwSciSyncCpuWaitContextAlloc(syncModule, &lwSciCtx);
  if (sciError != LwSciError_Success) {
    DPRINTF(
        "Error in allocating cpu wait context LwSciSyncCpuWaitContextAlloc\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.lwSciCtx = lwSciCtx;

  sciError = LwSciSyncAttrListCreate(syncModule, &signalerAttrListObj2);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in creating LwSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.signalerAttrListObj2 = signalerAttrListObj2;

  sciError = LwSciSyncAttrListCreate(syncModule, &waiterAttrListObj2);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in creating LwSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.waiterAttrListObj2 = waiterAttrListObj2;

  err = lwdlaGetLwSciSyncAttributes(
      reinterpret_cast<uint64_t*>(signalerAttrListObj2),
      LWDLA_LWSCISYNC_ATTR_SIGNAL);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in getting lwDLA's LwSciSync attributes\n");
    cleanUp(&resourceList);
    return 1;
  }

  sciError = fillCpuWaiterAttrList(waiterAttrListObj2);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in setting LwSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }

  syncAttrListObj2[0] = signalerAttrListObj2;
  syncAttrListObj2[1] = waiterAttrListObj2;
  sciError = LwSciSyncAttrListReconcile(syncAttrListObj2, 2,
                                        &lwSciSyncReconciledListObj2,
                                        &lwSciSyncConflictListObj2);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in reconciling LwSciSync's attribute lists\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.lwSciSyncConflictListObj2 = lwSciSyncConflictListObj2;
  resourceList.lwSciSyncReconciledListObj2 = lwSciSyncReconciledListObj2;

  sciError = LwSciSyncObjAlloc(lwSciSyncReconciledListObj2, &syncObj2);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in allocating LwSciSync object\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.syncObj2 = syncObj2;

  // importing external semaphore
  uint64_t* lwSciSyncObjRegPtr1 = NULL;
  uint64_t* lwSciSyncObjRegPtr2 = NULL;
  lwdlaExternalSemaphoreHandleDesc semaMemDesc = {0};
  memset(&semaMemDesc, 0, sizeof(semaMemDesc));
  semaMemDesc.extSyncObject = syncObj1;
  err = lwdlaImportExternalSemaphore(devHandle, &semaMemDesc,
                                     &lwSciSyncObjRegPtr1, 0);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in importing external semaphore = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  memset(&semaMemDesc, 0, sizeof(semaMemDesc));
  semaMemDesc.extSyncObject = syncObj2;
  err = lwdlaImportExternalSemaphore(devHandle, &semaMemDesc,
                                     &lwSciSyncObjRegPtr2, 0);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in importing external semaphore = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  DPRINTF("ALL MEMORY REGISTERED SUCCESSFULLY\n");

  // Wait events
  LwSciSyncFence preFence = LwSciSyncFenceInitializer;
  sciError = LwSciSyncObjGenerateFence(syncObj1, &preFence);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in generating LwSciSyncObj fence %x\n", sciError);
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.preFence = preFence;

  lwdlaWaitEvents* waitEvents;
  waitEvents = (lwdlaWaitEvents*)malloc(sizeof(lwdlaWaitEvents));
  if (waitEvents == NULL) {
    DPRINTF("Error in allocating wait events\n");
    cleanUp(&resourceList);
    return 1;
  }

  waitEvents->numEvents = 1;
  LwdlaFence* preFences =
      (LwdlaFence*)malloc(waitEvents->numEvents * sizeof(LwdlaFence));
  if (preFences == NULL) {
    DPRINTF("Error in allocating preFence array\n");
    cleanUp(&resourceList);
    return 1;
  }

  preFences[0].fence = &preFence;
  preFences[0].type = LWDLA_LWSCISYNC_FENCE;
  waitEvents->preFences = preFences;
  resourceList.preFences = preFences;
  resourceList.waitEvents = waitEvents;

  // Signal Events
  lwdlaSignalEvents* signalEvents;
  signalEvents = (lwdlaSignalEvents*)malloc(sizeof(lwdlaSignalEvents));
  if (signalEvents == NULL) {
    DPRINTF("Error in allocating signal events\n");
    cleanUp(&resourceList);
    return 1;
  }

  signalEvents->numEvents = 1;
  uint64_t** devPtrs =
      (uint64_t**)malloc(signalEvents->numEvents * sizeof(uint64_t*));
  if (devPtrs == NULL) {
    DPRINTF(
        "Error in allocating output pointer's array of registered objects\n");
    cleanUp(&resourceList);
    return 1;
  }
  devPtrs[0] = lwSciSyncObjRegPtr2;
  signalEvents->devPtrs = devPtrs;
  resourceList.devPtrs = devPtrs;

  signalEvents->eofFences =
      (LwdlaFence*)malloc(signalEvents->numEvents * sizeof(LwdlaFence));
  if (signalEvents->eofFences == NULL) {
    DPRINTF("Error in allocating eofFence array\n");
    cleanUp(&resourceList);
    return 1;
  }

  LwSciSyncFence eofFence = LwSciSyncFenceInitializer;
  signalEvents->eofFences[0].fence = &eofFence;
  signalEvents->eofFences[0].type = LWDLA_LWSCISYNC_FENCE;
  resourceList.signalEvents = signalEvents;
  resourceList.eofFence = eofFence;

  // Enqueue a lwDLA task.
  lwdlaTask task;
  task.moduleHandle = moduleHandle;
  task.outputTensor = &outputBufObjRegPtr;
  task.numOutputTensors = 1;
  task.numInputTensors = 1;
  task.inputTensor = &inputBufObjRegPtr;
  task.waitEvents = waitEvents;
  task.signalEvents = signalEvents;
  err = lwdlaSubmitTask(devHandle, &task, 1, NULL, 0);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in submitting task\n");
    cleanUp(&resourceList);
    return 1;
  }
  DPRINTF("SUBMIT IS DONE !!!\n");

  // Signal wait events
  LwSciSyncObjSignal(syncObj1);
  DPRINTF("SIGNALED WAIT EVENTS SUCCESSFULLY\n");

  // Wait for operations to finish and bring output buffer to CPU.
  sciError = LwSciSyncFenceWait(
      reinterpret_cast<LwSciSyncFence*>(signalEvents->eofFences[0].fence),
      lwSciCtx, -1);
  if (sciError != LwSciError_Success) {
    DPRINTF("Error in waiting on LwSciSyncFence\n");
    cleanUp(&resourceList);
    return 1;
  }

  memcpy(outputBuffer, outputBufObjBuffer, outputTensorDesc[0].size);

  // Output is available in outputBuffer.

  // Teardown.
  err = lwdlaMemUnregister(devHandle, inputBufObjRegPtr);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in unregistering external memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  err = lwdlaMemUnregister(devHandle, outputBufObjRegPtr);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in unregistering external memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  err = lwdlaMemUnregister(devHandle, lwSciSyncObjRegPtr1);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in unregistering external semaphore = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  err = lwdlaMemUnregister(devHandle, lwSciSyncObjRegPtr2);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in unregistering external semaphore = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  DPRINTF("ALL MEMORY UNREGISTERED SUCCESSFULLY\n");

  free(inputTensorDesc);
  free(outputTensorDesc);
  free(loadableData);
  free(inputBuffer);
  free(outputBuffer);
  LwSciBufObjFree(inputBufObj);
  LwSciBufObjFree(outputBufObj);
  LwSciBufAttrListFree(reconciledInputAttrList);
  LwSciBufAttrListFree(inputConflictList);
  LwSciBufAttrListFree(inputAttrList);
  LwSciBufAttrListFree(reconciledOutputAttrList);
  LwSciBufAttrListFree(outputConflictList);
  LwSciBufAttrListFree(outputAttrList);
  LwSciBufModuleClose(bufModule);
  LwSciSyncObjFree(syncObj1);
  LwSciSyncObjFree(syncObj2);
  LwSciSyncAttrListFree(signalerAttrListObj1);
  LwSciSyncAttrListFree(waiterAttrListObj1);
  LwSciSyncAttrListFree(signalerAttrListObj2);
  LwSciSyncAttrListFree(waiterAttrListObj2);
  LwSciSyncAttrListFree(lwSciSyncConflictListObj1);
  LwSciSyncAttrListFree(lwSciSyncReconciledListObj1);
  LwSciSyncAttrListFree(lwSciSyncConflictListObj2);
  LwSciSyncAttrListFree(lwSciSyncReconciledListObj2);
  LwSciSyncCpuWaitContextFree(lwSciCtx);
  LwSciSyncModuleClose(syncModule);
  free(waitEvents);
  free(preFences);
  free(signalEvents->eofFences);
  free(signalEvents);
  free(devPtrs);
  LwSciSyncFenceClear(&preFence);
  LwSciSyncFenceClear(&eofFence);

  resourceList.inputTensorDesc = NULL;
  resourceList.outputTensorDesc = NULL;
  resourceList.loadableData = NULL;
  resourceList.inputBuffer = NULL;
  resourceList.outputBuffer = NULL;
  resourceList.inputBufObj = NULL;
  resourceList.outputBufObj = NULL;
  resourceList.reconciledInputAttrList = NULL;
  resourceList.inputConflictList = NULL;
  resourceList.inputAttrList = NULL;
  resourceList.reconciledOutputAttrList = NULL;
  resourceList.outputConflictList = NULL;
  resourceList.outputAttrList = NULL;
  resourceList.bufModule = NULL;
  resourceList.syncObj1 = NULL;
  resourceList.syncObj2 = NULL;
  resourceList.signalerAttrListObj1 = NULL;
  resourceList.waiterAttrListObj1 = NULL;
  resourceList.signalerAttrListObj2 = NULL;
  resourceList.waiterAttrListObj2 = NULL;
  resourceList.lwSciSyncConflictListObj1 = NULL;
  resourceList.lwSciSyncReconciledListObj1 = NULL;
  resourceList.lwSciSyncConflictListObj2 = NULL;
  resourceList.lwSciSyncReconciledListObj2 = NULL;
  resourceList.lwSciCtx = NULL;
  resourceList.syncModule = NULL;
  resourceList.waitEvents = NULL;
  resourceList.signalEvents = NULL;
  resourceList.preFences = NULL;
  resourceList.devPtrs = NULL;

  err = lwdlaModuleUnload(moduleHandle, 0);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in lwdlaModuleUnload = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  } else {
    DPRINTF("Successfully unloaded module\n");
  }

  resourceList.moduleHandle = NULL;

  err = lwdlaDestroyDevice(devHandle);
  if (err != lwdlaSuccess) {
    DPRINTF("Error in lwDLA destroy device = %d\n", err);
    return 1;
  }
  DPRINTF("Device destroyed successfully\n");

  resourceList.devHandle = NULL;

  DPRINTF("lwDLAStandaloneMode DONE !!!\n");

  return 0;
}
