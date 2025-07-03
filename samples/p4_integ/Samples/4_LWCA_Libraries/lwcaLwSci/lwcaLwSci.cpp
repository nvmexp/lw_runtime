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

#include "lwdaLwSci.h"
#include <lwca.h>
#include <condition_variable>
#include <iostream>
#include <thread>

std::mutex m_mutex;
std::condition_variable m_condVar;
bool workSubmitted = false;

class lwdaLwSciSignal {
 private:
  LwSciSyncModule m_syncModule;
  LwSciBufModule m_bufModule;

  LwSciSyncAttrList m_syncAttrList;
  LwSciSyncFence *m_fence;

  LwSciBufAttrList m_rawBufAttrList;
  LwSciBufAttrList m_imageBufAttrList;
  LwSciBufAttrList m_buffAttrListOut[2];
  LwSciBufAttrKeyValuePair pairArrayOut[10];

  lwdaExternalMemory_t extMemRawBuf, extMemImageBuf;
  lwdaMipmappedArray_t d_mipmapArray;
  lwdaArray_t d_mipLevelArray;
  lwdaTextureObject_t texObject;
  lwdaExternalSemaphore_t signalSem;

  lwdaStream_t streamToRun;
  int m_lwdaDeviceId;
  LWuuid m_devUUID;
  uint64_t m_imageWidth;
  uint64_t m_imageHeight;
  void *d_outputBuf;
  size_t m_bufSize;

 public:
  lwdaLwSciSignal(LwSciBufModule bufModule, LwSciSyncModule syncModule,
                  int lwdaDeviceId, int bufSize, uint64_t imageWidth,
                  uint64_t imageHeight, LwSciSyncFence *fence)
      : m_syncModule(syncModule),
        m_bufModule(bufModule),
        m_lwdaDeviceId(lwdaDeviceId),
        d_outputBuf(NULL),
        m_bufSize(bufSize),
        m_imageWidth(imageWidth),
        m_imageHeight(imageHeight),
        m_fence(fence) {
    initLwda();

    checkLwSciErrors(LwSciSyncAttrListCreate(m_syncModule, &m_syncAttrList));
    checkLwSciErrors(LwSciBufAttrListCreate(m_bufModule, &m_rawBufAttrList));
    checkLwSciErrors(LwSciBufAttrListCreate(m_bufModule, &m_imageBufAttrList));

    setRawBufAttrList(m_bufSize);
    setImageBufAttrList(m_imageWidth, m_imageHeight);

    checkLwdaErrors(lwdaDeviceGetLwSciSyncAttributes(
        m_syncAttrList, m_lwdaDeviceId, lwdaLwSciSyncAttrSignal));
  }

  ~lwdaLwSciSignal() {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));
    checkLwdaErrors(lwdaFreeMipmappedArray(d_mipmapArray));
    checkLwdaErrors(lwdaFree(d_outputBuf));
    checkLwdaErrors(lwdaDestroyExternalSemaphore(signalSem));
    checkLwdaErrors(lwdaDestroyExternalMemory(extMemRawBuf));
    checkLwdaErrors(lwdaDestroyExternalMemory(extMemImageBuf));
    checkLwdaErrors(lwdaDestroyTextureObject(texObject));
    checkLwdaErrors(lwdaStreamDestroy(streamToRun));
  }

  void initLwda() {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));
    checkLwdaErrors(
        lwdaStreamCreateWithFlags(&streamToRun, lwdaStreamNonBlocking));

    int major = 0, minor = 0;
    checkLwdaErrors(lwdaDeviceGetAttribute(
        &major, lwdaDevAttrComputeCapabilityMajor, m_lwdaDeviceId));
    checkLwdaErrors(lwdaDeviceGetAttribute(
        &minor, lwdaDevAttrComputeCapabilityMinor, m_lwdaDeviceId));
    printf(
        "[lwdaLwSciSignal] GPU Device %d: \"%s\" with compute capability "
        "%d.%d\n\n",
        m_lwdaDeviceId, _ColwertSMVer2ArchName(major, minor), major, minor);

#ifdef lwDeviceGetUuid_v2
    LWresult res = lwDeviceGetUuid_v2(&m_devUUID, m_lwdaDeviceId);
#else
    LWresult res = lwDeviceGetUuid(&m_devUUID, m_lwdaDeviceId);
#endif

    if (res != LWDA_SUCCESS) {
      fprintf(stderr, "Driver API error = %04d \n", res);
      exit(EXIT_FAILURE);
    }
  }

  void setRawBufAttrList(uint64_t size) {
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    bool cpuAccess = false;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
    LwSciBufAttrKeyValuePair rawBufAttrs[] = {
        {LwSciBufRawBufferAttrKey_Size, &size, sizeof(size)},
        {LwSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {LwSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
        {LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {LwSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkLwSciErrors(LwSciBufAttrListSetAttrs(
        m_rawBufAttrList, rawBufAttrs,
        sizeof(rawBufAttrs) / sizeof(LwSciBufAttrKeyValuePair)));
  }

  void setImageBufAttrList(uint32_t width, uint32_t height) {
    LwSciBufType bufType = LwSciBufType_Image;
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    uint32_t planeCount = 1;
    uint32_t planeWidths[] = {width};
    uint32_t planeHeights[] = {height};
    uint64_t lrpad = 0, tbpad = 100;

    bool cpuAccessFlag = false;

    LwSciBufAttrValColorFmt planecolorfmts[] = {LwSciColor_B8G8R8A8};
    LwSciBufAttrValColorStd planecolorstds[] = {LwSciColorStd_SRGB};
    LwSciBufAttrValImageScanType planescantype[] = {LwSciBufScan_InterlaceType};

    LwSciBufAttrKeyValuePair imgBufAttrs[] = {
        {LwSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {LwSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
        {LwSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
        {LwSciBufImageAttrKey_TopPadding, &tbpad, sizeof(tbpad)},
        {LwSciBufImageAttrKey_BottomPadding, &tbpad, sizeof(tbpad)},
        {LwSciBufImageAttrKey_LeftPadding, &lrpad, sizeof(lrpad)},
        {LwSciBufImageAttrKey_RightPadding, &lrpad, sizeof(lrpad)},
        {LwSciBufImageAttrKey_PlaneColorFormat, planecolorfmts,
         sizeof(planecolorfmts)},
        {LwSciBufImageAttrKey_PlaneColorStd, planecolorstds,
         sizeof(planecolorstds)},
        {LwSciBufImageAttrKey_PlaneWidth, planeWidths, sizeof(planeWidths)},
        {LwSciBufImageAttrKey_PlaneHeight, planeHeights, sizeof(planeHeights)},
        {LwSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag,
         sizeof(cpuAccessFlag)},
        {LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {LwSciBufImageAttrKey_PlaneScanType, planescantype,
         sizeof(planescantype)},
        {LwSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkLwSciErrors(LwSciBufAttrListSetAttrs(
        m_imageBufAttrList, imgBufAttrs,
        sizeof(imgBufAttrs) / sizeof(LwSciBufAttrKeyValuePair)));
  }

  LwSciSyncAttrList getLwSciSyncAttrList() { return m_syncAttrList; }

  LwSciBufAttrList getLwSciRawBufAttrList() { return m_rawBufAttrList; }

  LwSciBufAttrList getLwSciImageBufAttrList() { return m_imageBufAttrList; }

  void runRotateImageAndSignal(unsigned char *imageData) {
    int numOfGPUs = 0;
    checkLwdaErrors(lwdaGetDeviceCount(&numOfGPUs));  // For lwca init purpose
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));

    copyDataToImageArray(imageData);
    createTexture();

    float angle = 0.5f;  // angle to rotate image by (in radians)
    rotateKernel(texObject, angle, (unsigned int *)d_outputBuf, m_imageWidth,
                 m_imageHeight, streamToRun);

    signalExternalSemaphore();
  }

  void lwdaImportLwSciSemaphore(LwSciSyncObj syncObj) {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));

    lwdaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = lwdaExternalSemaphoreHandleTypeLwSciSync;
    extSemDesc.handle.lwSciSyncObj = (void *)syncObj;

    checkLwdaErrors(lwdaImportExternalSemaphore(&signalSem, &extSemDesc));
  }

  void signalExternalSemaphore() {
    lwdaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    // For cross-process signaler-waiter applications need to use LwSciIpc
    // and LwSciSync[Export|Import] utilities to share the LwSciSyncFence
    // across process. This step is optional in single-process.
    signalParams.params.lwSciSync.fence = (void *)m_fence;
    signalParams.flags = 0;

    checkLwdaErrors(lwdaSignalExternalSemaphoresAsync(&signalSem, &signalParams,
                                                      1, streamToRun));
  }

  void lwdaImportLwSciRawBuf(LwSciBufObj inputBufObj) {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));
    checkLwSciErrors(
        LwSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut[0]));

    memset(pairArrayOut, 0, sizeof(LwSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = LwSciBufRawBufferAttrKey_Size;

    checkLwSciErrors(
        LwSciBufAttrListGetAttrs(m_buffAttrListOut[0], pairArrayOut, 1));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;

    lwdaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = lwdaExternalMemoryHandleTypeLwSciBuf;
    memHandleDesc.handle.lwSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkLwdaErrors(lwdaImportExternalMemory(&extMemRawBuf, &memHandleDesc));

    lwdaExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    m_bufSize = size;
    checkLwdaErrors(lwdaExternalMemoryGetMappedBuffer(
        &d_outputBuf, extMemRawBuf, &bufferDesc));
  }

  void lwdaImportLwSciImage(LwSciBufObj inputBufObj) {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));
    checkLwSciErrors(
        LwSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut[1]));

    memset(pairArrayOut, 0, sizeof(LwSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = LwSciBufImageAttrKey_Size;
    pairArrayOut[1].key = LwSciBufImageAttrKey_Alignment;
    pairArrayOut[2].key = LwSciBufImageAttrKey_PlaneCount;
    pairArrayOut[3].key = LwSciBufImageAttrKey_PlaneWidth;
    pairArrayOut[4].key = LwSciBufImageAttrKey_PlaneHeight;

    checkLwSciErrors(
        LwSciBufAttrListGetAttrs(m_buffAttrListOut[1], pairArrayOut, 5));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;
    uint64_t alignment = *(uint64_t *)pairArrayOut[1].value;
    uint64_t planeCount = *(uint64_t *)pairArrayOut[2].value;
    uint64_t imageWidth = *(uint64_t *)pairArrayOut[3].value;
    uint64_t imageHeight = *(uint64_t *)pairArrayOut[4].value;

    lwdaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = lwdaExternalMemoryHandleTypeLwSciBuf;
    memHandleDesc.handle.lwSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkLwdaErrors(lwdaImportExternalMemory(&extMemImageBuf, &memHandleDesc));

    lwdaExtent extent = {};
    memset(&extent, 0, sizeof(extent));
    extent.width = imageWidth;
    extent.height = imageHeight;
    extent.depth = 0;

    lwdaChannelFormatDesc desc;
    desc.x = 8;
    desc.y = 8;
    desc.z = 8;
    desc.w = 8;
    desc.f = lwdaChannelFormatKindUnsigned;

    lwdaExternalMemoryMipmappedArrayDesc mipmapDesc = {0};
    mipmapDesc.offset = 0;
    mipmapDesc.formatDesc = desc;
    mipmapDesc.extent = extent;
    mipmapDesc.flags = 0;

    mipmapDesc.numLevels = 1;
    checkLwdaErrors(lwdaExternalMemoryGetMappedMipmappedArray(
        &d_mipmapArray, extMemImageBuf, &mipmapDesc));
  }

  void copyDataToImageArray(unsigned char *imageData) {
    uint32_t mipLevelId = 0;
    checkLwdaErrors(lwdaGetMipmappedArrayLevel(&d_mipLevelArray, d_mipmapArray,
                                               mipLevelId));

    checkLwdaErrors(lwdaMemcpy2DToArrayAsync(
        d_mipLevelArray, 0, 0, imageData, m_imageWidth * sizeof(unsigned int),
        m_imageWidth * sizeof(unsigned int), m_imageHeight,
        lwdaMemcpyHostToDevice, streamToRun));
  }

  void createTexture() {
    lwdaResourceDesc texRes;
    memset(&texRes, 0, sizeof(lwdaResourceDesc));

    texRes.resType = lwdaResourceTypeArray;
    texRes.res.array.array = d_mipLevelArray;

    lwdaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(lwdaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = lwdaFilterModeLinear;
    texDescr.addressMode[0] = lwdaAddressModeWrap;
    texDescr.addressMode[1] = lwdaAddressModeWrap;
    texDescr.readMode = lwdaReadModeNormalizedFloat;

    checkLwdaErrors(
        lwdaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
  }
};

class lwdaLwSciWait {
 private:
  LwSciSyncModule m_syncModule;
  LwSciBufModule m_bufModule;

  LwSciSyncAttrList m_syncAttrList;
  LwSciBufAttrList m_rawBufAttrList;
  LwSciBufAttrList m_buffAttrListOut;
  LwSciBufAttrKeyValuePair pairArrayOut[10];
  LwSciSyncFence *m_fence;

  lwdaExternalMemory_t extMemRawBuf;
  lwdaExternalSemaphore_t waitSem;
  lwdaStream_t streamToRun;
  int m_lwdaDeviceId;
  LWuuid m_devUUID;
  void *d_outputBuf;
  size_t m_bufSize;
  size_t imageWidth;
  size_t imageHeight;

 public:
  lwdaLwSciWait(LwSciBufModule bufModule, LwSciSyncModule syncModule,
                int lwdaDeviceId, int bufSize, LwSciSyncFence *fence)
      : m_bufModule(bufModule),
        m_syncModule(syncModule),
        m_lwdaDeviceId(lwdaDeviceId),
        m_bufSize(bufSize),
        m_fence(fence) {
    initLwda();
    checkLwSciErrors(LwSciSyncAttrListCreate(m_syncModule, &m_syncAttrList));
    checkLwSciErrors(LwSciBufAttrListCreate(m_bufModule, &m_rawBufAttrList));

    setRawBufAttrList(m_bufSize);
    checkLwdaErrors(lwdaDeviceGetLwSciSyncAttributes(
        m_syncAttrList, m_lwdaDeviceId, lwdaLwSciSyncAttrWait));
  }

  ~lwdaLwSciWait() {
    checkLwdaErrors(lwdaStreamDestroy(streamToRun));
    checkLwdaErrors(lwdaDestroyExternalSemaphore(waitSem));
    checkLwdaErrors(lwdaDestroyExternalMemory(extMemRawBuf));
    checkLwdaErrors(lwdaFree(d_outputBuf));
  }

  void initLwda() {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));
    checkLwdaErrors(
        lwdaStreamCreateWithFlags(&streamToRun, lwdaStreamNonBlocking));
#ifdef lwDeviceGetUuid_v2
    LWresult res = lwDeviceGetUuid_v2(&m_devUUID, m_lwdaDeviceId);
#else
    LWresult res = lwDeviceGetUuid(&m_devUUID, m_lwdaDeviceId);
#endif
    if (res != LWDA_SUCCESS) {
      fprintf(stderr, "Driver API error = %04d \n", res);
      exit(EXIT_FAILURE);
    }

    int major = 0, minor = 0;
    checkLwdaErrors(lwdaDeviceGetAttribute(
        &major, lwdaDevAttrComputeCapabilityMajor, m_lwdaDeviceId));
    checkLwdaErrors(lwdaDeviceGetAttribute(
        &minor, lwdaDevAttrComputeCapabilityMinor, m_lwdaDeviceId));
    printf(
        "[lwdaLwSciWait] GPU Device %d: \"%s\" with compute capability "
        "%d.%d\n\n",
        m_lwdaDeviceId, _ColwertSMVer2ArchName(major, minor), major, minor);
  }

  void setRawBufAttrList(uint64_t size) {
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    bool cpuAccess = false;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
    LwSciBufAttrKeyValuePair rawBufAttrs[] = {
        {LwSciBufRawBufferAttrKey_Size, &size, sizeof(size)},
        {LwSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {LwSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
        {LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {LwSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkLwSciErrors(LwSciBufAttrListSetAttrs(
        m_rawBufAttrList, rawBufAttrs,
        sizeof(rawBufAttrs) / sizeof(LwSciBufAttrKeyValuePair)));
  }

  LwSciSyncAttrList getLwSciSyncAttrList() { return m_syncAttrList; }

  LwSciBufAttrList getLwSciRawBufAttrList() { return m_rawBufAttrList; }

  void runImageGrayscale(std::string image_filename, size_t imageWidth,
                         size_t imageHeight) {
    int numOfGPUs = 0;
    checkLwdaErrors(lwdaGetDeviceCount(&numOfGPUs));  // For lwca init purpose
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));

    waitExternalSemaphore();
    launchGrayScaleKernel((unsigned int *)d_outputBuf, image_filename,
                          imageWidth, imageHeight, streamToRun);
  }

  void lwdaImportLwSciSemaphore(LwSciSyncObj syncObj) {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));

    lwdaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = lwdaExternalSemaphoreHandleTypeLwSciSync;
    extSemDesc.handle.lwSciSyncObj = (void *)syncObj;

    checkLwdaErrors(lwdaImportExternalSemaphore(&waitSem, &extSemDesc));
  }

  void waitExternalSemaphore() {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));

    lwdaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    // For cross-process signaler-waiter applications need to use LwSciIpc
    // and LwSciSync[Export|Import] utilities to share the LwSciSyncFence
    // across process. This step is optional in single-process.
    waitParams.params.lwSciSync.fence = (void *)m_fence;
    waitParams.flags = 0;

    checkLwdaErrors(
        lwdaWaitExternalSemaphoresAsync(&waitSem, &waitParams, 1, streamToRun));
  }

  void lwdaImportLwSciRawBuf(LwSciBufObj inputBufObj) {
    checkLwdaErrors(lwdaSetDevice(m_lwdaDeviceId));

    checkLwSciErrors(LwSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut));

    memset(pairArrayOut, 0, sizeof(LwSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = LwSciBufRawBufferAttrKey_Size;

    checkLwSciErrors(
        LwSciBufAttrListGetAttrs(m_buffAttrListOut, pairArrayOut, 1));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;

    lwdaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = lwdaExternalMemoryHandleTypeLwSciBuf;
    memHandleDesc.handle.lwSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkLwdaErrors(lwdaImportExternalMemory(&extMemRawBuf, &memHandleDesc));

    lwdaExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    m_bufSize = size;

    checkLwdaErrors(lwdaExternalMemoryGetMappedBuffer(
        &d_outputBuf, extMemRawBuf, &bufferDesc));
  }
};

void thread_rotateAndSignal(lwdaLwSciSignal *lwdaLwSciSignalObj,
                            unsigned char *imageData) {
  std::lock_guard<std::mutex> guard(m_mutex);
  lwdaLwSciSignalObj->runRotateImageAndSignal(imageData);
  workSubmitted = true;
  m_condVar.notify_one();
}

void thread_waitAndGrayscale(lwdaLwSciWait *lwdaLwSciWaitObj,
                             std::string image_filename, size_t imageWidth,
                             size_t imageHeight) {
  // Acquire the lock
  std::unique_lock<std::mutex> mlock(m_mutex);
  m_condVar.wait(mlock, [] { return workSubmitted; });
  lwdaLwSciWaitObj->runImageGrayscale(image_filename, imageWidth, imageHeight);
}

lwdaLwSci::lwdaLwSci(int isMultiGPU, std::vector<int> &deviceIds,
                     unsigned char *imageData, size_t width, size_t height)
    : m_isMultiGPU(isMultiGPU),
      image_data(imageData),
      imageWidth(width),
      imageHeight(height) {
  if (isMultiGPU) {
    m_lwdaLwSciSignalDeviceId = deviceIds[0];
    m_lwdaLwSciWaitDeviceId = deviceIds[1];
  } else {
    m_lwdaLwSciSignalDeviceId = m_lwdaLwSciWaitDeviceId = deviceIds[0];
  }

  m_bufSize = imageWidth * imageHeight * sizeof(unsigned int);
}

void lwdaLwSci::initLwSci() {
  checkLwSciErrors(LwSciSyncModuleOpen(&syncModule));
  checkLwSciErrors(LwSciBufModuleOpen(&buffModule));
  fence = (LwSciSyncFence *)calloc(1, sizeof(LwSciSyncFence));
}

void lwdaLwSci::runLwdaLwSci(std::string &image_filename) {
  initLwSci();

  lwdaLwSciSignal rotateAndSignal(buffModule, syncModule,
                                  m_lwdaLwSciSignalDeviceId, m_bufSize,
                                  imageWidth, imageHeight, fence);
  lwdaLwSciWait waitAndGrayscale(buffModule, syncModule,
                                 m_lwdaLwSciWaitDeviceId, m_bufSize, fence);

  rawBufUnreconciledList[0] = rotateAndSignal.getLwSciRawBufAttrList();
  rawBufUnreconciledList[1] = waitAndGrayscale.getLwSciRawBufAttrList();

  createLwSciRawBufObj();

  imageBufUnreconciledList[0] = rotateAndSignal.getLwSciImageBufAttrList();

  createLwSciBufImageObj();

  rotateAndSignal.lwdaImportLwSciRawBuf(rawBufObj);
  rotateAndSignal.lwdaImportLwSciImage(imageBufObj);

  waitAndGrayscale.lwdaImportLwSciRawBuf(rawBufObj);

  synlwnreconciledList[0] = rotateAndSignal.getLwSciSyncAttrList();
  synlwnreconciledList[1] = waitAndGrayscale.getLwSciSyncAttrList();

  createLwSciSyncObj();

  rotateAndSignal.lwdaImportLwSciSemaphore(syncObj);
  waitAndGrayscale.lwdaImportLwSciSemaphore(syncObj);

  std::thread rotateThread(&thread_rotateAndSignal, &rotateAndSignal,
                           image_data);

  std::thread grayscaleThread(&thread_waitAndGrayscale, &waitAndGrayscale,
                              image_filename, imageWidth, imageHeight);

  rotateThread.join();
  grayscaleThread.join();
}

void lwdaLwSci::createLwSciRawBufObj() {
  int numAttrList = 2;
  checkLwSciErrors(LwSciBufAttrListReconcile(rawBufUnreconciledList,
                                             numAttrList, &rawBufReconciledList,
                                             &buffConflictList));
  checkLwSciErrors(LwSciBufObjAlloc(rawBufReconciledList, &rawBufObj));
  printf("created LwSciBufObj\n");
}

void lwdaLwSci::createLwSciBufImageObj() {
  int numAttrList = 1;
  checkLwSciErrors(LwSciBufAttrListReconcile(
      imageBufUnreconciledList, numAttrList, &imageBufReconciledList,
      &imageBufConflictList));
  checkLwSciErrors(LwSciBufObjAlloc(imageBufReconciledList, &imageBufObj));
  printf("created LwSciBufImageObj\n");
}

void lwdaLwSci::createLwSciSyncObj() {
  int numAttrList = 2;
  checkLwSciErrors(LwSciSyncAttrListReconcile(synlwnreconciledList, numAttrList,
                                              &syncReconciledList,
                                              &syncConflictList));
  checkLwSciErrors(LwSciSyncObjAlloc(syncReconciledList, &syncObj));
  printf("created LwSciSyncObj\n");
}
