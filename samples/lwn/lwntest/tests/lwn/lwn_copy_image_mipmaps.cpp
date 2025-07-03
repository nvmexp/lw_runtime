/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;


#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINT(x) do { \
    printf x; \
    fflush(stdout); \
} while (0)
#else
#define DEBUG_PRINT(x)
#endif


class LWNGenerateMipmaps
{
    const static int cellWidth = 10;
    const static int cellHeight = 10;

public:
    LWNTEST_CppMethods();
    void DisplayResult(QueueCommandBuffer &cmd, int cellNum, LWNboolean success) const;
};

lwString LWNGenerateMipmaps::getDescription() const
{
    lwStringBuf sb;
    sb << "Tests the LWN API's CopyImage entry point, to verify that we can generate valid\n"
          "2D mipmaps with it using linear filtering. We do this by generating some large\n"
          "textures and generating mipmaps down to 1x1, then comparing against a precallwlated\n"
          "result. Texture values are chosen to ensure deterministic results.\n\n"
          "Tested texture targets:\n"
          "* 1D texture of dimension 128\n"
          "* 2D texture of dimensions 128x128\n"
          "* 2D_ARRAY texture of dimensions 128x128x2\n";
    return sb.str();
}

int LWNGenerateMipmaps::isSupported() const
{
    return lwogCheckLWNAPIVersion(31, 1);
}

void LWNGenerateMipmaps::DisplayResult(QueueCommandBuffer &cmd, int cellNum, LWNboolean success) const
{
    SetCellViewportScissorPadded(cmd, cellNum % cellWidth, cellNum / cellWidth, 1);
    cmd.ClearColor(0, success ? 0.0 : 1.0, success ? 1.0 : 0.0, 0.0);
}

static LWNuint lowerMipDim(LWNuint dim)
{
    LWNuint t = dim >> 1;
    if (t == 0)
        return 1;
    else
        return t;
}

static void generateMipmaps2D(Device *device, Queue *queue, QueueCommandBuffer &cmd, Texture *tex, int width, int height, int layer, int mipLevels)
{

    // NOTES:
    // * CopyImage only does 2D sampling, so we can only generate 2D mipmaps
    //   with it. To generate mipmaps for 3D textures, use the 3D pipeline.
    // * sRGB formats require gamma correction; use the 3D pipeline.
    // * For non-renderable formats; callwlate on the CPU.
    CopyRegion reg0 = { 0, 0, layer, width, height, 1 };
    CopyRegion reg1 = { 0, 0, layer, 0, 0, 1 };
    TextureView view0, view1;
    view0.SetDefaults();
    view1.SetDefaults();

    // We ping-pong back and forth where generating odd levels will use <reg0>
    // and <view0> as the source and <reg1> and <view1> as the destination.
    // Even levels will use <reg1> and <view1> as the source.  The observation
    // here is that level N will be used as the destination when generating
    // from N-1, and then the same level will be used as the source next time.

    for (int level = 1; level < mipLevels+1; level++)
    {
        if (level % 2) {
            reg1.width = lowerMipDim(reg0.width);
            reg1.height = lowerMipDim(reg0.height);
            view1.SetLevels(level, 1);
            cmd.CopyTextureToTexture(tex, &view0, &reg0, tex, &view1, &reg1, CopyFlags::LINEAR_FILTER);
            DEBUG_PRINT(("src lev: %u  reg0 %u %u %u %u %u %u\n", level - 1, reg0.xoffset, reg0.yoffset, reg0.zoffset, reg0.width, reg0.height, reg0.depth));
            DEBUG_PRINT(("dst lev: %u  reg1 %u %u %u %u %u %u\n", level,     reg1.xoffset, reg1.yoffset, reg1.zoffset, reg1.width, reg1.height, reg1.depth));
        } else {
            reg0.width = lowerMipDim(reg1.width);
            reg0.height = lowerMipDim(reg1.height);
            view0.SetLevels(level, 0);
            cmd.CopyTextureToTexture(tex, &view1, &reg1, tex, &view0, &reg0, CopyFlags::LINEAR_FILTER);
            DEBUG_PRINT(("src lev: %u  reg1 %u %u %u %u %u %u\n", level - 1, reg1.xoffset, reg1.yoffset, reg1.zoffset, reg1.width, reg1.height, reg1.depth));
            DEBUG_PRINT(("dst lev: %u  reg0 %u %u %u %u %u %u\n", level,     reg0.xoffset, reg0.yoffset, reg0.zoffset, reg0.width, reg0.height, reg0.depth));
        }
    }
}

void LWNGenerateMipmaps::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    int cellNum = 0;
    cellTestInit(cellWidth, cellHeight);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.submit();

    // memory pools: allocate enough space for the largest formats
    const int texSize = 128;
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(texSize, texSize);
    tb.SetFormat(Format::RGBA32F);
    tb.SetLevels(8);
    LWNsizeiptr texStorageSize = tb.GetPaddedStorageSize();
    MemoryPoolAllocator texAllocator(device, NULL, 6 * texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // allocate enough memory for our largest formats; 32-bit 4-component textures
    LWNuint maxMemSize = 4 * 4 * texSize * texSize;
    MemoryPoolAllocator bufAllocator(device, NULL, 2 * maxMemSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *srcBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, maxMemSize);
    BufferAddress srcBufAddr = srcBuf->GetAddress();
    Buffer *dstBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, maxMemSize);
    BufferAddress dstBufAddr = dstBuf->GetAddress();
    uint8_t * srcMem = (uint8_t *) srcBuf->Map();
    uint8_t * dstMem = (uint8_t *) dstBuf->Map();

    TextureView copyView;
    copyView.SetDefaults().SetLevels(0, 1);

    CopyRegion copyRegion = { 0, 0, 0, 1, 1, 1 };

    int x, y;
    bool result = true;
    Texture *tex;

    // TEXTURE_1D
    tb.SetTarget(TextureTarget::TARGET_1D);
    tb.SetFormat(Format::RG32F);
    tb.SetSize1D(texSize);
    tex = texAllocator.allocTexture(&tb);
    const LWNfloat tex1d_r[] = {15, -67, 32, 1024.25, -856.125, 12, -28, 88};
    const LWNfloat tex1d_r_result = 27.515625;
    const LWNfloat tex1d_g[] = {64, -32};
    const LWNfloat tex1d_g_result = 16;
    for (x = 0; x < texSize; x++) {
        LWNfloat *mem = (LWNfloat*) srcMem;
        mem[x*2+0] = tex1d_r[(x*8)/texSize]; // R
        mem[x*2+1] = tex1d_g[(x*2)/texSize]; // G
    }
    copyRegion.width = texSize;
    copyView.SetLevels(0, 1);
    queueCB.CopyBufferToTexture(srcBufAddr, tex, &copyView, &copyRegion, CopyFlags::NONE);
    generateMipmaps2D(device, queue, queueCB, tex, texSize, 1, 0, 7);
    copyView.SetLevels(7, 1);
    copyRegion.width = 1;
    queueCB.CopyTextureToBuffer(tex, &copyView, &copyRegion, dstBufAddr, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    DEBUG_PRINT(("tex1d: %1.6f %1.6f\n", ((LWNfloat*)dstMem)[0], ((LWNfloat*)dstMem)[1]));
    result = (((LWNfloat*)dstMem)[0] == tex1d_r_result) &&
             (((LWNfloat*)dstMem)[1] == tex1d_g_result);
    DisplayResult(queueCB, cellNum++, result);
    texAllocator.freeTexture(tex);

    // TEXTURE_2D
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(texSize, texSize);
    tb.SetFormat(Format::R8SN);
    tex = texAllocator.allocTexture(&tb);
    const int8_t tex2d[] = {32, 72, -16, -56}; // 2x2 matrix
    const int8_t tex2d_result = 8; // average
    for (y = 0; y < texSize; y++) {
        for (x = 0; x < texSize; x++) {
            int8_t *mem = (int8_t*) srcMem;
            mem[y * texSize + x] = tex2d[((y*2)/texSize)*2 + (x*2)/texSize];
        }
    }
    copyRegion.width = texSize;
    copyRegion.height = texSize;
    copyView.SetLevels(0, 1);
    queueCB.CopyBufferToTexture(srcBufAddr, tex, &copyView, &copyRegion, CopyFlags::NONE);
    generateMipmaps2D(device, queue, queueCB, tex, texSize, texSize, 0, 7);
    copyRegion.width = 1;
    copyRegion.height = 1;
    copyView.SetLevels(7, 1);
    queueCB.CopyTextureToBuffer(tex, &copyView, &copyRegion, dstBufAddr, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    DEBUG_PRINT(("tex2d: %d\n", ((int8_t*)dstMem)[0]));
    result = ((int8_t*)dstMem)[0] == tex2d_result;
    DisplayResult(queueCB, cellNum++, result);
    texAllocator.freeTexture(tex);

    // TEXTURE_2D_ARRAY
    tb.SetTarget(TextureTarget::TARGET_2D_ARRAY);
    tb.SetFormat(Format::RGBA8);
    tb.SetSize3D(texSize, texSize, 2);
    tex = texAllocator.allocTexture(&tb);
    const uint8_t tex2d_array_a[] = {32, 64, 0, 64}; // 2x2 matrix
    const uint8_t tex2d_array_a_result = 40; // average
    const uint8_t tex2d_array_b[] = {128, 160, 192, 64}; // 2x2 matrix
    const uint8_t tex2d_array_b_result = 136; // average
    for (y = 0; y < texSize; y++) {
        for (x = 0; x < texSize; x++) {
            srcMem[(y*texSize+x)*4 + 0] = tex2d_array_a[((y*2)/texSize)*2 + (x*2)/texSize]; // R
            srcMem[(y*texSize+x)*4 + 1] = tex2d_array_a[((y*2)/texSize)*2 + (x*2)/texSize]; // G
            srcMem[(y*texSize+x)*4 + 2] = tex2d_array_a[((y*2)/texSize)*2 + (x*2)/texSize]; // B
            srcMem[(y*texSize+x)*4 + 3] = tex2d_array_b[((y*2)/texSize)*2 + (x*2)/texSize]; // A
        }
    }
    copyRegion.width = texSize;
    copyRegion.height = texSize;
    copyView.SetLevels(0, 1);
    queueCB.CopyBufferToTexture(srcBufAddr, tex, &copyView, &copyRegion, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    for (y = 0; y < texSize; y++) {
        for (x = 0; x < texSize; x++) {
            srcMem[(y*texSize+x)*4 + 0] = tex2d_array_b[((y*2)/texSize)*2 + (x*2)/texSize]; // R
            srcMem[(y*texSize+x)*4 + 1] = tex2d_array_a[((y*2)/texSize)*2 + (x*2)/texSize]; // G
            srcMem[(y*texSize+x)*4 + 2] = tex2d_array_b[((y*2)/texSize)*2 + (x*2)/texSize]; // B
            srcMem[(y*texSize+x)*4 + 3] = tex2d_array_a[((y*2)/texSize)*2 + (x*2)/texSize]; // A
        }
    }
    copyRegion.zoffset = 1;
    queueCB.CopyBufferToTexture(srcBufAddr, tex, &copyView, &copyRegion, CopyFlags::NONE);
    generateMipmaps2D(device, queue, queueCB, tex, texSize, texSize, 0, 7);
    generateMipmaps2D(device, queue, queueCB, tex, texSize, texSize, 1, 7);
    copyRegion.zoffset = 0;
    copyRegion.width = 1;
    copyRegion.height = 1;
    copyView.SetLevels(7, 1);
    queueCB.CopyTextureToBuffer(tex, &copyView, &copyRegion, dstBufAddr, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    DEBUG_PRINT(("tex2d_array layer 0: %d %d %d %d\n", dstMem[0], dstMem[1], dstMem[2], dstMem[3]));
    result = (dstMem[0] == tex2d_array_a_result) &&
             (dstMem[1] == tex2d_array_a_result) &&
             (dstMem[2] == tex2d_array_a_result) &&
             (dstMem[3] == tex2d_array_b_result);
    DisplayResult(queueCB, cellNum++, result);

    copyRegion.zoffset = 1;
    queueCB.CopyTextureToBuffer(tex, &copyView, &copyRegion, dstBufAddr, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    DEBUG_PRINT(("tex2d_array layer 1: %d %d %d %d\n", dstMem[0], dstMem[1], dstMem[2], dstMem[3]));
    result = (dstMem[0] == tex2d_array_b_result) &&
             (dstMem[1] == tex2d_array_a_result) &&
             (dstMem[2] == tex2d_array_b_result) &&
             (dstMem[3] == tex2d_array_a_result);
    DisplayResult(queueCB, cellNum++, result);
    texAllocator.freeTexture(tex);

    // Future: TEXTURE_LWBEMAP
    // Future: TEXTURE_LWBEMAP_ARRAY
    // These should both behave exactly like TEXTURE_2D_ARRAY, so they are
    // omitted for now.

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    bufAllocator.freeBuffer(srcBuf);
    bufAllocator.freeBuffer(dstBuf);
}


OGTEST_CppTest(LWNGenerateMipmaps, lwn_copy_image_mipmaps, );
