/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
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

class LWNMultibindComputeCpp
{
public:
    LWNTEST_CppMethods();
};

lwString LWNMultibindComputeCpp::getDescription() const
{
    return "Test binds different combinations of SSBOs, UBOs, textures and\n"
            "images to compute shader, using multibind feature. It then\n"
            "tests if the binding was correct by comparing bound resources'\n"
            "values against expected values for each combination.";
}

int LWNMultibindComputeCpp::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 9);
}

void LWNMultibindComputeCpp::doGraphics() const
{
    // vars
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int tileSize = 8;
    const GLuint groupX = tileSize, groupY = tileSize;
    const GLuint sharedMemorySizeBytes = groupX * groupY * 16;
    const int gridX = 1, gridY = 1, gridZ = 1;
    const int MXT = 4;
    const float vals[4] = {1.0, 0.6, 0.3, 0.0};
    const int outputImgWidth = 4*tileSize + 3;
    const int outputImgHeight = tileSize;

    // nearest sampler
    Sampler* nearestSampler;
    LWNuint nearestSamplerID;
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    nearestSampler = sb.CreateSampler();
    nearestSamplerID = nearestSampler->GetRegisteredID();

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();

    // output image and texture (and clear it)
    Texture* outputTexture;
    TextureHandle outputTextureHandle;
    ImageHandle outputImageHandle;
    tb.SetFlags(TextureFlags::IMAGE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(outputImgWidth, outputImgHeight);
    tb.SetFormat(Format::RGBA8);
    LWNsizeiptr outputTexStorageSize = tb.GetPaddedStorageSize();
    MemoryPool* outputTexGpuMemPool = device->CreateMemoryPool(NULL, outputTexStorageSize, MemoryPoolType::GPU_ONLY);
    MemoryPool* outBufCpuMemPool = device->CreateMemoryPool(NULL, outputTexStorageSize, MemoryPoolType::CPU_COHERENT);
    outputTexture = tb.CreateTextureFromPool(outputTexGpuMemPool, 0);
    Buffer* outTexBuff = bb.CreateBufferFromPool(outBufCpuMemPool, 0, outputTexStorageSize);
    dt::u8lwec4* outPtr = static_cast<dt::u8lwec4*>(outTexBuff->Map());
    for (int i=0; i<outputImgWidth; i++) {
        for (int j=0; j<outputImgHeight; j++) {
            outPtr[j*outputImgWidth+i] = dt::u8lwec4(0.0,0.0,0.0,1.0);
        }
    }
    CopyRegion cr = {0,0,0,outputImgWidth,outputImgHeight,1};
    queueCB.CopyBufferToTexture(outTexBuff->GetAddress(), outputTexture, 0, &cr, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    outTexBuff->Free();
    outputTextureHandle = device->GetTextureHandle(outputTexture->GetRegisteredTextureID(), nearestSamplerID);
    outputImageHandle = device->GetImageHandle(g_lwnTexIDPool->RegisterImage(outputTexture));

    // setup UBOs
    struct UBOBlock_t
    {
        float val[4];
    };
    BufferRange uboRanges[MXT];
    LWNint uboAlignment = 0;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
    LWNint uboStorageSize = ((sizeof(UBOBlock_t) + uboAlignment - 1) / uboAlignment) * uboAlignment;
    MemoryPool* uboMemPool = device->CreateMemoryPool(NULL, MXT*uboStorageSize, MemoryPoolType::CPU_COHERENT);
    for (int i=0; i<MXT; i++) {
        uboRanges[i].address = uboMemPool->GetBufferAddress() + i*uboStorageSize;
        uboRanges[i].size = uboStorageSize;
        UBOBlock_t* uboPtr = reinterpret_cast<UBOBlock_t*>((char*)uboMemPool->Map()+i*uboStorageSize);
        uboPtr->val[0] = vals[i];
        uboPtr->val[1] = 0.0;
        uboPtr->val[2] = 0.0;
        uboPtr->val[3] = 0.0;
    }

    // SSBOs (alignment is 4 bytes, same as sizeof(float) )
    BufferRange ssboRanges[MXT+1];
    MemoryPool* ssboMemPool = device->CreateMemoryPool(NULL, 4*4 + 3*4, MemoryPoolType::CPU_COHERENT);
    char* ssboMemPtr = static_cast<char*>(ssboMemPool->Map());
    for (int i=0; i<MXT; i++)
    {
        ssboRanges[i].address = ssboMemPool->GetBufferAddress() + i*4;
        ssboRanges[i].size = 4;
        float* ssboPtr = reinterpret_cast<float*>(ssboMemPtr+i*4);
        *ssboPtr = vals[i];
    }
    ssboRanges[MXT].address = ssboMemPool->GetBufferAddress() + 16;
    ssboRanges[MXT].size = 12;
    float* expectedValuesSSBOPtr = reinterpret_cast<float*>(ssboMemPtr + 16);

    // setup images and textures
    Texture* teximg_textures[MXT];
    TextureHandle teximg_texHandles[MXT];
    ImageHandle teximg_imgHandles[MXT];
    tb.SetDefaults();
    tb.SetFlags(TextureFlags::IMAGE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(1,1);
    tb.SetFormat(Format::RGBA32F);
    size_t texStorageSize = tb.GetStorageSize();
    MemoryPool* bufCpuMemPool = device->CreateMemoryPool(NULL, MXT*texStorageSize, MemoryPoolType::CPU_COHERENT);
    MemoryPool* texGpuMemPool = device->CreateMemoryPool(NULL, MXT*texStorageSize, MemoryPoolType::GPU_ONLY);
    for (int i=0; i<MXT; i++) {
        teximg_textures[i] = tb.CreateTextureFromPool(texGpuMemPool, texStorageSize*i);
        teximg_texHandles[i] = device->GetTextureHandle(teximg_textures[i]->GetRegisteredTextureID(), nearestSamplerID);
        teximg_imgHandles[i] = device->GetImageHandle(g_lwnTexIDPool->RegisterImage(teximg_textures[i]));
        Buffer* textureBuffer = bb.CreateBufferFromPool(bufCpuMemPool, texStorageSize*i, texStorageSize);
        float* ptr = static_cast<float*>(textureBuffer->Map());
        ptr[0] = vals[i];
        ptr[1] = 0.0;
        ptr[2] = 0.0;
        ptr[3] = 0.0;
        CopyRegion cr = {0,0,0,1,1,1};
        queueCB.CopyBufferToTexture(textureBuffer->GetAddress(), teximg_textures[i], 0, &cr, CopyFlags::NONE);
        queueCB.submit();
        queue->Finish();
        textureBuffer->Free();
    }

    // compute shader
    Program* pgm;
    lwShader cs;
    cs = ComputeShader(440);
    cs <<
          // input UBOs
          "layout(binding=0, std140) uniform Block0 {\n"
          "  vec4 ubo_val0;\n"
          "};\n"
          "layout(binding=1, std140) uniform Block1 {\n"
          "  vec4 ubo_val1;\n"
          "};\n"
          "layout(binding=2, std140) uniform Block2 {\n"
          "  vec4 ubo_val2;\n"
          "};\n"
          // input SSBOs
          "layout(std430, binding = 0) buffer SSBO0 {\n"
          "  float ssboVals0[];\n"
          "};\n"
          "layout(std430, binding = 1) buffer SSBO1 {\n"
          "  float ssboVals1[];\n"
          "};\n"
          "layout(std430, binding = 2) buffer SSBO2 {\n"
          "  float ssboVals2[];\n"
          "};\n"
          // input Images
          "layout(binding=0,rgba32f) uniform image2D img0;\n"
          "layout(binding=1,rgba32f) uniform image2D img1;\n"
          "layout(binding=2,rgba32f) uniform image2D img2;\n"
          // input Textures
          "layout (binding=0) uniform sampler2D tex0;\n"
          "layout (binding=1) uniform sampler2D tex1;\n"
          "layout (binding=2) uniform sampler2D tex2;\n"
          // input expected values
          "layout(std430, binding = 3) buffer SSBOExpected {\n"
          "  float eVals[];\n"
          "};\n"
          // output Image
          "layout(binding=3,rgba8) uniform image2D outputImage;\n"

          "vec4 sameAsExpected(vec3 v)\n"
          "{\n"
          "  if (v.r==eVals[0] && v.g==eVals[1] && v.b==eVals[2])\n"
          "    return vec4(0.0, 1.0, 0.0, 1.0);\n"
          "  return vec4(1.0, 0.0, 0.0, 1.0);\n"
          "}\n"

          "void main()\n"
          "{\n"
          "  ivec2 lID = ivec2(gl_LocalIlwocationID.xy);\n"
          "  vec2 a = vec2(lID);\n"
          "  vec2 b = vec2(gl_WorkGroupSize.xy);\n"
          // ubo test
          "  vec3 uboVal = vec3(ubo_val0.r, ubo_val1.r, ubo_val2.r);\n"
          "  imageStore(outputImage, lID, sameAsExpected(uboVal));\n"
          // ssbo test
          "  vec3 ssboVal = vec3(ssboVals0[0], ssboVals1[0], ssboVals2[0]);\n"
          "  imageStore(outputImage, lID + ivec2(9,0), sameAsExpected(ssboVal));\n"
          // image test
          "  vec3 imgVal = vec3(0.0);\n"
          "  imgVal.r = imageLoad(img0, ivec2(0,0)).r;\n"
          "  imgVal.g = imageLoad(img1, ivec2(0,0)).r;\n"
          "  imgVal.b = imageLoad(img2, ivec2(0,0)).r;\n"
          "  imageStore(outputImage, lID + ivec2(18,0), sameAsExpected(imgVal));\n"
          // texture test
          "  vec2 uv = vec2(a.x/b.x, a.y/b.y);\n"
          "  vec3 texVal = vec3(0.0);\n"
          "  texVal.r = texture(tex0,uv).r;\n"
          "  texVal.g = texture(tex1,uv).r;\n"
          "  texVal.b = texture(tex2,uv).r;\n"
          "  imageStore(outputImage, lID + ivec2(27,0), sameAsExpected(texVal));\n"
          "}\n";
    cs.setCSGroupSize(groupX,groupY);
    cs.setCSSharedMemory(sharedMemorySizeBytes);
    pgm = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgm, cs);

    // display shader
    Program* pgmDisplay;
    VertexShader vs(430);
    FragmentShader fs(430);
    vs <<
        "out vec2 uv;\n"
        "void main() {\n"
        "  vec2 pos; "
        "  if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);"
        "  if (gl_VertexID == 1) pos = vec2(1.0, -1.0);"
        "  if (gl_VertexID == 2) pos = vec2(1.0, 1.0);"
        "  if (gl_VertexID == 3) pos = vec2(-1.0, 1.0);"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  uv = pos * 0.5 + 0.5;\n"
        "}\n";
    fs <<
        "layout(binding=0) uniform sampler2D tex;\n"
        "in vec2 uv;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, uv);\n"
        "}\n";
    pgmDisplay = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgmDisplay, vs, fs);

    // clear
    queueCB.ClearColor();
    queueCB.submit();

    // execute compute and display results
    int execCnt = 0;
    for (int max=MXT-1; max>0; max--) {
        for (int first=0; first<=MXT-1-max; first++) {
            queueCB.BindProgram(pgm, ShaderStageBits::COMPUTE);
            // reset vals (aka bind resources that have all zero values)
            for (int i=0; i<MXT-1; i++) {
                queueCB.BindStorageBuffer(ShaderStage::COMPUTE, i, ssboRanges[3].address, ssboRanges[3].size);
                queueCB.BindUniformBuffer(ShaderStage::COMPUTE, i, uboRanges[3].address, uboRanges[3].size);
                queueCB.BindImage(ShaderStage::COMPUTE, i, teximg_imgHandles[3]);
                queueCB.BindTexture(ShaderStage::COMPUTE, i, teximg_texHandles[3]);
            }
            // multibind resources
            queueCB.BindStorageBuffers(ShaderStage::COMPUTE, first, max, ssboRanges);
            queueCB.BindUniformBuffers(ShaderStage::COMPUTE, first, max, uboRanges);
            queueCB.BindTextures(ShaderStage::COMPUTE, first, max, teximg_texHandles);
            queueCB.BindImages(ShaderStage::COMPUTE, first, max, teximg_imgHandles);
            // setup and bind SSBO that holds expected values
            expectedValuesSSBOPtr[0] = 0.0;
            expectedValuesSSBOPtr[1] = 0.0;
            expectedValuesSSBOPtr[2] = 0.0;
            for (int i=0; i<max; i++) {
                expectedValuesSSBOPtr[first+i] = vals[i];
            }
            queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 3, ssboRanges[MXT].address, ssboRanges[MXT].size);
            // bind output image
            queueCB.BindImage(ShaderStage::COMPUTE, 3, outputImageHandle);
            // run compute
            queueCB.DispatchCompute(gridX,gridY,gridZ);

            // display
            queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_TEXTURE | BarrierBits::ILWALIDATE_SHADER);
            queueCB.SetViewportScissor(0, outputImgHeight*execCnt+execCnt, outputImgWidth, outputImgHeight);
            queueCB.ClearColor(0, 1.0, 1.0, 1.0);
            queueCB.BindProgram(pgmDisplay, ShaderStageBits::ALL_GRAPHICS_BITS);
            queueCB.BindTexture(ShaderStage::FRAGMENT, 0, outputTextureHandle);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
            queueCB.submit();
            queue->Finish();

            execCnt++;
        }
    }

    // cleanup
    for (int i=0; i<MXT; i++) {
        teximg_textures[i]->Free();
    }
    outputTexture->Free();
    nearestSampler->Free();
    outputTexGpuMemPool->Free();
    outBufCpuMemPool->Free();
    uboMemPool->Free();
    ssboMemPool->Free();
    bufCpuMemPool->Free();
    texGpuMemPool->Free();
    pgm->Free();
    pgmDisplay->Free();
}

OGTEST_CppTest(LWNMultibindComputeCpp, lwn_multibind_compute, );
