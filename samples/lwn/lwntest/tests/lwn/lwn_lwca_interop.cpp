/*
* Copyright (c) 2021 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include "lwntest_cpp.h"
#include "cmdline.h"
#include "lwn_utils.h"

#include "lwca.h"
#include "lwdaLWN.h"

#include <array>
#include <vector>

#define DEBUG_MODE 0
#if DEBUG_MODE
#define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
#define DEBUG_PRINT(x)
#endif

using namespace lwn;

class LWNLWDAInteropTest
{
public:
    enum TestMode { BUFFER, TEXTURE };
    enum TestCase { LWDA_TO_LWN, LWN_TO_LWDA, BOTH };

    LWNLWDAInteropTest(TestMode mode, TestCase testCase = BOTH) : m_mode{ mode }, m_testCase{ testCase }
    {}

    LWNTEST_CppMethods();

private:
    const TestMode    m_mode;
    const TestCase    m_testCase;
};

namespace
{
    struct TexFormat
    {
        Format          format;
        LWarray_format  lwFormat;
        uint32_t        numComponents;
        uint32_t        texelSize;
    };

    const std::array<TexFormat, 21> TEX_FORMATS = { { {Format::R8,       LW_AD_FORMAT_UNSIGNED_INT8,  1,  1},
                                                      {Format::R8I,      LW_AD_FORMAT_SIGNED_INT8,    1,  1},
                                                      {Format::R16SN,    LW_AD_FORMAT_SIGNED_INT16,   1,  2},
                                                      {Format::R16UI,    LW_AD_FORMAT_UNSIGNED_INT16, 1,  2},
                                                      {Format::R32F,     LW_AD_FORMAT_FLOAT,          1,  4},
                                                      {Format::R32UI,    LW_AD_FORMAT_UNSIGNED_INT32, 1,  4},
                                                      {Format::R32I,     LW_AD_FORMAT_SIGNED_INT32,   1,  4},
                                                      {Format::RG8,      LW_AD_FORMAT_UNSIGNED_INT8,  2,  2},
                                                      {Format::RG8I,     LW_AD_FORMAT_SIGNED_INT8,    2,  2},
                                                      {Format::RG16,     LW_AD_FORMAT_UNSIGNED_INT16, 2,  4},
                                                      {Format::RG16I ,   LW_AD_FORMAT_SIGNED_INT16,   2,  4},
                                                      {Format::RG32F,    LW_AD_FORMAT_FLOAT,          2,  8},
                                                      {Format::RG32UI,   LW_AD_FORMAT_UNSIGNED_INT32, 2,  8},
                                                      {Format::RG32I,    LW_AD_FORMAT_SIGNED_INT32,   2,  8},
                                                      {Format::RGBA8,    LW_AD_FORMAT_UNSIGNED_INT8,  4,  4},
                                                      {Format::RGBA8I,   LW_AD_FORMAT_SIGNED_INT8,    4,  4},
                                                      {Format::RGBA16,   LW_AD_FORMAT_UNSIGNED_INT16, 4,  8},
                                                      {Format::RGBA16I,  LW_AD_FORMAT_SIGNED_INT16,   4,  8},
                                                      {Format::RGBA32F,  LW_AD_FORMAT_FLOAT,          4, 16},
                                                      {Format::RGBA32UI, LW_AD_FORMAT_UNSIGNED_INT32, 4, 16},
                                                      {Format::RGBA32I,  LW_AD_FORMAT_SIGNED_INT32,   4, 16},
                                                  } };

    const std::array<LWNmemoryPoolFlags, 3> MEM_TYPES = { { LWN_MEMORY_POOL_TYPE_GPU_ONLY,
                                                            LWN_MEMORY_POOL_TYPE_CPU_COHERENT,
                                                            LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT } };

    struct TexSize
    {
        int width;
        int height;
    };

    const std::array<TexSize, 3> TEX_SIZES = { { {32, 16}, {64, 64}, {100, 120} } };

    const std::array<size_t, 5> BUFFER_SIZES = { { 8, 24, 128, 200, 512 } };

    class LwdaInteropBufferTest
    {
    public:
        LwdaInteropBufferTest() : m_device{ DeviceState::GetActive()->getDevice() },
                                  m_queue{ DeviceState::GetActive()->getQueue() },
                                  m_queueCB{ DeviceState::GetActive()->getQueueCB() }
        {}

        void run();

    private:
        void runLwn2Lwda(std::vector<bool>& results);
        void runLwda2Lwn(std::vector<bool>& results);

        void fillBuffer(uint8_t* buffer, size_t bufferSize) const;

        lwn::Device*        m_device;
        lwn::Queue*         m_queue;
        QueueCommandBuffer& m_queueCB;
    };


    class LwdaInteropTexTest
    {
    public:
        LwdaInteropTexTest() : m_device{ DeviceState::GetActive()->getDevice() },
                               m_queue{ DeviceState::GetActive()->getQueue() },
                               m_queueCB{ DeviceState::GetActive()->getQueueCB() }
        {}

        void run(LWNLWDAInteropTest::TestCase testCase);

        static const uint8_t QUAD_SIZE = 8;

    private:
        void runLwn2Lwda(std::vector<bool>& results);
        void runLwda2Lwn(std::vector<bool>& results);

        void fillBuffer(int w, int h, const TexFormat& format, void* buffer) const;

        size_t getMaxTexStorageSize() const;

        Device*             m_device;
        Queue*              m_queue;
        QueueCommandBuffer& m_queueCB;
    };


    inline float clamp_01(float x)
    {
        if (x >= 0.0f) {
            return (x > 1.0f) ? 1.0f : x;
        } else {
            return 0.0f;
        }
    }

    // Helper class to colwert floating point colors im the range of [0.0,1.0] to integral type colors.
    template<typename T>
    class TexColor
    {
    public:
        TexColor(float r, float g, float b, float a) : m_color { static_cast<T>((clamp_01(r) * maxValue) + 0.5f),
                                                                 static_cast<T>((clamp_01(g) * maxValue) + 0.5f),
                                                                 static_cast<T>((clamp_01(b) * maxValue) + 0.5f),
                                                                 static_cast<T>((clamp_01(a) * maxValue) + 0.5f) }
        {
        }

        T operator[](size_t i) const { return m_color[i]; }

    private:
        static constexpr double maxValue = static_cast<double>(std::numeric_limits<T>::max());
        const T m_color[4];
    };


    void drawResults(QueueCommandBuffer& queueCB, std::vector<bool>& results)
    {
        queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
        queueCB.ClearColor(0, 0.0f, 0.0f, 0.8f);

        const float aspectRatio = static_cast<float>(lwrrentWindowWidth) / lwrrentWindowHeight;
        const float cellsY = ceil(sqrt(results.size() * 1.0f / aspectRatio));
        const float cellsX = aspectRatio * cellsY;

        const int cellWidth = lwrrentWindowWidth / static_cast<int>(cellsX);
        const int cellHeight = lwrrentWindowHeight / static_cast<int>(cellsY);
        int cellX = 0;
        int cellY = 0;

        for (auto b : results) {
            queueCB.SetViewportScissor(cellX + 1, cellY + 1, cellWidth - 1, cellHeight - 1);

            if (b) {
                queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f);
            } else {
                queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f);
            }

            cellX += cellWidth;
            if ((cellX + cellWidth) > lwrrentWindowWidth) {
                cellX = 0;
                cellY += cellHeight;
            }
        }
    }

    inline bool verifyBuffer(uint8_t* srcBuffer, uint8_t* resultBuffer, size_t size)
    {
        // Do a byte by byte compare instead of memcmp for easier debugging on failures.
        for (size_t i = 0; i < size; ++i) {
            if (srcBuffer[i] != resultBuffer[i]) {
                DEBUG_PRINT(("Missmatch found at offset: %d. Expected %d got %d", i, srcBuffer[i], resultBuffer[i]));
                return false;
            }
        }
        return true;
    }

    // Write checkboard pattern to buffer.
    template<typename T>
    void writeBuffer(int w, int h, int numComponents, T* buffer)
    {
        assert(numComponents <= 4);

        const int pitch = w * numComponents;

        for (int y = 0; y < h; ++y) {
            const int idx_y = y / LwdaInteropTexTest::QUAD_SIZE;

            for (int x = 0; x < w; ++x) {
                const int idx_x = x / LwdaInteropTexTest::QUAD_SIZE;
                const int colorIdx = ((idx_x + idx_y) & 1);

                const TexColor<T> color[2] = { TexColor<T>((float)x / w, (float)y / h, 0.0, 1.0f),
                                               TexColor<T>(0.0f, 0.0f, 1.0f, 1.0f) };

                T* ptr = &buffer[y * pitch + x * numComponents];

                for (int i = 0; i < numComponents; ++i) {
                    ptr[i] = color[colorIdx][i];
                }
            }
        }
    }

    void LwdaInteropTexTest::fillBuffer(int w, int h, const TexFormat& format, void* buffer) const
    {
        switch (format.lwFormat)
        {
        case LW_AD_FORMAT_UNSIGNED_INT8:
            writeBuffer(w, h, format.numComponents, static_cast<uint8_t*>(buffer));
            break;
        case LW_AD_FORMAT_SIGNED_INT8:
            writeBuffer(w, h, format.numComponents, static_cast<int8_t*>(buffer));
            break;
        case LW_AD_FORMAT_UNSIGNED_INT16:
            writeBuffer(w, h, format.numComponents, static_cast<uint16_t*>(buffer));
            break;
        case LW_AD_FORMAT_SIGNED_INT16:
            writeBuffer(w, h, format.numComponents, static_cast<int16_t*>(buffer));
            break;
        case LW_AD_FORMAT_UNSIGNED_INT32:
            writeBuffer(w, h, format.numComponents, static_cast<uint32_t*>(buffer));
            break;
        case LW_AD_FORMAT_SIGNED_INT32:
            writeBuffer(w, h, format.numComponents, static_cast<int32_t*>(buffer));
            break;
        case LW_AD_FORMAT_FLOAT:
            writeBuffer(w, h, format.numComponents, static_cast<float*>(buffer));
            break;
        default:
            assert(!"Unsupported format");
        }
    }

    size_t LwdaInteropTexTest::getMaxTexStorageSize() const
    {
        const TexFormat* maxTex = &TEX_FORMATS[0];
        for (auto tf : TEX_FORMATS) {
            if (tf.texelSize > maxTex->texelSize) {
                maxTex = &tf;
            }
        }

        TextureBuilder tb;
        tb.SetDefaults()
          .SetDevice(m_device)
          .SetTarget(TextureTarget::TARGET_2D)
          .SetFormat(maxTex->format);

        size_t maxStorageSize = 0;
        for (auto& ts : TEX_SIZES) {
            tb.SetSize2D(ts.width, ts.height);
            maxStorageSize = LW_MAX(tb.GetPaddedStorageSize(), maxStorageSize);
        }

        return maxStorageSize;
    }

#define LW_CALL_AND_CHECK_RESULT(cmd, action) \
    if (cmd != LWDA_SUCCESS) { \
        results.push_back(false);  \
        action; \
    }

    void LwdaInteropTexTest::run(LWNLWDAInteropTest::TestCase testCase)
    {
        std::vector<bool> results;

        if (testCase == LWNLWDAInteropTest::TestCase::LWN_TO_LWDA) {
            runLwn2Lwda(results);
        } else if (testCase == LWNLWDAInteropTest::TestCase::LWDA_TO_LWN) {
            runLwda2Lwn(results);
        }

        drawResults(m_queueCB, results);
        m_queueCB.submit();
        m_queue->Finish();
    }

    void LwdaInteropTexTest::runLwda2Lwn(std::vector<bool>& results)
    {
        Sync lwnLwdaReady;
        lwnLwdaReady.Initialize(m_device);

        LWevent lwLwdaReady = NULL;
        LW_CALL_AND_CHECK_RESULT(lwEventCreateFromLWNSync(&lwLwdaReady, (LWNsync*)&lwnLwdaReady, 0), return);

        BufferBuilder bb;
        bb.SetDevice(m_device).SetDefaults();

        TextureView view;
        view.SetDefaults();

        size_t maxRequiredPoolSize = getMaxTexStorageSize();

        // Create source buffer
        std::vector<uint8_t> srcBuffer(maxRequiredPoolSize, 0);

        MemoryPoolAllocator bufferAllocator(m_device, NULL, maxRequiredPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

        for (auto memType : MEM_TYPES) {

            MemoryPoolAllocator texAllocator(m_device, NULL, maxRequiredPoolSize, memType);

            for (const auto& t : TEX_FORMATS) {

                TextureBuilder tb;
                tb.SetDevice(m_device).SetDefaults()
                  .SetTarget(TextureTarget::TARGET_2D)
                  .SetFormat(t.format);

                for (const auto& ts : TEX_SIZES) {

                    tb.SetSize2D(ts.width, ts.height);

                    const size_t texSize = ts.width * ts.height * t.texelSize;

                    // Fill buffer with checkerboard pattern
                    fillBuffer(ts.width, ts.height, t, srcBuffer.data());

                    // Create result buffer
                    Buffer* resultBuffer = bufferAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texSize);
                    // Create shared texture
                    Texture* tex = texAllocator.allocTexture(&tb);

                    LWarray lwTex;
                    LW_CALL_AND_CHECK_RESULT(lwLWNtextureGetArray(&lwTex, (LWNtexture*)tex, (LWNtextureView*)&view, LW_GRAPHICS_REGISTER_FLAGS_NONE), continue);

                    // Copy checkerboard from buffer to texture
                    LWDA_MEMCPY2D cp = {};
                    cp.srcMemoryType = LW_MEMORYTYPE_HOST;
                    cp.srcHost = srcBuffer.data();
                    cp.srcPitch = ts.width * t.texelSize;
                    cp.dstMemoryType = LW_MEMORYTYPE_ARRAY;
                    cp.dstArray = lwTex;
                    cp.WidthInBytes = ts.width * t.texelSize;
                    cp.Height = ts.height;

                    LW_CALL_AND_CHECK_RESULT(lwMemcpy2DAsync(&cp, 0), continue);
                    LW_CALL_AND_CHECK_RESULT(lwEventRecord(lwLwdaReady, 0), continue);

                    // Wait for LWCA to be ready
                    m_queueCB.WaitSync(&lwnLwdaReady);

                    // Copy texture to result buffer
                    CopyRegion region = { 0, 0, 0, ts.width, ts.height, 1 };
                    m_queueCB.CopyTextureToBuffer(tex, NULL, &region, resultBuffer->GetAddress(), CopyFlags::NONE);
                    m_queueCB.submit();
                    m_queue->Finish();

                    if (memType == LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT) {
                        // In case of non-coherent memory we need to ilwalidate the
                        // cache after the GPU has written the memory.
                        resultBuffer->IlwalidateMappedRange(0, texSize);
                    }

                    results.push_back(verifyBuffer(srcBuffer.data(), static_cast<uint8_t*>(resultBuffer->Map()), texSize));

                    bufferAllocator.freeBuffer(resultBuffer);
                    texAllocator.freeTexture(tex);
                }
            }
        }

        lwEventDestroy(lwLwdaReady);
        lwnLwdaReady.Finalize();
    }

    void LwdaInteropTexTest::runLwn2Lwda(std::vector<bool>& results)
    {
        Sync lwnReady;
        lwnReady.Initialize(m_device);

        LWevent lwLwnReady = NULL;
        LW_CALL_AND_CHECK_RESULT(lwEventCreateFromLWNSync(&lwLwnReady, (LWNsync*)&lwnReady, 0), return);

        BufferBuilder bb;
        bb.SetDevice(m_device).SetDefaults();

        TextureView view;
        view.SetDefaults();

        size_t maxRequiredPoolSize = getMaxTexStorageSize();

        // Create source buffer used to fill the texture
        MemoryPoolAllocator bufferAllocator(m_device, NULL, maxRequiredPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
        Buffer* srcBuffer = bufferAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, maxRequiredPoolSize);

        for (auto memType : MEM_TYPES) {
            MemoryPoolAllocator texAllocator(m_device, NULL, maxRequiredPoolSize, memType);

            for (const auto& t : TEX_FORMATS) {

                TextureBuilder tb;
                tb.SetDevice(m_device).SetDefaults()
                  .SetTarget(TextureTarget::TARGET_2D)
                  .SetFormat(t.format);

                for (const auto& ts : TEX_SIZES) {

                    tb.SetSize2D(ts.width, ts.height);

                    const size_t texSize = ts.width * ts.height * t.texelSize;
                    Texture* tex = texAllocator.allocTexture(&tb);

                    LWarray lwTex;
                    LW_CALL_AND_CHECK_RESULT(lwLWNtextureGetArray(&lwTex, (LWNtexture*)tex, (LWNtextureView*)&view, LW_GRAPHICS_REGISTER_FLAGS_NONE), continue);

                    // Fill buffer with checkerboard pattern
                    fillBuffer(ts.width, ts.height, t, srcBuffer->Map());

                    // Create result buffer
                    std::vector<uint8_t> resultBuffer(texSize, 0);

                    // Copy checkerboard from buffer to texture
                    CopyRegion region = { 0, 0, 0, ts.width, ts.height, 1 };
                    m_queueCB.CopyBufferToTexture(srcBuffer->GetAddress(), tex, NULL, &region, CopyFlags::NONE);
                    m_queueCB.FenceSync(&lwnReady, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
                    m_queueCB.submit();
                    m_queue->Flush();

                    // Copy texture to result buffer
                    LWDA_MEMCPY2D cp = {};
                    cp.srcMemoryType = LW_MEMORYTYPE_ARRAY;
                    cp.srcArray = lwTex;
                    cp.dstMemoryType = LW_MEMORYTYPE_HOST;
                    cp.dstHost = resultBuffer.data();
                    cp.dstPitch = ts.width * t.texelSize;
                    cp.WidthInBytes = ts.width * t.texelSize;
                    cp.Height = ts.height;

                    // Wait for LWN to be ready
                    LW_CALL_AND_CHECK_RESULT(lwStreamWaitEvent(NULL, lwLwnReady, 0), continue);
                    // Since we copy from Array to Host memory, lwMemcpy2D only returns once the copy is completed
                    LW_CALL_AND_CHECK_RESULT(lwMemcpy2D(&cp), continue);

                    results.push_back(verifyBuffer(static_cast<uint8_t*>(srcBuffer->Map()), resultBuffer.data(), texSize));

                    texAllocator.freeTexture(tex);
                }
            }
        }
        lwEventDestroy(lwLwnReady);
        lwnReady.Finalize();
    }



    void LwdaInteropBufferTest::fillBuffer(uint8_t* buffer, size_t bufferSize) const
    {
        for (size_t i = 0; i < bufferSize; i++) {
            buffer[i] = i & 0xff;
        }
    }

    void LwdaInteropBufferTest::run()
    {
        std::vector<bool> results;

        runLwn2Lwda(results);
        runLwda2Lwn(results);

        drawResults(m_queueCB, results);
        m_queueCB.submit();
        m_queue->Finish();
    }

    void LwdaInteropBufferTest::runLwda2Lwn(std::vector<bool>& results)
    {
        // Create sync to notify that LWCA is done writing the buffer.
        Sync lwnLwdaReady;
        lwnLwdaReady.Initialize(m_device);;

        LWevent lwLwdaReady = NULL;
        LW_CALL_AND_CHECK_RESULT(lwEventCreateFromLWNSync(&lwLwdaReady, (LWNsync*)&lwnLwdaReady, 0), return);

        // Create sync to notify LWCA that LWN has finished clearing the buffer.
        Sync lwnReady;
        lwnReady.Initialize(m_device);

        LWevent lwLwnReady = NULL;
        LW_CALL_AND_CHECK_RESULT(lwEventCreateFromLWNSync(&lwLwnReady, (LWNsync*)&lwnReady, 0), return);

        BufferBuilder bb;
        bb.SetDevice(m_device).SetDefaults();

        for (auto sharedBufferSize : BUFFER_SIZES) {
            std::vector<uint8_t> lwSrcBuffer(sharedBufferSize);

            fillBuffer(lwSrcBuffer.data(), sharedBufferSize);

            // Create intermediate buffer to readback results from a GPU only LWNbuffer
            MemoryPoolAllocator bufferAllocator(m_device, NULL, sharedBufferSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
            Buffer* rbBuffer = bufferAllocator.allocBuffer(&bb, BufferAlignBits(BUFFER_ALIGN_COPY_WRITE_BIT | BUFFER_ALIGN_COPY_READ_BIT), sharedBufferSize);

            // Create LWCA source buffer on the GPU
            LWdeviceptr lwGpuSrcBuffer = NULL;
            LW_CALL_AND_CHECK_RESULT(lwMemAlloc(&lwGpuSrcBuffer, sharedBufferSize), return);

            // Fill LWCA source buffer on the GPU
            LW_CALL_AND_CHECK_RESULT(lwMemcpyHtoD(lwGpuSrcBuffer, lwSrcBuffer.data(), sharedBufferSize), return);

            for (auto memType : MEM_TYPES) {
                MemoryPoolAllocator sharedAllocator(m_device, NULL, sharedBufferSize, memType);
                Buffer* sharedBuffer = sharedAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, sharedBufferSize);

                m_queueCB.ClearBuffer(sharedBuffer->GetAddress(), sharedBufferSize, 0);
                m_queueCB.FenceSync(&lwnReady, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
                m_queueCB.submit();
                m_queue->Flush();

                // Import the shared LWNbuffer to LWCA
                LWdeviceptr lwLwnBufferPtr = 0;
                size_t lwBufferSize = 0;
                LW_CALL_AND_CHECK_RESULT(lwLWNbufferGetPointer(&lwLwnBufferPtr, (LWNbuffer*)sharedBuffer, 0, &lwBufferSize), continue);

                // Wait for LWN's buffer clear to be ready
                LW_CALL_AND_CHECK_RESULT(lwStreamWaitEvent(NULL, lwLwnReady, 0), continue);

                // Copy LWCA GPU buffer to the shared buffer
                LW_CALL_AND_CHECK_RESULT(lwMemcpyDtoDAsync(lwLwnBufferPtr, lwGpuSrcBuffer, lwBufferSize, 0), continue);
                LW_CALL_AND_CHECK_RESULT(lwEventRecord(lwLwdaReady, 0), continue);

                // Make sure the LWCA copy finished before accessing the buffer.
                m_queueCB.WaitSync(&lwnLwdaReady);

                // Readback and verify the content of the shared buffer. If the shared buffer is
                // GPU only, we need to copy it to a CPU accessible buffer first. If the shared
                // buffer is CPU accessible we can map it right away.
                Buffer* resultBuffer = sharedBuffer;

                if (memType == LWN_MEMORY_POOL_TYPE_GPU_ONLY) {
                    m_queueCB.CopyBufferToBuffer(sharedBuffer->GetAddress(), rbBuffer->GetAddress(), sharedBufferSize, CopyFlags::NONE);
                    resultBuffer = rbBuffer;
                }

                m_queueCB.submit();
                m_queue->Finish();

                if (memType == LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT) {
                    resultBuffer->IlwalidateMappedRange(0, sharedBufferSize);
                }

                results.push_back(verifyBuffer(lwSrcBuffer.data(), static_cast<uint8_t*>(resultBuffer->Map()), sharedBufferSize));
            }

            lwMemFree(lwGpuSrcBuffer);
        }
        lwEventDestroy(lwLwnReady);
        lwEventDestroy(lwLwdaReady);

        lwnReady.Finalize();
        lwnLwdaReady.Finalize();
    }

    void LwdaInteropBufferTest::runLwn2Lwda(std::vector<bool>& results)
    {
        // Create sync to notify LWCA that lwn is done writing the buffer.
        Sync lwnReady;
        lwnReady.Initialize(m_device);

        LWevent lwLwnReady = NULL;
        LW_CALL_AND_CHECK_RESULT(lwEventCreateFromLWNSync(&lwLwnReady, (LWNsync*)&lwnReady, 0), return);

        BufferBuilder bb;
        bb.SetDevice(m_device).SetDefaults();

        for (auto sharedBufferSize : BUFFER_SIZES) {

            MemoryPoolAllocator bufferAllocator(m_device, NULL, sharedBufferSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
            Buffer* srcBuffer = bufferAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, sharedBufferSize);

            fillBuffer(static_cast<uint8_t*>(srcBuffer->Map()), sharedBufferSize);

            for (auto memType : MEM_TYPES) {
                MemoryPoolAllocator sharedAllocator(m_device, NULL, sharedBufferSize, memType);
                Buffer* sharedBuffer = sharedAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, sharedBufferSize);

                // Import buffer and sync to LWCA
                LWdeviceptr lwLwnBufferPtr = 0;
                size_t lwBufferSize = 0;
                LW_CALL_AND_CHECK_RESULT(lwLWNbufferGetPointer(&lwLwnBufferPtr, (LWNbuffer*)sharedBuffer, 0, &lwBufferSize), continue);

                m_queueCB.ClearBuffer(sharedBuffer->GetAddress(), sharedBufferSize, 0);

                // Use LWN to fill the shared buffer.
                m_queueCB.CopyBufferToBuffer(srcBuffer->GetAddress(), sharedBuffer->GetAddress(), sharedBufferSize, 0);
                m_queueCB.FenceSync(&lwnReady, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
                m_queueCB.submit();
                m_queue->Flush();

                // Wait for LWN to be ready
                LW_CALL_AND_CHECK_RESULT(lwStreamWaitEvent(NULL, lwLwnReady, 0), continue);

                std::vector<uint8_t> lwResultBuffer(sharedBufferSize);
                LW_CALL_AND_CHECK_RESULT(lwMemcpyDtoH(lwResultBuffer.data(), lwLwnBufferPtr, sharedBufferSize), continue);

                // lwMemcpyDtoH is synchronous with respect to the host, so we can read the data immediately.
                results.push_back(verifyBuffer(static_cast<uint8_t*>(srcBuffer->Map()), lwResultBuffer.data(), sharedBufferSize));
            }
        }
        m_queue->Finish();

        lwEventDestroy(lwLwnReady);
        lwnReady.Finalize();
    }

#undef LW_CALL_AND_CHECK_RESULT

} // namespace

lwString LWNLWDAInteropTest::getDescription() const
{
    lwStringBuf sb;
    if (m_mode == TestMode::BUFFER) {
        sb << "Basic test to verify the LWN-LWCA buffer interop functionality. The test creates "
              "buffers of different sizes in memory pools that use different flags.These buffers "
              "are shared with LWCA.In one pass the buffer is written by LWN and LWCA reads "
              "the buffer and checks if it contains the expected results.In the other pass "
              "the buffer is written by LWDAand LWN reads the buffer and checks if it contains "
              "the expected results. ";
    } else if (m_mode == TestMode::TEXTURE) {
        sb << "Basic test to verify the LWN_LWDA texture interop functionality. The test creates "
              "textures of different formats and dimensions in memory pools that use different "
              "memory types. These textures are shared with LWCA. ";
        if (m_testCase == TestCase::LWDA_TO_LWN) {
            sb << "LWCA will write to the textures and LWN will copy the texture into a buffer "
                  "and verify if it contains the expected values.";
        } else if (m_testCase == TestCase::LWN_TO_LWDA) {
            sb << "LWN will write to the textures and LWCA will copy them into host memory and "
                  "verify if they contain the expected values.";
        }

    }
    sb << "For each subtest that succeeds a green quad is drawn if the subtest fails a red "
          "quad is drawn.\n";

    return sb.str();
}

int LWNLWDAInteropTest::isSupported() const
{
    if (!lwdaEnabled) {
        return 0;
    }

    return lwogCheckLWNAPIVersion(55, 11);
}


void LWNLWDAInteropTest::doGraphics() const
{
    LWresult lwresult;
    LWdevice lwdev;
    LWcontext lwctx;

    // Initialize the LWCA subsystem.
    lwresult = lwDeviceGet(&lwdev, 0);
    if (lwresult != LWDA_SUCCESS) {
        LWNFailTest();
        return;
    }
    lwresult = lwCtxCreate(&lwctx, 0, lwdev);
    if (lwresult != LWDA_SUCCESS) {
        LWNFailTest();
        return;
    }

    if (m_mode == LWNLWDAInteropTest::TestMode::BUFFER) {
        LwdaInteropBufferTest bufferTest;
        bufferTest.run();
    } else {
        LwdaInteropTexTest texTest;
        texTest.run(m_testCase);
    }

    // Clean up LWCA resources.
    lwresult = lwCtxDestroy(lwctx);
    if (lwresult != LWDA_SUCCESS) {
        LWNFailTest();
        return;
    }
}

OGTEST_CppTest(LWNLWDAInteropTest, lwn_lwda_buffer_interop,    (LWNLWDAInteropTest::TestMode::BUFFER));
OGTEST_CppTest(LWNLWDAInteropTest, lwn_lwda_tex2d_c2n_interop, (LWNLWDAInteropTest::TestMode::TEXTURE, LWNLWDAInteropTest::TestCase::LWDA_TO_LWN));
OGTEST_CppTest(LWNLWDAInteropTest, lwn_lwda_tex2d_n2c_interop, (LWNLWDAInteropTest::TestMode::TEXTURE, LWNLWDAInteropTest::TestCase::LWN_TO_LWDA));
