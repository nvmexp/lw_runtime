/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

// High dynamic range / wide color gamut on LWN demo application.

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "GL/gl.h"
#include "GL/wglext.h"
#include "lwn/lwn_Cpp.h"
#include "lwn/lwn_CppFuncPtr.h"
#include "lwn/lwn_CppFuncPtrImpl.h"
#include "lwn/lwn_CppMethods.h"

const int WIDTH = 640;
const int HEIGHT = 480;

PFNWGLCHOOSEPIXELFORMATARBPROC pWglChoosePixelFormatARB = nullptr;

// The debug layer will warn about the creation of a memory pool without backing system memory,
// which is only required on the NX device.
#define USE_DEBUG_LAYER 0

// There is a chick-and-egg problem to solve. We need WGL_ARB_pixel_format to create a window that
// supports HDR/WCG. However, we need a window to create a GL context that provides access to WGL
// extensions. Thus, we use a temporary dummy window that provides access to the extension.
class DummyWindow
{
public:
    DummyWindow()
    {
        mWindow = CreateWindow(L"static", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        int pixelFormat;
        PIXELFORMATDESCRIPTOR pfd = {0};
        pfd.lwersion = 1;
        pfd.dwFlags = PFD_SUPPORT_OPENGL;
        HDC dc = GetDC(mWindow);
        pixelFormat = ChoosePixelFormat(dc, &pfd);
        SetPixelFormat(dc, pixelFormat, &pfd);
        wglMakeLwrrent(dc, wglCreateContext(dc));
    }

    ~DummyWindow()
    {
        wglDeleteContext(wglGetLwrrentContext());
        DestroyWindow(mWindow);
    }

private:
    HWND mWindow;
};

// The "real" application window.
class Window
{
public:
    Window()
    {
        mInst = GetModuleHandle(NULL);

        WNDCLASS wc = {0};
        wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        wc.lpfnWndProc = DefWindowProc;
        wc.hInstance = mInst;
        wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
        wc.hLwrsor = LoadLwrsor(NULL, IDC_ARROW);
        wc.lpszClassName = WindowClassName();
        RegisterClass(&wc);
        mWindow = CreateWindow(WindowClassName(), 0, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 20, 20,
                               WIDTH, HEIGHT, 0, 0, 0, 0);
        SetWindowText(mWindow, L"HDR/WCG Demo");
        mHdc = GetDC(mWindow);

        // This is the important of enabling HDR/WCG on Windows. Use a 16-bit floating point
        // RGBA pixel format for the default frame buffer of a donor OpenGL context.
        // Note that the Win32 PIXELFORMATDESCRIPTOR interface does not support floating-point
        // formats; we must use WGL_ARB_pixel_format.
        int attribs[] = {
            WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
            WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
            WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
            WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_FLOAT_ARB,
            WGL_RED_BITS_ARB, 16,
            WGL_GREEN_BITS_ARB, 16,
            WGL_BLUE_BITS_ARB, 16,
            WGL_ALPHA_BITS_ARB, 16,
            0
        };
        int pixelFormat;
        UINT numFormats;
        pWglChoosePixelFormatARB(mHdc, attribs, nullptr, 1, &pixelFormat, &numFormats);
        PIXELFORMATDESCRIPTOR pfd = {0};
        DescribePixelFormat(mHdc, pixelFormat, sizeof(pfd), &pfd);
        SetPixelFormat(mHdc, pixelFormat, &pfd);
        wglMakeLwrrent(mHdc, wglCreateContext(mHdc));
    }

    ~Window()
    {
        wglDeleteContext(wglGetLwrrentContext());
        DestroyWindow(WindowFromDC(mHdc));
        UnregisterClass(WindowClassName(), mInst);
    }

    bool HandleUi()
    {
        if (!IsWindow(mWindow)) {
            return false;
        }
        if (GetAsyncKeyState(VK_ESCAPE)) {
            return false;
        }
        MSG msg;
        while (PeekMessage(&msg, mWindow, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_CLOSE) {
                return false;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        SwapBuffers(mHdc);
        return true;
    }

private:
    inline const wchar_t* WindowClassName() { return L"HDRtest"; }
    HINSTANCE mInst;
    HWND mWindow;
    HDC mHdc;
};

class LwnGraphics
{
public:
    LwnGraphics()
    {
        // LWN setup, including bootstrapping, window swap chain, and queue/command buffer
        using namespace lwn;
        auto pBootstrapLoader = (BootstrapLoaderFunc)wglGetProcAddress("rq34nd2ffz");
        auto pGetProcAddress =
            (DeviceGetProcAddressFunc)pBootstrapLoader("lwnDeviceGetProcAddress");
        lwnLoadCPPProcs(nullptr, pGetProcAddress);
        DeviceBuilder deviceBuilder;
        deviceBuilder.SetDefaults();
#if USE_DEBUG_LAYER
        deviceBuilder.SetFlags(DeviceFlagBits::DEBUG_ENABLE_LEVEL_4);
#endif
        mDevice.Initialize(&deviceBuilder);
#if USE_DEBUG_LAYER
        mDevice.InstallDebugCallback(DebugCallback, this, true);
#endif
        lwnLoadCPPProcs(&mDevice, pGetProcAddress);
        QueueBuilder queueBuilder;
        queueBuilder.SetDefaults().SetDevice(&mDevice);
        mQueue.Initialize(&queueBuilder);
        TextureBuilder textureBuilder;
        textureBuilder.SetDefaults()
                      .SetDevice(&mDevice)
                      .SetFormat(Format::RGBA16F) // Match the format of the donor OpenGL context.
                      .SetSize2D(WIDTH, HEIGHT)
                      .SetFlags(TextureFlags::DISPLAY);
        size_t texSize = textureBuilder.GetStorageSize();
        MemoryPoolBuilder memPoolBuilder;
        static constexpr int NUM_BUFFERS = 2;
        // NON-PORTABLE: Windows doesn't require system memory to back memory pools, but NX would
        // need it.
        memPoolBuilder.SetDefaults()
                      .SetDevice(&mDevice)
                      .SetFlags(MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::CPU_NO_ACCESS)
                      .SetStorage(nullptr, texSize * NUM_BUFFERS);
        mWindowTexturePool.Initialize(&memPoolBuilder);
        for (int i = 0; i < NUM_BUFFERS; ++i) {
            textureBuilder.SetStorage(&mWindowTexturePool, i * texSize);
            mWindowTextures[i].Initialize(&textureBuilder);
        }
        WindowBuilder windowBuilder;
        Texture* const windowTexturePtrs[2] = {mWindowTextures, mWindowTextures + 1};
        // SetNativeWindow with a non-NULL HWND is not lwrrently supported. Instead, rely on the
        // window associated with the current OpenGL context.
        windowBuilder.SetDefaults()
                     .SetDevice(&mDevice)
                     .SetTextures(2, windowTexturePtrs);
        mWindow.Initialize(&windowBuilder);
        mWindowTextureAvailableSync.Initialize(&mDevice);
        mControlMemory.resize(2 * CB_SIZE);
        memPoolBuilder.SetDefaults()
                      .SetDevice(&mDevice)
                      .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_UNCACHED)
                      // See CB_SIZE and OOM callback definitions.
                      .SetStorage(nullptr, 2 * CB_SIZE);
        mCommandMemory.Initialize(&memPoolBuilder);
        mCommandBuffer.Initialize(&mDevice);
        mCommandBuffer.SetMemoryCallback(OutOfMemoryCallback);
        mCommandBuffer.SetMemoryCallbackData(this);
    }

    ~LwnGraphics()
    {
        mQueue.Finish();
        mCommandBuffer.Finalize();
        mCommandMemory.Finalize();
        mWindow.Finalize();
        mWindowTextures[1].Finalize();
        mWindowTextures[0].Finalize();
        mWindowTexturePool.Finalize();
        mQueue.Finalize();
        mDevice.Finalize();
    }

    void DrawAndPresent()
    {
        // Draw HDR/WCG content. This application renders animated color blocks with channels
        // outside the [0,1] range, and the same blocks clamped to that range. The result is what
        // appears to be four rows of blocks on a display with HDR/WCG enabled, but only two rows on
        // other displays. The only draw command used is ClearColor, which lets us avoid the
        // complexity of setting up a shader pipeline.
        using namespace lwn;
        int textureIndex;
        mWindow.AcquireTexture(&mWindowTextureAvailableSync, &textureIndex);
        mQueue.WaitSync(&mWindowTextureAvailableSync);
        mCommandBuffer.BeginRecording();
        const Texture* windowTexture = mWindowTextures + textureIndex;
        mCommandBuffer.SetRenderTargets(1, &windowTexture, nullptr, nullptr, nullptr);
        // Draw an in-gamut region and an out-of-gamut region. Should look like a solid color on an
        // LDR display and two colors on HDR.
        static constexpr int NUM_SEGMENTS = 32;
        assert(WIDTH % NUM_SEGMENTS == 0);
        static constexpr int SEGMENT_WIDTH = WIDTH / NUM_SEGMENTS;
        assert(HEIGHT % 4 == 0);
        for (int i = 0; i < NUM_SEGMENTS; ++i) {
            int segLeft = i * WIDTH / NUM_SEGMENTS;
            // High dynamic range
            float angle = mBaseAngle - i * 6.28319f / NUM_SEGMENTS;
            float sa = sin(angle);
            float luminance = sa + 1.0f;
            float color[] = {luminance, luminance, luminance, 1.0f};
            mCommandBuffer.SetScissor(segLeft, 0, SEGMENT_WIDTH, HEIGHT / 4);
            mCommandBuffer.ClearColor(0, color, ClearColorMask::RGBA);
            // Low dynamic range
            luminance = std::min(luminance, 1.0f);
            color[0] = color[1] = color[2] = luminance;
            mCommandBuffer.SetScissor(segLeft, HEIGHT / 4, SEGMENT_WIDTH, HEIGHT / 4);
            mCommandBuffer.ClearColor(0, color, ClearColorMask::RGBA);
            // Wide color gamut
            // Rotate through a circle in scRGB color space, centered at (0.5, 0.5, 0.5) and
            // perpendilwlar to (1, 1, 1)
            static const float RADIUS = 1.4f;
            float ca = cos(angle);
            color[0] = 0.5f + 0.707107f * RADIUS * sa - 0.408248f * RADIUS * ca;
            color[1] = 0.5f - 0.707107f * RADIUS * sa - 0.408248f * RADIUS * ca;
            color[2] = 0.5f + 0.816497f * RADIUS * ca;
            mCommandBuffer.SetScissor(segLeft, HEIGHT / 2, SEGMENT_WIDTH, HEIGHT / 4);
            mCommandBuffer.ClearColor(0, color, ClearColorMask::RGBA);
            // Narrow color gamut
            color[0] = std::max(0.0f, std::min(1.0f, color[0]));
            color[1] = std::max(0.0f, std::min(1.0f, color[1]));
            color[2] = std::max(0.0f, std::min(1.0f, color[2]));
            mCommandBuffer.SetScissor(segLeft, 3 * HEIGHT / 4, SEGMENT_WIDTH, HEIGHT / 4);
            mCommandBuffer.ClearColor(0, color, ClearColorMask::RGBA);
        }
        CommandHandle commandHandle = mCommandBuffer.EndRecording();
        mQueue.SubmitCommands(1, &commandHandle);
        mQueue.PresentTexture(&mWindow, textureIndex);
        mBaseAngle += 0.01f;
        if (mBaseAngle > 6.28319f) {
            mBaseAngle -= 6.28319f;
        }
    }

private:
    // A size large enough to fully satisfy command or control memory demands at any given point.
    static constexpr size_t CB_SIZE = 65536;

    static void LWNAPIENTRY OutOfMemoryCallback(lwn::CommandBuffer* cmdBuf,
                                                lwn::CommandBufferMemoryEvent::Enum event,
                                                size_t minSize, void* callbackData)
    {
        LwnGraphics* self = static_cast<LwnGraphics*>(callbackData);
        self->HandleOutOfMemory(cmdBuf, event, minSize);
    }

    void HandleOutOfMemory(lwn::CommandBuffer* cmdBuf, lwn::CommandBufferMemoryEvent::Enum event,
                           size_t minSize)
    {
        // A chunk of memory that is double CB_SIZE is allocated for each type of command buffer
        // memory. To satisfy the allocation requests, ping-pong between each half of each chunk
        // of memory.
        if (event == lwn::CommandBufferMemoryEvent::OUT_OF_COMMAND_MEMORY) {
            cmdBuf->AddCommandMemory(&mCommandMemory, mCommandMemoryOffset, CB_SIZE);
            mCommandMemoryOffset = CB_SIZE - mCommandMemoryOffset;
        } else {
            cmdBuf->AddControlMemory(mControlMemory.data() + mControlMemoryOffset, CB_SIZE);
            mControlMemoryOffset = CB_SIZE - mControlMemoryOffset;
        }
    }

#if USE_DEBUG_LAYER
    static void LWNAPIENTRY DebugCallback(lwn::DebugCallbackSource::Enum source,
                                          lwn::DebugCallbackType::Enum type, int id,
                                          lwn::DebugCallbackSeverity::Enum severity,
                                          const char* message, void* userParam)
    {
        std::cout << "LWN debug: " << message << std::endl;
    }
#endif

    lwn::Device mDevice;
    lwn::Queue mQueue;
    lwn::MemoryPool mWindowTexturePool;
    lwn::Texture mWindowTextures[2];
    lwn::Window mWindow;
    lwn::Sync mWindowTextureAvailableSync;
    lwn::CommandBuffer mCommandBuffer;
    lwn::CommandHandle mCommandHandle = 0;
    std::vector<uint8_t> mControlMemory;
    int mControlMemoryOffset = 0;
    lwn::MemoryPool mCommandMemory;
    int mCommandMemoryOffset = 0;
    float mBaseAngle = 0.0f;
};

int main()
{
    {
        // A GL context is needed to get at the extension functions we want, including the one that
        // lets us initialize the "real" window.
        DummyWindow dummyWindow;
        pWglChoosePixelFormatARB =
            (PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB");
        assert(pWglChoosePixelFormatARB);
    }
    Window glWindow;
    LwnGraphics lwnGraphics;
    while (true) {
        lwnGraphics.DrawAndPresent();
        if (!glWindow.HandleUi()) {
            break;
        }
    }
    return 0;
}