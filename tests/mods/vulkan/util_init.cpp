/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
VULKAN_SAMPLE_DESCRIPTION
samples "init" utility functions
*/

#include <assert.h>
#include "util_init.hpp"

#ifdef _WIN32
static void run(struct sample_info *info) { /* Placeholder for samples that want to show dynamic content */ }

// MS-Windows event handling function:
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    struct sample_info *info = reinterpret_cast<struct sample_info *>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

    switch (uMsg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            break;
        case WM_PAINT:
            run(info);
            return 0;
        default:
            break;
    }
    return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}

DWORD WINAPI init_window_thread(LPVOID param) {
    sample_info &info = *(sample_info*)param;

    WNDCLASSEX win_class;
    assert(info.width > 0);
    assert(info.height > 0);

    info.connection = GetModuleHandle(NULL);

    // Initialize the window class structure:
    win_class.cbSize = sizeof(WNDCLASSEX);
    win_class.style = CS_HREDRAW | CS_VREDRAW;
    win_class.lpfnWndProc = WndProc;
    win_class.cbClsExtra = 0;
    win_class.cbWndExtra = 0;
    win_class.hInstance = info.connection;  // hInstance
    win_class.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    win_class.hLwrsor = LoadLwrsor(NULL, IDC_ARROW);
    win_class.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    win_class.lpszMenuName = NULL;
    win_class.lpszClassName = info.name.c_str();
    win_class.hIconSm = LoadIcon(NULL, IDI_WINLOGO);
    // Register window class:
    if (!RegisterClassEx(&win_class)) {
        // It didn't work, so try to give a useful error:
        printf("Unexpected error trying to start the application!\n");
        fflush(stdout);
        exit(1);
    }
    // Create window with the registered class:
    RECT wr = {0, 0, info.width, info.height};
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    info.window = CreateWindowEx(0,
                                 info.name.c_str(),   // class name
                                 info.name.c_str(),   // app name
                                 WS_OVERLAPPEDWINDOW |  // window style
                                     WS_VISIBLE | WS_SYSMENU,
                                 100, 100,            // x/y coords
                                 wr.right - wr.left,  // width
                                 wr.bottom - wr.top,  // height
                                 NULL,                // handle to parent
                                 NULL,                // handle to menu
                                 info.connection,     // hInstance
                                 NULL);               // no extra parameters
    if (!info.window) {
        // It didn't work, so try to give a useful error:
        printf("Cannot create a window in which to draw!\n");
        fflush(stdout);
        exit(1);
    }
    SetWindowLongPtr(info.window, GWLP_USERDATA, (LONG_PTR)&info);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}

void init_window(struct sample_info &info)
{
    CreateThread(nullptr, 0, init_window_thread, &info, 0, NULL);

    while (info.window == 0)
    {
        Sleep(0);
    }
}

void destroy_window(struct sample_info &info)
{
    PostMessage(info.window, WM_CLOSE, 0, 0);
}

#elif defined(__linux__)

#ifdef NDEBUG
static constexpr void dprintf(const char*, ...) { }
#else
#define dprintf printf
#endif

static void* EventLoop(void* arg)
{
    struct sample_info& info = *static_cast<struct sample_info*>(arg);

    while (info.keepRunning)
    {
        xcb_generic_event_t* const event = xcb_wait_for_event(info.connection);

        if (!event)
        {
            dprintf("Received nullptr event, connection to X server is broken\n");
            break;
        }

        dprintf("Received event 0x%x\n", static_cast<unsigned>(event->response_type));

        free(event);
    }

    return nullptr;
}

void init_window(struct sample_info& info)
{
    dprintf("init_window %dx%d\n", info.width, info.height);

    xcb_connection_t* const conn = xcb_connect(nullptr, nullptr);

    if (!conn)
    {
        fprintf(stderr, "Failed to connect to X server\n");
        exit(1);
    }

    xcb_screen_t* const screen = xcb_setup_roots_iterator(xcb_get_setup(conn)).data;

    const xcb_window_t window = xcb_generate_id(conn);

    uint32_t values[2] = {
        screen->black_pixel,
        XCB_EVENT_MASK_KEY_PRESS | XCB_EVENT_MASK_EXPOSURE
    };

    xcb_create_window(conn,
                      XCB_COPY_FROM_PARENT,
                      window,
                      screen->root,
                      0, 0, info.width, info.height,
                      0,
                      XCB_WINDOW_CLASS_INPUT_OUTPUT,
                      XCB_COPY_FROM_PARENT,
                      XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK,
                      values);

    xcb_change_property(conn,
                        XCB_PROP_MODE_REPLACE,
                        window,
                        XCB_ATOM_WM_NAME,
                        XCB_ATOM_STRING,
                        8,
                        info.name.size(),
                        info.name.data());

    xcb_map_window(conn, window);

    xcb_flush(conn);

    info.connection  = conn;
    info.window      = window;
    info.keepRunning = true;

    pthread_create(&info.thread, nullptr, EventLoop, &info);

    dprintf("Created window 0x%x\n", info.window);
}

void destroy_window(struct sample_info& info)
{
    dprintf("Closing window 0x%x\n", info.window);

    info.keepRunning = false;

    xcb_client_message_event_t dummyEvent = { };
    dummyEvent.response_type = XCB_CLIENT_MESSAGE;
    dummyEvent.format        = 32;
    dummyEvent.window        = info.window;

    xcb_send_event(info.connection,
                   false,
                   info.window,
                   XCB_EVENT_MASK_NO_EVENT,
                   reinterpret_cast<const char*>(&dummyEvent));

    xcb_flush(info.connection);

    pthread_join(info.thread, nullptr);

    xcb_destroy_window(info.connection, info.window);

    xcb_disconnect(info.connection);

    info.connection = nullptr;
    info.window     = 0;

    dprintf("Closed window\n");
}

#endif

void init_window_size(struct sample_info &info, int32_t default_width, int32_t default_height) {
#ifdef __ANDROID__
    AndroidGetWindowSize(&info.width, &info.height);
#else
    info.width = default_width;
    info.height = default_height;
#endif
}
