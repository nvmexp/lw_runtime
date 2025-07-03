/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

#include <stdio.h>
#include <wayland-client.h>
#include "wayland_win.h"
#include "wayland_input.h"
#include "lwnExt/lwnExt_waylandWin.h"
#include "lwnrender.h"
#include "error.h"
#include "args.h"

static void redraw(void *data, struct wl_callback *callback, uint32_t time);    // Forward decl
static const struct wl_callback_listener frame_listener =
{
    redraw,
};

static void redraw(void *data, struct wl_callback *callback, uint32_t time)
{
    static struct wl_callback *frame_callback = NULL;

    auto *winfo = (WaylandWindowInfo *) data;

    if (frame_callback) wl_callback_destroy(frame_callback);
    frame_callback = wl_surface_frame(winfo->wl_surface);
    wl_callback_add_listener(frame_callback, &frame_listener, winfo);

    RenderFrame(&eventInfo);
}

int main(int argc, char *argv[])
{
    g_args.ParseArgs(argc, argv);

    int windowWidth  = 640;
    int windowHeight = 480;

    if (g_args.m_fullScreen) {
        windowWidth = 0;
        windowHeight = 0;
    }
#ifdef WAYLAND_LWN_PRESENT
    printf("Wayland program to test out effect of CPU/GPU loading on framerate\n"
           "    q/rightmouse : quits the program\n"
           "    leftmouse    : positions the animating square\n\n");
#else
    printf("Wayland example with Wayland dispatch loop & mouse/keyboard interaction\n"
           "    q/rightmouse : quits the program\n"
           "    leftmouse    : positions the animating square\n"
           "    if cursor is within window, green software cursor tracks mouse position\n\n");
#endif
    fflush(stdout);

    // Setup mouse/keyboard event listener procs. Must be done before
    // Wayland initalization.
    InitializeSeatListener();

    WaylandWin *win = WaylandWin::GetInstance();
    if (win == NULL) {
        Error("Cannot obtain window interface");
        return 1;
    }

    WaylandWindowInfo *winfo = (WaylandWindowInfo *)
                    win->CreateWindow("wayland-lwn-example", windowWidth, windowHeight);
    if (!winfo) {
        Error("Cannot create window");
        return 1;
    }
    if (g_args.m_debug) {
        printf("Window size: (%d, %d)\n", windowWidth, windowHeight);
    }
    InitGraphics(winfo, windowWidth, windowHeight);

    redraw(winfo, NULL, 0);

    // Wayland dispatch loop. eventInfo structure captures mouse/keyboard information.
    // This sample application is designed to quit either when "q" key is pressed or
    // the right mouse key is pressed.

    while (wl_display_dispatch(winfo->wl_display) != -1)
    {
        if ((eventInfo.quit) || (eventInfo.rightPressed)) {
            break;
        }
    }

    // Ensure graphics objects are destroyed before Wayland window is destroyed.
    TerminateGraphics();
    win->DestroyWindow(winfo);

    return 0;
}
