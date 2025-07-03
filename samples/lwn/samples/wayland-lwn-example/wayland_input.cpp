/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

#include "wayland_input.h"

#include <wayland-client.h>
#include <xkbcommon/xkbcommon.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>

static struct xkb_context *xkb_context = NULL;
static struct xkb_state   *xkb_state   = NULL;
static struct xkb_keymap  *keymap      = NULL;

EventInfo eventInfo;

// By default disable printing of all events
#define printf(...)

static void pointer_enter (void *data, struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
    printf ("pointer enter\n");
    eventInfo.active = true;
}
static void pointer_leave (void *data, struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface) {
    printf ("pointer leave\n");
    eventInfo.active = false;
}
static void pointer_motion (void *data, struct wl_pointer *pointer, uint32_t time, wl_fixed_t x, wl_fixed_t y) {
    printf ("pointer motion %f %f\n", wl_fixed_to_double(x), wl_fixed_to_double(y));
    eventInfo.active = true;
    eventInfo.mouseX = wl_fixed_to_double(x);
    eventInfo.mouseY = wl_fixed_to_double(y);
}
static void pointer_button (void *data, struct wl_pointer *pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
    printf ("pointer button (button %d, state %d)\n", button, state);
    if (button == 272) {
        eventInfo.leftPressed = true;
        eventInfo.leftMouseX = eventInfo.mouseX;
        eventInfo.leftMouseY = eventInfo.mouseY;
    } else if (button == 273) {
        eventInfo.rightPressed = true;
        eventInfo.rightMouseX = eventInfo.mouseX;
        eventInfo.rightMouseY = eventInfo.mouseY;
    }
}
static void pointer_axis (void *data, struct wl_pointer *pointer, uint32_t time, uint32_t axis, wl_fixed_t value) {
    printf ("pointer axis\n");
}

static struct wl_pointer_listener pointer_listener =
{
    &pointer_enter,
    &pointer_leave,
    &pointer_motion,
    &pointer_button,
    &pointer_axis
};

static void keyboard_keymap (void *data, struct wl_keyboard *keyboard, uint32_t format, int32_t fd, uint32_t size)
{
    if (xkb_context == NULL) {
        xkb_context = xkb_context_new (XKB_CONTEXT_NO_FLAGS);
    }

    char *keymap_string = (char *) mmap (NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    xkb_keymap_unref (keymap);
    keymap = xkb_keymap_new_from_string (xkb_context, keymap_string, XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);
    munmap (keymap_string, size);
    close (fd);

    xkb_state_unref (xkb_state);
    xkb_state = xkb_state_new (keymap);
}

static void keyboard_enter (void *data, struct wl_keyboard *keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys) {}

static void keyboard_leave (void *data, struct wl_keyboard *keyboard, uint32_t serial, struct wl_surface *surface) {}

static void keyboard_key (void *data, struct wl_keyboard *keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state)
{
    if (state == WL_KEYBOARD_KEY_STATE_PRESSED) {
        xkb_keysym_t keysym = xkb_state_key_get_one_sym (xkb_state, key+8);
        uint32_t utf32 = xkb_keysym_to_utf32 (keysym);
        if (utf32) {
            if (utf32 >= 0x21 && utf32 <= 0x7E) {
                printf ("the key %c was pressed\n", (char)utf32);
                if (utf32 == 'q') {
                    eventInfo.quit = true;
                }
            }
            else {
                printf ("the key U+%04X was pressed\n", utf32);
            }
        }
        else {
            char name[64];
            xkb_keysym_get_name (keysym, name, 64);
            printf ("the key %s was pressed\n", name);
        }
    }
}
static void keyboard_modifiers (void *data, struct wl_keyboard *keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
    xkb_state_update_mask (xkb_state, mods_depressed, mods_latched, mods_locked, 0, 0, group);
}

static struct wl_keyboard_listener keyboard_listener =
{
    &keyboard_keymap,
    &keyboard_enter,
    &keyboard_leave,
    &keyboard_key,
    &keyboard_modifiers
};

static void seat_capabilities (void *data, struct wl_seat *seat, uint32_t capabilities)
{
    if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
        struct wl_pointer *pointer = wl_seat_get_pointer (seat);
        wl_pointer_add_listener (pointer, &pointer_listener, NULL);
    }
    if (capabilities & WL_SEAT_CAPABILITY_KEYBOARD) {
        struct wl_keyboard *keyboard = wl_seat_get_keyboard (seat);
        wl_keyboard_add_listener (keyboard, &keyboard_listener, NULL);
    }
}

static struct wl_seat_listener seat_listener =
{
    seat_capabilities
};

// Initialize the global seat listener struct
// (back door hack in waylan_win.cpp)
extern struct wl_seat_listener *s_pwl_seat_listener;

void InitializeSeatListener()
{
    s_pwl_seat_listener = &seat_listener;
}
