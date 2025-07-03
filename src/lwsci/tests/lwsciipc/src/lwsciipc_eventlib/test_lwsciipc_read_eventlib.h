/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef __TEST_LWSCIIPC_READ_EVENTLIB_H__
#define __TEST_LWSCIIPC_READ_EVENTLIB_H__

#if (LW_IS_SAFETY == 0) && defined(IVC_EVENTLIB)
#include "test_lwsciipc_read_events.h"
#include "test_lwsciipc_read_log.h"

#define SHARD_MEMORY_REGISTER_DONE 0x0001U
#define EVENTLIB_INIT_DONE 0x0002U
#define EVENT_LOG_SIZE (8 * 1024 * 1024)
#define LWSCIIPC_READ_TEST_DUMP_FILENAME "/tmp/dump_test_lwsciipc_read.bin"

typedef struct {
    struct lwsciip_read_test_log_ctx el_ctx;
    uint32_t init_status;
} lwsciipc_eventlib_handle_t;

extern void lwsciipc_eventlib_close(void);
extern int32_t lwsciipc_eventlib_init(void);
extern void dump_lwsciipc_eventlibs(char *testapp_filename);
extern lwsciipc_eventlib_handle_t g_el_handle;

#define LWSCIIPC_EVENTLIB_LOG(_type, ...) do { \
    if ((g_el_handle.init_status & EVENTLIB_INIT_DONE) != 0) { \
        if ((g_el_handle.el_ctx.ctx.priv != NULL) && \
            (lwsciip_read_test_##_type##_check(&g_el_handle.el_ctx) > 0)) { \
            lwsciip_read_test_##_type##_write(&g_el_handle.el_ctx, 0, __VA_ARGS__); } \
    } \
} while(false)
#else
#define LWSCIIPC_EVENTLIB_LOG(_type, ...)
#endif

#endif
