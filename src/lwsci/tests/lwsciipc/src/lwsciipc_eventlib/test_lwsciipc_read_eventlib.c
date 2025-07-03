/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "test_lwsciipc_read_events.h"
#include "test_lwsciipc_read_log.h"
#include "test_lwsciipc_read_eventlib.h"
#include "test_lwsciipc_read_events_json.h"

static int32_t dump_eventlib(char *filename, void* mem_addr, uint32_t mem_size)
{
    struct eventlib_ctx ctx;
    void *msg = NULL;
    uint32_t size = 0;
    int fd = -1;
    bool is_eventlib_init = false;
    int32_t ret = -1;

    memset(&ctx, 0, sizeof(ctx));

    ctx.direction = EVENTLIB_DIRECTION_READER;
    ctx.w2r_shm = mem_addr;
    ctx.w2r_shm_size = mem_size;

    msg = malloc(mem_size);
    if (msg == NULL) {
        ret = -ENOMEM;
        fprintf(stderr, "memory allocation failed, size=%u\n", mem_size);
        goto exit;
    }

    ret = eventlib_init(&ctx);
    if (ret != 0) {
        fprintf(stderr, "eventlib_init() failed for %s\n", filename);
        goto exit;
    }
    is_eventlib_init = true;

    fd = open(filename,
              O_WRONLY | O_TRUNC | O_CREAT,
              S_IRWXU | S_IRWXG | S_IRWXO);
    if (fd == -1) {
        ret = -1;
        fprintf(stderr, "Cannot open file, filename=%s", filename);
        goto exit;
    }

    size = mem_size;
    ret = eventlib_read(&ctx, (void *)(msg), &size, NULL);
    if (ret == 0) {
        write(fd, msg, size);
    } else {
        fprintf(stderr, "eventlib_read failed");
    }

exit:
    if (fd == -1) {
        close(fd);
    }
    if (msg != NULL) {
        free(msg);
    }
    if (is_eventlib_init) {
        eventlib_close(&ctx);
    }

    return ret;
}

void dump_lwsciipc_eventlibs(char *testapp_filename)
{
    struct lwsciip_read_test_log_ctx *ptr_ctx = &g_el_handle.el_ctx;

    if ((g_el_handle.init_status & EVENTLIB_INIT_DONE) != 0) {
        dump_eventlib(testapp_filename,
                      ptr_ctx->ctx.w2r_shm, ptr_ctx->ctx.w2r_shm_size);
    }
}

void lwsciipc_eventlib_close(void)
{
    if ((g_el_handle.init_status & EVENTLIB_INIT_DONE) != 0) {
        eventlib_close(&g_el_handle.el_ctx.ctx);
    }
    if ((g_el_handle.init_status & SHARD_MEMORY_REGISTER_DONE) != 0) {
        (void)eventlib_unregister_shmem(LWSCIIP_READ_TEST_EVENT_PROVIDER_NAME,
                                        g_el_handle.el_ctx.ctx.w2r_shm,
                                        EVENT_LOG_SIZE);
    }
    g_el_handle.init_status = 0;
}

int32_t lwsciipc_eventlib_init(void)
{
    int32_t i;
    int32_t ret;
    struct lwsciip_read_test_log_ctx *ptr_ctx = &g_el_handle.el_ctx;

    ptr_ctx->get_timestamp = eventlib_get_timer_counter;

    for (i = 0; i < LWSCIIP_READ_TEST_EVENT_BITMASK_SIZE; i++) {
        ptr_ctx->bitmask[i] = ~0U;
    }

    ptr_ctx->ctx.w2r_shm = eventlib_register_shmem(LWSCIIP_READ_TEST_EVENT_PROVIDER_NAME,
                                                   EVENT_LOG_SIZE,
                                                   (char *)test_lwsciipc_read_events_json,
                                                   test_lwsciipc_read_events_json_len);
    if (ptr_ctx->ctx.w2r_shm == NULL) {
        fprintf(stderr, "eventlib_register_shmem() failed\n");
        return -1;
    }
    g_el_handle.init_status |= SHARD_MEMORY_REGISTER_DONE;

    memset(ptr_ctx->ctx.w2r_shm, 0, EVENT_LOG_SIZE);
    ptr_ctx->ctx.w2r_shm_size = EVENT_LOG_SIZE;
    ptr_ctx->ctx.direction = EVENTLIB_DIRECTION_WRITER;
    ptr_ctx->ctx.r2w_shm = NULL;
    ptr_ctx->ctx.r2w_shm_size = 0;
    ptr_ctx->ctx.flags = 0;
    ptr_ctx->ctx.num_buffers = 2;

    ret = eventlib_init(&ptr_ctx->ctx);
    if (ret == 0) {
        g_el_handle.init_status |= EVENTLIB_INIT_DONE;
    } else {
        fprintf(stderr, "eventlib_init() failed for read\n");
        lwsciipc_eventlib_close();
    }

    return ret;
}

lwsciipc_eventlib_handle_t g_el_handle = {
    .init_status = 0,
};
