/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <ucm/api/ucm.h>
#include <common/test.h>
#include <lwca.h>
#include <lwda_runtime.h>

static ucm_event_t alloc_event, free_event;

static void lwda_mem_alloc_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    alloc_event.mem_type.address  = event->mem_type.address;
    alloc_event.mem_type.size     = event->mem_type.size;
    alloc_event.mem_type.mem_type = event->mem_type.mem_type;
}

static void lwda_mem_free_callback(ucm_event_type_t event_type,
                                   ucm_event_t *event, void *arg)
{
    free_event.mem_type.address  = event->mem_type.address;
    free_event.mem_type.size     = event->mem_type.size;
    free_event.mem_type.mem_type = event->mem_type.mem_type;
}


class lwda_hooks : public ucs::test {
protected:

    virtual void init() {
        ucs_status_t result;
        LWresult ret;
        ucs::test::init();

        /* intialize device context */
        if (lwdaSetDevice(0) != lwdaSuccess) {
            UCS_TEST_SKIP_R("can't set lwca device");
        }

        ret = lwInit(0);
        if (ret != LWDA_SUCCESS) {
            UCS_TEST_SKIP_R("can't init lwca device");
        }

        ret = lwDeviceGet(&device, 0);
        if (ret != LWDA_SUCCESS) {
            UCS_TEST_SKIP_R("can't get lwca device");
        }

        ret = lwCtxCreate(&context, 0, device);
        if (ret != LWDA_SUCCESS) {
            UCS_TEST_SKIP_R("can't create lwca context");
        }

        /* install memory hooks */
        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 0, lwda_mem_alloc_callback,
                                       reinterpret_cast<void*>(this));
        ASSERT_UCS_OK(result);

        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 0, lwda_mem_free_callback,
                                       reinterpret_cast<void*>(this));
        ASSERT_UCS_OK(result);
    }

    virtual void cleanup() {
        LWresult ret;

        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, lwda_mem_alloc_callback,
                                reinterpret_cast<void*>(this));
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE, lwda_mem_free_callback,
                                reinterpret_cast<void*>(this));

        ret = lwCtxDestroy(context);
        EXPECT_EQ(ret, LWDA_SUCCESS);

        ucs::test::cleanup();
    }


    void check_mem_alloc_events(void *ptr, size_t size,
                                int expect_mem_type = UCS_MEMORY_TYPE_LWDA)  {
        ASSERT_EQ(ptr, alloc_event.mem_type.address);
        ASSERT_EQ(size, alloc_event.mem_type.size);
        ASSERT_EQ(expect_mem_type, alloc_event.mem_type.mem_type);
    }

    void check_mem_free_events(void *ptr, size_t size,
                               int expect_mem_type = UCS_MEMORY_TYPE_LWDA) {
        ASSERT_EQ(ptr, free_event.mem_type.address);
        ASSERT_EQ(expect_mem_type, free_event.mem_type.mem_type);
    }

    LWdevice   device;
    LWcontext  context;
};

UCS_TEST_F(lwda_hooks, test_lwMem_Alloc_Free) {
    LWresult ret;
    LWdeviceptr dptr, dptr1;

    /* small allocation */
    ret = lwMemAlloc(&dptr, 64);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, 64);

    ret = lwMemFree(dptr);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_free_events((void *)dptr, 64);

    /* large allocation */
    ret = lwMemAlloc(&dptr, (256 * 1024 *1024));
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, (256 * 1024 *1024));

    ret = lwMemFree(dptr);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_free_events((void *)dptr, (256 * 1024 *1024));

    /* multiple allocations, lwdafree in reverse order */
    ret = lwMemAlloc(&dptr, (1 * 1024 *1024));
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, (1 * 1024 *1024));

    ret = lwMemAlloc(&dptr1, (1 * 1024 *1024));
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_alloc_events((void *)dptr1, (1 * 1024 *1024));

    ret = lwMemFree(dptr1);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_free_events((void *)dptr1, (1 * 1024 *1024));

    ret = lwMemFree(dptr);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_free_events((void *)dptr, (1 * 1024 *1024));
}

UCS_TEST_F(lwda_hooks, test_lwMemAllocManaged) {
    LWresult ret;
    LWdeviceptr dptr;

    ret = lwMemAllocManaged(&dptr, 64, LW_MEM_ATTACH_GLOBAL);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, 64, UCS_MEMORY_TYPE_LWDA_MANAGED);

    ret = lwMemFree(dptr);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_free_events((void *)dptr, 0);
}

UCS_TEST_F(lwda_hooks, test_lwMemAllocPitch) {
    LWresult ret;
    LWdeviceptr dptr;
    size_t pitch;

    ret = lwMemAllocPitch(&dptr, &pitch, 4, 8, 4);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, (4 * 8));

    ret = lwMemFree(dptr);
    ASSERT_EQ(ret, LWDA_SUCCESS);
    check_mem_free_events((void *)dptr, 0);
}

UCS_TEST_F(lwda_hooks, test_lwda_Malloc_Free) {
    lwdaError_t ret;
    void *ptr, *ptr1;

    /* small allocation */
    ret = lwdaMalloc(&ptr, 64);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_alloc_events(ptr, 64);

    ret = lwdaFree(ptr);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_free_events(ptr, 64);

    /* large allocation */
    ret = lwdaMalloc(&ptr, (256 * 1024 *1024));
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_alloc_events(ptr, (256 * 1024 *1024));

    ret = lwdaFree(ptr);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_free_events(ptr, (256 * 1024 *1024));

    /* multiple allocations, lwdafree in reverse order */
    ret = lwdaMalloc(&ptr, (1 * 1024 *1024));
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_alloc_events(ptr, (1 * 1024 *1024));

    ret = lwdaMalloc(&ptr1, (1 * 1024 *1024));
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_alloc_events(ptr1, (1 * 1024 *1024));

    ret = lwdaFree(ptr1);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_free_events(ptr1, (1 * 1024 *1024));

    ret = lwdaFree(ptr);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_free_events(ptr, (1 * 1024 *1024));

    /* lwdaFree with NULL */
    ret = lwdaFree(NULL);
    ASSERT_EQ(ret, lwdaSuccess);
}

UCS_TEST_F(lwda_hooks, test_lwdaMallocManaged) {
    lwdaError_t ret;
    void *ptr;

    ret = lwdaMallocManaged(&ptr, 64, lwdaMemAttachGlobal);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_LWDA_MANAGED);

    ret = lwdaFree(ptr);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_free_events(ptr, 0);
}

UCS_TEST_F(lwda_hooks, test_lwdaMallocPitch) {
    lwdaError_t ret;
    void *devPtr;
    size_t pitch;

    ret = lwdaMallocPitch(&devPtr, &pitch, 4, 8);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_alloc_events(devPtr, (4 * 8));

    ret = lwdaFree(devPtr);
    ASSERT_EQ(ret, lwdaSuccess);
    check_mem_free_events(devPtr, 0);
}
