/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2011-2015 LWPU Corporation.  All rights reserved.
 * Copyright (c) 2015      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * This file contains various support functions for doing LWCA
 * operations.
 */
#include "opal_config.h"

#include <errno.h>
#include <unistd.h>
#include <lwca.h>

#include "opal/align.h"
#include "opal/datatype/opal_colwertor.h"
#include "opal/datatype/opal_datatype_lwda.h"
#include "opal/util/output.h"
#include "opal/util/show_help.h"
#include "opal/util/proc.h"
#include "opal/util/argv.h"

#include "opal/mca/rcache/base/base.h"
#include "opal/runtime/opal_params.h"
#include "opal/mca/timer/base/base.h"
#include "opal/mca/dl/base/base.h"

#include "common_lwda.h"

/**
 * Since function names can get redefined in lwca.h file, we need to do this
 * stringifying to get the latest function name from the header file.  For
 * example, lwca.h may have something like this:
 * #define lwMemFree lwMemFree_v2
 * We want to make sure we find lwMemFree_v2, not lwMemFree.
 */
#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

#define OPAL_LWDA_DLSYM(libhandle, funcName)                                         \
do {                                                                                 \
 char *err_msg;                                                                      \
 void *ptr;                                                                          \
 if (OPAL_SUCCESS !=                                                                 \
     opal_dl_lookup(libhandle, STRINGIFY(funcName), &ptr, &err_msg)) {               \
        opal_show_help("help-mpi-common-lwca.txt", "dlsym failed", true,             \
                       STRINGIFY(funcName), err_msg);                                \
        return 1;                                                                    \
    } else {                                                                         \
        *(void **)(&lwFunc.funcName) = ptr;                                          \
        opal_output_verbose(15, mca_common_lwda_output,                              \
                            "LWCA: successful dlsym of %s",                          \
                            STRINGIFY(funcName));                                    \
    }                                                                                \
} while (0)

/* Structure to hold LWCA function pointers that get dynamically loaded. */
struct lwdaFunctionTable {
    int (*lwPointerGetAttribute)(void *, LWpointer_attribute, LWdeviceptr);
    int (*lwMemcpyAsync)(LWdeviceptr, LWdeviceptr, size_t, LWstream);
    int (*lwMemcpy)(LWdeviceptr, LWdeviceptr, size_t);
    int (*lwMemAlloc)(LWdeviceptr *, unsigned int);
    int (*lwMemFree)(LWdeviceptr buf);
    int (*lwCtxGetLwrrent)(void *lwContext);
    int (*lwStreamCreate)(LWstream *, int);
    int (*lwEventCreate)(LWevent *, int);
    int (*lwEventRecord)(LWevent, LWstream);
    int (*lwMemHostRegister)(void *, size_t, unsigned int);
    int (*lwMemHostUnregister)(void *);
    int (*lwEventQuery)(LWevent);
    int (*lwEventDestroy)(LWevent);
    int (*lwStreamWaitEvent)(LWstream, LWevent, unsigned int);
    int (*lwMemGetAddressRange)(LWdeviceptr*, size_t*, LWdeviceptr);
    int (*lwIpcGetEventHandle)(LWipcEventHandle*, LWevent);
    int (*lwIpcOpenEventHandle)(LWevent*, LWipcEventHandle);
    int (*lwIpcOpenMemHandle)(LWdeviceptr*, LWipcMemHandle, unsigned int);
    int (*lwIpcCloseMemHandle)(LWdeviceptr);
    int (*lwIpcGetMemHandle)(LWipcMemHandle*, LWdeviceptr);
    int (*lwCtxGetDevice)(LWdevice *);
    int (*lwDeviceCanAccessPeer)(int *, LWdevice, LWdevice);
    int (*lwDeviceGet)(LWdevice *, int);
#if OPAL_LWDA_GDR_SUPPORT
    int (*lwPointerSetAttribute)(const void *, LWpointer_attribute, LWdeviceptr);
#endif /* OPAL_LWDA_GDR_SUPPORT */
    int (*lwCtxSetLwrrent)(LWcontext);
    int (*lwEventSynchronize)(LWevent);
    int (*lwStreamSynchronize)(LWstream);
    int (*lwStreamDestroy)(LWstream);
#if OPAL_LWDA_GET_ATTRIBUTES
    int (*lwPointerGetAttributes)(unsigned int, LWpointer_attribute *, void **, LWdeviceptr);
#endif /* OPAL_LWDA_GET_ATTRIBUTES */
};
typedef struct lwdaFunctionTable lwdaFunctionTable_t;
static lwdaFunctionTable_t lwFunc;

static int stage_one_init_ref_count = 0;
static bool stage_three_init_complete = false;
static bool common_lwda_initialized = false;
static bool common_lwda_mca_parames_registered = false;
static int mca_common_lwda_verbose;
static int mca_common_lwda_output = 0;
bool mca_common_lwda_enabled = false;
static bool mca_common_lwda_register_memory = true;
static bool mca_common_lwda_warning = false;
static opal_list_t common_lwda_memory_registrations;
static LWstream ipcStream = NULL;
static LWstream dtohStream = NULL;
static LWstream htodStream = NULL;
static LWstream memcpyStream = NULL;
static int mca_common_lwda_gpu_mem_check_workaround = (LWDA_VERSION > 7000) ? 0 : 1;
static opal_mutex_t common_lwda_init_lock;
static opal_mutex_t common_lwda_htod_lock;
static opal_mutex_t common_lwda_dtoh_lock;
static opal_mutex_t common_lwda_ipc_lock;

/* Functions called by opal layer - plugged into opal function table */
static int mca_common_lwda_is_gpu_buffer(const void*, opal_colwertor_t*);
static int mca_common_lwda_memmove(void*, void*, size_t);
static int mca_common_lwda_lw_memcpy_async(void*, const void*, size_t, opal_colwertor_t*);
static int mca_common_lwda_lw_memcpy(void*, const void*, size_t);

/* Function that gets plugged into opal layer */
static int mca_common_lwda_stage_two_init(opal_common_lwda_function_table_t *);

/* Structure to hold memory registrations that are delayed until first
 * call to send or receive a GPU pointer */
struct common_lwda_mem_regs_t {
    opal_list_item_t super;
    void *ptr;
    size_t amount;
    char *msg;
};
typedef struct common_lwda_mem_regs_t common_lwda_mem_regs_t;
OBJ_CLASS_DECLARATION(common_lwda_mem_regs_t);
OBJ_CLASS_INSTANCE(common_lwda_mem_regs_t,
                   opal_list_item_t,
                   NULL,
                   NULL);

static int mca_common_lwda_async = 1;
static int mca_common_lwda_lwmemcpy_async;
#if OPAL_ENABLE_DEBUG
static int mca_common_lwda_lwmemcpy_timing;
#endif /* OPAL_ENABLE_DEBUG */

/* Array of LWCA events to be queried for IPC stream, sending side and
 * receiving side. */
LWevent *lwda_event_ipc_array = NULL;
LWevent *lwda_event_dtoh_array = NULL;
LWevent *lwda_event_htod_array = NULL;

/* Array of fragments lwrrently being moved by lwca async non-blocking
 * operations */
struct mca_btl_base_descriptor_t **lwda_event_ipc_frag_array = NULL;
struct mca_btl_base_descriptor_t **lwda_event_dtoh_frag_array = NULL;
struct mca_btl_base_descriptor_t **lwda_event_htod_frag_array = NULL;

/* First free/available location in lwda_event_status_array */
static int lwda_event_ipc_first_avail, lwda_event_dtoh_first_avail, lwda_event_htod_first_avail;

/* First lwrrently-being used location in the lwda_event_status_array */
static int lwda_event_ipc_first_used, lwda_event_dtoh_first_used, lwda_event_htod_first_used;

/* Number of status items lwrrently in use */
static int lwda_event_ipc_num_used, lwda_event_dtoh_num_used, lwda_event_htod_num_used;

/* Size of array holding events */
int lwda_event_max = 400;
static int lwda_event_ipc_most = 0;
static int lwda_event_dtoh_most = 0;
static int lwda_event_htod_most = 0;

/* Handle to liblwda.so */
opal_dl_handle_t *liblwda_handle = NULL;

/* Unused variable that we register at init time and unregister at fini time.
 * This is used to detect if user has done a device reset prior to MPI_Finalize.
 * This is a workaround to avoid SEGVs.
 */
static int checkmem;
static int ctx_ok = 1;

#define LWDA_COMMON_TIMING 0
#if OPAL_ENABLE_DEBUG
/* Some timing support structures.  Enable this to help analyze
 * internal performance issues. */
static opal_timer_t ts_start;
static opal_timer_t ts_end;
static double aclwm;
#define THOUSAND  1000L
#define MILLION   1000000L
static float mydifftime(opal_timer_t ts_start, opal_timer_t ts_end);
#endif /* OPAL_ENABLE_DEBUG */

/* These functions are typically unused in the optimized builds. */
static void lwda_dump_evthandle(int, void *, char *) __opal_attribute_unused__ ;
static void lwda_dump_memhandle(int, void *, char *) __opal_attribute_unused__ ;
#if OPAL_ENABLE_DEBUG
#define LWDA_DUMP_MEMHANDLE(a) lwda_dump_memhandle a
#define LWDA_DUMP_EVTHANDLE(a) lwda_dump_evthandle a
#else
#define LWDA_DUMP_MEMHANDLE(a)
#define LWDA_DUMP_EVTHANDLE(a)
#endif /* OPAL_ENABLE_DEBUG */

/* This is a seperate function so we can see these variables with ompi_info and
 * also set them with the tools interface */
void mca_common_lwda_register_mca_variables(void)
{

    if (false == common_lwda_mca_parames_registered) {
        common_lwda_mca_parames_registered = true;
    }
    /* Set different levels of verbosity in the lwca related code. */
    mca_common_lwda_verbose = 0;
    (void) mca_base_var_register("ompi", "mpi", "common_lwda", "verbose",
                                 "Set level of common lwca verbosity",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_9,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &mca_common_lwda_verbose);

    /* Control whether system buffers get LWCA pinned or not.  Allows for
     * performance analysis. */
    mca_common_lwda_register_memory = true;
    (void) mca_base_var_register("ompi", "mpi", "common_lwda", "register_memory",
                                 "Whether to lwMemHostRegister preallocated BTL buffers",
                                 MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                 OPAL_INFO_LVL_9,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &mca_common_lwda_register_memory);

    /* Control whether we see warnings when LWCA memory registration fails.  This is
     * useful when LWCA support is configured in, but we are running a regular MPI
     * application without LWCA. */
    mca_common_lwda_warning = true;
    (void) mca_base_var_register("ompi", "mpi", "common_lwda", "warning",
                                 "Whether to print warnings when LWCA registration fails",
                                 MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                 OPAL_INFO_LVL_9,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &mca_common_lwda_warning);

    /* Use this flag to test async vs sync copies */
    mca_common_lwda_async = 1;
    (void) mca_base_var_register("ompi", "mpi", "common_lwda", "memcpy_async",
                                 "Set to 0 to force LWCA sync copy instead of async",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_9,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &mca_common_lwda_async);

    /* Use this parameter to increase the number of outstanding events allows */
    (void) mca_base_var_register("ompi", "mpi", "common_lwda", "event_max",
                                 "Set number of oustanding LWCA events",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_9,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &lwda_event_max);

    /* Use this flag to test lwMemcpyAsync vs lwMemcpy */
    mca_common_lwda_lwmemcpy_async = 1;
    (void) mca_base_var_register("ompi", "mpi", "common_lwda", "lwmemcpy_async",
                                 "Set to 0 to force LWCA lwMemcpy instead of lwMemcpyAsync/lwStreamSynchronize",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_5,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &mca_common_lwda_lwmemcpy_async);

#if OPAL_ENABLE_DEBUG
    /* Use this flag to dump out timing of lwmempcy sync and async */
    mca_common_lwda_lwmemcpy_timing = 0;
    (void) mca_base_var_register("ompi", "mpi", "common_lwda", "lwmemcpy_timing",
                                 "Set to 1 to dump timing of eager copies",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_5,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &mca_common_lwda_lwmemcpy_timing);
#endif /* OPAL_ENABLE_DEBUG */

    (void) mca_base_var_register("ompi", "mpi", "common_lwda", "gpu_mem_check_workaround",
                                 "Set to 0 to disable GPU memory check workaround. A user would rarely have to do this.",
                                 MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                 OPAL_INFO_LVL_9,
                                 MCA_BASE_VAR_SCOPE_READONLY,
                                 &mca_common_lwda_gpu_mem_check_workaround);
}

/**
 * This is the first stage of initialization.  This function is called
 * explicitly by any BTLs that can support LWCA-aware. It is called during
 * the component open phase of initialization. This fuction will look for
 * the SONAME of the library which is liblwda.so.1. In most cases, this will
 * result in the library found.  However, there are some setups that require
 * the extra steps for searching. This function will then load the symbols
 * needed from the LWCA driver library. Any failure will result in this
 * initialization failing and status will be set showing that.
 */
int mca_common_lwda_stage_one_init(void)
{
    int retval, i, j;
    char *lwdalibs[] = {"liblwda.so.1", "liblwda.dylib", NULL};
    char *searchpaths[] = {"", "/usr/lib64", NULL};
    char **errmsgs = NULL;
    char *errmsg = NULL;
    int errsize;
    bool stage_one_init_passed = false;

    stage_one_init_ref_count++;
    if (stage_one_init_ref_count > 1) {
        opal_output_verbose(10, mca_common_lwda_output,
                            "LWCA: stage_one_init_ref_count is now %d, no need to init",
                            stage_one_init_ref_count);
        return OPAL_SUCCESS;
    }

    /* This is a no-op in most cases as the parameters were registered earlier */
    mca_common_lwda_register_mca_variables();

    OBJ_CONSTRUCT(&common_lwda_init_lock, opal_mutex_t);
    OBJ_CONSTRUCT(&common_lwda_htod_lock, opal_mutex_t);
    OBJ_CONSTRUCT(&common_lwda_dtoh_lock, opal_mutex_t);
    OBJ_CONSTRUCT(&common_lwda_ipc_lock, opal_mutex_t);

    mca_common_lwda_output = opal_output_open(NULL);
    opal_output_set_verbosity(mca_common_lwda_output, mca_common_lwda_verbose);

    opal_output_verbose(10, mca_common_lwda_output,
                        "LWCA: stage_one_init_ref_count is now %d, initializing",
                        stage_one_init_ref_count);

    /* First check if the support is enabled.  In the case that the user has
     * turned it off, we do not need to continue with any LWCA specific
     * initialization.  Do this after MCA parameter registration. */
    if (!opal_lwda_support) {
        return 1;
    }

    if (!OPAL_HAVE_DL_SUPPORT) {
        opal_show_help("help-mpi-common-lwca.txt", "dlopen disabled", true);
        return 1;
    }

    /* Now walk through all the potential names liblwda and find one
     * that works.  If it does, all is good.  If not, print out all
     * the messages about why things failed.  This code was careful
     * to try and save away all error messages if the loading ultimately
     * failed to help with debugging.
     *
     * NOTE: On the first loop we just utilize the default loading
     * paths from the system.  For the second loop, set /usr/lib64 to
     * the search path and try again.  This is done to handle the case
     * where we have both 32 and 64 bit liblwda.so libraries
     * installed.  Even when running in 64-bit mode, the /usr/lib
     * directory is searched first and we may find a 32-bit
     * liblwda.so.1 library.  Loading of this library will fail as the
     * OPAL DL framework does not handle having the wrong ABI in the
     * search path (unlike ld or ld.so).  Note that we only set this
     * search path after the original search.  This is so that
     * LD_LIBRARY_PATH and run path settings are respected.  Setting
     * this search path overrides them (rather then being
     * appended). */
    j = 0;
    while (searchpaths[j] != NULL) {
        i = 0;
        while (lwdalibs[i] != NULL) {
            char *filename = NULL;
            char *str = NULL;

            /* If there's a non-empty search path, prepend it
               to the library filename */
            if (strlen(searchpaths[j]) > 0) {
                asprintf(&filename, "%s/%s", searchpaths[j], lwdalibs[i]);
            } else {
                filename = strdup(lwdalibs[i]);
            }
            if (NULL == filename) {
                opal_show_help("help-mpi-common-lwca.txt", "No memory",
                               true, OPAL_PROC_MY_HOSTNAME);
                return 1;
            }

            retval = opal_dl_open(filename, false, false,
                                  &liblwda_handle, &str);
            if (OPAL_SUCCESS != retval || NULL == liblwda_handle) {
                if (NULL != str) {
                    opal_argv_append(&errsize, &errmsgs, str);
                } else {
                    opal_argv_append(&errsize, &errmsgs,
                                     "opal_dl_open() returned NULL.");
                }
                opal_output_verbose(10, mca_common_lwda_output,
                                    "LWCA: Library open error: %s",
                                    errmsgs[errsize-1]);
            } else {
                opal_output_verbose(10, mca_common_lwda_output,
                                    "LWCA: Library successfully opened %s",
                                    lwdalibs[i]);
                stage_one_init_passed = true;
                break;
            }
            i++;

            free(filename);
        }
        if (true == stage_one_init_passed) {
            break; /* Break out of outer loop */
        }
        j++;
    }

    if (true != stage_one_init_passed) {
        errmsg = opal_argv_join(errmsgs, '\n');
        if (opal_warn_on_missing_liblwda) {
            opal_show_help("help-mpi-common-lwca.txt", "dlopen failed", true,
                           errmsg);
        }
        opal_lwda_support = 0;
    }
    opal_argv_free(errmsgs);
    free(errmsg);

    if (true != stage_one_init_passed) {
        return 1;
    }
    opal_lwda_add_initialization_function(&mca_common_lwda_stage_two_init);
    OBJ_CONSTRUCT(&common_lwda_memory_registrations, opal_list_t);

    /* Map in the functions that we need.  Note that if there is an error
     * the macro OPAL_LWDA_DLSYM will print an error and call return.  */
    OPAL_LWDA_DLSYM(liblwda_handle, lwStreamCreate);
    OPAL_LWDA_DLSYM(liblwda_handle, lwCtxGetLwrrent);
    OPAL_LWDA_DLSYM(liblwda_handle, lwEventCreate);
    OPAL_LWDA_DLSYM(liblwda_handle, lwEventRecord);
    OPAL_LWDA_DLSYM(liblwda_handle, lwMemHostRegister);
    OPAL_LWDA_DLSYM(liblwda_handle, lwMemHostUnregister);
    OPAL_LWDA_DLSYM(liblwda_handle, lwPointerGetAttribute);
    OPAL_LWDA_DLSYM(liblwda_handle, lwEventQuery);
    OPAL_LWDA_DLSYM(liblwda_handle, lwEventDestroy);
    OPAL_LWDA_DLSYM(liblwda_handle, lwStreamWaitEvent);
    OPAL_LWDA_DLSYM(liblwda_handle, lwMemcpyAsync);
    OPAL_LWDA_DLSYM(liblwda_handle, lwMemcpy);
    OPAL_LWDA_DLSYM(liblwda_handle, lwMemFree);
    OPAL_LWDA_DLSYM(liblwda_handle, lwMemAlloc);
    OPAL_LWDA_DLSYM(liblwda_handle, lwMemGetAddressRange);
    OPAL_LWDA_DLSYM(liblwda_handle, lwIpcGetEventHandle);
    OPAL_LWDA_DLSYM(liblwda_handle, lwIpcOpenEventHandle);
    OPAL_LWDA_DLSYM(liblwda_handle, lwIpcOpenMemHandle);
    OPAL_LWDA_DLSYM(liblwda_handle, lwIpcCloseMemHandle);
    OPAL_LWDA_DLSYM(liblwda_handle, lwIpcGetMemHandle);
    OPAL_LWDA_DLSYM(liblwda_handle, lwCtxGetDevice);
    OPAL_LWDA_DLSYM(liblwda_handle, lwDeviceCanAccessPeer);
    OPAL_LWDA_DLSYM(liblwda_handle, lwDeviceGet);
#if OPAL_LWDA_GDR_SUPPORT
    OPAL_LWDA_DLSYM(liblwda_handle, lwPointerSetAttribute);
#endif /* OPAL_LWDA_GDR_SUPPORT */
    OPAL_LWDA_DLSYM(liblwda_handle, lwCtxSetLwrrent);
    OPAL_LWDA_DLSYM(liblwda_handle, lwEventSynchronize);
    OPAL_LWDA_DLSYM(liblwda_handle, lwStreamSynchronize);
    OPAL_LWDA_DLSYM(liblwda_handle, lwStreamDestroy);
#if OPAL_LWDA_GET_ATTRIBUTES
    OPAL_LWDA_DLSYM(liblwda_handle, lwPointerGetAttributes);
#endif /* OPAL_LWDA_GET_ATTRIBUTES */
    return 0;
}

/**
 * This function is registered with the OPAL LWCA support.  In that way,
 * these function pointers will be loaded into the OPAL LWCA code when
 * the first colwertor is initialized.  This does not trigger any LWCA
 * specific initialization as this may just be a host buffer that is
 * triggering this call.
 */
static int mca_common_lwda_stage_two_init(opal_common_lwda_function_table_t *ftable)
{
    if (OPAL_UNLIKELY(!opal_lwda_support)) {
        return OPAL_ERROR;
    }

    ftable->gpu_is_gpu_buffer = &mca_common_lwda_is_gpu_buffer;
    ftable->gpu_lw_memcpy_async = &mca_common_lwda_lw_memcpy_async;
    ftable->gpu_lw_memcpy = &mca_common_lwda_lw_memcpy;
    ftable->gpu_memmove = &mca_common_lwda_memmove;

    opal_output_verbose(30, mca_common_lwda_output,
                        "LWCA: support functions initialized");
    return OPAL_SUCCESS;
}

/**
 * This is the last phase of initialization.  This is triggered when we examine
 * a buffer pointer and determine it is a GPU buffer.  We then assume the user
 * has selected their GPU and we can go ahead with all the LWCA related
 * initializations.  If we get an error, just return.  Cleanup of resources
 * will happen when fini is called.
 */
static int mca_common_lwda_stage_three_init(void)
{
    int i, s, rc;
    LWresult res;
    LWcontext lwContext;
    common_lwda_mem_regs_t *mem_reg;

    OPAL_THREAD_LOCK(&common_lwda_init_lock);
    opal_output_verbose(20, mca_common_lwda_output,
                        "LWCA: entering stage three init");

/* Compiled without support or user disabled support */
    if (OPAL_UNLIKELY(!opal_lwda_support)) {
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: No mpi lwca support, exiting stage three init");
        stage_three_init_complete = true;
        OPAL_THREAD_UNLOCK(&common_lwda_init_lock);
        return OPAL_ERROR;
    }

    /* In case another thread snuck in and completed the initialization */
    if (true == stage_three_init_complete) {
        if (common_lwda_initialized) {
            opal_output_verbose(20, mca_common_lwda_output,
                                "LWCA: Stage three already complete, exiting stage three init");
            OPAL_THREAD_UNLOCK(&common_lwda_init_lock);
            return OPAL_SUCCESS;
        } else {
            opal_output_verbose(20, mca_common_lwda_output,
                                "LWCA: Stage three already complete, failed during the init");
            OPAL_THREAD_UNLOCK(&common_lwda_init_lock);
            return OPAL_ERROR;
        }
    }

    /* Check to see if this process is running in a LWCA context.  If
     * so, all is good.  If not, then disable registration of memory. */
    res = lwFunc.lwCtxGetLwrrent(&lwContext);
    if (LWDA_SUCCESS != res) {
        if (mca_common_lwda_warning) {
            /* Check for the not initialized error since we can make suggestions to
             * user for this error. */
            if (LWDA_ERROR_NOT_INITIALIZED == res) {
                opal_show_help("help-mpi-common-lwca.txt", "lwCtxGetLwrrent failed not initialized",
                               true);
            } else {
                opal_show_help("help-mpi-common-lwca.txt", "lwCtxGetLwrrent failed",
                               true, res);
            }
        }
        mca_common_lwda_enabled = false;
        mca_common_lwda_register_memory = false;
    } else if ((LWDA_SUCCESS == res) && (NULL == lwContext)) {
        if (mca_common_lwda_warning) {
            opal_show_help("help-mpi-common-lwca.txt", "lwCtxGetLwrrent returned NULL",
                           true);
        }
        mca_common_lwda_enabled = false;
        mca_common_lwda_register_memory = false;
    } else {
        /* All is good.  mca_common_lwda_register_memory will retain its original
         * value.  Normally, that is 1, but the user can override it to disable
         * registration of the internal buffers. */
        mca_common_lwda_enabled = true;
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: lwCtxGetLwrrent succeeded");
    }

    /* No need to go on at this point.  If we cannot create a context and we are at
     * the point where we are making MPI calls, it is time to fully disable
     * LWCA support.
     */
    if (false == mca_common_lwda_enabled) {
        OPAL_THREAD_UNLOCK(&common_lwda_init_lock);
        return OPAL_ERROR;
    }

    if (true == mca_common_lwda_enabled) {
        /* Set up an array to store outstanding IPC async copy events */
        lwda_event_ipc_num_used = 0;
        lwda_event_ipc_first_avail = 0;
        lwda_event_ipc_first_used = 0;

        lwda_event_ipc_array = (LWevent *) calloc(lwda_event_max, sizeof(LWevent *));
        if (NULL == lwda_event_ipc_array) {
            opal_show_help("help-mpi-common-lwca.txt", "No memory",
                           true, OPAL_PROC_MY_HOSTNAME);
            rc = OPAL_ERROR;
            goto cleanup_and_error;
        }

        /* Create the events since they can be reused. */
        for (i = 0; i < lwda_event_max; i++) {
            res = lwFunc.lwEventCreate(&lwda_event_ipc_array[i], LW_EVENT_DISABLE_TIMING);
            if (LWDA_SUCCESS != res) {
                opal_show_help("help-mpi-common-lwca.txt", "lwEventCreate failed",
                               true, OPAL_PROC_MY_HOSTNAME, res);
                rc = OPAL_ERROR;
                goto cleanup_and_error;
            }
        }

        /* The first available status index is 0.  Make an empty frag
           array. */
        lwda_event_ipc_frag_array = (struct mca_btl_base_descriptor_t **)
            malloc(sizeof(struct mca_btl_base_descriptor_t *) * lwda_event_max);
        if (NULL == lwda_event_ipc_frag_array) {
            opal_show_help("help-mpi-common-lwca.txt", "No memory",
                           true, OPAL_PROC_MY_HOSTNAME);
            rc = OPAL_ERROR;
            goto cleanup_and_error;
        }
    }

    if (true == mca_common_lwda_enabled) {
        /* Set up an array to store outstanding async dtoh events.  Used on the
         * sending side for asynchronous copies. */
        lwda_event_dtoh_num_used = 0;
        lwda_event_dtoh_first_avail = 0;
        lwda_event_dtoh_first_used = 0;

        lwda_event_dtoh_array = (LWevent *) calloc(lwda_event_max, sizeof(LWevent *));
        if (NULL == lwda_event_dtoh_array) {
            opal_show_help("help-mpi-common-lwca.txt", "No memory",
                           true, OPAL_PROC_MY_HOSTNAME);
            rc = OPAL_ERROR;
            goto cleanup_and_error;
        }

        /* Create the events since they can be reused. */
        for (i = 0; i < lwda_event_max; i++) {
            res = lwFunc.lwEventCreate(&lwda_event_dtoh_array[i], LW_EVENT_DISABLE_TIMING);
            if (LWDA_SUCCESS != res) {
                opal_show_help("help-mpi-common-lwca.txt", "lwEventCreate failed",
                               true, OPAL_PROC_MY_HOSTNAME, res);
                rc = OPAL_ERROR;
                goto cleanup_and_error;
            }
        }

        /* The first available status index is 0.  Make an empty frag
           array. */
        lwda_event_dtoh_frag_array = (struct mca_btl_base_descriptor_t **)
            malloc(sizeof(struct mca_btl_base_descriptor_t *) * lwda_event_max);
        if (NULL == lwda_event_dtoh_frag_array) {
            opal_show_help("help-mpi-common-lwca.txt", "No memory",
                           true, OPAL_PROC_MY_HOSTNAME);
            rc = OPAL_ERROR;
            goto cleanup_and_error;
        }

        /* Set up an array to store outstanding async htod events.  Used on the
         * receiving side for asynchronous copies. */
        lwda_event_htod_num_used = 0;
        lwda_event_htod_first_avail = 0;
        lwda_event_htod_first_used = 0;

        lwda_event_htod_array = (LWevent *) calloc(lwda_event_max, sizeof(LWevent *));
        if (NULL == lwda_event_htod_array) {
            opal_show_help("help-mpi-common-lwca.txt", "No memory",
                           true, OPAL_PROC_MY_HOSTNAME);
           rc = OPAL_ERROR;
           goto cleanup_and_error;
        }

        /* Create the events since they can be reused. */
        for (i = 0; i < lwda_event_max; i++) {
            res = lwFunc.lwEventCreate(&lwda_event_htod_array[i], LW_EVENT_DISABLE_TIMING);
            if (LWDA_SUCCESS != res) {
                opal_show_help("help-mpi-common-lwca.txt", "lwEventCreate failed",
                               true, OPAL_PROC_MY_HOSTNAME, res);
               rc = OPAL_ERROR;
               goto cleanup_and_error;
            }
        }

        /* The first available status index is 0.  Make an empty frag
           array. */
        lwda_event_htod_frag_array = (struct mca_btl_base_descriptor_t **)
            malloc(sizeof(struct mca_btl_base_descriptor_t *) * lwda_event_max);
        if (NULL == lwda_event_htod_frag_array) {
            opal_show_help("help-mpi-common-lwca.txt", "No memory",
                           true, OPAL_PROC_MY_HOSTNAME);
           rc = OPAL_ERROR;
           goto cleanup_and_error;
        }
    }

    s = opal_list_get_size(&common_lwda_memory_registrations);
    for(i = 0; i < s; i++) {
        mem_reg = (common_lwda_mem_regs_t *)
            opal_list_remove_first(&common_lwda_memory_registrations);
        if (mca_common_lwda_enabled && mca_common_lwda_register_memory) {
            res = lwFunc.lwMemHostRegister(mem_reg->ptr, mem_reg->amount, 0);
            if (res != LWDA_SUCCESS) {
                /* If registering the memory fails, print a message and continue.
                 * This is not a fatal error. */
                opal_show_help("help-mpi-common-lwca.txt", "lwMemHostRegister during init failed",
                               true, mem_reg->ptr, mem_reg->amount,
                               OPAL_PROC_MY_HOSTNAME, res, mem_reg->msg);
            } else {
                opal_output_verbose(20, mca_common_lwda_output,
                                    "LWCA: lwMemHostRegister OK on rcache %s: "
                                    "address=%p, bufsize=%d",
                                    mem_reg->msg, mem_reg->ptr, (int)mem_reg->amount);
            }
        }
        free(mem_reg->msg);
        OBJ_RELEASE(mem_reg);
    }

    /* Create stream for use in ipc asynchronous copies */
    res = lwFunc.lwStreamCreate(&ipcStream, 0);
    if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwStreamCreate failed",
                       true, OPAL_PROC_MY_HOSTNAME, res);
        rc = OPAL_ERROR;
        goto cleanup_and_error;
    }

    /* Create stream for use in dtoh asynchronous copies */
    res = lwFunc.lwStreamCreate(&dtohStream, 0);
    if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwStreamCreate failed",
                       true, OPAL_PROC_MY_HOSTNAME, res);
        rc = OPAL_ERROR;
        goto cleanup_and_error;
    }

    /* Create stream for use in htod asynchronous copies */
    res = lwFunc.lwStreamCreate(&htodStream, 0);
    if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwStreamCreate failed",
                       true, OPAL_PROC_MY_HOSTNAME, res);
        rc = OPAL_ERROR;
        goto cleanup_and_error;
    }

    if (mca_common_lwda_lwmemcpy_async) {
        /* Create stream for use in lwMemcpyAsync synchronous copies */
        res = lwFunc.lwStreamCreate(&memcpyStream, 0);
        if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwStreamCreate failed",
                           true, OPAL_PROC_MY_HOSTNAME, res);
            rc = OPAL_ERROR;
            goto cleanup_and_error;
        }
    }

    res = lwFunc.lwMemHostRegister(&checkmem, sizeof(int), 0);
    if (res != LWDA_SUCCESS) {
        /* If registering the memory fails, print a message and continue.
         * This is not a fatal error. */
        opal_show_help("help-mpi-common-lwca.txt", "lwMemHostRegister during init failed",
                       true, &checkmem, sizeof(int),
                       OPAL_PROC_MY_HOSTNAME, res, "checkmem");

    } else {
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: lwMemHostRegister OK on test region");
    }

    opal_output_verbose(20, mca_common_lwda_output,
                        "LWCA: the extra gpu memory check is %s", (mca_common_lwda_gpu_mem_check_workaround == 1) ? "on":"off");

    opal_output_verbose(30, mca_common_lwda_output,
                        "LWCA: initialized");
    opal_atomic_mb();  /* Make sure next statement does not get reordered */
    common_lwda_initialized = true;
    stage_three_init_complete = true;
    OPAL_THREAD_UNLOCK(&common_lwda_init_lock);
    return OPAL_SUCCESS;

    /* If we are here, something went wrong.  Cleanup and return an error. */
 cleanup_and_error:
    opal_atomic_mb(); /* Make sure next statement does not get reordered */
    stage_three_init_complete = true;
    OPAL_THREAD_UNLOCK(&common_lwda_init_lock);
    return rc;
}

/**
 * Cleanup all LWCA resources.
 *
 * Note: Still figuring out how to get lwMemHostUnregister called from the smlwda sm
 * rcache.  Looks like with the memory pool from openib (grdma), the unregistering is
 * called as the free list is destructed.  Not true for the sm mpool.  This means we
 * are lwrrently still leaking some host memory we registered with LWCA.
 */
void mca_common_lwda_fini(void)
{
    int i;
    LWresult res;

    if (false == common_lwda_initialized) {
        stage_one_init_ref_count--;
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: mca_common_lwda_fini, never completed initialization so "
                            "skipping fini, ref_count is now %d", stage_one_init_ref_count);
        return;
    }

    if (0 == stage_one_init_ref_count) {
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: mca_common_lwda_fini, ref_count=%d, fini is already complete",
                            stage_one_init_ref_count);
        return;
    }

    if (1 == stage_one_init_ref_count) {
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: mca_common_lwda_fini, ref_count=%d, cleaning up started",
                            stage_one_init_ref_count);

        /* This call is in here to make sure the context is still valid.
         * This was the one way of checking which did not cause problems
         * while calling into the LWCA library.  This check will detect if
         * a user has called lwdaDeviceReset prior to MPI_Finalize. If so,
         * then this call will fail and we skip cleaning up LWCA resources. */
        res = lwFunc.lwMemHostUnregister(&checkmem);
        if (LWDA_SUCCESS != res) {
            ctx_ok = 0;
        }
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: mca_common_lwda_fini, lwMemHostUnregister returned %d, ctx_ok=%d",
                            res, ctx_ok);

        if (NULL != lwda_event_ipc_array) {
            if (ctx_ok) {
                for (i = 0; i < lwda_event_max; i++) {
                    if (NULL != lwda_event_ipc_array[i]) {
                        lwFunc.lwEventDestroy(lwda_event_ipc_array[i]);
                    }
                }
            }
            free(lwda_event_ipc_array);
        }
        if (NULL != lwda_event_htod_array) {
            if (ctx_ok) {
                for (i = 0; i < lwda_event_max; i++) {
                    if (NULL != lwda_event_htod_array[i]) {
                        lwFunc.lwEventDestroy(lwda_event_htod_array[i]);
                    }
                }
            }
            free(lwda_event_htod_array);
        }

        if (NULL != lwda_event_dtoh_array) {
            if (ctx_ok) {
                for (i = 0; i < lwda_event_max; i++) {
                    if (NULL != lwda_event_dtoh_array[i]) {
                        lwFunc.lwEventDestroy(lwda_event_dtoh_array[i]);
                    }
                }
            }
            free(lwda_event_dtoh_array);
        }

        if (NULL != lwda_event_ipc_frag_array) {
            free(lwda_event_ipc_frag_array);
        }
        if (NULL != lwda_event_htod_frag_array) {
            free(lwda_event_htod_frag_array);
        }
        if (NULL != lwda_event_dtoh_frag_array) {
            free(lwda_event_dtoh_frag_array);
        }
        if ((NULL != ipcStream) && ctx_ok) {
            lwFunc.lwStreamDestroy(ipcStream);
        }
        if ((NULL != dtohStream) && ctx_ok) {
            lwFunc.lwStreamDestroy(dtohStream);
        }
        if ((NULL != htodStream) && ctx_ok) {
            lwFunc.lwStreamDestroy(htodStream);
        }
        if ((NULL != memcpyStream) && ctx_ok) {
            lwFunc.lwStreamDestroy(memcpyStream);
        }
        OBJ_DESTRUCT(&common_lwda_init_lock);
        OBJ_DESTRUCT(&common_lwda_htod_lock);
        OBJ_DESTRUCT(&common_lwda_dtoh_lock);
        OBJ_DESTRUCT(&common_lwda_ipc_lock);
        if (NULL != liblwda_handle) {
            opal_dl_close(liblwda_handle);
        }

        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: mca_common_lwda_fini, ref_count=%d, cleaning up all done",
                            stage_one_init_ref_count);

        opal_output_close(mca_common_lwda_output);

    } else {
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: mca_common_lwda_fini, ref_count=%d, lwca still in use",
                            stage_one_init_ref_count);
    }
    stage_one_init_ref_count--;
}

/**
 * Call the LWCA register function so we pin the memory in the LWCA
 * space.
 */
void mca_common_lwda_register(void *ptr, size_t amount, char *msg) {
    int res;

    /* Always first check if the support is enabled.  If not, just return */
    if (!opal_lwda_support)
        return;

    if (!common_lwda_initialized) {
        OPAL_THREAD_LOCK(&common_lwda_init_lock);
        if (!common_lwda_initialized) {
            common_lwda_mem_regs_t *regptr;
            regptr = OBJ_NEW(common_lwda_mem_regs_t);
            regptr->ptr = ptr;
            regptr->amount = amount;
            regptr->msg = strdup(msg);
            opal_list_append(&common_lwda_memory_registrations,
                             (opal_list_item_t*)regptr);
            OPAL_THREAD_UNLOCK(&common_lwda_init_lock);
            return;
        }
        OPAL_THREAD_UNLOCK(&common_lwda_init_lock);
    }

    if (mca_common_lwda_enabled && mca_common_lwda_register_memory) {
        res = lwFunc.lwMemHostRegister(ptr, amount, 0);
        if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
            /* If registering the memory fails, print a message and continue.
             * This is not a fatal error. */
            opal_show_help("help-mpi-common-lwca.txt", "lwMemHostRegister failed",
                           true, ptr, amount,
                           OPAL_PROC_MY_HOSTNAME, res, msg);
        } else {
            opal_output_verbose(20, mca_common_lwda_output,
                                "LWCA: lwMemHostRegister OK on rcache %s: "
                                "address=%p, bufsize=%d",
                                msg, ptr, (int)amount);
        }
    }
}

/**
 * Call the LWCA unregister function so we unpin the memory in the LWCA
 * space.
 */
void mca_common_lwda_unregister(void *ptr, char *msg) {
    int res, i, s;
    common_lwda_mem_regs_t *mem_reg;

    /* This can happen if memory was queued up to be registered, but
     * no LWCA operations happened, so it never was registered.
     * Therefore, just release any of the resources. */
    if (!common_lwda_initialized) {
        s = opal_list_get_size(&common_lwda_memory_registrations);
        for(i = 0; i < s; i++) {
            mem_reg = (common_lwda_mem_regs_t *)
                opal_list_remove_first(&common_lwda_memory_registrations);
            free(mem_reg->msg);
            OBJ_RELEASE(mem_reg);
        }
        return;
    }

    if (mca_common_lwda_enabled && mca_common_lwda_register_memory) {
        res = lwFunc.lwMemHostUnregister(ptr);
        if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
            /* If unregistering the memory fails, just continue.  This is during
             * shutdown.  Only print when running in verbose mode. */
            opal_output_verbose(20, mca_common_lwda_output,
                                "LWCA: lwMemHostUnregister failed: ptr=%p, res=%d, rcache=%s",
                                ptr, res, msg);

        } else {
            opal_output_verbose(20, mca_common_lwda_output,
                                "LWCA: lwMemHostUnregister OK on rcache %s: "
                                "address=%p",
                                msg, ptr);
        }
    }
}

/*
 * Get the memory handle of a local section of memory that can be sent
 * to the remote size so it can access the memory.  This is the
 * registration function for the sending side of a message transfer.
 */
int lwda_getmemhandle(void *base, size_t size, mca_rcache_base_registration_t *newreg,
                      mca_rcache_base_registration_t *hdrreg)

{
    LWmemorytype memType;
    LWresult result;
    LWipcMemHandle *memHandle;
    LWdeviceptr pbase;
    size_t psize;

    mca_rcache_common_lwda_reg_t *lwda_reg = (mca_rcache_common_lwda_reg_t*)newreg;
    memHandle = (LWipcMemHandle *)lwda_reg->data.memHandle;

    /* We should only be there if this is a LWCA device pointer */
    result = lwFunc.lwPointerGetAttribute(&memType,
                                          LW_POINTER_ATTRIBUTE_MEMORY_TYPE, (LWdeviceptr)base);
    assert(LWDA_SUCCESS == result);
    assert(LW_MEMORYTYPE_DEVICE == memType);

    /* Get the memory handle so we can send it to the remote process. */
    result = lwFunc.lwIpcGetMemHandle(memHandle, (LWdeviceptr)base);
    LWDA_DUMP_MEMHANDLE((100, memHandle, "GetMemHandle-After"));

    if (LWDA_SUCCESS != result) {
        opal_show_help("help-mpi-common-lwca.txt", "lwIpcGetMemHandle failed",
                       true, result, base);
        return OPAL_ERROR;
    } else {
        opal_output_verbose(20, mca_common_lwda_output,
                            "LWCA: lwIpcGetMemHandle passed: base=%p size=%d",
                            base, (int)size);
    }

    /* Need to get the real base and size of the memory handle.  This is
     * how the remote side saves the handles in a cache. */
    result = lwFunc.lwMemGetAddressRange(&pbase, &psize, (LWdeviceptr)base);
    if (LWDA_SUCCESS != result) {
        opal_show_help("help-mpi-common-lwca.txt", "lwMemGetAddressRange failed",
                       true, result, base);
        return OPAL_ERROR;
    } else {
        opal_output_verbose(10, mca_common_lwda_output,
                            "LWCA: lwMemGetAddressRange passed: addr=%p, size=%d, pbase=%p, psize=%d ",
                            base, (int)size, (void *)pbase, (int)psize);
    }

    /* Store all the information in the registration */
    lwda_reg->base.base = (void *)pbase;
    lwda_reg->base.bound = (unsigned char *)pbase + psize - 1;
    lwda_reg->data.memh_seg_addr.pval = (void *) pbase;
    lwda_reg->data.memh_seg_len = psize;

#if OPAL_LWDA_SYNC_MEMOPS
    /* With LWCA 6.0, we can set an attribute on the memory pointer that will
     * ensure any synchronous copies are completed prior to any other access
     * of the memory region.  This means we do not need to record an event
     * and send to the remote side.
     */
    memType = 1; /* Just use this variable since we already have it */
    result = lwFunc.lwPointerSetAttribute(&memType, LW_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                          (LWdeviceptr)base);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwPointerSetAttribute failed",
                       true, OPAL_PROC_MY_HOSTNAME, result, base);
        return OPAL_ERROR;
    }
#else
    /* Need to record the event to ensure that any memcopies into the
     * device memory have completed.  The event handle associated with
     * this event is sent to the remote process so that it will wait
     * on this event prior to copying data out of the device memory.
     * Note that this needs to be the NULL stream to make since it is
     * unknown what stream any copies into the device memory were done
     * with. */
    result = lwFunc.lwEventRecord((LWevent)lwda_reg->data.event, 0);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwEventRecord failed",
                       true, result, base);
        return OPAL_ERROR;
    }
#endif /* OPAL_LWDA_SYNC_MEMOPS */

    return OPAL_SUCCESS;
}

/*
 * This function is called by the local side that called the lwda_getmemhandle.
 * There is nothing to be done so just return.
 */
int lwda_ungetmemhandle(void *reg_data, mca_rcache_base_registration_t *reg)
{
    opal_output_verbose(10, mca_common_lwda_output,
                        "LWCA: lwda_ungetmemhandle (no-op): base=%p", reg->base);
    LWDA_DUMP_MEMHANDLE((100, ((mca_rcache_common_lwda_reg_t *)reg)->data.memHandle, "lwda_ungetmemhandle"));

    return OPAL_SUCCESS;
}

/*
 * Open a memory handle that refers to remote memory so we can get an address
 * that works on the local side.  This is the registration function for the
 * remote side of a transfer.  newreg contains the new handle.  hddrreg contains
 * the memory handle that was received from the remote side.
 */
int lwda_openmemhandle(void *base, size_t size, mca_rcache_base_registration_t *newreg,
                       mca_rcache_base_registration_t *hdrreg)
{
    LWresult result;
    LWipcMemHandle *memHandle;
    mca_rcache_common_lwda_reg_t *lwda_newreg = (mca_rcache_common_lwda_reg_t*)newreg;

    /* Save in local variable to avoid ugly casting */
    memHandle = (LWipcMemHandle *)lwda_newreg->data.memHandle;
    LWDA_DUMP_MEMHANDLE((100, memHandle, "Before call to lwIpcOpenMemHandle"));

    /* Open the memory handle and store it into the registration structure. */
    result = lwFunc.lwIpcOpenMemHandle((LWdeviceptr *)&newreg->alloc_base, *memHandle,
                                       LW_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);

    /* If there are some stale entries in the cache, they can cause other
     * registrations to fail.  Let the caller know that so that can attempt
     * to clear them out. */
    if (LWDA_ERROR_ALREADY_MAPPED == result) {
        opal_output_verbose(10, mca_common_lwda_output,
                            "LWCA: lwIpcOpenMemHandle returned LWDA_ERROR_ALREADY_MAPPED for "
                            "p=%p,size=%d: notify memory pool\n", base, (int)size);
        return OPAL_ERR_WOULD_BLOCK;
    }
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwIpcOpenMemHandle failed",
                       true, OPAL_PROC_MY_HOSTNAME, result, base);
        /* Lwrrently, this is a non-recoverable error */
        return OPAL_ERROR;
    } else {
        opal_output_verbose(10, mca_common_lwda_output,
                            "LWCA: lwIpcOpenMemHandle passed: base=%p (remote base=%p,size=%d)",
                            newreg->alloc_base, base, (int)size);
        LWDA_DUMP_MEMHANDLE((200, memHandle, "lwIpcOpenMemHandle"));
    }

    return OPAL_SUCCESS;
}

/*
 * Close a memory handle that refers to remote memory.
 */
int lwda_closememhandle(void *reg_data, mca_rcache_base_registration_t *reg)
{
    LWresult result;
    mca_rcache_common_lwda_reg_t *lwda_reg = (mca_rcache_common_lwda_reg_t*)reg;

    /* Only attempt to close if we have valid context.  This can change if a call
     * to the fini function is made and we discover context is gone. */
    if (ctx_ok) {
        result = lwFunc.lwIpcCloseMemHandle((LWdeviceptr)lwda_reg->base.alloc_base);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            if (LWDA_ERROR_DEINITIALIZED != result) {
                opal_show_help("help-mpi-common-lwca.txt", "lwIpcCloseMemHandle failed",
                true, result, lwda_reg->base.alloc_base);
            }
            /* We will just continue on and hope things continue to work. */
        } else {
            opal_output_verbose(10, mca_common_lwda_output,
                                "LWCA: lwIpcCloseMemHandle passed: base=%p",
                                lwda_reg->base.alloc_base);
            LWDA_DUMP_MEMHANDLE((100, lwda_reg->data.memHandle, "lwIpcCloseMemHandle"));
        }
    }

    return OPAL_SUCCESS;
}

void mca_common_lwda_construct_event_and_handle(uintptr_t *event, void *handle)
{
    LWresult result;

    result = lwFunc.lwEventCreate((LWevent *)event, LW_EVENT_INTERPROCESS | LW_EVENT_DISABLE_TIMING);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwEventCreate failed",
                       true, OPAL_PROC_MY_HOSTNAME, result);
    }

    result = lwFunc.lwIpcGetEventHandle((LWipcEventHandle *)handle, (LWevent)*event);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwIpcGetEventHandle failed",
                       true, result);
    }

    LWDA_DUMP_EVTHANDLE((10, handle, "construct_event_and_handle"));

}

void mca_common_lwda_destruct_event(uintptr_t event)
{
    LWresult result;

    /* Only attempt to destroy if we have valid context.  This can change if a call
     * to the fini function is made and we discover context is gone. */
    if (ctx_ok) {
        result = lwFunc.lwEventDestroy((LWevent)event);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwEventDestroy failed",
                           true, result);
        }
    }
}


/*
 * Put remote event on stream to ensure that the the start of the
 * copy does not start until the completion of the event.
 */
void mca_common_wait_stream_synchronize(mca_rcache_common_lwda_reg_t *rget_reg)
{
#if OPAL_LWDA_SYNC_MEMOPS
    /* No need for any of this with SYNC_MEMOPS feature */
    return;
#else /* OPAL_LWDA_SYNC_MEMOPS */
    LWipcEventHandle evtHandle;
    LWevent event;
    LWresult result;

    memcpy(&evtHandle, rget_reg->data.evtHandle, sizeof(evtHandle));
    LWDA_DUMP_EVTHANDLE((100, &evtHandle, "stream_synchronize"));

    result = lwFunc.lwIpcOpenEventHandle(&event, evtHandle);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwIpcOpenEventHandle failed",
                       true, result);
    }

    /* BEGIN of Workaround - There is a bug in LWCA 4.1 RC2 and earlier
     * versions.  Need to record an event on the stream, even though
     * it is not used, to make sure we do not short circuit our way
     * out of the lwStreamWaitEvent test.
     */
    result = lwFunc.lwEventRecord(event, 0);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwEventRecord failed",
                       true, OPAL_PROC_MY_HOSTNAME, result);
    }
    /* END of Workaround */

    result = lwFunc.lwStreamWaitEvent(0, event, 0);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwStreamWaitEvent failed",
                       true, result);
    }

    /* All done with this event. */
    result = lwFunc.lwEventDestroy(event);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwEventDestroy failed",
                       true, result);
    }
#endif /* OPAL_LWDA_SYNC_MEMOPS */
}

/*
 * Start the asynchronous copy.  Then record and save away an event that will
 * be queried to indicate the copy has completed.
 */
int mca_common_lwda_memcpy(void *dst, void *src, size_t amount, char *msg,
                           struct mca_btl_base_descriptor_t *frag, int *done)
{
    LWresult result;
    int iter;

    OPAL_THREAD_LOCK(&common_lwda_ipc_lock);
    /* First make sure there is room to store the event.  If not, then
     * return an error.  The error message will tell the user to try and
     * run again, but with a larger array for storing events. */
    if (lwda_event_ipc_num_used == lwda_event_max) {
        opal_show_help("help-mpi-common-lwca.txt", "Out of lwEvent handles",
                       true, lwda_event_max, lwda_event_max+100, lwda_event_max+100);
        OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    if (lwda_event_ipc_num_used > lwda_event_ipc_most) {
        lwda_event_ipc_most = lwda_event_ipc_num_used;
        /* Just print multiples of 10 */
        if (0 == (lwda_event_ipc_most % 10)) {
            opal_output_verbose(20, mca_common_lwda_output,
                                "Maximum ipc events used is now %d", lwda_event_ipc_most);
        }
    }

    /* This is the standard way to run.  Running with synchronous copies is available
     * to measure the advantages of asynchronous copies. */
    if (OPAL_LIKELY(mca_common_lwda_async)) {
        result = lwFunc.lwMemcpyAsync((LWdeviceptr)dst, (LWdeviceptr)src, amount, ipcStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwMemcpyAsync failed",
                           true, dst, src, amount, result);
            OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
            return OPAL_ERROR;
        } else {
            opal_output_verbose(20, mca_common_lwda_output,
                                "LWCA: lwMemcpyAsync passed: dst=%p, src=%p, size=%d",
                                dst, src, (int)amount);
        }
        result = lwFunc.lwEventRecord(lwda_event_ipc_array[lwda_event_ipc_first_avail], ipcStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwEventRecord failed",
                           true, OPAL_PROC_MY_HOSTNAME, result);
            OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
            return OPAL_ERROR;
        }
        lwda_event_ipc_frag_array[lwda_event_ipc_first_avail] = frag;

        /* Bump up the first available slot and number used by 1 */
        lwda_event_ipc_first_avail++;
        if (lwda_event_ipc_first_avail >= lwda_event_max) {
            lwda_event_ipc_first_avail = 0;
        }
        lwda_event_ipc_num_used++;

        *done = 0;
    } else {
        /* Mimic the async function so they use the same memcpy call. */
        result = lwFunc.lwMemcpyAsync((LWdeviceptr)dst, (LWdeviceptr)src, amount, ipcStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwMemcpyAsync failed",
                           true, dst, src, amount, result);
            OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
            return OPAL_ERROR;
        } else {
            opal_output_verbose(20, mca_common_lwda_output,
                                "LWCA: lwMemcpyAsync passed: dst=%p, src=%p, size=%d",
                                dst, src, (int)amount);
        }

        /* Record an event, then wait for it to complete with calls to lwEventQuery */
        result = lwFunc.lwEventRecord(lwda_event_ipc_array[lwda_event_ipc_first_avail], ipcStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwEventRecord failed",
                           true, OPAL_PROC_MY_HOSTNAME, result);
            OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
            return OPAL_ERROR;
        }

        lwda_event_ipc_frag_array[lwda_event_ipc_first_avail] = frag;

        /* Bump up the first available slot and number used by 1 */
        lwda_event_ipc_first_avail++;
        if (lwda_event_ipc_first_avail >= lwda_event_max) {
            lwda_event_ipc_first_avail = 0;
        }
        lwda_event_ipc_num_used++;

        result = lwFunc.lwEventQuery(lwda_event_ipc_array[lwda_event_ipc_first_used]);
        if ((LWDA_SUCCESS != result) && (LWDA_ERROR_NOT_READY != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwEventQuery failed",
                           true, result);
            OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
            return OPAL_ERROR;
        }

        iter = 0;
        while (LWDA_ERROR_NOT_READY == result) {
            if (0 == (iter % 10)) {
                opal_output(-1, "EVENT NOT DONE (iter=%d)", iter);
            }
            result = lwFunc.lwEventQuery(lwda_event_ipc_array[lwda_event_ipc_first_used]);
            if ((LWDA_SUCCESS != result) && (LWDA_ERROR_NOT_READY != result)) {
                opal_show_help("help-mpi-common-lwca.txt", "lwEventQuery failed",
                               true, result);
            OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
                return OPAL_ERROR;
            }
            iter++;
        }

        --lwda_event_ipc_num_used;
        ++lwda_event_ipc_first_used;
        if (lwda_event_ipc_first_used >= lwda_event_max) {
            lwda_event_ipc_first_used = 0;
        }
        *done = 1;
    }
    OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
    return OPAL_SUCCESS;
}

/*
 * Record an event and save the frag.  This is called by the sending side and
 * is used to queue an event when a htod copy has been initiated.
 */
int mca_common_lwda_record_dtoh_event(char *msg, struct mca_btl_base_descriptor_t *frag)
{
    LWresult result;

    /* First make sure there is room to store the event.  If not, then
     * return an error.  The error message will tell the user to try and
     * run again, but with a larger array for storing events. */
    OPAL_THREAD_LOCK(&common_lwda_dtoh_lock);
    if (lwda_event_dtoh_num_used == lwda_event_max) {
        opal_show_help("help-mpi-common-lwca.txt", "Out of lwEvent handles",
                       true, lwda_event_max, lwda_event_max+100, lwda_event_max+100);
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    if (lwda_event_dtoh_num_used > lwda_event_dtoh_most) {
        lwda_event_dtoh_most = lwda_event_dtoh_num_used;
        /* Just print multiples of 10 */
        if (0 == (lwda_event_dtoh_most % 10)) {
            opal_output_verbose(20, mca_common_lwda_output,
                                "Maximum DtoH events used is now %d", lwda_event_dtoh_most);
        }
    }

    result = lwFunc.lwEventRecord(lwda_event_dtoh_array[lwda_event_dtoh_first_avail], dtohStream);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwEventRecord failed",
                       true, OPAL_PROC_MY_HOSTNAME, result);
        OPAL_THREAD_UNLOCK(&common_lwda_dtoh_lock);
        return OPAL_ERROR;
    }
    lwda_event_dtoh_frag_array[lwda_event_dtoh_first_avail] = frag;

    /* Bump up the first available slot and number used by 1 */
    lwda_event_dtoh_first_avail++;
    if (lwda_event_dtoh_first_avail >= lwda_event_max) {
        lwda_event_dtoh_first_avail = 0;
    }
    lwda_event_dtoh_num_used++;

    OPAL_THREAD_UNLOCK(&common_lwda_dtoh_lock);
    return OPAL_SUCCESS;
}

/*
 * Record an event and save the frag.  This is called by the receiving side and
 * is used to queue an event when a dtoh copy has been initiated.
 */
int mca_common_lwda_record_htod_event(char *msg, struct mca_btl_base_descriptor_t *frag)
{
    LWresult result;

    OPAL_THREAD_LOCK(&common_lwda_htod_lock);
    /* First make sure there is room to store the event.  If not, then
     * return an error.  The error message will tell the user to try and
     * run again, but with a larger array for storing events. */
    if (lwda_event_htod_num_used == lwda_event_max) {
        opal_show_help("help-mpi-common-lwca.txt", "Out of lwEvent handles",
                       true, lwda_event_max, lwda_event_max+100, lwda_event_max+100);
        OPAL_THREAD_UNLOCK(&common_lwda_htod_lock);
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    if (lwda_event_htod_num_used > lwda_event_htod_most) {
        lwda_event_htod_most = lwda_event_htod_num_used;
        /* Just print multiples of 10 */
        if (0 == (lwda_event_htod_most % 10)) {
            opal_output_verbose(20, mca_common_lwda_output,
                                "Maximum HtoD events used is now %d", lwda_event_htod_most);
        }
    }

    result = lwFunc.lwEventRecord(lwda_event_htod_array[lwda_event_htod_first_avail], htodStream);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwEventRecord failed",
                       true, OPAL_PROC_MY_HOSTNAME, result);
        OPAL_THREAD_UNLOCK(&common_lwda_htod_lock);
        return OPAL_ERROR;
    }
    lwda_event_htod_frag_array[lwda_event_htod_first_avail] = frag;

   /* Bump up the first available slot and number used by 1 */
    lwda_event_htod_first_avail++;
    if (lwda_event_htod_first_avail >= lwda_event_max) {
        lwda_event_htod_first_avail = 0;
    }
    lwda_event_htod_num_used++;

    OPAL_THREAD_UNLOCK(&common_lwda_htod_lock);
    return OPAL_SUCCESS;
}

/**
 * Used to get the dtoh stream for initiating asynchronous copies.
 */
void *mca_common_lwda_get_dtoh_stream(void) {
    return (void *)dtohStream;
}

/**
 * Used to get the htod stream for initiating asynchronous copies.
 */
void *mca_common_lwda_get_htod_stream(void) {
    return (void *)htodStream;
}

/*
 * Function is called every time progress is called with the sm BTL.  If there
 * are outstanding events, check to see if one has completed.  If so, hand
 * back the fragment for further processing.
 */
int progress_one_lwda_ipc_event(struct mca_btl_base_descriptor_t **frag) {
    LWresult result;

    OPAL_THREAD_LOCK(&common_lwda_ipc_lock);
    if (lwda_event_ipc_num_used > 0) {
        opal_output_verbose(20, mca_common_lwda_output,
                           "LWCA: progress_one_lwda_ipc_event, outstanding_events=%d",
                            lwda_event_ipc_num_used);

        result = lwFunc.lwEventQuery(lwda_event_ipc_array[lwda_event_ipc_first_used]);

        /* We found an event that is not ready, so return. */
        if (LWDA_ERROR_NOT_READY == result) {
            opal_output_verbose(20, mca_common_lwda_output,
                                "LWCA: lwEventQuery returned LWDA_ERROR_NOT_READY");
            *frag = NULL;
            OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
            return 0;
        } else if (LWDA_SUCCESS != result) {
            opal_show_help("help-mpi-common-lwca.txt", "lwEventQuery failed",
                           true, result);
            *frag = NULL;
            OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
            return OPAL_ERROR;
        }

        *frag = lwda_event_ipc_frag_array[lwda_event_ipc_first_used];
        opal_output_verbose(10, mca_common_lwda_output,
                            "LWCA: lwEventQuery returned %d", result);

        /* Bump counters, loop around the cirlwlar buffer if necessary */
        --lwda_event_ipc_num_used;
        ++lwda_event_ipc_first_used;
        if (lwda_event_ipc_first_used >= lwda_event_max) {
            lwda_event_ipc_first_used = 0;
        }
        /* A return value of 1 indicates an event completed and a frag was returned */
        OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
        return 1;
    }
    OPAL_THREAD_UNLOCK(&common_lwda_ipc_lock);
    return 0;
}

/**
 * Progress any dtoh event completions.
 */
int progress_one_lwda_dtoh_event(struct mca_btl_base_descriptor_t **frag) {
    LWresult result;

    OPAL_THREAD_LOCK(&common_lwda_dtoh_lock);
    if (lwda_event_dtoh_num_used > 0) {
        opal_output_verbose(30, mca_common_lwda_output,
                           "LWCA: progress_one_lwda_dtoh_event, outstanding_events=%d",
                            lwda_event_dtoh_num_used);

        result = lwFunc.lwEventQuery(lwda_event_dtoh_array[lwda_event_dtoh_first_used]);

        /* We found an event that is not ready, so return. */
        if (LWDA_ERROR_NOT_READY == result) {
            opal_output_verbose(30, mca_common_lwda_output,
                                "LWCA: lwEventQuery returned LWDA_ERROR_NOT_READY");
            *frag = NULL;
            OPAL_THREAD_UNLOCK(&common_lwda_dtoh_lock);
            return 0;
        } else if (LWDA_SUCCESS != result) {
            opal_show_help("help-mpi-common-lwca.txt", "lwEventQuery failed",
                           true, result);
            *frag = NULL;
            OPAL_THREAD_UNLOCK(&common_lwda_dtoh_lock);
            return OPAL_ERROR;
        }

        *frag = lwda_event_dtoh_frag_array[lwda_event_dtoh_first_used];
        opal_output_verbose(30, mca_common_lwda_output,
                            "LWCA: lwEventQuery returned %d", result);

        /* Bump counters, loop around the cirlwlar buffer if necessary */
        --lwda_event_dtoh_num_used;
        ++lwda_event_dtoh_first_used;
        if (lwda_event_dtoh_first_used >= lwda_event_max) {
            lwda_event_dtoh_first_used = 0;
        }
        /* A return value of 1 indicates an event completed and a frag was returned */
        OPAL_THREAD_UNLOCK(&common_lwda_dtoh_lock);
        return 1;
    }
    OPAL_THREAD_UNLOCK(&common_lwda_dtoh_lock);
    return 0;
}

/**
 * Progress any dtoh event completions.
 */
int progress_one_lwda_htod_event(struct mca_btl_base_descriptor_t **frag) {
    LWresult result;

    OPAL_THREAD_LOCK(&common_lwda_htod_lock);
    if (lwda_event_htod_num_used > 0) {
        opal_output_verbose(30, mca_common_lwda_output,
                           "LWCA: progress_one_lwda_htod_event, outstanding_events=%d",
                            lwda_event_htod_num_used);

        result = lwFunc.lwEventQuery(lwda_event_htod_array[lwda_event_htod_first_used]);

        /* We found an event that is not ready, so return. */
        if (LWDA_ERROR_NOT_READY == result) {
            opal_output_verbose(30, mca_common_lwda_output,
                                "LWCA: lwEventQuery returned LWDA_ERROR_NOT_READY");
            *frag = NULL;
            OPAL_THREAD_UNLOCK(&common_lwda_htod_lock);
            return 0;
        } else if (LWDA_SUCCESS != result) {
            opal_show_help("help-mpi-common-lwca.txt", "lwEventQuery failed",
                           true, result);
            *frag = NULL;
            OPAL_THREAD_UNLOCK(&common_lwda_htod_lock);
            return OPAL_ERROR;
        }

        *frag = lwda_event_htod_frag_array[lwda_event_htod_first_used];
        opal_output_verbose(30, mca_common_lwda_output,
                            "LWCA: lwEventQuery returned %d", result);

        /* Bump counters, loop around the cirlwlar buffer if necessary */
        --lwda_event_htod_num_used;
        ++lwda_event_htod_first_used;
        if (lwda_event_htod_first_used >= lwda_event_max) {
            lwda_event_htod_first_used = 0;
        }
        /* A return value of 1 indicates an event completed and a frag was returned */
        OPAL_THREAD_UNLOCK(&common_lwda_htod_lock);
        return 1;
    }
    OPAL_THREAD_UNLOCK(&common_lwda_htod_lock);
    return OPAL_ERR_RESOURCE_BUSY;
}


/**
 * Need to make sure the handle we are retrieving from the cache is still
 * valid.  Compare the cached handle to the one received.
 */
int mca_common_lwda_memhandle_matches(mca_rcache_common_lwda_reg_t *new_reg,
                                      mca_rcache_common_lwda_reg_t *old_reg)
{

    if (0 == memcmp(new_reg->data.memHandle, old_reg->data.memHandle, sizeof(new_reg->data.memHandle))) {
        return 1;
    } else {
        return 0;
    }

}

/*
 * Function to dump memory handle information.  This is based on
 * definitions from lwiinterprocess_private.h.
 */
static void lwda_dump_memhandle(int verbose, void *memHandle, char *str) {

    struct InterprocessMemHandleInternal
    {
        /* The first two entries are the LWinterprocessCtxHandle */
        int64_t ctxId; /* unique (within a process) id of the sharing context */
        int     pid;   /* pid of sharing context */

        int64_t size;
        int64_t blocksize;
        int64_t offset;
        int     gpuId;
        int     subDeviceIndex;
        int64_t serial;
    } memH;

    if (NULL == str) {
        str = "LWCA";
    }
    memcpy(&memH, memHandle, sizeof(memH));
    opal_output_verbose(verbose, mca_common_lwda_output,
                        "%s:ctxId=0x%" PRIx64 ", pid=%d, size=%" PRIu64 ", blocksize=%" PRIu64 ", offset=%"
                        PRIu64 ", gpuId=%d, subDeviceIndex=%d, serial=%" PRIu64,
                        str, memH.ctxId, memH.pid, memH.size, memH.blocksize, memH.offset,
                        memH.gpuId, memH.subDeviceIndex, memH.serial);
}

/*
 * Function to dump memory handle information.  This is based on
 * definitions from lwiinterprocess_private.h.
 */
static void lwda_dump_evthandle(int verbose, void *evtHandle, char *str) {

    struct InterprocessEventHandleInternal
    {
        unsigned long pid;
        unsigned long serial;
        int index;
    } evtH;

    if (NULL == str) {
        str = "LWCA";
    }
    memcpy(&evtH, evtHandle, sizeof(evtH));
    opal_output_verbose(verbose, mca_common_lwda_output,
                        "LWCA: %s:pid=%lu, serial=%lu, index=%d",
                        str, evtH.pid, evtH.serial, evtH.index);
}


/* Return microseconds of elapsed time. Microseconds are relevant when
 * trying to understand the fixed overhead of the communication. Used
 * when trying to time various functions.
 *
 * Cut and past the following to get timings where wanted.
 *
 *   clock_gettime(CLOCK_MONOTONIC, &ts_start);
 *   FUNCTION OF INTEREST
 *   clock_gettime(CLOCK_MONOTONIC, &ts_end);
 *   aclwm = mydifftime(ts_start, ts_end);
 *   opal_output(0, "Function took   %7.2f usecs\n", aclwm);
 *
 */
#if OPAL_ENABLE_DEBUG
static float mydifftime(opal_timer_t ts_start, opal_timer_t ts_end) {
    return (ts_end - ts_start);
}
#endif /* OPAL_ENABLE_DEBUG */

/* Routines that get plugged into the opal datatype code */
static int mca_common_lwda_is_gpu_buffer(const void *pUserBuf, opal_colwertor_t *colwertor)
{
    int res;
    LWmemorytype memType = 0;
    LWdeviceptr dbuf = (LWdeviceptr)pUserBuf;
    LWcontext ctx = NULL, memCtx = NULL;
#if OPAL_LWDA_GET_ATTRIBUTES
    uint32_t isManaged = 0;
    /* With LWCA 7.0, we can get multiple attributes with a single call */
    LWpointer_attribute attributes[3] = {LW_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         LW_POINTER_ATTRIBUTE_CONTEXT,
                                         LW_POINTER_ATTRIBUTE_IS_MANAGED};
    void *attrdata[] = {(void *)&memType, (void *)&memCtx, (void *)&isManaged};

    res = lwFunc.lwPointerGetAttributes(3, attributes, attrdata, dbuf);
    OPAL_OUTPUT_VERBOSE((101, mca_common_lwda_output,
                        "dbuf=%p, memType=%d, memCtx=%p, isManaged=%d, res=%d",
                         (void *)dbuf, (int)memType, (void *)memCtx, isManaged, res));

    /* Mark unified memory buffers with a flag.  This will allow all unified
     * memory to be forced through host buffers.  Note that this memory can
     * be either host or device so we need to set this flag prior to that check. */
    if (1 == isManaged) {
        if (NULL != colwertor) {
            colwertor->flags |= COLWERTOR_LWDA_UNIFIED;
        }
    }
    if (res != LWDA_SUCCESS) {
        /* If we cannot determine it is device pointer,
         * just assume it is not. */
        return 0;
    } else if (memType == LW_MEMORYTYPE_HOST) {
        /* Host memory, nothing to do here */
        return 0;
    } else if (memType == 0) {
        /* This can happen when LWCA is initialized but dbuf is not valid LWCA pointer */
        return 0;
    }
    /* Must be a device pointer */
    assert(memType == LW_MEMORYTYPE_DEVICE);
#else /* OPAL_LWDA_GET_ATTRIBUTES */
    res = lwFunc.lwPointerGetAttribute(&memType,
                                       LW_POINTER_ATTRIBUTE_MEMORY_TYPE, dbuf);
    if (res != LWDA_SUCCESS) {
        /* If we cannot determine it is device pointer,
         * just assume it is not. */
        return 0;
    } else if (memType == LW_MEMORYTYPE_HOST) {
        /* Host memory, nothing to do here */
        return 0;
    }
    /* Must be a device pointer */
    assert(memType == LW_MEMORYTYPE_DEVICE);
#endif /* OPAL_LWDA_GET_ATTRIBUTES */

    /* This piece of code was added in to handle in a case ilwolving
     * OMP threads.  The user had initialized LWCA and then spawned
     * two threads.  The first thread had the LWCA context, but the
     * second thread did not.  We therefore had no context to act upon
     * and future LWCA driver calls would fail.  Therefore, if we have
     * GPU memory, but no context, get the context from the GPU memory
     * and set the current context to that.  It is rare that we will not
     * have a context. */
    res = lwFunc.lwCtxGetLwrrent(&ctx);
    if (OPAL_UNLIKELY(NULL == ctx)) {
        if (LWDA_SUCCESS == res) {
#if !OPAL_LWDA_GET_ATTRIBUTES
            res = lwFunc.lwPointerGetAttribute(&memCtx,
                                               LW_POINTER_ATTRIBUTE_CONTEXT, dbuf);
            if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
                opal_output(0, "LWCA: error calling lwPointerGetAttribute: "
                            "res=%d, ptr=%p aborting...", res, pUserBuf);
                return OPAL_ERROR;
            }
#endif /* OPAL_LWDA_GET_ATTRIBUTES */
            res = lwFunc.lwCtxSetLwrrent(memCtx);
            if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
                opal_output(0, "LWCA: error calling lwCtxSetLwrrent: "
                            "res=%d, ptr=%p aborting...", res, pUserBuf);
                return OPAL_ERROR;
            } else {
                OPAL_OUTPUT_VERBOSE((10, mca_common_lwda_output,
                                     "LWCA: lwCtxSetLwrrent passed: ptr=%p", pUserBuf));
            }
        } else {
            /* Print error and proceed */
            opal_output(0, "LWCA: error calling lwCtxGetLwrrent: "
                        "res=%d, ptr=%p aborting...", res, pUserBuf);
            return OPAL_ERROR;
        }
    }

    /* WORKAROUND - They are times when the above code determines a pice of memory
     * is GPU memory, but it actually is not.  That has been seen on multi-GPU systems
     * with 6 or 8 GPUs on them. Therefore, we will do this extra check.  Note if we
     * made it this far, then the assumption at this point is we have GPU memory.
     * Unfotunately, this extra call is costing us another 100 ns almost doubling
     * the cost of this entire function. */
    if (OPAL_LIKELY(mca_common_lwda_gpu_mem_check_workaround)) {
        LWdeviceptr pbase;
        size_t psize;
        res = lwFunc.lwMemGetAddressRange(&pbase, &psize, dbuf);
        if (LWDA_SUCCESS != res) {
            opal_output_verbose(5, mca_common_lwda_output,
                                "LWCA: lwMemGetAddressRange failed on this pointer: res=%d, buf=%p "
                                "Overriding check and setting to host pointer. ",
                              res, (void *)dbuf);
            /* This cannot be GPU memory if the previous call failed */
            return 0;
        }
    }

    /* First access on a device pointer finalizes LWCA support initialization.
     * If initialization fails, disable support. */
    if (!stage_three_init_complete) {
        if (0 != mca_common_lwda_stage_three_init()) {
            opal_lwda_support = 0;
        }
    }

    return 1;
}

static int mca_common_lwda_lw_memcpy_async(void *dest, const void *src, size_t size,
                                         opal_colwertor_t* colwertor)
{
    return lwFunc.lwMemcpyAsync((LWdeviceptr)dest, (LWdeviceptr)src, size,
                                (LWstream)colwertor->stream);
}

/**
 * This function is plugged into various areas where a lwMemcpy would be called.
 * This is a synchronous operation that will not return until the copy is complete.
 */
static int mca_common_lwda_lw_memcpy(void *dest, const void *src, size_t size)
{
    LWresult result;
#if OPAL_ENABLE_DEBUG
    LWmemorytype memTypeSrc, memTypeDst;
    if (OPAL_UNLIKELY(mca_common_lwda_lwmemcpy_timing)) {
        /* Nice to know type of source and destination for timing output. Do
         * not care about return code as memory type will just be set to 0 */
        result = lwFunc.lwPointerGetAttribute(&memTypeDst,
                                              LW_POINTER_ATTRIBUTE_MEMORY_TYPE, (LWdeviceptr)dest);
        result = lwFunc.lwPointerGetAttribute(&memTypeSrc,
                                              LW_POINTER_ATTRIBUTE_MEMORY_TYPE, (LWdeviceptr)src);
        ts_start = opal_timer_base_get_usec();
    }
#endif
    if (mca_common_lwda_lwmemcpy_async) {
        result = lwFunc.lwMemcpyAsync((LWdeviceptr)dest, (LWdeviceptr)src, size, memcpyStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwMemcpyAsync failed",
                           true, dest, src, size, result);
            return OPAL_ERROR;
        }
        result = lwFunc.lwStreamSynchronize(memcpyStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwStreamSynchronize failed",
                           true, OPAL_PROC_MY_HOSTNAME, result);
            return OPAL_ERROR;
        }
    } else {
         result = lwFunc.lwMemcpy((LWdeviceptr)dest, (LWdeviceptr)src, size);
         if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
             opal_show_help("help-mpi-common-lwca.txt", "lwMemcpy failed",
                            true, OPAL_PROC_MY_HOSTNAME, result);
             return OPAL_ERROR;
         }
    }
#if OPAL_ENABLE_DEBUG
    if (OPAL_UNLIKELY(mca_common_lwda_lwmemcpy_timing)) {
        ts_end = opal_timer_base_get_usec();
        aclwm = mydifftime(ts_start, ts_end);
        if (mca_common_lwda_lwmemcpy_async) {
            opal_output(0, "lwMemcpyAsync took   %7.2f usecs, size=%d, (src=%p (%d), dst=%p (%d))\n",
                        aclwm, (int)size, src, memTypeSrc, dest, memTypeDst);
        } else {
            opal_output(0, "lwMemcpy took   %7.2f usecs, size=%d,  (src=%p (%d), dst=%p (%d))\n",
                        aclwm, (int)size, src, memTypeSrc, dest, memTypeDst);
        }
    }
#endif
    return OPAL_SUCCESS;
}

static int mca_common_lwda_memmove(void *dest, void *src, size_t size)
{
    LWdeviceptr tmp;
    int result;

    result = lwFunc.lwMemAlloc(&tmp,size);
    if (mca_common_lwda_lwmemcpy_async) {
        result = lwFunc.lwMemcpyAsync(tmp, (LWdeviceptr)src, size, memcpyStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwMemcpyAsync failed",
                           true, tmp, src, size, result);
            return OPAL_ERROR;
        }
        result = lwFunc.lwMemcpyAsync((LWdeviceptr)dest, tmp, size, memcpyStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwMemcpyAsync failed",
                           true, dest, tmp, size, result);
            return OPAL_ERROR;
        }
        result = lwFunc.lwStreamSynchronize(memcpyStream);
        if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
            opal_show_help("help-mpi-common-lwca.txt", "lwStreamSynchronize failed",
                           true, OPAL_PROC_MY_HOSTNAME, result);
            return OPAL_ERROR;
        }
    } else {
        result = lwFunc.lwMemcpy(tmp, (LWdeviceptr)src, size);
        if (OPAL_UNLIKELY(result != LWDA_SUCCESS)) {
            opal_output(0, "LWCA: memmove-Error in lwMemcpy: res=%d, dest=%p, src=%p, size=%d",
                        result, (void *)tmp, src, (int)size);
            return OPAL_ERROR;
        }
        result = lwFunc.lwMemcpy((LWdeviceptr)dest, tmp, size);
        if (OPAL_UNLIKELY(result != LWDA_SUCCESS)) {
            opal_output(0, "LWCA: memmove-Error in lwMemcpy: res=%d, dest=%p, src=%p, size=%d",
                        result, dest, (void *)tmp, (int)size);
            return OPAL_ERROR;
        }
    }
    lwFunc.lwMemFree(tmp);
    return OPAL_SUCCESS;
}

int mca_common_lwda_get_device(int *devicenum)
{
    LWdevice lwDev;
    int res;

    res = lwFunc.lwCtxGetDevice(&lwDev);
    if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
        opal_output(0, "LWCA: lwCtxGetDevice failed: res=%d",
                    res);
        return res;
    }
    *devicenum = lwDev;
    return 0;
}

int mca_common_lwda_device_can_access_peer(int *access, int dev1, int dev2)
{
    int res;
    res = lwFunc.lwDeviceCanAccessPeer(access, (LWdevice)dev1, (LWdevice)dev2);
    if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
        opal_output(0, "LWCA: lwDeviceCanAccessPeer failed: res=%d",
                    res);
        return res;
    }
    return 0;
}

int mca_common_lwda_get_address_range(void *pbase, size_t *psize, void *base)
{
    LWresult result;
    result = lwFunc.lwMemGetAddressRange((LWdeviceptr *)pbase, psize, (LWdeviceptr)base);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwMemGetAddressRange failed 2",
                       true, OPAL_PROC_MY_HOSTNAME, result, base);
        return OPAL_ERROR;
    } else {
        opal_output_verbose(50, mca_common_lwda_output,
                            "LWCA: lwMemGetAddressRange passed: addr=%p, pbase=%p, psize=%lu ",
                            base, *(char **)pbase, *psize);
    }
    return 0;
}

#if OPAL_LWDA_GDR_SUPPORT
/* Check to see if the memory was freed between the time it was stored in
 * the registration cache and now.  Return true if the memory was previously
 * freed.  This is indicated by the BUFFER_ID value in the registration cache
 * not matching the BUFFER_ID of the buffer we are checking.  Return false
 * if the registration is still good.
 */
bool mca_common_lwda_previously_freed_memory(mca_rcache_base_registration_t *reg)
{
    int res;
    unsigned long long bufID;
    unsigned char *dbuf = reg->base;

    res = lwFunc.lwPointerGetAttribute(&bufID, LW_POINTER_ATTRIBUTE_BUFFER_ID,
                                       (LWdeviceptr)dbuf);
    /* If we cannot determine the BUFFER_ID, then print a message and default
     * to forcing the registration to be kicked out. */
    if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
        opal_show_help("help-mpi-common-lwca.txt", "bufferID failed",
                       true, OPAL_PROC_MY_HOSTNAME, res);
        return true;
    }
    opal_output_verbose(50, mca_common_lwda_output,
                        "LWCA: base=%p, bufID=%llu, reg->gpu_bufID=%llu, %s", dbuf, bufID, reg->gpu_bufID,
                        (reg->gpu_bufID == bufID ? "BUFFER_ID match":"BUFFER_ID do not match"));
    if (bufID != reg->gpu_bufID) {
        return true;
    } else {
        return false;
    }
}

/*
 * Get the buffer ID from the memory and store it in the registration.
 * This is needed to ensure the cached registration is not stale.  If
 * we fail to get buffer ID, print an error and set buffer ID to 0.
 * Also set SYNC_MEMOPS on any GPU registration to ensure that
 * synchronous copies complete before the buffer is accessed.
 */
void mca_common_lwda_get_buffer_id(mca_rcache_base_registration_t *reg)
{
    int res;
    unsigned long long bufID = 0;
    unsigned char *dbuf = reg->base;
    int enable = 1;

    res = lwFunc.lwPointerGetAttribute(&bufID, LW_POINTER_ATTRIBUTE_BUFFER_ID,
                                       (LWdeviceptr)dbuf);
    if (OPAL_UNLIKELY(res != LWDA_SUCCESS)) {
        opal_show_help("help-mpi-common-lwca.txt", "bufferID failed",
                       true, OPAL_PROC_MY_HOSTNAME, res);
    }
    reg->gpu_bufID = bufID;

    res = lwFunc.lwPointerSetAttribute(&enable, LW_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                       (LWdeviceptr)dbuf);
    if (OPAL_UNLIKELY(LWDA_SUCCESS != res)) {
        opal_show_help("help-mpi-common-lwca.txt", "lwPointerSetAttribute failed",
                       true, OPAL_PROC_MY_HOSTNAME, res, dbuf);
    }
}
#endif /* OPAL_LWDA_GDR_SUPPORT */
