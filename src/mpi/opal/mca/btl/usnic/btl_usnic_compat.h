/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2013-2018 Cisco Systems, Inc.  All rights reserved
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2018      Intel, Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* This header contains macros to help minimize usnic BTL differences
 * between v1.7/v1.8, v1.9/v2.0, and v2.0/v2.1. */

#ifndef BTL_USNIC_COMPAT_H
#define BTL_USNIC_COMPAT_H

#include "opal/mca/rcache/rcache.h"
#include "opal/mca/btl/btl.h"

/************************************************************************/

/* v2.0 and beyond */

#if (OPAL_MAJOR_VERSION >= 2)

/* OMPI_ERROR_LOG and friends */
#  include "opal/util/error.h"

/* PMIX / modex stuff */
#  include "opal/mca/pmix/pmix.h"

/* Proc stuff */
#  include "opal/util/proc.h"

/* Free lists are unified into OPAL free lists */
#  include "opal/class/opal_free_list.h"

/* Inclue the progress thread stuff */
#  include "opal/runtime/opal_progress_threads.h"

/* Hwloc support is now guaranteed, and the rest of the code base does
   not define OPAL_HAVE_HWLOC any more (because it would always be 1).

   Note: The usnic BTL still uses OPAL_HAVE_HWLOC because Cisco
   continues to sync it against a v1.10-based tree (where
   OPAL_HAVE_HWLOC may still be 0 or 1).  Once Cisco stops syncing the
   usnic BTL against v1.10.x, all the OPAL_HAVE_HWLOC code in the
   usnic BTL can go away. */
#  define OPAL_HAVE_HWLOC 1

#  define USNIC_OUT opal_btl_base_framework.framework_output
/* JMS Really want to be able to get the job size somehow...  But for
   now, so that we can compile, just set it to a constant :-( */
#  define USNIC_MCW_SIZE 2
#if OPAL_HAVE_HWLOC
#    define proc_bound() (NULL != opal_process_info.cpuset ? 1 : 0)
#else
#    define proc_bound() 0
#endif
#  define USNIC_BTL_DEFAULT_VERSION(name) MCA_BTL_DEFAULT_VERSION(name)

#  define USNIC_SEND_LOCAL        des_segments
#  define USNIC_SEND_LOCAL_COUNT  des_segment_count
#  define USNIC_SEND_REMOTE       des_segments
#  define USNIC_SEND_REMOTE_COUNT des_segment_count

#  define USNIC_RECV_LOCAL        des_segments
#  define USNIC_RECV_LOCAL_COUNT  des_segment_count
#  define USNIC_RECV_REMOTE       des_segments
#  define USNIC_RECV_REMOTE_COUNT des_segment_count

#  define USNIC_PUT_LOCAL         des_segments
#  define USNIC_PUT_LOCAL_COUNT   des_segment_count
#  define USNIC_PUT_REMOTE        des_segments
#  define USNIC_PUT_REMOTE_COUNT  des_segments_count

// Starting after Open MPI v3.1.0, the BTL_VERSION macro was defined
// by btl.h (it'll likely get into v4.0.0 -- don't know if this change
// will migrate to the v3.x.y branches).  So if BTL_VERSION is already
// defined, then we don't need to define it again.  As of this writing
// (Feb 2018), this set of defines works fine with BTL v3.0.0 and
// v3.1.0.  So we'll set the BTL version to the minimium acceptable
// value: 3.0.0.
#  if !defined(BTL_VERSION)
#      define BTL_VERSION 300
#  endif

#  define USNIC_COMPAT_FREE_LIST_GET(list, item) \
    (item) = opal_free_list_get((list))
#  define USNIC_COMPAT_FREE_LIST_RETURN(list, item) \
    opal_free_list_return((list), (item))

#  define usnic_compat_free_list_init opal_free_list_init

/*
 * Performance critical; needs to be inline
 */
static inline int
usnic_compat_proc_name_compare(opal_process_name_t a,
                               opal_process_name_t b)
{
    return (bool) (a.jobid == b.jobid && a.vpid == b.vpid);
}

/************************************************************************/

/* v1.7, v1.8, and v1.10 (there was no v1.9) */

#elif (OPAL_MAJOR_VERSION == 1 && OPAL_MINOR_VERSION >= 7)

/* OMPI_ERROR_LOG and friends */
#  include "ompi/mca/rte/rte.h"

/* Proc stuff */
#  include "ompi/proc/proc.h"

/* Use OMPI free lists in v1.8 */
#  include "ompi/class/ompi_free_list.h"

#  define USNIC_OUT ompi_btl_base_framework.framework_output
#  define USNIC_MCW_SIZE ompi_process_info.num_procs
#  define proc_bound() (ompi_rte_proc_is_bound)
#  define opal_proc_local_get() ompi_proc_local()

#  define opal_sync_event_base opal_event_base

#  define opal_process_info orte_process_info

#  define opal_proc_t ompi_proc_t
#  define opal_process_name_t ompi_process_name_t

#  define opal_btl_usnic_modex_t ompi_btl_usnic_modex_t
#  define opal_btl_usnic_component_t ompi_btl_usnic_component_t
#  define opal_btl_usnic_module_t ompi_btl_usnic_module_t
#  define opal_btl_usnic_endpoint_t ompi_btl_usnic_endpoint_t
#  define opal_btl_usnic_endpoint_t_class ompi_btl_usnic_endpoint_t_class
#  define opal_btl_usnic_frag_t ompi_btl_usnic_frag_t
#  define opal_btl_usnic_frag_t_class ompi_btl_usnic_frag_t_class
#  define opal_btl_usnic_send_frag_t ompi_btl_usnic_send_frag_t
#  define opal_btl_usnic_send_frag_t_class ompi_btl_usnic_send_frag_t_class
#  define opal_btl_usnic_large_send_frag_t ompi_btl_usnic_large_send_frag_t
#  define opal_btl_usnic_large_send_frag_t_class ompi_btl_usnic_large_send_frag_t_class
#  define opal_btl_usnic_small_send_frag_t ompi_btl_usnic_small_send_frag_t
#  define opal_btl_usnic_small_send_frag_t_class ompi_btl_usnic_small_send_frag_t_class
#  define opal_btl_usnic_put_dest_frag_t ompi_btl_usnic_put_dest_frag_t
#  define opal_btl_usnic_put_dest_frag_t_class ompi_btl_usnic_put_dest_frag_t_class
#  define opal_btl_usnic_rx_buf_t ompi_btl_usnic_rx_buf_t
#  define opal_btl_usnic_rx_buf_t_class ompi_btl_usnic_rx_buf_t_class
#  define opal_btl_usnic_segment_t ompi_btl_usnic_segment_t
#  define opal_btl_usnic_segment_t_class ompi_btl_usnic_segment_t_class
#  define opal_btl_usnic_frag_segment_t ompi_btl_usnic_frag_segment_t
#  define opal_btl_usnic_frag_segment_t_class ompi_btl_usnic_frag_segment_t_class
#  define opal_btl_usnic_chunk_segment_t ompi_btl_usnic_chunk_segment_t
#  define opal_btl_usnic_chunk_segment_t_class ompi_btl_usnic_chunk_segment_t_class
#  define opal_btl_usnic_recv_segment_t ompi_btl_usnic_recv_segment_t
#  define opal_btl_usnic_recv_segment_t_class ompi_btl_usnic_recv_segment_t_class
#  define opal_btl_usnic_ack_segment_t ompi_btl_usnic_ack_segment_t
#  define opal_btl_usnic_ack_segment_t_class ompi_btl_usnic_ack_segment_t_class

#  define opal_btl_usnic_run_tests ompi_btl_usnic_run_tests

#  define USNIC_SEND_LOCAL        des_src
#  define USNIC_SEND_LOCAL_COUNT  des_src_cnt
#  define USNIC_SEND_REMOTE       des_dst
#  define USNIC_SEND_REMOTE_COUNT des_dst_cnt

#  define USNIC_RECV_LOCAL        des_dst
#  define USNIC_RECV_LOCAL_COUNT  des_dst_cnt
#  define USNIC_RECV_REMOTE       des_src
#  define USNIC_RECV_REMOTE_COUNT des_src_cnt

#  define USNIC_PUT_LOCAL         des_dst
#  define USNIC_PUT_LOCAL_COUNT   des_dst_cnt
#  define USNIC_PUT_REMOTE        des_dst
#  define USNIC_PUT_REMOTE_COUNT  des_dst_cnt

#  define BTL_VERSION 20

#  define opal_free_list_t              ompi_free_list_t
#  define opal_free_list_item_t         ompi_free_list_item_t
#  define opal_free_list_item_init_fn_t ompi_free_list_item_init_fn_t

#  define USNIC_COMPAT_FREE_LIST_GET(list, item) \
    OMPI_FREE_LIST_GET_MT(list, (item))
#  define USNIC_COMPAT_FREE_LIST_RETURN(list, item) \
    OMPI_FREE_LIST_RETURN_MT((list), (item))

#  define USNIC_COMPAT_BASE_VERSION                                 \
    MCA_BASE_VERSION_2_0_0,                                         \
        .mca_type_name = "btl",                                     \
        .mca_type_major_version = OMPI_MAJOR_VERSION,               \
        .mca_type_minor_version = OMPI_MINOR_VERSION,               \
        .mca_type_release_version = OMPI_RELEASE_VERSION

#  define USNIC_BTL_DEFAULT_VERSION(name)                       \
    USNIC_COMPAT_BASE_VERSION,                                  \
        .mca_component_name = name,                             \
        .mca_component_major_version = OPAL_MAJOR_VERSION,      \
        .mca_component_minor_version = OPAL_MINOR_VERSION,      \
        .mca_component_release_version = OPAL_RELEASE_VERSION

#define OPAL_BTL_USNIC_UNIT_TESTS OMPI_BTL_USNIC_UNIT_TESTS

/*
 * Performance critical; needs to be inline
 */
static inline int
usnic_compat_proc_name_compare(opal_process_name_t a,
                               opal_process_name_t b)
{
    return ompi_rte_compare_name_fields(OMPI_RTE_CMP_ALL, &a, &b);
}

/* Hotels in v1.8 */
#  include "opal/class/opal_hotel.h"

/*
 * Performance critical; needs to be inline
 */
static inline int
usnic_compat_opal_hotel_init(opal_hotel_t *hotel, int num_rooms,
                             opal_event_base_t *evbase,
                             uint32_t eviction_timeout,
                             int eviction_event_priority,
                             opal_hotel_eviction_callback_fn_t evict_callback_fn)
{
    return opal_hotel_init(hotel, num_rooms, eviction_timeout,
                           eviction_event_priority, evict_callback_fn);
}
#define opal_hotel_init usnic_compat_opal_hotel_init


/*
 * Replicate functions that exist on master
 */
char* opal_get_proc_hostname(opal_proc_t *proc);

/*
 * Wrapper to call ompi_free_list_init
 */
int usnic_compat_free_list_init(opal_free_list_t *free_list,
                                size_t frag_size,
                                size_t frag_alignment,
                                opal_class_t* frag_class,
                                size_t payload_buffer_size,
                                size_t payload_buffer_alignment,
                                int num_elements_to_alloc,
                                int max_elements_to_alloc,
                                int num_elements_per_alloc,
                                struct mca_mpool_base_module_t *mpool,
                                int mpool_reg_flags,
                                void *unused0,
                                opal_free_list_item_init_fn_t item_init,
                                void *ctx);

/*
 * Start the connectivity checker progress thread
 */
opal_event_base_t *opal_progress_thread_init(const char *name);

/*
 * Stop the connectivity checker progress thread
 */
int opal_progress_thread_finalize(const char *name);


/************************************************************************/

#else
#  error OMPI version too old (< 1.7)
#endif

/************************************************************************
 * Common to all versions
 ************************************************************************/

/* Forward declare to avoid #include ordering complications */
struct opal_btl_usnic_modex_t;

void usnic_compat_modex_send(int *rc,
                             mca_base_component_t *component,
                             struct opal_btl_usnic_modex_t *modexes,
                             size_t size);

void usnic_compat_modex_recv(int *rc,
                             mca_base_component_t *component,
                             opal_proc_t *proc,
                             struct opal_btl_usnic_modex_t **modexes,
                             size_t *size);

uint64_t usnic_compat_rte_hash_name(opal_process_name_t *pname);
const char *usnic_compat_proc_name_print(opal_process_name_t *pname);

/************************************************************************/

/* BTL 2.0 vs 3.0 compatibilty functions (specifically: some BTL API
   functions changed signatures between 2.0 and 3.0) */

struct mca_btl_base_module_t;
struct mca_btl_base_endpoint_t;

/* BTL 2.0 (i.e., v1.7/v1.8, but listed separately because these are
   really BTL API issues) */

#if BTL_VERSION == 20

/* This function changed signature in BTL 3.0 */
mca_btl_base_descriptor_t*
opal_btl_usnic_prepare_src(
    struct mca_btl_base_module_t* base_module,
    struct mca_btl_base_endpoint_t* endpoint,
    struct mca_mpool_base_registration_t* registration,
    struct opal_colwertor_t* colwertor,
    uint8_t order,
    size_t reserve,
    size_t* size,
    uint32_t flags);

/* This function no longer exists in BTL 3.0 */
mca_btl_base_descriptor_t*
opal_btl_usnic_prepare_dst(
    struct mca_btl_base_module_t* base_module,
    struct mca_btl_base_endpoint_t* endpoint,
    struct mca_mpool_base_registration_t* registration,
    struct opal_colwertor_t* colwertor,
    uint8_t order,
    size_t reserve,
    size_t* size,
    uint32_t flags);

/* This function changed signature in BTL 3.0 */
int
opal_btl_usnic_put(
    struct mca_btl_base_module_t *btl,
    struct mca_btl_base_endpoint_t *endpoint,
    struct mca_btl_base_descriptor_t *desc);

/************************************************************************/

/* BTL 3.0 (i.e., >=v1.9, but listed separately because these are
   really BTL API issues) */

#elif BTL_VERSION >= 300

/* This function changed signature compared to BTL 2.0 */
struct mca_btl_base_descriptor_t *
opal_btl_usnic_prepare_src(struct mca_btl_base_module_t *base_module,
                           struct mca_btl_base_endpoint_t *endpoint,
                           struct opal_colwertor_t *colwertor,
                           uint8_t order,
                           size_t reserve,
                           size_t *size,
                           uint32_t flags);

/* This function changed signature compared to BTL 2.0 */
int
opal_btl_usnic_put(struct mca_btl_base_module_t *base_module,
                   struct mca_btl_base_endpoint_t *endpoint,
                   void *local_address, uint64_t remote_address,
                   struct mca_btl_base_registration_handle_t *local_handle,
                   struct mca_btl_base_registration_handle_t *remote_handle,
                   size_t size, int flags, int order,
                   mca_btl_base_rdma_completion_fn_t cbfunc,
                   void *cbcontext, void *cbdata);

#endif /* BTL_VERSION */

#if defined(RCACHE_MAJOR_VERSION) && RCACHE_MAJOR_VERSION >= 3

#define RCACHE_VERSION 30

/* these structures got renamed with the mpool/rcache rewrite */
#define mca_mpool_base_registration_t mca_rcache_base_registration_t
#define mca_mpool_base_resources_t mca_rcache_base_resources_t

#else

#define RCACHE_VERSION 20

#endif

#endif /* BTL_USNIC_COMPAT_H */
