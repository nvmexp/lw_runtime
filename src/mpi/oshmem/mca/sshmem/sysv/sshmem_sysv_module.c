/*
 * Copyright (c) 2014      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "oshmem_config.h"

#include <errno.h>
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif  /* HAVE_FCNTL_H */
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif /* HAVE_SYS_MMAN_H */
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif /* HAVE_SYS_TYPES_H */
#ifdef HAVE_SYS_IPC_H
#include <sys/ipc.h>
#endif /* HAVE_SYS_IPC_H */
#if HAVE_SYS_SHM_H
#include <sys/shm.h>
#endif /* HAVE_SYS_SHM_H */
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif /* HAVE_SYS_STAT_H */
#include <string.h>
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif /* HAVE_NETDB_H */

#include "opal/constants.h"
#include "opal_stdint.h"
#include "opal/util/output.h"
#include "opal/util/path.h"
#include "opal/util/show_help.h"
#include "orte/util/show_help.h"

#include "oshmem/proc/proc.h"
#include "oshmem/mca/sshmem/sshmem.h"
#include "oshmem/mca/sshmem/base/base.h"

#include "sshmem_sysv.h"


/* ////////////////////////////////////////////////////////////////////////// */
/* local functions */
static int
module_init(void);

static int
segment_create(map_segment_t *ds_buf,
               const char *file_name,
               size_t size, long hint);

static void *
segment_attach(map_segment_t *ds_buf, sshmem_mkey_t *mkey);

static int
segment_detach(map_segment_t *ds_buf, sshmem_mkey_t *mkey);

static int
segment_unlink(map_segment_t *ds_buf);

static int
module_finalize(void);

/* sysv shmem module */
mca_sshmem_sysv_module_t mca_sshmem_sysv_module = {
    /* super */
    {
        module_init,
        segment_create,
        segment_attach,
        segment_detach,
        segment_unlink,
        module_finalize
    }
};


/* ////////////////////////////////////////////////////////////////////////// */
static int
module_init(void)
{
    /* nothing to do */
    return OSHMEM_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
module_finalize(void)
{
    /* nothing to do */
    return OSHMEM_SUCCESS;
}


/* ////////////////////////////////////////////////////////////////////////// */
static int
segment_create(map_segment_t *ds_buf,
               const char *file_name,
               size_t size, long hint)
{
    int rc = OSHMEM_SUCCESS;
    void *addr = NULL;
    int shmid = MAP_SEGMENT_SHM_ILWALID;
    int flags;
    int try_hp;

    assert(ds_buf);

    if (hint) {
        return OSHMEM_ERR_NOT_IMPLEMENTED;
    }

    /* init the contents of map_segment_t */
    shmem_ds_reset(ds_buf);

    /* for sysv shared memory we don't have to worry about the backing store
     * being located on a network file system... so no check is needed here.
     */

    /* create a new shared memory segment and save the shmid. note the use of
     * real_size here
     */
    flags = IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR;
    try_hp = mca_sshmem_sysv_component.use_hp;
#if defined (SHM_HUGETLB)
    flags |= ((0 != try_hp) ? SHM_HUGETLB : 0);
    size = ((size + sshmem_sysv_gethugepagesize() - 1) / sshmem_sysv_gethugepagesize()) * sshmem_sysv_gethugepagesize();
#endif

    /* Create a new shared memory segment and save the shmid. */
retry_alloc:
    shmid = shmget(IPC_PRIVATE, size, flags);
    if (shmid == MAP_SEGMENT_SHM_ILWALID) {
        /* hugepage alloc was set to auto. Hopefully it failed because there are no
         * enough hugepages on the system. Turn it off and retry.
         */
        if (-1 == try_hp) {
            OPAL_OUTPUT_VERBOSE(
                    (10, oshmem_sshmem_base_framework.framework_output,
                     "failed to allocate %llu bytes with huge pages. "
                     "Using regular pages", (unsigned long long)size));
            flags = IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR;
            try_hp = 0;
            goto retry_alloc;
        }
        opal_show_help("help-oshmem-sshmem.txt",
                       "create segment failure",
                       true,
                       "sysv",
                       orte_process_info.nodename, (unsigned long long) size,
                       strerror(errno), errno);
        opal_show_help("help-oshmem-sshmem-sysv.txt",
                       "sysv:create segment failure",
                       true);
        return OSHMEM_ERROR;
    }

    /* Attach to the segment */
    addr = shmat(shmid, (void *) mca_sshmem_base_start_address, 0);
    if (addr == (void *) -1L) {
        opal_show_help("help-oshmem-sshmem.txt",
                       "create segment failure",
                       true,
                       "sysv",
                       orte_process_info.nodename, (unsigned long long) size,
                       strerror(errno), errno);
        opal_show_help("help-oshmem-sshmem-sysv.txt",
                       "sysv:create segment failure",
                       true);
        shmctl(shmid, IPC_RMID, NULL);
        return OSHMEM_ERR_OUT_OF_RESOURCE;
    }

    shmctl(shmid, IPC_RMID, NULL );

    ds_buf->type = MAP_SEGMENT_ALLOC_SHM;
    ds_buf->seg_id = shmid;
    ds_buf->super.va_base = addr;
    ds_buf->seg_size = size;
    ds_buf->super.va_end = (void*)((uintptr_t)ds_buf->super.va_base + ds_buf->seg_size);

    OPAL_OUTPUT_VERBOSE(
          (70, oshmem_sshmem_base_framework.framework_output,
           "%s: %s: create %s "
           "(id: %d, addr: %p size: %lu)\n",
           mca_sshmem_sysv_component.super.base_version.mca_type_name,
           mca_sshmem_sysv_component.super.base_version.mca_component_name,
           (rc ? "failure" : "successful"),
           ds_buf->seg_id, ds_buf->super.va_base, (unsigned long)ds_buf->seg_size)
      );

    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
/**
 * segment_attach can only be called after a successful call to segment_create
 */
static void *
segment_attach(map_segment_t *ds_buf, sshmem_mkey_t *mkey)
{
    assert(ds_buf);
    assert(mkey->va_base == 0);

    if (MAP_SEGMENT_SHM_ILWALID == (int)(mkey->u.key)) {
        return (mkey->va_base);
    }

    mkey->va_base = shmat((int)(mkey->u.key), 0, 0);

    OPAL_OUTPUT_VERBOSE(
        (70, oshmem_sshmem_base_framework.framework_output,
         "%s: %s: attach successful "
            "(id: %d, addr: %p size: %lu | va_base: 0x%p len: %d key %llx)\n",
            mca_sshmem_sysv_component.super.base_version.mca_type_name,
            mca_sshmem_sysv_component.super.base_version.mca_component_name,
            ds_buf->seg_id, ds_buf->super.va_base, (unsigned long)ds_buf->seg_size,
            mkey->va_base, mkey->len, (unsigned long long)mkey->u.key)
    );

    /* update returned base pointer with an offset that hides our stuff */
    return (mkey->va_base);
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
segment_detach(map_segment_t *ds_buf, sshmem_mkey_t *mkey)
{
    int rc = OSHMEM_SUCCESS;

    assert(ds_buf);

    OPAL_OUTPUT_VERBOSE(
        (70, oshmem_sshmem_base_framework.framework_output,
         "%s: %s: detaching "
            "(id: %d, addr: %p size: %lu)\n",
            mca_sshmem_sysv_component.super.base_version.mca_type_name,
            mca_sshmem_sysv_component.super.base_version.mca_component_name,
            ds_buf->seg_id, ds_buf->super.va_base, (unsigned long)ds_buf->seg_size)
    );

    if (ds_buf->seg_id != MAP_SEGMENT_SHM_ILWALID) {
        shmctl(ds_buf->seg_id, IPC_RMID, NULL );
    }

    if (mca_sshmem_sysv_component.use_hp != 0) {
        /**
         *  Workaround kernel panic when detaching huge pages from user space simultanously from several processes
         *  dont detach here instead let kernel do it during process cleanup
         */
        /* shmdt((void *)ds_buf->seg_base_addr); */
    }

    /* reset the contents of the map_segment_t associated with this
     * shared memory segment.
     */
    shmem_ds_reset(ds_buf);

    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
segment_unlink(map_segment_t *ds_buf)
{
    /* not much unlink work needed for sysv */

    OPAL_OUTPUT_VERBOSE(
        (70, oshmem_sshmem_base_framework.framework_output,
         "%s: %s: unlinking "
         "(id: %d, size: %lu)\n",
         mca_sshmem_sysv_component.super.base_version.mca_type_name,
         mca_sshmem_sysv_component.super.base_version.mca_component_name,
         ds_buf->seg_id, (unsigned long)ds_buf->seg_size)
    );

    /* don't completely reset.  in particular, only reset
     * the id and flip the invalid bit.  size and name values will remain valid
     * across unlinks. other information stored in flags will remain untouched.
     */
    ds_buf->seg_id = MAP_SEGMENT_SHM_ILWALID;
    /* note: this is only changing the valid bit to 0. */
    MAP_SEGMENT_ILWALIDATE(ds_buf);

    return OSHMEM_SUCCESS;
}

/*
 * Get current huge page size
 *
 */
size_t sshmem_sysv_gethugepagesize(void)
{
    static size_t huge_page_size = 0;
    char buf[256];
    int size_kb;
    FILE *f;

    /* Cache the huge page size value */
    if (huge_page_size == 0) {
        f = fopen("/proc/meminfo", "r");
        if (f != NULL) {
            while (fgets(buf, sizeof(buf), f)) {
                if (sscanf(buf, "Hugepagesize: %d kB", &size_kb) == 1) {
                    huge_page_size = size_kb * 1024L;
                    break;
                }
            }
            fclose(f);
        }

        if (huge_page_size == 0) {
            huge_page_size = 2 * 1024L *1024L;
        }
    }

    return huge_page_size;
}


