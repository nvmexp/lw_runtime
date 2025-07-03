/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2014      LWPU Corporation.  All rights reserved.
 * Copyright (c) 2017-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2017      Intel, Inc. All rights reserved
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_COLWERTOR_H_HAS_BEEN_INCLUDED
#define OPAL_COLWERTOR_H_HAS_BEEN_INCLUDED

#include "opal_config.h"

#ifdef HAVE_SYS_UIO_H
#include <sys/uio.h>
#endif

#include "opal/constants.h"
#include "opal/datatype/opal_datatype.h"
#include "opal/prefetch.h"

BEGIN_C_DECLS
/*
 * COLWERTOR SECTION
 */
/* keep the last 16 bits free for data flags */
#define COLWERTOR_DATATYPE_MASK    0x0000FFFF
#define COLWERTOR_SEND_COLWERSION  0x00010000
#define COLWERTOR_RECV             0x00020000
#define COLWERTOR_SEND             0x00040000
#define COLWERTOR_HOMOGENEOUS      0x00080000
#define COLWERTOR_NO_OP            0x00100000
#define COLWERTOR_WITH_CHECKSUM    0x00200000
#define COLWERTOR_LWDA             0x00400000
#define COLWERTOR_LWDA_ASYNC       0x00800000
#define COLWERTOR_TYPE_MASK        0x10FF0000
#define COLWERTOR_STATE_START      0x01000000
#define COLWERTOR_STATE_COMPLETE   0x02000000
#define COLWERTOR_STATE_ALLOC      0x04000000
#define COLWERTOR_COMPLETED        0x08000000
#define COLWERTOR_LWDA_UNIFIED     0x10000000
#define COLWERTOR_HAS_REMOTE_SIZE  0x20000000
#define COLWERTOR_SKIP_LWDA_INIT   0x40000000

union dt_elem_desc;
typedef struct opal_colwertor_t opal_colwertor_t;

typedef int32_t (*colwertor_advance_fct_t)( opal_colwertor_t* pColwertor,
                                            struct iovec* iov,
                                            uint32_t* out_size,
                                            size_t* max_data );
typedef void*(*memalloc_fct_t)( size_t* pLength, void* userdata );
typedef void*(*memcpy_fct_t)( void* dest, const void* src, size_t n, opal_colwertor_t* pColwertor );

/* The master colwertor struct (defined in colwertor_internal.h) */
struct opal_colwertor_master_t;

struct dt_stack_t {
    int32_t           index;    /**< index in the element description */
    int16_t           type;     /**< the type used for the last pack/unpack (original or OPAL_DATATYPE_UINT1) */
    int16_t           padding;
    size_t            count;    /**< number of times we still have to do it */
    ptrdiff_t         disp;     /**< actual displacement depending on the count field */
};
typedef struct dt_stack_t dt_stack_t;

/**
 *
 */
#define DT_STATIC_STACK_SIZE   5                /**< This should be sufficient for most applications */

struct opal_colwertor_t {
    opal_object_t                 super;          /**< basic superclass */
    uint32_t                      remoteArch;     /**< the remote architecture */
    uint32_t                      flags;          /**< the properties of this colwertor */
    size_t                        local_size;     /**< overall length data on local machine, compared to bColwerted */
    size_t                        remote_size;    /**< overall length data on remote machine, compared to bColwerted */
    const opal_datatype_t*        pDesc;          /**< the datatype description associated with the colwertor */
    const dt_type_desc_t*         use_desc;       /**< the version used by the colwertor (normal or optimized) */
    opal_datatype_count_t         count;          /**< the total number of full datatype elements */

    /* --- cacheline boundary (64 bytes - if 64bits arch and !OPAL_ENABLE_DEBUG) --- */
    uint32_t                      stack_size;     /**< size of the allocated stack */
    unsigned char*                pBaseBuf;       /**< initial buffer as supplied by the user */
    dt_stack_t*                   pStack;         /**< the local stack for the actual colwersion */
    colwertor_advance_fct_t       fAdvance;       /**< pointer to the pack/unpack functions */

    /* --- cacheline boundary (96 bytes - if 64bits arch and !OPAL_ENABLE_DEBUG) --- */
    struct opal_colwertor_master_t* master;       /**< the master colwertor */

    /* All others fields get modified for every call to pack/unpack functions */
    uint32_t                      stack_pos;      /**< the actual position on the stack */
    size_t                        partial_length; /**< amount of data left over from the last unpack */
    size_t                        bColwerted;     /**< # of bytes already colwerted */

    /* --- cacheline boundary (128 bytes - if 64bits arch and !OPAL_ENABLE_DEBUG) --- */
    uint32_t                      checksum;       /**< checksum computed by pack/unpack operation */
    uint32_t                      csum_ui1;       /**< partial checksum computed by pack/unpack operation */
    size_t                        csum_ui2;       /**< partial checksum computed by pack/unpack operation */

    /* --- fields are no more aligned on cacheline --- */
    dt_stack_t                    static_stack[DT_STATIC_STACK_SIZE];  /**< local stack for small datatypes */

#if OPAL_LWDA_SUPPORT
    memcpy_fct_t                  cbmemcpy;       /**< memcpy or lwMemcpy */
    void *                        stream;         /**< LWstream for async copy */
#endif
};
OPAL_DECLSPEC OBJ_CLASS_DECLARATION( opal_colwertor_t );


/*
 *
 */
static inline uint32_t opal_colwertor_get_checksum( opal_colwertor_t* colwertor )
{
    return colwertor->checksum;
}


/*
 *
 */
OPAL_DECLSPEC int32_t opal_colwertor_pack( opal_colwertor_t* pColw, struct iovec* iov,
                                           uint32_t* out_size, size_t* max_data );

/*
 *
 */
OPAL_DECLSPEC int32_t opal_colwertor_unpack( opal_colwertor_t* pColw, struct iovec* iov,
                                             uint32_t* out_size, size_t* max_data );

/*
 *
 */
OPAL_DECLSPEC opal_colwertor_t* opal_colwertor_create( int32_t remote_arch, int32_t mode );


/**
 * The cleanup function will put the colwertor in exactly the same state as after a call
 * to opal_colwertor_construct. Therefore, all PML can call OBJ_DESTRUCT on the request's
 * colwertors without having to call OBJ_CONSTRUCT everytime they grab a new one from the
 * cache. The OBJ_CONSTRUCT on the colwertor should be called only on the first creation
 * of a request (not when extracted from the cache).
 */
static inline int opal_colwertor_cleanup( opal_colwertor_t* colwertor )
{
    if( OPAL_UNLIKELY(colwertor->stack_size > DT_STATIC_STACK_SIZE) ) {
        free( colwertor->pStack );
        colwertor->pStack     = colwertor->static_stack;
        colwertor->stack_size = DT_STATIC_STACK_SIZE;
    }
    colwertor->pDesc     = NULL;
    colwertor->stack_pos = 0;
    colwertor->flags     = OPAL_DATATYPE_FLAG_NO_GAPS | COLWERTOR_COMPLETED;

    return OPAL_SUCCESS;
}


/**
 * Return:   0 if no packing is required for sending (the upper layer
 *             can use directly the pointer to the contiguous user
 *             buffer).
 *           1 if data does need to be packed, i.e. heterogeneous peers
 *             (source arch != dest arch) or non contiguous memory
 *             layout.
 */
static inline int32_t opal_colwertor_need_buffers( const opal_colwertor_t* pColwertor )
{
    if (OPAL_UNLIKELY(0 == (pColwertor->flags & COLWERTOR_HOMOGENEOUS))) return 1;
#if OPAL_LWDA_SUPPORT
    if( pColwertor->flags & (COLWERTOR_LWDA | COLWERTOR_LWDA_UNIFIED)) return 1;
#endif
    if( pColwertor->flags & OPAL_DATATYPE_FLAG_NO_GAPS ) return 0;
    if( (pColwertor->count == 1) && (pColwertor->flags & OPAL_DATATYPE_FLAG_CONTIGUOUS) ) return 0;
    return 1;
}

/**
 * Update the size of the remote datatype representation. The size will
 * depend on the configuration of the master colwertor. In homogeneous
 * elwironments, the local and remote sizes are identical.
 */
size_t
opal_colwertor_compute_remote_size( opal_colwertor_t* pColw );

/**
 * Return the local size of the colwertor (count times the size of the datatype).
 */
static inline void opal_colwertor_get_packed_size( const opal_colwertor_t* pColw,
                                                   size_t* pSize )
{
    *pSize = pColw->local_size;
}


/**
 * Return the remote size of the colwertor (count times the remote size of the
 * datatype). On homogeneous elwironments the local and remote sizes are
 * identical.
 */
static inline void opal_colwertor_get_unpacked_size( const opal_colwertor_t* pColw,
                                                     size_t* pSize )
{
    if( pColw->flags & COLWERTOR_HOMOGENEOUS ) {
        *pSize = pColw->local_size;
        return;
    }
    if( 0 == (COLWERTOR_HAS_REMOTE_SIZE & pColw->flags) ) {
        assert(! (pColw->flags & COLWERTOR_SEND));
        opal_colwertor_compute_remote_size( (opal_colwertor_t*)pColw);
    }
    *pSize = pColw->remote_size;
}

/**
 * Return the current absolute position of the next pack/unpack. This function is
 * mostly useful for contiguous datatypes, when we need to get the pointer to the
 * contiguous piece of memory.
 */
static inline void opal_colwertor_get_lwrrent_pointer( const opal_colwertor_t* pColw,
                                                       void** position )
{
    unsigned char* base = pColw->pBaseBuf + pColw->bColwerted + pColw->pDesc->true_lb;
    *position = (void*)base;
}

static inline void opal_colwertor_get_offset_pointer( const opal_colwertor_t* pColw,
                                                      size_t offset, void** position )
{
    unsigned char* base = pColw->pBaseBuf + offset + pColw->pDesc->true_lb;
    *position = (void*)base;
}


/*
 *
 */
OPAL_DECLSPEC int32_t opal_colwertor_prepare_for_send( opal_colwertor_t* colwertor,
                                                       const struct opal_datatype_t* datatype,
                                                       size_t count,
                                                       const void* pUserBuf);

static inline int32_t opal_colwertor_copy_and_prepare_for_send( const opal_colwertor_t* pSrcColw,
                                                                const struct opal_datatype_t* datatype,
                                                                size_t count,
                                                                const void* pUserBuf,
                                                                int32_t flags,
                                                                opal_colwertor_t* colwertor )
{
    colwertor->remoteArch = pSrcColw->remoteArch;
    colwertor->flags      = pSrcColw->flags | flags;
    colwertor->master     = pSrcColw->master;

    return opal_colwertor_prepare_for_send( colwertor, datatype, count, pUserBuf );
}

/*
 *
 */
OPAL_DECLSPEC int32_t opal_colwertor_prepare_for_recv( opal_colwertor_t* colwertor,
                                                       const struct opal_datatype_t* datatype,
                                                       size_t count,
                                                       const void* pUserBuf );
static inline int32_t opal_colwertor_copy_and_prepare_for_recv( const opal_colwertor_t* pSrcColw,
                                                                const struct opal_datatype_t* datatype,
                                                                size_t count,
                                                                const void* pUserBuf,
                                                                int32_t flags,
                                                                opal_colwertor_t* colwertor )
{
    colwertor->remoteArch = pSrcColw->remoteArch;
    colwertor->flags      = (pSrcColw->flags | flags);
    colwertor->master     = pSrcColw->master;

    return opal_colwertor_prepare_for_recv( colwertor, datatype, count, pUserBuf );
}

/*
 * Give access to the raw memory layout based on the datatype.
 */
OPAL_DECLSPEC int32_t
opal_colwertor_raw( opal_colwertor_t* colwertor,  /* [IN/OUT] */
                    struct iovec* iov,            /* [IN/OUT] */
                    uint32_t* iov_count,          /* [IN/OUT] */
                    size_t* length );             /* [OUT]    */


/*
 * Upper level does not need to call the _nocheck function directly.
 */
OPAL_DECLSPEC int32_t
opal_colwertor_set_position_nocheck( opal_colwertor_t* colwertor,
                                     size_t* position );
static inline int32_t
opal_colwertor_set_position( opal_colwertor_t* colwertor,
                             size_t* position )
{
    /*
     * Do not allow the colwertor to go outside the data boundaries. This test include
     * the check for datatype with size zero as well as for colwertors with a count of zero.
     */
    if( OPAL_UNLIKELY(colwertor->local_size <= *position) ) {
        colwertor->flags |= COLWERTOR_COMPLETED;
        colwertor->bColwerted = colwertor->local_size;
        *position = colwertor->bColwerted;
        return OPAL_SUCCESS;
    }

    /*
     * If the colwertor is already at the correct position we are happy.
     */
    if( OPAL_LIKELY((*position) == colwertor->bColwerted) ) return OPAL_SUCCESS;

    /* Remove the completed flag if it's already set */
    colwertor->flags &= ~COLWERTOR_COMPLETED;

    if( (colwertor->flags & OPAL_DATATYPE_FLAG_NO_GAPS) &&
#if defined(CHECKSUM)
        !(colwertor->flags & COLWERTOR_WITH_CHECKSUM) &&
#endif  /* defined(CHECKSUM) */
        (colwertor->flags & (COLWERTOR_SEND | COLWERTOR_HOMOGENEOUS)) ) {
        /* Contiguous and no checkpoint and no homogeneous unpack */
        colwertor->bColwerted = *position;
        return OPAL_SUCCESS;
    }

    return opal_colwertor_set_position_nocheck( colwertor, position );
}

/*
 *
 */
static inline int32_t
opal_colwertor_personalize( opal_colwertor_t* colwertor,
                            uint32_t flags,
                            size_t* position )
{
    colwertor->flags |= flags;

    if( OPAL_UNLIKELY(NULL == position) )
        return OPAL_SUCCESS;
    return opal_colwertor_set_position( colwertor, position );
}

/*
 *
 */
OPAL_DECLSPEC int
opal_colwertor_clone( const opal_colwertor_t* source,
                      opal_colwertor_t* destination,
                      int32_t copy_stack );

static inline int
opal_colwertor_clone_with_position( const opal_colwertor_t* source,
                                    opal_colwertor_t* destination,
                                    int32_t copy_stack,
                                    size_t* position )
{
    (void)opal_colwertor_clone( source, destination, copy_stack );
    return opal_colwertor_set_position( destination, position );
}

/*
 *
 */
OPAL_DECLSPEC void
opal_colwertor_dump( opal_colwertor_t* colwertor );

OPAL_DECLSPEC void
opal_datatype_dump_stack( const dt_stack_t* pStack,
                          int stack_pos,
                          const union dt_elem_desc* pDesc,
                          const char* name );

/*
 *
 */
OPAL_DECLSPEC int
opal_colwertor_generic_simple_position( opal_colwertor_t* pColwertor,
                                        size_t* position );

END_C_DECLS

#endif  /* OPAL_COLWERTOR_H_HAS_BEEN_INCLUDED */
