/*
 * Copyright (c) 2011-2014 LWPU Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef _OPAL_DATATYPE_LWDA_H
#define _OPAL_DATATYPE_LWDA_H

/* Structure to hold LWCA support functions that gets filled in when the
 * common lwca code is initialized.  This removes any dependency on <lwca.h>
 * in the opal lwca datatype code. */
struct opal_common_lwda_function_table {
    int (*gpu_is_gpu_buffer)(const void*, opal_colwertor_t*);
    int (*gpu_lw_memcpy_async)(void*, const void*, size_t, opal_colwertor_t*);
    int (*gpu_lw_memcpy)(void*, const void*, size_t);
    int (*gpu_memmove)(void*, void*, size_t);
};
typedef struct opal_common_lwda_function_table opal_common_lwda_function_table_t;

void mca_lwda_colwertor_init(opal_colwertor_t* colwertor, const void *pUserBuf);
bool opal_lwda_check_bufs(char *dest, char *src);
bool opal_lwda_check_one_buf(char *buf, opal_colwertor_t *colwertor );
void* opal_lwda_memcpy(void * dest, const void * src, size_t size, opal_colwertor_t* colwertor);
void* opal_lwda_memcpy_sync(void * dest, const void * src, size_t size);
void* opal_lwda_memmove(void * dest, void * src, size_t size);
void opal_lwda_add_initialization_function(int (*fptr)(opal_common_lwda_function_table_t *));
void opal_lwda_set_copy_function_async(opal_colwertor_t* colwertor, void *stream);

#endif
