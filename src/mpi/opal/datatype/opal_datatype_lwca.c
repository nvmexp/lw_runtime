/*
 * Copyright (c) 2011-2014 LWPU Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "opal/align.h"
#include "opal/util/output.h"
#include "opal/datatype/opal_colwertor.h"
#include "opal/datatype/opal_datatype_lwda.h"

static bool initialized = false;
int opal_lwda_verbose = 0;
static int opal_lwda_enabled = 0; /* Starts out disabled */
static int opal_lwda_output = 0;
static void opal_lwda_support_init(void);
static int (*common_lwda_initialization_function)(opal_common_lwda_function_table_t *) = NULL;
static opal_common_lwda_function_table_t ftable;

/* This function allows the common lwca code to register an
 * initialization function that gets called the first time an attempt
 * is made to send or receive a GPU pointer.  This allows us to delay
 * some LWCA initialization until after MPI_Init().
 */
void opal_lwda_add_initialization_function(int (*fptr)(opal_common_lwda_function_table_t *)) {
    common_lwda_initialization_function = fptr;
}

/**
 * This function is called when a colwertor is instantiated.  It has to call
 * the opal_lwda_support_init() function once to figure out if LWCA support
 * is enabled or not.  If LWCA is not enabled, then short circuit out
 * for all future calls.
 */
void mca_lwda_colwertor_init(opal_colwertor_t* colwertor, const void *pUserBuf)
{
    /* Only do the initialization on the first GPU access */
    if (!initialized) {
        opal_lwda_support_init();
    }

    /* This is needed to handle case where colwertor is not fully initialized
     * like when trying to do a sendi with colwertor on the statck */
    colwertor->cbmemcpy = (memcpy_fct_t)&opal_lwda_memcpy;

    /* If not enabled, then nothing else to do */
    if (!opal_lwda_enabled) {
        return;
    }

    if (ftable.gpu_is_gpu_buffer(pUserBuf, colwertor)) {
        colwertor->flags |= COLWERTOR_LWDA;
    }
}

/* Checks the type of pointer
 *
 * @param dest   One pointer to check
 * @param source Another pointer to check
 */
bool opal_lwda_check_bufs(char *dest, char *src)
{
    /* Only do the initialization on the first GPU access */
    if (!initialized) {
        opal_lwda_support_init();
    }

    if (!opal_lwda_enabled) {
        return false;
    }

    if (ftable.gpu_is_gpu_buffer(dest, NULL) || ftable.gpu_is_gpu_buffer(src, NULL)) {
        return true;
    } else {
        return false;
    }
}

/*
 * With LWCA enabled, all contiguous copies will pass through this function.
 * Therefore, the first check is to see if the colwertor is a GPU buffer.
 * Note that if there is an error with any of the LWCA calls, the program
 * aborts as there is no recovering.
 */

/* Checks the type of pointer
 *
 * @param buf   check one pointer providing a colwertor.
 *  Provides aditional information, e.g. managed vs. unmanaged GPU buffer
 */
bool  opal_lwda_check_one_buf(char *buf, opal_colwertor_t *colwertor )
{
    /* Only do the initialization on the first GPU access */
    if (!initialized) {
        opal_lwda_support_init();
    }

    if (!opal_lwda_enabled) {
        return false;
    }

    return ( ftable.gpu_is_gpu_buffer(buf, colwertor));
}

/*
 * With LWCA enabled, all contiguous copies will pass through this function.
 * Therefore, the first check is to see if the colwertor is a GPU buffer.
 * Note that if there is an error with any of the LWCA calls, the program
 * aborts as there is no recovering.
 */

void *opal_lwda_memcpy(void *dest, const void *src, size_t size, opal_colwertor_t* colwertor)
{
    int res;

    if (!(colwertor->flags & COLWERTOR_LWDA)) {
        return memcpy(dest, src, size);
    }

    if (colwertor->flags & COLWERTOR_LWDA_ASYNC) {
        res = ftable.gpu_lw_memcpy_async(dest, (void *)src, size, colwertor);
    } else {
        res = ftable.gpu_lw_memcpy(dest, (void *)src, size);
    }

    if (res != 0) {
        opal_output(0, "LWCA: Error in lwMemcpy: res=%d, dest=%p, src=%p, size=%d",
                    res, dest, src, (int)size);
        abort();
    } else {
        return dest;
    }
}

/*
 * This function is needed in cases where we do not have contiguous
 * datatypes.  The current code has macros that cannot handle a colwertor
 * argument to the memcpy call.
 */
void *opal_lwda_memcpy_sync(void *dest, const void *src, size_t size)
{
    int res;
    res = ftable.gpu_lw_memcpy(dest, src, size);
    if (res != 0) {
        opal_output(0, "LWCA: Error in lwMemcpy: res=%d, dest=%p, src=%p, size=%d",
                    res, dest, src, (int)size);
        abort();
    } else {
        return dest;
    }
}

/*
 * In some cases, need an implementation of memmove.  This is not fast, but
 * it is not often needed.
 */
void *opal_lwda_memmove(void *dest, void *src, size_t size)
{
    int res;

    res = ftable.gpu_memmove(dest, src, size);
    if(res != 0){
        opal_output(0, "LWCA: Error in gpu memmove: res=%d, dest=%p, src=%p, size=%d",
                    res, dest, src, (int)size);
        abort();
    }
    return dest;
}

/**
 * This function gets called once to check if the program is running in a lwca
 * environment.
 */
static void opal_lwda_support_init(void)
{
    if (initialized) {
        return;
    }

    /* Set different levels of verbosity in the lwca related code. */
    opal_lwda_output = opal_output_open(NULL);
    opal_output_set_verbosity(opal_lwda_output, opal_lwda_verbose);

    /* Callback into the common lwca initialization routine. This is only
     * set if some work had been done already in the common lwca code.*/
    if (NULL != common_lwda_initialization_function) {
        if (0 == common_lwda_initialization_function(&ftable)) {
            opal_lwda_enabled = 1;
        }
    }

    if (1 == opal_lwda_enabled) {
        opal_output_verbose(10, opal_lwda_output,
                            "LWCA: enabled successfully, LWCA device pointers will work");
    } else {
        opal_output_verbose(10, opal_lwda_output,
                            "LWCA: not enabled, LWCA device pointers will not work");
    }

    initialized = true;
}

/**
 * Tell the colwertor that copies will be asynchronous LWCA copies.  The
 * flags are cleared when the colwertor is reinitialized.
 */
void opal_lwda_set_copy_function_async(opal_colwertor_t* colwertor, void *stream)
{
    colwertor->flags |= COLWERTOR_LWDA_ASYNC;
    colwertor->stream = stream;
}
