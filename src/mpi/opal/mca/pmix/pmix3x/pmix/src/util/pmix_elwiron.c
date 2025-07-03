/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2007-2013 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2014-2019 Intel, Inc.  All rights reserved.
 * Copyright (c) 2016      IBM Corporation.  All rights reserved.
 * Copyright (c) 2019      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <src/include/pmix_config.h>

#include <pmix_common.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/util/printf.h"
#include "src/util/error.h"
#include "src/util/argv.h"
#include "src/util/pmix_elwiron.h"

#define PMIX_DEFAULT_TMPDIR "/tmp"
#define PMIX_MAX_ELWAR_LENGTH   100000

/*
 * Merge two elwiron-like char arrays, ensuring that there are no
 * duplicate entires
 */
char **pmix_elwiron_merge(char **minor, char **major)
{
    int i;
    char **ret = NULL;
    char *name, *value;

    /* Check for bozo cases */

    if (NULL == major) {
        if (NULL == minor) {
            return NULL;
        } else {
            return pmix_argv_copy(minor);
        }
    }

    /* First, copy major */

    ret = pmix_argv_copy(major);

    /* Do we have something in minor? */

    if (NULL == minor) {
        return ret;
    }

    /* Now go through minor and call pmix_setelw(), but with overwrite
       as false */

    for (i = 0; NULL != minor[i]; ++i) {
        value = strchr(minor[i], '=');
        if (NULL == value) {
            pmix_setelw(minor[i], NULL, false, &ret);
        } else {

            /* strdup minor[i] in case it's a constant string */

            name = strdup(minor[i]);
            value = name + (value - minor[i]);
            *value = '\0';
            pmix_setelw(name, value + 1, false, &ret);
            free(name);
        }
    }

    /* All done */

    return ret;
}

/*
 * Portable version of setelw(), allowing editing of any elwiron-like
 * array
 */
 pmix_status_t pmix_setelw(const char *name, const char *value, bool overwrite,
                char ***elw)
{
    int i;
    char *newvalue, *compare;
    size_t len;
    bool valid;

    /* Check the bozo case */
    if( NULL == elw ) {
        return PMIX_ERR_BAD_PARAM;
    }

    if (NULL != value) {
        /* check the string for unacceptable length - i.e., ensure
         * it is NULL-terminated */
        valid = false;
        for (i=0; i < PMIX_MAX_ELWAR_LENGTH; i++) {
            if ('\0' == value[i]) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            PMIX_ERROR_LOG(PMIX_ERR_BAD_PARAM);
            return PMIX_ERR_BAD_PARAM;
        }
    }

    /* If this is the "elwiron" array, use putelw or setelw */
    if (*elw == elwiron) {
        /* THIS IS POTENTIALLY A MEMORY LEAK!  But I am doing it
           because so that we don't violate the law of least
           astonishmet for PMIX developers (i.e., those that don't
           check the return code of pmix_setelw() and notice that we
           returned an error if you passed in the real elwiron) */
#if defined (HAVE_SETELW)
        if (NULL == value) {
            /* this is actually an unsetelw request */
            unsetelw(name);
        } else {
            setelw(name, value, overwrite);
        }
#else
        /* Make the new value */
        if (NULL == value) {
            i = asprintf(&newvalue, "%s=", name);
        } else {
            i = asprintf(&newvalue, "%s=%s", name, value);
        }
        if (NULL == newvalue || 0 > i) {
            return PMIX_ERR_OUT_OF_RESOURCE;
        }
        putelw(newvalue);
        /* cannot free it as putelw doesn't copy the value */
#endif
        return PMIX_SUCCESS;
    }

    /* Make the new value */
    if (NULL == value) {
        i = asprintf(&newvalue, "%s=", name);
    } else {
        i = asprintf(&newvalue, "%s=%s", name, value);
    }
    if (NULL == newvalue || 0 > i) {
        return PMIX_ERR_OUT_OF_RESOURCE;
    }

    if (NULL == *elw) {
        i = 0;
        pmix_argv_append(&i, elw, newvalue);
        free(newvalue);
        return PMIX_SUCCESS;
    }

    /* Make something easy to compare to */

    i = asprintf(&compare, "%s=", name);
    if (NULL == compare || 0 > i) {
        free(newvalue);
        return PMIX_ERR_OUT_OF_RESOURCE;
    }
    len = strlen(compare);

    /* Look for a duplicate that's already set in the elw */

    for (i = 0; (*elw)[i] != NULL; ++i) {
        if (0 == strncmp((*elw)[i], compare, len)) {
            if (overwrite) {
                free((*elw)[i]);
                (*elw)[i] = newvalue;
                free(compare);
                return PMIX_SUCCESS;
            } else {
                free(compare);
                free(newvalue);
                return PMIX_EXISTS;
            }
        }
    }

    /* If we found no match, append this value */

    i = pmix_argv_count(*elw);
    pmix_argv_append(&i, elw, newvalue);

    /* All done */

    free(compare);
    free(newvalue);
    return PMIX_SUCCESS;
}


/*
 * Portable version of unsetelw(), allowing editing of any
 * elwiron-like array
 */
 pmix_status_t pmix_unsetelw(const char *name, char ***elw)
{
    int i;
    char *compare;
    size_t len;
    bool found;

    /* Check for bozo case */

    if (NULL == *elw) {
        return PMIX_SUCCESS;
    }

    /* Make something easy to compare to */

    i = asprintf(&compare, "%s=", name);
    if (NULL == compare || 0 > i) {
        return PMIX_ERR_OUT_OF_RESOURCE;
    }
    len = strlen(compare);

    /* Look for a duplicate that's already set in the elw.  If we find
       it, free it, and then start shifting all elements down one in
       the array. */

    found = false;
    for (i = 0; (*elw)[i] != NULL; ++i) {
        if (0 != strncmp((*elw)[i], compare, len))
            continue;
        if (elwiron != *elw) {
            free((*elw)[i]);
        }
        for (; (*elw)[i] != NULL; ++i)
            (*elw)[i] = (*elw)[i + 1];
        found = true;
        break;
    }
    free(compare);

    /* All done */

    return (found) ? PMIX_SUCCESS : PMIX_ERR_NOT_FOUND;
}

const char* pmix_tmp_directory( void )
{
    const char* str;

    if( NULL == (str = getelw("TMPDIR")) )
        if( NULL == (str = getelw("TEMP")) )
            if( NULL == (str = getelw("TMP")) )
                str = PMIX_DEFAULT_TMPDIR;
    return str;
}

const char* pmix_home_directory( void )
{
    char* home = getelw("HOME");

    return home;
}
