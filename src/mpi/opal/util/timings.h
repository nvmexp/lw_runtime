/*
 * Copyright (C) 2014      Artem Polyakov <artpol84@gmail.com>
 * Copyright (c) 2014-2017 Intel, Inc. All rights reserved.
 * Copyright (c) 2017-2018 Mellanox Technologies Ltd. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_UTIL_TIMING_H
#define OPAL_UTIL_TIMING_H

#include "opal_config.h"

#include "opal/class/opal_list.h"
#include "opal/runtime/opal_params.h"

typedef enum {
    OPAL_TIMING_AUTOMATIC_TIMER,
    OPAL_TIMING_GET_TIME_OF_DAY,
    OPAL_TIMING_CYCLE_NATIVE,
    OPAL_TIMING_USEC_NATIVE
} opal_timer_type_t;

#if OPAL_ENABLE_TIMING

typedef double (*opal_timing_ts_func_t)(void);

#define OPAL_TIMING_STR_LEN 256

typedef struct {
    char id[OPAL_TIMING_STR_LEN], cntr_elw[OPAL_TIMING_STR_LEN];
    int enabled, error;
    int cntr;
    double ts;
    opal_timing_ts_func_t get_ts;
} opal_timing_elw_t;

opal_timing_ts_func_t opal_timing_ts_func(opal_timer_type_t type);

#define OPAL_TIMING_ELW_START_TYPE(func, _nm, type, prefix)                       \
    do {                                                                          \
        char *ptr = NULL;                                                         \
        char *_prefix = prefix;                                                   \
        int n;                                                                    \
        if( NULL == prefix ){                                                     \
            _prefix = "";                                                         \
        }                                                                         \
        (_nm)->error = 0;                                                         \
        n = snprintf((_nm)->id, OPAL_TIMING_STR_LEN, "%s_%s", _prefix, func);     \
        if( n > OPAL_TIMING_STR_LEN ){                                            \
             (_nm)->error = 1;                                                    \
        }                                                                         \
        n = sprintf((_nm)->cntr_elw,"OMPI_TIMING_%s_CNT", (_nm)->id);             \
        if( n > OPAL_TIMING_STR_LEN ){                                            \
            (_nm)->error = 1;                                                     \
        }                                                                         \
        ptr = getelw((_nm)->id);                                                  \
        if( NULL == ptr || strcmp(ptr, "1")){                                     \
            (_nm)->enabled = 0;                                                   \
        }                                                                         \
        (_nm)->get_ts = opal_timing_ts_func(type);                                \
        ptr = getelw("OPAL_TIMING_ENABLE");                                       \
        if (NULL != ptr) {                                                        \
            (_nm)->enabled = atoi(ptr);                                           \
        }                                                                         \
        (_nm)->cntr = 0;                                                          \
        ptr = getelw((_nm)->id);                                                  \
        if( NULL != ptr ){                                                        \
            (_nm)->cntr = atoi(ptr);                                              \
        }                                                                         \
        (_nm)->ts = (_nm)->get_ts();                                              \
        if ( 0 != (_nm)->error ){                                                 \
            (_nm)->enabled = 0;                                                   \
        }                                                                         \
    } while(0)

/* We use function names for identification
 * however this might be a problem for the private
 * functions declared as static as their names may
 * conflict.
 * Use prefix to do a finer-grained identification if needed
 */
#define OPAL_TIMING_ELW_INIT_PREFIX(prefix, name)                                 \
    opal_timing_elw_t name ## _val, *name = &(name ## _val);                      \
    OPAL_TIMING_ELW_START_TYPE(__func__, name, OPAL_TIMING_AUTOMATIC_TIMER, prefix);

#define OPAL_TIMING_ELW_INIT(name) OPAL_TIMING_ELW_INIT_PREFIX("", name)

#define OPAL_TIMING_ELW_NEXT(h, ...)                                              \
    do {                                                                          \
        int n;                                                                    \
        char buf1[OPAL_TIMING_STR_LEN], buf2[OPAL_TIMING_STR_LEN];                \
        double time;                                                              \
        char *filename;                                                           \
        if( h->enabled ){                                                         \
            /* enabled codepath */                                                \
            time = h->get_ts() - h->ts;                                           \
            n = snprintf(buf1, OPAL_TIMING_STR_LEN, "OMPI_TIMING_%s_DESC_%d", h->id, h->cntr); \
            if ( n > OPAL_TIMING_STR_LEN ){                                       \
                h->error = 1;                                                     \
            }                                                                     \
            n = snprintf(buf2, OPAL_TIMING_STR_LEN, __VA_ARGS__ );                \
            if ( n > OPAL_TIMING_STR_LEN ){                                       \
                h->error = 1;                                                     \
            }                                                                     \
            setelw(buf1, buf2, 1);                                                \
            n = snprintf(buf1, OPAL_TIMING_STR_LEN, "OMPI_TIMING_%s_VAL_%d", h->id, h->cntr);  \
            if ( n > OPAL_TIMING_STR_LEN ){                                       \
                h->error = 1;                                                     \
            }                                                                     \
            n = snprintf(buf2, OPAL_TIMING_STR_LEN, "%lf", time);                 \
            if ( n > OPAL_TIMING_STR_LEN ){                                       \
                h->error = 1;                                                     \
            }                                                                     \
            setelw(buf1, buf2, 1);                                                \
            filename = strrchr(__FILE__, '/');                                    \
            filename = (filename == NULL) ? strdup(__FILE__) : filename+1;        \
            n = snprintf(buf1, OPAL_TIMING_STR_LEN, "OMPI_TIMING_%s_FILE_%d", h->id, h->cntr); \
            if ( n > OPAL_TIMING_STR_LEN ){                                       \
                h->error = 1;                                                     \
            }                                                                     \
            n = snprintf(buf2, OPAL_TIMING_STR_LEN, "%s", filename);              \
            if ( n > OPAL_TIMING_STR_LEN ){                                       \
                h->error = 1;                                                     \
            }                                                                     \
            setelw(buf1, buf2, 1);                                                \
            h->cntr++;                                                            \
            sprintf(buf1, "%d", h->cntr);                                         \
            setelw(h->cntr_elw, buf1, 1);                                         \
            /* We don't include elw operations into the consideration.
             * Hopefully this will help to make measurements more accurate.
             */                                                                   \
            h->ts = h->get_ts();                                                  \
        }                                                                         \
        if (h->error) {                                                           \
            n = snprintf(buf1, OPAL_TIMING_STR_LEN, "OMPI_TIMING_%s_ERROR", h->id);\
            if ( n > OPAL_TIMING_STR_LEN ){                                       \
                h->error = 1;                                                     \
            }                                                                     \
            n = snprintf(buf2, OPAL_TIMING_STR_LEN, "%d", h->error);              \
            if ( n > OPAL_TIMING_STR_LEN ){                                       \
                h->error = 1;                                                     \
            }                                                                     \
            setelw(buf1, buf2, 1);                                                \
        }                                                                         \
    } while(0)

/* This function supposed to be called from the code that will
 * do the postprocessing, i.e. OMPI timing portion that will
 * do the reduction of aclwmulated values
 */
#define OPAL_TIMING_ELW_CNT_PREFIX(prefix, func, _cnt)                            \
    do {                                                                          \
        char ename[OPAL_TIMING_STR_LEN];                                          \
        char *ptr = NULL;                                                         \
        int n = snprintf(ename, OPAL_TIMING_STR_LEN, "OMPI_TIMING_%s_%s_CNT", prefix, func);    \
        (_cnt) = 0;                                                               \
        if ( n <= OPAL_TIMING_STR_LEN ){                                          \
            ptr = getelw(ename);                                                  \
            if( NULL != ptr ){ (_cnt) = atoi(ptr); };                             \
        }                                                                         \
    } while(0)

#define OPAL_TIMING_ELW_ERROR_PREFIX(prefix, func, _err)                          \
    do {                                                                          \
        char ename[OPAL_TIMING_STR_LEN];                                          \
        (_err) = 0;                                                               \
        char *ptr = NULL;                                                         \
        int n = snprintf(ename, OPAL_TIMING_STR_LEN, "OMPI_TIMING_%s%s_ERROR", prefix, func);    \
        if ( n <= OPAL_TIMING_STR_LEN ){                                          \
            ptr = getelw(ename);                                                  \
            if( NULL != ptr ){ (_err) = atoi(ptr); };                             \
        }                                                                         \
    } while(0)

#define OPAL_TIMING_ELW_GETDESC_PREFIX(prefix, filename, func, i, desc, _t)       \
    do {                                                                          \
        char vname[OPAL_TIMING_STR_LEN];                                          \
        (_t) = 0.0;                                                               \
        sprintf(vname, "OMPI_TIMING_%s_%s_FILE_%d", prefix, func, i);              \
        *filename = getelw(vname);                                                \
        sprintf(vname, "OMPI_TIMING_%s_%s_DESC_%d", prefix, func, i);              \
        *desc = getelw(vname);                                                    \
        sprintf(vname, "OMPI_TIMING_%s_%s_VAL_%d", prefix, func, i);               \
        char *ptr = getelw(vname);                                                \
        if ( NULL != ptr ) {                                                      \
            sscanf(ptr,"%lf", &(_t));                                             \
        }                                                                         \
    } while(0)

#define OPAL_TIMING_ELW_GETDESC(file, func, index, desc)                          \
    OPAL_TIMING_ELW_GETDESC_PREFIX("", file, func, index, desc)

#else

#define OPAL_TIMING_ELW_START_TYPE(func, type, prefix)

#define OPAL_TIMING_ELW_INIT(name)

#define OPAL_TIMING_ELW_INIT_PREFIX(prefix, name)

#define OPAL_TIMING_ELW_NEXT(h, ... )

#define OPAL_TIMING_ELW_CNT_PREFIX(prefix, func)

#define OPAL_TIMING_ELW_CNT(func)

#define OPAL_TIMING_ELW_GETDESC_PREFIX(prefix, func, i, desc)

#define OPAL_TIMING_ELW_GETDESC(func, index, desc)

#define OPAL_TIMING_ELW_ERROR_PREFIX(prefix, func)

#endif

#endif
