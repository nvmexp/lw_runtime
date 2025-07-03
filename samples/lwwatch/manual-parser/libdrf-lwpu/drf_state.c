/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifndef DRF_WINDOWS
#include <unistd.h>
#endif

#include <drf_types.h>
#include "drf.h"
#include "drf_state.h"

int drf_errno;

int drf_state_alloc(const char **files, drf_state_t **__state)
{
    __drf_state_t *state;
    int ret, debug;

    debug = (getelw("DRF_DEBUG") != NULL);
    state = calloc(sizeof(__drf_state_t), 1);
    if (!state) {
        if (debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        return -1;
    } else {
        if (files) {
            ret = drf_state_add_files((void *)state, files);
            if (ret < 0)
                goto failed;
            if (!state->manuals) {
                drf_set_errno(ENOENT);
                goto failed;
            }
        }
        state->debug = debug;
    }
    *__state = (drf_state_t *)state;
    return 0;
failed:
    drf_state_free(state);
    return -1;
}

int drf_state_free(drf_state_t *__state)
{
    int debug;
    __drf_state_t *state = __state;
    drf_device_t *device;
    drf_field_t *field;
    drf_register_t *_register;
    uint32_t i, j, k, l;

    debug = (getelw("DRF_DEBUG") != NULL);
    if (!state) {
        if (debug) {
            fprintf(stderr, "%s() failed (Invalid argument)!\n",
                    __FUNCTION__);
        }
        drf_set_errno(EILWAL);
        return -1;
    } else {
        if (state->addendums) {
            for (i = 0; state->addendums[i]; i++)
                free(state->addendums[i]);
            free(state->addendums);
        }
        if (state->manuals) {
            for (i = 0; state->manuals[i]; i++)
                free(state->manuals[i]);
            free(state->manuals);
        }
        for (i = 0; i < state->n_devices; i++) {
            device = state->devices[i];
            if (device) {
                for (j = 0; j < device->n_registers; j++) {
                    _register = device->registers[j];
                    if (_register) {
                        for (k = 0; k < _register->n_fields; k++) {
                            field = _register->fields[k];
                            if (field) {
                                for (l = 0; l < field->n_defines; l++)
                                    free(field->defines[l]);
                            }
                            free(field);
                        }
                    }
                    free(_register);
                }
            }
            free((char *)device->fname);
            free(device);
        }
        free(state->devices);
        free(state);
        return 0;
    }
}

int drf_state_add_files(drf_state_t *__state, const char **files)
{
    int ret, debug;
    __drf_state_t *state = __state;
    struct stat stat_buffer;
    char **manuals = NULL, **addendums = NULL;
    uint32_t n_manuals = 0, n_addendums = 0;
    char *str, *file = NULL;
    uint32_t i;

    debug = (getelw("DRF_DEBUG") != NULL);
    if (!state || !files || !files[0]) {
        if (debug) {
            fprintf(stderr, "%s() failed (Invalid argument)!\n",
                    __FUNCTION__);
        }
        drf_set_errno(EILWAL);
        return -1;
    } else {
        if (state->addendums) {
            for (i = 0; state->addendums[i]; i++)
                n_addendums++;
        }
        if (state->manuals) {
            for (i = 0; state->manuals[i]; i++)
                n_manuals++;
        }
        for (i = 0; files[i]; i++) {
            memset(&stat_buffer, 0, sizeof(stat_buffer));
            ret = stat(files[i], &stat_buffer);            
            if (ret < 0) {
                if (debug) {
                    fprintf(stderr, "stat() failed (%s)!\n",
                            strerror(errno));
                }
                drf_set_errno(errno);
                goto failed;
            }
            file = malloc(strlen(files[i]) + 1);
            if (!file) {
                if (debug) {
                    fprintf(stderr, "malloc() failed (%s)!\n",
                            strerror(errno));
                }
                drf_set_errno(errno);
                goto failed;
            }
            str = strcasestr(files[i], "_addendum.h");
            if (str) {
                if (strcasecmp(str, "_addendum.h"))
                    str = NULL;
            }
            strcpy(file, files[i]);
            if (!str) {
                n_manuals++;
                manuals = realloc(state->manuals,
                        (sizeof(char *) * (n_manuals + 1)));
                if (!manuals) {
                    if (debug) {
                        fprintf(stderr, "realloc() failed (%s)!\n",
                                strerror(errno));
                    }
                    free(file);
                    drf_set_errno(errno);
                    goto failed;
                }
                manuals[n_manuals-1] = file;
                manuals[n_manuals] = NULL;
            } else {
                n_addendums++;
                addendums = realloc(state->addendums,
                        (sizeof(char *) * (n_addendums + 1)));
                if (!addendums) {
                    if (debug) {
                        fprintf(stderr, "realloc() failed (%s)!\n",
                                strerror(errno));
                    }
                    free(file);
                    drf_set_errno(errno);
                    goto failed;
                }
                addendums[n_addendums-1] = file;
                addendums[n_addendums] = NULL;
            }
            state->manuals = manuals;
            state->addendums = addendums;
        }
    }
    return 0;
failed:
    return -1;
}

int drf_state_add_buffers(drf_state_t *__state, const char **buffers,
        const uint32_t *buffer_sizes)
{
    int debug;
    __drf_state_t *state = __state;
    drf_mem_buffer_t *mem_buffer, *mem_buffers = NULL;
    uint32_t n_mem_buffers = 0;
    uint32_t i;

    debug = (getelw("DRF_DEBUG") != NULL);
    if (!state || !buffers || !buffer_sizes) {
        if (debug) {
            fprintf(stderr, "%s() failed (Invalid argument)!\n",
                    __FUNCTION__);
        }
        drf_set_errno(EILWAL);
        return -1;
    } else {
        if (state->mem_buffers) {
            for (i = 0; state->mem_buffers[i].data; i++)
                n_mem_buffers++;
        }
        for (i = 0; buffers[i]; i++) {
            if (buffer_sizes[i] != 0) {
                n_mem_buffers++;
                mem_buffers = realloc(state->mem_buffers,
                        (sizeof(*mem_buffer) * (n_mem_buffers + 1)));
                if (!mem_buffers) {
                    if (debug) {
                        fprintf(stderr, "realloc() failed (%s)!\n",
                                strerror(errno));
                    }
                    drf_set_errno(errno);
                    goto failed;
                }
                mem_buffer = &mem_buffers[n_mem_buffers-1];
                mem_buffer->data = buffers[i];
                mem_buffer->data_size = buffer_sizes[i];
                memset(&mem_buffers[n_mem_buffers], 0, sizeof(*mem_buffer));
            }
            state->mem_buffers = mem_buffers;
        }
    }
    return 0;
failed:
    return -1;
}
