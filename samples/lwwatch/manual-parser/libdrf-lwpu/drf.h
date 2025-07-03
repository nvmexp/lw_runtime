/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_H__
#define __DRF_H__

#include <drf_types.h>

typedef void drf_state_t;

extern int drf_errno;

#define drf_set_errno(e) drf_errno = (e)

#ifdef DEBUG
#define DRF_ASSERT(x) {                                           \
    if (x) {} else {                                              \
        fprintf(stderr, "drf: DRF_ASSERT(%s) failed @ %s:%u!\n",  \
            #x, __FILE__, __LINE__);                              \
    }                                                             \
}
#else
#define DRF_ASSERT(x)
#endif

int drf_state_alloc(const char **files, drf_state_t **state);
int drf_state_free(drf_state_t *state);

int drf_state_add_files(drf_state_t *state, const char **files);
int drf_state_add_buffers(drf_state_t *state,
        const char **buffers, const uint32_t *buffer_sizes);

#endif /* __DRF_H__ */
