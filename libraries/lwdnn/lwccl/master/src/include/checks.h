/*************************************************************************
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"

// Check LWCA calls
#define LWDACHECK(cmd) do {                                 \
    lwdaError_t e = cmd;                                    \
    if( e != lwdaSuccess ) {                                \
        WARN("Lwca failure '%s'", lwdaGetErrorString(e));   \
        return ncclUnhandledLwdaError;                      \
    }                                                       \
} while(false)

#define LWDACHECKGOTO(cmd, res, label) do {                 \
    lwdaError_t e = cmd;                                    \
    if( e != lwdaSuccess ) {                                \
        WARN("Lwca failure '%s'", lwdaGetErrorString(e));   \
        res = ncclUnhandledLwdaError;                       \
        goto label;                                         \
    }                                                       \
} while(false)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#endif
