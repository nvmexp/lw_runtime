/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_LWDA_IFACE_H
#define UCT_LWDA_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/sys/preprocessor.h>
#include <lwda_runtime.h>
#include <lwca.h>


#define UCT_LWDA_DEV_NAME       "lwca"


#define UCT_LWDA_FUNC(_func)                                    \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            lwdaError_t _result = (_func);                      \
            if (lwdaSuccess != _result) {                       \
                ucs_error("%s is failed. ret:%s",               \
                          UCS_PP_MAKE_STRING(_func),            \
                          lwdaGetErrorString(_result));         \
                _status = UCS_ERR_IO_ERROR;                     \
            }                                                   \
        } while (0);                                            \
        _status;                                                \
    })


#define UCT_LWDADRV_FUNC(_func)                                 \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            LWresult _result = (_func);                         \
            const char *lw_err_str;                             \
            if (LWDA_ERROR_NOT_READY == _result) {              \
                _status = UCS_INPROGRESS;                       \
            } else if (LWDA_SUCCESS != _result) {               \
                lwGetErrorString(_result, &lw_err_str);         \
                ucs_error("%s is failed. ret:%s",               \
                          UCS_PP_MAKE_STRING(_func),lw_err_str);\
                _status = UCS_ERR_IO_ERROR;                     \
            }                                                   \
        } while (0);                                            \
        _status;                                                \
    })


#define UCT_LWDADRV_CTX_ACTIVE(_state)                             \
    {                                                              \
        LWcontext lwr_ctx;                                         \
        LWdevice dev;                                              \
        unsigned flags;                                            \
                                                                   \
        _state = 0;                                                \
        /* avoid active state check if no lwca activity */         \
        if ((LWDA_SUCCESS == lwCtxGetLwrrent(&lwr_ctx)) &&         \
            (NULL != lwr_ctx)) {                                   \
            UCT_LWDADRV_FUNC(lwCtxGetDevice(&dev));                \
            UCT_LWDADRV_FUNC(lwDevicePrimaryCtxGetState(dev,       \
                                                        &flags,    \
                                                        &_state)); \
        }                                                          \
    }


ucs_status_t
uct_lwda_base_query_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                           unsigned *num_tl_devices_p);

#endif
