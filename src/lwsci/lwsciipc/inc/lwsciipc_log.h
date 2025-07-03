/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIIPC_LOG_H
#define INCLUDED_LWSCIIPC_LOG_H

#if defined(LW_IS_TRACER_ENABLED)
#include <tracer.h>
#endif

/* log mutex error only */
#define log_mutex_err(err, str) \
        do { \
            if(EOK != (err)) { \
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC str, err);   \
            } \
        } while (LW_FALSE)

/* log os lwsci error only */
#define log_os_scierr(err, str) \
        do { \
            if(LwSciError_Success != (err)) { \
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC str, (int32_t)err);   \
            } \
        } while (LW_FALSE)

/* 1) covert errno to LwSciError with logging simply then report it.
 * 2) error ret is set to LwSciError_IlwalidState forcedly.
 * 3) simple version of LwSciIpcErrnoToLwSciErr()
 */
#define report_os_errno(err, str) \
        do { \
            if(EOK != (err)) { \
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC str, err);   \
                ret = LwSciError_IlwalidState;  \
            } \
            else { \
                ret = LwSciError_Success;  \
            } \
        } while (LW_FALSE)

/* log mutex error then goto label */
#define log_mutex_errto(err, str, label) \
        do { \
            if(EOK != (err)) { \
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC str, err);   \
                goto label; \
            } \
        } while (LW_FALSE)

/* log os error then goto label */
#define log_os_errto(err, str, label) log_mutex_errto(err, str, label)

/* 1) log and report mutex error then goto label.
 * 2) error ret is set to LwSciError_IlwalidState forcedly.
 */
#define report_mutex_errto(err, str, label) \
        do { \
            if(EOK != (err)) { \
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC str, err);   \
                ret = LwSciError_IlwalidState;  \
                goto label; \
            } \
        } while (LW_FALSE)

/* 1) log and report os error then goto label.
 * 2) error ret is set to LwSciError_IlwalidState forcedly.
 */
#define report_os_errto(err, str, label) report_mutex_errto(err, str, label)

/* 1) log and report string truncation error then goto label.
 *    strlcat
 * 2) error ret is set to LwSciError_IlwalidState forcedly.
 */
#define report_trunc_errto(val, size, str, label) \
        do { \
            if((val) >= (size)) { \
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC str, err);   \
                ret = LwSciError_IlwalidState;  \
                goto label; \
            } \
        } while (LW_FALSE)

/* 1) log mutex error
 * 2) update mutex error only when ret is LwSciError_Success
 */
#define update_mutex_err(err, str) \
        do { \
            if(EOK != (err)) { \
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC str, err);   \
                if (LwSciError_Success == ret) { \
                    ret = LwSciError_IlwalidState; \
                } \
            } \
        } while (LW_FALSE)

/* 1) log os error
 * 2) update os error only when ret is LwSciError_Success
 */
#define update_os_err(err, str) update_mutex_err(err, str)

/* 1) log mutex error
 * 2) update mutex error only when ret is LwSciError_Success
 * 3) then goto label
 */
#define update_mutex_errto(err, str, label) \
        do { \
            if(EOK != (err)) { \
                LWSCIIPC_ERR_STRINT("error: " LIB_FUNC str, err);   \
                if (LwSciError_Success == ret) { \
                    ret = LwSciError_IlwalidState; \
                } \
                goto label; \
            } \
        } while (LW_FALSE)

/* 1) log os error
 * 2) update os error only when ret is LwSciError_Success
 * 3) then goto label
 */
#define update_os_errto(err, str, label) update_mutex_errto(err, str, label)

/* 1) colwert sivc API error to LwSciError
 * 2) retranslate these two errors to the known error code for clarity
 *      LwSciError_TooBig caused by input, to LwSciError_BadParameter
 *      LwSciError_Overflow cased by internal state, to LwSciError_IlwalidState
 */
#define update_sivc_err(err) \
        do { \
            ret = ErrnoToLwSciErr(err); \
            if (ret == LwSciError_TooBig) { \
                ret = LwSciError_BadParameter; \
            } \
            if (ret == LwSciError_Overflow) { \
                ret = LwSciError_IlwalidState; \
            } \
        } while (LW_FALSE)

#endif /* INCLUDED_LWSCIIPC_LOG_H */

