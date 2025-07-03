/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_fsm.h"

#include "lwscicommon_os.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"

void LwSciBufFsmInit(
    LwSciBufFsm* fsm,
    const LwSciBufFsmState* initialState,
    void* context)
{
    LWSCI_FNENTRY("");

    if ((fsm == NULL) || (initialState == NULL)) {
        /* These variables are controlled by LwSci APIs and are not exposed to
         * the caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LwSciCommonPanic();
    }

    fsm->lwrrState = initialState;
    fsm->context = context;

    LWSCI_FNEXIT("");
}

bool LwSciBufFsmEventProcess(
    LwSciBufFsm* fsm,
    const void* data,
    LwSciError* propagatedError)
{
    bool transitioned = false;

    LWSCI_FNENTRY("");

    if ((fsm == NULL) || (data == NULL) || (propagatedError == NULL)) {
        /* These variables are controlled by LwSci APIs and are not exposed to
         * the caller of any Public API. As such, we can panic here since this
         * indicates incorrect usage. */
        LwSciCommonPanic();
    }

    if ((fsm->lwrrState == NULL) || (fsm->lwrrState->stateTransition == NULL)) {
        /* These variables are controlled by LwSci APIs and are not exposed to
         * the caller of any Public API. As such, we can panic here since this
         * indicates the FSM wasn't correctly set up. */
        LwSciCommonPanic();
    }

    fsm->lwrrState->stateTransition(fsm, data, &transitioned);
    if (transitioned == true) {
        if (fsm->lwrrState == NULL) {
            /* These variables are controlled by LwSci APIs and are not exposed
             * to the caller of any Public API. As such, we can panic here
             * since this indicates the FSM wasn't correctly set up. */
            LwSciCommonPanic();
        }

        if (fsm->lwrrState->output != NULL) {
            /* Call the output function if it exists */
            *propagatedError = fsm->lwrrState->output(fsm->context, data);

            if (*propagatedError != LwSciError_Success) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

ret:
    LWSCI_FNEXIT("");
    return transitioned;
}
