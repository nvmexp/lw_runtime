/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_FSM_H
#define INCLUDED_LWSCIBUF_FSM_H

/* Implements a Mealy FSM.
 * We choose not to support hierarchical FSMs for now since we don't expect to
 * support many states, which means we shouldn't deal with state explosion.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "lwscicommon_covanalysis.h"
#include "lwscierror.h"

/* Typedef for the output function that implements the state behaviour.
 */
typedef LwSciError (*LwSciBufFsmOutputFn)(void* context, const void* data);

/* Forward declare */
struct LwSciBufFsmStruct;
typedef struct LwSciBufFsmStruct LwSciBufFsm;

/* In the interests of saving space, we implement the state transition table
 * using a function that returns the new state.
 */
typedef void (*LwSciBufFsmStateTransition)
    (LwSciBufFsm* fsm, const void* data, bool* transitioned);

/* This encapsulates the state output function and the state transition table
 * for a particular state.
 */
typedef struct {
    /* The output function. This implements the logic for this state. */
    LwSciBufFsmOutputFn output;

    /* This implements the FSM state transition table. */
    LwSciBufFsmStateTransition stateTransition;
} LwSciBufFsmState;

struct LwSciBufFsmStruct {
    /* Current state. */
    const LwSciBufFsmState* lwrrState;

    /* Context data accessible via every state's output function */
    void* context;
};

/* Define a state and its associated output function implementing the state
 * behaviour.
 *
 * States should be declared via FSM_DEFINE_STATE prior to declaring a
 * transition table associated with the state. */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_10), "LwSciBuf-ADV-MISRAC2012-021")
#define FSM_DEFINE_STATE(state, outputfn) \
    static void fsmTransition##state( \
        LwSciBufFsm* fsm, \
        const void* data, \
        bool* transitioned); \
    static const LwSciBufFsmState state = {\
        .output = outputfn, \
        .stateTransition = fsmTransition##state, \
    }
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_10))

/* State Transition Macros
 *
 * The following macros are intended to be used in conjunction with each other.
 * For example:
 *
 *     FSM_DEFINE_STATE_TRANSITION_TABLE(stateFoo) {
 *         FSM_ADD_TRANSITION_STATE(stateBar, BarGuard);
 *         FSM_ADD_DEFAULT_TRANSITION(stateError);
 *         FSM_DEFINE_STATE_TRANSITION_TABLE_END;
 *     }
 *
 * This makes it harder to mess up setting up an attribute on the
 * LwSciBufFsmState or condition in the transition table.
 */

/* Define a State Transition table.
 *
 * This should be called after FSM_DEFINE_STATE.
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_10), "LwSciBuf-ADV-MISRAC2012-021")
#define FSM_DEFINE_STATE_TRANSITION_TABLE(state) \
    static void fsmTransition##state( \
        LwSciBufFsm* fsm, \
        const void* data, \
        bool* transitioned)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_10))

/* Needed for MISRA 2.2, in addition to keeping the abstraction nice and not
 * requiring labels to be manually created within the block.
 *
 * This must be the last item in a FSM_DEFINE_STATE_TRANSITION_TABLE block.
 */
#define FSM_DEFINE_STATE_TRANSITION_TABLE_END \
    ret: \
        return

/* Add a transition that only oclwrs whenever some condition is true. In other
 * words, the state transition is _guarded_ by a Guard function.
 * Note: Guard functions must _never_ error.
 *
 * This should be called within a FSM_DEFINE_STATE_TRANSITION_TABLE block.
 */
#define FSM_ADD_TRANSITION_STATE(state, guard) \
    do { \
        if ((guard)(fsm->context, data)) { \
            fsm->lwrrState = &(state); \
            *transitioned = true; \
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")\
            goto ret; \
        } \
    } while (1U == 0U)

/* For MISRA 2.2 purposes, we introduce a separate default transition macro.
 * Otherwise we would implement this via a guard that simply allows every state
 * transition.
 *
 * If used, this should be the last state declared in the transition table.
 *
 * This should be called within a FSM_DEFINE_STATE_TRANSITION_TABLE block.
 */
#define FSM_ADD_DEFAULT_TRANSITION(state) \
    do { \
        (void)fsm; \
        (void)data; \
        fsm->lwrrState = &(state); \
        *transitioned = true; \
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")\
        goto ret; \
    } while (1U == 0U)

/**
 * @brief Initializes the LwSciBufFsm to the starting LwSciBufFsmState and any
 * context data that should be accessible within the FSM exelwtion.
 *
 * @param[in,out] fsm LwSciBufFsm to initialize
 * @param[in] initialState LwSciBufFsmState to start the LwSciBufFsm in
 * @param[in] context Any global context that various states need to access
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *   - fsm is NULL
 *   - initialState is NULL
 */
void LwSciBufFsmInit(
    LwSciBufFsm* fsm,
    const LwSciBufFsmState* initialState,
    void* context);

/**
 * @brief Passes data into the LwSciBufFSm for the current state to process. If
 * a state transition is possible, the transition will first occur before the
 * new LwSciBufFsmState's output function is called to process the data. Any
 * errors that occur from with the LwSciBufFsmState output function are
 * propagated out.
 *
 * @param[in,out] fsm The LwSciBufFsm to advance
 * @param[in] data The data the LwSciBufFsm is processing
 * @param[out] propagatedError The propagated LwSciError oclwrring from within
 * the LwSciBufFsmState output function.
 *
 * @return bool, the completion status of the operation:
 * - true if successful.
 * - false if no state transition oclwrred.
 * - Panics if any of the following oclwrs:
 *   - fsm is NULL
 *   - fsm is invalid
 *   - data is NULL
 *   - propagatedError is NULL
 */
bool LwSciBufFsmEventProcess(
    LwSciBufFsm* fsm,
    const void* data,
    LwSciError* propagatedError);

#endif /* INCLUDED_LWSCIBUF_FSM_H */
