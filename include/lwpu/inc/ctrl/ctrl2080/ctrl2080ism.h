/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2011 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080ism.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_XX ISM control commands and parameters */

/*
 * LW2080_CTRL_ISM_MINI_PARAMS_1CLK
 *
 * See LW_ISM_MINI_1CLK_*.
 * Fields not represented are either output fields or configured automatically.
 */
typedef struct LW2080_CTRL_ISM_MINI_PARAMS_1CLK {
    LwU32 srcSel;
    LwU32 outDiv;
    LwU32 mode;
    LwU32 init;
    LwU32 finit;
} LW2080_CTRL_ISM_MINI_PARAMS_1CLK;

/*
 * LW2080_CTRL_ISM_MINI_PARAMS_1CLK_DBG
 *
 * See LW_ISM_MINI_1CLK_DBG_*.
 * Fields not represented are either output fields or configured automatically.
 */
typedef struct LW2080_CTRL_ISM_MINI_PARAMS_1CLK_DBG {
    LwU32 srcSel;
    LwU32 outDiv;
    LwU32 mode;
    LwU32 init;
    LwU32 finit;
    LwU32 clampDbgClkout;
} LW2080_CTRL_ISM_MINI_PARAMS_1CLK_DBG;

/*
 * LW2080_CTRL_ISM_MINI_PARAMS_2CLK
 *
 * See LW_ISM_MINI_2CLK_*.
 * Fields not represented are either output fields or configured automatically.
 */
typedef struct LW2080_CTRL_ISM_MINI_PARAMS_2CLK {
    LwU32 srcSel;
    LwU32 outDiv;
    LwU32 mode;
    LwU32 init;
    LwU32 finit;
    LwU32 refclkSel;
} LW2080_CTRL_ISM_MINI_PARAMS_2CLK;

/*
 * LW2080_CTRL_ISM_MINI_PARAMS
 *
 * Union of all ISM_MINI input structs that can be specified for an experiment.
 */


/*
 * LW2080_CTRL_ISM_MINI_TYPE
 *
 * Enumeration of all ISM_MINI types that can be specified for an experiment.
 */
#define LW2080_CTRL_ISM_MINI_TYPE_1CLK     (0x00000001)
#define LW2080_CTRL_ISM_MINI_TYPE_1CLK_DBG (0x00000002)
#define LW2080_CTRL_ISM_MINI_TYPE_2CLK     (0x00000003)

/*
 * LW2080_CTRL_ISM_MINI
 * 
 *   startBit
 *       Starting bit within the chain of this ISM_MINI.
 *   type
 *       One of LW2080_CTRL_ISM_MINI_TYPE_*.
 *   params
 *       The parameters corresponding to this ISM_MINI_TYPE.
 */
typedef struct LW2080_CTRL_ISM_MINI {
    LwU32 startBit;
    LwU32 type;
    union {
        LW2080_CTRL_ISM_MINI_PARAMS_1CLK     oneClk;
        LW2080_CTRL_ISM_MINI_PARAMS_1CLK_DBG oneClkDbg;
        LW2080_CTRL_ISM_MINI_PARAMS_2CLK     twoClk;
    } params;
} LW2080_CTRL_ISM_MINI;

/*
 * LW2080_CTRL_ISM_CHAIN
 * 
 *   id
 *       Chain ID. One of LW_ISM_CHAIN_*_ID.
 *   width
 *       Length of chain in bits. One of LW_ISM_CHAIN_*_WIDTH.
 *   chipletSel
 *       Chiplet selection. One of LW_ISM_CHAIN_*_CHIPLET_SEL.
 *   miniIndex
 *       Index into this experiment's array of ISM_MINIs of the first
 *       ISM_MINI that is on this chain.
 *   miniCount
 *       Number of ISM_MINIs on this chain used in this experiment.
 */
typedef struct LW2080_CTRL_ISM_CHAIN {
    LwU32 id;
    LwU32 width;
    LwU32 chipletSel;
    LwU32 miniIndex;
    LwU32 miniCount;
} LW2080_CTRL_ISM_CHAIN;

/*
 * LW2080_CTRL_ISM_CTRL
 *
 * Configuration for LW_ISM_CTRL_*.
 * Fields not represented are either output fields or configured automatically.
 *
 *   duration
 *       See LW_ISM_CTRL_DURATION.
 *   delay
 *       See LW_ISM_CTRL_DELAY. 
 *   triggerSrcInitial
 *       See LW_ISM_CTRL_TRIGGER_SRC.
 *       This is the TRIGGER_SRC used for the first iteration of the experiment.
 *       Having a different trigger source for the first iteration allows experiments
 *       to trigger on some event and then run continuously (as well as other combinations).
 *   triggerSrc
 *       See LW_ISM_CTRL_TRIGGER_SRC.
 *       This is the TRIGGER_SRC used for the remaining iterations of the experiment.   
 */
typedef struct LW2080_CTRL_ISM_CTRL {
    LwU32 duration;
    LwU32 delay;
    LwU32 triggerSrcInitial;
    LwU32 triggerSrc;
} LW2080_CTRL_ISM_CTRL;

/*
 * LW2080_CTRL_CMD_ISM_START_EXPERIMENT
 *
 * Start an ISM experiment to collect ISM_MINI count data.
 * This call specifies the entire configuration needed to run the experiment
 * and aborts any previously running experiment.
 *
 * NOTE: To abort an experiment without starting a new experiment you
 *       can call this with 0 iterations, 0 chainCount, and 0 miniCount.
 *       This should effectively release the resources used by RM.
 *
 *   ctrl
 *       ISM control parameters affecting the entire experiment.
 *   iterCount
 *       The number of iterations to run the experiment.
 *   iterDuration
 *       The MINIMUM time in microseconds that must elapse between
 *       the timestamps of each iteration.
 *   pChains
 *       Pointer to an array of <chainCount> LW2080_CTRL_ISM_CHAIN structs.
 *       The <miniIndex> fields within these structs index into <pMinis>.
 *   chainCount
 *       Number of ISM_CHAINs in this experiment.
 *   pMinis
 *       Pointer to an array of <miniCount> LW2080_CTRL_ISM_MINI structs.
 *   miniCount
 *       Number of ISM_MINIs in this experiment.
 *
 * Possible status values returned are:
 *   TBD
 */
#define LW2080_CTRL_CMD_ISM_START_EXPERIMENT (0x20800001) /* finn: Evaluated from "(0x2080 << 16) | 0x1" */

typedef struct LW2080_CTRL_ISM_START_EXPERIMENT_PARAMS {
    LwU32 ctrl;
    LwU32 iterCount;
    LwU32 iterDuration;
    LW_DECLARE_ALIGNED(LwP64 pChains, 8);
    LwU32 chainCount;
    LW_DECLARE_ALIGNED(LwP64 pMinis, 8);
    LwU32 miniCount;
} LW2080_CTRL_ISM_START_EXPERIMENT_PARAMS;

/*
 * LW2080_CTRL_ISM_ITERATION_HEADER
 *
 *   timestamp
 *       PTIMER value read on this iteration.
 *   tsense
 *       TSENSE value read on this iteration.
 */
typedef struct LW2080_CTRL_ISM_ITERATION_HEADER {
    LW_DECLARE_ALIGNED(LwU64 timestamp, 8);
    LwU32 tsense;
} LW2080_CTRL_ISM_ITERATION_HEADER;

/*
 * LW2080_CTRL_ISM_MINI_ITERATION
 *
 *   count
 *       See LW_ISM_MINI_*_COUNT.
 */
typedef struct LW2080_CTRL_ISM_MINI_ITERATION {
    LwU32 count;
} LW2080_CTRL_ISM_MINI_ITERATION;

/*
 * LW2080_CTRL_CMD_ISM_CHECK_EXPERIMENT
 *
 * Check the status of a lwrrently running experiment and copy out the
 * lwrrently available results.
 *
 * Must be called only after a call to ISM_START_EXPERIMENT.
 * The <iterCount> and <miniCount> parameters from that call referenced here are
 * stored internally by RM.
 *
 *   pHeaders
 *       Array of <iterCount> LW2080_CTRL_ISM_ITERATION_HEADER structures.
 *       On return, this array contains valid data up to <iterIndex> - 1.
 *   pMinis
 *       Array of <iterCount> * <miniCount> LW2080_CTRL_ISM_MINI_ITERATION structures.
 *       On return, this array contains valid data up to (<iterIndex> * <miniCount>) - 1.
 *   iterIndex
 *       On input, this indicates the index of the first iteration to be be copied out (if completed).
 *       On return, this indicates the number of iterations that have completed.
 *
 *       This should normally be 0 on the first call and untouched on subsequent calls
 *       until the experiment is complete. This avoids redundant copying and allows
 *       the client to use data from completed iterations immediately.
 *
 *       Once <iterIndex> == <iterCount>, RM releases its internal resources
 *       and the experiment is considered complete. Further calls to ISM_CHECK_EXPERIMENT
 *       are illegal until another experiment is started.
 *
 * Possible status values returned are:
 *   TBD
 */
#define LW2080_CTRL_CMD_ISM_CHECK_EXPERIMENT (0x20800002) /* finn: Evaluated from "(0x2080 << 16) | 0x2" */

typedef struct LW2080_CTRL_ISM_CHECK_EXPERIMENT_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pHeaders, 8);
    LW_DECLARE_ALIGNED(LwP64 pMinis, 8);
    LwU32 iterIndex;
} LW2080_CTRL_ISM_CHECK_EXPERIMENT_PARAMS;
