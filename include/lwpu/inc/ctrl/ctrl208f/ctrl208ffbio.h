/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2010-2015 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl208f/ctrl208ffbio.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
 * LW208F_CTRL_CMD_FBIO_SETUP_TRAINING_EXP
 *
 * This command defines an fbio training experiment for later use.
 *
 * This command has the form of a 'cmd' operation descriminant
 * followed by a union populated with an operand to match the 'cmd'.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *
 */
#define LW208F_CTRL_CMD_FBIO_SETUP_TRAINING_EXP                     (0x208f0a03) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FBIO_INTERFACE_ID << 8) | LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_PARAMS_MESSAGE_ID" */

/*
 * LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_NULL
 * 
 * This command has no effect when used.
 */
#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_NULL                    0

/*
 * LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_NUMBER_OF_MOD_SLOTS
 *
 * This command defines the number of register modification slots to be used
 * in the setup phase of the pass/fail training exercise.  Using this cmd
 * causes any previously defined modification table to be cleared/released.
 * The maximum size of the table is defined by
 *     _SETUP_FBIO_TRAINING_EXP_SET_NUMBER_OF_MOD_SLOTS__MAX
 * Using the value of zero for number of mod slots is valid.  "Mod slots"
 * are defined below.
 *
 * Possible status values returned are:
 *   LW_ERR_ILWALID_LIMIT
 *     if modSlots is > __MAX
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *     if we're out of memory setting up the mod slot table
 */
#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_NUMBER_OF_MOD_SLOTS 1
typedef struct LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_NUMBER_OF_MOD_SLOTS_OPERANDS {
    LwU32 modSlots;
} LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_NUMBER_OF_MOD_SLOTS_OPERANDS;

#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_NUMBER_OF_MOD_SLOTS__MAX 256

/*
 * LW208F_CTRL_FBIO_SETUPFBIO_TRAINING_EXP_SET_MOD_SLOT
 *
 * This command is used to define each individual element of the the sequence
 * of operations which will be performed before running the pass/fail training
 * exercise.  Note that this cmd is multi-cmd (with a null cmd all its own,
 * etc).
 *
 * Possible status values returned are:
 *   LW_ERR_ILWALID_INDEX
 *     if seq is out of range for previously set SET_NUMBER_OF_MOD_SLOTS
 *     operation.
 *   LW_ERR_ILWALID_COMMAND
 *     if cmd isn't recognized
 */
#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT                 2

#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_NULL            0

/*
 * LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_SET_REGISTER
 *
 * This cmd is used to define a register transaction to be applied in
 * sequence before running the pass/fail experiment.  This is where
 * LW_PFB_TRAINING_ADR, LW_PFB_TRAINING_CMD, LW_PFB_TRAINING_DP(i),
 * LW_PFB_TRAINING_THRESHOLD and LW_PFB_TRAINING_MASK, etc. should be 
 * configured before calling back with cmd
 * LW208F_CTRL_CMD_FBIO_RUN_TRAINING_EXP.
 *
 *   reg
 *     This parameter specifies the BAR0 register offset to affect.
 *   andMask
 *   orMask
 *     These parameter specify the RMW values to be used in the following:
 *       write32(reg, (read32(reg) & andMask) | orMask)
 */
#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_SET_REGISTER    1
typedef struct LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_SET_REGISTER_OPERANDS {
    LwU32 reg;
    LwU32 andMask;
    LwU32 orMask;
} LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_SET_REGISTER_OPERANDS;

/*
 * LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_DELAY
 *
 * This cmd is used to define a delay to be applied in the setup sequence
 * before running the pass/fail experiment.
 *
 *  usec
 *    Specifies delay to be used in microseconds.
 */

#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_DELAY 2
typedef struct LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_DELAY_OPERANDS {
    LwU32 usec;
} LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_DELAY_OPERANDS;

/*
 * LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_INIT_LT
 *
 * This cmd is used to define a point where normal link training initialization
 * may be exelwted in the sequence before running the pass/fail experiment.  
 * In general, this is not needed since it is done during normal
 * initialization, but does allow re-initialization during the sequence.
 */
#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_INIT_LT  3

/*
 * LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_RUN_LT
 *
 * This cmd is used to define a point where normal link training may be
 * exelwted in the sequence before running the pass/fail experiment.
 * In general, this is not needed since it is somewhat redundant.
 */
#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_RUN_LT   4

/*
 * LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_STOP_SEQ
 *
 * This cmd is used to define a point where the sequence stops without
 * running the pass/fail experiment.  
 */
#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_STOP_SEQ 5



typedef struct LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_PARAMS {
    LwU32 cmd;
    /* C form: LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_OPERANDS op; */
    union {

        LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_SET_REGISTER_OPERANDS setRegister;

        LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_DELAY_OPERANDS        delay;
    } op;
    LwU32 seq;
} LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_PARAMS;



#define LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_PARAMS {
    LwU32 cmd;

    /* C form: LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_OPERANDS op; */
    union {

        LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_NUMBER_OF_MOD_SLOTS_OPERANDS setNumberOfModSlots;

        LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_SET_MOD_SLOT_PARAMS              setModSlot;
    } op;
} LW208F_CTRL_FBIO_SETUP_TRAINING_EXP_PARAMS;

/*
 * LW208F_CTRL_CMD_FBIO_RUN_TRAINING_EXP
 *
 * This command runs the previously defined experiment and returns status on
 * pass/fail.  LW_OK is returned in the case of both pass
 * and fail.
 *
 * The risingPasses and fallingPasses outputs represents the results across
 * all partitions and bytelanes.  Each array entry corresponds to a partition
 * and the bits within each member represent the bytelane.  So e.g.:
 * risingPasses[4] represents the rising pass results for all of partition 4's
 * byte lanes.
 *
 * The partitionsValid bitmask represents the partitions for which the results
 * in risingPasses and fallingPasses are valid (not floorswept).
 *
 * The bytelanesValid bitmask represents the bytelanes for which the results
 * are valid (available bytelanes).
 *
 * The failingDebug[] represent debug data for why (if so) a test fails.
 * This is basically LW_PFB_TRAINING_DEBUG(i)
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_TIMEOUT
 *       if hardware doesn't respond to request in a timely manner.
 *   LW_ERR_ILWALID_DATA
 *       if there was some sort of very weird data corruption issue.
 */
#define LW208F_CTRL_CMD_FBIO_RUN_TRAINING_EXP          (0x208f0a04) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FBIO_INTERFACE_ID << 8) | LW208F_CTRL_FBIO_RUN_TRAINING_EXP_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FBIO_RUN_TRAINING_EXP_RESULT__SIZE 8
#define LW208F_CTRL_FBIO_RUN_TRAINING_EXP_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW208F_CTRL_FBIO_RUN_TRAINING_EXP_PARAMS {LwU32 risingPasses[LW208F_CTRL_FBIO_RUN_TRAINING_EXP_RESULT__SIZE];
    LwU32 fallingPasses[LW208F_CTRL_FBIO_RUN_TRAINING_EXP_RESULT__SIZE];
    LwU32 failingDebug[LW208F_CTRL_FBIO_RUN_TRAINING_EXP_RESULT__SIZE];
    LwU32 partitionsValid;
    LwU32 bytelanesValid;
} LW208F_CTRL_FBIO_RUN_TRAINING_EXP_PARAMS;

#define LW208F_CTRL_CMD_FBIO_GET_TRAINING_CAPS (0x208f0a05) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FBIO_INTERFACE_ID << 8) | LW208F_CTRL_FBIO_GET_TRAINING_CAPS_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FBIO_GET_TRAINING_CAPS_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW208F_CTRL_FBIO_GET_TRAINING_CAPS_PARAMS {
    LwU32 supported;
} LW208F_CTRL_FBIO_GET_TRAINING_CAPS_PARAMS;

// _ctrl208ffbio_h_
