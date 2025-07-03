/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or dis  closure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file
 * @see     https://confluence.lwpu.com/display/RMCLOC/Clocks+3.0
 * @author  Daniel Worpell
 * @author  Eric Colter
 * @author  Antone Vogt-Varvak
 * @author  Lawrence Chang
 * @author  Manisha Choudhury
 */

#ifndef CLK3_FS_MUX_H
#define CLK3_FS_MUX_H


/* ------------------------ Includes --------------------------------------- */

#include "clk/fs/clkfreqsrc.h"
#include "clk/clkfieldvalue.h"

/* ------------------------ Macros ----------------------------------------- */
/*******************************************************************************
 * ClkSignalPath/ClkSignalNode Constants
*******************************************************************************/
typedef LwU8    ClkSignalNode;

// Number of bits for each signal node.
#define CLK_SIGNAL_NODE_WIDTH           4

//
// Number of potential nodes that can be represented with a single node value.
// We subtract one so because _INDETERMINATE is not included with the count.
//
#define CLK_SIGNAL_NODE_COUNT           (BIT(CLK_SIGNAL_NODE_WIDTH) - 1)

// Value to indicate that the node is not applicable nor specified
#define CLK_SIGNAL_NODE_INDETERMINATE   ((ClkSignalNode) CLK_SIGNAL_NODE_COUNT)

/* ------------------------ Datatypes -------------------------------------- */

typedef ClkFreqSrc_Virtual ClkMux_Virtual;


/*!
 * @extends     ClkFreqSrc
 * @brief       Multi-Input Frequency Source
 *
 * @details     These classes contain the logic for frequency source objects that
 *              have more than one input.  This may mean a simple multiplexer or
 *              an array of multiplexers (e.g. OSM) provided they are controlled
 *              by the same register.
 */
typedef struct
{
/*!
 * @brief       Inherited state
 * @ilwariant   Inherited state must always be first.
 */
    ClkFreqSrc  super;

/*!
 * @brief       Map between switch register values and node numbers of potential inputs
 * @see         muxGateValue
 * @protected
 *
 * @details     This final member maps a potential input to the field mask and
 *              value required to program the multiplexer to that input.
 *              These values apply to both 'muxReg' and 'statusReg'.
 *
 *              If inputs of this object can be gated with a field in 'muxReg',
 *              then each element of this array must include the field value
 *              to ungate the inputs.  This is smart even if we don't use the
 *              gating feature, but essential if we do (i.e. if 'muxGateValue'
 *              is used).  OSMs are an example.
 *
 * @ilwariant   The number of elements in this array must equal ClkMux::count.
 *
 * @note        final:          The pointer is initialized by the constructor and never subsequently changed.
 *                              Moreover, the data in the array is never changed after construction.
 */
    ClkFieldValueMap muxValueMap;

/*!
 * @brief       Potential input array
 * @protected
 *
 * @details     This array contains pointers to the frequency source objecs that
 *              that object may use as inputs.  Unconnected inputs are set to
 *              the NULL pointer.
 *
 * @see         ClkMux::count
 *
 * @note        final:          Value does not change after construction.
 */
    ClkFreqSrc**  input;

/*!
 * @brief       Register to switch multiplexer among the inputs
 * @protected
 *
 * @details     For Turing One-Source Modules (OSMs), the name of this register
 *              in the manuals is generally LW_PTRIM_SYS_xxxCLK_yyy_SWITCH where
 *              xxx is the name of the clock domain and yyy represents the path.
 *              The generic name for the register is LW_PTRIM_SYS_CLK_SWITCH.
 *              OSMs generally do not share this register with other objects.
 *
 *              For Turing linear dividers, LW_PTRIM_SYS_SEL_VCO is the name
 *              given in the manuals.  In contrast to OSMs, this register is
 *              generally shared with other linear dividers.
 *
 *              For Ampere and later Switch Dividers, this register is usually
 *              named LW_PVTRIM_SYS_xxx_SWITCH_DIVIDER where xxx is the name of
 *              the clock domains.
 *
 *              This member can not be 'const' because there is floorsweeping
 *              logic for GA100 in 'clkConstruct_SchematicDag'.  However, its
 *              values does not change after construction.
 */
    LwU32 muxRegAddr;

/*!
 * @brief       Switch register value to disable all potential inputs
 * @see         CLK_DRF__FIELDVALUE (for initialization)
 * @see         muxValueMap
 * @protected
 *
 * @details     For OSM3s, this is LW_PTRIM_SYS_CLK_SWITCH_STOPCLK_YES.
 *
 *              Leave this field zero to disable gating (or if this object can
 *              not be gated).
 *
 *              If this object can be gated, but not with 'muxReg', then use a
 *              'ClkGate' object instead of this field.
 *
 *              If this member contains something other than zeroes, then the
 *              ungating field value must be set in all entries of 'muxValueMap'.
 */
    const ClkFieldValue muxGateField;

/*!
 * @brief       Length of potential input array
 * @protected
 *
 * @details     This is how long the array pointed to by ClkMux::input is.
 *
 * @see         ClkMux::input
 */
    LwU8 count;

} ClkMux;


/* ------------------------ External Definitions --------------------------- */

extern ClkFreqSrc_Virtual clkVirtual_Mux;


/* ------------------------ Function Prototypes ---------------------------- */

/*!
 * @brief       Implementation of the virtual function
 * @memberof    ClkMux
 * @protected
 */
void clkReadAndPrint_Mux(ClkMux *this, LwU32 *pFreqKHz);

/* ------------------------ Include Derived Types -------------------------- */

#endif // CLK3_FS_MUX_H

