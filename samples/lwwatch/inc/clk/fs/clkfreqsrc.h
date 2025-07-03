/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
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
 */

#ifndef CLK3_FS_FREQSRC_H
#define CLK3_FS_FREQSRC_H

#include "lwtypes.h"
/* ------------------------ Includes --------------------------------------- */
/* ------------------------ Macros ----------------------------------------- */

/*!
 * @brief       Virtual Interface Point Thunks
 * @memberof    ClkFreqSrc
 * @see         ClkFreqSrc_Virtual
 * @see         https://wiki.lwpu.com/engwiki/index.php/Resman/RM_Foundations/Lwrrent_Projects/RTOS-FLCN-Scripts#rtos-flcn-script_-_Static_UCode_Analysis_on_objdump
 * @public
 *
 * @details     Thunk (thin wrappers) used to call virtual interface pointer.
 *
 *              In general, virtual Implementation should be called via this
 *              '_FreqSrc_VIP' thunk.  In effect, it defines the virtual
 *              interface for 'ClkFreqSrc'.  See comments in 'ClkFreqSrc_Virtual'
 *              for each function for detail information.
 *
 *
 * @note        Although this thunk is public, the implementation itself
 *              is protected, which means that subclasses may call it directly
 *              for the purpose of daisy-chaining up the inheritance tree.
 */
#define clkReadAndPrint_FreqSrc_VIP(this, pOutput) ((this)->pVirtual->clkReadAndPrint((this), (pOutput)))

/*!
 * @brief       Define virtual table
 * @memberof    ClkFreqSrc
 * @see         ClkFreqSrc_Virtual
 * @see         drivers/resman/arch/lwalloc/common/inc/pmu/pmuifclk.h
 *
 * @details     This macros defines the content (using initializers) of the
 *              table of virtual function pointers for the specified class.
 *
 *              The class is named 'Clk[class]' where '[class]' is specified
 *              by '_CLASS'
 *
 *              The class must declare all the members of 'ClkFreqSrc_Virtual'
 *              in the form of '[fun]_[class]'. For example: 'clkReadAndPrint_Pll'
 *
 *              In the case where the class inherits an implementation from its
 *              super class, a macro should be used to define an alias.  For
 *              example, 'ClkPdiv' inherits 'clkReadAndPrint' from 'ClkWire',
 *              its super class, rather than define its own implementation.
 *              As such, its header file contains this macro:
 *                  #define clkReadAndPrint_Multiplier clkReadAndPrint_Wire
 *
 *              Usage of this macro should be followed with a semicolon.
 *
 * @param[in]   _CLASS      Bare name of Clocks 3.x class (e.g. 'Pll')
 */
#define CLK_DEFINE_VTABLE__FREQSRC(_CLASS)                                          \
ClkFreqSrc_Virtual clkVirtual_##_CLASS = { (ClkReadAndPrint_FreqSrc_VIP(*)) (clkReadAndPrint_##_CLASS) }

/* ------------------------ Datatypes -------------------------------------- */


typedef struct ClkFreqSrc           ClkFreqSrc;         //!< One per object -- Statically allocated
typedef struct ClkFreqSrc_Virtual   ClkFreqSrc_Virtual; //!< One per class -- Statically allocatied

/*!
 * @brief       Function Pointer Types for 'ClkFreqSrc_Virtual'
 * @memberof    ClkFreqSrc
 * @see         ClkFreqSrc_Virtual
 */
typedef void ClkReadAndPrint_FreqSrc_VIP(ClkFreqSrc *pFreqSrc, LwU32 *pFreqKHz);

/*!
 * @brief       Table of virtual interface points
 * @memberof    ClkFreqSrc
 *
 * @details     Each function pointer represents a virtual interface point (VIP).
 *              There is a thunk (thin wrapper) for each virtual interface point
 *              to simplify the calling syntax.  By convention, each of these
 *              thunks has the same name as the corresponding pointer with the
 *              suffix _FreqSrc_VIP attached.  Conceptually, these xxx_VIP thunks
 *              are roughly equivalent to the xxx_HAL wrappers used by the
 *              rmconfig/LWOC code generators.
 *
 *              Clocks 3.x differs from Clocks 2.x in that Clocks 2.x used a
 *              dynamic vtable in which the pointer value would change.  In
 *              costrast, the member function pointers in Clocks 3.x are 'const'
 *              and do not change.
 *
 *              The Implementation referenced by each pointer has a slightly
 *              different parameter list v. the pointer itself in that the
 *              first parameter is the specific sublcass in the implementation,
 *              but is 'ClkFreqSrc' in the pointer.  For example, since
 *              'ClkPll <: ClkFreqSrc', the 'ClkPll' implementation of 'clkRead' is:
 *                  FLCN_STATUS clkReadAndPrint_Pll(ClkPll *this, ClkSignal *pOutput)
 *              even though the pointer typedef is:
 *                  FLCN_STATUS ClkReadAndPrint_FreqSrc_VIP(ClkFreqSrc *this, ClkSignal *pOutput)
 *
 * @note        const:  All structures of this type should be 'const'.
 */
struct ClkFreqSrc_Virtual
{
/*!
 * @brief       Read hardware.
 * @see         ClkFreqSrc::cycle
 *
 * @details     The software state (i.e. configuration) for all phases is
 *              computed based on the state of the hardware.  In other words,
 *              the hardware is read and the results are placed in all elements
 *              of the phase array.
 *
 *              All fields of 'pFreqKHz' are assigned appropriate values upon
 *              successful completion.
 *
 *              Cycles must be detected along the active signal path using
 *              the 'ClkFreqSrc::cycle' flag.  The documentation for this
 *              flag contains details.
 *
 * @pre         'this' may not be NULL.
 *
 * @pre         'pFreqKHz' may not be NULL.
 *
 * @ilwariant   This function does not write any registers.
 *
 * @ilwariant   This function does not throw a DBG_BREAKPOINT (or assertion) due
 *              to an invalid hardware state.
 *
 * @param[in]   this        Instance of ClkWire from which to read
 * @param[out]  pFreqKHz    pointer to be filled with callwlated frequency
 */
    ClkReadAndPrint_FreqSrc_VIP   *clkReadAndPrint;
};


/*!
 * @class       ClkFreqSrc
 * @brief       Frequency Source Class
 * @see         ClkFreqSrc_Virtual
 *
 * @details     For the most part, the data in objects of subclasses does not
 *              change after initialization.  Specifically, the data in these
 *              objects are not epxosed to through the RM_PMU_ data structures.
 *
 *              As such, each concrete subclass contains a pointer to the
 *              subclass-specific RM_PMU_ data structure.  The pointer is not
 *              placed here in the base class since such a pointer would have to
 *              be down-casted in the subclass logic, something we'd like to
 *              avoid.  However, this means that we do no support any sort of
 *              inheritance in the RM_PMU_ data structures.
 *
 *              For each concrete subclass, there is exactly one instance of a
 *              virtual table and all objects of that subclass point to it via
 *              the 'pVirtual' member.
 *
 * @note        In the case where all the members of a struct are inherited from
 *              the superclass, then they are aliased using 'typedef'.
 *
 * @protected   Only functions belonging to the class and subclasses may access
 *              or modify anything in this struct.
 *
 * @note        abstract:  This class does not have a vtable if its own.
 */
struct ClkFreqSrc
{
/*!
 * @brief       Table of Virtual Interface Points (Vtable)
 */
    ClkFreqSrc_Virtual* pVirtual;
    
/*!
 * @brief       Cycle Detector
 * @see         ClkFreqSrc_Virtual::clkRead
 *
 * @details     This member is used to detect cycles within the schematic dag.
 *              In theory, such cycles should not happen.
 *
 *              If the flag is zero on entry, it should set the flag to nonzero
 *              before daisy-chaining to an input, then reset it to zero upon
 *              return from the daisy-chain.
 *
 */
    LwBool      bCycle;
    
/*!
 * @brief       ClkFreqSrc Name
 * @see         ClkFreqSrc_Virtual::clkRead
 *
 * @details     This give each base ClkFreqSrc Object within a schmeatic dag
 *              a unique name to be identified by. This name should be
 *              agnostic to any manual definitions and refer only to the
 *              structure within the schematic dag. For example, the mux
 *              object, which is part of the swdiv structure, in pwrclk will
 *              be given the name 'pwrclk.swdiv.mux'. Each period ('.') 
 *              is the serparation of each containing structure. E.g. 
 *              semantically reading the name 'pwrclk.swdiv.mux' says 
 *              'This mux exists within a swdiv structure which exists in
 *              the pwrclk domain'.
 */    
    const char *name; 
};


/* ------------------------ External Definitions --------------------------- */
/* ------------------------ Function Prototypes ---------------------------- */
/* ------------------------ Include Derived Types -------------------------- */

// Get name of the super FreqSrc object
#define CLK_NAME(x) (((ClkFreqSrc *)x)->name)

#endif // CLK3_FS_FREQSRC_H

