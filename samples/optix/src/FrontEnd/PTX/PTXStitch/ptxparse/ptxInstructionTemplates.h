/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : ptxInstructionTemplates.h
 *
 *  Description              :
 *
 */

#ifndef ptxInstructionTemplates_INCLUDED
#define ptxInstructionTemplates_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "ptxIR.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Functions --------------------------------*/

/*
 * Function         : Initialize this module, by defining 
 *                    all instruction templates
 * Parameters       : 
 * Function Result  : 
 */
void ptxDefineInstructionTemplates(ptxParseData parseData, cString extDescFileName, cString extDescAsString);

uInt ptxGetInstructionOpcode(ptxParseData parseData, String name);

/*
 * Function         : Match parsed instruction information to templates
 * Parameters       : parseData      (I) PTX parsing Data
 *                    name           (I) Instruction name
 *                    storage        (I) For Memory operations: storage space
 *                    arguments      (I) Parsed instruction arguments
 *                    instrType      (I) Imposed instruction type
 *                    nrofArguments  (I) Number of arguments parsed
 *                    nrofInstrTypes (I) Number of instruction types parsed
 *                    vectorMode     (I) True iff. instruction has vector modifier
 *                    parsingMacro   (I) True iff. within macro expansion
 *                    sourcePos      (I) source location of reference
 * Function Result  : 
 */
ptxInstructionTemplate
            ptxMatchInstruction( 
                ptxParseData       parseData,
                String             name, 
                ptxStorageClass    storage[ptxMAX_INSTR_MEMSPACE],
                uInt               nrofInstrMemspace,
                ptxExpression     *arguments,
                ptxType           *instrType,
                uInt               nrofArguments,
                uInt               nrofInstrTypes,
                Bool               parsingMacro,
                msgSourcePos_t     sourcePos
            );

Bool areExtendedInstructionsEnabled(ptxParseData parseData);
#if     defined(__cplusplus)
}
#endif 

#endif
