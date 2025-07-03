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
 *  Module name              : ptxIR.c
 *
 *  Description              :
 *
 */

/*--------------------------------- Includes ---------------------------------*/

#include <g_lwconfig.h>

#include "ptx.h"
#include "ptxIR.h"
#include "ptxConstructors.h"
#include "ptxInstructions.h"
#include "ptxMacroUtils.h"
#include "ptxparseMessageDefs.h"
#include "ctMessages.h"
#include "stdUtils.h"
#include "ptxPragmaUtils.h"
#include "stdVector.h"
#include "stdString.h"
#include "ctLog.h"
#include "interfaceUtils.h"
#include "DebugInfo.h"

// should be sorted in ascending order for binary search
static const int ptxIsaVersions[] = {
      10 
    , 11
    , 12
    , 13
    , 14
    , 15
    , 20
    , 21
    , 22
    , 23
    , 30
    , 31
    , 32
    , 40
    , 41
    , 42
    , 43
    , 50
    , 51
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_60)
    , 60
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_61)
    , 61
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_62)
    , 62
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_63)
    , 63
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_64)
    , 64
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_65)
    , 65
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_70)
    , 70
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_71)
    , 71
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_72)
    , 72
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_73)
    , 73
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_74)
    , 74
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_75)
    , 75
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_76)
    , 76
#endif
};

int ptxGetLatestMajorVersion(void)
{
    int numVersions = sizeof(ptxIsaVersions) / sizeof(ptxIsaVersions[0]);
    return ptxIsaVersions[numVersions - 1] / 10;
}

int ptxGetLatestMinorVersion(void)
{
    int numVersions = sizeof(ptxIsaVersions) / sizeof(ptxIsaVersions[0]);
    return ptxIsaVersions[numVersions - 1] % 10;
}

static int comparePtxVersions(const void *first, const void *second)
{
    return (*(int*)first - *(int*)second);
}

Bool ptxIsSupportedIsaVersion(int version)
{
    int *ptr = (int *)bsearch(&version, ptxIsaVersions, sizeof(ptxIsaVersions) / sizeof(ptxIsaVersions[0]), sizeof(ptxIsaVersions[0]), comparePtxVersions);
    return ptr != NULL;
}

/*-------------------------------- Query --------------------------------*/

/*
* Function         : ptxDwarfGetSectionPointer 
* Parameters       : ptxparsingstate and DwarfSectionType
* Comments         : Look in secList in ptxParsingState and return pointer for matching input
*                    section name.
*/

ptxDwarfSection ptxDwarfGetSectionPointer(ptxParsingState p, DwarfSectionType sectionType)
{
    stdList_t dwarfSecList = p->dwarfSections;
    while (dwarfSecList) {
        ptxDwarfSection lwrPtxDwarfSection = (ptxDwarfSection) dwarfSecList->head;
        if (lwrPtxDwarfSection->sectionType == sectionType)
            return lwrPtxDwarfSection;
        dwarfSecList = dwarfSecList->tail;
    }
    return NULL;
}

/*
* Function         : ptxNeedVecModifier 
*                  : more than one argument of type vector.
* Parameters       : ptxInstruction
* Function Result  : True if the input instruction required vector modifier for its operands.
*/

Bool ptxNeedVecModifierForVecArgs(ptxInstruction instr)
{
    uInt tcode = instr->tmplate->code;
    switch(tcode){
    case ptx_ld_Instr:
    case ptx_ldu_Instr:
    case ptx_st_Instr:
        return True;
    default:
        return False;
    }
}

/*
 * Function         : Get argument number of vector type argument.
 * Parameters       : ptxInstruction
 * Function Result  : argument number
*/

uInt ptxGetVecArgNumber(ptxInstruction instr)
{
    uInt code = instr->tmplate->code;
    switch(code) { 
    case ptx_ld_Instr:
    case ptx_ldu_Instr:
        return 0;
    case ptx_st_Instr:
        return 1;
    default:
        stdASSERT(0, ("Unexpected instruction"));
        return ~0;
    }
}

/*
* Function         : Get the vector component from a vector 
* Parameters       : vecExp : ptxExpression which represents the vector
*                    index  : Index of the element to be obtained from the vector 
* Function Result  : Element of the vector specified by the index 
*/
ptxExpression ptxGetElementExprFromVectorExpr(ptxExpression vecExp, uInt index)
{
    stdList_t elems;
    uInt counter;
    stdASSERT(vecExp->kind == ptxVectorExpression, ("unexpected type"));
    
    for (elems = vecExp->cases.Vector.elements, counter = 0; elems != NULL && counter < index; elems = elems->tail, counter++);
    stdASSERT( elems != NULL, ("Trying to access invalid index"));
    
    return elems->head;
}

/*
* Function         : Get argument number of co-ordinate vector in texture/surface instruction.
* Parameters       : ptxInstruction code
* Function Result  : argument number
*/
int ptxTexSurfGetCoordArgNumber(ptxParsingState parseState, uInt code)
{
    stdASSERT( ptxIsTextureInstr(code) || ptxIsSurfaceInstr(code), ("texture/surface instruction expected") );

    if (ptxIsTextureInstr(code)){
        return ptxTexHasSampler(parseState) ? 3 : 2;
    } else if(code == ptx_sust_b_Instr || code == ptx_sust_p_Instr || code == ptx_sured_b_Instr || code == ptx_sured_p_Instr) {
        return 1;
    } else {
        return 2;
    }
}

/*
* Function         : Get argument number of texture component in the PTX instruction
* Parameters       : ptxInstruction  :  The instruction in which we are querying the positional arguement information
*                    component       :  The component for which the positional arguement number is to be determined 
* Function Result  : argument number
*/
int ptxTexComponentArgNumber(ptxParsingState parseState, ptxInstruction instr, ptxTexComponentType component)
{
    int pos = -1;
    int coordPos = ptxTexSurfGetCoordArgNumber(parseState, instr->tmplate->code);
    switch (component) {
    case TEX_COMPONENT_TYPE_TEXSAMP:
        return 1;                                              // Assumes only texture, Samplers will be (texturePosition + 1)
    case TEX_COMPONENT_TYPE_COORDS:
        return coordPos;
    case TEX_COMPONENT_TYPE_LODBIAS:
        pos =  (instr->tmplate->code == ptx_tex_level_Instr) ? (coordPos + 1) : -1;
        return pos;
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    case TEX_COMPONENT_TYPE_LODCLAMP:
        if (instr->tmplate->code == ptx_tex_clamp_Instr) {
            pos = coordPos + 1;
        } else if (instr->tmplate->code == ptx_tex_grad_clamp_Instr) {
            pos = coordPos + 3;
        }
        return pos;
#endif
    case TEX_COMPONENT_TYPE_MULTISAMPLE:
        return (ptxIsTextureInstrUsesMultiSampleModifier(instr->modifiers)) ? coordPos : -1;
    case TEX_COMPONENT_TYPE_ARRAYINDEX:
        return (ptxIsTexSurfInstrUsesArrayModifier(instr->modifiers)) ? coordPos : -1;
    case TEX_COMPONENT_TYPE_AOFFSET:
        return (ptxTexHasOffsetArg(parseState, instr) ?  ptxTexGetMinNumberOfArgs(parseState, instr->tmplate->code) : -1);
    case TEX_COMPONENT_TYPE_DEPTH:
        // As depth compare argument will always be the last argument 
        return (ptxTexHasDepthCompareArg(parseState, instr) ? (instr->tmplate->nrofArguments - 1) : -1);
    case TEX_COMPONENT_TYPE_DSDX:
    case TEX_COMPONENT_TYPE_DTDX:
    case TEX_COMPONENT_TYPE_DRDX:
        pos = ptxIsTexGradInstr(instr->tmplate->code) ? (coordPos + 1) : -1;
        return pos;
    case TEX_COMPONENT_TYPE_DSDY:
    case TEX_COMPONENT_TYPE_DTDY:
    case TEX_COMPONENT_TYPE_DRDY:
        pos = ptxIsTexGradInstr(instr->tmplate->code) ? (coordPos + 2) : -1;
        return pos;
    case TEX_COMPONENT_TYPE_GRANULARITY:
        return (ptxHasFOOTPRINT_MOD(instr->modifiers) ?  ptxTexGetMinNumberOfArgs(parseState, instr->tmplate->code) : -1);
    default:
        return -1;
    }
}

/*
* Function         : Get minimum number of arguments in texture/surface instruction
* Parameters       : ptxInstruction code
* Function Result  : minimum number of arguments based on independent/unified mode
*/

uInt ptxTexGetMinNumberOfArgs(ptxParsingState parseState, uInt code)
{
    int texCordPos = ptxTexSurfGetCoordArgNumber(parseState, code);
    stdASSERT( ptxIsTextureInstr(code) && texCordPos >= 0, ("texture instruction expected") );

    switch (code){
    case ptx_tex_grad_Instr:
        return texCordPos + 3;

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    case ptx_tex_grad_clamp_Instr:
        return texCordPos + 4;
#endif

    case ptx_tex_level_Instr:
        return texCordPos + 2;

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    case ptx_tex_clamp_Instr:
        return texCordPos + 2;
#endif

    default:
        return texCordPos + 1;
    }
}

/*
* Function         : Get argument number of symbol representing surface in surface instruction.
* Parameters       : ptxInstruction
* Function Result  : argument number
*/
int ptxSurfGetSurfaceSymbolArgNumber(ptxInstruction instr)
{
    switch (instr->tmplate->code) {
    case ptx_suld_b_Instr:
    case ptx__sulea_p_Instr:
    case ptx__sulea_b_Instr:
    case ptx_suq_Instr:
        return 1;
    case ptx_sust_b_Instr:
    case ptx_sust_p_Instr:
    case ptx_sured_b_Instr:
    case ptx_sured_p_Instr:
        return 0;
    default:
        stdASSERT(False, ("Unexpected instruction"));
        return -1;
    }
}

/*
 * Function         : ptxGetSymEntFromExpr
 * Parameters       : ptxExpression
 * Function Result  : Returns symbol table entry for the given ptxExpression
 */
ptxSymbolTableEntry ptxGetSymEntFromExpr(ptxExpression expr)
{
    ptxSymbolTableEntry symbolTabEntry;

    stdASSERT(expr, ("Invalid addr expression"));
    switch (expr->kind) {
    case ptxBinaryExpression:
        symbolTabEntry = ptxGetSymEntFromExpr(expr->cases.Binary->left);
        if (symbolTabEntry) {
            return symbolTabEntry;
        } else {
            return ptxGetSymEntFromExpr(expr->cases.Binary->right);
        }                                                                        
    case ptxUnaryExpression:
        return ptxGetSymEntFromExpr(expr->cases.Unary->arg);
    case ptxAddressOfExpression:
        return ptxGetSymEntFromExpr(expr->cases.AddressOf.lhs);  
    case ptxAddressRefExpression: 
        return ptxGetSymEntFromExpr(expr->cases.AddressRef.arg); 
    case ptxSymbolExpression:
        return expr->cases.Symbol.symbol;       
    case ptxVideoSelectExpression:
        return expr->cases.VideoSelect->arg->cases.Symbol.symbol;
    case ptxByteSelectExpression:
        return expr->cases.ByteSelect->arg->cases.Symbol.symbol;
    case ptxVectorSelectExpression:
        return expr->cases.VectorSelect->arg->cases.Symbol.symbol;
    case ptxIntConstantExpression:
    case ptxFloatConstantExpression:
    case ptxSinkExpression:
    case ptxLabelReferenceExpression:
        return NULL;
    case ptxArrayIndexExpression:
        return ptxGetSymEntFromExpr(expr->cases.ArrayIndex->arg);
    default:
        stdASSERT(0, ("Unexpected address expression"));
        return NULL;
    }
}

/*
 * Function         : ptxGetAddressArgBase
 * Parameters       : ptxExpression
 * Function Result  : Returns the base expression incase of AddressOf or AddressRef expression
 *                    Else returns the same expression
 */
ptxExpression ptxGetAddressArgBase(ptxExpression arg)
{
    switch (arg->kind) {
    case ptxAddressOfExpression:    return (ptxGetAddressArgBase(arg->cases.AddressOf.lhs));
    case ptxAddressRefExpression:   return (ptxGetAddressArgBase(arg->cases.AddressRef.arg));
    default:                        return arg;
    }
}

/*
 * Function         : ptxGetAddressArgBaseType
 * Parameters       : ptxType
 * Function Result  : Returns the base expression type incase of AddressOf or AddressRef expression
 *                    Else type of the input expression
 */
ptxType ptxGetAddressArgBaseType(ptxExpression arg)
{
    return ptxGetAddressArgBase(arg)->type;
}


/*
 * Function         : ptxGetAddrOperandPos
 * Parameters       : instruction opcode
 * Function Result  : Returns position of address operand (counting from 0) in Instruction
 *                    Returns -1 for instructiosn which don't have memory operand
 */
int ptxGetAddrOperandPos(uInt code)
{
    switch (code) {
    case ptx_ld_Instr:
    case ptx_ldu_Instr:
    case ptx_mov_Instr:
    case ptx_cvta_Instr:
    case ptx_cvta_to_Instr:
    case ptx_atom_Instr:
#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_63)
    case ptx__ldsm_Instr:
#endif
        return 1;
    case ptx_st_Instr:
    case ptx_red_Instr:    
    case ptx_prefetch_Instr:
    case ptx_prefetchu_Instr:
#if LWCFG(GLOBAL_ARCH_AMPERE)
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL) || LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_71)
    case ptx_cachepolicy_Instr:
#endif // internal || ISA_71
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    case ptx_destroy_Instr:
#endif // internal
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_74)
    case ptx_discard_Instr:
    case ptx_applypriority_Instr:
#endif // ISA_74
#endif // ampere
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_PTX_ISA_FUTURE)
    case ptx_stmatrix_Instr:
#endif
        return 0;
    }
    return -1;
}

#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
/*
Functions for printing macro expansions

Document for more details - https://drive.google.com/open?id=1IHfzb-nwo7WmsWFjx8XpslPHKw1MY4cG
Presentation link - https://drive.google.com/open?id=1l-RVWmO7DGI_1Z-E7dij8ZeZF_iVe9ERleQvNzkUNxY
*/
/*
 * Function         : appendTagToMacro
 * Function Result  : Append begin-end tags to macro expansion
 */
static String appendTagToMacro( String expansion, Bool isUserPTX, int *tagLength, printMacroExpansionInfoRec *expansionInfo,
                                ptxParsingState parseState)
{
    char beginTag[1000], endTag[1000];
    stdString_t expansionWithTag = stringCreate(8);
    if (strlen(expansionInfo->insName) > 900) {
        expansionInfo->insName[900] = '\0';
        stdCHECK_WITH_POS( False, (ptxMacroExpansionTagLength, ptxsLwrPos(parseState)) );
    }
    if (isUserPTX) {
        sprintf(beginTag,"\n        /****  Macro instruction - %s (%s:%d) begin  ****/\n",
                expansionInfo->insName, parseState->parseData->ptxfilename, ptxget_lineno(parseState->scanner));
        sprintf(endTag,"\n        /****  Macro instruction - %s (%s:%d) end  ****/\n",
                expansionInfo->insName, parseState->parseData->ptxfilename, ptxget_lineno(parseState->scanner));
    } else {
        sprintf(beginTag,"\n\n        /****  Expansion of %s begin  ****/\n",
                expansionInfo->insName);
        sprintf(endTag,"\n        /****  Expansion of %s end  ****/\n\n",
                expansionInfo->insName);
    }
    if (expansionInfo->expandNestedMacro || isUserPTX) {
        *tagLength = strlen(beginTag);
        stringAddBuf(expansionWithTag, beginTag);
        stringAddBuf(expansionWithTag, expansion);
        stringAddBuf(expansionWithTag, endTag);
    } else {
        *tagLength = 0;
        stringAddBuf(expansionWithTag, expansion);
    }
    return(stringStripToBuf(expansionWithTag));
}

/*
 * Function         : setNestedExpansion
 * Function Result  : Sets expandNestedMacro as True for debug level 2
 */
static void setNestedExpansion( printMacroExpansionInfoRec *expansionInfo )
{
    expansionInfo->expandNestedMacro = True;
}

/*
 * Function         : initializeUserMacro
 * Function Result  : Initialize required fields for printing expansions for
 *                    each user written macro instruction (UserPTX)
 */
static String initializeUserMacro( String expansion, int *orgloc, int *tagLength, printMacroExpansionInfoRec *expansionInfo,
                                   ptxParsingState parseState)
{
    // initialize required fields
    expansionInfo->locationStackTop = 0;
    expansionInfo->expansionLength = 0;
    expansionInfo->expansionLengthStackTop = 0;
    expansionInfo->prev_macro_stack_ptr = 0;
    *orgloc = 0;
    expansion = appendTagToMacro(expansion, True, tagLength, expansionInfo, parseState);
    return(expansion);
}

/*
 * Function         : initializeForFirstMacro
 * Function Result  : Initialize structure first time
 */
static void initializeForFirstMacro( printMacroExpansionInfoRec *expansionInfo )
{
    expansionInfo->expandNestedMacro = False;
    CT_DEBUG_DO("macro_expansion", 2, setNestedExpansion(expansionInfo););
    stdXArrayInit(expansionInfo->locationStack);
    stdXArrayInit(expansionInfo->expansionLengthStack);
    stdXArrayInit(expansionInfo->parentStack);
    expansionInfo->nofExpansion = 0;
    expansionInfo->expansionStack = vectorCreate(8);
}

/*
 * Function         : storeExpansionDetails
 * Function Result  : Store expansion location and length
 */
static void storeExpansionDetails( int originalLocation, int lwrExpLen, int macro_stack_ptr, printMacroExpansionInfoRec *expansionInfo )
{
    stdXArrayAssign(expansionInfo->locationStack, expansionInfo->locationStackTop,
                    (originalLocation + expansionInfo->expansionLength));
    expansionInfo->locationStackTop++;
    stdXArrayAssign(expansionInfo->expansionLengthStack,
                    expansionInfo->expansionLengthStackTop, lwrExpLen);
    expansionInfo->expansionLengthStackTop++;
    expansionInfo->prev_macro_stack_ptr = macro_stack_ptr;
}

/*
 * Function         : setLwrrentExpansion
 * Function Result  : Returns string to store in expansionStack
 */
static String setLwrrentExpansion( String expansion, int macro_stack_ptr, int *originalLocation, int *tagLength,
                                  int parentLength, printMacroExpansionInfoRec *expansionInfo,
                                  ptxParsingState parseState)
{
    String macrolabel, lwrExp;
    // for debug level one just add label instead of expansion of nested macro
    if (!expansionInfo->expandNestedMacro && macro_stack_ptr == 2) {
        if (stdIS_SUFFIX(" (FORCE INLINE FUNCTION)", expansionInfo->insName)) {
            macrolabel = "    // FORCE INLINE Function\n";
        } else {
            macrolabel = "    // Macro Instruction\n";
        }
        lwrExp =  stdCOPYSTRING(macrolabel);
    } else {
        lwrExp = stdCOPYSTRING(expansion);
    }

    if (macro_stack_ptr == 1) {
        // required initialization for user written macro
        lwrExp = initializeUserMacro(lwrExp, originalLocation, tagLength, expansionInfo,
                                     parseState);
        stdXArrayAssign(expansionInfo->parentStack, 0, *tagLength);
    } else {
        lwrExp = appendTagToMacro(lwrExp, False, tagLength, expansionInfo, parseState);
        // find location of nested expansion wrt to parent macro
        *originalLocation = (expansionInfo->parentStack[macro_stack_ptr - 2] +
                            parentLength);
    }
    return lwrExp;
}

/*
 * Function         : generateMacroExpansionDetails
 * Function Result  : Process expansion of macro instruction and generate expansion details
 */
void generateMacroExpansionDetails( String expansion, int macro_stack_ptr, String parentStr, ptxParsingState parseState )
{
    int originalLocation, tagLength, expLengthInd;
    String lwrExp, parent;
    printMacroExpansionInfoRec *expansionInfo = &(parseState->printMacroExpansionInfo);

    // initialize printMacroExpansionInfo for first macro
    if (expansionInfo->nofExpansion == -1) {
        initializeForFirstMacro(expansionInfo);
    }
    // for debug level one don't process 3rd and above nesting level
    if (!expansionInfo->expandNestedMacro && macro_stack_ptr > 2)
        return;

    if (parseState->lwrInstrSrc == MacroUtilFunction) {
        return;
    }
    parent = stdCOPYSTRING(parentStr);

    lwrExp = setLwrrentExpansion(expansion, macro_stack_ptr, &originalLocation,
                                 &tagLength, strlen(parent), expansionInfo,
                                 parseState);

    vectorAddTo(lwrExp, expansionInfo->expansionStack);

    if (expansionInfo->prev_macro_stack_ptr < macro_stack_ptr) {
        // storing length of parent macro instruction for nested macro
        if (macro_stack_ptr > 1) {
            stdXArrayAssign(expansionInfo->parentStack, macro_stack_ptr - 1,
                            (originalLocation + tagLength));
        }
    } else {
        // computing expansion length of previously completed macro expansions
        for (expLengthInd = 0; expLengthInd < (expansionInfo->prev_macro_stack_ptr - macro_stack_ptr + 1); expLengthInd++) {
            expansionInfo->expansionLength +=
                expansionInfo->expansionLengthStack[expansionInfo->expansionLengthStackTop - 1];
            expansionInfo->expansionLengthStackTop--;
        }
    }
    storeExpansionDetails(originalLocation, strlen(lwrExp), macro_stack_ptr, expansionInfo);
}

/*
 * Function         : isNestingPresent
 * Function Result  : Check if the expansion has nested macro instruction
 */
static Bool isNestingPresent( String exp, int *seenCharCount, int *lwrExpInd, printMacroExpansionInfoRec *expansionInfo )
{
    // check if no nesting for current expansion or all nested espansions are done
    if (*seenCharCount + strlen(exp) < expansionInfo->locationStack[*lwrExpInd]) {
        return False;
    }
    if (*lwrExpInd >= vectorSize(expansionInfo->expansionStack) - expansionInfo->nofExpansion) {
        return False;
    }
    return True;
}

/*
 * Function         : generateNestedMacroExpansion
 * Function Result  : Generates single string of expansion of userPTX (including nested
 *                    expansions) by processing nested macros and appeding it to final string
 */
static stdString_t generateNestedMacroExpansion( String exp, stdString_t finalExp, int *lwrExpInd,
                                                int *seenCharCount, printMacroExpansionInfoRec *expansionInfo )
{
    int len;
    *lwrExpInd += 1;
    while (isNestingPresent(exp, seenCharCount, lwrExpInd, expansionInfo)) {
        len = expansionInfo->locationStack[*lwrExpInd] - *seenCharCount;
        *seenCharCount += len;
        len++;
        String sub = stdCOPY_N(exp, len);        // copy extra one character for '\0' by len++
        sub[len - 1] = '\0';
        stringAddBuf(finalExp, sub);
        exp = exp + len - 1;              // move pointer by length (len - 1 due to len++)
        if (*lwrExpInd < vectorSize(expansionInfo->expansionStack) - expansionInfo->nofExpansion) {
            String expansion = (String)vectorIndex(expansionInfo->expansionStack,
                                            expansionInfo->nofExpansion + *lwrExpInd);
            finalExp = generateNestedMacroExpansion(expansion, finalExp, lwrExpInd, seenCharCount, expansionInfo);
        }
    }
    len = strlen(exp);
    stringAddBuf(finalExp, exp);
    *seenCharCount += len;
    return finalExp;
}

/*
 * Function         : storeFinalExpansionOfUserMacro
 * Function Result  : Set fields to generates single string of expansion and
 *                    cleanup unnecessary data from expansionStack
 */
void storeFinalExpansionOfUserMacro( ptxParsingState parseState )
{
    printMacroExpansionInfoRec *expansionInfo = &(parseState->printMacroExpansionInfo);
    if (parseState->lwrInstrSrc != UserPTX) {
        return;
    }
    if (expansionInfo->nofExpansion == -1) {
        return;
    }
    if (vectorSize(expansionInfo->expansionStack) == (uInt)expansionInfo->nofExpansion) {
        return;
    }
    int lwrExpInd = 0, seenCharCount = 0;
    String finalExp, expansion;
    expansion = (String)vectorIndex(expansionInfo->expansionStack, expansionInfo->nofExpansion);
    finalExp = stringStripToBuf(generateNestedMacroExpansion(expansion, stringCreate(8),
                                                            &lwrExpInd, &seenCharCount, expansionInfo));
    vectorSetElement(expansionInfo->expansionStack, expansionInfo->nofExpansion, finalExp);
    expansionInfo->nofExpansion++;
    // cleanup unnecessary data from expansionStack to match size with nofExpansion
    while (vectorSize(expansionInfo->expansionStack) > (uInt)expansionInfo->nofExpansion) {
        String dummy = vectorPop(expansionInfo->expansionStack);
        stdFREE(dummy);
    }
}

/*
 * Function         : setInstrName
 * Function Result  : Set insName as current instruction name
 */
static void setInstrName( String name, ptxParsingState parseState )
{
    printMacroExpansionInfoRec *expansionInfo = &(parseState->printMacroExpansionInfo);
    expansionInfo->insName = name;  //set instruction name
}

/*
 * Function         : blankoutComment
 * Function Result  : Remove comments from macro expansion if present
 */
static void blankoutComment( String str )
{
    // Lwrrently removal of comments in macro expansion is required because presence of comments
    // leads to incomplete data in macro_stack(ptx.l) which hampers logic of printing.

    // TODO: Fix the issue of incorrect data in macro_stack and print comments in macro expansion
    int ind = 0;
    while (str[ind] != '\0') {
        if (str[ind] == '/' && str[ind+1] == '/') {
            while (str[ind] != '\n' && str[ind] != '\0') {
                str[ind] = ' ';
                ind++;
            }
        }
        if (str[ind] == '/' && str[ind] == '*') {
            while (!(str[ind] == '*' && str[ind+1] == '/') && str[ind] != '\0') {
                str[ind] = ' ';
                ind++;
            }
            str[ind] = ' ';
            str[ind+1] = ' ';
        }
        ind++;
    }
}

/*
 * Function         : preProcessMacroForPrint
 * Function Result  : Remove comments from expansion and set insName
 */
void preProcessMacroForPrint( String str, String name, ptxParsingState parseState )
{
    setInstrName(name, parseState);
    blankoutComment(str);
}

/*
 * Function         : deallocateMacroExpansionInfo
 * Function Result  : Deallocates the memory allocated to expansionInfo
 */
static void deallocateMacroExpansionInfo( printMacroExpansionInfoRec *expansionInfo )
{
    if (expansionInfo->nofExpansion != -1) {
        vectorDelete(expansionInfo->expansionStack);
        stdFREE(expansionInfo->expansionLengthStack);
        stdFREE(expansionInfo->locationStack);
        stdFREE(expansionInfo->insName);
    }
}

/*
 * Function         : printExpansion
 * Function Result  : Process macro expansion if available and print all expansions
 */
void printExpansion( ptxParsingState parseState )
{
    int expInd;
    printMacroExpansionInfoRec *expansionInfo = &(parseState->printMacroExpansionInfo);
    for (expInd = 0; expInd < expansionInfo->nofExpansion; expInd++) {
        stdSYSLOG("%s",(String)vectorIndex(expansionInfo->expansionStack, expInd));
    }
    deallocateMacroExpansionInfo(expansionInfo);
}
#endif

/*-------------------------------- Instruction --------------------------------*/

/*
 * Function         : Check if instruction is bar or barrier instruction
 * Function Result  : True iff bar or barrier instruction
 */
Bool ptxIsBarOrBarrierInstr(uInt code)
{
    return (code == ptx_bar_Instr        ||
        code == ptx_bar_arrive_Instr     ||
        code == ptx_bar_red_Instr        ||
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        code == ptx_bar_scan_Instr       ||
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_60)
        code == ptx_bar_warp_Instr       ||
        code == ptx_barrier_Instr        ||
        code == ptx_barrier_arrive_Instr ||
        code == ptx_barrier_red_Instr    ||
#endif
        False                            );
}

/*
 * Function : Check if the instruction is either wmma.load or wmma.store instruction
 */
Bool ptxIsWMMALoadStore(uInt tcode)
{
    switch(tcode) {
    case ptx_wmma_load_a_Instr:
    case ptx_wmma_load_b_Instr:
    case ptx_wmma_load_c_Instr:
    case ptx_wmma_store_d_Instr:
        return True;
    default:
        return False;
    }
}

/*
 * Function : Check if the instruction is a wmma.* instruction
 */
Bool ptxIsWMMAInstr(uInt tcode)
{
    return ptxIsWMMALoadStore(tcode) || tcode == ptx_wmma_mma_Instr;
}

/*
 * Function         : Check Vector argument. This function is not meant to be used if there are 
 *                  : more than one argument of type vector.
 * Parameters       : ptxInstruction
 * Function Result  : True if the desired operand is of type vector else False.
 */

Bool ptxIsArgVecType(ptxInstruction instr)
{
    uInt code = instr->tmplate->code;
    if (ptxNeedVecModifierForVecArgs(instr)) {
        switch(code) { 
        case ptx_ld_Instr:
        case ptx_ldu_Instr:
            return (instr->arguments[0]->type->kind == ptxVectorType);
        case ptx_st_Instr:
            return (instr->arguments[1]->type->kind == ptxVectorType); 
        default:
            stdASSERT(0, ("Unexpected instruction"));
        }
    }
    return False;
}

/*
 * Function         : Check if tex,tld4 instruction has an explicit sampler
 * Function Result  : True iff texture instruction has an explicit sampler
 */

Bool ptxTexHasSampler(ptxParsingState parseState)
{
    return checkTargetOpts(parseState, "texmode_independent");
}



/*
 * Function         : Check if texture instruction has offset
 * Function Result  : True iff texture instruction has offset
 */

Bool ptxTexHasOffsetArg(ptxParsingState parseState, ptxInstruction instr)
{
    uInt code = instr->tmplate->code;
    uInt offsetPos = ptxTexGetMinNumberOfArgs(parseState, code);
    stdASSERT( ptxIsTextureInstr(code), ("texture instruction expected") );
 
    if (ptxHasFOOTPRINT_MOD(instr->modifiers)) {
        return False;
    }

    if (instr->tmplate->nrofArguments > offsetPos) {
        // As offset argument will be always be next to surface argument
        return instr->arguments[offsetPos]->type->kind == ptxVectorType;
    } else {
        return False;
    }
}

/*
 * Function         : Check if texture instruction has depth compare arg
 * Function Result  : True iff texture instruction has depth compare arg
 */
Bool ptxTexHasDepthCompareArg(ptxParsingState parseState, ptxInstruction instr)
{
    uInt code = instr->tmplate->code;
    uInt minArgs = ptxTexGetMinNumberOfArgs(parseState, code);
    ptxType type = instr->arguments[instr->tmplate->nrofArguments - 1]->type;
    ptxExpressionKind exprKind = instr->arguments[instr->tmplate->nrofArguments - 1]->kind;
    stdASSERT( ptxIsTextureInstr(code), ("texture instruction expected") );
 
    if (ptxHasFOOTPRINT_MOD(instr->modifiers)) {
        return False;
    }

    if (instr->tmplate->nrofArguments > minArgs) {
        // As depth compare argument will always be the last argument 
        // checking for last argument will be enough.
        return isF32(type) || isB32(type) || exprKind == ptxFloatConstantExpression;
    } else {
        return False;
    }
}

#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)

/*
 * Function         : Check if instruction is a TTU instruction
 * Function Result  : True iff TTU instruction
 */
Bool ptxIsTTUInstr(uInt code)
{
    return (code == ptx__ttuopen_Instr   ||
            code == ptx__ttust_Instr     ||
            code == ptx__ttuld_Instr     ||
            code == ptx__ttugo_Instr     ||
            code == ptx_ttucctl_Instr     );
}

/*
 * Function         : Check if instruction is any TTU instruction other than ttucctl
 * Function Result  : True iff any TTU instruction other than ttucctl
 */
Bool ptxIsTTUInstrExceptTTUCCL(uInt code)
{
    return ptxIsTTUInstr(code) && (code != ptx_ttucctl_Instr);
}
#endif

/*
 * Function         : Check if instruction is tex or tex.{base,level,grad} instruction
 * Function Result  : True iff texture instruction
 */
Bool ptxIsTexInstr(uInt code)
{
    return (code == ptx_tex_Instr          ||
            code == ptx_tex_base_Instr     ||
            code == ptx_tex_level_Instr    ||
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
            code == ptx_tex_clamp_Instr    ||
#endif
            ptxIsTexGradInstr(code));
}

/*
 * Function         : Check if instruction is tex.grad or tex.grad.clamp instruction
 * Function Result  : True iff texture instruction
 */
Bool ptxIsTexGradInstr(uInt code)
{
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    return (code == ptx_tex_grad_Instr || code == ptx_tex_grad_clamp_Instr);
#else
    return code == ptx_tex_grad_Instr;
#endif
}

/*
 * Function         : Check if instruction is a texture instruction (excludes txq)
 * Function Result  : True iff texture instruction
 */
Bool ptxIsTextureInstr(uInt code)
{
    return (ptxIsTexInstr(code) || code == ptx_tld4_Instr);
}

/*
 * Function         : Check if instruction is a texture query
 * Function Result  : True iff texture query instruction
 */
Bool ptxIsTxqInstr(uInt code)
{
    return code == ptx_txq_Instr || code == ptx_txq_level_Instr;
}

/* Function          : Check if the texture instruction uses indirect texture, sampler and surface 
 * Parameters        : instr  : Instruction in which access type of the opaque entities needs to be determined
 *                     entity : which among Textures, Sampler, Surfaces should be considered for indirect access checking  
 * Function Result   : True iff texture instruction has -
 *                                                    1. indirect texture when Texture entity is set
 *                                                    2. indirect sampler when Sampler entity is set
 *                                                    3. indirect surface when Surface entity is set
 *                                                    4. OR of any of the above
 */
Bool ptxTexSurfUsesIndirectAccess(ptxParsingState parseState, ptxInstruction instr, uInt entity)
{
    int pos = 0, code;
    Bool isTexRefIndirectAccess = False;
    ptxSymbolTableEntry texSymEnt = NULL, sampSymEnt = NULL, surfSymEnt = NULL;
    code = instr->tmplate->code;

    if (entity & TEXTURE_SYMBOL) {
        stdASSERT(ptxIsTextureInstr(code) || ptxIsTxqInstr(code), ("Unexpected instruction"));
        texSymEnt  = ptxGetSymEntFromExpr(instr->arguments[1]);
        isTexRefIndirectAccess |= !isTEXREF(texSymEnt->symbol->type);
    }
    if (entity & SAMPLER_SYMBOL) {
        if (ptxIsTextureInstr(code)) {
            stdASSERT(ptxTexHasSampler(parseState), ("Sampler not present"));
            sampSymEnt = ptxGetSymEntFromExpr(instr->arguments[2]);
        } else if (ptxIsTxqInstr(code)) {
            sampSymEnt = ptxGetSymEntFromExpr(instr->arguments[1]);
        } else {
            /* OPTIX_HAND_EDIT : sampSymEnt may appear to be uninitialized in release builds */
            sampSymEnt = 0;
            stdASSERT(0, ("Unexpected instruction"));
        }
        isTexRefIndirectAccess |= !isSAMPLERREF(sampSymEnt->symbol->type);
    }
    if (entity & SURFACE_SYMBOL) {
        stdASSERT((ptxIsSurfaceInstr(code) || code == ptx_suq_Instr), ("Unexpected instruction"));
        pos = ptxSurfGetSurfaceSymbolArgNumber(instr);
        surfSymEnt = ptxGetSymEntFromExpr(instr->arguments[pos]);
        isTexRefIndirectAccess |= !isSURFREF(surfSymEnt->symbol->type);
    }

    return isTexRefIndirectAccess;
}

/* Function          : Check if the texture/surface instruction uses Array modifier
 * Function Result   : True iff texture/surface instruction has A1D or A2D or ALWBE or A2DMS modifiers
 */
Bool ptxIsTexSurfInstrUsesArrayModifier( ptxModifier modifier )
{
    switch (modifier.TEXTURE) {
    case ptxA1D_MOD:
    case ptxA2D_MOD:
    case ptxALWBE_MOD:
    case ptxA2DMS_MOD:
        return True;
    default :
        return False;
    }
}

/* Function          : Check if the texture instruction uses multi-sample modifier
 * Function Result   : True iff texture instruction has 2DMS or A2DMS modifiers
 */
Bool ptxIsTextureInstrUsesMultiSampleModifier( ptxModifier modifier )
{
    switch (modifier.TEXTURE) {
    case ptx2DMS_MOD:
    case ptxA2DMS_MOD:
        return True;
    case ptx1D_MOD:
    case ptx2D_MOD:
    case ptx3D_MOD:
    case ptxLWBE_MOD:
    case ptxA1D_MOD:
    case ptxA2D_MOD:
    case ptxALWBE_MOD:
        return False;
    default :
        stdASSERT(False, ("Unexpected modifier"));
        return False;
    }
}

/*
 * ptxIsImmediateExpr() - Checks if the expression is an immediate constant literal
 *
 */
Bool ptxIsImmediateExpr(ptxExpression exp)
{
    return (exp->kind == ptxIntConstantExpression) || (exp->kind == ptxFloatConstantExpression);
}

/*
 * isSinkExpression() : Checks if input expression is sink ('_') expression
 */
Bool isSinkExpression(ptxExpression expr)
{
    return expr && expr->kind == ptxSinkExpression;
}

/*
 * ptxGetImmediateIntVal() - Returns the immediate integer value from IntConstant ptxExpression
 *
 */
Int64 ptxGetImmediateIntVal(ptxExpression exp)
{
    stdASSERT(exp->kind == ptxIntConstantExpression, ("unexpected kind"));
    return exp->cases.IntConstant.i;
}

/*
* Function         : Check whether the immediate arguement is a multiple of the specified value
* Parameters       : arguments : List of all the arguments of the current instuction
*                    argIdx    : The index of the immediate argument which is to be checked for the value multiple
*                    multiple  : The number which is suppose to divide the immediate argument
*                    name      : The name of the instruction to report error if imm. argument is not a mutliple
*                    sourcePos : The sourcePos of the instruction to report error if imm. argument is not a mutliple
*/
void ptxCheckImmediateArgForMultipleOf(ptxExpression *arguments, uInt argIdx,
                                       uInt multiple, String name,
                                       msgSourcePos_t sourcePos)
{
    uInt64 argVal = 0;
    stdASSERT (ptxIsImmediateExpr(arguments[argIdx]), ("immediate expected"));
    argVal = ptxGetImmediateIntVal(arguments[argIdx]);
    stdCHECK_WITH_POS( stdMULTIPLEOF(argVal, multiple),
                       (ptxMsgArgValueReqMultiple, sourcePos, argIdx,
                        name, argVal, multiple));
}

/*
 * Function         : Check if instruction is a surface instruction introduced in PTX ISA 1.5 (excludes suq)
 * Function Result  : True iff surface instruction defined in PTX 1.5
 */
Bool isPTX15SurfaceInstr( uInt code )
{
    return ( code == ptx_suld_b_Instr   ||
             code == ptx_sust_b_Instr   );
}

/*
 * Function         : Check if instruction is a surface instruction (excludes suq)
 * Function Result  : True iff surface instruction
 */
Bool ptxIsSurfaceInstr(uInt code)
{
    return ( code == ptx_suld_b_Instr   ||
             code == ptx_sust_b_Instr   ||
             code == ptx_sust_p_Instr   ||
             code == ptx_sured_b_Instr  ||
             code == ptx_sured_p_Instr  ||
             code == ptx__sulea_p_Instr ||
             code == ptx__sulea_b_Instr );
}
            
/*
 * Function         : Check if instruction is a scalar/simd2/simd4 video instruction
 * Function Result  : True iff video instruction
 */
Bool ptxIsVideoScalarInstr(uInt code)
{
    switch (code) {
    case ptx_vadd_Instr:
    case ptx_vsub_Instr:
    case ptx_vabsdiff_Instr:
    case ptx_vmin_Instr:
    case ptx_vmax_Instr:
    case ptx_vshl_Instr:
    case ptx_vshr_Instr:
    case ptx_vmad_Instr:
    case ptx_vset_Instr:
        return True;
    default:
        return False;
    }
}

Bool ptxIsVideoSIMD2Instr(uInt code)
{
    switch (code) {
    case ptx_vadd2_Instr:
    case ptx_vsub2_Instr:
    case ptx_vavrg2_Instr:
    case ptx_vabsdiff2_Instr:
    case ptx_vmin2_Instr:
    case ptx_vmax2_Instr:
    case ptx_vset2_Instr:
        return True;
    default:
        return False;
    }
}

Bool ptxIsVideoSIMD4Instr(uInt code)
{
    switch (code) {
    case ptx_vadd4_Instr:
    case ptx_vsub4_Instr:
    case ptx_vavrg4_Instr:
    case ptx_vabsdiff4_Instr:
    case ptx_vmin4_Instr:
    case ptx_vmax4_Instr:
    case ptx_vset4_Instr:
        return True;
    default:
        return False;
    }
}

Bool ptxIsVideoInstruction(uInt code)
{
    return ptxIsVideoScalarInstr(code) || ptxIsVideoSIMD2Instr(code) || ptxIsVideoSIMD4Instr(code);
}

uInt ptxGetVideoInstrSIMDWidth(uInt code)
{
    stdASSERT(ptxIsVideoInstruction(code), ("Video instruction expected\n"));

    if (ptxIsVideoScalarInstr(code)) { 
        return 1;
    } else if (ptxIsVideoSIMD2Instr(code)) {
        return 2;
    } else if (ptxIsVideoSIMD4Instr(code)) {
        return 4;
    } else {
        stdASSERT(0, ("Unexpected video instruction type\n"));
        return 0;
    }
}

/*
 * Function         : Check if instruction supports f16 arithmetic
 * Function Result  : True iff instruction opcode supports f16 type
 */
Bool ptxIsF16ArithmeticInstr(uInt code)
{
    switch (code) {
    case ptx_add_Instr:
    case ptx_sub_Instr:
    case ptx_mul_Instr:
    case ptx_fma_Instr:
    case ptx_neg_Instr:
    case ptx_min_Instr:
    case ptx_max_Instr:
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_65)
    case ptx_abs_Instr:
#endif
        return True;
    default:
        return False;
    }
}

/*
 * Function         : Check if instruction supports f16 comparison
 * Function Result  : True iff instruction opcode supports f16 type
 */
Bool ptxIsF16CompareInstr(uInt code)
{
    switch (code) {
    case ptx_set_Instr:
    case ptx_setp_Instr:
        return True;
    default:
        return False;
    }
}

Bool isDenseIMMAWithExplicitTypes(uInt shape) {
    switch (shape) {
    case ptxSHAPE_080816_MOD:
    case ptxSHAPE_080832_MOD:
        return False;
#if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_70)
    case ptxSHAPE_160816_MOD:
    case ptxSHAPE_160832_MOD:
    case ptxSHAPE_160864_MOD:
        return True;
#endif
    default:
        stdASSERT(False, ("unexpected shape for integer _mma"));
        // To avoid warning/error
        return False;
    }
}

/*
 * Function         : Check if instruction supports memory descriptor
 * Function Result  : True iff instruction opcode supports memory descriptor
 */
Bool ptxInstrSupportsMemDesc(uInt code)
{
#if LWCFG(GLOBAL_ARCH_AMPERE) && (LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL) || LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_71))
    Bool isMemDescSupported = False;
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    isMemDescSupported = code == ptx_wmma_load_a_Instr
        || code == ptx_wmma_load_b_Instr
        || code == ptx_wmma_load_c_Instr
        || code == ptx_wmma_store_d_Instr;
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_71)
    isMemDescSupported |= code == ptx_cp_async_Instr
        || code == ptx_ld_Instr
        || code == ptx_st_Instr
        || code == ptx_atom_Instr
        || code == ptx_red_Instr;
#endif
    return isMemDescSupported;
#endif
}

/*-------------------------------- Modifiers ---------------------------------*/

/*
 * Function         : Find the vector size of a modifier
 * Function Result  : vector size of a modifier
 */
uInt ptxGetVectorSize( ptxModifier modifier )
{
    switch (modifier.VECTOR) {
    case ptxV2_MOD : return 2;
    case ptxV4_MOD : return 4;
    case ptxV8_MOD : return 8;
    default        : return 1;
    }
}

/*
 * Function         : Find the vector length of an expression
 * Function Result  : vector length of the expression 'arg'
 */
uInt ptxGetArgVectorLength( ptxExpression arg )
{
    if (arg->kind != ptxVectorExpression)
        return 1;

    return arg->type->cases.Vector.N;
}

/*
 * Function         : This logically represents number of the components needed to pin-point the texel being referred to.
 * Function Result  : Number of entities needed to pin-point the texel being refered to in the instruction 
 */
uInt ptxGetTexSurfNumPosComponents( ptxModifier modifier )
{
    switch (modifier.TEXTURE) {
    case ptx1D_MOD    : return 1;
    case ptx2D_MOD    : return 2;
    case ptx3D_MOD    : return 3;
    case ptxA1D_MOD   : return 2;
    case ptxA2D_MOD   : return 3;
    case ptxLWBE_MOD  : return 3;
    case ptxALWBE_MOD : return 4;
    case ptx2DMS_MOD  : return 3;
    case ptxA2DMS_MOD : return 4;
    default           : return 1;  // default is 1d (e.g. for _suld/_sust)
    }
}

/*
 * Function         : Find the dimension of base texture or surface
 *                    For array textures, dimension of texture refers to dimension of the texture element in the array
 * Function Result  : dimension of a texture/surface
 */
uInt ptxGetTextureDim( ptxModifier modifier )
{
    switch (modifier.TEXTURE) {
    case ptx1D_MOD    :
    case ptxA1D_MOD   :
        return 1;
    case ptx2D_MOD    :
    case ptxA2D_MOD   :
    case ptx2DMS_MOD  :
    case ptxA2DMS_MOD :
        return 2;
    case ptx3D_MOD    :
    case ptxLWBE_MOD  :
    case ptxALWBE_MOD :
        return 3;
    default           : 
        return 1;  // default is 1d (e.g. for _suld/_sust)
    }
}

/*
 * Function         : Check if a modifier has a floating point round
 * Function Result  : True iff floating point round
 */
Bool ptxHasRoundFModifier( ptxModifier modifier)
{
    switch (modifier.ROUND) {
    case ptxRN_MOD  :
    case ptxRM_MOD  :
    case ptxRP_MOD  :
    case ptxRZ_MOD  : return True;
    default         : return False;
    }
}

/*
 * Function         : Check if a modifier has a integer round
 * Function Result  : True iff floating point round
 */
Bool ptxHasRoundIModifier( ptxModifier modifier )
{
    switch (modifier.ROUND) {
    case ptxRNI_MOD :
    case ptxRMI_MOD :
    case ptxRPI_MOD :
    case ptxRZI_MOD : return True;
    default         : return False;
    }
}

/*
 * Function         : Check if a modifier is a texture query other than the queries in common with surfaces
 * Function Result  : True iff normalized_coords or has a surface query modifier (subset of tex query mods)
 */
Bool ptxHasTexQueryModifier( ptxModifier modifier )
{
    switch (modifier.QUERY) {
    case ptxQUERY_NORM_MOD     : return True;
    case ptxQUERY_ARRSIZE_MOD  : return True;
    case ptxQUERY_MIPLEVEL_MOD : return True;
    case ptxQUERY_SAMPLES_MOD  : return True;
    default                    : return ptxHasSurfQueryModifier(modifier);  // All surf query mods are also tex query mods
    }
}

/*
 * Function         : Check if a modifier is for sampler property
 * Function Result  : True iff filter mode, addressing mode, or unnormalized query
 */
Bool ptxHasSamplerQueryModifier( ptxModifier modifier )
{
    switch (modifier.QUERY) {
    case ptxQUERY_FILTER_MOD  :
    case ptxQUERY_ADDR0_MOD   :
    case ptxQUERY_ADDR1_MOD   :
    case ptxQUERY_ADDR2_MOD   :
    case ptxQUERY_UNNORM_MOD  : return True;
    default                   : return False;
    }
}

/*
 * Function         : Check if a modifier is a surface query
 * Function Result  : True iff width, height, depth, channel_type, or channel_order query
 */
Bool ptxHasSurfQueryModifier( ptxModifier modifier )
{
    switch (modifier.QUERY) {
    case ptxQUERY_WIDTH_MOD   :
    case ptxQUERY_HEIGHT_MOD  :
    case ptxQUERY_DEPTH_MOD   :
    case ptxQUERY_CHTYPE_MOD  :
    case ptxQUERY_CHORDER_MOD :
    case ptxQUERY_ARRSIZE_MOD : return True;
    default                   : return False;
    }
}

// Get index of memspace corresponding to argument
int ptxGetStorageIndex(uInt tcode, uInt argId, uInt nrofInstrMemspace)
{

    if (nrofInstrMemspace < 1) {
        // We already have reported error in case of incorrect memspace
        return 0;
    }

    // Note: If more than one memspace is specified then the assumption is that
    // the memspaces correspond to arguments in order of specificaton.
    if (argId <= (nrofInstrMemspace - 1)) {
        return argId;
    }

    switch (tcode) {
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    case ptx_cp_async_bulk_tensor_Instr:
    case ptx_cp_reduce_async_bulk_tensor_Instr:
    // For cp{.reduce}.async.bulk.tensor.global.shared instruction 2nd argument corresponds
    // to .shared memspace
        switch (argId) {
        case 2: return 1;
        default:
            return 0;
        }
        break;
#endif
    default:
        return 0;
    }
}

/*
 * Function : Check if cp.async instruction has src-size operand
 */
Bool isSrcSizePresentForCopyInstruction(uInt tcode, ptxExpression *args, uInt nargs, Bool hasCacheHint)
{
#if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_70)
    if (tcode != ptx_cp_async_Instr) {
        return False;
    }
    if (nargs == 5) {
         return (!isPRED(args[3]->type));
    }
    if (nargs == 4) {
         return (!hasCacheHint && !isPRED(args[3]->type));
    }
#endif
    return False;
}

/*
 * Function : Return ignore-src argument for cp.async
 */
ptxExpression getIgnoreSrcArgForCopyInstr(uInt tcode, ptxExpression *args, uInt nargs)
{
#if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_75)
    if (tcode != ptx_cp_async_Instr) {
        return NULL;
    }
    if (nargs == 5 || nargs == 4) {
        if (isPRED(args[3]->type)) {
            return args[3];
        }
        return NULL;
    }
#endif
    return NULL;
}

/*
 * Function : Check if cp.async instruction has ignore-src operand
 */
Bool isIgnoreSrcPresentForCopyInstruction(uInt tcode, ptxExpression *args, uInt nargs)
{
#if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_75)
    if (tcode != ptx_cp_async_Instr) {
        return False;
    }
    if (getIgnoreSrcArgForCopyInstr(tcode, args, nargs)) return True;
#endif
    return False;
}

/*
 * Function : Check if instruction is 256 bit wide load-store
 *            ld.global.v8.type
 *            st.global.v8.type
 *
 *            type = {.b32, .f32, .u32, s32}
 */
#if LWCFG(GLOBAL_ARCH_ADA) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
Bool isWideLoadStoreInstr(uInt tcode, ptxType type, ptxModifier modifiers)
{
    if (tcode != ptx_ld_Instr && tcode != ptx_st_Instr) {
        return False;
    }

    return ptxGetVectorSize(modifiers) == 8 && (isB32(type) || isI32(type) || isF32(type));
}
#endif // ada, internal

/*
 * Function : Check if instruction has implicit memspace
 */
Bool ptxInstrHasImplicitMemSpace(uInt tcode)
{
    switch (tcode) {
#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_63)
    case ptx__ldsm_Instr:
        return True;
#endif
    default:
        return False;
    }
}

// Here we need 64 bit integer as a key in Map, But stdMap Key is type of
// 'Pointer', So prevents us to store integer directly on 32 bit system.
// Hence, here we dynamically allocate 64 bit number and use its address
// as a key
Pointer ptxCreateKeyFromLoc(Int fileId, Int line, Int linePos)
{
    Int64 *key;
    stdNEW(key)

    // NOTE: In current implementation 16 bits are allocated to filePos and linePos
    stdASSERT((fileId < (1 << 16) && linePos < (1 << 16)),
            ("File Index/column Pos is out of range"));
    *key = ((Int64) fileId << 48) + ((Int64) linePos << 32) + line;
    return key;
}
/*---------------------------- Type Compatibility ----------------------------*/


/*
 * Function         : Maximize information on types, and return whether type components were compatible
 * Function Result  : True iff compatible
 */
Bool ptxMaximizeType( ptxTypeKind *lkind, uInt64 *lsize,
                      ptxTypeKind  rkind, uInt64  rsize, Bool  rConst )
{
        if (!rConst && *lsize != rsize) {
            return False;
        } else {
            *lsize= stdMAX(*lsize,rsize);
        
                if (isBitTypeKind(*lkind)) {
                    *lkind= rkind;
                    return True;
                } else if (isBitTypeKind(rkind)) {
                    ; // BitType compatible with all types, so *lkind unchanged
                    return True;
                } else {
                    return *lkind == rkind || (isIntegerKind(*lkind) && isIntegerKind(rkind))
                                           || (  isFloatKind(*lkind) && isFloatKind(rkind));
                }
    }
}

/*
 * Function         : Decide if expression can be assigned to location of specified type.
 * Parameters       : l   (I) Type of location to assign to
 *                    r   (I) Type of expression to assign
 *                    rConst (I) True iff rhs expression is constant expression
 * Function Result  : True iff castable
 */
Bool ptxAssignmentCompatible( ptxType l, ptxType r, Bool rConst)
{
    if (l==r) { 
        return True; 
    } else {
    
                ptxTypeKind lkind= l->kind;
                ptxTypeKind rkind= r->kind;

                if (lkind == ptxVectorType) {
                        if (rkind != lkind) { return False; }
                        return l->cases.Vector.N == r->cases.Vector.N
                                && ptxAssignmentCompatible( l->cases.Vector.base, r->cases.Vector.base,
                                                            rConst);
                } else {

                        uInt64 lsize= ptxGetTypeSizeInBits(l);
                        uInt64 rsize= ptxGetTypeSizeInBits(r);

                        return ptxMaximizeType( &lkind, &lsize, rkind, rsize, rConst);
            }
        }
}

/*----------------------------- Unique Type Table ----------------------------*/

    stdUNUSED(static uInt32 storageHash( ptxStorageClass storage ));
    static uInt32 storageHash( ptxStorageClass storage )
    { return storage.kind ^ storage.bank; }

    stdUNUSED(static Bool storageEqual( ptxStorageClass l, ptxStorageClass r ));
    static Bool storageEqual( ptxStorageClass l, ptxStorageClass r )
    { return l.kind==r.kind && l.bank==r.bank; }

    static void hashSymbol( ptxSymbol symbol, uInt32 *hash )
    {
       *hash ^= stdAddressHash(symbol->type)
              ^ stdStringHash (symbol->unMangledName);
    }

    static Bool symbolEqual( ptxSymbol l, ptxSymbol r )
    {
        return stdEQSTRING(l->unMangledName,   r->unMangledName)
            &&            (l->type == r->type);
    }

    static Bool symbolsEqual( stdList_t l, stdList_t r )
    {
        if (l==NULL) { return r==NULL; }
        if (r==NULL) { return l==NULL; }
        
        return symbolEqual (l->head,r->head)
            && symbolsEqual(l->tail,r->tail);
    }

    static uInt32 fieldsHash( stdList_t fields )
    {
        uInt32 result= 0;
        listTraverse(fields, (stdEltFun)hashSymbol, &result );
        return result;
    }

static Bool TypeEqual( ptxType l, ptxType r )
{
    if (l->kind    != r->kind   ) { return False; }

    switch (l->kind) {
    case ptxLabelType           : return True;
    case ptxMacroType           : return True;
    case ptxTypePred            : return True;
    case ptxConditionCodeType   : return True;
    case ptxParamListType       : return True;
    case ptxTypeE4M3            :
    case ptxTypeE5M2            :
    case ptxTypeE4M3x2          :
    case ptxTypeE5M2x2          :
    case ptxTypeF16             :
    case ptxTypeF32             :
    case ptxTypeF64             :
    case ptxTypeF16x2           :
    case ptxTypeBF16            :
    case ptxTypeBF16x2          :
    case ptxTypeTF32            :
    case ptxTypeB1              :
    case ptxTypeB2              :
    case ptxTypeB4              :
    case ptxTypeB8              :
    case ptxTypeB16             :
    case ptxTypeB32             :
    case ptxTypeB64             :
    case ptxTypeB128            :
    case ptxTypeU2              :
    case ptxTypeU4              :
    case ptxTypeU8              :
    case ptxTypeU16             :
    case ptxTypeU32             :
    case ptxTypeU64             :
    case ptxTypeS2              :
    case ptxTypeS4              :
    case ptxTypeS8              :
    case ptxTypeS16             :
    case ptxTypeS32             :
    case ptxTypeS64             : return True;
    case ptxOpaqueType          : return stdEQSTRING(l->cases.Opaque.name, r->cases.Opaque.name) && symbolsEqual( l->cases.Opaque.fields, r->cases.Opaque.fields ); 
    case ptxVectorType          : return l->cases.Vector         .base == r->cases.Vector         .base  &&  l->cases.Vector .N == r->cases.Vector .N;  
    case ptxArrayType           : return l->cases.Array          .base == r->cases.Array          .base  &&  l->cases.Array  .N == r->cases.Array  .N; 
    case ptxIncompleteArrayType : return l->cases.IncompleteArray.base == r->cases.IncompleteArray.base; 
    default                     : stdASSERT( False, ("Case label out of bounds") );
    }
    
    return False;
}

static uInt64 TypeHash( ptxType t )
{
    switch (t->kind) {
    case ptxLabelType           : return t->kind;
    case ptxMacroType           : return t->kind;
    case ptxTypePred            : return t->kind;
    case ptxConditionCodeType   : return t->kind;
    case ptxParamListType       : return t->kind;
    case ptxTypeB1              :
    case ptxTypeB2              :
    case ptxTypeB4              :
    case ptxTypeB8              :
    case ptxTypeB16             :
    case ptxTypeB32             :
    case ptxTypeB64             :
    case ptxTypeB128            :
    case ptxTypeE4M3            :
    case ptxTypeE5M2            :
    case ptxTypeE4M3x2          :
    case ptxTypeE5M2x2          :
    case ptxTypeF16             :
    case ptxTypeBF16            :
    case ptxTypeBF16x2          :
    case ptxTypeTF32            :
    case ptxTypeF32             :
    case ptxTypeF64             :
    case ptxTypeF16x2           : return t->kind ^ ptxGetTypeSizeInBits(t);
    case ptxTypeU2              :
    case ptxTypeU4              :
    case ptxTypeU8              :
    case ptxTypeU16             :
    case ptxTypeU32             :
    case ptxTypeU64             :
    case ptxTypeS2              :
    case ptxTypeS4              :
    case ptxTypeS8              :
    case ptxTypeS16             :
    case ptxTypeS32             :
    case ptxTypeS64             : return t->kind ^ ptxGetTypeSizeInBits(t) ^ isSignedInt(t);
    case ptxOpaqueType          : return t->kind ^ fieldsHash    ( t->cases.Opaque.fields ) ^ stdStringHash(t->cases.Opaque.name);
    case ptxVectorType          : return t->kind ^ stdAddressHash( t->cases.Vector        .base ) ^ t->cases.Vector.N;  
    case ptxArrayType           : return t->kind ^ stdAddressHash( t->cases.Array         .base ) ^ t->cases.Array .N; 
    case ptxIncompleteArrayType : return t->kind ^ stdAddressHash( t->cases.IncompleteArray.base ) ^ t->cases.IncompleteArray.logAlignment;
    default                     : stdASSERT( False, ("Case label out of bounds") );
    }
    return 0;
}


    
static ptxType uniqueType( ptxType candidate, ptxParsingState parseState)
{
    
    ptxType result;
    
    if (!parseState->uniqueTypes) {parseState-> uniqueTypes= mapXNEW(Type, 64); }

    result= mapApply(parseState->uniqueTypes,candidate);
    
    if (!result) {
        result= stdCOPY(candidate);
        mapDefine(parseState->uniqueTypes,result,result);
    }

    return result;
}

/*------------------------ Type Constructor Functions ------------------------*/

uInt64 ptxGetTypeSizeInBits(ptxType type)
{
    switch(type->kind) {
    case ptxTypeB1:
        return 1;
    case ptxTypeB2:
    case ptxTypeU2:
    case ptxTypeS2:
        return 2;
    case ptxTypeU4:
    case ptxTypeS4:
    case ptxTypeB4:
        return 4;
    case ptxTypeB8:
    case ptxTypeU8:
    case ptxTypeS8:
    case ptxTypeE4M3:
    case ptxTypeE5M2:
        return 8;
    case ptxTypeE4M3x2:
    case ptxTypeE5M2x2:
    case ptxTypeB16:
    case ptxTypeU16:
    case ptxTypeS16:
    case ptxTypeF16:
    case ptxTypeBF16:
        return 16;
    case ptxTypeTF32:
        return 32;
    case ptxTypeB32:
    case ptxTypeU32:
    case ptxTypeS32:
    case ptxTypeF32:
    case ptxTypeBF16x2:
        return 32;
    case ptxTypeB64:
    case ptxTypeU64:
    case ptxTypeS64:
    case ptxTypeF64:
        return 64;
    case ptxTypeB128:
        return 128;
    case ptxOpaqueType:
        return type->cases.Opaque.sizeInBits;
    case ptxTypePred:     // @@ ?? how big should a predicate be in memory?
    case ptxConditionCodeType: // @@ ?? how big should a predicate be in memory?
    case ptxTypeF16x2:
        return 32;
    case ptxVectorType:
        return (type->cases.Vector.N) *
               ptxGetTypeSizeInBits(type->cases.Vector.base);
    case ptxArrayType:
        return type->cases.Array.N *
               stdROUNDUP((int)ptxGetTypeSizeInBits(type->cases.Array.base) ,
                          1 <<ptxGetTypeLogAlignment(type->cases.Array.base));
    case ptxLabelType:
    case ptxMacroType:
    case ptxParamListType:
        return -1;
    case ptxIncompleteArrayType:
        return 0;
    default:
        stdASSERT(False, ("Unexpected PTX type"));
        return 0;
    }
}

uInt64 ptxGetTypeSizeInBytes(ptxType type)
{
    return ptxGetTypeSizeInBits(type) / stdBITSPERBYTE;
}

uInt ptxGetTypeLogAlignment(ptxType type)
{
    switch(type->kind) {
    case ptxTypeB8 :
    case ptxTypeB16:
    case ptxTypeB32:
    case ptxTypeB64:
    case ptxTypeB128:
    case ptxTypeU8 :
    case ptxTypeU16:
    case ptxTypeU32:
    case ptxTypeU64:
    case ptxTypeS8 :
    case ptxTypeS16:
    case ptxTypeS32:
    case ptxTypeS64:
    case ptxTypeE4M3:
    case ptxTypeE5M2:
    case ptxTypeE4M3x2:
    case ptxTypeE5M2x2:
    case ptxTypeF16:
    case ptxTypeF32:
    case ptxTypeF64:
    case ptxTypePred:
    case ptxConditionCodeType:
    case ptxTypeF16x2:
    case ptxTypeBF16:
    case ptxTypeBF16x2:
        return stdLOG2((int) ptxGetTypeSizeInBytes(type));
    case ptxOpaqueType:
        return type->cases.Opaque.logAlignment;
    case ptxIncompleteArrayType:
        return type->cases.IncompleteArray.logAlignment;
    case ptxVectorType:
        return stdLOG2(type->cases.Vector.N) +
               ptxGetTypeLogAlignment(type->cases.Vector.base);
    case ptxArrayType:
        return ptxGetTypeLogAlignment(type->cases.Array.base);
    case ptxLabelType:
    case ptxMacroType:
    case ptxParamListType:
        return 0;
    default:
        stdASSERT(False, ("Unexpected PTX type"));
        return 0;
    }
}

/*
 * Function         : Create macro type representation.
 * Parameters       :
 * Function Result  : Requested type
 */
ptxType ptxCreateMacroType(ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    
    result.kind         = ptxMacroType;
    
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create label type representation.
 * Parameters       :
 * Function Result  : Requested type
 */
ptxType ptxCreateLabelType(ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    
    result.kind         = ptxLabelType;
    
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create predicate type representation.
 * Parameters       :
 * Function Result  : Requested type
 */
ptxType ptxCreatePredicateType(ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    
    result.kind         = ptxTypePred;
    
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create condition code type representation.
 * Parameters       :
 * Function Result  : Requested type
 */
ptxType ptxCreateConditionCodeType(ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    
    result.kind         = ptxConditionCodeType;
    
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create parameter list type representation.
 * Parameters       :
 * Function Result  : Requested type
 */
static ptxType ptxCreateParamListType(ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    
    result.kind         = ptxParamListType;
    
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create bit type representation.
 * Parameters       : size    (I) Representation size in bits
 * Function Result  : Requested type
 */
ptxType ptxCreateBitType( uInt64 size, ptxParsingState parseState)
{
    stdASSERT(FITS_IN_INT32(size), ("Size of data type more than expected"));
    ptxTypeRec result;
    stdMEMCLEAR(&result);

    switch(size) {
    case   1: result.kind = ptxTypeB1;   break;
    case   2: result.kind = ptxTypeB2;   break;
    case   4: result.kind = ptxTypeB4;   break;
    case   8: result.kind = ptxTypeB8;   break;
    case  16: result.kind = ptxTypeB16;  break;
    case  32: result.kind = ptxTypeB32;  break;
    case  64: result.kind = ptxTypeB64;  break;
    case 128: result.kind = ptxTypeB128; break;
    default: stdASSERT(False, ("Unexpected bit type size"));
             result.kind = ptxTypeB8;
    }

    return uniqueType(&result, parseState);
}

/*
 * Function         : Create float type representation.
 * Parameters       : size    (I) Representation size in bits (16, 32 or 64)
 * Function Result  : Requested type
 */
ptxType ptxCreateFloatType( uInt64 size, ptxParsingState parseState)
{
    ptxTypeRec result;

    stdASSERT(FITS_IN_INT32(size), ("Size of data type more than expected"));
    stdASSERT(size == 16 || size == 32 || size == 64, ("Illegal float size"));

    stdMEMCLEAR(&result);
    switch (size) {
    case 16: result.kind = ptxTypeF16; break;
    case 32: result.kind = ptxTypeF32; break;
    case 64: result.kind = ptxTypeF64; break;
    }
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create packed half float type representation.
 * Parameters       : size    (I) Representation size in bits (32)
 *                                As of now only f16x2 is supported.
 * Function Result  : Requested type
 */
ptxType ptxCreatePackedHalfFloatType( uInt64 size, ptxParsingState parseState)
{
    stdASSERT(FITS_IN_INT32(size), ("Size of data type more than expected"));
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    
    stdASSERT(size == 32, ("Illegal size")); 
    result.kind         = ptxTypeF16x2;
        
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create custom float type representation.
 * Parameters       : e   (I) Size of Exponent in bits (As of now 4, 5 and 8 bits are supported)
 *                  : m   (I) Size of mantissa in bits (As of now 2, 3, 7 and 10 bits are supported)
 *                  : num (I) number of elements packed
 * Function Result  : Requested type
 */
ptxType ptxCreateLwstomFloatType(uInt e, uInt m, uInt num, ptxParsingState parseState)
{
    uInt size = (e + m + 1)*num;
    stdASSERT(FITS_IN_INT32(size), ("Size of data type more than expected"));
    ptxTypeRec result;
    stdMEMCLEAR(&result);

    stdASSERT(num == 1 || num == 2, ("Unexpected number of packed elements"));
    stdASSERT(e == 8 || e == 4 || e == 5, ("Unexpected exponent size"));
    stdASSERT(m == 7 || m == 10 || m == 2 || m == 3, ("Unexpected mantissa size"));
    stdASSERT((e == 8 && (m == 7 || m == 10)) ||
              (e == 4 && m == 3)              ||
              (e == 5 && m == 2),
              ("Unexpected exponent & mantissa combination"));

    if (num == 1) {
        if (e == 8) {
            result.kind = m == 7 ? ptxTypeBF16 : ptxTypeTF32;
        } else {
            result.kind = m == 3 ? ptxTypeE4M3 : ptxTypeE5M2;
#if !(LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL))
            stdCHECK_WITH_POS(False, (ptxMsgUnknownModifier,  ptxsLwrPos(parseState),
                                      getTypeEnumAsString(parseState->parseData->deobfuscatedStringMapPtr, result.kind)));
#endif
        }
    } else {
        stdASSERT((m == 3 || m == 2 || m == 7) && num == 2, ("Only two elements bf16 or f8 types are allowed to pack"));
        if (m == 7) {
            result.kind = ptxTypeBF16x2;
        } else {
            result.kind = m == 3 ? ptxTypeE4M3x2 : ptxTypeE5M2x2;
#if !(LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL))
            stdCHECK_WITH_POS(False, (ptxMsgUnknownModifier,  ptxsLwrPos(parseState),
                                      getTypeEnumAsString(parseState->parseData->deobfuscatedStringMapPtr, result.kind)));
#endif
        }
    }

    return uniqueType(&result, parseState);
}


/*
 * Function         : Create integer type representation.
 * Parameters       : size     (I) Representation size in bits (8, 16, 32 or 64)
 *                    isSigned (I) integer type attribute 'signed'
 * Function Result  : Requested type
 */
ptxType ptxCreateIntType( uInt64 size, Bool isSigned, ptxParsingState parseState)
{
    stdASSERT(FITS_IN_INT32(size), ("Size of data type more than expected"));
    ptxTypeRec result;
    stdMEMCLEAR(&result);
   
    switch(size) {
    case  2: result.kind = isSigned ?  ptxTypeS2 : ptxTypeU2;  break;
    case  4: result.kind = isSigned ?  ptxTypeS4 : ptxTypeU4;  break;
    case  8: result.kind = isSigned ?  ptxTypeS8 : ptxTypeU8;  break;
    case 16: result.kind = isSigned ? ptxTypeS16 : ptxTypeU16; break;
    case 32: result.kind = isSigned ? ptxTypeS32 : ptxTypeU32; break;
    case 64: result.kind = isSigned ? ptxTypeS64 : ptxTypeU64; break;
    default: stdASSERT(False, ("Unexpected int type size"));
             result.kind = ptxTypeS8;
    }
    
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create opaque type representation.
 * Parameters       : name     (I) name of type
 *                    fields   (I) list of ptxSymbols, opaque fields
 * Function Result  : Requested type
 */
 static void mapOpaqueFields( ptxSymbol field, ptxType type ) 
 { 
     stdASSERT((type->kind == ptxOpaqueType), ("Opaque type is expected"))

     type->cases.Opaque.sizeInBits  = stdROUNDUP( ptxGetTypeSizeInBits(type), 1<< ptxGetTypeLogAlignment(field->type));
     type->cases.Opaque.sizeInBits += ptxGetTypeSizeInBits(field->type);

     type->cases.Opaque.logAlignment = stdMAX( type->cases.Opaque.logAlignment,
                                               ptxGetTypeLogAlignment(field->type));
 }
 
ptxType ptxCreateOpaqueType( String name, stdList_t fields , ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    
    result.kind                   = ptxOpaqueType;
    result.cases.Opaque.sizeInBits = 0;
    result.cases.Opaque.name   = name;
    result.cases.Opaque.fields = fields;
    result.cases.Opaque.logAlignment = 0;
    
    listTraverse( fields, (stdEltFun)mapOpaqueFields, &result );  

    return uniqueType(&result, parseState);
}

/*
 * Function         : Get the operation-modifier from MMA_OPMOD table corresponding to the POSTOP
 * Parameters       : ptxOperator     (I) Postop whose corresponding entry
 *                                        in MMA_OPMOD table is to be found
 * Function Result  : MMA_OPMOD table entry which corresponds to ptxOperator 'op'
 */
uInt colwertPostOpToMMAOperation(ptxOperator op) {
     switch(op) {
#if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_70)
     case ptxANDOp : return ptxMMA_AND_MOD;
#endif
     case ptxXOROp : return ptxMMA_XOR_MOD;
     case ptxPOPCOp: return ptxMMA_POPC_MOD;
     default:
        return ptxNOMMA_MOD;
     }
}

/*
 * Function         : Get the type-modifier from TYPEMOD table corresponding to the ptxType
 * Parameters       : type     (I) PTX Type whose corresponding entry in TYPEMOD table is to be found
 * Function Result  : TYPEMOD table entry which corresponds to ptxType 'type'
 */
uInt ptxGetTypeModFromType(ptxType type)
{
    if (isU2(type))  return ptxTYPE_u2_MOD;
    if (isU4(type))  return ptxTYPE_u4_MOD;
    if (isU8(type))  return ptxTYPE_u8_MOD;
    if (isU16(type)) return ptxTYPE_u16_MOD;
    if (isU32(type)) return ptxTYPE_u32_MOD;
    if (isU64(type)) return ptxTYPE_u64_MOD;
    if (isS2(type))  return ptxTYPE_s2_MOD;
    if (isS4(type))  return ptxTYPE_s4_MOD;
    if (isS8(type))  return ptxTYPE_s8_MOD;
    if (isS16(type)) return ptxTYPE_s16_MOD;
    if (isS32(type)) return ptxTYPE_s32_MOD;
    if (isS64(type)) return ptxTYPE_s64_MOD;
    if (isB1(type))  return ptxTYPE_b1_MOD;
    if (isB2(type))  return ptxTYPE_b2_MOD;
    if (isB4(type))  return ptxTYPE_b4_MOD;
    if (isB8(type))  return ptxTYPE_b8_MOD;
    if (isB16(type)) return ptxTYPE_b16_MOD;
    if (isB32(type)) return ptxTYPE_b32_MOD;
    if (isB64(type)) return ptxTYPE_b64_MOD;
    if (isB128(type)) return ptxTYPE_b128_MOD;
    if (isE4M3(type)) return ptxTYPE_e4m3_MOD;
    if (isE5M2(type)) return ptxTYPE_e5m2_MOD;
    if (isF16(type)) return ptxTYPE_f16_MOD;
    if (isF32(type)) return ptxTYPE_f32_MOD;
    if (isF16x2(type)) return ptxTYPE_f16x2_MOD;
    if (isF64(type)) return ptxTYPE_f64_MOD;
    if (isBF16(type)) return ptxTYPE_BF16_MOD;
    if (isBF16x2(type)) return ptxTYPE_BF16x2_MOD;
    if (isTF32(type)) return ptxTYPE_TF32_MOD;
    return ptxNOTYPE_MOD;
}

/*
 * Function         : Get the instruction type from TYPEMOD
 * Parameters       : typemod     (I) PTX Type modifier
 * Function Result  : instruction type corresponding to the TYPEMOD
 */
ptxType ptxGetTypeFromTypeMod(uInt typeMod, ptxParsingState parseState)
{
    switch (typeMod) {
    case ptxTYPE_u2_MOD:     return ptxCreateIntType(2,  False, parseState);
    case ptxTYPE_u4_MOD:     return ptxCreateIntType(4,  False, parseState);
    case ptxTYPE_u8_MOD:     return ptxCreateIntType(8,  False, parseState);
    case ptxTYPE_u16_MOD:    return ptxCreateIntType(16, False, parseState);
    case ptxTYPE_u32_MOD:    return ptxCreateIntType(32, False, parseState);
    case ptxTYPE_u64_MOD:    return ptxCreateIntType(64, False, parseState);
    case ptxTYPE_s2_MOD:     return ptxCreateIntType(2,  True, parseState);
    case ptxTYPE_s4_MOD:     return ptxCreateIntType(4,  True, parseState);
    case ptxTYPE_s8_MOD:     return ptxCreateIntType(8,  True, parseState);
    case ptxTYPE_s16_MOD:    return ptxCreateIntType(16, True, parseState);
    case ptxTYPE_s32_MOD:    return ptxCreateIntType(32, True, parseState);
    case ptxTYPE_s64_MOD:    return ptxCreateIntType(64, True, parseState);
    case ptxTYPE_b1_MOD:     return ptxCreateBitType(1, parseState);
    case ptxTYPE_b2_MOD:     return ptxCreateBitType(2, parseState);
    case ptxTYPE_b4_MOD:     return ptxCreateBitType(4, parseState);
    case ptxTYPE_b8_MOD:     return ptxCreateBitType(8, parseState);
    case ptxTYPE_b16_MOD:    return ptxCreateBitType(16, parseState);
    case ptxTYPE_b32_MOD:    return ptxCreateBitType(32, parseState);
    case ptxTYPE_b64_MOD:    return ptxCreateBitType(64, parseState);
    case ptxTYPE_b128_MOD:   return ptxCreateBitType(128, parseState);
    case ptxTYPE_e4m3_MOD:   return ptxCreateLwstomFloatType(4, 3, 1, parseState);
    case ptxTYPE_e5m2_MOD:   return ptxCreateLwstomFloatType(5, 2, 1, parseState);
    case ptxTYPE_f16_MOD:    return ptxCreateFloatType(16, parseState);
    case ptxTYPE_f32_MOD:    return ptxCreateFloatType(32, parseState);
    case ptxTYPE_f64_MOD:    return ptxCreateFloatType(64, parseState);     
    case ptxTYPE_f16x2_MOD:  return ptxCreatePackedHalfFloatType(32, parseState);
    case ptxTYPE_BF16_MOD:   return ptxCreateLwstomFloatType(8, 7, 1, parseState);
    case ptxTYPE_BF16x2_MOD: return ptxCreateLwstomFloatType(8, 7, 2, parseState);
    case ptxTYPE_TF32_MOD:   return ptxCreateLwstomFloatType(8, 10, 1, parseState);
    default:
        stdASSERT(False, ("Unexpected type modifier"));
        return NULL;
    }
}

/*
 * Function         : Get the type's size in bits
 * Parameters       : type     (I) Type from TYPEMOD table whose size is to be queried
 * Function Result  : Size of 'type' in bits
 */
uInt ptxGetTypeModSize(uInt type)
{
    switch(type) {
    case ptxTYPE_s2_MOD  : return 2;
    case ptxTYPE_s4_MOD  : return 4;
    case ptxTYPE_s8_MOD  : return 8;
    case ptxTYPE_s32_MOD : return 32;
    case ptxTYPE_u2_MOD  : return 2;
    case ptxTYPE_u4_MOD  : return 4;
    case ptxTYPE_u8_MOD  : return 8;
    case ptxTYPE_b1_MOD  : return 1;
    case ptxTYPE_b2_MOD  : return 2;
    case ptxTYPE_b4_MOD  : return 4;
    case ptxTYPE_b8_MOD  : return 8;
    case ptxTYPE_b16_MOD : return 16;
    case ptxTYPE_u16_MOD : return 16;
    case ptxTYPE_e4m3_MOD: return 8;
    case ptxTYPE_e5m2_MOD: return 8;
    case ptxTYPE_f16_MOD : return 16;
    case ptxTYPE_BF16_MOD : return 16;
    case ptxTYPE_BF16x2_MOD : return 32;
    case ptxTYPE_f32_MOD : return 32;
    case ptxTYPE_TF32_MOD : return 32;
    default:
        stdASSERT(False, ("unexpected type modifier"));
        return 0;
    }
}

/*
 * Function         : Recognize TypeMod from a given string
 * Parameters       : modNameStr     (I) Name of the TypeMod to be recognized
 * Function Result  : ptxTYPEmod enum value corresponding to the input string
 */
uInt recognizeTypeMod(ptxParseData parseData, cString modNameStr)
{
#define recognizeAs(typeMod, recognizedTypeMod) \
    if (stdEQSTRING(modNameStr, getTYPEMODAsStringRaw(parseData->deobfuscatedStringMapPtr, typeMod))) \
        return recognizedTypeMod;

#define recognize(typeMod)  recognizeAs(typeMod, typeMod)

#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_63)
    // Since, .b1,.b2 or .b4 might clash with video selectors to avoid that,
    // so do not identify them in lexer. If the string reaches here then,
    // treat it as a type modifier.
    recognize(ptxTYPE_b4_MOD)
    recognize(ptxTYPE_b2_MOD)
    recognize(ptxTYPE_b1_MOD)

    recognize(ptxTYPE_TF32_MOD)
#endif

#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_64)
    recognize(ptxTYPE_BF16_MOD)
#endif

#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_65)
    recognize(ptxTYPE_BF16x2_MOD)
#endif

    return ptxNOTYPE_MOD;

#undef recognizeAs
#undef recognize
}

/*
 * Function         : Get number of group represented by group modifier
 * Parameters       : group modifier
 * Function Result  : number of group represented
 */
uInt ptxGetNumOfGroups(uInt groupMod)
{
    switch (groupMod) {
    case ptxGROUP_g1_MOD: return 1;
    case ptxGROUP_g2_MOD: return 2;
    case ptxGROUP_g4_MOD: return 4;
    default: stdASSERT(False, ("Unexpected group modifier")); break;
    }
    return 0;
}

/*
 * Function         : Create pointer type representation.
 * Parameters       : storage  (I) storage class of pointer target
 *                                 Precondition is that ptxIsAddressableStorage(storage) is True
 *                    base     (I) pointer target type
 * Function Result  : Requested type
 */
 stdUNUSED(static uInt getPointerTypeSize( ptxStorageClass storage ));
 static uInt getPointerTypeSize( ptxStorageClass storage ) 
 {
     switch (storage.kind) {
     case ptxConstStorage       :
     case ptxLocalStorage       :
     case ptxParamStorage       :
     case ptxSharedStorage      : return 2;   // 16 bit pointer

     case ptxCodeStorage        :
     case ptxGenericStorage     :
     case ptxGlobalStorage      : return 4;   // @@ disregarding storage specific pointer size

     case ptxRegStorage         :
     case ptxSregStorage        :
     case ptxSurfStorage        :
     case ptxTexStorage         :
     case ptxTexSamplerStorage  :
     case ptxUNSPECIFIEDStorage :
     default                    : return 0xffffffff;
     }
 }

/*
 * Function         : Create representation of array type with unspecified N.
 * Parameters       : base     (I) Array element type
 * Function Result  : Requested type
 */
ptxType ptxCreateIncompleteArrayType(ptxType base, uInt logAlignment, ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    
    result.kind             = ptxIncompleteArrayType;
    result.cases.IncompleteArray.base = base;
    result.cases.IncompleteArray.logAlignment = stdMAX(logAlignment,
                                                       ptxGetTypeLogAlignment(base));
    
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create array type representation.
 * Parameters       : base     (I) Array element type
 *                    N        (I) Number of elements of array type
 * Function Result  : Requested type
 */
ptxType ptxCreateArrayType( uInt64 N, ptxType base , ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
   
    result.kind             = ptxArrayType;
    result.cases.Array.base = base;
    result.cases.Array.N    = N;
    
    return uniqueType(&result, parseState);
}

/*
 * Function         : Create vector type representation.
 * Parameters       : base     (I) Vector element element type,
 *                                 which must be basic type (Bit, Int or Float)
 *                    N        (I) Number of elements of array type
 * Function Result  : Requested type
 */
ptxType ptxCreateVectorType( uInt N, ptxType base , ptxParsingState parseState)
{
    ptxTypeRec result;
    stdMEMCLEAR(&result);
    Bool isPredVec = False;
    Bool allowVarSizeVec = False;

#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    allowVarSizeVec = True;
#endif

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_62)
    isPredVec = base->kind == ptxTypePred && N < 8;
#endif

// Extended instructions P2R and R2P operates on the vectors
// of predicates with size less than 8
// Instruction 'spmetadata' creates vector of length '3' in DAG generation phase
// Instruciton _mma.warpgroup can have vectors of length up to 128
#if LWCFG(GLOBAL_ARCH_VOLTA)
    stdASSERT(N == 1 || N == 2 || N == 3 || N == 4 || N == 8 || isPredVec || allowVarSizeVec,
              ("Illegal number of vector elements"));
#else
    stdASSERT(N == 1 || N == 2 || N == 4 || isPredVec,
              ("Illegal number of vector elements"));
#endif

    result.kind              = ptxVectorType;
    result.cases.Vector.N    = N;
    result.cases.Vector.base = base;
    
    return uniqueType(&result, parseState);
}

Bool ptxIsBasicTypeKind( ptxTypeKind kind )
{
    switch (kind) {
    case ptxTypeB1             :
    case ptxTypeB2             :
    case ptxTypeB4             :
    case ptxTypeB8             :
    case ptxTypeB16            :
    case ptxTypeB32            :
    case ptxTypeB64            :
    case ptxTypeB128           :
    case ptxTypeU2             :
    case ptxTypeU4             :
    case ptxTypeU8             :
    case ptxTypeU16            :
    case ptxTypeU32            :
    case ptxTypeU64            :
    case ptxTypeS2             :
    case ptxTypeS4             :
    case ptxTypeS8             :
    case ptxTypeS16            :
    case ptxTypeS32            :
    case ptxTypeS64            :
    case ptxTypeF16            :
    case ptxTypeF32            :
    case ptxTypeF64            :
    case ptxTypeF16x2          :
    case ptxTypePred           :
    case ptxConditionCodeType  : return True;

    default                    : return False;
    }
}

/*
 * Function         : Test if type is a basic type (Bit, Int, Float, or Predicate).
 * Parameters       : type     (I) Type to inspect
 * Function Result  : True iff. type is a basic type
 */
Bool ptxIsBasicType( ptxType type )
{
    return ptxIsBasicTypeKind(type->kind);
}


/*
 * Function         : Test if enough of the type is known 
 *                    in order to allocate an instance from it.
 * Parameters       : type     (I) Type to inspect
 * Function Result  : True iff. type is a complete type
 */
Bool ptxIsCompleteType( ptxType type )
{
    switch (type->kind) {
    case ptxIncompleteArrayType :
    case ptxMacroType           :
    case ptxLabelType           : return False;
    
    case ptxTypeU2              :
    case ptxTypeU4              :
    case ptxTypeU8              :
    case ptxTypeU16             :
    case ptxTypeU32             :
    case ptxTypeU64             :
    case ptxTypeS2              :
    case ptxTypeS4              :
    case ptxTypeS8              :
    case ptxTypeS16             :
    case ptxTypeS32             :
    case ptxTypeS64             :
    case ptxOpaqueType          :
    case ptxTypeE4M3            :
    case ptxTypeE5M2            :
    case ptxTypeE4M3x2          :
    case ptxTypeE5M2x2          :
    case ptxTypeF16             :
    case ptxTypeF32             :
    case ptxTypeF64             :
    case ptxTypeF16x2           :
    case ptxTypeBF16            :
    case ptxTypeBF16x2          :
    case ptxTypeTF32            :
    case ptxTypeB1              :
    case ptxTypeB2              :
    case ptxTypeB4              :
    case ptxTypeB8              :
    case ptxTypeB16             :
    case ptxTypeB32             :
    case ptxTypeB64             :
    case ptxTypeB128            :
    case ptxVectorType          :
    case ptxArrayType           :
    case ptxTypePred            :
    case ptxConditionCodeType   : return True;
    
    default                     : stdASSERT( False, ("Case label out of bounds") );
    }
    return False;
}


/*
 * Function         : Test if instances of type may be placed in registers.
 * Parameters       : type     (I) Type to inspect
 * Function Result  : True iff. type is 'registerable'
 */
Bool ptxIsRegisterType( ptxType type )
{
    switch (type->kind) {
    case ptxIncompleteArrayType :
    case ptxLabelType           :
    case ptxMacroType           :
    case ptxParamListType       :
    case ptxOpaqueType          :
    case ptxArrayType           : return False;
    
    case ptxTypeU8              :
    case ptxTypeU16             :
    case ptxTypeU32             :
    case ptxTypeU64             :
    case ptxTypeS8              :
    case ptxTypeS16             :
    case ptxTypeS32             :
    case ptxTypeS64             :
    case ptxTypeE4M3            :
    case ptxTypeE5M2            :
    case ptxTypeE4M3x2          :
    case ptxTypeE5M2x2          :
    case ptxTypeF16             :
    case ptxTypeF32             :
    case ptxTypeF64             :
    case ptxTypeF16x2           :
    case ptxTypeB8              :
    case ptxTypeB16             :
    case ptxTypeB32             :
    case ptxTypeB64             :
    case ptxTypeB128            :
    case ptxVectorType          :
    case ptxTypePred            :
    case ptxConditionCodeType   : return True;
    
    default                     : stdASSERT( False, ("Case label out of bounds") );
    }
    return False;
}


/*
 * Function         : Test if instances of type may be placed in parameter state space.
 * Parameters       : type     (I) Type to inspect
 *                    isEntry  (I) True iff. function is a CTA entry
 * Function Result  : True iff. type is 'parameterizable'
 */
Bool ptxIsParameterType( ptxType type, Bool isEntry )
{
    switch (type->kind) {
    case ptxLabelType           :
    case ptxMacroType           :
    case ptxParamListType       :
    case ptxVectorType          : return False;
    
    case ptxOpaqueType          : return isEntry;

    case ptxIncompleteArrayType :
    case ptxArrayType           :
    case ptxTypeB8              :
    case ptxTypeB16             :
    case ptxTypeB32             :
    case ptxTypeB64             :
    case ptxTypeB128            :
    case ptxTypeF16             :
    case ptxTypeF32             :
    case ptxTypeF64             :
    case ptxTypeU8              :
    case ptxTypeU16             :
    case ptxTypeU32             :
    case ptxTypeU64             :
    case ptxTypeS8              :
    case ptxTypeS16             :
    case ptxTypeS32             :
    case ptxTypeS64             :
    case ptxTypePred            : return True;
    
    default                     : stdASSERT( False, ("Case label out of bounds") );
    }
    return False;
}


/*
 * Function         : Test if storage class is addressable.
 * Parameters       : storage (I) Storage to inspect
 * Function Result  : True iff. storage is addressable
 */
Bool ptxIsAddressableStorage( ptxStorageClass storage ) 
{
    switch (storage.kind) {
    case ptxCodeStorage        :
    case ptxConstStorage       :
    case ptxGenericStorage     :
    case ptxGlobalStorage      :
    case ptxLocalStorage       :
    case ptxParamStorage       :
    case ptxSharedStorage      : return True;

    case ptxSurfStorage        :
    case ptxTexStorage         :
    case ptxTexSamplerStorage  :
    case ptxRegStorage         :
    case ptxSregStorage        :
    case ptxUNSPECIFIEDStorage :
    default                    : return False;
    }
}


/*
 * Function         : Test if storage class is initializable
 * Parameters       : storage (I) Storage to inspect
 * Function Result  : True iff. storage is initializable
 */
Bool ptxIsInitializableStorage( ptxStorageClass storage ) 
{
    switch (storage.kind) {
    case ptxCodeStorage        :
    case ptxConstStorage       :
    case ptxGlobalStorage      : return True;

    case ptxUNSPECIFIEDStorage :
    case ptxRegStorage         :
    case ptxSregStorage        :
    case ptxLocalStorage       :
    case ptxParamStorage       :
    case ptxSharedStorage      :
    case ptxGenericStorage     :
    case ptxSurfStorage        :
    case ptxTexStorage         :
    case ptxTexSamplerStorage  :
    default                    : return False;
    }
}


/*
 * Function         : Test if storage class represents 
 *                    some form of register space.
 * Parameters       : storage (I) Storage to inspect
 * Function Result  : True iff. storage is a register
 */
Bool ptxIsRegisterStorage( ptxStorageClass storage )
{
    switch (storage.kind) {
    case ptxUNSPECIFIEDStorage :
    case ptxCodeStorage        :
    case ptxConstStorage       :
    case ptxGenericStorage     :
    case ptxGlobalStorage      :
    case ptxLocalStorage       :
    case ptxParamStorage       :
    case ptxSharedStorage      :
    case ptxSurfStorage        :
    case ptxTexStorage         :
    case ptxTexSamplerStorage  : return False;

    case ptxRegStorage         :
    case ptxSregStorage        :
    default                    : return True;
    }
}


/*
 * Function         : Create basic type from type info.
 * Parameters       : kind     (I) Kind of basic type to create
 *                    size     (I) Size of basic type to create
 * Function Result  : Requested basic type
 */
ptxType ptxCreateBasicType( ptxTypeKind kind, uInt64 size, ptxParsingState parseState)
{
    stdASSERT(ptxIsBasicTypeKind(kind), ("Expected basic type"));

    switch (kind) {
    case ptxTypeB8              :
    case ptxTypeB16             :
    case ptxTypeB32             :
    case ptxTypeB64             :
    case ptxTypeB128            : return ptxCreateBitType            (size, parseState);
    case ptxTypeU8              :
    case ptxTypeU16             :
    case ptxTypeU32             :
    case ptxTypeU64             : return ptxCreateIntType            (size, False,  parseState);
    case ptxTypeS8              :
    case ptxTypeS16             :
    case ptxTypeS32             :
    case ptxTypeS64             : return ptxCreateIntType            (size, True, parseState);
    case ptxTypeF16             :
    case ptxTypeF32             :
    case ptxTypeF64             : return ptxCreateFloatType          (size, parseState);
    case ptxTypeF16x2           : return ptxCreatePackedHalfFloatType(size, parseState);
    case ptxTypePred            : return ptxCreatePredicateType      (parseState);
    case ptxConditionCodeType   : return ptxCreateConditionCodeType  (parseState);
    default                     : return NULL;
    }
}

/* 
 * Function         : Get base type 
 * Parameters       : type  (I) Type 
 * Function Result  : For aggregate types (array, vector) return base type,
 *                    otherwise return same type
 */
ptxType ptxGetBaseType( ptxType type )
{
    while (type->kind == ptxVectorType || type->kind == ptxArrayType) {
        if (type->kind == ptxVectorType) {
            type = type->cases.Vector.base;
        } else if (type->kind == ptxArrayType) {
            type = type->cases.Array.base;
        }
    }
    return type;
}


/*--------------------- Expression Constructor Functions ---------------------*/

/*
 * Function         : Create binary expression
 * Parameters       : type       (I) result type
 *                    op         (I) binary operator
 *                    left,right (I) binary arguments
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateBinaryExpr( ptxType type, ptxOperator op, ptxExpression left, ptxExpression right )
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind               = ptxBinaryExpression;
    result->type               = type;
    result->isConstant         = left->isConstant && right->isConstant;
    result->isLhs              = False;

    stdNEW(result->cases.Binary);
    result->cases.Binary->op    = op;
    result->cases.Binary->left  = left;
    result->cases.Binary->right = right;
    return result;
}

/*
 * Function         : Create unary expression
 * Parameters       : type       (I) result type
 *                    op         (I)  unary operator
 *                    arg        (I)  operation argument
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateUnaryExpr( ptxType type, ptxOperator op, ptxExpression arg )
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind            = ptxUnaryExpression;
    result->type            = type;
    result->isConstant      = arg->isConstant;
    result->isLhs           = False;

    stdNEW(result->cases.Unary);
    result->cases.Unary->op  = op;
    result->cases.Unary->arg = arg;
    return result;
}

/*
 * Function         : Create integer constant expression
 * Parameters       : i          (I) Integer constant
 *                    isSigned   (I) True iff. integer is signed
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateIntConstantExpr( Int64 i, Bool isSigned , ptxParsingState parseState)
{
    ptxExpression result;

    stdNEW(result);
    
    result->kind                = ptxIntConstantExpression;
    // integer constants are represented as 64b types
    result->type                = ptxCreateIntType(64, isSigned, parseState);
    result->isConstant          = True;
    result->isLhs               = False;
    result->cases.IntConstant.i = i;
    
    return result;
}

/*
 * Function         : Create single-precision floating-point constant expression
 * Parameters       : f          (I) Float constant
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateF32FloatConstantExpr( Float f , ptxParsingState parseState)
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind                    = ptxFloatConstantExpression;
    result->type                    = ptxCreateFloatType(32, parseState);
    result->isConstant              = True;
    result->isLhs                   = False;
    result->cases.FloatConstant.flt = f;
    
    return result;
}

/*
 * Function         : Create double-precision floating-point constant expression
 * Parameters       : d          (I) Double constant
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateF64FloatConstantExpr( Double d , ptxParsingState parseState)
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind                    = ptxFloatConstantExpression;
    result->type                    = ptxCreateFloatType(64, parseState);
    result->isConstant              = True;
    result->isLhs                   = False;
    result->cases.DoubleConstant.dbl = d;
    
    return result;
}

/*
 * Function         : Get single-precision value of FloatConstant expression, casting from double if necessary
 * Parameters       : e          (I) Float Constant Expression
 * Function Result  : Requested value
 */
Float ptxGetF32FloatConstantExpr(ptxExpression e)
{
    stdASSERT((e->kind == ptxFloatConstantExpression && (ptxGetTypeSizeInBits(e->type) == 32 || ptxGetTypeSizeInBits(e->type) == 64)),("Illegal FloatConstantExpression"));

    if (ptxGetTypeSizeInBits(e->type) == 64)
        return (Float)(e->cases.DoubleConstant.dbl);
    else
        return e->cases.FloatConstant.flt;
}

uInt32 ptxColwert32FloatToUnsignedIntExpr(ptxExpression e)
{
    uInt32 *uVal;
    stdASSERT((e->kind == ptxFloatConstantExpression && (ptxGetTypeSizeInBits(e->type) == 32 || ptxGetTypeSizeInBits(e->type) == 64)),
              ("Illegal FloatConstantExpression for colwersion. Expected 32 bit/64 bit floating point constant."));
    if(ptxGetTypeSizeInBits(e->type) == 32) {
        uVal = (uInt32 *) &(e->cases.FloatConstant.flt);
    } else {
        uVal = (uInt32 *) &(e->cases.DoubleConstant.dbl);
    }
    return *uVal;
}

uInt64 ptxColwert64FloatToUnsignedLongExpr(ptxExpression e)
{
    uInt64 *ulVal;
    stdASSERT((e->kind == ptxFloatConstantExpression && ptxGetTypeSizeInBits(e->type) == 64),
        ("Illegal FloatConstantExpression for colwersion. Expected 64 bit floating point constant."));
    ulVal =  (uInt64 *) &(e->cases.DoubleConstant.dbl);
    return *ulVal;
}

/*
 * Function         : Get double-precision value of FloatConstant expression; no up-casting from f32 allowed
 * Parameters       : e          (I) Float Constant Expression
 * Function Result  : Requested value
 */
Double ptxGetF64FloatConstantExpr(ptxExpression e)
{
    stdASSERT((e->kind == ptxFloatConstantExpression && ptxGetTypeSizeInBits(e->type) == 64),("Illegal FloatConstantExpression"));
    return e->cases.DoubleConstant.dbl;
}

/*
 * Function         : Set the symbol in a symbol expression
 * Parameters       : expr       (I) symbol reference expression
 *                    symbol     (I) symbol referred
 */
void ptxInitSymbolExpr(ptxExpression expr, ptxSymbolTableEntry symbol)
{
    expr->kind                = ptxSymbolExpression;
    expr->type                = symbol->symbol->type;
    expr->cases.Symbol.symbol = symbol;
    expr->isConstant          = True;   // @@ This will depend: only if it is not a local symbol, in a deeper symbol table
    expr->isLhs               = symbol->kind == ptxVariableSymbol;
}

/*
 * Function         : Create symbol reference expression
 * Parameters       : symbol     (I) symbol to reference
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateSymbolExpr( ptxSymbolTableEntry symbol )
{
    ptxExpression result;
    stdNEW(result);
    ptxInitSymbolExpr(result, symbol);
    return result;
}

/*
 * Function         : Create array- or vector indexing expression
 * Parameters       : arg        (I) array- or vector valued selectee
 *                    index      (I) index
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateArrayIndexExpr( ptxExpression arg, ptxExpression index )
{
    ptxExpression result;
    
    stdNEW(result);
    
    stdASSERT( arg->type->kind == ptxArrayType || arg->type->kind == ptxIncompleteArrayType, ("Array type expected") );
    
        if (arg->type->kind == ptxArrayType) {
        result->type = arg->type->cases.Array.base;
        } else {
        result->type = arg->type->cases.IncompleteArray.base;
        }
        
    result->kind                   = ptxArrayIndexExpression;
    result->isConstant             = False;
    result->isLhs                  = arg->isLhs;

    stdNEW(result->cases.ArrayIndex);
    result->cases.ArrayIndex->arg   = arg;
    result->cases.ArrayIndex->index = index;
    return result;
}

/*
 * Function         : Create array- or vector indexing expression
 * Parameters       : arg        (I) array- or vector valued selectee
 *                    dimension  (I) number of vector selectors, up to maximum of 4
 *                    selector   (I) selector string, e.g  "xxy", with length indicated by dimension
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateVectorSelectExpr( ptxExpression arg, uInt dimension, ptxVectorSelector *selector , ptxParsingState parseState)
{
    ptxExpression result;
    
    stdNEW(result);
    
    stdASSERT( arg->type->kind == ptxVectorType, ("Vector type expected") );
    
    result->kind                         = ptxVectorSelectExpression;
    result->type                         = arg->type->cases.Vector.base;
    result->isConstant                   = False;
    result->isLhs                        = arg->isLhs;

    stdNEW(result->cases.VectorSelect);
    result->cases.VectorSelect->arg       = arg;
    result->cases.VectorSelect->dimension = dimension;

    stdMEMCOPY_N(&result->cases.VectorSelect->selector[0], selector, dimension);
    
    if (dimension>1) {
       result->type= ptxCreateVectorType(dimension,result->type, parseState);
    }
    
    return result;
}
 
/*
 * Function         : Create video sub-word select expression
 * Parameters       : arg        (I) register symbol
 *                    N          (I) number of video selectors, up to maximum of 4
 *                    selector   (I) video selector
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateVideoSelectExpr( ptxExpression arg, uInt N, ptxVideoSelector *selector )
{
    ptxExpression result;
    
    stdNEW(result);
    
    stdASSERT (arg->kind == ptxSymbolExpression && arg->cases.Symbol.symbol->storage.kind == ptxRegStorage && ptxGetTypeSizeInBits(arg->type) == 32, ("Register symbol expected") );
    
    result->kind                       = ptxVideoSelectExpression;
    result->type                       = arg->type;
    result->isConstant                 = False;
    result->isLhs                      = arg->isLhs;

    stdNEW(result->cases.VideoSelect);
    result->cases.VideoSelect->arg      = arg;
    result->cases.VideoSelect->N        = N;

    stdMEMCOPY_N(&result->cases.VectorSelect->selector[0], selector, N);
    return result;
}

/*
 * Function         : Create byte sub-word select expression
 * Parameters       : arg        (I) register symbol
 *                    N          (I) number of byte selectors, up to maximum of 1
 *                    selector   (I) byte selector
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateByteSelectExpr( ptxExpression arg, uInt N, ptxByteSelector *selector )
{
    ptxExpression result;
    
    stdNEW(result);
    
    stdASSERT (arg->kind == ptxSymbolExpression && arg->cases.Symbol.symbol->storage.kind == ptxRegStorage && ptxGetTypeSizeInBits(arg->type) == 32, ("Register symbol expected") );
    
    result->kind                       = ptxByteSelectExpression;
    result->type                       = arg->type;
    result->isConstant                 = False;
    result->isLhs                      = arg->isLhs;

    stdNEW(result->cases.ByteSelect);
    result->cases.ByteSelect->arg      = arg;
    result->cases.ByteSelect->N        = N;

    stdMEMCOPY_N(&result->cases.VectorSelect->selector[0], selector, N);
    return result;
}
/*
 * Function         : Create vector expression from element list
 * Parameters       : elements (I) list of element
 *                    type     (I) type of vector expression
 * Function Result  : Requested expression
 */

        static void completeVectorExpr( ptxExpression element, ptxExpression result )
        {
            if (!element->type || !ptxIsBasicType(element->type)) return;
            result->isConstant &= element->isConstant;
            result->isLhs      &= element->isLhs;
        } 
 
ptxExpression ptxCreateVectorExpr( stdList_t elements, ptxType type )
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind                  = ptxVectorExpression;
    result->type                  = type;
    result->isConstant            = True;
    result->isLhs                 = True;
    result->cases.Vector.elements = elements;
    result->cases.Vector.reverseMod = False;
        
    listTraverse( elements, (stdEltFun)completeVectorExpr, result );
        
    return result;
}
 
/*
 * Function         : Create address take expression
 * Parameters       : lhs        (I) access path to memory location
 * Function Result  : Requested expression
 */
static void isCAP(ptxExpression e, Bool *result) { *result &= e->isConstant; }

static Bool isConstantAccessPath( ptxExpression lhs, ptxStorageClass *storage )
{
    switch (lhs->kind) {
    case ptxArrayIndexExpression:
        return lhs->cases.ArrayIndex->index->isConstant && isConstantAccessPath( lhs->cases.ArrayIndex->arg, storage );

    case ptxVectorSelectExpression:
        return isConstantAccessPath( lhs->cases.VectorSelect->arg, storage );

    case ptxVideoSelectExpression:
        return isConstantAccessPath( lhs->cases.VideoSelect->arg, storage );

    case ptxByteSelectExpression:
        return isConstantAccessPath( lhs->cases.ByteSelect->arg, storage );

    case ptxSymbolExpression:
        *storage = lhs->cases.Symbol.symbol->storage;
        return lhs->isConstant;

    case ptxVectorExpression: {
        Bool result = True;
        listTraverse( lhs->cases.Vector.elements, (stdEltFun)isCAP,&result);
        *storage = ptxCreateStorageClass( ptxRegStorage, -1 );
        return result;
    }
    case ptxPredicateExpression:
        return isConstantAccessPath( lhs->cases.Predicate.arg, storage );

    default:
        return False;
    }
    return False;
}
 
ptxExpression ptxCreateAddressOfExpr( ptxExpression lhs , ptxParsingState parseState)
{
    ptxExpression   result;
    ptxStorageClass storage;
    
    stdNEW(result);
    
    result->kind                = ptxAddressOfExpression;
    result->isConstant          = isConstantAccessPath(lhs,&storage);
    result->isLhs               = False;
    // Ideally size of type should be obtain from getPointerTypeSize as it will
    // return appropriate sizes depending on state space (e.g. 16bits for shared/local/const,
    // for generic/global based on address size).
    // However as of now doing that results in assertions in OCG as OCG doesn't handle
    // 16 bit for address and expects DT_UINT. To keep this change only refactoring
    // and w/o impacting functional behavior for now keep size as 4 bytes and visit later
    // FIXME: Fix this issue working with DAG consumption in OCG
    result->type                = ptxCreateIntType(32, False, parseState);
    result->cases.AddressOf.lhs = lhs;
    
    return result;
}
 
/*
 * Function         : Create memory address reference expression
 * Parameters       : arg        (I) address expression
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateAddressRefExpr( ptxExpression arg )
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind                 = ptxAddressRefExpression;
    result->isConstant           = False;
    result->isLhs                = True;
    result->type                 = arg->type;
    result->cases.AddressRef.arg = arg;
    
    return result;
}
 
/*
 * Function         : Create predicate expression
 * Parameters       : neg        (I) True iff. predicate should be negated
 *                  : arg        (I) predicate
 * Function Result  : Requested expression
 */
ptxExpression ptxCreatePredicateExpr( Bool neg, ptxExpression pred , ptxParsingState parseState)
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind                = ptxPredicateExpression;
    result->type                = ptxCreatePredicateType(parseState);
    result->isConstant          = False;
    result->isLhs               = False;
    result->neg                 = neg;
    result->cases.Predicate.arg = pred;
    return result;
}

/*
 * Function         : Create label reference expression
 * Parameters       : name       (I) name of referenced label
 *                    symbtab    (I) context of reference
 *                    sourcePos  (I) source location of reference
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateLabelReferenceExpr( String name, ptxSymbolTable symbtab, msgSourcePos_t sourcePos, ptxParsingState parseState)
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind                           = ptxLabelReferenceExpression;
    result->type                           = ptxCreateLabelType(parseState);
    result->isConstant                     = True;
    result->isLhs                          = False;

    stdNEW(result->cases.LabelReference);
    result->cases.LabelReference->name      = name;
    result->cases.LabelReference->sourcePos = sourcePos;
    return result;
}

/*
 * Function         : Create parameter list expression from element list
 * Parameters       : elements (I) list of element
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateParamListExpr( stdList_t elements, ptxParsingState parseState)
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind                     = ptxParamListExpression;
    result->type                     = ptxCreateParamListType(parseState);
    result->isConstant               = True;
    result->isLhs                    = False;
    result->cases.ParamList.elements = elements;
        
    return result;
}
 
/*
 * Function         : Create sink expression
 * Parameters       : None
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateSinkExpr( void )
{
    ptxExpression result;
    
    stdNEW(result);
    
    result->kind                     = ptxSinkExpression;
    result->type                     = NULL;
    result->isConstant               = False;
    result->isLhs                    = True;
        
    return result;
}

/*
 * Function         : Change Int Constant Expression Size
 * Parameters       : expr   (I) Integer Constant Expression
 *                    size   (I) new size of resulting expression
 * Function Result  : Requested expression
 */
ptxExpression ptxChangeIntConstantExpressionSize(ptxExpression expr, uInt size, ptxParsingState parseState)  {
    ptxExpression result;

    result = ptxCreateIntConstantExpr(expr->cases.IntConstant.i, isSignedInt(expr->type), parseState);
    result->type = ptxCreateIntType(size, isSignedInt(expr->type), parseState);
    return result;
}

/*
 * Function         : Obtain storage class related to
 *                    specified access path.
 * Parameters       : lhs        (I) access path to inspect
 * Function Result  : Requested storage id
 */
ptxStorageClass ptxGetStorage( ptxExpression lhs )
{
    ptxStorageClass result;

    result = ptxCreateStorageClass(ptxUNSPECIFIEDStorage, -1);
    isConstantAccessPath(lhs,&result);
    return result;
}
 

/*
 * Function         : Obtain storage kind related to
 *                    specified access path.
 * Parameters       : lhs        (I) access path to inspect
 * Function Result  : Requested storage kind
 */
ptxStorageKind ptxGetStorageKind( ptxExpression lhs )
{
    ptxStorageClass result;

    result = ptxCreateStorageClass(ptxUNSPECIFIEDStorage, -1);
    isConstantAccessPath(lhs,&result);

    return result.kind;
}
 
/*-------------------------- Symbol Table Functions --------------------------*/

/*
 * Function         : Mangle the symbol passed
 * Parameters       : symbol       (I) symbol to be mangled
 * Function Result  : new mangled name of the original name
 */
String ptxMangleName(ptxSymbol symbol, uInt numScopesOnLine)
{
    String mangledName;

    stdASSERT(numScopesOnLine < 100, ("Number scopes on same line extended its maximum limit"));
    stdASSERT(symbol->sourcePos->lineNo < 100000000, ("Number of lines in input PTX extended its maximum limit"));
    if (numScopesOnLine == 0) {
        // Assuming Max Line number of the input PTX file is a 8-digit no
        mangledName = (String) stdMALLOC(strlen(symbol->unMangledName) 
                                        + 8 + 6);
        sprintf(mangledName, "$__%s__%d", symbol->unMangledName, symbol->sourcePos->lineNo);
    } else {
        // Assuming Max Line number of the input PTX file is a 8-digit no and max scopes on same line would be a 2-digit no.
        mangledName = (String) stdMALLOC(strlen(symbol->unMangledName) 
                                        + 8 + 6 + 2 + 1);
        //If variables having same name are declared on same line in different scopes, the count of scopes is appended to the mangled name
        sprintf(mangledName, "$__%s__%d$%d", symbol->unMangledName, symbol->sourcePos->lineNo, numScopesOnLine);
    }
    return mangledName;
}

/*
 * Function         : Increment global variable numScopesOnLine
 * Parameters       : None
 * Function Result  : None
 */
void ptxIncrementNumScopesOnLine(ptxParseData parseData)
{
    parseData->numScopesOnLine++;
}

/*
 * Function         : Reset global variable numScopesOnLine to zero.
 * Parameters       : None
 * Function Result  : None
 */
void ptxResetNumScopesOnLine(ptxParseData parseData)
{
    parseData->numScopesOnLine  = 0;
}

/*
 * Function         : Determine whether mangling is required
 * Parameters       : storage       (I) storage class of the variable, whose mangling decision must be made 
 * Function Result  : True  iff. symbol is to be mangled
 *                    False iff. symbol is not to be mangled
 */
Bool ptxIsManglingNeeded(ptxStorageClass storage)
{
    return (storage.kind != ptxRegStorage && 
            storage.kind != ptxSregStorage && 
            storage.kind != ptxParamStorage );
}

/*
 * ptxAddToAtomTable()
 *
 */
static void ptxAddToAtomTable(ptxSymbolTableEntry symbolTableEntry, ptxParsingState parseState)
{
    symbolTableEntry->symbol->index = AddIAtom(parseState->atoms, symbolTableEntry->symbol->name);
}

/*
 * Function         : Add all the mangled symbol names in symbol table to Atom Table
 * Parameters       : table      (I) symbol table whose varible contents should be mangled
 *                    parseState (I) parsing state which stores all parsing related information
 */
void ptxAssignIndexToSymbols(ptxSymbolTable table, ptxParsingState parseState)
{
    // Atom table already contains UnMangled Name -> Index Map
    // This is to ensure that even mangled names are added to the Atom Table
    listTraverse(table->VariableSeq, (stdEltFun)ptxAddToAtomTable, parseState);
    listTraverse(table->LabelSeq, (stdEltFun)ptxAddToAtomTable, parseState);
    listTraverse(table->FunctionSeq, (stdEltFun)ptxAddToAtomTable, parseState);
    listTraverse(table->MacroSeq, (stdEltFun)ptxAddToAtomTable, parseState);
    listTraverse(table->MarkerSeq, (stdEltFun)ptxAddToAtomTable, parseState);
}

/*
 * Function         : Create storage class from components
 * Parameters       : kind       (I) kind of storage class
 *                    bank       (I) if applicable, memory bank, or -1 otherwise
 * Function Result  : requested new storage class representation
 */
ptxStorageClass ptxCreateStorageClass( ptxStorageKind kind, Int bank )
{
    ptxStorageClass result;
    result.kind = kind; 
    result.bank = bank;
    return result;
}

/*
 * Function         : Create symbol with source information
 * Parameters       : type         (I) type of symbol
 *                    name         (I) name of symbol
 *                    logAlignment (I) log2 of required alignment of variable
 *                    sourcePos    (I) location in source file where symbol was declared
 * Function Result  : requested new symbol representation
 */
ptxSymbol ptxCreateSymbol(ptxParsingState parseState, ptxType type, String name, uInt logAlignment, uInt64 attributeFlags, msgSourcePos_t sourcePos )
{
    ptxSymbol symbol;
    
    stdNEW(symbol);
    symbol->type      = type;
    symbol->unMangledName = name;
    symbol->name = name;
    symbol->index = AddIAtom(parseState->atoms, symbol->unMangledName);
    symbol->logAlignment = stdMAX( logAlignment, ptxGetTypeLogAlignment(type));
    symbol->sourcePos = sourcePos;
    symbol->attributeFlags = attributeFlags;

    return symbol;
}

/*
 * Function         : Set a symbol attribute
 * Parameters       : symbol       (I) ptxSymbol
 *                    attribute    (I) ptxSymbolAttribute
 */
void ptxSetSymbolAttribute( ptxSymbol symbol, ptxSymbolAttribute attribute )
{
    symbol->attributeFlags |= attribute;
}

/*
 * Function         : Checks if a symbol attribute is set
 * Parameters       : symbol       (I) ptxSymbol
 *                    attribute    (I) ptxSymbolAttribute
 * Function Result  : True iff. symbol has attribute set
 *                    False iff. symbol doesn't have attribute set
 */
Bool ptxCheckSymbolAttribute( ptxSymbol symbol, ptxSymbolAttribute attribute )
{
    return ((symbol->attributeFlags & attribute) != 0);
}

/*
 * Function         : Determine if the symbol is user-defined
 * Parameters       : symbol       (I) symbol to check
 * Function Result  : True  iff. symbol is user symbol
 *                    False iff. symbol is non user symbol
 */
Bool ptxIsUserSymbol( ptxSymbol symbol )
{
    return isPtxUserInput(msgGetFileName(symbol->sourcePos));
}

Bool ptxIsAddressableSymbol(ptxSymbolTableEntry symbol)
{
    return symbol->kind == ptxVariableSymbol
        || symbol->kind == ptxFunctionSymbol
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        || symbol->kind == ptxLabelSymbol
#endif
        ;
}

/*
 * Function         : Create symbol table
 * Parameters       : parent       (I) parent symbol table
 * Function Result  : new symbol table
 */
ptxSymbolTable ptxCreateSymbolTable(ptxSymbolTable parent )
{
    ptxSymbolTable result;
    
    stdNEW(result);
    
    result->parent            = parent;
    result->symbolIndexMap    = mapNEW(uInt,64);
    result->opaques           = mapNEW(String,64);

    listXInit( result->LabelSeq    );
    listXInit( result->LabelRefSeq );
    listXInit( result->VariableSeq );
    listXInit( result->ConstrainedExternalVarSeq);
    listXInit( result->InitializableVarSeq);
    listXInit( result->symbolsWithAttributeFlags);
    listXInit( result->FunctionSeq );
    listXInit( result->MarkerSeq   );
    listXInit( result->MacroSeq    );
    listXInit( result->VariablesToPromoteSeq );
    listXInit( result->statements );

    return result;
}

static Bool checkVarRequiresPromotion(ptxSymbolTableEntry result)
{
    // global or constant user variables with ptxLocalScope will be promoted
    if (result->storage.kind != ptxGlobalStorage && result->storage.kind != ptxConstStorage) {
        return False;
    }
    if (result->scope != ptxLocalScope) {
        return False;
    }
    if (!ptxIsUserSymbol(result->symbol)) {
        return False;
    }
    if (result->storage.kind == ptxConstStorage) {
        return(!stdIS_PREFIX("__lwdart_", result->symbol->name));
    }
    return True;
}

/*
 * Function         : Add definition of variable to symbol table
 * Parameters       : symbtab      (I) symbol table to add to
 *                    symbol       (I) symbol to add
 *                    scope        (I) scope of symbol
 *                    storage      (I) storage where symbol must be allocated,
 *                    initialValue (I) Initial value specification, or NULL
 *                    range        (I) range for paramerized variable
 * Function Result  : True  iff. symbol could be added 
 *                    False iff. symbol name clashes with current contents of symbol table
 */
Bool ptxAddVariableSymbol(ptxSymbolTable symbtab, ptxSymbol symbol, ptxDeclarationScope scope, 
                          ptxStorageClass storage, ptxInitializer initialValue, uInt range)
{
    ptxSymbolTableEntry old = mapApply(symbtab->symbolIndexMap, (Pointer)(Address)symbol->index);

    if (old) {
        return False;
    } else {
        ptxSymbolTableEntry result;

        stdNEW(result);
        result->kind         = ptxVariableSymbol;
        result->symbol       = symbol;
        result->scope        = scope;

        result->symbtab      = symbtab;
        result->storage      = storage;
        result->initialValue = initialValue;
        result->range        = range;
        result->aux          = NULL;

        mapDefine(symbtab->symbolIndexMap, (Pointer)(Address)symbol->index, result);
        if (checkVarRequiresPromotion(result)) {
            listXPutAfter(symbtab->VariablesToPromoteSeq, result);
        } else {
            listXPutAfter( symbtab->VariableSeq, result );
        }

        if (scope == ptxExternalScope && !(storage.kind == ptxSregStorage || storage.kind == ptxRegStorage)) {
            listXPutAfter(symbtab->ConstrainedExternalVarSeq, result);
        }

        if (storage.kind == ptxGlobalStorage || (storage.kind == ptxConstStorage && !stdIS_PREFIX("__lwdart_", symbol->name))) {
            listXPutAfter(symbtab->InitializableVarSeq, result);
        }

        if (symbol->attributeFlags) {
            listXPutAfter(symbtab->symbolsWithAttributeFlags, result);
        } 

        if (range > 0) { // for parameterized variables, allocate an array of pointers to symtab entries
            stdNEW(result->aux);
            stdNEW(result->aux->funcProtoAttrInfo);
            stdNEW_N(result->aux->parameterizedVars, range);
        }
        return True;
    }
}

/*
 * ptxAddParameterizedVariableSymbol()
 *
 */

static void ptxAddParameterizedVariableSymbol(ptxSymbolTable symbtab, ptxSymbol symbol, 
                                              ptxSymbolTableEntry paramEntry, uInt idx)

{
    ptxSymbolTableEntry result;

    stdNEW(result);
    result->kind         = ptxVariableSymbol;
    result->symbol       = symbol;
    result->scope        = paramEntry->scope;

    result->symbtab      = symbtab;
    result->storage      = paramEntry->storage;
    result->initialValue = NULL;
    result->range        = 0;
    result->aux          = NULL;

    stdASSERT(idx < paramEntry->range, ("not a paramaterized var"));
    stdASSERT(paramEntry->aux->parameterizedVars[idx] == NULL, ("already exists"));
    paramEntry->aux->parameterizedVars[idx] = result;
    if (checkVarRequiresPromotion(result)) {
        listXPutAfter(symbtab->VariablesToPromoteSeq, result);
    } else {
        listXPutAfter( symbtab->VariableSeq, result );
    }
}

/*
 * Function         : Add definition of function to symbol table
 * Parameters       : symbtab    (I) symbol table to add to
 *                    symbol     (I) symbol to add
 *                    isEntry    (I) True iff. function is a CTA entry
 *                    isInlineFunc (I) True iff. function gets inlined
 *                    scope      (I) scope of symbol
 *                    body       (I) definition of function's contents
 *                    rparams    (I) function's return parameters
 *                    fparams    (I) function's formal parameters
 *                    hasAllocatedParams (I) True iff. function has allocated parameters
 *                    hasNoReturn (I) True iff .noreturn specfied, applicable only on device functions
 *                    isUnique   (I) True iff .unique specfied, applicable only on macro functions
 *                    retAddrAllocno(I) register specifying return address
 *                    scratchRegs(I) List of scratch registers
 * Function Result  : True  iff. symbol could be added 
 *                    False iff. symbol name clashes with current contents of symbol table
 */
Bool ptxAddFunctionSymbol(ptxSymbolTable symbtab, ptxSymbol symbol, Bool isEntry,
                          Bool isInlineFunc, ptxDeclarationScope scope,
                          ptxSymbolTable body, stdList_t rparams, stdList_t fparams,
                          Bool hasAllocatedParams, Bool hasNoReturn, Bool isUnique,
                          uInt retAddrAllocno, stdList_t scratchRegs)
{
    ptxSymbolTableEntry old= mapApply(symbtab->symbolIndexMap, (Pointer)(Address)symbol->index);

    if (old) {
        return False;
    } else {
        ptxSymbolTableEntry result;

        stdNEW(result);
        result->kind       = ptxFunctionSymbol;
        result->symbol     = symbol;
        result->scope      = scope;
        result->symbtab    = symbtab;

        stdNEW(result->aux);
        stdNEW(result->aux->funcProtoAttrInfo);
        listXInit( result->aux->pragmas );
        result->aux->isEntry      = isEntry;
        result->aux->isInlineFunc   = isInlineFunc;
        result->aux->funcProtoAttrInfo->hasAllocatedParams = hasAllocatedParams;
        result->aux->funcProtoAttrInfo->hasNoReturn  = hasNoReturn;
        // Flag isUnique indicates there will be single copy of function
        // in noCloning, SC and EWP compilation modes
        result->aux->isUnique     = isUnique;
        result->aux->funcIndex    = ~0;
        result->aux->maxnreg      = 0;
        result->aux->maxntid[0]   = 0;
        result->aux->maxntid[1]   = 0;
        result->aux->maxntid[2]   = 0;
        result->aux->minnctapersm = 0;
        result->aux->reqntid[0]   = 0;
        result->aux->reqntid[1]   = 0;
        result->aux->reqntid[2]   = 0;
        result->aux->body         = body;
        result->aux->funcProtoAttrInfo->rparams      = rparams;
        result->aux->funcProtoAttrInfo->fparams      = fparams;
        result->aux->funcProtoAttrInfo->scratchRegs  = scratchRegs;
        result->aux->funcProtoAttrInfo->retAddrAllocno = retAddrAllocno;
        result->aux->funcProtoAttrInfo->numAbiParamRegs = UNSPECIFIED_ABI_PARAM_REGS; // initial value, will be changed based pragma abi_param_reg
        result->aux->funcProtoAttrInfo->firstParamReg     = UNSPECIFIED_ABI_REG;
        result->aux->usesWMMAInstrs = False;
        result->aux->aliasee        = NULL;
        result->aux->localMaxNReg   = 0;
        result->aux->funcProtoAttrInfo->retAddrBeforeParams = UNSPECIFIED_RET_ADDR_BEFORE_PARAMS;
        result->aux->funcProtoAttrInfo->retAddrReg          = UNSPECIFIED_ABI_REG;
        result->aux->funcProtoAttrInfo->retAddrUReg         = UNSPECIFIED_ABI_REG;
        result->aux->funcProtoAttrInfo->relRetAddrReg       = UNSPECIFIED_ABI_REG;

        mapDefine(symbtab->symbolIndexMap, (Pointer)(Address)symbol->index, result);
        
        listXPutAfter( symbtab->FunctionSeq, result );

        return True;
    }
}

void ptxSetUniqueFuncIndex(ptxSymbolTableEntry funcSym, ptxParsingState parseState)
{
    stdASSERT(funcSym->kind == ptxFunctionSymbol && funcSym->aux->body,
                ("Defined function expected"));
    funcSym->aux->funcIndex = parseState->nextUniqueFuncIndex++;
}

uInt ptxGetDefinedFunctionCount(ptxParsingState parseState)
{
    return parseState->nextUniqueFuncIndex;
}

/*
 * Function         : Add label definition to symbol table, pointing to its current instruction position
 * Parameters       : symbtab    (I) symbol table to add to
 *                    symbol     (I) symbol to add
 * Function Result  : True  iff. symbol could be added 
 *                    False iff. symbol name clashes with current contents of symbol table
 */
Bool ptxAddLabelSymbol(ptxSymbolTable symbtab, ptxSymbol symbol)
{
    ptxSymbolTableEntry old= mapApply(symbtab->symbolIndexMap, (Pointer)(Address)symbol->index);

    if (old) {
        return False;
    } else {
        ptxSymbolTableEntry result;

        stdNEW(result);
        result->kind        = ptxLabelSymbol;
        result->symbol      = symbol;
        result->scope       = ptxLocalScope;
        result->symbtab     = symbtab;
        result->storage     = ptxNOSTORAGECLASS;

        stdNEW(result->aux);
        stdNEW(result->aux->funcProtoAttrInfo);
        listXInit( result->aux->pragmas );  // not needed by Label
        result->aux->listIndex = symbtab->numStmts;

        mapDefine(symbtab->symbolIndexMap, (Pointer)(Address)symbol->index, result);

        listXPutAfter( symbtab->LabelSeq, result );

        return True;
    }
}

/*
 * Function         : Add macro definition to symbol table
 * Parameters       : symbtab    (I) symbol table to add to
 *                    symbol     (I) symbol to add
 *                    formals    (I) list of formal macro parameter names
 *                    body       (I) macro body string
 *                    sourcePos  (I) start source position of body definition.
 * Function Result  : True  iff. symbol could be added 
 *                    False iff. symbol name clashes with current contents of symbol table
 */
Bool ptxAddMacroSymbol(ptxSymbolTable symbtab, ptxSymbol symbol, stdList_t formals, String body, msgSourcePos_t sourcePos)
{
    ptxSymbolTableEntry old= mapApply(symbtab->symbolIndexMap, (Pointer)(Address)symbol->index);

    if (old) {
        return False;
    } else {
        ptxSymbolTableEntry result;

        stdNEW(result);
        result->kind        = ptxMacroSymbol;
        result->symbol      = symbol;
        result->scope       = ptxGlobalScope;
        result->symbtab     = symbtab;
        result->storage     = ptxNOSTORAGECLASS;

        stdNEW(result->aux);
        stdNEW(result->aux->funcProtoAttrInfo);
        listXInit( result->aux->pragmas );  // not needed by Macro
        result->aux->funcProtoAttrInfo->fparams     = formals;
        result->aux->mbody                     = body;
        result->aux->mbodyPos                  = sourcePos;

        mapDefine(symbtab->symbolIndexMap, (Pointer)(Address)symbol->index, result);

        listXPutAfter( symbtab->MacroSeq, result );
        
        return True;
    }
}

/*
 * Function         : Add opaque definition to symbol table
 * Parameters       : symbtab    (I) symbol table to add to
 *                    symbol     (I) type symbol to add
 * Function Result  : True  iff. type could be added 
 *                    False iff. type name clashes with current contents of symbol table
 */
Bool ptxAddOpaque(ptxSymbolTable symbtab, ptxSymbol symbol )
{
    ptxType old= mapApply(symbtab->opaques,symbol->unMangledName);

    if (old) {
        return False;
    } else {
        mapDefine(symbtab->opaques,symbol->unMangledName,symbol);

        return True;
    }
}

/*
 * Function         : Lookup symbol in symbol table
 * Parameters       : symbtab         (I) symbol table to inspect
 *                    unMangledName   (I) original or un-mangled name of symbol to lookup
 *                    inspectParents  (I) ancestor symbol tables will be inspected
 *                                         iff. inspectParents equals True.
 *                    parseState      (I) parsing state which stores all parsing related information
 * Function Result  : requested symbol, or NULL when not found
 */
ptxSymbolTableEntry ptxLookupSymbol(ptxSymbolTable symbtab, String unMangledName, Bool inspectParent, ptxParsingState parseState)
{
    ptxSymbolTableEntry result;
    uInt idx, suffixStart, symbolIndex;
    ptxParamVarSaveRec save;

    if (symbtab == NULL) {
        return NULL;
    }

    if (ptxIsParameterizedVariableName(unMangledName, &idx, &suffixStart)) {
        ptxGetParameterizedVariableName(unMangledName, suffixStart, &save);
        symbolIndex = LookUpIString(parseState->atoms, unMangledName);
        result = mapApply(symbtab->symbolIndexMap, (Pointer)(Address)symbolIndex);
        ptxRestoreParameterizedVariableName(unMangledName, suffixStart, &save);
        if (result && idx < result->range)
            return result->aux->parameterizedVars[idx];
    }

    symbolIndex = LookUpIString(parseState->atoms, unMangledName);    
    result = mapApply(symbtab->symbolIndexMap, (Pointer)(Address)symbolIndex);
    if (!result && inspectParent)
        result = ptxLookupSymbol(symbtab->parent, unMangledName, inspectParent, parseState);
    return result;
}

static Bool isGlobalMemoryStorage( ptxStorageClass storage )
{
    switch (storage.kind) {
    case ptxCodeStorage        : stdASSERT(False, ("Unexpected variable storage") );

    case ptxConstStorage       :
    case ptxGenericStorage     :
    case ptxGlobalStorage      :
    case ptxLocalStorage       :
    case ptxParamStorage       :
    case ptxSharedStorage      :
    case ptxSurfStorage        :
    case ptxTexStorage         : return True;

    case ptxRegStorage         :
    case ptxSregStorage        :
    default                    : return False;
    }
}


/*
 * Function         : Store debug information in variable's symbol table entry
 * Parameters       : name            (I) name of symbol to lookup
 *                    symbtab         (I) symbol table to inspect
 *                    scope           (I) scope of symbol
 *                    storage         (I) storage where symbol must be allocated,
 *                    state           (I) parsing state
 * Function Result  :
 */
void ptxSetVariableDebugInfo(String name, ptxSymbolTable symbtab, ptxDeclarationScope scope, ptxStorageClass storage, ptxParsingState state)
{
   /*
    * For debugging purposes, map variable in a virtual ptx memory space
    * corresponding to each of the storage classes. It is the job of OCG
    * to translate this virtual address into a real hardware address:
    */
    if ( scope != ptxExternalScope && isGlobalMemoryStorage(storage) ) {
        ptxSymbolTableEntry   entry       = ptxLookupSymbol( symbtab, name, True, state );
        entry->virtualAddress             = state->virtualSize[storage.kind];
        state->virtualSize[storage.kind] += ptxGetTypeSizeInBytes(entry->symbol->type);

        listXPutAfter(state->dwarfLocations[storage.kind].l, entry );
    }
}

/*
 * Function         : Check whether symbol represents parameterized auxiliary variable
 * Parameters       : ptxsymEnt       (I) symbol table entry of variable to check
 * Function Result  : True if symbol is parameterized auxiliary variable
 */
Bool ptxIsParameterizedAuxSymbol(ptxSymbolTableEntry ptxsymEnt)
{
    String name = ptxsymEnt->symbol->name;
    return (ptxsymEnt->range > 0 && name && name[strlen(name) - 1] == '<');
}

/*
 * Function         : Check whether string has the form of a parameterized variable name
 * Parameters       : name            (I) name of variable
 *                    suffix          (O) numeric suffix, returned as an integer
 *                    suffixStart     (O) starting position in 'name' of this numeric suffix
 * Function Result  : True if string has a numeric suffix
 */
Bool ptxIsParameterizedVariableName( String name, uInt *suffix, uInt *suffixStart )
{
    char letter;
    Bool InDigit;
    uInt lSuffix, ii, lSuffixStart;

    if (!name || isdigit(*name))
        return False;

    ii = 1;
    InDigit = False;
    lSuffix = 0;
    lSuffixStart = 0;

    for (ii = 1; (letter = name[ii]) != 0; ii++) {
        if (isdigit(letter)) {
            lSuffix = 10 * lSuffix + (letter - '0');
            if (!InDigit)
                lSuffixStart = ii;
            InDigit = True;
        } else {
            InDigit = False;
            lSuffix = 0;
            lSuffixStart = 0;
        }
    }

    if (suffix) *suffix = lSuffix;
    if (suffixStart) *suffixStart = lSuffixStart;
    return InDigit;
}

/*
 * Function         : Check whether string has the form of a parameterized variable name
 * Parameters       : name            (IO) name of variable. It will be modified inplace.
 *                    suffixStart     (I)  numeric suffix, returned as an integer
 *                    save            (O)  saved information to restore the name
 * Function Result  : 
 */
void ptxGetParameterizedVariableName( String name, uInt suffixStart, ptxParamVarSave save )
{
    stdASSERT( ptxIsParameterizedVariableName(name, NULL, NULL), ("not a parameterized variable") );
    save->save[0] = name[suffixStart];
    save->save[1] = name[suffixStart + 1];

    name[suffixStart] = '<';
    name[suffixStart + 1] = 0; // terminate with '<'
}

/*
 * Function         : colwert the name of a parameterized variable from canonical form back to original name
 * Parameters       : name            (IO) name of variable. It will be modified inplace.
 *                    suffixStart     (I)  numeric suffix, returned as an integer
 *                    save            (O)  saved information returned by previous ptxGetParameterizedVariableName call
 * Function Result  :
 */
void ptxRestoreParameterizedVariableName( String name, uInt suffixStart, ptxParamVarSave save )
{
    stdASSERT(name && name[strlen(name) - 1] == '<', ("not a canonicalized name") );
    name[suffixStart] = save->save[0];
    name[suffixStart + 1] = save->save[1];
}

/*
 * Function         : Lookup symbol in symbol table, and lazily create parameterized variables
 * Parameters       : symbtab         (I) symbol table to inspect
 *                    name            (I) name of symbol to lookup
 *                    inspectParents  (I) ancestor symbol tables will be inspected
 *                                         iff. inspectParents equals True.
 *                    state           (I) parsing state
 * Function Result  : requested symbol, or NULL when not found
 */
ptxSymbolTableEntry ptxLookupSymbolLazyCreate(ptxSymbolTable symbtab, String name, Bool inspectParent, ptxParsingState state)
{
    uInt idx, suffixStart, symbolIndex;
    ptxSymbolTableEntry result;
    ptxParamVarSaveRec save;

    if (symbtab == NULL)
        return NULL;

    if (!ptxIsParameterizedVariableName(name, &idx, &suffixStart)) {
        return ptxLookupSymbol(symbtab, name, inspectParent, state);
    }

    ptxGetParameterizedVariableName(name, suffixStart, &save);
    symbolIndex = LookUpIString(state->atoms, name);
    result = mapApply(symbtab->symbolIndexMap, (Pointer)(Address)symbolIndex);
    ptxRestoreParameterizedVariableName(name, suffixStart, &save);

    if (result && idx < result->range) {
        if (result->aux->parameterizedVars[idx] == NULL) {
            ptxSymbol old = result->symbol;
            ptxSymbol var = ptxCreateSymbol(state, old->type, name, old->logAlignment, 0, old->sourcePos );
            ptxAddParameterizedVariableSymbol( symbtab, var, result, idx );
            ptxSetVariableDebugInfo( name, symbtab, result->scope, result->storage, state );
        }
        return result->aux->parameterizedVars[idx];
    }

    symbolIndex = LookUpIString(state->atoms, name);
    result = mapApply(symbtab->symbolIndexMap, (Pointer)(Address)symbolIndex);
    if (!result && inspectParent)
        result = ptxLookupSymbolLazyCreate( symbtab->parent, name, inspectParent, state );
    return result;
}

/*
 * Function         : Lookup opaque struct in symbol table
 * Parameters       : symbtab         (I) symbol table to inspect
 *                    name            (I) name of symbol to lookup
 *                    inspectParents  (I) ancestor symbol tables will be inspected
 *                                         iff. inspectParents equals True.
 * Function Result  : requested symbol, or NULL when not found
 */
ptxSymbol ptxLookupOpaque(ptxSymbolTable symbtab, String name, Bool inspectParent )
{
    if (symbtab == NULL) {
        return NULL;
    } else {
        ptxSymbol result= mapApply(symbtab->opaques,name);
        
        if (!result && inspectParent) {
            return ptxLookupOpaque(symbtab->parent,name,inspectParent);
        } else {
            return result;
        }
    }
}

/*
 * Function         : Add statement at the end of statement list of specified symbol table
 * Parameters       : symbtab    (I) symbol table to add to
 *                    statement  (I) statement to add
 * Function Result  :
 */
void ptxAddStatement(ptxSymbolTable symbtab, ptxStatement statement )
{
    symbtab->numStmts++;
    listXPutAfter(symbtab->statements,statement);
}



/*---------------------- Statement Constructor Functions ---------------------*/
/*
 * Function         : Create instruction statement representation from components
 * Parameters       : instruction (I) instruction to promote to statement
 * Function Result  : requested new statement representation
 */
ptxStatement ptxCreateInstructionStatement( ptxInstruction instruction )
{
    ptxStatement result;
    
    stdNEW(result);
    
    result->kind                          = ptxInstructionStatement;
    result->cases.Instruction.instruction = instruction;
    
    return result;
}

/*
 * Function         : Create pragma statement representation from components
 * Parameters       : pragmas (I) list of pragma strings to promote to statement
 * Function Result  : requested new statement representation
 */
ptxStatement ptxCreatePragmaStatement( stdList_t pragmas )
{
    ptxStatement result;
    
    stdNEW(result);
    
    result->kind                          = ptxPragmaStatement;
    listXInit(result->cases.Pragma.pragmas);

    while (pragmas) {
        String line = pragmas->head;
        pragmas     = pragmas->tail;

        listXPutAfter( result->cases.Pragma.pragmas, line );
    }
    return result;
}

/*
 * Function         : Create metadata value of integer type
 * Parameters       : integer
 * Function Result  : requested metadata value
 */
ptxMetaDataValue ptxCreateMetadataValueInt( uInt val )
{
    ptxMetaDataValue metadata;
    stdNEW(metadata);

    metadata->metadataKind = ptxMetaDataValueInt;
    metadata->cases.val    = val;
    return metadata;
}

/*
 * Function         : Create metadata value of index type
 * Parameters       : metadata index
 * Function Result  : requested metadata value
 */
ptxMetaDataValue ptxCreateMetadataValueIndex( uInt index )
{
    ptxMetaDataValue metadata;
    stdNEW(metadata);

    metadata->metadataKind = ptxMetaDataValueIndex;
    metadata->cases.metadataIndex = index;
    return metadata;
}

/*
 * Function         : Create metadata value of string type
 * Parameters       : string
 * Function Result  : requested metadata value
 */
ptxMetaDataValue ptxCreateMetadataValueString( String str )
{
    ptxMetaDataValue metadata;
    stdNEW(metadata);

    metadata->metadataKind = ptxMetaDataValueString;
    metadata->cases.str    = stdCOPYSTRING(str);
    return metadata;
}

/*---------------------------- Parsing Functions -----------------------------*/

/*
 * Function         : Incremental parse of assembly source file.
 *                    The result of parsing will be added as a new symbol table
 *                    to the parsedObjects set in the state parameter.
 *                    global symbol resolution will take place in order to
 *                    resolve 'extern' symbols to definitions in other parsed files.
 *                    Hence, this function also serves as a linker.
 * Parameters       : inputFileName  (I)  name of ptx assembly file to parse
 *                    obfuscationKey (I)  value by which the ptx file was obfuscated, or zero
 *                    object         (IO) Constructed parsing state
 *                    debugInfo      (I)  True iff. debug info needs be generated
 *                    debugOneLineBB (I)  True iff. a basic block is needed per source line
 *                    lineInfo       (I)  True iff. line info needs to be generated
 * Function Result  : True iff. parsing succeeded
 */
Bool ptxParseInputFile( String inputFileName, uInt32 obfuscationKey, ptxParsingState object, Bool debugInfo, Bool debugOneLineBB, Bool lineInfo )
{
    stdMemSpace_t savedSpace = stdSwapMemSpace( object->memSpace );

    msgTry(True) {
            ptxInitScanner(object);
            
            if (obfuscationKey) {
                object->ptxObfuscation= stdCreateObfuscation(obfuscationKey);
            } else {
                object->ptxObfuscation= NULL;
            }

            object->ptxin              = fopen(inputFileName, object->ptxObfuscation?"rb":"r");
            if (object->ptxObfuscation) {
                // find size of file, because obfuscation may create false EOF
                fseek(object->ptxin, 0, SEEK_END);
                object->ptxLength = ftell(object->ptxin);
                fseek(object->ptxin, 0, SEEK_SET); // go back to beginning of file
            }
            object->ptxDebugInfo       = debugInfo || lineInfo;
            object->ptxDebugOneLineBB  = debugOneLineBB;
            object->ptxStringInput     = False;

            stdCHECK( object->ptxin, (ptxMsgInputFileNotFound, inputFileName) ) {
                object->parseData->target_arch = object->target_arch = NULL;  // reset target_arch
                object->macroSymbolTable = object->macroSymbols;
                object->ptxAllowMacros   = False;
                object->globalSymbolTable = object->globalSymbols;

                object->inputFileName    = inputFileName;

                ptxInitLexState(object);
                ptxparse(object->scanner, object);
                ptxDestroyLexState(object);

                setInsert( object->parsedObjects, object->objectSymbolTable );
            }
    }
    msgOtherwise {
    }
    msgEndTry;

    if (object->ptxin) { fclose(object->ptxin); }
    
    if (object->ptxObfuscation) {
        stdDeleteObfuscation(object->ptxObfuscation);
    }

    stdSwapMemSpace(savedSpace);

    // OPTIX_HAND_EDIT reset error state
    Bool noErrorsFound = !msgErrorsFound();
    msgSetError( False );
    return noErrorsFound;
}

static void deObfuscateMacro(ptxParsingState state, char *obfuscatedMacro, uInt obfuscationKey, int size)
{
    char *ptr;
    int i;
    stdObfuscationState obstate;

    if (obfuscationKey) {
        obstate= stdCreateObfuscation(obfuscationKey);
    } else {
        obstate= NULL;
    }

    state->deobfuscatedMacro = stdMALLOC(size);

    ptr = (char*)obfuscatedMacro;
    for(i = 0; i < size; i++)
    {
        if (obstate) {
            state->deobfuscatedMacro[i] = stdDeobfuscate(obstate, ptr[i]);
        }
    }

    if (obfuscationKey) {
        stdDeleteObfuscation(obstate);
    }
}

// OPTIX_HAND_EDIT
/*
 * Function         : Incremental parse of assembly, directly from ptx string.
 *          i         The result of parsing will be added as a new symbol table
 *                    to the parsedObjects set in the state parameter.
 *                    global symbol resolution will take place in order to
 *                    resolve 'extern' symbols to definitions in other parsed files.
 *                    Hence, this function also serves as a linker.
 * Parameters       : ident          (I)  identifier for error handling. start with '<' for non user input
 *                    ptx            (I)  ptx assembly program as string
 *                    obfuscationKey (I)  value by which the ptx string was obfuscated, or zero
 *                    object         (IO) Parsing state to receive result
 *                    debugInfo      (I)  True iff. debug info needs be generated
 *                    debugOneLineBB (I)  True iff. a basic block is needed per source line
 *                    lineInfo       (I)  True iff. line info needs to be generated
 *                    ptxStringLength(I)  length of ptx input
 *                    decrypter      (I)  Instance of OptiX 7 EncryptionManager
 *                    decryptionCB   (I)  Decryption callback
 * Function Result  : True iff. parsing succeeded
 */
Bool ptxParseInputString( cString ident, String ptx, uInt32 obfuscationKey, ptxParsingState object, Bool debugInfo, Bool debugOneLineBB, Bool lineInfo, uInt32 ptxStringLength, void* decrypter, GenericCallback decryptionCB )
{
    stdMemSpace_t savedSpace = stdSwapMemSpace( object->memSpace );

    msgTry(True) {

            if (obfuscationKey) {
                object->ptxObfuscation= stdCreateObfuscation(obfuscationKey);
                object->ptxLength = ptxStringLength;
            } else {
                object->ptxObfuscation= NULL;
            }

            {
                ptxInitScanner(object);
                object->ptxin              = NULL;
                object->ptxDebugInfo       = debugInfo || lineInfo;
                object->ptxDebugOneLineBB  = debugOneLineBB;
                object->ptxStringInput     = True;
                object->inputFileName      = "";
                object->parseData->target_arch = object->target_arch = NULL;  // reset target_arch
                object->globalSymbolTable  = object->globalSymbols;
                object->macroSymbolTable   = object->macroSymbols;
                object->ptxAllowMacros     = !(isPtxUserInput(ident));  // allow macros for non-user input

                ptxInitLexState(object);
                // OPTIX_HAND_EDIT
                ptxPushInput(object, ptx, ptxStringLength, object->ptxObfuscation,ident, 1, decrypter, decryptionCB);
                ptxparse(object->scanner, object);
                ptxDestroyLexState(object);

                setInsert( object->parsedObjects, object->objectSymbolTable );
            }
    }
    msgOtherwiseSelect
    msgCatch(ptxMsgVersionUnsupported) { msgPropagate(); }
    msgDefault { }
    msgEndTry;

    stdSwapMemSpace(savedSpace);

    // OPTIX_HAND_EDIT reset error state
    Bool noErrorsFound = !msgErrorsFound();
    msgSetError( False );
    return noErrorsFound;
}


/*
 * Function         : Create new, empty parsed object, to incrementally
 *                    fill by repeated calls to ptxParseInputFile.
 * Parameters       : gpuInfo        (I)  spec of gpu to parse for
 * Function Result  : Fresh, empty ptx parsed object
 */

extern const unsigned long long ptxFmtTesla[];
extern const unsigned long long ptxInstructionMacrosFermi[];
extern const unsigned long long ptxFmtFermi[];
extern const int ptxFmtFermiSize;

    static void addSymbol( ptxSymbolTableEntry e, stdList_t *l )
    {
        listAddTo(e->symbol,l);
    }

    static stdList_t getFieldList( ptxSymbolTable fields )
    {
        stdList_t result= NULL;

        listTraverse( fields->VariableSeq, (stdEltFun)addSymbol, &result );

        return listReverse(result);
    }

static ptxParseData createParseData(void)
{
    ptxParseData result;
    stdNEW(result);
    result->isF64Allowed = False;
    result->moduleScopeNumAbiParamReg = UNSPECIFIED_ABI_PARAM_REGS;
    result->moduleScopeRetAddrBeforeParams = UNSPECIFIED_RET_ADDR_BEFORE_PARAMS;
    result->moduleScopeRetAddrReg = UNSPECIFIED_ABI_REG;
    result->moduleScopeRetAddrUReg = UNSPECIFIED_ABI_REG;
    result->moduleScopeRelRetAddrReg = UNSPECIFIED_ABI_REG;
    result->moduleScopeScratchRRegs = UNSPECIFIED_ABI_REGS;
    result->moduleScopeScratchBRegs = UNSPECIFIED_ABI_REGS;
    result->moduleScopeFirstParamReg = UNSPECIFIED_ABI_REG;
    result->moduleScopeCoroutinePragma = False;
    result->doubleUse = False;
    result->numScopesOnLine = 0;
    result->macro_stack_ptr = 0;
    result->ptxCount = 0;
    result->isTexModeIndependent = False;
    result->version = NULL;
    result->target_arch = NULL;
    result->ptxfilename = NULL;
    result->deobfuscatedStringMapPtr = NULL;
    return result;
}

ptxParsingState ptxCreateEmptyState(void  *ptxInfo,
                                    gpuFeaturesProfile gpuInfo,
                                    IAtomTable* lAtoms,
                                    stdMap_t* lfuncIndexToSymbolMap,
                                    void (*AddExtraPreProcessorMacroFlags)(ptxParsingState state, void *ptxInfo),
                                    uInt generatePrefetchSizeSeed,
                                    cString extDescFileName, cString extDescAsString, stdMap_t* deobfuscatedStringMapPtr)
{
    ptxStorageKind  s;
    ptxParsingState result;
    stdMemSpace_t   memSpace   = memspCreate( "PTX parsing state", stdLwrrentMemspace, memspDEFAULT_PAGESIZE);
    stdMemSpace_t   savedSpace = stdSwapMemSpace( memSpace );
    String ptxMajorVersion, ptxMinorVersion;

    
   /*
    * Initialize the parser.
    */

   /* -.- */

    stdNEW(result);

    result->scanner             = NULL;
    result->version             = NULL;
    result->target_arch         = NULL;
    result->addr_size           = 0;    // keep invalid to identify if address_size is specified or not
    result->max_target          = 0;
    result->target_opts         = mapNEW(String,64);
    result->memSpace            = memSpace;
    result->gpuInfo             = gpuInfo;
    result->preprocessorMacros  = mapNEW(String,64);
    result->atoms               = lAtoms;
    result->inputFileName       = NULL;

    result->warn_on_double_demotion = False;

    result->usesModuleScopedRegOrLocalVars = False;
    result->moduleScopedRegOrLocalVarName  = NULL;
    result->usesFuncPointer                = False;
    result->callsExternFunc                = False;
    result->usesFuncWithMultipleRets       = False;
    result->funcWithMultipleRetsName       = NULL;
    result->enablePtxDebug                 = False;
    result->enableLineInfoGeneration       = False;
    // This flag will be set to False in parsing if PTX is non-empty
    result->isEmptyUserPTX                 = True;
    result->nextUniqueFuncIndex            = 0;
    result->ptxBuiltInSourceStruct         = msgCreateSourceStructure("<builtin>");
    result->ptxFileSourceStruct            = msgCreateSourceStructure("");
    result->generatePrefetchSizeSeed       = generatePrefetchSizeSeed;
    result->parseData                      = createParseData();
    result->parseData->deobfuscatedStringMapPtr       = deobfuscatedStringMapPtr;
    ptxDefineInstructionTemplates(result->parseData, extDescFileName, extDescAsString);
    mapDefine(result->preprocessorMacros, "GPU_ARCH", gpuInfo->internalName);

    // set flag for macro expansion of intrinsincs
    AddExtraPreProcessorMacroFlags(result, ptxInfo);

    ptxMajorVersion = stdMALLOC(3);
    sprintf(ptxMajorVersion, "%d", ptxGetLatestMajorVersion());
    ptxMinorVersion = stdMALLOC(3);
    sprintf(ptxMinorVersion, "%d", ptxGetLatestMinorVersion());
    mapDefine(result->preprocessorMacros, "PTX_MAJOR_VERSION", ptxMajorVersion);
    mapDefine(result->preprocessorMacros, "PTX_MINOR_VERSION", ptxMinorVersion);

    initMacroElwVar(result->parseData, PTX_MAJOR_VERSION, ptxGetLatestMajorVersion());
    initMacroElwVar(result->parseData, PTX_MINOR_VERSION, ptxGetLatestMinorVersion());

    result->globalSymbols          = ptxCreateSymbolTable(NULL);
    result->globalSymbols->data    = (Pointer)ptxGlobalScope;
    result->macroSymbols           = ptxCreateSymbolTable(NULL);
    result->macroSymbols->data     = (Pointer)ptxGlobalScope;
    result->parsedObjects          = setNEW(Pointer,   8);
    result->ptxToSourceLine.instructionMap  = rangemapCreate();
    result->ptxToSourceLine.labelMap        = mapNEW(uInt, 8192);
    result->ptxToSourceLine.inlinedLocMap   = mapNEW(pInt64, 1024);
    result->ptxToSourceLine.functionPtxLineRangeMap = rangemapCreate();
    result->dwarfFiles             = mapNEW(uInt,      8);
    result->internalDwarfLabel     = mapNEW(String, 1024);
    result->locationLabels         = mapNEW(String, 1024);
    result->sassAddresses          = mapNEW(String, 1024);
    result->paramOffset            = mapNEW(String, 1024);
    result->symbolNames            = mapNEW(Pointer,8192);
    result->symbolNamesCnt         = mapNEW(String, 8192);
    result->arrayInit              = mapNEW(String, 1024);
    result->dwarfLiveRangeMap      = mapNEW(String, 8192);
    result->parsingTTUBlock        = False;
    result->funcIndexToSymbolMap   = lfuncIndexToSymbolMap;
    result->ptxLength              = 0;
    listXInit( result->pragmas );
    listXInit( result->dwarfBytes );
    listXInit( result->dwarfSections );
    listXInit( result->nonMercFeaturesUsed );
    result->metadataSection = NULL;
    result->uniqueTypes     = NULL;
    result->lwrInstrSrc     = UserPTX;
    result->isVersionTargetMismatch = False;
    result->targetDirectivePos      = NULL;


    for (s=0; s<ptxMAXStorage; s++) {
        listXInit( result->dwarfLocations[s].l );
    }
    

    {
        int idx;
        msgSourcePos_t  builtinPos = ctMsgCreateSourcePos("<builtin>", &(result->ptxBuiltInSourceStruct), 0);

        // .texref type
        //
        {
            ptxSymbolTable opaqueSymtab = ptxCreateSymbolTable(result->globalSymbols);

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "width", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "height", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "depth", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "channel_data_type", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "channel_order", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "normalized_coords", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "filter_mode", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "addr_mode_0", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "addr_mode_1", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "addr_mode_2", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );
            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "array_size", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "num_mipmap_levels", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "num_samples", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddOpaque( result->globalSymbols,
                          ptxCreateSymbol(result,
                              ptxCreateOpaqueType( ".texref", getFieldList( opaqueSymtab ) , result),
                              ".texref",
                              0, 0, builtinPos
                          )
                        );
        }

        // .samplerref type
        //
        {
            ptxSymbolTable opaqueSymtab = ptxCreateSymbolTable(result->globalSymbols);

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "force_unnormalized_coords", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "filter_mode", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "addr_mode_0", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "addr_mode_1", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "addr_mode_2", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddOpaque( result->globalSymbols,
                          ptxCreateSymbol(result,
                              ptxCreateOpaqueType( ".samplerref", getFieldList( opaqueSymtab ) , result),
                              ".samplerref",
                              0, 0, builtinPos
                          )
                        );
        }

        // .surfref type
        //
        {
            ptxSymbolTable opaqueSymtab = ptxCreateSymbolTable(result->globalSymbols);

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "width", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "height", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "depth", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "channel_data_type", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "channel_order", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );
            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "array_size", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );
            ptxAddVariableSymbol( opaqueSymtab,
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), "memory_layout", 0, 0, builtinPos ),
                                  ptxLocalScope,
                                  ptxCreateStorageClass( ptxConstStorage, 0 ),
                                  NULL, 0 );

            ptxAddOpaque( result->globalSymbols,
                          ptxCreateSymbol(result,
                              ptxCreateOpaqueType( ".surfref", getFieldList( opaqueSymtab ) , result),
                              ".surfref",
                              0, 0, builtinPos
                          )
                        );
        }

        // A7
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "A7", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxRegStorage, -1 ),
                              NULL, 0 ); 

# define PTX_SREG_VEC_BITS  (32)  // PTX 2.0 defines vector sregs to be .v4.b32 instead of .v4.b16

        // %tid and %ntid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateVectorType(4, ptxCreateBitType(PTX_SREG_VEC_BITS, result) , result), "%tid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateVectorType(4, ptxCreateBitType(PTX_SREG_VEC_BITS, result) , result), "%ntid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %laneid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%laneid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %warpid and %nwarpid
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%warpid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%nwarpid", 0, 0, builtinPos ),
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %smid and %nsmid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%smid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%nsmid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %ctaid and %nctaid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateVectorType(4, ptxCreateBitType(PTX_SREG_VEC_BITS, result), result), "%ctaid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateVectorType(4, ptxCreateBitType(PTX_SREG_VEC_BITS, result), result), "%nctaid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

# define PTX_SREG_GRIDID_BITS  (64)  // PTX 3.0 defines %gridid to be .b64 instead of .b32

        // %gridid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(PTX_SREG_GRIDID_BITS, result), "%gridid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %clock
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%clock", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %clock_hi
        //
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%clock_hi", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %clock64
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(64, result), "%clock64", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %pm0, %pm1, %pm2, %pm3
        //
        for (idx=0; idx<4; idx++) {
            char buffer[100];
            sprintf(buffer, "%%pm%d", idx);
            ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), stdCOPYSTRING(buffer), 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        }
        // %pm4, %pm5, %pm6, %pm7
        //
        for (idx=4; idx<8; idx++) {
            char buffer[100];
            sprintf(buffer, "%%pm%d", idx);
            ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), stdCOPYSTRING(buffer), 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        }
        // %pm<8>_64
        //
        for (idx=0; idx<8; idx++) {
            char buffer[100];
            sprintf(buffer, "%%pm%d_64", idx);
            ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(64, result), stdCOPYSTRING(buffer), 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        }

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
#if (LWCFG(GLOBAL_CHIP_T194) || LWCFG(GLOBAL_GPU_IMPL_GV11B)) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_60)
        for (idx=0; idx<8; idx++) {
            char buffer[100];
            sprintf(buffer, "%%pm%d_snap_64", idx);
            ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(64, result), stdCOPYSTRING(buffer), 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        }
#endif
#endif

        // %lanemask_{eq,le,lt,ge,gt}
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%lanemask_eq", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%lanemask_le", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%lanemask_lt", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%lanemask_ge", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%lanemask_gt", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %elwreg<32>
        //
        for (idx=0; idx<32; idx++) {
            char buffer[100];

            sprintf(buffer, "%%elwreg%d", idx);

            ptxAddVariableSymbol( result->globalSymbols, 
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), stdCOPYSTRING(buffer), 0, 0, builtinPos ), 
                                  ptxExternalScope, 
                                  ptxCreateStorageClass( ptxSregStorage, -1 ),
                                  NULL, 0 ); 
        }

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        // %affinity
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%affinity", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %sm_ctaid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%sm_ctaid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );
#endif

        // %globaltimer_lo
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%globaltimer_lo", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %globaltimer_hi
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%globaltimer_hi", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %globaltimer
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(64, result), "%globaltimer", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        // %cq_entryid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%cq_entryid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %cq_entryaddr
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(64, result), "%cq_entryaddr", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %cq_incr_minus1
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%cq_incr_minus1", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %is_queue_cta
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%is_queue_cta", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %bar<16>
        //
        for (idx=0; idx<16; idx++) {
            char buffer[100];
            sprintf(buffer, "%%bar%d", idx);
            ptxAddVariableSymbol( result->globalSymbols, 
                                  ptxCreateSymbol(result, ptxCreateBitType(32, result), stdCOPYSTRING(buffer), 0, 0, builtinPos ), 
                                  ptxExternalScope, 
                                  ptxCreateStorageClass( ptxSregStorage, -1 ),
                                  NULL, 0 ); 
        }

        // %bar_warp
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%bar_warp", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %bar_warp_result
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%bar_warp_result", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %bar_warp_resultp
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreatePredicateType(result), "%bar_warp_resultp", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

#if LWCFG(GLOBAL_ARCH_VOLTA)
        // %virtual_engineid
        //
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%virtual_engineid", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );
#endif // #if LWCFG(GLOBAL_ARCH_VOLTA)

#endif // #if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)

        // %total_smem_size
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%total_smem_size", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

        // %dynamic_smem_size
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%dynamic_smem_size", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 ); 

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_76)

        // %reserved_smem_offset_begin
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%reserved_smem_offset_begin", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %reserved_smem_offset_end
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%reserved_smem_offset_end", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %reserved_smem_offset_cap
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%reserved_smem_offset_cap", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %reserved_smem_offset_0
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%reserved_smem_offset_0", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %reserved_smem_offset_1
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%reserved_smem_offset_1", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );
#endif // ISA 7.6

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        // %hwtaskid
        //
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%hwtaskid", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %nlatc
        //
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%nlatc", 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );
#endif

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_64)
        // %stackend
        //
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), getSRAsString(result->parseData->deobfuscatedStringMapPtr, ptxStackend_STR), 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %stackinit_entry
        //
        ptxAddVariableSymbol( result->globalSymbols,
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), getSRAsString(result->parseData->deobfuscatedStringMapPtr, ptxStackinit_entry_STR), 0, 0, builtinPos ),
                              ptxExternalScope,
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );
#endif

        // cluster related sregs
#if LWCFG(GLOBAL_ARCH_HOPPER)
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_FUTURE)
        // %is_cluster_cta
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreatePredicateType(result), "%is_cluster_cta", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %clusterid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateVectorType( 4, ptxCreateBitType(PTX_SREG_VEC_BITS, result), result), "%clusterid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %nclusterid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateVectorType( 4, ptxCreateBitType(PTX_SREG_VEC_BITS, result), result), "%nclusterid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %cluster_ctaid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateVectorType( 4, ptxCreateBitType(PTX_SREG_VEC_BITS, result), result), "%cluster_ctaid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %cluster_ctarank
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%cluster_ctarank", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %cluster_nctaid
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateVectorType( 4, ptxCreateBitType(PTX_SREG_VEC_BITS, result), result), "%cluster_nctaid", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );

        // %cluster_nctarank
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%cluster_nctarank", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );
#endif // FUTURE
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        // %clusterid_gpc
        //
        ptxAddVariableSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateBitType(32, result), "%clusterid_gpc", 0, 0, builtinPos ), 
                              ptxExternalScope, 
                              ptxCreateStorageClass( ptxSregStorage, -1 ),
                              NULL, 0 );
#endif // INTERNAL
#endif // HOPPER

        // Builtin functions :
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        ptxAddFunctionSymbol( result->globalSymbols, 
                              ptxCreateSymbol(result, ptxCreateLabelType(result), getBUILTINSAsString(result->parseData->deobfuscatedStringMapPtr, ptxSuspendBuiltin_STR), 0, 0, builtinPos ),
                              False, False, ptxStaticScope, NULL, NULL, NULL, False, False, False, -1, NULL);
#endif

    }
#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
    printMacroExpansionInfoRec *expansionInfo = &(result->printMacroExpansionInfo);
    expansionInfo->nofExpansion = -1;
#endif
    stdSwapMemSpace(savedSpace);

    result->macroMap = mapNEW(String, 100);
    result->inlineFuncsMap = mapNEW(String, 100);
#if 0
    /* OPTIX_HAND_EDIT */
    /* We don't need these macros, because they will be expanded in the driver */

    initMacroProfileFermi(result);
    initMacroUtilFuncParseState(result);
    deObfuscateMacro(result, (String)ptxFmtFermi, INSTRUCTIONMACROS_KEY, ptxFmtFermiSize);
    ptxParseInputString( "<fermi macros>", (String)ptxInstructionMacrosFermi, INSTRUCTIONMACROS_KEY, result, False, False, False, 0 );

#endif        
    return result;
}

Bool ptxIsExpandedInternally(ptxInstructionSource lwrInstrSrc)
{
    return (lwrInstrSrc == Macro) || (lwrInstrSrc == InlineFunction);
}

Bool ptxIsInternalSource(ptxInstructionSource lwrInstrSrc)
{
    return ptxIsExpandedInternally(lwrInstrSrc) || (lwrInstrSrc == MacroUtilFunction);
}

Bool areRestrictedUseFeaturesAllowed(ptxParsingState ptxIR)
{
    return areExtendedInstructionsEnabled(ptxIR->parseData) || ptxIsInternalSource(ptxIR->lwrInstrSrc);
}

/*
 * Function         : parse one macro util func
 * Parameters       : object   (I) parsing state to use
 * Function Result  :
 */
static void ptxParseMacroUtilFunc( String ptx, ptxParsingState object)
{
    object->lwrInstrSrc = MacroUtilFunction;
    // OPTIX_HAND_EDIT TODO: When and how gets this called? For now replace the encrypted string by empty ptx
    ptxParseInputString( "<macro util>", ptx, INSTRUCTIONMACROS_KEY, object, False, False, False, 0, ptx, 0 );
    // Since ptxParseMacroUtilFunc is called after parsing of input files,
    // we should be okay simply setting the lwrInstrSrc back to UserPTX
    object->lwrInstrSrc = UserPTX;
}

/*
 * Function         : Iteratively parse all called macro util funcs
 * Parameters       : object   (I) parsing state to use 
 * Function Result  :
 */
void ptxProcessMacroUtilFuncs( ptxParsingState object )
{
   stdList_t funcList = NULL;

   String userInputVersion = object->version;
   String userInputArch    = object->target_arch;
   uInt addr_size          = object->addr_size;
   Bool foundDebugInfo     = object->foundDebugInfo;

   while (getPendingMacroUtilFuncList(&funcList, object)) {
      listTraverse(funcList, (stdEltFun)ptxParseMacroUtilFunc, object);
      listDelete(funcList);
      funcList = NULL;
   }

   object->parseData->version     = object->version        = userInputVersion;
   object->parseData->target_arch = object->target_arch    = userInputArch;
   object->addr_size      = addr_size;
   object->foundDebugInfo = foundDebugInfo;
}

/*
 * Function         : Close object, finalize parsing state
 * Parameters       : object   (I) object to close
 * Function Result  :
 */
void ptxCloseObject( ptxParsingState object )
{
    // set default texmode to texmode_unified if no texmode specified
    if (mapApply(object->target_opts, "texmode_unified"    )==NULL &&
        mapApply(object->target_opts, "texmode_independent")==NULL) {
        mapDefine(object->target_opts, "texmode_unified", (Pointer)True);
    }
}


/*
 * Function         : Discard object, with entire parsing state that it contains
 * Parameters       : object   (I) object to delete
 * Function Result  :
 */
void ptxDeleteObject( ptxParsingState object )
{
    if (object->ptxBuiltInSourceStruct) {
        msgDeleteSourceStructure(object->ptxBuiltInSourceStruct);
        object->ptxBuiltInSourceStruct = NULL;
    }

    if (object->ptxFileSourceStruct) {
        msgDeleteSourceStructure(object->ptxFileSourceStruct);
        object->ptxFileSourceStruct = NULL;
    }

    memspDelete(object->memSpace,False);
}


/*--------------------------- Debug Info Printing ----------------------------*/

static void printString( String s, FILE *f )
{
    fprintf(f, "%s\n", s );
}

static Bool ldFileLessEq( stdList_t e1, stdList_t e2 )
{
    return ( (uInt)(Address)listIndex(e1, 0) < (uInt)(Address)listIndex(e2, 0));
}

static void dumpPtxFiles( String file, FILE *f)
{
    char _buf[4096];
    FILE *input   = fopen(file, "r");
    int   pending = 0;
    char  *savedStr; 
    while (fgets(_buf, sizeof(_buf), input) != NULL) {
       char *buf = stdSTRTOK(_buf + strspn(_buf, " \t"), "\r\n", &savedStr);
       
       if (buf == NULL ||
           stdIS_PREFIX("#", buf) ||
           stdIS_PREFIX("//", buf) ||
           stdIS_PREFIX(".loc", buf) ||
           stdIS_PREFIX(".file", buf) ||
           stdIS_PREFIX("@@DWARF", buf)) {
           pending++;
       } else {
           char *sep1 = ".byte ", *sep2 = "";
           while (pending) {
               fprintf(f, "%s0", sep1);
               pending--;
               sep1 = ",";
               sep2 = "\n";
           }
           fprintf(f, "%s.string \"%s\"\n", sep2, buf);
       }
    }
    if (input)
        fclose(input);
}

static int string_starts_with(String s1, String s2, String s3)
{
    String base = s1;

    if (s2) {
      uInt len = (uInt)( strlen(s2));
      uInt off = (uInt)(strspn(base, " \t"));

      base += off;
      if (strncmp(s2, base, len) != 0) {
          return 0;
      }
      base += len;
    }

    if (s3) {
      uInt len = (uInt)( strlen(s3));
      uInt off = (uInt)(strspn(base, " \t"));

      base += off;
      if (strncmp(s3, base, len) != 0) {
          return 0;
      }
      base += len;
    }

    return 1;
}

static void dumpDebugInfoSections(ptxParsingState state, FILE *f)
{
    // FIXME: Use "state->dwarfSections" to dump .debug_info section
    stdList_t *dwarf = &state->dwarfBytes;
    stdString_t debugInfoAddress = stringNEW();
    stdString_t debugInfoPtx = stringNEW();

    while (*dwarf) {
        String str = (String)(*dwarf)->head;

        if (string_starts_with(str, ".section", ".debug_info")) {
            stdList_t end = (*dwarf)->tail;

            while (!string_starts_with((String)end->head, ".section", NULL))
            {
                Char *savedStr;
                String line =  stdSTRTOK((String)end->head, " \t", &savedStr);
                Char sep = '\t';

                stringAddBuf(debugInfoAddress, line);
                stringAddBuf(debugInfoPtx, line);
                
                while ((line = stdSTRTOK(NULL, " \t\r\n,", &savedStr)) != NULL)
                {
                    stringAddChar(debugInfoAddress, sep);
                    stringAddChar(debugInfoPtx, sep);
                    if (!isdigit(line[0]) && mapIsDefined(state->sassAddresses, line)) {
                        ptxSymLocInfo syminfo;
                        ptxSymLocInfo ptxsyminfo;

                        syminfo = (ptxSymLocInfo)(uintptr_t)mapApply(state->sassAddresses, line); 
                        ptxsyminfo = (ptxSymLocInfo)(uintptr_t)mapApply(state->locationLabels, line); 
                        
                        stringAddFormat(debugInfoAddress, "0x%08x", syminfo->offset);
                        stringAddFormat(debugInfoPtx, "%d", ptxsyminfo->offset);
                    } else {
                        stringAddBuf(debugInfoAddress, line);
                        stringAddBuf(debugInfoPtx, line);
                    }
                    sep = ',';
                };
                stringAddChar(debugInfoAddress, '\n');
                stringAddChar(debugInfoPtx, '\n');
                end = end->tail;
            }
            *dwarf = end;
            break;
        }

        dwarf = &(*dwarf)->tail;
    }

    fputs(".section .debug_info, \"\",@progbits\n", f);
    fputs(stringStripToBuf(stringCopy(debugInfoAddress)), f);
    fputs(".section .lw_debug_info_ptx, \"\",@progbits\n", f);
    fputs(stringStripToBuf(stringCopy(debugInfoPtx)), f);
}

/*
 * Function         : Print parsed debug information to specified file,
 *                    and builds up state->locationLabels as a side effect.
 * Parameters       : files         (I) the input files that where parsed.
 *                    state         (I) state to print
 *                    f             (I) file to print to
 * Function Result  : 
 */
void ptxPrintDebugInfo( stdList_t files, ptxParsingState state, FILE *f )
{
    stdList_t ldFiles= NULL;

    ldFiles = mapToList( state->dwarfFiles );
    listSort( &ldFiles, (stdLessEqFun)ldFileLessEq);
    dumpDebugInfoSections(state, f);
    fprintf(f, ".section .lw_debug_ptx_txt, \"\", @progbits\n");
    listTraverse(files, (stdEltFun)dumpPtxFiles, f);

    // FIXME: Use "state->dwarfSections" to print debug section
    listTraverse( state->dwarfBytes, (stdEltFun)printString, f );
    fprintf(f, "\n\n");
    listDelete(ldFiles);
}

typedef struct {
    String                   prefix;
    FILE                    *f;
    ptxDeclarationScope      scope;
    ptxParsingState          state;
} LinkingRec;


static void printImportExport( uInt symbolIndex, ptxSymbolTableEntry entry, LinkingRec *rec )
{
    String name = mapApply(*rec->state->funcIndexToSymbolMap, (Pointer)(Address)symbolIndex);
    if (entry->scope == rec->scope) {
        fprintf(rec->f,"%s%s",rec->prefix,name);
        rec->prefix= ",";
    }
}



/*
 * Function         : Print import/export symbol information in option format for fatbin command
 * Parameters       : state       (I) state to print
 *                    f           (I) file to print to
 * Function Result  : 
 */
void ptxPrintLinkInfo( ptxParsingState state, FILE *f )
{
    LinkingRec rec;
    
    rec.f      = f;
    
    rec.prefix = " --import ";
    rec.scope  = ptxExternalScope;
    rec.state  = state;
    mapTraverse( state->globalSymbols->symbolIndexMap, (stdPairFun)printImportExport, &rec );
    
    rec.prefix = " --export ";
    rec.scope  = ptxGlobalScope;
    mapTraverse( state->globalSymbols->symbolIndexMap, (stdPairFun)printImportExport, &rec );
}



/*--------------------------- Expression Printing ----------------------------*/

static void ptxPrintVectorSelector( ptxVectorSelector *sel, uInt dim, stdString_t s )
{
    while (dim--) {
        switch (*(sel++)) {
        case ptxComp_X : stringAddBuf(s,"x"); break;
        case ptxComp_Y : stringAddBuf(s,"y"); break;
        case ptxComp_Z : stringAddBuf(s,"z"); break;
        case ptxComp_W : stringAddBuf(s,"w"); break;
        default        : stdASSERT( False, ("Case label out of bounds") );
        }
    }
}

static void ptxPrintVideoSelector( ptxVideoSelector *sel, uInt N, stdString_t s )
{
    // print leading 'b' or 'h' based on kind of first selector
    uInt i;
    for (i=0; i<N; i++) {
        char *str = NULL;

        switch (*(sel++)) {
        case ptxCOMP_NONE :                             break;
        case ptxCOMP_H0   : str = (i==0 ? "h0" : "0");  break;
        case ptxCOMP_H1   : str = (i==0 ? "h1" : "1");  break;
        case ptxCOMP_H2   : str = (i==0 ? "h2" : "2");  break;
        case ptxCOMP_H3   : str = (i==0 ? "h3" : "3");  break;
        case ptxCOMP_B0   : str = (i==0 ? "b0" : "0");  break;
        case ptxCOMP_B1   : str = (i==0 ? "b1" : "1");  break;
        case ptxCOMP_B2   : str = (i==0 ? "b2" : "2");  break;
        case ptxCOMP_B3   : str = (i==0 ? "b3" : "3");  break;
        case ptxCOMP_B4   : str = (i==0 ? "b4" : "4");  break;
        case ptxCOMP_B5   : str = (i==0 ? "b5" : "5");  break;
        case ptxCOMP_B6   : str = (i==0 ? "b6" : "6");  break;
        case ptxCOMP_B7   : str = (i==0 ? "b7" : "7");  break;
        default           : stdASSERT( False, ("Case label out of bounds") );
        }
        if (str) {
            stringAddBuf(s,str);
        }
    }
}

static void ptxPrintByteSelector(ptxByteSelector *sel, uInt N, stdString_t s)
{
    // print leading 'b' based on kind of first selector
    uInt i;
    for (i = 0; i < N; i++) {
        char *str = NULL;
        switch (*(sel++)) {
        case ptxBYTE_NONE :             break;
        case ptxBYTE_B0   : str = (i == 0 ? "b0" : "0");  break;
        case ptxBYTE_B1   : str = (i == 0 ? "b1" : "1");  break;
        case ptxBYTE_B2   : str = (i == 0 ? "b2" : "2");  break;
        case ptxBYTE_B3   : str = (i == 0 ? "b3" : "3");  break;
        default           : stdASSERT(False, ("invalid byte selector"));
        }
        if (str) {
            stringAddBuf(s, str);
        }
    }
}


/*
 * Function         : Print ptx comparison operator to specified string.
 * Parameters       : c           (I) Comparison operator to print
 *                    s           (I) string to print to
 * Function Result  :
 */
void ptxPrintComparison( ptxComparison c, stdString_t s )
{
    String repr= "?";

    switch (c) {
    case ptxEQ_Comparison  : repr= "eq";  break;
    case ptxNE_Comparison  : repr= "ne";  break;
    case ptxLT_Comparison  : repr= "lt";  break;
    case ptxLE_Comparison  : repr= "le";  break;
    case ptxGT_Comparison  : repr= "gt";  break;
    case ptxGE_Comparison  : repr= "ge";  break;
    case ptxLO_Comparison  : repr= "lo";  break;
    case ptxLS_Comparison  : repr= "ls";  break;
    case ptxHI_Comparison  : repr= "hi";  break;
    case ptxHS_Comparison  : repr= "hs";  break;
    case ptxNUM_Comparison : repr= "num"; break;
    case ptxNAN_Comparison : repr= "nan"; break;
    case ptxEQU_Comparison : repr= "equ"; break;
    case ptxNEU_Comparison : repr= "neu"; break;
    case ptxLTU_Comparison : repr= "ltu"; break;
    case ptxLEU_Comparison : repr= "leu"; break;
    case ptxGTU_Comparison : repr= "gtu"; break;
    case ptxGEU_Comparison : repr= "geu"; break;
    case ptxNO_Comparison  : repr= "";    break;
    default                : stdASSERT( False, ("Case label out of bounds") );
    }
    
    stringAddBuf(s,repr);
}

/*
 * Function         : Print ptx operator to specified string.
 * Parameters       : o           (I) Operator to print
 *                    s           (I) string to print to
 * Function Result  :
 */
void ptxPrintOperator( ptxOperator o, stdString_t s )
{
    String repr= "?";

    switch (o) {
    case ptxLTOp     : repr= "<";  break;
    case ptxLTEQOp   : repr= "<="; break;
    case ptxGTOp     : repr= ">";  break;  
    case ptxGTEQOp   : repr= ">="; break;
    case ptxEQOp     : repr= "=="; break;  
    case ptxNEQOp    : repr= "!="; break; 
    case ptxOROp     : repr= "|";  break;
    case ptxOROROp   : repr= "||"; break;
    case ptxANDOp    : repr= "&";  break;
    case ptxANDANDOp : repr= "&&"; break;
    case ptxXOROp    : repr= "^";  break;
    case ptxADDOp    : repr= "+";  break;
    case ptxSUBOp    : repr= "-";  break;
    case ptxSHLOp    : repr= "<<"; break;
    case ptxSHROp    : repr= ">>"; break;
    case ptxMULOp    : repr= "*";  break;
    case ptxDIVOp    : repr= "/";  break;
    case ptxREMOp    : repr= "%";  break;
    case ptxNOTOp    : repr= "!";  break;
    case ptxILWOp    : repr= "~";  break;
    
  /* 
   * The following are instruction modifiers:
   */
    case ptxMINOp      : repr= ".min";       break;
    case ptxMAXOp      : repr= ".max";       break;
    case ptxINCOp      : repr= ".inc";       break;
    case ptxDECOp      : repr= ".dec";       break;
    case ptxPOPCOp     : repr= ".popc";      break;
    case ptxCASOp      : repr= ".cas";       break;
    case ptxEXCHOp     : repr= ".exch";      break;
    case ptxSAFEADDOp  : repr= ".safeadd";   break;
    case ptxNOP        : repr= "";           break;
    default            : stdASSERT( False, ("Case label out of bounds") );
    }
    
    stringAddBuf(s,repr);
}

/*
 * Function         : Print ptx operator to specified string (in modifier form).
 * Parameters       : o           (I) Operator to print
 *                    s           (I) string to print to
 * Function Result  :
 */
void ptxPrintOperatorAsModifier( ptxOperator o, stdString_t s )
{
    String repr= "?";

    switch (o) {
    case ptxLTOp       : repr= ".lt";        break;
    case ptxLTEQOp     : repr= ".le";        break;
    case ptxGTOp       : repr= ".gt";        break;  
    case ptxGTEQOp     : repr= ".ge";        break;
    case ptxEQOp       : repr= ".eq";        break;  
    case ptxNEQOp      : repr= ".ne";        break; 
    case ptxOROp       : repr= ".or";        break;
    case ptxANDOp      : repr= ".and";       break;
    case ptxXOROp      : repr= ".xor";       break;
    case ptxADDOp      : repr= ".add";       break;
    case ptxMINOp      : repr= ".min";       break;
    case ptxMAXOp      : repr= ".max";       break;
    case ptxINCOp      : repr= ".inc";       break;
    case ptxDECOp      : repr= ".dec";       break;
    case ptxPOPCOp     : repr= ".popc";      break;
    case ptxCASOp      : repr= ".cas";       break;
    case ptxEXCHOp     : repr= ".exch";      break;
    case ptxSAFEADDOp  : repr= ".safeadd";   break;
    case ptxNOP        : repr= "";           break;
    default            : stdASSERT( False, ("Case label out of bounds") );
    }
    
    stringAddBuf(s,repr);
}

    typedef struct {
        String      separator;
        stdString_t s;
    } PrintRec;

    static void prexprs( ptxExpression e, PrintRec *rec )
    {
        stringAddBuf(rec->s,rec->separator);
        ptxPrintExpression(e, rec->s );
        rec->separator=",";
    }

static void ptxPrintExpressions( stdList_t l, stdString_t s )
{
    PrintRec rec;
    
    rec.separator = "";
    rec.s         = s;

    listTraverse(l, (stdEltFun)prexprs, &rec);
}


/*
 * Function         : Print parsed expression to specified string.
 * Parameters       : e           (I) Expression to print
 *                    s           (I) string to print to
 * Function Result  :
 */
void ptxPrintExpression( ptxExpression e, stdString_t s )
{
    switch (e->kind) {
    case ptxBinaryExpression :
        ptxPrintExpression(e->cases.Binary->left , s );
        ptxPrintOperator  (e->cases.Binary->op   , s );
        ptxPrintExpression(e->cases.Binary->right, s );
        break;
        
    case ptxUnaryExpression :
        ptxPrintOperator  (e->cases.Unary->op , s );
        ptxPrintExpression(e->cases.Unary->arg, s );
        break;
        
    case ptxIntConstantExpression :
        stringAddFormat(s,"%"stdFMT_LLD,e->cases.IntConstant.i);
        break;
        
    case ptxFloatConstantExpression :
        if (ptxGetTypeSizeInBits(e->type) == 64) {
            union {
              uInt64 i;
              Double f;
            } v;
            v.f = ptxGetF64FloatConstantExpr(e);
            stringAddFormat(s,"0D%016"stdFMT_LLX,v.i);
        } else if (ptxGetTypeSizeInBits(e->type) == 32) {
            union {
              uInt32 i;
              Float f;
            } v;
            v.f = ptxGetF32FloatConstantExpr(e);
            stringAddFormat(s,"0F%08x",v.i);
        } else {
            stdASSERT( False, ("Illegal FloatConstExpression") );
        }
        break;
        
    case ptxSymbolExpression :
        stringAddBuf(s,e->cases.Symbol.symbol->symbol->unMangledName);
        break;
        
    case ptxArrayIndexExpression :
        ptxPrintExpression(e->cases.ArrayIndex->arg, s );
        stringAddBuf(s,"[");
        ptxPrintExpression(e->cases.ArrayIndex->index, s );
        stringAddBuf(s,"]");
        break;
        
    case ptxVectorSelectExpression :
        ptxPrintExpression(e->cases.VectorSelect->arg, s );
        stringAddBuf(s,".");
        ptxPrintVectorSelector(e->cases.VectorSelect->selector, e->cases.VectorSelect->dimension, s );
        break;
        
    case ptxVideoSelectExpression :
        ptxPrintExpression(e->cases.VideoSelect->arg, s );
        stringAddBuf(s,".");
        ptxPrintVideoSelector(e->cases.VideoSelect->selector, e->cases.VideoSelect->N, s );
        break;

    case ptxByteSelectExpression :
        ptxPrintExpression(e->cases.ByteSelect->arg, s );
        stringAddBuf(s,".");
        ptxPrintByteSelector(e->cases.ByteSelect->selector, e->cases.ByteSelect->N, s );
        break;

    case ptxPredicateExpression :
        if (e->neg) { stringAddBuf(s,"!"); }
        ptxPrintExpression(e->cases.Predicate.arg, s );
        break;
        
        break;
        
    case ptxAddressOfExpression :
        ptxPrintExpression(e->cases.AddressOf.lhs, s );
        break;
        
    case ptxAddressRefExpression :
        stringAddBuf(s,"[");
        ptxPrintExpression(e->cases.AddressRef.arg, s );
        stringAddBuf(s,"]");
        break;
        
    case ptxLabelReferenceExpression :
        stringAddBuf(s,e->cases.LabelReference->name);
        break;
        
    case ptxVectorExpression :
        stringAddBuf(s,"{");
        ptxPrintExpressions(e->cases.Vector.elements, s );
        stringAddBuf(s,"}");
        break;
        
    case ptxParamListExpression :
        stringAddBuf(s,"(");
        ptxPrintExpressions(e->cases.ParamList.elements, s );
        stringAddBuf(s,")");
        break;
        
    case ptxSinkExpression :
        stringAddBuf(s,"_");
        break;
    
    default : stdASSERT( False, ("Case label out of bounds") );
    }
}




/*
 * Function         : Print ptx type to specified string.
 * Parameters       : t           (I) Type expression to print
 *                    s           (I) string to print to
 * Function Result  :
 */
     static void prScalarType(String reprs, uInt64 size, Bool isSigned, stdString_t s)
     {
         // OPTIX_HAND_EDIT %d needs to be %llu since it is uInt64
         stringAddFormat(s,".%c%llu",reprs[isSigned == True],size);
     }
 
void ptxPrintType( stdMap_t* deobfuscatedStringMapPtr, ptxType t, stdString_t s )
{
    switch (t->kind) {
    case ptxTypeB1:
    case ptxTypeB2:
    case ptxTypeB4:
    case ptxTypeB8:
    case ptxTypeB16:
    case ptxTypeB32:
    case ptxTypeB64:
    case ptxTypeB128:
        prScalarType("bb", ptxGetTypeSizeInBits(t), False, s);
        break;
    
    case ptxTypeF16:
    case ptxTypeF32:
    case ptxTypeF64:
        prScalarType("ff", ptxGetTypeSizeInBits(t), False, s);
        break;
    
    case ptxTypeU2 :
    case ptxTypeU4 :
    case ptxTypeU8 :
    case ptxTypeU16:
    case ptxTypeU32:
    case ptxTypeU64:
    case ptxTypeS2 :
    case ptxTypeS4 :
    case ptxTypeS8 :
    case ptxTypeS16:
    case ptxTypeS32:
    case ptxTypeS64:
        prScalarType("us", ptxGetTypeSizeInBits(t), isSignedInt(t), s);
        break;

    case ptxTypeF16x2 :
        stringAddBuf(s, ".f16x2");
        break;

    case ptxTypeE4M3:
    case ptxTypeE5M2:
    case ptxTypeE4M3x2:
    case ptxTypeE5M2x2:
    case ptxTypeBF16:
    case ptxTypeBF16x2:
    case ptxTypeTF32:
        stringAddBuf(s, getTypeEnumAsString(deobfuscatedStringMapPtr, t->kind));
        break;

    case ptxVectorType :
        stringAddFormat(s, ".v%d ", t->cases.Vector.N);
        ptxPrintType(deobfuscatedStringMapPtr, t->cases.Vector.base, s);
        break;

    case ptxLabelType :
    case ptxMacroType :
    case ptxTypePred :
    case ptxConditionCodeType:
    case ptxOpaqueType:
    case ptxIncompleteArrayType :
    case ptxParamListType :
    case ptxArrayType :
    default : stdASSERT( False, ("Case label out of bounds") );
    }
}

/*
 * Function         : Return a unique name of the specified symbol that 
 *                    can be used in a global namespace, for instance in the
 *                    symbol map assembly file generated for debugging
 * Parameters       : state       (I) ptx object that defines the symbol
 *                    symbol      (I) Symbol to list
 * Function Result  : 
 */
String ptxGetUniqueName( ptxParsingState state, ptxSymbol symbol )
{
    String result= mapApply(state->symbolNames,symbol);
    
    if (!result) {
        uInt cnt=(uInt)( (Address)mapApply(state->symbolNamesCnt, symbol->unMangledName));
        mapDefine(state->symbolNamesCnt, symbol->unMangledName, (Pointer)(Address)(cnt+1));
        
        if (cnt == 0) {
            result= symbol->unMangledName;
        } else {
            stdString_t name= stringNEW();
            stringAddFormat(name,"%s.%d",symbol->unMangledName,cnt);
            result= stringStripToBuf(name);
        }
        
        mapDefine(state->symbolNames,symbol,result);
    }

    return result;
}

/*
 * Function         : It gives the pre-image of the uniquename generated by ptxGetUniqueName(),
                      essentially it is the reverse of the algo used in ptxGetUniqueName().
 * Parameters       : uname       (I) ptr to uniquename
 * Function Result  : 
 */

String ptxGetPreUniqueName(String uname)
{
    int size = strlen(uname);
    String name = uname + size;
    while(*name != '.' && name != uname)
        --name;
    if(name != uname)
        size = name - uname;

    name=stdMALLOC(size + 1);
    strncpy(name, uname, size);
    name[size]='\0';

    return name;
}



/* Utility Functions */

Bool ptxVersionAtLeast( int major, int minor, ptxParsingState parseState )
{
    if (parseState->version) {
        int m, n;

        // OPTIX_HAND_EDIT use preparsed version values
#if 0
        sscanf(parseState->version, "%d.%d", &m, &n);
#else
        m = parseState->version_major;
        n = parseState->version_minor;
#endif
        if (m == 5 && n == 1 && major == 6 && minor == 0) {
            // FIXME: This is hack to allow transition from PTX 5.1 to PTX 6.0
            // Any feature in PTX 6.0, allow even in PTX 5.1
            return True;
        } else {
            return ((m > major) || (m == major && n >= minor));
        }
    } else
        return False;
}

/*
 * Function : Checks if the input function is an OCG compiler understood builtin genomics function
 */
Bool ptxIsOcgBuiltinGenomicsFunc(ptxParseData parseData, cString name)
{
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    if (name == NULL)
        return False;

    if (stdEQSTRING(name, ""))
        return False;

    return stdIS_PREFIX(getBUILTINSAsString(parseData->deobfuscatedStringMapPtr, ptxOcgBuiltinGenomics_STR), name);
#else
    return False;
#endif
}

/*
 * Function : Checks if the input function is an OCG compiler understood builtin function
 */
Bool ptxIsOcgBuiltinFunc(ptxParseData parseData, cString name)
{
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    if (name == NULL)
        return False;

    if (stdEQSTRING(name, ""))
        return False;

    return stdIS_PREFIX(getBUILTINSAsString(parseData->deobfuscatedStringMapPtr, ptxOcgBuiltin_STR), name);
#else
    return False;
#endif
}
/*
 * Function : Checks if the input function is a compiler understood builtin function
 */
Bool ptxIsBuiltinFunc(ptxParseData parseData, cString name)
{
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    if (name == NULL)
        return False;

    if (stdEQSTRING(name, ""))
        return False;

    return stdIS_PREFIX(getBUILTINSAsString(parseData->deobfuscatedStringMapPtr, ptxBuiltin_STR), name);
#else
    return False;
#endif
}

/*
 * Function         : Appends the feature to list ptxIR->nonMercFeaturesUsed, only if it's not in the list.
 * Parameters       : feature : ptxNonMercFeature whcih is to be added to the list.
 */
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
void ptxSetNonMercFeatureUsed(ptxParsingState parseState, ptxNonMercFeature feature)
{
    if (!listContains(parseState->nonMercFeaturesUsed, (Pointer)feature)) {
        // inferred address_size is determined latter in compilation process
        // but the error/warning for address_size should be first in order. 
        if (feature == ptxAddressSize32) {
            listXPutInFront(parseState->nonMercFeaturesUsed, (Pointer)feature);
        // else in order of oclwrance of the feature.
        } else {
            listXPutAfter(parseState->nonMercFeaturesUsed, (Pointer)feature);
        }
    }
}
#endif

/*
 * Function : returns the symbol table entry of the callee function from the call instruction
 */
ptxSymbolTableEntry ptxCallSymTabEntFromCallInsr(ptxInstruction instr)
{
    if (instr->tmplate->code != ptx_call_Instr)
        return NULL;

    int funcExprIdx          = ptxGetCallTargetArgNumber(instr);
    ptxExpression funcExpr   = instr->arguments[funcExprIdx];
    ptxSymbolTableEntry func;
    stdASSERT(funcExpr->kind == ptxSymbolExpression, (""));

    func = ptxGetSymEntFromExpr(funcExpr);
    stdASSERT(func->kind == ptxFunctionSymbol ||
              func->kind == ptxVariableSymbol, (""));

    return func;
}
/*
 * Function : returns the name of the callee function from the call instruction
 */
cString ptxGetFuncNameFromCallInsr(ptxInstruction instr)
{
    ptxSymbolTableEntry func = ptxCallSymTabEntFromCallInsr(instr);
    if (func == NULL)
        return "";

    return func->symbol->name;
}

Bool isDirectCall(ptxExpression *arguments, uInt nrofArguments)
{
    uInt i, n=0;

    // count number of non-paramListExpr arguments; indirect calls have two, direct calls have one
    for (i=0; i<nrofArguments; i++) {
        ptxExpression arg = arguments[i];
        if (arg->kind != ptxParamListExpression)  n++;
    }

    return (n!=2);
}

/*
* Function         : Get argument number of call target in the PTX call instruction
* Parameters       : ptxInstruction  :  The instruction in which we are querying the positional arguement information
* Function Result  : call target argument number
*/
int ptxGetCallTargetArgNumber(ptxInstruction instr)
{
    uInt i;
    stdASSERT(instr->tmplate->code == ptx_call_Instr,
              ("Unexpected instruction"));

    for (i = 0; i < instr->tmplate->nrofArguments; i++) {
        ptxExpression arg = instr->arguments[i];
        if (arg->kind != ptxParamListExpression) {
            return i;
        }
    }
    stdASSERT( False, ("Call target not found.\n") );
    return 0;
}

/*
 * Function         : Get argument number of the return argument in the PTX call instruction
 * Parameters       : ptxInstruction  :  The instruction in which we are querying the positional arguement information
 * Function Result  : return argument number
 */
static int ptxGetCallReturnArgNumber(ptxInstruction instr)
{
    if (instr->tmplate->code != ptx_call_Instr)
        return -1;

    if (instr->arguments[0]->kind == ptxParamListExpression)
        return 0;

    return -1;
}

/*
 * Function         : Get argument number of the input argument in the PTX call instruction
 * Parameters       : ptxInstruction  :  The instruction in which we are querying the positional arguement information
 * Function Result  : input argument number
 */
static int ptxGetCallInputArgNumber(ptxInstruction instr)
{
    int callTargetIdx ;
    if (instr->tmplate->code != ptx_call_Instr)
        return -1;

    callTargetIdx = ptxGetCallTargetArgNumber(instr);

    if (callTargetIdx + 1 > instr->tmplate->nrofArguments - 1)
        return -1;

    return callTargetIdx + 1;
}

/*
 * Function         : Define the new label into the given symbol table
 */
static ptxSymbolTableEntry ptxDefineLabelInSymTab(ptxSymbolTable ptxSymTab,
                                                  ptxParsingState parseState,
                                                  String name,
                                                  msgSourcePos_t sourcePos)
{
    stdCHECK_WITH_POS(ptxAddLabelSymbol(ptxSymTab,
                                        ptxCreateSymbol(parseState,
                                                        ptxCreateLabelType(parseState),
                                                        name, 0, 0, sourcePos)),
                      (ptxMsgDuplicateLabel, sourcePos, name));
    return ptxLookupSymbol(ptxSymTab, name, True, parseState);
}

/*
 * Function         : Do the deep copy for FunctionPrototypeAttrInfo struct.
 */
static void ptxCopyFunctionPrototypeAttrInfo(ptxFunctionPrototypeAttrInfo dst,
                                             ptxFunctionPrototypeAttrInfo src)
{
    // Do a blind shallow copy of entire object.
    memcpy(dst, src, sizeof(*(src)));
    // Now Do deep copy for lists
    dst->scratchRegs = listCopy(src->scratchRegs);
    dst->scratchRRegs = listCopy(src->scratchRRegs);
    dst->scratchBRegs = listCopy(src->scratchBRegs);
    dst->rparams = listCopy(src->rparams);
    dst->fparams = listCopy(src->fparams);
}

/*
 * Function         : Get the (mangled) unique name for the newly created function
 *                    prototype out of the given symbol.
 */
static String ptxGetUniqueNameForPrototype(ptxSymbol symbol, uInt64 virtualAddress, uInt numScopesOnLine)
{
    String mangledName, protoName;

    mangledName = ptxMangleName(symbol, numScopesOnLine);
    protoName = (String) stdMALLOC(strlen(mangledName) + 10 + 30);
    sprintf(protoName, "cvt_proto_%s_%lld", mangledName, virtualAddress);
    stdFREE(mangledName);
    return protoName;
}

/*
 * Function         : Creates a dummy symbol table entry but having the specified ptxSymbolKind.
 * Parameters       : parseState     :  Parsing state
 *                    instr          :  ptx instruction using the given symbol
 *                    ptxSymEnt      :  The original symbol table entry whose dummy
 *                                      entry of specified kind needs to be created.
 *                    kind           :  The kind of the new symbol table entry to create.
 * Function Result  : New symbol table entry having the specified kind.
 */

ptxSymbolTableEntry ptxCreateDummySymbolTableEntryOfKind(ptxParsingState parseState,
                                                         ptxInstruction instr,
                                                         ptxSymbolTableEntry ptxSymEnt,
                                                         ptxSymbolKind kind)
{
    ptxSymbolTableEntry dummyEntry;
    String dummyName = ptxGetUniqueNameForPrototype(ptxSymEnt->symbol, instr->virtualAddress, parseState->parseData->numScopesOnLine);

    // Lwrrently we only support prototypeSymbol kind.
    // In future we might need to enhance the function body to support other kind's.
    stdASSERT(kind == ptxCallPrototypeSymbol, ("Unexpected symbol kind"));

    stdASSERT(ptxSymEnt->symbtab, ("Unable to locate symbol table"));

    /* Look up symtab for symbol. If found simply re-use that. */
    dummyEntry = ptxLookupSymbol(ptxSymEnt->symbtab, dummyName, False, parseState);
    if (dummyEntry) {
        stdASSERT(dummyEntry->aux != NULL &&
                  dummyEntry->aux->funcProtoAttrInfo != NULL, ("Corrupt label"));

        // Ensure entry is of the kind as requested.
        stdASSERT(dummyEntry->kind == kind, ("Found multiple entries with same name but differrent kind"));
        return dummyEntry;
    }

    dummyEntry = ptxDefineLabelInSymTab(ptxSymEnt->symbtab, parseState,
                                        dummyName, ptxSymEnt->symbol->sourcePos);

    stdASSERT( dummyEntry != NULL && dummyEntry->aux != NULL &&
               dummyEntry->aux->funcProtoAttrInfo != NULL &&
               dummyEntry->aux->funcProtoAttrInfo->rparams == NULL &&
               dummyEntry->aux->funcProtoAttrInfo->fparams == NULL, ("Corrupt label") );

    stdASSERT(ptxSymEnt->aux->funcProtoAttrInfo, ("Unable to locate function prototype attribute info"));

    ptxCopyFunctionPrototypeAttrInfo(dummyEntry->aux->funcProtoAttrInfo, ptxSymEnt->aux->funcProtoAttrInfo);

    dummyEntry->kind = kind;
    return dummyEntry;
}

/*
 * Function         : Checks if the specified label immediately follows the specified instruction
 * Parameters       : instr      :  The instruction which is to be checked for immediately preceeding a label
 *                  : labSymEnt  :  The label which is to be checked for immedaiately succedding an instruction
 * Function Result  : True if the label 'labSymEnt' is immediately after the instruction 'instr'; 
 *                    Otherwise False
 */
Bool ptxIsLabelImmediatelyFollowsInstr(ptxInstruction instr, ptxSymbolTableEntry labSymEnt)
{
    if (!labSymEnt || labSymEnt->kind != ptxLabelSymbol)
        return False;

    if (!labSymEnt->aux)
        return False;

    if (!instr)
        return False;

    return labSymEnt->aux->listIndex == instr->stmtId + 1;
}

/*
 * Function         : Checks if the specified call instruction uses .param space at either output/input
 * Parameters       : instr           :  The call instruction which is to be checked for using .param spaced argument
 *                  : useReturnParam  :  Specifies whether the arguments to be checked for .param space is for return or input arguments
 * Function Result  : True if the input or output argument of the specified call instruction is in .param space
 *                    Otherwise False
 */
Bool ptxIsCallArgsUsingParamSpace(ptxInstruction instr, Bool useReturnParam)
{
    int argIdx = -1;
    ptxSymbolTableEntry arg;

    if (!instr || instr->tmplate->code != ptx_call_Instr)
        return False;

    argIdx = useReturnParam ? ptxGetCallReturnArgNumber(instr):
                              ptxGetCallInputArgNumber(instr);

    if (argIdx == -1)
        return False;

    // assuming the first argument is representative of the entire list
    ptxExpression argExp = instr->arguments[argIdx]->cases.ParamList.elements->head;
    stdASSERT(argExp, (""));
    arg = ptxGetSymEntFromExpr(argExp);
    stdASSERT(arg, (""));

    return arg->storage.kind == ptxParamStorage;
}

Bool isImmediate(ptxExpression Expr)
{
    switch (Expr->kind) {
    case ptxIntConstantExpression:
    case ptxFloatConstantExpression:
        return True;
    default:
        break;
    }

    return False;
}

/*
 * Function         : Checks if any of the LWCA SASS directives are used for a function
 * Parameters       : function on which presence of LWCA SASS directive is to be checked
 * Function Result  : True if LWCA SASS directives are used on th function symbol
 *                    False, otherwise
 */
Bool usesLwdaSass(ptxSymbolTableEntry func)
{
    return func->aux->funcProtoAttrInfo->hasAllocatedParams ||
           func->aux->funcProtoAttrInfo->scratchRegs        ||
           (func->aux->funcProtoAttrInfo->retAddrAllocno >= 0);
}


/*
 * Function         : Checks if any of the Custom ABI pragmas are used for a function
 * Parameters       : function on which presence of Custom ABI pragmas is to be checked
 * Function Result  : True if Custom ABI pragmas are used on th function symbol
 *                    False, otherwise
 */
Bool usesLwstomABI(ptxSymbolTableEntry func)
{
    return (func->aux->funcProtoAttrInfo->numAbiParamRegs  !=  UNSPECIFIED_ABI_PARAM_REGS) ||
           (func->aux->funcProtoAttrInfo->retAddrReg       != UNSPECIFIED_ABI_REG)   ||
           (func->aux->funcProtoAttrInfo->relRetAddrReg    != UNSPECIFIED_ABI_REG)   ||
           (func->aux->funcProtoAttrInfo->retAddrUReg      != UNSPECIFIED_ABI_REG)   ||
           (func->aux->funcProtoAttrInfo->scratchRRegs     != UNSPECIFIED_ABI_REGS)  ||
           (func->aux->funcProtoAttrInfo->scratchBRegs     != UNSPECIFIED_ABI_REGS);
}

/*
 * Function         : Checks if the function is decorated with coroutine pragma. Such function can suspend.
 * Parameters       : function on which presence of coroutine pragmas is to be checked
 * Function Result  : True if coroutine pragmas are used on th function symbol
 *                    False, otherwise
 */
Bool usesCoroutine(ptxSymbolTableEntry func)
{
    return func->aux->funcProtoAttrInfo->isCoroutine;
}

static uInt32 ptxGetScratchBRegInfo(ptxSymbolTableEntry func)
{
    uInt32 scratchB = 0;
    stdList_t l = func->aux->funcProtoAttrInfo->scratchBRegs;

    if (l == UNSPECIFIED_ABI_REGS) {
        return 0;
    }

    for (; l && l->tail; l = l->tail) {
        int regNo = (int)(Address)l->head;
        scratchB |= (1 << regNo);
    }
    return scratchB;
}

static void ptxGetScratchRRegInfo(ptxSymbolTableEntry func,
                           uInt64* scratchR63To0,    uInt64* scratchR127To64,
                           uInt64* scratchR191To128, uInt64* scratchR255To192)
{
    stdList_t l = func->aux->funcProtoAttrInfo->scratchRRegs;
    uInt64 scratchR[4] = {0, 0, 0, 0};

    *scratchR63To0    = 0;
    *scratchR127To64  = 0;
    *scratchR191To128 = 0;
    *scratchR255To192 = 0;

    if (l == UNSPECIFIED_ABI_REGS) {
        return;
    }

    for (; l && l->tail; l = l->tail) {
        int regNo = (int)(Address)l->head;
        int scratchBucket = (regNo >> 6) & 0x3;
        scratchR[scratchBucket] |= 0x1ULL << (regNo & 0x3F);
    }

    stdMEMCOPY(scratchR63To0,    &scratchR[0]);
    stdMEMCOPY(scratchR127To64,  &scratchR[1]);
    stdMEMCOPY(scratchR191To128, &scratchR[2]);
    stdMEMCOPY(scratchR255To192, &scratchR[3]);
}

/*
 * Function         : Obtain the scratch registers in groups of uInt64s for a given function
 * Parameters       : (I) func             - Function whose scratch registers are to be queried
 *                    (O) scratchB         - Barrier registers which are marked as scratch
 *                    (O) scratchR63To0    - RRegisters from   0 to  63 which are marked scratch
 *                    (O) scratchR127To64  - RRegisters from  64 to 127 which are marked scratch
 *                    (O) scratchR191To128 - RRegisters from 128 to 191 which are marked scratch
 *                    (O) scratchR255To192 - RRegisters from 192 to 255 which are marked scratch
 */
void ptxGetScratchRegInfo(ptxSymbolTableEntry func, uInt32* scratchB,
                          uInt64* ScratchR63To0,    uInt64* ScratchR127To64,
                          uInt64* ScratchR191To128, uInt64* ScratchR255To192)
{
    *scratchB = ptxGetScratchBRegInfo(func);
    ptxGetScratchRRegInfo(func, ScratchR63To0,    ScratchR127To64,
                                ScratchR191To128, ScratchR255To192);
}

/*
 * Function         : Parses value from a known pragma
 * Parameters       : parseData, Pragma
 * Function Result  : Pragma value
 */
int ptxGetPragmaValue(ptxParseData parseData, String pragma)
{
    char *end;
    int pragmaValue;
    char* ptr;
    if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxLocalMaxNReg_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxLocalMaxNReg_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxRetAddrRegRRel32_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxRetAddrRegRRel32_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxRetAddrRegR_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxRetAddrRegR_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxRetAddrRegU_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxRetAddrRegU_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxRetAddrReg_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxRetAddrReg_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallRetAddrRegRRel32_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallRetAddrRegRRel32_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallRetAddrRegR_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallRetAddrRegR_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallRetAddrRegU_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallRetAddrRegU_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallRetAddrReg_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallRetAddrReg_PRG)) + 1;
    } else {
        stdASSERT(0, ("unexpected pragma"));
        return -1;
    }
    // Ensure that pragma value is non-empty.
    stdCHECK(*ptr != 0, (ptxMsgNonEmptyPragmaValue, pragma));
    pragmaValue = strtol(ptr, &end, 0);
    stdCHECK(*end == 0, (ptxMsgIlwalidPragmaValue, ptr, pragma));
    return pragmaValue;
}

typedef struct {
    stdList_t* l;
    cString pragma;
    ptxParseData parseData;
} pragmaParsingData;

#define PragmaRangeSizeThreshold 300

static Bool isAbiParamRegPragma(ptxParseData parseData, cString pragma)
{
    return (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxAbiParamReg_PRG), pragma) ||
            stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallAbiParamReg_PRG), pragma));
}

static void populateList(String reg, pragmaParsingData* data)
{
    int i, j, milwal, maxVal, val, bound1Val, bound2Val;
    char *sep, *end;
    char *bound1, *bound2;
    stdList_t* list = data->l;

    if ((sep = strchr(reg, '-')) == NULL) {
        val = strtol(reg, &end, 0);
        if (val == 0 && !strncmp(reg, "all", 3) && isAbiParamRegPragma(data->parseData, data->pragma)) {
            listAddTo((Pointer)(Address)ABI_PARAM_REGS_ALL, list);
            return;
        }
        stdCHECK(*end == 0, (ptxMsgIlwalidPragmaValue, reg, data->pragma));
        listAddTo((Pointer)(Address)val, list);
        return;
    }

    if (reg == sep) {
        stdCHECK(False, (ptxMsgIlwalidPragmaRange, reg, data->pragma));
        return;
    }
    if (*(String)(sep + 1) == '\0') {
        stdCHECK(False, (ptxMsgIlwalidPragmaRange, reg, data->pragma));
        return;
    }

    bound1 = stdCOPY_S(reg, sep - reg + 1);
    bound1[sep - reg + 1] = '\0';
    bound2 = stdCOPYSTRING(sep + 1);
    bound1Val = strtol(bound1, &end, 0);
    if (*end != '-') {
        stdCHECK(False, (ptxMsgIlwalidPragmaRange, reg, data->pragma));
        return;
    }

    bound2Val = strtol(bound2, &end, 0);
    if (*end != 0) {
        stdCHECK(*end == 0, (ptxMsgIlwalidPragmaRange, reg, data->pragma));
        return;
    }

    if (bound1Val > bound2Val) {
        stdSWAP(bound1Val, bound2Val, int);
    }
    milwal = bound1Val;
    maxVal = bound2Val;
    // add all the numbers in the range to the list
    for (i = milwal, j = 0; i <= maxVal; i++, j++) {
        listAddTo((Pointer)(Address)i, list);
        if (j > PragmaRangeSizeThreshold) break;
    }
}

/*
 * Function         : Parses multiple values from a known pragma
 * Parameters       : Pragma - pragma to be parsed
 *                    sortFinalList - whether to sort the final list in ascending order
 *                    addDummyTerminator - whether to add a dummy element (-1) at the end of the result list
 * Function Result  : Pragma value as a list
 */
stdList_t ptxGetPragmaValueList(ptxParseData parseData, String pragma, Bool sortFinalList, Bool addDummyTerminator)
{
    char  *ptr;
    stdList_t pragmaValueList = NULL;
    const int dummyTerminatorValue = -1;

    if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxAbiParamReg_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxAbiParamReg_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallAbiParamReg_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallAbiParamReg_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxScratchRegsR_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxScratchRegsR_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallScratchRegsR_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallScratchRegsR_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxScratchRegsB_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxScratchRegsB_PRG)) + 1;
    } else if (stdIS_PREFIX(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallScratchRegsB_PRG), pragma)) {
        ptr = pragma + strlen(getLWSTOMABIPrgAsString(parseData->deobfuscatedStringMapPtr, ptxCallScratchRegsB_PRG)) + 1;
    } else {
        stdASSERT(0, ("unexpected pragma"));
        return NULL;
    }

    pragmaParsingData data = {&pragmaValueList, pragma, parseData};
    stdTokenizeString(ptr, ",", False, False, (stdEltFun)populateList, (Pointer)&data, False, False);

    if (sortFinalList) {
        listSort(&pragmaValueList, (stdLessEqFun)stdIntLessEq);

        if (addDummyTerminator) {
            pragmaValueList = listAppend(pragmaValueList, (Pointer)(Address) dummyTerminatorValue);
        }

        return pragmaValueList;
    }

    if (addDummyTerminator) {
        // This terminator element being added in front now will become the last element during reversal
        pragmaValueList = listCons((Pointer)(Address) dummyTerminatorValue, pragmaValueList);
    }

    return listReverse(pragmaValueList);
}

/*
 * Function         : Parses value of abi_param_reg pragma
 * Parameters       : (I) Pragma   - pragma to be parsed
 *                    (O) numReg   - number of registers to be used for parameters
 *                    (O) firstReg - starting register to be used for parameters
 */
void ptxGetAbiParamRegPragma(String pragma, int* numReg, int* firstReg, ptxParsingState parseState)
{
    stdList_t regs = ptxGetPragmaValueList(parseState->parseData, pragma, False, False);
    int maxRegsPerThread = parseState->gpuInfo->maxRegsPerThread;
    cString profileName = parseState->gpuInfo->profileName;

    if (regs == NULL) {
        stdCHECK(False, (ptxMsgIncorrectPragmaArgs, pragma));
        *numReg = UNSPECIFIED_ABI_PARAM_REGS;
        *firstReg = UNSPECIFIED_ABI_REG;
        return;
    }

    *numReg = (int)(Address)regs->head;
    if (*numReg != ABI_PARAM_REGS_ALL) {
        stdCHECK(*numReg > 0 && *numReg <= maxRegsPerThread,
                 (ptxMsgWrongValue, getLWSTOMABIPrgAsString(parseState->parseData->deobfuscatedStringMapPtr, ptxAbiParamReg_PRG),
                  *numReg, profileName));
    }
    regs = regs->tail;
    if (regs == NULL) {
        *firstReg = UNSPECIFIED_ABI_REG;
        return;
    }
    *firstReg = (int)(Address)regs->head;
    stdCHECK(*firstReg > 0 || *firstReg <= maxRegsPerThread,
             (ptxMsgWrongValue, getLWSTOMABIPrgAsString(parseState->parseData->deobfuscatedStringMapPtr, ptxAbiParamReg_PRG), *firstReg, profileName));
    stdCHECK(*firstReg + *numReg <= maxRegsPerThread,
             (ptxMsgParameterPoolOverflow, *firstReg, *numReg,
              maxRegsPerThread, profileName));


    if (regs->tail != NULL) {
        stdCHECK(False, (ptxMsgIncorrectPragmaArgs, pragma));
        return;
    }
}

/*
 * Function         : Returns the value of a known jetfire pragma.
 * Parameters       : Pragma - Jetfire pragma whose value to be parsed.
                      isPragmaValueInt - Flag helps to interpret the returned pragma-value.
                                         (pragma-value can be either of int or String)
                      pragmaValue - Integer pragma value if `isPragmaValueInt` is True.
 * Function Result  : pragma-value as a string.
 */
String ptxGetJetfirePragmaValue(String pragma, Bool *isPragmaValueInt, Int64 *pragmaValue, ptxParseData parseData)
{
    char *ptr, *end;

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_65)
    if (stdIS_PREFIX(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxNextKnob_PRG), pragma)) {
        ptr = pragma + strlen(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxNextKnob_PRG)) + 1;
        *isPragmaValueInt = False;
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_71)
    } else if (stdIS_PREFIX(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxGlobalKnob_PRG), pragma)) {
        ptr = pragma + strlen(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxGlobalKnob_PRG)) + 1;
        *isPragmaValueInt = False;
    } else if (stdIS_PREFIX(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxSetKnob_PRG), pragma)) {
        ptr = pragma + strlen(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxSetKnob_PRG)) + 1;
        *isPragmaValueInt = False;
    } else if (stdIS_PREFIX(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxResetKnob_PRG), pragma)) {
        ptr = pragma + strlen(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxResetKnob_PRG)) + 1;
        *isPragmaValueInt = False;
#endif
    } else if (stdIS_PREFIX(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxLwopt_PRG), pragma)) {
        ptr = pragma + strlen(getJETFIREPrgAsString(parseData->deobfuscatedStringMapPtr, ptxLwopt_PRG)) + 1;
        *isPragmaValueInt = True;
    } else {
        stdASSERT(0, ("unexpected pragma"));
        return "";
    }
#else
    return "";
#endif // ISA_65

    if (*isPragmaValueInt) {
        *pragmaValue = strtoll(ptr, &end, 0);
        if (*end != 0) {
            return "";
        }
    } else {
        *pragmaValue = ~0;
    }
    return ptr;
}

/*
 * Function         : Sanitizes the value of a known jetfire pragma.
 * Parameters       : Pragma - Jetfire pragma whose value to be sanitized.
 */
void ptxSanitizeJetfirePragmaValue(String pragma, msgSourcePos_t sourcePos, ptxParseData parseData)
{
    char *ptr;
    Bool isPragmaValueInt = False;
    Int64 pragmaValue;

    ptr = ptxGetJetfirePragmaValue(pragma, &isPragmaValueInt, &pragmaValue, parseData);

    if (*ptr == 0) {
        // Empty pragma Value.
        stdCHECK_WITH_POS(False, (ptxMsgNonEmptyPragmaValue, sourcePos, pragma));
    }

    if (isPragmaValueInt) {
        if (pragmaValue < 0 || pragmaValue == LLONG_MIN || pragmaValue == LLONG_MAX) {
            stdCHECK_WITH_POS(False, (ptxMsgIlwalidPragmaValue, sourcePos, ptr, pragma));
        }
    }
}

/*
 * Function         : Sanitizes the syntax and value of a known sync pragma.
 * Parameters       : Pragma - Sync pragma whose syntax and value to be sanitized.
 */
void ptxSanitizeSyncPragma(String pragma, msgSourcePos_t sourcePos)
{
    char *str;
    char *end;
    Int64 pragmaValue = ~0;
    str = strchr(pragma, '=');
    if (!str) {
        //Empty pragma value
        stdCHECK_WITH_POS(False, (ptxMsgNonEmptyPragmaValue, sourcePos, pragma));
        return;
    } else {
        str = str + 1;
        pragmaValue = strtoll(str, &end, 0);
        if (*end != 0 || pragmaValue < 0 || pragmaValue == LLONG_MIN || pragmaValue == LLONG_MAX) {
            stdCHECK_WITH_POS(False, (ptxMsgIlwalidPragmaValue, sourcePos, str, pragma));
        }
    }
}

/*
 * Function         : Used to check whether the input function is a alias
 * Parameters       : symEnt   (I) function symbol which is to be checked for its alias-ness
 * Function Result  : True if 'symEnt' is an alias symbol
 *                    False, otherwise
 */
Bool isAliasSymbol(ptxSymbolTableEntry symEnt)
{
    if (symEnt->kind != ptxFunctionSymbol) {
        return False;
    }

    if (symEnt->aux->aliasee) {
        return True;
    }

    return False;
}

/*
 * Function         : Used to query aliasee function for an alias function
 * Parameters       : symEnt   (I) function symbol whose aliasee is to be determined
 * Function Result  : aliasee function symbol if input function 'symEnt' is an alias
 *                    'symeEnt' function symbol if input function 'symEnt' is not an alias
 */
ptxSymbolTableEntry ptxResolveAliasSymbol(ptxSymbolTableEntry alias)
{
    ptxSymbolTableEntry aliasee = NULL;

    if (!isAliasSymbol(alias)) {
        return alias;
    }

    aliasee = alias->aux->aliasee;
    if (aliasee->aux->body) {
        return aliasee;
    }

    return alias;
}


Bool isB1  (ptxType type) { return (type != NULL && type->kind == ptxTypeB1 );  }
Bool isB2  (ptxType type) { return (type != NULL && type->kind == ptxTypeB2 );  }
Bool isB4  (ptxType type) { return (type != NULL && type->kind == ptxTypeB4 );  }
Bool isB8  (ptxType type) { return (type != NULL && type->kind == ptxTypeB8 );  }
Bool isB16 (ptxType type) { return (type != NULL && type->kind == ptxTypeB16);  }
Bool isB32 (ptxType type) { return (type != NULL && type->kind == ptxTypeB32);  }
Bool isB64 (ptxType type) { return (type != NULL && type->kind == ptxTypeB64);  }
Bool isB128(ptxType type) { return (type != NULL && type->kind == ptxTypeB128); }
Bool isS2  (ptxType type) { return (type != NULL && type->kind == ptxTypeS2);  }
Bool isS4  (ptxType type) { return (type != NULL && type->kind == ptxTypeS4);  }
Bool isS8  (ptxType type) { return (type != NULL && type->kind == ptxTypeS8);  }
Bool isS16 (ptxType type) { return (type != NULL && type->kind == ptxTypeS16);  }
Bool isS32 (ptxType type) { return (type != NULL && type->kind == ptxTypeS32);  }
Bool isS64 (ptxType type) { return (type != NULL && type->kind == ptxTypeS64);  }
Bool isU2  (ptxType type) { return (type != NULL && type->kind == ptxTypeU2); }
Bool isU4  (ptxType type) { return (type != NULL && type->kind == ptxTypeU4); }
Bool isU8  (ptxType type) { return (type != NULL && type->kind == ptxTypeU8); }
Bool isU16 (ptxType type) { return (type != NULL && type->kind == ptxTypeU16); }
Bool isU32 (ptxType type) { return (type != NULL && type->kind == ptxTypeU32); }
Bool isU64 (ptxType type) { return (type != NULL && type->kind == ptxTypeU64); }
Bool isI2  (ptxType type) { return (isU2(type) || isS2(type)); }
Bool isI4  (ptxType type) { return (isU4(type) || isS4(type)); }
Bool isI8  (ptxType type) { return (isU8(type) || isS8(type)); }
Bool isI16 (ptxType type) { return (isU16(type) || isS16(type)); }
Bool isI32 (ptxType type) { return (isU32(type) || isS32(type)); }
Bool isI64 (ptxType type) { return (isU64(type) || isS64(type)); }
Bool isE4M3 (ptxType type) { return (type != NULL && type->kind == ptxTypeE4M3); }
Bool isE5M2 (ptxType type) { return (type != NULL && type->kind == ptxTypeE5M2); }
Bool isF8  (ptxType type) { return (isE4M3(type) || isE5M2(type)); }
Bool isE4M3x2 (ptxType type) { return (type != NULL && type->kind == ptxTypeE4M3x2); }
Bool isE5M2x2 (ptxType type) { return (type != NULL && type->kind == ptxTypeE5M2x2); }
Bool isF8x2  (ptxType type) { return (isE4M3x2(type) || isE5M2x2(type)); }
Bool isF16 (ptxType type) { return (type != NULL && type->kind == ptxTypeF16); }
Bool isF16x2 (ptxType type) { return (type != NULL && type->kind == ptxTypeF16x2); }
Bool isF32 (ptxType type) { return (type != NULL && type->kind == ptxTypeF32); }
Bool isF64 (ptxType type) { return (type != NULL && type->kind == ptxTypeF64); }
Bool isBF16  (ptxType type)  { return (type != NULL && type->kind == ptxTypeBF16); }
Bool isBF16x2 (ptxType type) { return (type != NULL && type->kind == ptxTypeBF16x2); }
Bool isTF32 (ptxType type)  { return (type != NULL && type->kind == ptxTypeTF32);}
Bool isPRED(ptxType type) { return (type != NULL && type->kind == ptxTypePred); }
Bool isTEXREF    (ptxType type) { return (type != NULL && type->kind==ptxOpaqueType && stdEQSTRING(type->cases.Opaque.name,".texref"));     }
Bool isSAMPLERREF(ptxType type) { return (type != NULL && type->kind==ptxOpaqueType && stdEQSTRING(type->cases.Opaque.name,".samplerref")); }
Bool isSURFREF   (ptxType type) { return (type != NULL && type->kind==ptxOpaqueType && stdEQSTRING(type->cases.Opaque.name,".surfref"));    }
Bool isArray (ptxType type) { return type != NULL && type->kind == ptxArrayType;}
Bool isSignedInt(ptxType type) { return isS8(type) || isS16(type) || isS32(type) || isS64(type); }
Bool isFloatKind(ptxTypeKind kind) { return kind == ptxTypeF16 || kind == ptxTypeF32 || kind == ptxTypeF64; }
Bool isFloat(ptxType type)  { return (type && isFloatKind(type->kind)); }

Bool isIntegerKind(ptxTypeKind kind)
{
    return kind == ptxTypeU8  || kind == ptxTypeU16 || kind == ptxTypeU32 ||
           kind == ptxTypeU64 || kind == ptxTypeS8  || kind == ptxTypeS16 ||
           kind == ptxTypeS32 || kind == ptxTypeS64;
}
Bool isInteger(ptxType type){ return (type && isIntegerKind(type->kind)); }

Bool isBitTypeKind(ptxTypeKind kind) {
    return kind == ptxTypeB8  || kind == ptxTypeB16 ||
           kind == ptxTypeB32 || kind == ptxTypeB64 ||
           kind == ptxTypeB128;
}
Bool isBitType(ptxType type) {
    return (type && isBitTypeKind(type->kind));
}


// This function returns True if all .atype, .btype, 
// .ctype and .dtype are .f16 else returns False
Bool areAllFourMatrixTypesF16(ptxType instructionType[ptxMAX_INSTR_ARGS], 
                              uInt nrofInstructionTypes)
{
    int i;

    if (nrofInstructionTypes < 4) { return False; }

    for (i = 0; i < 4 ; ++i) {
        if (!isF16(instructionType[i])) {
            return False;
        }
    }
    return True;
}

// Functions to check Type modifiers

Bool isU4Mod(uInt x)         { return (x == ptxTYPE_u4_MOD);}
Bool isS4Mod(uInt x)         { return (x == ptxTYPE_s4_MOD);}
Bool isS2Mod(uInt x)         { return (x == ptxTYPE_s2_MOD);}
Bool isS32Mod(uInt x)        { return (x == ptxTYPE_s32_MOD);}
Bool isB1Mod(uInt x)         { return (x == ptxTYPE_b1_MOD);}
Bool isB32Mod(uInt x)        { return (x == ptxTYPE_b32_MOD);}
Bool isI4Mod(uInt x)         { return (x == ptxTYPE_u4_MOD || x == ptxTYPE_s4_MOD); }
Bool isI2Mod(uInt x)         { return (x == ptxTYPE_u2_MOD || x == ptxTYPE_s2_MOD); }
Bool isI8Mod(uInt x)         { return (x == ptxTYPE_u8_MOD || x == ptxTYPE_s8_MOD); }
Bool isI16Mod(uInt x)        { return (x == ptxTYPE_u16_MOD || x == ptxTYPE_s16_MOD); }
Bool isF32Mod(uInt x)        { return (x == ptxTYPE_f32_MOD);  }
Bool isE4M3Mod(uInt x)       { return (x == ptxTYPE_e4m3_MOD); }
Bool isE5M2Mod(uInt x)       { return (x == ptxTYPE_e5m2_MOD); }
Bool isBF16Mod(uInt x)       { return (x == ptxTYPE_BF16_MOD); }
Bool isU8U4(uInt x, uInt y)  { return (x == ptxTYPE_u8_MOD && y == ptxTYPE_u4_MOD); }
Bool isS8S4(uInt x, uInt y)  { return (x == ptxTYPE_s8_MOD && y == ptxTYPE_s4_MOD); }
Bool isU4U2(uInt x, uInt y)  { return (x == ptxTYPE_u4_MOD && y == ptxTYPE_u2_MOD); }
Bool isS4S2(uInt x, uInt y)  { return (x == ptxTYPE_s4_MOD && y == ptxTYPE_s2_MOD); }
// end
