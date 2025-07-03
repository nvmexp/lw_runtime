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
 *  Module name              : ptxInstructionTemplates.c
 *
 *  Description              :
 *
 */

/*--------------------------------- Includes ---------------------------------*/

#include "g_lwconfig.h"
#include "ptxInstructionTemplates.h"
#include "ptxInstructions.h"
#include "ptxConstructors.h"
#include "ptxparseMessageDefs.h"
#include "ptxMacroUtils.h"
#include "ctLog.h"
#include "stdBitSet.h"
#include "ptxDescriptorReader.h"

#define ptxMAX_TEMPLATE_MATCHES 128

/*------------------------------- Module State -------------------------------*/


/*------------------------------ Template Match ------------------------------*/

Bool areExtendedInstructionsEnabled(ptxParseData parseData)
{
    int size = mapSize(parseData->extTemplates);
    return size != 0;
}

static Bool isRegisterArgument (ptxExpression arg)
{
    if (arg->kind == ptxSymbolExpression && ptxIsRegisterStorage((arg->cases.Symbol.symbol)->storage)) return True;
    if (arg->kind == ptxVectorExpression) return True;

    return False;
}

  /*
   * Allow legacy ptx code for special registers whose size was redefined:
   * - PTX 2.0 redefined %tid, %ntid, %ctaid, and %nctaid to be .v4.u32 instead of .v4.u16
   * - PTX 1.3 redefined %gridid to be .u32 instead of .u16
   * - PTX 3.0 redefined %gridid to be .u64 instead of .u32
   */
  static Bool isLegacySRegArgument (ptxExpression arg, uInt64 size, uInt64 typeSize)
  {
      if (arg->kind == ptxVectorSelectExpression) {
          return isLegacySRegArgument(arg->cases.VectorSelect->arg, size, typeSize);
      }
      if (arg->kind == ptxSymbolExpression) {
          ptxSymbolTableEntry sym = arg->cases.Symbol.symbol;

          if ( sym->storage.kind == ptxSregStorage &&
               ( strcmp(sym->symbol->name, "%tid"   ) == 0 ||
                 strcmp(sym->symbol->name, "%ntid"  ) == 0 ||
                 strcmp(sym->symbol->name, "%ctaid" ) == 0 ||
                 strcmp(sym->symbol->name, "%nctaid") == 0 ) ) {
              return ( size == 16 && typeSize == 32 );
          }
          if ( sym->storage.kind == ptxSregStorage && strcmp(sym->symbol->name, "%gridid") == 0 ) {
            return ( (size == 16 && typeSize == 32) ||  // allow .u16 reads in PTX ISA < 3.0
                     (size == 16 && typeSize == 64) ||  // allow .u16 reads in PTX ISA >= 3.0
                     (size == 32 && typeSize == 64) );  // allow .u32 reads in PTX ISA >= 3.0
          }
      }
      return False;
  }

/*
 * Match arguments in FollowAType positions to corresponding instruction type.
 */
  static Bool largeArgFeatureApplies(ptxInstructionFeature features, ptxInstructionType iType, ptxTypeKind argTypeKind)
  {
      // Argument may be larger than instruction-type size iff:
      // . instruction has 'large argument' feature flag, AND
      // . (inst-type is .bX) OR (arg-type is .bX) OR (inst-type not .fX AND arg-type not .fX)

      return ( ptxHasLARG_Feature(features) &&
               ((iType == ptxBitIType) || (isBitTypeKind(argTypeKind) || (iType != ptxFloatIType  && !isFloatKind(argTypeKind)))));
  }

static Bool matchArgType( ptxExpression argument, uInt argIndex, ptxInstructionType iType, uInt size, ptxInstructionFeature features, Bool vectorMode )
{
    ptxType           type       = argument->type;
    ptxExpressionKind kind       = argument->kind;
    Bool              isConstant = argument->isConstant;
    Bool              isRegister = isRegisterArgument(argument);
    ptxTypeKind       typeKind   = type->kind;
    uInt64            typeSize   = ptxGetTypeSizeInBits(type);

    if (typeKind == ptxVectorType) {
        if (kind == ptxVectorExpression) {
            ptxExpression expr = ptxGetElementExprFromVectorExpr(argument, 0);
            kind = expr->kind;
        }
        typeKind = (type->cases.Vector.base)->kind;
        typeSize = ptxGetTypeSizeInBits(type->cases.Vector.base);

        // if vector operand is matching a non-vector bit-type instruction type, get total operand size
        if (iType == ptxBitIType && !vectorMode) {
            typeSize *= type->cases.Vector.N;
        }
    } else {
        /*
         * 'normalize' the argument type size values and prevent match in boundary cases:
         */
        // DOUBLERES applies to DEST and SRC3 (see MUL, MAD instructions)
        if ( (argIndex == 0 || argIndex == 3) && ptxHasDOUBLERES_Feature(features) ) {
            if ( typeSize >= 16 ) { typeSize /= 2; } else { typeSize= 0xffff; }
        }    
    }

   /*
    * Allow Bit types in place of Int and Float types (and vice verse), as long as they
    * match in size.
    */
    if (iType == ptxPackedHalfFloatIType) {
       if (kind     == ptxIntConstantExpression  ) { return False; }
       if (kind     == ptxFloatConstantExpression) { return False; }
       if (isFloatKind(typeKind)                 ) { return False; }
       if (isIntegerKind(typeKind)               ) { return False; }
       if (typeKind == ptxTypePred               ) { return False; }
       if (typeKind == ptxOpaqueType             ) { return False; }
    } else
    if (iType == ptxFloatIType) {
       if (kind     == ptxFloatConstantExpression) { return size == 32 || size == 64; }
       if (typeKind == ptxTypeF16x2              ) { return False; }
       if (isInteger(type)                       ) { return False; }
       if (typeKind == ptxTypePred               ) { return False; }
       if (typeKind == ptxOpaqueType             ) { return False; }

       if (kind == ptxIntConstantExpression && isBitTypeKind(typeKind)) {
           // special case to allow vector with first element as integer and rest as float for texture arrays.
           return True;
       }
    } else 
    if (iType == ptxIntIType) {
       if (kind     == ptxIntConstantExpression  ) { return True;  }
       if (isFloatKind(typeKind)                 ) { return False; }
       if (typeKind == ptxTypePred               ) { return False; }
       if (typeKind == ptxOpaqueType             ) { return False; }
    } else 
    if (iType == ptxBitIType) {
       if (kind     == ptxIntConstantExpression  ) { return True;  }
       if (kind     == ptxFloatConstantExpression) { return (size == typeSize);  }
       if (typeKind == ptxTypePred               ) { return False; }
       if (typeKind == ptxOpaqueType             ) { return False; }
    } else 
    if (iType == ptxPredicateIType) {
       if (typeKind == ptxTypePred               ) { return True;  }
       if (kind     == ptxIntConstantExpression  ) { return True;  }
       if (isIntegerKind(typeKind)               ) { return False; }
       if (isFloatKind(typeKind)                 ) { return False; }
       if (isBitTypeKind(typeKind)               ) { return False; }
       if (typeKind == ptxOpaqueType             ) { return False; }
    } else
    if (iType == ptxOpaqueIType) {
       if (isInteger(type)                       ) { return (size == 8 && !isSignedInt(type)); }
       else                                        { return False; }
    }

    // *** FIXME *** we should eventually get rid of this, and match types more precisely
    // - start by disallowing labels
    if (isConstant && !isRegister) {
        ptxExpression expr = argument;
        if (expr->kind == ptxLabelReferenceExpression) return False;
        if (expr->kind == ptxBinaryExpression) {
            expr = expr->cases.Binary->left;
        }
        if (expr->kind == ptxSymbolExpression
            && !ptxIsAddressableSymbol(expr->cases.Symbol.symbol))
        {
            return False;
        }
        return True;
    }

    // okay if:
    // . instruction-type and operand sizes match
    // . instruction-type size is smaller than operand size, and instruction has 'large argument' feature
    // . legacy special-register %tid, %ntid, %ctaid, %ctaid, %gridid
    return ( (size == typeSize) ||
             (size < typeSize && largeArgFeatureApplies(features, iType, typeKind)) ||
             (isLegacySRegArgument(argument, size, typeSize)) );
}

static Bool matchArguments(ptxInstructionTemplate t,
                           uInt nrofArguments,
                           ptxExpression arguments[],
                           Bool vectorMode,
                           ptxInstructionType instrTypeBuffer[],
                           uInt instrTypeSizeBuffer[])
{
    // OPTIX_HAND_EDIT Fix signed/unsigned comparison warnings
    uInt i;

    for (i = 0; i < t->nrofArguments; i++) {
        ptxExpression     argument   = arguments[i];
        ptxType           type       = argument->type;
        ptxExpressionKind kind       = argument->kind;

        if (kind == ptxSinkExpression) { /* 'sink' variable, continue */
            continue;
        }

        switch (t->argType[i]) {
        case ptxFollowAType        : if (!matchArgType( argument, i, instrTypeBuffer[t->followMap[i]], instrTypeSizeBuffer[t->followMap[i]],
                                                        t->features, vectorMode ) )
                                     { return False; }
                                     break;
        case ptxU16AType           : // match I16, or vectors of I16
                                     if (kind == ptxVectorExpression) {
                                         ptxExpression expr;
                                         expr = ptxGetElementExprFromVectorExpr(argument, 0);
                                         kind = expr->kind;
                                     }
                                     if (type->kind == ptxVectorType) { type = type->cases.Vector.base; }
                                     if ( !isI16(type) && !isB16(type) && (kind != ptxIntConstantExpression) ) { return False; }
                                     break;
        case ptxU32AType           : if ((!isI32(type) && !isB32(type) && (kind != ptxIntConstantExpression))
                                          || (kind == ptxAddressRefExpression)
                                          || (kind == ptxAddressOfExpression))
                                     { return False; }
                                     break;
        case ptxU64AType           : if ((!isI64(type) && !isB64(type) && (kind != ptxIntConstantExpression))
                                          || (kind == ptxAddressRefExpression)
                                          || (kind == ptxAddressOfExpression))
                                     { return False; }
                                     break;
        case ptxS32AType           : // match I32/B32, or vectors of I32/B32
                                     if (kind == ptxVectorExpression) {
                                         ptxExpression expr;
                                         expr = ptxGetElementExprFromVectorExpr(argument, 0);
                                         kind = expr->kind;
                                     }
                                     if (type->kind == ptxVectorType) { type = type->cases.Vector.base; }
                                     if ((!isI32(type) && !isB32(type) && kind != ptxIntConstantExpression)
                                          || (kind == ptxAddressRefExpression)
                                          || (kind == ptxAddressOfExpression))
                                     { return False; }
                                     break;
        case ptxF32AType           : // match F32/B32, or vectors of F32/B32
                                     if (kind == ptxVectorExpression) {
                                         ptxExpression expr;
                                         expr = ptxGetElementExprFromVectorExpr(argument, 0);
                                         kind = expr->kind;
                                     }
                                     if (type->kind == ptxVectorType) { type = type->cases.Vector.base; }
                                     if ( !isF32(type) && !isB32(type) && kind != ptxFloatConstantExpression ) { return False; }
                                     break;
        case ptxB32AType           : // match any type of size 32 bit or its vector                               
                                     if (type->kind == ptxVectorType) { type = type->cases.Vector.base; }
                                     if (ptxGetTypeSizeInBits(type) != 32) { return False; }
                                     // note : type->kind is not checked here
                                     break;
        case ptxB64AType           : // match any type of size 64 bit or its vector
                                     if (type->kind == ptxVectorType) { type = type->cases.Vector.base; }
                                     if (ptxGetTypeSizeInBits(type) != 64) { return False; }
                                     // note : type->kind is not checked here
                                     break;
        case ptxImageAType         : {
                                         ptxType baseType = ptxGetAddressArgBaseType(argument);
                                         if (!(isTEXREF(baseType) || isSAMPLERREF(baseType) || isSURFREF(baseType)
                                               || isI64(baseType) || isS64(baseType)        || isB64(baseType)))
                                         { return False; }
                                     }
                                     break;
        case ptxScalarF32AType     : if (!(isF32(type) || isB32(type) || kind == ptxFloatConstantExpression)) { return False; }
                                     break;
        case ptxF16x2AType         : // match F16x2/B32, or vectors of F16x2/B32
                                     if (type->kind == ptxVectorType) { type = type->cases.Vector.base; }
                                     if ( !isF16x2(type) && !isB32(type) ) { return False; }
                                     break;
        case ptxConstantIntAType   : if (kind != ptxIntConstantExpression)   { return False; }
                                     break;
        case ptxConstantFloatAType : if (kind != ptxFloatConstantExpression) { return False; }
                                     break;
        case ptxPredicateAType     : if (type->kind != ptxTypePred && kind != ptxIntConstantExpression ) { return False; }
                                     break;
        case ptxPredicateVectorAType: // match predicate or vector of predicates
                                     if (kind == ptxVectorExpression) {
                                         ptxExpression expr;
                                         expr = ptxGetElementExprFromVectorExpr(argument, 0);
                                         kind = expr->kind;
                                     }
                                     if (type->kind == ptxVectorType) { type = type->cases.Vector.base; }
                                     if ( !isPRED(type) && kind != ptxIntConstantExpression ) { return False; }
                                     break;
        case ptxMemoryAType        : // Don't infer anything from memory argument type
                                     break;
        case ptxSymbolAType        : if (kind != ptxSymbolExpression ) { return False; }
                                     break;
        case ptxTargetAType        : if (type->kind != ptxLabelType && !isInteger(type) && !isBitTypeKind(type->kind)) { return False; }
                                     break;
        case ptxParamListAType     : if (kind != ptxParamListExpression ) { return False; }
                                     break;
        case ptxVoidAType          : break;   // 'void' type: any type checking is instruction-specific
        case ptxLabelAType         : if (type->kind != ptxLabelType) { return False; }
                                     // for an already defined symbol, make sure it is a label symbol
                                     if (kind == ptxSymbolExpression &&
                                         (ptxGetSymEntFromExpr(argument)->kind != ptxLabelSymbol))
                                     { return False; }
                                     break;
        default                    : stdASSERT( False, ("Case label out of bounds") );
        }
    }

    return True;
}

static Bool matchInstructionTypes(ptxInstructionTemplate t,
                                  ptxType instrType[],
                                  uInt nrofInstrTypes,
                                  ptxInstructionType instrTypeBuffer[],
                                  uInt instrTypeSizeBuffer[])
{
    // OPTIX_HAND_EDIT Fix signed/unsigned comparison warnings
    uInt i;

    if (nrofInstrTypes != t->nrofInstrTypes) return False;
    if (!nrofInstrTypes) {
        // No instruction type. Matching template cannot have FollowAType arguments.
        for (i = 0; i < t->nrofArguments; i++) {
            if (t->argType[i] == ptxFollowAType) return False;
        }
    }

    // Check if the current template matches the imposed instruction type:
    for (i = 0; i < t->nrofInstrTypes; i++) {
        if (instrTypeBuffer[i] != t->instrType[i]) return False;
        if (instrType[i]->kind != ptxOpaqueType) {
            if (!(bitSetElement(t->instrTypeSizes[i], instrTypeSizeBuffer[i])))
                return False;
        }
    }

    return True;
}

static Bool checkUsesSYNC(ptxParseData parseData)
{
    return ptxHasSYNC_MOD(parseData->modifiers);
}

static Bool checkAllowsSYNC(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasSYNC_Feature(t->features);
}

static Bool checkUsesPOSTOP(ptxParseData parseData)
{
    return parseData->postopclass != NoOperationClass;
}

static Bool checkAllowsPOSTOP(ptxParseData parseData, ptxInstructionTemplate t)
{
    if (parseData->postop == ptxCASOp) return ptxHasCAS_Feature(t->features);
    // No separate check for AtomicOperationClass. Pass all postops
    // through for atomics, since there are better checks for atomic
    // operations in the parser.
    if (ptxHasATOMIC_Feature(t->features)) return True;
    if (parseData->postopclass & BoolOperationClass) return ptxHasBOP_Feature(t->features);
    if (parseData->postopclass & ArithOperationClass) return ptxHasARITHOP_Feature(t->features);
    return True;
}

static Bool checkUsesDESC(ptxParseData parseData)
{
    return ptxHasDESC_MOD(parseData->modifiers);
}

static Bool checkAllowsDESC(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasDESC_Feature(t->features);
}

static Bool checkUsesRELU(ptxParseData parseData)
{
    return ptxHasRELU_MOD(parseData->modifiers);
}

static Bool checkAllowsRELU(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasRELU_Feature(t->features);
}

static Bool checkUsesSATF(ptxParseData parseData)
{
    return ptxHasSATF_MOD(parseData->modifiers);
}

static Bool checkAllowsSATF(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasSATF_Feature(t->features);
}

static Bool checkUsesADDRTYPE(ptxParseData parseData)
{
    return ptxHasADDRTYPE_MOD(parseData->modifiers);
}

static Bool checkAllowsADDRTYPE(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasADDRTYPE_Feature(t->features);
}

static Bool checkUsesMULTICAST(ptxParseData parseData)
{
    return ptxHasMULTICAST_MOD(parseData->modifiers);
}

static Bool checkAllowsMULTICAST(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasMULTICAST_Feature(t->features);
}

static Bool checkUsesIM2COL(ptxParseData parseData)
{
    return ptxHasIM2COL_MOD(parseData->modifiers);
}

static Bool checkAllowsIM2COL(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasIM2COL_Feature(t->features);
}

static Bool checkUsesPACKEDOFF(ptxParseData parseData)
{
    return ptxHasPACKEDOFF_MOD(parseData->modifiers);
}

static Bool checkAllowsPACKEDOFF(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasPACKEDOFF_Feature(t->features);
}

static Bool checkUsesCACHEPREFETCH(ptxParseData parseData)
{
    return ptxHasCACHEPREFETCH_MOD(parseData->modifiers);
}

static Bool checkAllowsCACHEPREFETCH(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasCACHEPREFETCH_Feature(t->features);
}

static Bool checkUsesPREFETCHSIZE(ptxParseData parseData)
{
    return ptxHasPREFETCHSIZE_MOD(parseData->modifiers);
}

static Bool checkAllowsPREFETCHSIZE(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasPREFETCHSIZE_Feature(t->features);
}

static Bool checkUsesCACHEHINT(ptxParseData parseData)
{
    return ptxHasCACHEHINT_MOD(parseData->modifiers);
}

static Bool checkAllowsCACHEHINT(ptxParseData parseData, ptxInstructionTemplate t)
{
    return ptxHasCACHEHINT_Feature(t->features);
}

// If the instruction uses feature XYZ, then remove templates that do
// not allow XYZ. This is useful when the feature affects the template
// signature (types and operands). For example, without this fix, if
// XYZ needs an extra argument, and if that argument is missing, then
// the wrong template may be matched. This can result in an incorrect
// error message saying feature XYZ is not supported.
#define PRUNE_FOR_FEATURE(XYZ)                                          \
    if (checkUses##XYZ(parseData)) {                                    \
        for (i = 0; i < totalMatches; ++i) {                            \
            ptxInstructionTemplate t = matched[i];                      \
            if (!t) continue;                                           \
            if (!checkAllows##XYZ(parseData, t)) {                      \
                matched[i] = NULL;                                      \
                --prunedMatches;                                        \
            }                                                           \
        }                                                               \
        if (prunedMatches == 0) {                                       \
            stdCHECK_WITH_POS(False, (ptxMsgIllegalModifier, sourcePos, \
                                      get_str##XYZ(parseData), name)); \
            return NULL;                                                \
        }                                                               \
    }

uInt ptxGetInstructionOpcode(ptxParseData parseData, String name)
{
    stdList_t templates = mapApply(parseData->stdTemplates, name);

    if (!templates) {
        templates = mapApply(parseData->extTemplates, name);
    }

    // We do not emit an error for unknown instruction here since this
    // is handled later in preprocessInstruction. That check also
    // filters instructions in macro bodies, which should not be
    // filtered here. Callers should be ready to handle an unknown
    // opcode returned from here.
    if (!templates) return ptx_unknown_Instr;

    return ((ptxInstructionTemplate)templates->head)->code;
}

static uInt copyTemplateListToArray(ptxInstructionTemplate dst[],
                                    stdList_t templates, uInt count)
{
    while (templates) {
        stdASSERT(count < ptxMAX_TEMPLATE_MATCHES,
                  ("exceeded available space for matches"));
        dst[count++] = templates->head;
        templates = templates->tail;
    }
    return count;
}

/*
 * Function         : Match parsed instruction information to templates
 * Parameters       : name           (I) Instruction name
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
            )
{
    uInt i;
    ptxInstructionType  instrTypeBuffer     [ptxMAX_INSTR_ARGS];
    uInt                instrTypeSizeBuffer [ptxMAX_INSTR_ARGS];
    Bool                vectorMode = ptxHasVECTOR_MOD(parseData->modifiers);
    stdList_t           templates;
    stdList_t           moreTemplates;

    ptxInstructionTemplate matched[ptxMAX_TEMPLATE_MATCHES];
    uInt                   totalMatches = 0;
    uInt                   prunedMatches = 0;

    stdASSERT( nrofArguments  <= ptxMAX_INSTR_ARGS, ("Arguments incorrectly filtered by caller") );
    stdASSERT( nrofInstrTypes <= ptxMAX_INSTR_ARGS, ("Arguments incorrectly filtered by caller") );

    // Impose instruction type from caller
    for (i = 0; i < nrofInstrTypes; ++i) {
        switch (instrType[i]->kind) {
        case ptxTypeF16:
        case ptxTypeF32:
        case ptxTypeF64:
            instrTypeBuffer[i] = ptxFloatIType;
            instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]);
            break;
        case ptxTypeF16x2:
            instrTypeBuffer[i] = ptxPackedHalfFloatIType;
            instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]);
            break;
        case ptxTypeBF16:
           instrTypeBuffer[i] = ptxLwstomFloatE8IType;
           instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]);
           break;
        case ptxTypeTF32:
           instrTypeBuffer[i] = ptxLwstomFloatTF32Type;
           instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]);
           break;
        case ptxTypeE4M3:
        case ptxTypeE5M2:
        case ptxTypeE4M3x2:
        case ptxTypeE5M2x2:
           instrTypeBuffer[i] = ptxLwstomFloatFP8Type;
           instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]);
           break;
        case ptxTypeBF16x2:
           instrTypeBuffer[i] = ptxLwstomFloatE8IType;
           instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]);
           break;
        case ptxTypeU2:
        case ptxTypeU4:
        case ptxTypeU8:
        case ptxTypeU16:
        case ptxTypeU32:
        case ptxTypeU64:
        case ptxTypeS2:
        case ptxTypeS4:
        case ptxTypeS8:
        case ptxTypeS16:
        case ptxTypeS32:
        case ptxTypeS64:
            instrTypeBuffer[i] = ptxIntIType;
            instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]);
            break;
        case ptxTypeB1:
        case ptxTypeB2:
        case ptxTypeB4:
        case ptxTypeB8:
        case ptxTypeB16:
        case ptxTypeB32:
        case ptxTypeB64:
        case ptxTypeB128:
            instrTypeBuffer[i] = ptxBitIType;
            instrTypeSizeBuffer[i] = (int) ptxGetTypeSizeInBits(instrType[i]);
            break;
        case ptxTypePred:
            instrTypeBuffer[i] = ptxPredicateIType;
            instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]);
            break;
        case ptxOpaqueType           : instrTypeBuffer[i] = ptxOpaqueIType;          instrTypeSizeBuffer[i]= (int) ptxGetTypeSizeInBits(instrType[i]); break;
        default                      : stdASSERT(0, ("Unhandled type"));
        }
    }

#if !defined(RELEASE) || LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    // Uncoditionally allow all instructions in an internal build.
    templates = mapApply( parseData->stdTemplates, name );
#else
    // For external builds:
    // If the name starts with an underscore, that instruction is
    // only recognized inside a macro.
    if (name[0] == '_' && !parsingMacro) {
       templates = NULL;
    } else {
        templates = mapApply(parseData->stdTemplates, name );
    }
#endif
    // Extended instructions are allowed unconditionally in user
    // code. Hence we do not need to check the initial underscore.
    //
    // Note that an instruction can be both standard and extended. In
    // addition, it can also start with an underscore. Such an
    // instruction will be treated as follows:
    // 1. It is allowed in a macro because it passes the earlier check.
    // 2. It is allowed in user code if extended templates are initialized.
    moreTemplates = mapApply(parseData->extTemplates, name);

    if (!templates && !moreTemplates) {
        stdCHECK_WITH_POS( False, (ptxMsgUnknownInstructionName, sourcePos, name) );
        return NULL;
    }

    // Copy the linked list of templates into a candidate array so
    // that we can remove candidates in subsequent pruning steps
    // without having to deal with linked list copies.
    //
    // TODO: Pruning introduces holes in the array, which need to be
    // skipped at each traversal. A further optimization could compact
    // the array instead, by either creating a copy, or compacting
    // in-place while pruning. But this does not seem super
    // important. Most instructions have less than 16 templates, and
    // only in the rarest of the rare case (textures), do we see more
    // than 32 templates.
    totalMatches = copyTemplateListToArray(matched, templates, totalMatches);
    totalMatches = copyTemplateListToArray(matched, moreTemplates, totalMatches);
    prunedMatches = totalMatches;

    PRUNE_FOR_FEATURE(SYNC);
    PRUNE_FOR_FEATURE(POSTOP);
    PRUNE_FOR_FEATURE(DESC);
    PRUNE_FOR_FEATURE(RELU);
    PRUNE_FOR_FEATURE(SATF);
    PRUNE_FOR_FEATURE(CACHEPREFETCH);
    PRUNE_FOR_FEATURE(CACHEHINT);
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
    PRUNE_FOR_FEATURE(ADDRTYPE);
    PRUNE_FOR_FEATURE(MULTICAST);
    PRUNE_FOR_FEATURE(IM2COL);
    PRUNE_FOR_FEATURE(PACKEDOFF);
    PRUNE_FOR_FEATURE(PREFETCHSIZE);
#endif

    for (i = 0; i < totalMatches; ++i) {
        ptxInstructionTemplate t = matched[i];
        if (!t) continue;
        if (!matchInstructionTypes(t, instrType, nrofInstrTypes,
                                  instrTypeBuffer, instrTypeSizeBuffer))
        {
            matched[i] = NULL;
            --prunedMatches;
        }
    }
    if (prunedMatches == 0) {
        stdCHECK_WITH_POS(False, (ptxMsgNonMatchingInstrTypes, sourcePos, name));
        return NULL;
    }

    for (i = 0; i < totalMatches; ++i) {
        ptxInstructionTemplate t = matched[i];
        if (!t) continue;
        if (nrofArguments != t->nrofArguments ) continue;

        if (matchArguments(t, nrofArguments, arguments, vectorMode,
                           instrTypeBuffer, instrTypeSizeBuffer))
        { return t; }
    }

    stdCHECK_WITH_POS( False, (ptxMsgNonMatchingInstrArgs, sourcePos, name ) );
    return NULL;
}

/*----------------------------- Template Storage -----------------------------*/

static void addTemplate(ptxParseData parseData, ptxInstructionTemplate instr, Bool isExtended)
{
    stdList_t prev, current;
    stdMap_t targetMap = isExtended ? parseData->extTemplates : parseData->stdTemplates;

    prev    = mapApply(targetMap, instr->name);
    current = listCons(instr, prev);

    mapDefine(targetMap, instr->name, current);
}

/*------------------------ Parsing Template Definition -----------------------*/
                
    static uInt countTypes( String s )
    {
        char c;
        uInt result= 0;
        while ((c = *s++)) {
            if ( isalpha(c) ) { result++; }
        }
        return result;
    }

static void addInstructionTemplate(ptxParseData parseData, String type, String name, String signature,
                                   ptxInstructionFeature features,
                                   ptxInstructionCode code, Bool isExtended)
{
    Int  t;
    uInt i;
    uInt size = 0;
    ptxInstructionTemplate result;

    uInt nrofArgTypes   = (uInt)(strlen(signature));
    uInt nrofInstrTypes = countTypes(type);

    stdASSERT( nrofArgTypes <= ptxMAX_INSTR_ARGS, ("ptxMAX_INSTR_ARGS too small, '%s'",name) );

    stdNEW(result);

    result->name           = name;
    result->code           = code;
    result->features       = features;
    result->nrofArguments  = nrofArgTypes;
    result->nrofInstrTypes = nrofInstrTypes;

    // unpack instruction types and allowed sizes

    for (i=0, t=-1; i<strlen(type); i++) {
        switch (type[i]) {
        case 'F' : result->instrType[++t] = ptxFloatIType;
            // if no size restrictions follow, default to byte sizes 4 and 8
            result->instrTypeSizes[t] = bitSetCreate();
            if (i+1==strlen(type) || isalpha(type[i+1])) {
                bitSetInsert(result->instrTypeSizes[t], 32);
                bitSetInsert(result->instrTypeSizes[t], 64);
            } else {
                bitSetInsert(result->instrTypeSizes[t], 0);
            }
            break;

        case 'H' : result->instrType[++t] = ptxPackedHalfFloatIType;
            // if no size restrictions follow, default to byte sizes 4
            result->instrTypeSizes[t] = bitSetCreate();
            if (i+1==strlen(type) || isalpha(type[i+1])) {
                bitSetInsert(result->instrTypeSizes[t], 32);
            } else {
                bitSetInsert(result->instrTypeSizes[t], 0);
            }
            break;

        case 'I' : result->instrType[++t] = ptxIntIType;
            // if no size restrictions follow, default to byte sizes 2, 4, and 8
            result->instrTypeSizes[t] = bitSetCreate();
            if (i+1==strlen(type) || isalpha(type[i+1])) {
                bitSetInsert(result->instrTypeSizes[t], 16);
                bitSetInsert(result->instrTypeSizes[t], 32);
                bitSetInsert(result->instrTypeSizes[t], 64);
            } else {
                bitSetInsert(result->instrTypeSizes[t], 0);
            }
            break;

        case 'B' : result->instrType[++t] = ptxBitIType;
            // if no size restrictions follow, default to byte sizes 2, 4, and 8
            result->instrTypeSizes[t] = bitSetCreate();
            if (i+1==strlen(type) || isalpha(type[i+1])) {
                bitSetInsert(result->instrTypeSizes[t], 16);
                bitSetInsert(result->instrTypeSizes[t], 32);
                bitSetInsert(result->instrTypeSizes[t], 64);
            } else {
                bitSetInsert(result->instrTypeSizes[t], 0);
            }
            break;

        case 'P' :
            stdASSERT( i+1==strlen(type) || isalpha(type[i+1]), ("Type size restrictions not allowed for 'P' type") );

            result->instrType[++t]    = ptxPredicateIType;
            result->instrTypeSizes[t] = bitSetCreate();
            bitSetInsert(result->instrTypeSizes[t], 32);   // predicates have size==4
            break;
        case 'O':
            stdASSERT( i+1==strlen(type) || isalpha(type[i+1]), ("Type size restrictions not allowed for 'O' type") );

            result->instrType[++t]    = ptxOpaqueIType;
            result->instrTypeSizes[t] = bitSetCreate();
            bitSetInsert(result->instrTypeSizes[t], 0);   // use size==0 for opaques and do necessary checks separately
            break;
        case 'E':
            // 'E' represent custom float with exponent 8 (e8m*) types and mantissa
            //  value is represented by size
            result->instrType[++t] = ptxLwstomFloatE8IType;
            result->instrTypeSizes[t] = bitSetCreate();
            break;
        case 'T':
            // 'T' represent custom float with exponent 8 and mantissa 10 -> .tf32
            //  only 32 value is allowed for size
            result->instrType[++t] = ptxLwstomFloatTF32Type;
            result->instrTypeSizes[t] = bitSetCreate();
            break;
        case 'Q':
            // 'Q' represent custom float with size 8 bit (e4m3/e5m2) types and mantissa
            result->instrType[++t] = ptxLwstomFloatFP8Type;
            result->instrTypeSizes[t] = bitSetCreate();
            if (i+1==strlen(type) || isalpha(type[i+1])) {
                bitSetInsert(result->instrTypeSizes[t], 8);
                bitSetInsert(result->instrTypeSizes[t], 16);
            } else {
                bitSetInsert(result->instrTypeSizes[t], 0);
            }
            break;
        case '[': break;
        case '|':
        case ']':
                bitSetInsert(result->instrTypeSizes[t], size);
                size = 0;
            break;
        default:
            if ('0' <= type[i] && type[i] <= '9') {
                size = size*10 + (type[i] - '0');
            } else {
                stdASSERT( False, ("Unknown instruction type: '%c'",type[i]) );
                break;
            }
            if (i+1==strlen(type) || isalpha(type[i+1])) {
                bitSetInsert(result->instrTypeSizes[t], size);
                size = 0;
            }
            break;
        }

        stdASSERT( t < (Int)nrofInstrTypes, ("Instruction type error, '%s'",name) );
    }

    /*
     * A single digit indicates that the argument in the current position follows the specified
     * instruction type, where instruction types are numbered starting with zero.
     *
     * For example, the instruction SET has argument type signature of '011'.  The instruction
     * SET.eq.f32.u32 d,a,b; would therefor map dest arg 'd' to type .f32, and map both source
     * args 'a' and 'b' to type .u32.
     */
    for (i=0; i<nrofArgTypes; i++) {
        switch (signature[i]) {
        case 'x' : result->argType[i]= ptxU16AType;                                                    break;
        case 'u' : result->argType[i]= ptxU32AType;                                                    break;
        case 'U' : result->argType[i]= ptxU64AType;                                                    break;
        case 'd' : result->argType[i]= ptxB32AType;                                                    break; 
        case 'e' : result->argType[i]= ptxB64AType;                                                    break;
        case 's' : result->argType[i]= ptxS32AType;                                                    break;
        case 'f' : result->argType[i]= ptxF32AType;                                                    break;
        case 'l' : result->argType[i]= ptxScalarF32AType;                                              break;
        case 'i' : result->argType[i]= ptxImageAType;                                                  break;           
        case 'h' : result->argType[i]= ptxF16x2AType;                                                  break;
        case 'C' : result->argType[i]= ptxConstantIntAType;                                            break;
        case 'D' : result->argType[i]= ptxConstantFloatAType;                                          break;
        case 'P' : result->argType[i]= ptxPredicateAType;                                              break;
        case 'Q' : result->argType[i]= ptxPredicateVectorAType;                                        break;
        case 'M' : result->argType[i]= ptxMemoryAType;                                                 break;
        case 'S' : result->argType[i]= ptxSymbolAType;                                                 break;
        case 'T' : result->argType[i]= ptxTargetAType;                                                 break;
        case 'A' : result->argType[i]= ptxParamListAType;                                              break;
        case 'V' : result->argType[i]= ptxVoidAType;                                                   break;
        case 'L' : result->argType[i]= ptxLabelAType;                                                  break;
        default  :
            {
                uInt tindex = signature[i]-'0';
                
                stdASSERT( isdigit(signature[i]), ("Unknown argument type: '%c'",signature[i]) );
                stdASSERT( tindex < nrofInstrTypes, ("Instruction type index '%d' out of range",tindex) );
                
                result->argType[i]   = ptxFollowAType;
                result->followMap[i] = tindex;
            }
        }
    }
    addTemplate(parseData, result, isExtended);
}

static String findNext(String ptr)
{
    Char c = *ptr++;
    stdCHECK(c != ' ' && c != 0,
             (ptxMsgParsingError, "unexpected end of string"));

    while (c != ' ' && c != 0) { c = *ptr++; }
    *(ptr - 1) = 0;

    return ptr;
}

static String consumeString(String ptr, String *dst)
{
    String next = findNext(ptr);
    if (stdEQSTRING(ptr, ".")) {
        // replace place-holder with empty string
        *ptr = 0;
    }
    *dst = ptr;
    return next;
}

static String consumeUInt32(String ptr, uInt32 *dst)
{
    sscanf(ptr, "%u", dst);
    return findNext(ptr);
}

static String consumeHexInt32(String ptr, uInt32 *dst)
{
    sscanf(ptr, "%x", dst);
    return findNext(ptr);
}

static void ptxDefineExtendedInstructions(ptxParseData parseData, cString extDescFileName, cString extDescAsString)
{
    String extDescBuffer = NULL;
    uInt32 token = 0;
    String ptr;
    int i;

    #include "ptxExtInstrFeatures.incl"

    // extDescBuffer is initialized in the above file only for offline
    // compiler builds with GLOBAL_FEATURE_PTX_ISA_INTERNAL defined.
    // This ensures that the extended instructions do not get hard-coded
    // into ptxas.

#if !defined(PTX_EXT_DESC_STRING_LEN) || PTX_EXT_DESC_STRING_LEN == 0
    #error "Expected non-empty descriptor string"
#endif

#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL) && !defined(GPGPUCOMP_DRV_BUILD)
    stdASSERT(extDescBuffer,
              ("extended descriptors should be hard-coded in internal builds"));

    // Override the hard-coded extDescBuffer in case user has specified.
    if (extDescFileName || extDescAsString) {
        extDescBuffer = obtainExtDescBuffer(extDescFileName, extDescAsString,
                                            PTX_EXT_DESC_STRING_LEN);
    }

#else
    // This stdCHECK acts as a poor man's assert in a Release
    // build. The expression in the check will be constant-folded
    // depending on whether extDescBuffer was initialized or not.
    stdCHECK(!extDescBuffer, (ptxMsgParsingError, "template initialization",
                              "failed"));

    extDescBuffer = obtainExtDescBuffer(extDescFileName, extDescAsString,
                                        PTX_EXT_DESC_STRING_LEN);
#endif

    if (!extDescBuffer) return;

    ptr = extDescBuffer;

#if !defined(PTX_EXTENSION_TOKEN) || PTX_EXTENSION_TOKEN == 0
    // Ensure that we have the correct token constant from geninstr.pl
    #error "Invalid token"
#endif

    ptr = consumeHexInt32(ptr, &token);

    CT_DEBUG_PRINT("extended-instructions", 1, "Expected token value 0x%x, token value in descriptor file  0x%x\n", PTX_EXTENSION_TOKEN, token);

    if (token != PTX_EXTENSION_TOKEN) {
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        stdCHECK(False, (ptxMsgParsingError, "template initialization",
                         "\n***** Token did not match for extended instructions.\n"
                         "***** Please copy the extended descriptor file from\n"
                         "***** the same build as the PTXAS binary."));
#else
        stdCHECK(False, (ptxMsgParsingError,
                         "template initialization", "failed"));
#endif
    }

    for (i = 0; i != PTX_NUM_EXTENDED_TEMPLATES; ++i) {
        String instrTypes;
        String instrName;
        String argTypes;
        uInt32 tcode;

        ptr = consumeString(ptr, &instrTypes);
        ptr = consumeString(ptr, &instrName);
        ptr = consumeString(ptr, &argTypes);
        ptr = consumeUInt32(ptr, &tcode);

        addInstructionTemplate(parseData, instrTypes, stdCOPYSTRING(instrName), argTypes,
                               extFeatures[i], tcode, True);
    }

    stdFREE(extDescBuffer);
}

/*----------------------------- Instruction Table ----------------------------*/

/*
 * Function         : Initialize this module, by defining
 *                    all instruction templates
 * Parameters       :
 * Function Result  :
 */

void ptxDefineInstructionTemplates(ptxParseData parseData, cString extDescFileName, cString extDescAsString)
{
    ptxInstructionFeature features;
    parseData->stdTemplates    = mapNEW(String, 128);
    parseData->extTemplates    = mapNEW(String, 16);

    #include "ptxInstructionDefs.incl"

    stdASSERT(ptx_unknown_Instr == 0,
              ("geninstr.pl should ensure that unknown is always zero"));

    ptxDefineExtendedInstructions(parseData, extDescFileName, extDescAsString);
}
