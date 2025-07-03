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
 *  Module name              : ptxConstructors.h
 *
 *  Description              :
 *
 */

#ifndef ptxConstructors_INCLUDED
#define ptxConstructors_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "ptxIR.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Query ---------------------------------*/

/*
 * Function         : Check if texture instruction has depth compare arg
 * Function Result  : True iff texture instruction has depth compare arg
 */
Bool ptxTexHasDepthCompareArg(ptxParsingState parseState, ptxInstruction instr);

/*
* Function         : Get argument number of texture component in the PTX instruction
* Parameters       : parseState      :  Current PTX Parsing State
*                    ptxInstruction  :  The instruction in which we are querying the positional arguement information
*                    component       :  The component for which the positional arguement number is to be determined 
* Function Result  : argument number
*/
int ptxTexComponentArgNumber(ptxParsingState parseState, ptxInstruction instr, ptxTexComponentType component);

/*
 * Function         : Check if tex instruction has depth compare, if so return its argument number
 * Function Result  : argument number; returns -1 if depth compare is not present
 */
uInt ptxTexGetDepthCompareArgNumber(ptxInstruction instr);

/*
* Function         : Get minimum number of arguments in texture/surface instruction
* Parameters       : parseState      :  Current PTX Parsing State
*                  : ptxInstruction code
* Function Result  : minimum number of arguments based on independent/unified mode
*/

uInt ptxTexGetMinNumberOfArgs(ptxParsingState parseState, uInt code);

/*
 * Function         : Check if texture instruction has offset
 * Function Result  : True iff texture instruction has offset
 */
Bool ptxTexHasOffsetArg(ptxParsingState parseState, ptxInstruction instr);

Bool checkTargetOpts( ptxParsingState gblState, String opt );

/*
* Function         : Get argument number of vector type argument.
* Parameters       : ptxInstruction
* Function Result  : argument number
*/
uInt ptxGetVecArgNumber(ptxInstruction instr);

/*
* Function         : Get the vector component from a vector 
* Parameters       : vecExp : ptxExpression which represents the vector
*                    index  : Index of the element to be obtained from the vector 
* Function Result  : Element of the vector specified by the index 
*/
ptxExpression ptxGetElementExprFromVectorExpr(ptxExpression vecExp, uInt index);

/*
* Function         : Get argument number of co-ordinate vector in texture instruction.
* Parameters       : ptxInstruction
* Function Result  : argument number
*/
int ptxTexSurfGetCoordArgNumber(ptxParsingState parseState, uInt code);

/*
* Function         : Get argument number of symbol representing surface in surface instruction.
* Parameters       : ptxInstruction
* Function Result  : argument number
*/
int ptxSurfGetSurfaceSymbolArgNumber(ptxInstruction instr);

/*
* Function         : Get argument number of call target in the PTX call instruction
* Parameters       : ptxInstruction  :  The instruction in which we are querying the positional arguement information
* Function Result  : call target argument number
*/
int ptxGetCallTargetArgNumber(ptxInstruction instr);

/*
* Function         : Check whether the input arguement expression is either a int OR float kind of expression 
* Parameters       : ptxExpression 
* Function Result  : True,  if input argument expression is either int OR float kind of expression 
*                    False, Otherwise
*/
Bool ptxIsImmediateExpr(ptxExpression exp);

/*
* Function         : Check whether the input argument is sink ('_') Expression
* Parameters       : ptxExpression
* Function Result  : True,  if input argument expression is sink ('_') Expression
*                    False, Otherwise
*/

Bool isSinkExpression(ptxExpression expr);

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
                                       msgSourcePos_t sourcePos);

typedef enum {
    TEXTURE_SYMBOL = 0x01,
    SAMPLER_SYMBOL = 0x02,
    SURFACE_SYMBOL = 0x04
} ptxOpaqueEntities;

/* Function          : Check if the texture instruction uses indirect texture, sampler and surface 
 * Parameters        : instr  : Instruction in which access type of the opaque entities needs to be determined
 *                     entity : which among Textures, Sampler, Surfaces should be considered for indirect access checking  
 * Function Result   : True iff texture instruction has -
 *                                                    1. indirect texture when Texture entity is set
 *                                                    2. indirect sampler when Sampler entity is set
 *                                                    3. indirect surface when Surface entity is set
 *                                                    4. OR of any of the above
 */
Bool ptxTexSurfUsesIndirectAccess(ptxParsingState parseState, ptxInstruction instr, uInt entity);

/*
 * Function         : ptxNeedVecModifier
 *                  : more than one argument of type vector.
 * Parameters       : ptxInstruction
 * Function Result  : True if the input instruction required vector modifier for its operands.
 */
Bool ptxNeedVecModifierForVecArgs(ptxInstruction instr);

/*
 * Function         : This logically represents number of the components needed to pin-point the texel being referred to.
 * Function Result  : Number of entities needed to pin-point the texel being refered to in the instruction 
 */
uInt ptxGetTexSurfNumPosComponents( ptxModifier modifier );

/*
 * Function         : Find the dimension of base texture or surface
 *                    For array textures, dimension of texture refers to dimension of the texture element in the array
 * Function Result  : dimension of a texture/surface
 */
uInt ptxGetTextureDim( ptxModifier modifier );

/*
 * Function         : ptxGetSymEntFromExpr
 * Parameters       : ptxExpression
 * Function Result  : Returns symbol table entry for the given ptxExpression
 */
ptxSymbolTableEntry ptxGetSymEntFromExpr(ptxExpression expr);

/*
 * Function         : ptxGetAddressArgBase
 * Parameters       : ptxExpression
 * Function Result  : Returns the base expression incase of AddressOf or AddressRef expression
 *                    Else returns the same expression
 */
ptxExpression ptxGetAddressArgBase(ptxExpression arg);

/*
 * Function         : ptxGetAddressArgBaseType
 * Parameters       : ptxType
 * Function Result  : Returns the base expression type incase of AddressOf or AddressRef expression
 *                    Else type of the input expression
 */
ptxType ptxGetAddressArgBaseType(ptxExpression arg);

/*
 * Function         : ptxGetAddrOperandPos
 * Parameters       : instruction opcode
 * Function Result  : Returns position of address operand (counting from 0) in Instruction
 *                    Returns -1 for instructiosn which don't have memory operand
 */
int ptxGetAddrOperandPos(uInt code);

/*-------------------------------- Instruction--------------------------------*/

/*
 * Function         : Check if instruction is bar or barrier instruction
 * Function Result  : True iff bar or barrier instruction
 */
Bool ptxIsBarOrBarrierInstr(uInt code);

/*
 * Function         : Check Vector argument. This function is not meant to be used if there are
 *                  : more than one argument of type vector.
 * Parameters       : ptxInstruction
 * Function Result  : True if the desired operand is of type vector else False.
 */
Bool ptxIsArgVecType(ptxInstruction instr);

/*
 * Function         : Check if tex,tld4 instruction has an explicit sampler
 * Function Result  : True iff texture instruction has an explicit sampler
 */
Bool ptxTexHasSampler(ptxParsingState parseState);

#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
/*
 * Function         : Check if instruction is a TTU instruction
 * Function Result  : True iff TTU instruction
 */
Bool ptxIsTTUInstr(uInt code);

/*
 * Function         : Check if instruction is any TTU instruction other than ttucctl
 * Function Result  : True iff any TTU instruction other than ttucctl
 */
Bool ptxIsTTUInstrExceptTTUCCL(uInt code);

#endif

/*
 * Function         : Check if instruction is tex or tex.{base,level,grad} instruction
 * Function Result  : True iff texture instruction
 */
Bool ptxIsTexInstr(uInt code);

/*
 * Function         : Check if instruction is tex or tex.grad or .tex.grad.clamp instruction
 * Function Result  : True iff texture instruction
 */
Bool ptxIsTexGradInstr(uInt code);

/*
 * Function         : Check if instruction is a texture instruction (excludes txq)
 * Function Result  : True iff texture instruction
 */
Bool ptxIsTextureInstr(uInt code);

/*
 * Function         : Check if instruction is a texture query
 * Function Result  : True iff texture query instruction
 */
Bool ptxIsTxqInstr(uInt code);

/* Function          : Check if the texture/surface instruction uses Array modifier
 * Function Result   : True iff texture/surface instruction has A1D or A2D or ALWBE or A2DMS modifiers
 */
Bool ptxIsTexSurfInstrUsesArrayModifier( ptxModifier modifier );
    
/* Function          : Check if the texture instruction uses multi-sample modifier
 * Function Result   : True iff texture instruction has 2DMS or A2DMS modifiers
 */
Bool ptxIsTextureInstrUsesMultiSampleModifier( ptxModifier modifier );

/*
 * Function         : Check if instruction is a surface instruction introduced in PTX ISA 1.5 (excludes suq)
 * Function Result  : True iff surface instruction defined in PTX 1.5
 */
Bool isPTX15SurfaceInstr(uInt code);

/*
 * Function         : Check if instruction is a surface instruction (excludes suq)
 * Function Result  : True iff surface instruction
 */
Bool ptxIsSurfaceInstr(uInt code);

/*
 * Function         : Check if instruction is a scalar/simd2/simd4 video instruction
 * Function Result  : True iff video instruction
 */
Bool ptxIsVideoScalarInstr(uInt code);
Bool ptxIsVideoSIMD2Instr(uInt code);
Bool ptxIsVideoSIMD4Instr(uInt code);
Bool ptxIsVideoInstruction(uInt code);

/*
 * Function         : Return SIMD width of video instruction
 * Function Result  : SIMD width, 1/2/4.
 */
uInt ptxGetVideoInstrSIMDWidth(uInt code);

/*
 * Function         : Check if instruction supports f16 arithmetic
 * Function Result  : True iff instruction opcode supports f16 type
 */
Bool ptxIsF16ArithmeticInstr(uInt code);

/*
 * Function         : Check if instruction supports f16 comparison
 * Function Result  : True iff instruction opcode supports f16 type
 */
Bool ptxIsF16CompareInstr(uInt code);

/*
 * Function         : Check if instruction supports memory descriptor
 * Function Result  : True iff instruction opcode supports memory descriptor
 */
Bool ptxInstrSupportsMemDesc(uInt code);

/*---------------------------- Type Compatibility ----------------------------*/

/*
 * Function         : Maximize information on types, and return whether type components were compatible
 * Function Result  : True iff compatible
 */
Bool ptxMaximizeType( ptxTypeKind *lkind, uInt64 *lsize,
                      ptxTypeKind  rkind, uInt64  rsize, Bool  rConst );

/*
 * Function         : Decide if expression can be assigned to location of specified type.
 * Parameters       : lhs    (I) Type of location to assign to
 *                    rhs    (I) Type of expression to assign
 *                    rConst (I) True iff rhs expression is constant expression
 * Function Result  : True iff assignable
 */
Bool ptxAssignmentCompatible( ptxType l, ptxType r, Bool rConst);


/*------------------------ Type Constructor Functions ------------------------*/

/*
 * Function         : Create macro type representation.
 * Parameters         parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateMacroType( ptxParsingState parseState );

/*
 * Function         : Create label type representation.
 * Parameters         parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateLabelType(ptxParsingState parseState);

/*
 * Function         : Create predicate type representation.
 * Parameters         parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreatePredicateType(ptxParsingState parseState);

/*
 * Function         : Create condition-code type representation.
 * Parameters         parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateConditionCodeType(ptxParsingState parseState );

/*
 * Function         : Create bit type representation.
 * Parameters       : size    (I) Representation size in bytes (1,2 or 4)
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateBitType( uInt64 size , ptxParsingState parseState);

/*
 * Function         : Create float type representation.
 * Parameters       : size    (I) Representation size in bytes (2, 4 or 8)
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateFloatType( uInt64 size, ptxParsingState parseState);

/*
 * Function         : Create packed half float type representation.
 * Parameters       : size    (I) Representation size in bytes (4)
 *                                As of now only f16x2 is supported.
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreatePackedHalfFloatType( uInt64 size, ptxParsingState parseState);

/*
 * Function         : Create custom float type representation.
 * Parameters       : e   (I) Size of Exponent in bits (As of now only 8 bit is supported)
 *                  : m   (I) Size of mantissa in bits (As of now only 7/10 bit are supported)
 *                  : num (I) number of elements to be packed
 * Function Result  : Requested type
 */
ptxType ptxCreateLwstomFloatType(uInt e, uInt m, uInt num, ptxParsingState parseState);

/*
 * Function         : Create integer type representation.
 * Parameters       : size     (I) Representation size in bytes (1, 2, 4 or 8)
 *                    isSigned (I) integer type attribute 'signed'
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateIntType( uInt64 size, Bool isSigned, ptxParsingState parseState);

/*
 * Function         : Create opaque type representation.
 * Parameters       : name     (I) name of type
 *                    fields   (I) list of ptxSymbols, opaque fields
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateOpaqueType( String name, stdList_t fields , ptxParsingState parseState);

/*
 * Function         : Create pointer type representation.
 * Parameters       : storage  (I) storage kind of pointer target
 *                                 Precondition is that ptxAddressableStorage(storage) is True
 *                    base     (I) pointer target type
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreatePointerType( ptxStorageClass storage, ptxType base , ptxParsingState parseState);

/*
 * Function         : Create representation of array type with unspecified N.
 * Parameters       : base         (I) Array element type
 *                    logAlignment (I) Alignment of the array type
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateIncompleteArrayType(ptxType base, uInt logAlignment, ptxParsingState parseState);

/*
 * Function         : Create array type representation.
 * Parameters       : base     (I) Array element type
 *                    N        (I) Number of elements of array type
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateArrayType( uInt64 N, ptxType base , ptxParsingState parseState);

/*
 * Function         : Create vector type representation.
 * Parameters       : base     (I) Vector element element type,
 *                                 which must be basic type (Bit, Int or Float)
 *                    N        (I) Number of elements of array type
                      parseState(I) ptx Parser State
 * Function Result  : Requested type
 */
ptxType ptxCreateVectorType( uInt N, ptxType base , ptxParsingState parseState);


          /* ---------- . ---------- */

Bool ptxIsBasicTypeKind( ptxTypeKind kind );

/*
 * Function         : Test if type is a basic type (Bit, Int, Float, or Predicate).
 * Parameters       : type     (I) Type to inspect
 * Function Result  : True iff. type is a basic type
 */
Bool ptxIsBasicType( ptxType type );

/*
 * Function         : Test if enough of the type is known 
 *                    in order to allocate an instance from it.
 * Parameters       : type     (I) Type to inspect
 * Function Result  : True iff. type is a complete type
 */
Bool ptxIsCompleteType( ptxType type );

/*
 * Function         : Test if instances of type may be placed in registers.
 * Parameters       : type     (I) Type to inspect
 * Function Result  : True iff. type is 'registerable'
 */
Bool ptxIsRegisterType( ptxType type );

/*
 * Function         : Test if instances of type may be placed in parameter state space.
 * Parameters       : type     (I) Type to inspect
 *                    isEntry  (I) True iff. function is a CTA entry
 * Function Result  : True iff. type is 'parameterizable'
 */
Bool ptxIsParameterType( ptxType type, Bool isEntry );

/*
 * Function         : Test if storage class is addressable.
 * Parameters       : storage (I) Storage to inspect
 * Function Result  : True iff. storage is addressable
 */
Bool ptxIsAddressableStorage( ptxStorageClass storage );

/*
 * Function         : Test if storage class is initializable
 * Parameters       : storage (I) Storage to inspect
 * Function Result  : True iff. storage is initializable
 */
Bool ptxIsInitializableStorage( ptxStorageClass storage );

/*
 * Function         : Test if storage class represents 
 *                    some form of register space.
 * Parameters       : storage (I) Storage to inspect
 * Function Result  : True iff. storage is a register
 */
Bool ptxIsRegisterStorage( ptxStorageClass storage );

/*
 * Function         : Create basic type from type info.
 * Parameters       : kind     (I) Kind of basic type to create
 *                    size     (I) Size of basic type to create
                      parseState(I) ptx Parser State
 * Function Result  : Requested basic type
 */
ptxType ptxCreateBasicType( ptxTypeKind kind, uInt64 size, ptxParsingState parseState);

/* 
 * Function         : Get base type 
 * Parameters       : type  (I) Type 
 * Function Result  : For aggregate types (array, vector) return base type,
 *                    otherwise return same type
 */
ptxType ptxGetBaseType( ptxType type );

/*
 * Function         : Get type logAlignment
 * Parameters       : type  (I) Type
 * Function Result  : returns logAlignment of input type
 */
uInt ptxGetTypeLogAlignment(ptxType type);

/*
 * Function         : Get type size
 * Parameters       : type  (I) Type
 * Function Result  : returns size of input type in bits
 */
uInt64 ptxGetTypeSizeInBits(ptxType type);

/*
 * Function         : Get type size in bytes
 * Parameters       : type  (I) Type
 * Function Result  : returns size of input type in bytes
 */
uInt64 ptxGetTypeSizeInBytes(ptxType type);

/*--------------------- Expression Constructor Functions ---------------------*/

/*
 * Function         : Create binary expression
 * Parameters       : type       (I) result type
 *                    op         (I) binary operator
 *                    left,right (I) binary arguments
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateBinaryExpr( ptxType type, ptxOperator op, ptxExpression left, ptxExpression right );

/*
 * Function         : Create unary expression
 * Parameters       : type       (I) result type
 *                    op         (I)  unary operator
 *                    arg        (I)  operation argument
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateUnaryExpr( ptxType type, ptxOperator op, ptxExpression arg );

/*
 * Function         : Create integer constant expression
 * Parameters       : i          (I) Integer constant
 *                    isSigned   (I) True iff. integer is signed
                      parseState(I) ptx Parser State
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateIntConstantExpr( Int64 i, Bool isSigned, ptxParsingState parseState);

/*
 * Function         : Create single-precision floating-point constant expression
 * Parameters       : f          (I) Float constant
                      parseState(I) ptx Parser State
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateF32FloatConstantExpr( Float f, ptxParsingState parseState);

/*
 * Function         : Create double-precision floating-point constant expression
 * Parameters       : d          (I) Double constant
                      parseState(I) ptx Parser State
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateF64FloatConstantExpr( Double d, ptxParsingState parseState);

/*
 * Function         : Get single-precision value of FloatConstant expression, casting from double if necessary
 * Parameters       : e          (I) Float Constant Expression
 * Function Result  : Requested value
 */
Float ptxGetF32FloatConstantExpr(ptxExpression e);

/*
 * Function         : Colwert 32 bit floating point number to 32 bit bitstream.
 * Parameters       : e          (I) Float constant expression
 * Function result  : *uVal holds the colwerted value
 */
uInt32 ptxColwert32FloatToUnsignedIntExpr(ptxExpression e);

/*
 * Function         : Colwert 64 bit Double number to 64 bit bitstream.
 * Parameters       : e          (I) Double constant expression
 * Function result  : *ulVal holds the colwerted value
 */
uInt64 ptxColwert64FloatToUnsignedLongExpr(ptxExpression e);

/*
 * Function         : Get double-precision value of FloatConstant expression; no up-casting from f32 allowed
 * Parameters       : e          (I) Float Constant Expression
 * Function Result  : Requested value
 */
Double ptxGetF64FloatConstantExpr(ptxExpression e);

/*
 * Function         : Create symbol reference expression
 * Parameters       : symbol     (I) symbol to reference
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateSymbolExpr( ptxSymbolTableEntry symbol );

/*
 * Function         : Create array- or vector indexing expression
 * Parameters       : arg        (I) array- or vector valued selectee
 *                    index      (I) index
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateArrayIndexExpr( ptxExpression arg, ptxExpression index );
 
/*
 * Function         : Create array- or vector indexing expression
 * Parameters       : arg        (I) array- or vector valued selectee
 *                    dimension  (I) number of vector selectors, up to maximum of 4
 *                    selector   (I) selector string, e.g  "xxy", with length indicated by dimension
                      parseState(I) ptx Parser State
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateVectorSelectExpr( ptxExpression arg, uInt dimension, ptxVectorSelector *selector, ptxParsingState parseState);
 
/*
 * Function         : Create video sub-word select expression
 * Parameters       : arg        (I) register symbol
 *                    N          (I) number of video selectors, up to maximum of 4
 *                    selector   (I) video selector
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateVideoSelectExpr( ptxExpression arg, uInt N, ptxVideoSelector *selector );

/*
 * Function         : Create video sub-word select expression
 * Parameters       : arg        (I) register symbol
 *                    N          (I) number of video selectors, up to maximum of 4
 *                    selector   (I) video selector
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateByteSelectExpr( ptxExpression arg, uInt N, ptxByteSelector *selector );

/*
 * Function         : Create vector expression from element list
 * Parameters       : elements (I) list of element
 *                    type     (I) type of vector expression
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateVectorExpr( stdList_t elements, ptxType type );

/*
 * Function         : Create address take expression
 * Parameters       : lhs        (I) access path to memory location
                      parseState(I) ptx Parser State
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateAddressOfExpr( ptxExpression lhs, ptxParsingState parseState);
 
/*
 * Function         : Create memory address reference expression
 * Parameters       : arg        (I) address expression
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateAddressRefExpr( ptxExpression arg );

/*
 * Function         : Create predicate expression
 * Parameters       : neg        (I) True iff. predicate should be negated
 *                  : arg        (I) predicate
                      parseState(I) ptx Parser State
 * Function Result  : Requested expression
 */
ptxExpression ptxCreatePredicateExpr( Bool neg, ptxExpression pred, ptxParsingState parseState);

/*
 * Function         : Create label reference expression
 * Parameters       : name       (I) name of referenced label
 *                    symbtab    (I) context of reference
 *                    sourcePos  (I) source location of reference
                      parseState(I) ptx Parser State
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateLabelReferenceExpr( String name, ptxSymbolTable symbtab, msgSourcePos_t sourcePos, ptxParsingState parseState);

/*
 * Function         : Create parameter list expression from element list
 * Parameters       : elements (I) list of element
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateParamListExpr( stdList_t elements, ptxParsingState parseState);

/*
 * Function         : Create sink expression
 * Parameters       : None
 * Function Result  : Requested expression
 */
ptxExpression ptxCreateSinkExpr( void );

/*
 * Function         : Change Int Constant Expression Size
 * Parameters       : expr   (I) Integer Constant Expression
 *                    size   (I) new size of resulting expression
                      parseState(I) ptx Parser State
 * Function Result  : Requested expression
 */
ptxExpression ptxChangeIntConstantExpressionSize(ptxExpression expr, uInt size,  ptxParsingState parseState);

          /* ---------- . ---------- */

/*
 * Function         : Obtain storage class related to
 *                    specified access path.
 * Parameters       : lhs        (I) access path to inspect
 * Function Result  : Requested storage class
 */
ptxStorageClass ptxGetStorage( ptxExpression lhs );

/*
 * Function         : Obtain storage kind related to
 *                    specified access path.
 * Parameters       : lhs        (I) access path to inspect
 * Function Result  : Requested storage kind
 */
ptxStorageKind ptxGetStorageKind( ptxExpression lhs );

/*-------------------------- Symbol Table Functions --------------------------*/

#define ptxNOSTORAGECLASS ptxCreateStorageClass(ptxUNSPECIFIEDStorage,-1)

/*
 * Function         : Create storage class from components
 * Parameters       : kind       (I) kind of storage class
 *                    bank       (I) if applicable, memory bank, or -1 otherwise
 * Function Result  : requested new storage class representation
 */
ptxStorageClass ptxCreateStorageClass( ptxStorageKind kind, Int bank );

/*
 * Function         : Create symbol with source information
 * Parameters       : parseState   (I) ptx Parsing State
 *                    type         (I) type of symbol
 *                    name         (I) name of symbol
 *                    logAlignment (I) log2 of required alignment of variable
 *                    sourcePos    (I) location in source file where symbol was declared
 * Function Result  : requested new symbol representation
 */
ptxSymbol ptxCreateSymbol(ptxParsingState parseState, ptxType type, String name, uInt logAlignment, uInt64 attributeFlags, msgSourcePos_t sourcePos );

/*
 * Function         : Create symbol table
 * Parameters       : parent       (I) parent symbol table
 * Function Result  : new symbol table
 */
ptxSymbolTable ptxCreateSymbolTable(ptxSymbolTable parent );

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
                          ptxStorageClass storage, ptxInitializer initialValue, uInt range);

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
 *                    scratchRegs(I) List of scratch registers
 *                    retAddrAllocno(I) register specifying return address
 * Function Result  : True  iff. symbol could be added 
 *                    False iff. symbol name clashes with current contents of symbol table
 */
Bool ptxAddFunctionSymbol(ptxSymbolTable symbtab, ptxSymbol symbol, Bool isEntry,
                          Bool isInlineFunc, ptxDeclarationScope scope,
                          ptxSymbolTable body, stdList_t rparams, stdList_t fparams,
                          Bool hasAllocatedParams, Bool hasNoReturn, Bool isUnique,
                          uInt retAddrAllocno, stdList_t scratchRegs);

/*
 * Function         : Add label definition to symbol table, pointing to its current instruction position
 * Parameters       : symbtab    (I) symbol table to add to
 *                    symbol     (I) symbol to add
 * Function Result  : True  iff. symbol could be added 
 *                    False iff. symbol name clashes with current contents of symbol table
 */
Bool ptxAddLabelSymbol(ptxSymbolTable symbtab, ptxSymbol symbol);

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
Bool ptxAddMacroSymbol(ptxSymbolTable symbtab, ptxSymbol symbol, stdList_t formals, String body, msgSourcePos_t sourcePos);

/*
 * Function         : Add opaque definition to symbol table
 * Parameters       : symbtab    (I) symbol table to add to
 *                    symbol     (I) type symbol to add
 * Function Result  : True  iff. type could be added 
 *                    False iff. type name clashes with current contents of symbol table
 */
Bool ptxAddOpaque (ptxSymbolTable symbtab, ptxSymbol symbol );

/*
 * Function         : Lookup symbol in symbol table
 * Parameters       : symbtab         (I) symbol table to inspect
 *                    name            (I) name of symbol to lookup
 *                    inspectParents  (I) ancestor symbol tables will be inspected
 *                                         iff. inspectParents equals True.
 *                    parseState      (I) parsing state which stores all parsing related
 *                                        information
 * Function Result  : requested symbol, or NULL when not found
 */
ptxSymbolTableEntry ptxLookupSymbol(ptxSymbolTable symbtab, String name, Bool inspectParent, ptxParsingState parseState);

/*
 * Function         : Store debug information in variable's symbol table entry
 * Parameters       : name            (I) name of symbol to lookup
 *                    symbtab         (I) symbol table to inspect
 *                    scope           (I) scope of symbol
 *                    storage         (I) storage where symbol must be allocated,
 *                    state           (I) parsing state
 * Function Result  :
 */
void ptxSetVariableDebugInfo(String name, ptxSymbolTable symbtab, ptxDeclarationScope scope, ptxStorageClass storage, ptxParsingState state);

/*
 * Function         : Check whether string has the form of a parameterized variable name
 * Parameters       : name            (I) name of variable
 *                    suffix          (O) numeric suffix, returned as an integer
 *                    suffixStart     (O) starting position in 'name' of this numeric suffix
 * Function Result  : True if string has a numeric suffix
 */
Bool ptxIsParameterizedVariableName( String name, uInt *suffix, uInt *suffixStart );

/*
 * Function         : colwert the name of a parameterized variable into canonical form
 * Parameters       : name            (IO) name of variable. It will be modified inplace.
 *                    suffixStart     (I)  numeric suffix, returned as an integer
 *                    save            (O)  saved information to restore the name
 * Function Result  :
 */
void ptxGetParameterizedVariableName( String name, uInt suffixStart, ptxParamVarSave save );

/*
 * Function         : colwert the name of a parameterized variable from canonical form back to original name
 * Parameters       : name            (IO) name of variable. It will be modified inplace.
 *                    suffixStart     (I)  numeric suffix, returned as an integer
 *                    save            (O)  saved information returned by previous ptxGetParameterizedVariableName call
 * Function Result  :
 */
void ptxRestoreParameterizedVariableName( String name, uInt suffixStart, ptxParamVarSave save );

/*
 * Function         : Lookup symbol in symbol table, and lazily create parameterized variables
 * Parameters       : symbtab         (I) symbol table to inspect
 *                    name            (I) name of symbol to lookup
 *                    inspectParents  (I) ancestor symbol tables will be inspected
 *                                         iff. inspectParents equals True.
 *                    state           (I) parsing state
 * Function Result  : requested symbol, or NULL when not found
 */
ptxSymbolTableEntry ptxLookupSymbolLazyCreate(ptxSymbolTable symbtab, String name, Bool inspectParent, ptxParsingState state);

/*
 * Function         : Lookup opaque struct in symbol table
 * Parameters       : symbtab         (I) symbol table to inspect
 *                    name            (I) name of symbol to lookup
 *                    inspectParents  (I) ancestor symbol tables will be inspected
 *                                         iff. inspectParents equals True.
 * Function Result  : requested symbol, or NULL when not found
 */
ptxSymbol ptxLookupOpaque (ptxSymbolTable symbtab, String name, Bool inspectParent );

/*
 * Function         : Add statement at the end of statement list of specified symbol table
 * Parameters       : symbtab    (I) symbol table to add to
 *                    statement  (I) statement to add
 * Function Result  : 
 */
void ptxAddStatement(ptxSymbolTable symbtab, ptxStatement statement );

/*---------------------- Statement Constructor Functions ---------------------*/
/*
 * Function         : Create instruction statement representation from components
 * Parameters       : instruction (I) instruction to promote to statement
 * Function Result  : requested new statement representation
 */
ptxStatement ptxCreateInstructionStatement( ptxInstruction instruction );

/*
 * Function         : Create pragma statement representation from components
 * Parameters       : pragmas (I) list of pragma strings to promote to statement
 * Function Result  : requested new statement representation
 */
ptxStatement ptxCreatePragmaStatement( stdList_t pragmas );

/*
 * Function         : Create metadata value of integer type
 * Parameters       : integer
 * Function Result  : requested metadata value
 */
ptxMetaDataValue ptxCreateMetadataValueInt( uInt val );

/*
 * Function         : Create metadata value of string type
 * Parameters       : string
 * Function Result  : requested metadata value
 */
ptxMetaDataValue ptxCreateMetadataValueString( String str );

/*
 * Function         : Create metadata value of index type
 * Parameters       : metadata index
 * Function Result  : requested metadata value
 */
ptxMetaDataValue ptxCreateMetadataValueIndex( uInt index );

#if     defined(__cplusplus)
}
#endif 

#endif /* ptxConstructors_INCLUDED */
