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
 *  Module name              : ptxIR.h
 *
 *  Description              :
 *
 */

#ifndef ptxIR_INCLUDED 
#define ptxIR_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdLocal.h"
#include "stdString.h"
#include "stdList.h"
#include "stdSet.h"
#include "stdMap.h"
#include "stdRangeMap.h"
#include "stdMessages.h"
#include "stdMemSpace.h"
#include "stdVector.h"
#include "stdLocal.h"
#include "gpuInfo.h"
#include "ptxObfuscatedIRdefs.h"
#include "copi_atom_interface.h"
#include "stdBitSet.h"
#include "stdObfuscate.h"
#include "DebugInfo.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef struct ptxSymbolRec                    *ptxSymbol;
typedef struct ptxTypeRec                      *ptxType;
typedef struct ptxExpressionRec                *ptxExpression;
typedef struct ptxSymbolTableEntryAuxRec       *ptxSymbolTableEntryAux;
typedef struct ptxFunctionPrototypeAttrInfoRec *ptxFunctionPrototypeAttrInfo;
typedef struct ptxSymbolTableEntryRec          *ptxSymbolTableEntry;
typedef struct ptxSymbolTableRec               *ptxSymbolTable;
typedef struct ptxInstructionTemplateRec       *ptxInstructionTemplate;
typedef struct ptxInstructionRec               *ptxInstruction;
typedef struct ptxStatementRec                 *ptxStatement;
typedef struct ptxParsingStateRec              *ptxParsingState;
typedef struct ptxParseInfoRec                 *ptxParseData;
typedef struct ptxInitializerRec               *ptxInitializer;
typedef struct ptxCodeLocationRec              *ptxCodeLocation;
typedef struct ptxSymLocInfoRec                *ptxSymLocInfo;
typedef struct ptxParamVarSaveRec              *ptxParamVarSave;
typedef struct ptxDwarfSectionRec              *ptxDwarfSection;
typedef struct ptxDwarfLineRec                 *ptxDwarfLine;
typedef struct ptxMetaDataSectionRec           *ptxMetaDataSection;
typedef struct ptxMetaDataNodesRec             *ptxMetaDataNode;
typedef struct ptxMetaDataValueRec             *ptxMetaDataValue;
typedef struct ptxDwarfLiveRangeMapListNodeRec *ptxDwarfLiveRangeMapListNode;

/*----------------------------------- Types ----------------------------------*/

typedef enum {
     ptxLocalScope,
     ptxStaticScope,
     ptxGlobalScope,
     ptxExternalScope,
     ptxWeakScope,
     ptxCommonScope
} ptxDeclarationScope;

typedef enum {
    NoOperationClass        = 0,
    AtomicOperationClass    = 1,
    ArithOperationClass     = 2,
    BoolOperationClass      = 4
} ptxOperatorClass;

#define AtomicOperations          (ArithOperationClass \
                                   | AtomicOperationClass \
                                   | BoolOperationClass)

#include "lwPtxStorage.h" // definition of ptxStorageKind

typedef struct {
    ptxStorageKind  kind;
    Byte            bank;
} ptxStorageClass;

typedef struct ptxDwarfLiveRangeMapListNodeRec
{
    Bool isFirstDefinition;         // To which list of OCG should PTXAS add this symbol in ptxOptimize? 
    String ptxRegisterName;         // String representation of ptxRegister encoded in locList/locExpr
    Bool isLocList;                 // Is the symbol coming from locList? 
                                    // This field is added just for debugging purpose
} ptxDwarfLiveRangeMapListNodeRec;

typedef enum {
    PTX_TYPES_TABLE(GET_ENUM),
    ptxLabelType,
    ptxMacroType,
    ptxConditionCodeType,
    ptxOpaqueType,
    ptxIncompleteArrayType,
    ptxVectorType,
    ptxParamListType,
    ptxArrayType
} ptxTypeKind;

typedef struct ptxTypeRec {
    ptxTypeKind     kind;
    union {
        struct {
            String            name;
            stdList_t         fields;     // ptxSymbol
            uInt64            sizeInBits; // Size of type represented in bits
            // Explicitly store logAlignment for opaque type as it cannot be
            // derived from its size.
            uInt              logAlignment;
        } Opaque;
        
        struct {
             uInt64           N;
             ptxType          base;
        } Array;
        
        struct {
             uInt             N;
             ptxType          base;
        } Vector;
        
        struct {
             ptxType          base;
             // Explicitly store logAlignment for incompleteArray type as it changes
             // from default value when incompleteArray is used as function param.
             // See function createArrayType for details
             uInt             logAlignment;
        } IncompleteArray;
    } cases;
} ptxTypeRec;

// OPTIX_HAND_EDIT
typedef void(*GenericCallback)();

/*-------------------------------- Expressions -------------------------------*/

typedef enum {
    ptxPtrNone,
    ptxPtrGeneric,
    ptxPtrConst,
    ptxPtrGlobal,
    ptxPtrLocal,
    ptxPtrShared,
    ptxPtrTexref,
    ptxPtrSamplerref,
    ptxPtrSurfref
} ptxPointerAttr;

/* Bit mask enum for different attributes */
typedef enum {
    ptxAttributeNone = 0,
    ptxAttributeManaged = 1
} ptxSymbolAttribute;

typedef enum {
    PTX_COMPARE_TABLE(GET_ENUM)
} ptxComparison;

typedef enum {
    PTX_OPERATOR_TABLE(GET_ENUM)
} ptxOperator;

typedef enum {
    ptxBinaryExpression,
    ptxUnaryExpression,
    ptxIntConstantExpression,
    ptxFloatConstantExpression,
    ptxSymbolExpression,
    ptxArrayIndexExpression,
    ptxVectorSelectExpression,
    ptxVideoSelectExpression,
    ptxByteSelectExpression,
    ptxPredicateExpression,
    ptxStructExpression,
    ptxAddressOfExpression,
    ptxAddressRefExpression,
    ptxLabelReferenceExpression,
    ptxVectorExpression,
    ptxParamListExpression,
    ptxSinkExpression
} ptxExpressionKind;

typedef enum {
    ptxComp_X,
    ptxComp_Y,
    ptxComp_Z,
    ptxComp_W
} ptxVectorSelector;

typedef enum {
    PTX_VIDEOSELECTOR_TABLE(GET_ENUM)
} ptxVideoSelector;

typedef enum {
    PTX_BYTESELECTOR_TABLE(GET_ENUM)
} ptxByteSelector;

typedef enum {    
    ptxTexWrap = 1,
    ptxTexMirror,
    ptxTexClampOGL,
    ptxTexClampEdge,
    ptxTexClampBorder,
    ptxTexNearest,
    ptxTexLinear
} ptxTexConstants;

typedef enum {
    TEX_COMPONENT_TYPE_TEXSAMP ,
    TEX_COMPONENT_TYPE_COORDS ,
    TEX_COMPONENT_TYPE_DEPTH ,
    TEX_COMPONENT_TYPE_LODBIAS ,
    TEX_COMPONENT_TYPE_MULTISAMPLE ,
    TEX_COMPONENT_TYPE_ARRAYINDEX ,
    TEX_COMPONENT_TYPE_AOFFSET ,
    TEX_COMPONENT_TYPE_PTP1 ,
    TEX_COMPONENT_TYPE_PTP2 ,
    TEX_COMPONENT_TYPE_SI ,
    TEX_COMPONENT_TYPE_TI ,
    TEX_COMPONENT_TYPE_DSDX ,
    TEX_COMPONENT_TYPE_DTDX ,
    TEX_COMPONENT_TYPE_DRDX ,
    TEX_COMPONENT_TYPE_DSDY ,
    TEX_COMPONENT_TYPE_DTDY ,
    TEX_COMPONENT_TYPE_DRDY ,
    TEX_COMPONENT_TYPE_LODCLAMP,
    TEX_COMPONENT_TYPE_GRANULARITY
} ptxTexComponentType;

typedef struct ptxExpressionRec {
    ptxExpressionKind               kind : 6;
    Bool                            isConstant : 1;
    Bool                            isLhs : 1;
    Bool                            neg : 1; // used with kind == ptxPredicateExpression
    ptxType                         type;

    union {
        struct {
            ptxOperator             op;
            ptxExpression           left;
            ptxExpression           right;
        } *Binary;

        struct {
            ptxOperator             op;
            ptxExpression           arg;
        } *Unary;

        struct {
            Int64                   i;
        } IntConstant;

        struct {
            Float                   flt;
        } FloatConstant;

        struct {
            Double                  dbl;
        } DoubleConstant;

        struct {
            ptxSymbolTableEntry     symbol;
        } Symbol;

        struct {
            ptxExpression           arg;
            ptxExpression           index;
        } *ArrayIndex;

        struct {
            ptxExpression           arg;
            uInt                    dimension;
            ptxVectorSelector       selector[4];
        } *VectorSelect;
        // TODO:  represent video instructions using ByteSelect
        struct {
            ptxExpression           arg;
            uInt                    N;
            ptxVideoSelector        selector[4];
        } *VideoSelect;

        // Used to represent instruction operands with byte specifier
        struct {
            ptxExpression           arg;
            uInt                    N;
            ptxByteSelector         selector[4];
        } *ByteSelect;

        struct {
            ptxExpression           lhs;
        } AddressOf;

        struct {
            ptxExpression           arg;
        } AddressRef;

        struct {
            // moved out of the struct for better packing
            // Bool                    neg; 
            ptxExpression           arg;
        } Predicate;

        struct {
            stdList_t               fields;
        } Struct;
        
        struct L {
            String                  name;
            msgSourcePos_t          sourcePos;  // Referring source location
        } *LabelReference;
        
        struct {
            stdList_t               elements;
            Bool                    reverseMod;
        } Vector;

        struct {
            stdList_t               elements;
        } ParamList;
    } cases;

} ptxExpressionRec;


/*------------------------------- Initializers -------------------------------*/

typedef enum {
    ptxExpressionInitializer,
    ptxNamedFieldInitializer,
    ptxStructuredInitializer
} ptxInitializerKind;

struct ptxInitializerRec {
    ptxInitializerKind  kind;
    msgSourcePos_t      sourcePos;
    
    union {
        struct {
            ptxExpression          expr;
            uInt64                 mask;
            Bool                   isGeneric;
        } Expression;
        
        struct {
            String                 name;
            ptxExpression          expr;
        } NamedField;
        
        struct {
            stdList_t              list;
        } Structured;
    } cases;
};

/*------------------------------- Symbol Table -------------------------------*/

typedef struct ptxSymbolRec {
    String                         name;          // Stores mangled name after symbol declaration is parsed
    String                         unMangledName; // Stores unmangled name after symbol declaration is parsed
    ptxType                        type;
    uInt                           index;
    uInt                           logAlignment; // alignment constraint in addition to natural type alignment
    uInt64                         attributeFlags; // mask holding state of each attribute
    msgSourcePos_t                 sourcePos;
} ptxSymbolRec;

typedef enum {
    ptxLabelSymbol,
    ptxBranchTargetSymbol,
    ptxCallTargetSymbol,
    ptxCallPrototypeSymbol,
    ptxVariableSymbol,
    ptxFunctionSymbol,
    ptxMarkerSymbol,
    ptxMacroSymbol
} ptxSymbolKind;

#define UNSPECIFIED_ABI_PARAM_REGS -1
#define ABI_PARAM_REGS_ALL -2
#define UNSPECIFIED_RET_ADDR_BEFORE_PARAMS -1
#define UNSPECIFIED_ABI_REG -1
#define UNSPECIFIED_ABI_REGS NULL

typedef struct ptxFunctionPrototypeAttrInfoRec {
    Bool                           hasAllocatedParams;      // Indicates if all parameters of device function are pre-allocated using .allocno

    int                            retAddrAllocno;          // In case of device function: register specifying return address

    stdList_t                      scratchRegs;             // List of scratch registers specified via .scratch

    int                            numAbiParamRegs;         // In case of device function: number of registers to be used for parameter passing
                                                            // UNSPECIFIED_ABI_PARAM_REGS is initial value
    int                            firstParamReg;           // In case of device function: minimum physical register where parameters are allocated
                                                            // UNSPECIFIED_ABI_REG is initial value
    int                            retAddrBeforeParams;     // In case of device function: allocate return addresses before parameters
                                                            // UNSPECIFIED_RET_ADDR_BEFORE_PARAMS is initial value

    int                            retAddrUReg;             // In case of device function: physical uniform register where absolute return address is allocated
                                                            // UNSPECIFIED_ABI_REG is initial value
    int                            retAddrReg;              // In case of device function: physical register where absolute return address is allocated
                                                            // UNSPECIFIED_ABI_REG is initial value
    int                            relRetAddrReg;           // In case of device function: physical register where relative return address is allocated
                                                            // UNSPECIFIED_ABI_REG is initial value
    stdList_t                      scratchRRegs;            // In case of device function: list of physical R registers which are marked as scratch
    stdList_t                      scratchBRegs;            // In case of device function: list of physical barrier registers which are marked as scratch
                                                            // UNSPECIFIED_ABI_REGS is initial value of scratch{R,B}Regs respectively

    Bool                           hasNoReturn;             // Indicates if .noreturn specfied, applicable only on device functions

    Bool                           isCoroutine;             // Indicates if the current function is a coroutine (i.e., calls __lw_ptx_builtin_suspend)

    stdList_t                      rparams;                 // In case of function symbol: function's return parameters (list of ptxVariableInfo structs)
    stdList_t                      fparams;                 // In case of function symbol: function's formal parameters (list of ptxVariableInfo structs)
                                                            // OR in case of macro symbol: list of formal parameter names
} ptxFunctionPrototypeAttrInfoRec;


typedef struct ptxSymbolTableEntryAuxRec {
    Bool                           isEntry;                 // In case of function symbol: if function is a CTA entry
    Bool                           isInlineFunc;            // In case of function symbol: if function call gets inlined with the function body
    Bool                           isRelwrsive;             // In case of function symbol: if function is part of relwrsive call chain
    Bool                           usesWMMAInstrs;          // Indicates use of wmma instructions
    Bool                           isUnique;                // Indicates there will be single copy of function in noCloning, SC and EWP compilation modes
    uInt                           funcIndex;               // For defined function: funcIndex = unique id (starting from '0')
                                                            // For non-defined function: funcIndex = ~0 (Invalid)
    uInt                           maxnreg;                 // In case of entry symbol: max registers per thread across entire call tree
    uInt                           localMaxNReg;            // In case of entry symbol: max registers per thread for the current compilation unit
    uInt                           maxntid[3];              // In case of entry symbol: max extent of each CTA dimension
    uInt                           minnctapersm;            // In case of entry symbol: min CTAs per SM
    uInt                           reqntid[3];              // In case of entry symbol: required extent of each CTA dimension

    ptxSymbolTableEntry            aliasee;                 // In case of function symbol: Real function to which current function is aliased to

    listXList                      (pragmas);               // list of kernel function-scoped pragma strings

    ptxSymbolTable                 body;                    // In case of function symbol: function's definition

    // Sequence number in a list. For function, it is the position in
    // the list of functions in the symbol table. For a label, it is
    // the position of the statement that follows it, in the list of
    // statements in the symbol table.
    uInt                           listIndex;

    String                         mbody;                   // In case of macro symbol : macro body text
    msgSourcePos_t                 mbodyPos;                // In case of macro symbol : source position of start of macro body text
                                                            // Also, in case of function symbol: source position of '}'
    msgSourcePos_t                 startPos;                // In case of macro symbol : source position of start of macro body text
                                                            // Also, in case of function symbol: source position of '{'
    ptxSymbolTableEntry           *parameterizedVars;       // array of symbol table entries for parameterized variables

    ptxFunctionPrototypeAttrInfo   funcProtoAttrInfo;       // Various function attributes related to the creation of function prototype.
} ptxSymbolTableEntryAuxRec;

typedef struct ptxSymbolTableEntryRec {
    ptxSymbolKind                  kind;
    ptxSymbol                      symbol;
    ptxDeclarationScope            scope;
    ptxInitializer                 initialValue;  // variable initializers, also used by .branchtargets and .calltargets label
    ptxStorageClass                storage;
    uInt64                         virtualAddress;
    ptxCodeLocation                location;
    ptxSymbolTable                 symbtab;       // symbol table in which this entry resides

    uInt                           range;         // range of lazily declared parameterized variables
    ptxSymbolTableEntryAux         aux;
} ptxSymbolTableEntryRec;


typedef struct ptxSymbolTableRec {
    stdMap_t                      opaques;        // String --> ptxType
    stdMap_t                      symbolIndexMap; // Symbol's Index --> ptxSymbolTableEntry

    listXList                    (LabelSeq);      // ptxSymbolTableEntry; in order of symbol insertion
    listXList                    (LabelRefSeq);   // ptxSymbolTableEntry; in order of symbol insertion
    listXList                    (VariableSeq);   // ptxSymbolTableEntry; in order of symbol insertion
    listXList                    (ConstrainedExternalVarSeq);    // externally scoped symbols which
                                                                 // don't have Sreg or Reg storage space
    listXList                    (symbolsWithAttributeFlags);    // Symbols which have attribute flags
    listXList                    (InitializableVarSeq);          // Symbols for initialization code in ptxAs 
    listXList                    (FunctionSeq);   // ptxSymbolTableEntry; in order of symbol insertion
    listXList                    (MarkerSeq);     // ptxSymbolTableEntry; in order of symbol insertion
    listXList                    (MacroSeq);      // ptxSymbolTableEntry; in order of symbol insertion

    // these variables will be promoted to VariableSeq of object symbol table, so list should not be accessed after parsing
    listXList                    (VariablesToPromoteSeq);

    ptxSymbolTable                parent;     
    
    listXList                    (statements);    // ptxStatement
    uInt                          numStmts;       // number of statements
    
    Pointer                       data;
} ptxSymbolTableRec;


// used to collect information needed to define a variable or parameter
typedef struct {
    ptxSymbol       symbol;
    ptxInitializer  initializer;
    ptxStorageClass storage;
    ptxPointerAttr  ptr_attr;          // optional kernel parameter attribute, specifies state space being pointed to by pointer parameter
    uInt            ptr_logAlignment;  // optional kernel parameter attribute, specifies log2(alignment) of object being pointed to by pointer parameter
    int             allocno;           // optional device function parameter attribute, specifies hard register to use for function parameter
    uInt            range;             // range of variables lazily declared using shorthand notation
} *ptxVariableInfo;

/*
 * Function         : Creates a dummy symbol table entry but having the specified ptxSymbolKind.
 * Parameters       : parseState :  Parsing state
 *                    instr      :  ptx instruction using the given symbol
 *                    ptxSymEnt  :  The original symbol table entry whose dummy entry of specified kind needs to be created.
 *                  : kind       :  The kind of the new symbol table entry to create.
 * Function Result  : New symbol table entry having the specified kind.
 */

ptxSymbolTableEntry ptxCreateDummySymbolTableEntryOfKind(ptxParsingState parseState,
                                                         ptxInstruction instr,
                                                         ptxSymbolTableEntry ptxSymEnt,
                                                         ptxSymbolKind kind);
/*
 * Function         : Set the globalState
 */
void ptxSetGlobalState(ptxParsingState state);

/*
 * Function         : Get .version String
 */
String getPtxVersionString();

/*
 * Function         : Get .target String
 */
String getTargetArchString();

/*
 * Function         : Set the symbol in a symbol expression
 */
void ptxInitSymbolExpr(ptxExpression expr, ptxSymbolTableEntry symbol);

/*
 * Function         : Set a symbol attribute
 * Parameters       : symbol       (I) ptxSymbol
 *                    attribute    (I) ptxSymbolAttribute
 */
void ptxSetSymbolAttribute( ptxSymbol symbol, ptxSymbolAttribute attribute );

/*
 * Function         : Checks if a symbol attribute is set
 * Parameters       : symbol       (I) ptxSymbol
 *                    attribute    (I) ptxSymbolAttribute
 * Function Result  : True iff. symbol has attribute set
 *                    False iff. symbol doesn't have attribute set
 */
Bool ptxCheckSymbolAttribute( ptxSymbol symbol, ptxSymbolAttribute attribute );

/*
 * Function         : Determine if the symbol is user-defined
 * Parameters       : symbol       (I) symbol to check
 * Function Result  : True  iff. symbol is user symbol
 *                    False iff. symbol is non user symbol
 */
Bool ptxIsUserSymbol( ptxSymbol symbol );

/*
 * Function         : Determine if the symbol can be used as an address argument
 * Parameters       : symbol       (I) symbol to check
 * Function Result  : True  iff. symbol has an address
 */
Bool ptxIsAddressableSymbol( ptxSymbolTableEntry symbol );

/*
 * Function         : Mangle the symbol passed
 * Parameters       : symbol       (I) symbol to be mangled
 * Function Result  : new mangled name of the original name
 */
String ptxMangleName( ptxSymbol symbol, uInt numScopesOnLine);

/*
 * Function         : Increment global variable numScopesOnLine
 * Parameters       : None
 * Function Result  : None
 */
void ptxIncrementNumScopesOnLine(ptxParseData parseData);

/*
 * Function         : Reset global variable numScopesOnLine to zero.
 * Parameters       : None
 * Function Result  : None
 */
void ptxResetNumScopesOnLine(ptxParseData parseData);

/*
 * Function         : Determine whether mangling is required
 * Parameters       : storage       (I) storage class of the variable, whose mangling decision must be made 
 * Function Result  : True  iff. symbol is to be mangled
 *                    False iff. symbol is not to be mangled
 */
Bool ptxIsManglingNeeded( ptxStorageClass storage );

/*
 * Function         : Add the symbol indecies of all symbols in symbol table to Atom Table
 * Parameters       : table      (I) symbol table whose varible contents should be mangled
 *                    parseState (I) parsing state which stores all parsing related information
 */
void ptxAssignIndexToSymbols(ptxSymbolTable table, ptxParsingState parseState);

/*
 * Function : Check if instruction has implicit memspace
 */
Bool ptxInstrHasImplicitMemSpace(uInt tcode);

/*
 * Function : Check if the instruction is either wmma.load or wmma.store instruction
 */
Bool ptxIsWMMALoadStore(uInt tcode);

/*
 * Function : Check if the instruction is a wmma.* instruction
 */
Bool ptxIsWMMAInstr(uInt tcode);

/*
 * Function : Checks if the input function is an OCG compiler understood builtin genomics function
 */
Bool ptxIsOcgBuiltinGenomicsFunc(ptxParseData parseData, cString name);

/*
 * Function : Checks if the input function is an OCG compiler understood builtin function
 */
Bool ptxIsOcgBuiltinFunc(ptxParseData parseData, cString name);

/*
 * Function : Checks if the input function is a compiler understood builtin function
 */
Bool ptxIsBuiltinFunc(ptxParseData parseData, cString name);

/*
 * Function : returns the symbol table entry of the callee function from the call instruction
 */
ptxSymbolTableEntry ptxCallSymTabEntFromCallInsr(ptxInstruction instr);

/*
 * Function : returns the name of the callee function from the call instruction
 */
cString ptxGetFuncNameFromCallInsr(ptxInstruction instr);

/*
 * Function         : Checks if the specified call instruction uses .param space at either output/input
 * Parameters       : instr           :  The call instruction which is to be checked for using .param spaced argument
 *                  : useReturnParam  :  Specifies whether the arguments to be checked for .param space is for return or input arguments
 * Function Result  : True if the input or output argument of the specified call instruction is in .param space
 *                    Otherwise False
 */
Bool ptxIsCallArgsUsingParamSpace(ptxInstruction instr, Bool useReturnParam);

/*
 * Function         : Checks if the specified label immediately follows the specified instruction
 * Parameters       : instr      :  The instruction which is to be checked for immediately preceeding a label
 *                  : labSymEnt  :  The label which is to be checked for immedaiately succedding an instruction
 * Function Result  : True if the label 'labSymEnt' is immediately after the instruction 'instr'; 
 *                    Otherwise False
 */
Bool ptxIsLabelImmediatelyFollowsInstr(ptxInstruction instr, ptxSymbolTableEntry labSymEnt);

/*
 * State change of lwrInstrSrc is as follows :
 *  1. Initially, its value is UserPTX
 *  2. On encountering .MACRO, it will become Macro
 *  3. In the popSymbolTable for the ending scope of .MACRO, it is set to UserPTX
 *  4. On encountering .FORCE_INLINE, it will become InlineFunction
 *  5. In the popSymbolTable for the ending scope of .FORCE_INLINE, it is set to UserPTX
 *  6. Cycles of (2)+(3) and (4)+(5) continues. After parseInputFile and parseInputString, (7) is entered
 *  7. When ptxParseMacroUtilFunc() begins and ends, the value of lwrInstrSrc changes from
 *     UserPTX -> MacroUtilFunction and MacroUtilFunction -> UserPTX respectively.
 */

typedef enum {
    UserPTX,
    Macro,
    MacroUtilFunction,
    InlineFunction,
} ptxInstructionSource;

#if LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
/*
 * Function : Set required fields to generate final expansion and call to
 *            generateNestedMacroExpansion
 */
void storeFinalExpansionOfUserMacro(ptxParsingState parseState);

/*
 * Function : Process current macro for printing
 */
void generateMacroExpansionDetails(String expansion, int macro_stack_ptr, String parentStr, ptxParsingState parseState);

/*
 * Function : Remove comments and set instruction name
 */
void preProcessMacroForPrint(String str, String name, ptxParsingState parseState);

/*
 * Function : process macro expansion if available and print all expansions
 */
void printExpansion(ptxParsingState parseState);
#endif

/*------------------------------- Instructions -------------------------------*/

typedef struct {
    uInt     APRX               : 1;
    uInt     RELU               : 1;
    uInt     FTZ                : 1;
    uInt     NOFTZ              : 1;
    uInt     KEEPREF            : 1;
    uInt     NOATEXIT           : 1;
    uInt     SAT                : 1;
    uInt     SATF               : 1;
    uInt     CC                 : 1;
    uInt     SHAMT              : 1;
    uInt     SCOPE              : 3;
    uInt     TADDR              : 1;
    uInt     ORDER              : 4;
    uInt     NC                 : 1;
    uInt     ROUND              : 4;
    uInt     TESTP              : 3;
    uInt     CACHEOP            : 4;
    uInt     LEVEL              : 2;
    uInt     EVICTPRIORITY      : 3;
    uInt     L2EVICTPRIORITY    : 3;
    uInt     LEVELEVICTPRIORITY : 4;
    uInt     FLOW               : 2;
    uInt     BRANCH             : 2;
    uInt     EXCLUSIVE          : 1;
    uInt     VECTOR             : 2;
    uInt     TEXTURE            : 4;
    uInt     DIM                : 3;
// TENSORDIM is specialized from DIM. Init value must be same for both.
    uInt     TENSORDIM          : 3;
    uInt     IM2COL             : 1;
    uInt     PACKEDOFF          : 1;
    uInt     MULTICAST          : 1;
    uInt     MBARRIER           : 1;
    uInt     FOOTPRINT          : 1;
    uInt     COARSE             : 1;
    uInt     COMPONENT          : 3;
    uInt     QUERY              : 4;
    uInt     VOTE               : 3;
    uInt     CLAMP              : 3;
    uInt     SHR                : 2;
    uInt     VMAD               : 2;
    uInt     PRMT               : 3;
    uInt     SHFL               : 3;
    uInt     ENDIS              : 2;
// .uni token, will be colwerted to UNI or VUNI after instruction is recognized
    uInt     UNIFORM            : 1;
    uInt     SYNC               : 1;
    uInt     NOINC              : 1;
    uInt     NOCOMPLETE         : 1;
    uInt     NOSLEEP            : 1;
    uInt     SHAREDSCOPE        : 2;
    uInt     BAR                : 2;
// .all/.any token, will be colwerted to BAR or VOTE after instruction is recognized
    uInt     THREADS            : 2;
    uInt     ALIGN              : 1;
    uInt     RAND               : 1;
    uInt     TTUSLOT            : 1;
    uInt     TTU                : 1;
    uInt     SHAPE              : 8;
    uInt     ALAYOUT            : 2;
    uInt     BLAYOUT            : 2;
    uInt     CACHEPREFETCH      : 3;
    uInt     PREFETCHSIZE       : 3;
    uInt     ATYPE              : 5;
    uInt     BTYPE              : 5;
    uInt     TRANS              : 1;
    uInt     EXPAND             : 1;
    uInt     NUM                : 2;
    uInt     GROUP              : 2;
    uInt     SEQ                : 1;
    uInt     COMPRESSED_FORMAT  : 5;
    uInt     EXPANDED_FORMAT    : 5;
    uInt     THREADGROUP        : 2;
    uInt     MMA_OP1            : 2;
    uInt     MMA_OP2            : 2;
    uInt     SPARSITY           : 2;
    uInt     SPFORMAT           : 2;
    uInt     DESC               : 1;
    uInt     NANMODE            : 1;
    uInt     XORSIGN            : 1;
    uInt     TRANSA             : 1;
    uInt     NEGA               : 1;
    uInt     TRANSB             : 1;
    uInt     NEGB               : 1;
    uInt     IGNOREC            : 1;
    uInt     ADDRTYPE           : 1;
    uInt     ABS                : 1;
    uInt     CACHEHINT          : 1;
    uInt     HITPRIORITY        : 3;
    uInt     MISSPRIORITY       : 3;
    uInt     OOB                : 1;
    uInt     PROXYKIND          : 1;
} ptxModifier;


typedef enum {
    PTX_APRXMOD_TABLE(GET_ENUM)
}ptxAPRXmod;

typedef enum {
    PTX_RELUMOD_TABLE(GET_ENUM)
}ptxRELUmod;

typedef enum {
    PTX_NANMODEMOD_TABLE(GET_ENUM)
}ptxNANMODEmod;

typedef enum {
    PTX_XORSIGNMOD_TABLE(GET_ENUM)
}ptxXORSIGNmod;

typedef enum {
    PTX_TRANSAMOD_TABLE(GET_ENUM)
}ptxTRANSAmod;

typedef enum {
    PTX_NEGAMOD_TABLE(GET_ENUM)
}ptxNEGAmod;

typedef enum {
    PTX_TRANSBMOD_TABLE(GET_ENUM)
}ptxTRANSBmod;

typedef enum {
    PTX_NEGBMOD_TABLE(GET_ENUM)
}ptxNEGBmod;

typedef enum {
    PTX_IGNORECMOD_TABLE(GET_ENUM)
}ptxIGNORECmod;

typedef enum {
    PTX_ADDRTYPEMOD_TABLE(GET_ENUM)
}ptxADDRTYPECmod;

typedef enum {
    PTX_FTZMOD_TABLE(GET_ENUM)
}ptxFTZmod;

typedef enum {
    PTX_OOBMOD_TABLE(GET_ENUM)
}ptxOOBmod;

typedef enum {
    PTX_NOFTZMOD_TABLE(GET_ENUM)
}ptxExplicitNOFTZmod;

typedef enum {
    PTX_KEEPREFMOD_TABLE(GET_ENUM)
}ptxKEEPREFmod;

typedef enum {
    PTX_NOATEXITMOD_TABLE(GET_ENUM)
}ptxNOATEXITmod;

typedef enum {
    PTX_SATMOD_TABLE(GET_ENUM)
}ptxSATmod;

typedef enum {
    PTX_SATFMOD_TABLE(GET_ENUM)
}ptxSATFmod;

typedef enum {
    PTX_CCMOD_TABLE(GET_ENUM)
}ptxCCmod;

typedef enum {
    PTX_SHAMTMOD_TABLE(GET_ENUM)
}ptxSHAMTmod;

typedef enum {
    PTX_SCOPEMOD_TABLE(GET_ENUM)
}ptxSCOPEmod;

typedef enum {
    PTX_TADDRMOD_TABLE(GET_ENUM)
}ptxTADDRmod;

typedef enum {
    PTX_ORDERMOD_TABLE(GET_ENUM)
}ptxORDERmod;

typedef enum {
    PTX_NCMOD_TABLE(GET_ENUM)
}ptxNCmod;

typedef enum {
    PTX_ROUNDMOD_TABLE(GET_ENUM)
}ptxROUNDmod;

typedef enum {
    PTX_TESTPMOD_TABLE(GET_ENUM)
}ptxTESTPmod;

typedef enum {
    PTX_LEVELMOD_TABLE(GET_ENUM)
}ptxLEVELmod;

typedef enum {
    PTX_EVICTPRIORITYMOD_TABLE(GET_ENUM)
}ptxEVICTPRIORITYmod;

typedef enum {
    PTX_LEVELEVICTPRIORITYMOD_TABLE(GET_ENUM)
}ptxLEVELEVICTPRIORITYmod;

typedef enum {
    PTX_CACHEOPMOD_TABLE(GET_ENUM)
}ptxCACHEmod;

typedef enum {
    PTX_FLOWMOD_TABLE(GET_ENUM)
}ptxFLOWmod;

typedef enum {
    PTX_BRANCHMOD_TABLE(GET_ENUM)
}ptxBRANCHmod;

typedef enum {
    PTX_EXCLUSIVEMOD_TABLE(GET_ENUM)
}ptxEXCLUSIVEmod;

typedef enum {
    PTX_VECTORMOD_TABLE(GET_ENUM)
}ptxVECTORmod;

typedef enum {
    PTX_TEXTUREMOD_TABLE(GET_ENUM)
}ptxTEXTUREmod;

typedef enum {
    PTX_DIMMOD_TABLE(GET_ENUM)
}ptxDIMmod;

typedef enum {
    PTX_IM2COLMOD_TABLE(GET_ENUM)
}ptxIM2COLmod;

typedef enum {
    PTX_PACKEDOFFMOD_TABLE(GET_ENUM)
}ptxPACKEDOFFmod;

typedef enum {
    PTX_MULTICASTMOD_TABLE(GET_ENUM)
}ptxMULTICASTmod;

typedef enum {
    PTX_MBARRIERMOD_TABLE(GET_ENUM)
}ptxMBARRIERmod;

typedef enum {
    PTX_FOOTPRINTMOD_TABLE(GET_ENUM)
}ptxFOOTPRINTmod;

typedef enum {
    PTX_COARSEMOD_TABLE(GET_ENUM)
}ptxCOARSEmod;

typedef enum {
    PTX_COMPONENTMOD_TABLE(GET_ENUM)
}ptxCOMPONENTmod;

typedef enum {
    PTX_QUERYMOD_TABLE(GET_ENUM)
}ptxQUERYmod;

// Following macros need to be adjusted when more modifiers are added
#define ptxQUERY_NUM_MODIFIERS           ptxQUERY_SAMPLES_MOD

typedef enum {
    PTX_VOTEMOD_TABLE(GET_ENUM)
}ptxVOTEmod;

typedef enum {
    PTX_CLAMPMOD_TABLE(GET_ENUM)
}ptxCLAMPmod;

typedef enum {
    PTX_SHRMOD_TABLE(GET_ENUM)
}ptxSHRmod;

typedef enum {
    PTX_VMADMOD_TABLE(GET_ENUM)
}ptxVMADmod;

typedef enum {
    PTX_PRMTMOD_TABLE(GET_ENUM)
}ptxPRMTmod;

typedef enum {
    PTX_SHFLMOD_TABLE(GET_ENUM)
}ptxSHFLmod;

typedef enum {
    PTX_ENDISMOD_TABLE(GET_ENUM)
}ptxENDISmod;

typedef enum {
    PTX_UNIFORMMOD_TABLE(GET_ENUM)
}ptxUNIFORMmod;

typedef enum {
    PTX_SYNCMOD_TABLE(GET_ENUM)
}ptxSYNCmod;

typedef enum {
    PTX_NOINCMOD_TABLE(GET_ENUM)
}ptxNOINCmod;

typedef enum {
    PTX_NOCOMPLETEMOD_TABLE(GET_ENUM)
}ptxNOCOMPLETEmod;

typedef enum {
    PTX_NOSLEEPMOD_TABLE(GET_ENUM)
}ptxNOSLEEPmod;

typedef enum {
    PTX_SHAREDSCOPEMOD_TABLE(GET_ENUM)
}ptxSHAREDSCOPEmod;

typedef enum {
    PTX_BARMOD_TABLE(GET_ENUM)
}ptxBARmod;

typedef enum {
    PTX_ALIGNMOD_TABLE(GET_ENUM)
}ptxALIGNmod;

typedef enum {
    PTX_THREADSMOD_TABLE(GET_ENUM)
}ptxTHREADSmod;

typedef enum {
    PTX_RANDMOD_TABLE(GET_ENUM)
}ptxRANDmod;

typedef enum {
    PTX_TTUSLOTMOD_TABLE(GET_ENUM)
}ptxTTUSLOTmod;

typedef enum {
    PTX_TTUMOD_TABLE(GET_ENUM)
}ptxTTUmod;

typedef enum {
    PTX_SHAPEMOD_TABLE(GET_ENUM),
    ptxSHAPE_MAX_MOD,
}ptxSHAPEmod;

typedef enum {
    PTX_LAYOUTMOD_TABLE(GET_ENUM)
}ptxLAYOUTmod;

typedef enum {
    PTX_SPARSITYMOD_TABLE(GET_ENUM)
}ptxSPARSITYmod;

typedef enum {
    PTX_SPFORMATMOD_TABLE(GET_ENUM)
}ptxSPFORMATmod;

typedef enum {
    PTX_CACHEPREFETCHMOD_TABLE(GET_ENUM)
}ptxCACHEPREFETCHmod;

typedef enum {
    PTX_PREFETCHSIZEMOD_TABLE(GET_ENUM)
}ptxPREFETCHSIZEmod;

typedef enum {
    PTX_DESCMOD_TABLE(GET_ENUM)
}ptxDESCmod;

typedef enum {
    PTX_TRANSMOD_TABLE(GET_ENUM)
}ptxTRANSmod;

typedef enum {
    PTX_EXPANDMOD_TABLE(GET_ENUM)
}ptxEXPANDmod;

typedef enum {
    PTX_SEQMOD_TABLE(GET_ENUM)
}ptxSEQmod;

typedef enum {
    PTX_NUMMOD_TABLE(GET_ENUM)
}ptxNUMmod;

typedef enum {
    PTX_GROUPMOD_TABLE(GET_ENUM)
}ptxGROUPmod;

typedef enum {
    PTX_TYPEMOD_TABLE(GET_ENUM)
}ptxTYPEmod;

typedef enum {
    PTX_ABSMOD_TABLE(GET_ENUM)
}ptxABSmod;

typedef enum {
    PTX_CACHEHINTMOD_TABLE(GET_ENUM)
}ptxCACHEHINTmod;

typedef enum {
    PTX_PROXYKINDMOD_TABLE(GET_ENUM)
}ptxPROXYKINDmod;

// Functions to check TYPE modifiers
Bool isI8Mod(uInt x);
Bool isF32Mod(uInt x);
Bool isS32Mod(uInt x);
Bool isB32Mod(uInt x);
Bool isB1Mod(uInt x);
Bool isI16Mod(uInt x);
Bool isI4Mod(uInt x);
Bool isI2Mod(uInt x);
Bool isS2Mod(uInt x);
Bool isU4Mod(uInt x);
Bool isS4Mod(uInt x);
Bool isE4M3Mod(uInt x);
Bool isE5M2Mod(uInt x);
Bool isBF16Mod(uInt x);
Bool isU8U4(uInt x, uInt y);
Bool isS8S4(uInt x, uInt y);
Bool isU4U2(uInt x, uInt y);
Bool isS4S2(uInt x, uInt y);

typedef enum {
    PTX_THREADGROUPMOD_TABLE(GET_ENUM)
} ptxTHREADGROUPMod;

typedef enum {
    PTX_MMA_OPMOD_TABLE(GET_ENUM)
} ptxMMA_OPmod;

#define ptxHasBYTE_MOD(mods)       (((mods).BYTE)      != 0)
#define ptxHasSCOPE_MOD(mods)      (((mods).SCOPE)     != 0)
#define ptxHasROUND_MOD(mods)      (((mods).ROUND)     != 0)
#define ptxHasTESTP_MOD(mods)      (((mods).TESTP)     != 0)
#define ptxHasXORSIGN_MOD(mods)    (((mods).XORSIGN)   != 0)
#define ptxHasCACHEOP_MOD(mods)    (((mods).CACHEOP)   != 0)
#define ptxHasLEVEL_MOD(mods)      (((mods).LEVEL)     != 0)
#define ptxHasEVICTPRIORITY_MOD(mods) (((mods).EVICTPRIORITY) != 0)
#define ptxHasVECTOR_MOD(mods)     (((mods).VECTOR)    != 0)
#define ptxHasFLOW_MOD(mods)       (((mods).FLOW)      != 0)
#define ptxHasBRANCH_MOD(mods)     (((mods).BRANCH)    != 0)
#define ptxHasEXCLUSIVE_MOD(mods)  (((mods).EXCLUSIVE) != 0)
#define ptxHasTEXTURE_MOD(mods)    (((mods).TEXTURE)   != 0)
#define ptxHasTENSORDIM_MOD(mods)  (((mods).TENSORDIM) != 0)
#define ptxHasDIM_MOD(mods)        (((mods).DIM)       != 0)
#define ptxHasIM2COL_MOD(mods)     (((mods).IM2COL)    != 0)
#define ptxHasPACKEDOFF_MOD(mods)  (((mods).PACKEDOFF) != 0)
#define ptxHasMULTICAST_MOD(mods)  (((mods).MULTICAST) != 0)
#define ptxHasMBARRIER_MOD(mods)   (((mods).MBARRIER)  != 0)
#define ptxHasFOOTPRINT_MOD(mods)  (((mods).FOOTPRINT) != 0)
#define ptxHasGRANULARTY_MOD(mods) (((mods).COARSE)    != 0)
#define ptxHasCOMPONENT_MOD(mods)  (((mods).COMPONENT) != 0)
#define ptxHasQUERY_MOD(mods)      (((mods).QUERY)     != 0)
#define ptxHasCLAMP_MOD(mods)      (((mods).CLAMP)     != 0)
#define ptxHasSHR_MOD(mods)        (((mods).SHR)       != 0)
#define ptxHasVMAD_MOD(mods)       (((mods).VMAD)      != 0)
#define ptxHasPRMT_MOD(mods)       (((mods).PRMT)      != 0)
#define ptxHasSHFL_MOD(mods)       (((mods).SHFL)      != 0)
#define ptxHasENDIS_MOD(mods)      (((mods).ENDIS)     != 0)
#define ptxHasRAND_MOD(mods)       (((mods).RAND)      != 0)
#define ptxHasVOTE_MOD(mods)       (((mods).VOTE)      != 0)
#define ptxHasSYNC_MOD(mods)       (((mods).SYNC)      != 0)
#define ptxHasNOINC_MOD(mods)      (((mods).NOINC)     != 0)
#define ptxHasNOCOMPLETE_MOD(mods) (((mods).NOCOMPLETE)!= 0)
#define ptxHasNOSLEEP_MOD(mods)    (((mods).NOSLEEP)   != 0)
#define ptxHasSHAREDSCOPE_MOD(mods) (((mods).SHAREDSCOPE) != 0)
#define ptxHasBAR_MOD(mods)        (((mods).BAR)       != 0)
#define ptxHasALIGN_MOD(mods)      (((mods).ALIGN)     != 0)
#define ptxHasTTUSLOT_MOD(mods)    (((mods).TTUSLOT)   != 0)
#define ptxHasTTU_MOD(mods)        (((mods).TTU)   != 0)
#define ptxHasSHAPE_MOD(mods)      (((mods).SHAPE)     != 0)
#define ptxHasLAYOUT_MOD(mods)     (((mods).LAYOUT)    != 0)
#define ptxHasALAYOUT_MOD(mods)    (((mods).ALAYOUT)   != 0)
#define ptxHasBLAYOUT_MOD(mods)    (((mods).BLAYOUT)   != 0)
#define ptxHasCACHEPREFETCH_MOD(mods)   (((mods).CACHEPREFETCH) != 0)
#define ptxHasPREFETCHSIZE_MOD(mods)    (((mods).PREFETCHSIZE)  != 0)
#define ptxHasATYPE_MOD(mods)      (((mods).ATYPE)     != 0)
#define ptxHasBTYPE_MOD(mods)      (((mods).BTYPE)     != 0)
#define ptxHasTRANS_MOD(mods)      ((mods).TRANS       != 0)
#define ptxHasEXPAND_MOD(mods)     ((mods).EXPAND      != 0)
#define ptxHasSEQ_MOD(mods)        ((mods).SEQ         != 0)
#define ptxHasNUM_MOD(mods)        ((mods).NUM         != 0)
#define ptxHasGROUP_MOD(mods)      ((mods).GROUP       != 0)
#define ptxHasTHREADS_MOD(mods)    ((mods).THREADS     != 0)
#define ptxHasCOMPRESSED_FORMAT_MOD(mods)  ((mods).COMPRESSED_FORMAT  != 0)
#define ptxHasEXPANDED_FORMAT_MOD(mods)    ((mods).EXPANDED_FORMAT    != 0)
#define ptxHasTHREADGROUP_MOD(mods)        ((mods).THREADGROUP        != 0)
#define ptxHasSPARSITY_MOD(mods)           ((mods).SPARSITY           != 0)
#define ptxHasSPFORMAT_MOD(mods)           ((mods).SPFORMAT           != 0)
#define ptxHasDESC_MOD(mods)               ((mods).DESC               != 0)
#define ptxHasRELU_MOD(mods)               ((mods).RELU               != 0)
#define ptxHasSATF_MOD(mods)               ((mods).SATF               != 0)
#define ptxHasADDRTYPE_MOD(mods)           (((mods).ADDRTYPE)         != 0)
#define ptxHasABS_MOD(mods)                ((mods).ABS                != 0)
#define ptxHasCACHEHINT_MOD(mods)          ((mods).CACHEHINT          != 0)
#define ptxHasHITPRIORITY_MOD(mods)        ((mods).HITPRIORITY        != 0)
#define ptxHasMISSPRIORITY_MOD(mods)       ((mods).MISSPRIORITY       != 0)
#define ptxHasLEVELEVICTPRIORITY_MOD(mods) ((mods).LEVELEVICTPRIORITY != 0)
#define ptxHasPROXYKIND_MOD(mods)          ((mods).PROXYKIND          != 0)

// various functions for modifiers
uInt ptxGetVectorSize( ptxModifier modifier );
uInt ptxGetArgVectorLength( ptxExpression arg );
Bool ptxHasRoundFModifier( ptxModifier modifier );
Bool ptxHasRoundIModifier( ptxModifier modifier );
Bool ptxHasTexQueryModifier( ptxModifier modifier );
Bool ptxHasSamplerQueryModifier( ptxModifier modifier );
Bool ptxHasSurfQueryModifier( ptxModifier modifier );

// Get index of memspace corresponding to argument
int ptxGetStorageIndex(uInt tcode, uInt argId, uInt nrofInstrMemspace);

typedef struct {
    uInt     BOP                : 1;
    uInt     BYTE               : 1;
    uInt     CMP                : 1;
    uInt     KEEPREF            : 1;
    uInt     NOATEXIT           : 1;
    uInt     RESULT             : 1;
    uInt     RESULTP            : 1;
    uInt     APRX               : 1;
    uInt     RELU               : 1;
    uInt     FTZ                : 1;
    uInt     NOFTZ              : 1;
    uInt     SAT                : 1;
    uInt     SATF               : 1;
    uInt     VSAT               : 1;
    uInt     CC                 : 1;
    uInt     SHAMT              : 1;
    uInt     ROUNDF             : 1;
    uInt     ROUNDI             : 1;
    uInt     SIGNED             : 1;
    uInt     FLOW               : 1;
    uInt     BRANCH             : 1;
    uInt     EXCLUSIVE          : 1;
    uInt     DOUBLERES          : 1;     // DOUBLERES applies to DEST and SRC3 (see MUL, MAD instructions)
    uInt     LARG               : 1;
    uInt     SREGARG            : 1;
    uInt     MEMSPACE           : 1;
    uInt     MEMSPACES          : 1;
    uInt     TESTP              : 1;
    uInt     CACHEOP            : 1;
    uInt     ORDER              : 1;
    uInt     LEVEL              : 1;
    uInt     EVICTPRIORITY      : 1;
    uInt     SCOPE              : 1;
    uInt     VECTORIZABLE       : 1;
    uInt     TEXADDR            : 1;
    uInt     TEXMOD             : 1;
    uInt     TENSORDIM          : 1;
    uInt     IM2COL             : 1;
    uInt     PACKEDOFF          : 1;
    uInt     MULTICAST          : 1;
    uInt     MBARRIER           : 1;
    uInt     FOOTPRINT          : 1;
    uInt     COARSE             : 1;
    uInt     COMPMOD            : 1;
    uInt     SURFQ              : 1;
    uInt     SMPLQ              : 1;
    uInt     TEXQ               : 1;
    uInt     VOTE               : 1;
    uInt     ATOMOPF            : 1;
    uInt     ATOMOPI            : 1;
    uInt     ATOMOPB            : 1;
    uInt     ARITHOP            : 1;
    uInt     CAS                : 1;
    uInt     CLAMP              : 1;
    uInt     SHR                : 1;
    uInt     VMAD               : 1;
    uInt     PRMT               : 1;
    uInt     SHFL               : 1;
    uInt     ENDIS              : 1;
    uInt     RAND               : 1;
    uInt     SYNC               : 1;
    uInt     NOINC              : 1;
    uInt     NOCOMPLETE         : 1;
    uInt     NOSLEEP            : 1;
    uInt     SHAREDSCOPE        : 1;
    uInt     BAR                : 1;
    uInt     ALIGN              : 1;
    uInt     TTUSLOT            : 1;
    uInt     TTU                : 1;
    uInt     SHAPE              : 1;
    uInt     CACHEPREFETCH      : 1;
    uInt     PREFETCHSIZE       : 1;
    uInt     TRANS              : 1;
    uInt     NUM                : 1;
    uInt     SEQ                : 1;
    uInt     GROUP              : 1;
    uInt     EXPAND             : 1;
    uInt     THREADGROUP        : 1;
    uInt     SPARSITY           : 1;
    uInt     SPFORMAT           : 1;
    uInt     DESC               : 1;
    uInt     NANMODE            : 1;
    uInt     XORSIGN            : 1;
    uInt     TRANSA             : 1;
    uInt     NEGA               : 1;
    uInt     TRANSB             : 1;
    uInt     NEGB               : 1;
    uInt     IGNOREC            : 1;
    uInt     ADDRTYPE           : 1;
    uInt     ABS                : 1;
    uInt     CACHEHINT          : 1;
    uInt     OOB                : 1;
    uInt     PROXYKIND          : 1;
} ptxInstructionFeature;

#define ptxHasBOP_Feature(f)           (((f).BOP)          != 0)
#define ptxHasBYTE_Feature(f)          (((f).BYTE)         != 0)
#define ptxHasCMP_Feature(f)           (((f).CMP)          != 0)
#define ptxHasKEEPREF_Feature(f)       (((f).KEEPREF)      != 0)
#define ptxHasNOATEXIT_Feature(f)      (((f).NOATEXIT)     != 0)
#define ptxHasRESULT_Feature(f)        (((f).RESULT)       != 0)
#define ptxHasRESULTP_Feature(f)       (((f).RESULTP)      != 0)
#define ptxHasAPRX_Feature(f)          (((f).APRX)         != 0)
#define ptxHasRELU_Feature(f)          (((f).RELU)         != 0)
#define ptxHasNANMODE_Feature(f)       (((f).NANMODE)      != 0)
#define ptxHasXORSIGN_Feature(f)       (((f).XORSIGN)      != 0)
#define ptxHasTRANSA_Feature(f)        (((f).TRANSA)       != 0)
#define ptxHasNEGA_Feature(f)          (((f).NEGA)         != 0)
#define ptxHasTRANSB_Feature(f)        (((f).TRANSB)       != 0)
#define ptxHasNEGB_Feature(f)          (((f).NEGB)         != 0)
#define ptxHasIGNOREC_Feature(f)       (((f).IGNOREC)      != 0)
#define ptxHasADDRTYPE_Feature(f)      (((f).ADDRTYPE)     != 0)
#define ptxHasFTZ_Feature(f)           (((f).FTZ)          != 0)
#define ptxHasNOFTZ_Feature(f)         (((f).NOFTZ)        != 0)
#define ptxHasSAT_Feature(f)           (((f).SAT)          != 0)
#define ptxHasSATF_Feature(f)          (((f).SATF)         != 0)
#define ptxHasVSAT_Feature(f)          (((f).VSAT)         != 0)
#define ptxHasCC_Feature(f)            (((f).CC)           != 0)
#define ptxHasSHAMT_Feature(f)         (((f).SHAMT)        != 0)
#define ptxHasROUNDF_Feature(f)        (((f).ROUNDF)       != 0)
#define ptxHasROUNDI_Feature(f)        (((f).ROUNDI)       != 0)
#define ptxHasSIGNED_Feature(f)        (((f).SIGNED)       != 0)
#define ptxHasFLOW_Feature(f)          (((f).FLOW)         != 0)
#define ptxHasBRANCH_Feature(f)        (((f).BRANCH)       != 0)
#define ptxHasEXCLUSIVE_Feature(f)     (((f).EXCLUSIVE)    != 0)
#define ptxHasDOUBLERES_Feature(f)     (((f).DOUBLERES)    != 0)
#define ptxHasLARG_Feature(f)          (((f).LARG)         != 0)
#define ptxHasSREGARG_Feature(f)       (((f).SREGARG)      != 0)
#define ptxHasMEMSPACE_Feature(f)      (((f).MEMSPACE)     != 0)
#define ptxHasMEMSPACES_Feature(f)     (((f).MEMSPACES)    != 0)
#define ptxHasTESTP_Feature(f)         (((f).TESTP)        != 0)
#define ptxHasCACHEOP_Feature(f)       (((f).CACHEOP)      != 0)
#define ptxHasORDER_Feature(f)         (((f).ORDER)        != 0)
#define ptxHasLEVEL_Feature(f)         (((f).LEVEL)        != 0)
#define ptxHasEVICTPRIORITY_Feature(f) (((f).EVICTPRIORITY)  != 0)
#define ptxHasSCOPE_Feature(f)         (((f).SCOPE)        != 0)
#define ptxHasVECTORIZABLE_Feature(f)  (((f).VECTORIZABLE) != 0)
#define ptxHasTEXADDR_Feature(f)       (((f).TEXADDR)      != 0)
#define ptxHasTEXMOD_Feature(f)        (((f).TEXMOD)       != 0)
#define ptxHasTENSORDIM_Feature(f)     (((f).TENSORDIM)    != 0)
#define ptxHasIM2COL_Feature(f)        (((f).IM2COL)       != 0)
#define ptxHasPACKEDOFF_Feature(f)     (((f).PACKEDOFF)    != 0)
#define ptxHasMULTICAST_Feature(f)     (((f).MULTICAST)    != 0)
#define ptxHasMBARRIER_Feature(f)      (((f).MBARRIER)     != 0)
#define ptxHasFOOTPRINT_Feature(f)     (((f).FOOTPRINT)    != 0)
#define ptxHasCOARSE_Feature(f)        (((f).COARSE)       != 0)
#define ptxHasCOMPMOD_Feature(f)       (((f).COMPMOD)      != 0)
#define ptxHasSURFQ_Feature(f)         (((f).SURFQ)        != 0)
#define ptxHasSMPLQ_Feature(f)         (((f).SMPLQ)        != 0)
#define ptxHasTEXQ_Feature(f)          (((f).TEXQ)         != 0)
#define ptxHasVOTE_Feature(f)          (((f).VOTE)         != 0)
#define ptxHasATOMOPF_Feature(f)       (((f).ATOMOPF)      != 0)
#define ptxHasATOMOPI_Feature(f)       (((f).ATOMOPI)      != 0)
#define ptxHasATOMOPB_Feature(f)       (((f).ATOMOPB)      != 0)
#define ptxHasARITHOP_Feature(f)       (((f).ARITHOP)      != 0)
#define ptxHasCAS_Feature(f)           (((f).CAS)          != 0)
#define ptxHasCLAMP_Feature(f)         (((f).CLAMP)        != 0)
#define ptxHasSHR_Feature(f)           (((f).SHR)          != 0)
#define ptxHasVMAD_Feature(f)          (((f).VMAD)         != 0)
#define ptxHasPRMT_Feature(f)          (((f).PRMT)         != 0)
#define ptxHasSHFL_Feature(f)          (((f).SHFL)         != 0)
#define ptxHasENDIS_Feature(f)         (((f).ENDIS)        != 0)
#define ptxHasRAND_Feature(f)          (((f).RAND)         != 0)
#define ptxHasSYNC_Feature(f)          (((f).SYNC)         != 0)
#define ptxHasNOINC_Feature(f)         (((f).NOINC)        != 0)
#define ptxHasNOCOMPLETE_Feature(f)    (((f).NOCOMPLETE)   != 0)
#define ptxHasNOSLEEP_Feature(f)       (((f).NOSLEEP)      != 0)
#define ptxHasSHAREDSCOPE_Feature(f)   (((f).SHAREDSCOPE)  != 0)
#define ptxHasBAR_Feature(f)           (((f).BAR)          != 0)
#define ptxHasALIGN_Feature(f)         (((f).ALIGN)        != 0)
#define ptxHasTTUSLOT_Feature(f)       (((f).TTUSLOT)      != 0)
#define ptxHasTTU_Feature(f)           (((f).TTU)          != 0)
#define ptxHasSHAPE_Feature(f)         (((f).SHAPE)        != 0)
#define ptxHasTRANS_Feature(f)         (((f).TRANS)        != 0)
#define ptxHasEXPAND_Feature(f)        (((f).EXPAND)       != 0)
#define ptxHasSEQ_Feature(f)           (((f).SEQ)          != 0)
#define ptxHasNUM_Feature(f)           (((f).NUM)          != 0)
#define ptxHasGROUP_Feature(f)         (((f).GROUP)        != 0)
#define ptxHasCACHEPREFETCH_Feature(f) (((f).CACHEPREFETCH) != 0)
#define ptxHasPREFETCHSIZE_Feature(f)  (((f).PREFETCHSIZE) != 0)
#define ptxHasTHREADGROUP_Feature(f)   (((f).THREADGROUP)  != 0)
#define ptxHasSPARSITY_Feature(f)      (((f).SPARSITY)     != 0)
#define ptxHasSPFORMAT_Feature(f)      (((f).SPFORMAT)     != 0)
#define ptxHasDESC_Feature(f)          (((f).DESC)         != 0)
#define ptxHasABS_Feature(f)           (((f).ABS)          != 0)
#define ptxHasCACHEHINT_Feature(f)     (((f).CACHEHINT)    != 0)
#define ptxHasOOB_Feature(f)           (((f).OOB)          != 0)
#define ptxHasPROXYKIND_Feature(f)     (((f).PROXYKIND)    != 0)

#define ptxHasATOMIC_Feature(f)        (ptxHasATOMOPF_Feature(f)        \
                                        || ptxHasATOMOPI_Feature(f)     \
                                        || ptxHasATOMOPB_Feature(f)     \
                                        || ptxHasCAS_Feature(f))

typedef enum {
    RealInstruction,
    MacroInstruction,
    InlineFunctionCall
} ptxInstructionKind;

typedef enum {
    ptxUnknownIType,
    ptxFloatIType,
    ptxPackedHalfFloatIType,
    ptxIntIType,
    ptxBitIType,
    ptxPredicateIType,
    ptxOpaqueIType,
    ptxLwstomFloatE8IType,
    ptxLwstomFloatTF32Type,
    ptxLwstomFloatFP8Type
} ptxInstructionType;

typedef enum {
    ptxFollowAType,
    ptxU16AType,
    ptxU32AType,
    ptxU64AType,
    ptxS32AType,
    ptxF32AType,
    ptxF16x2AType,
    ptxScalarF32AType, 
    ptxB32AType,
    ptxB64AType,
    ptxImageAType,
    ptxConstantIntAType,
    ptxConstantFloatAType,
    ptxPredicateAType,
    ptxPredicateVectorAType,
    ptxMemoryAType,
    ptxSymbolAType,
    ptxTargetAType,
    ptxParamListAType,
    ptxVoidAType,
    ptxLabelAType
} ptxArgumentType;

typedef enum {
    ptxMbarrierInstruction,
    ptxAddressSize32,
    ptxTexGrad3DLwbeAlwbe,
    ptxTexGradDepthCompare,
    ptxGenomics,
    ptxCoroutine,
    ptxSyscallCompilation,
    ptxTexInstruction,
    ptxSurfInstruction
} ptxNonMercFeature;

#define ptxMAX_INSTR_NAME 128
#define ptxMAX_INSTR_MEMSPACE 2
#define ptxMAX_EVICTPRIORITY_MODS 2
#define ptxMAX_LEVEL_MODS 2
#define ptxMAX_INSTR_ARGS 9
#define ptxMAX_MODS 256
#define ptxMAX_INLINE_FUNCTION_INPUT_ARGS  30
#define ptxMAX_INLINE_FUNCTION_OUTPUT_ARGS 30

typedef struct ptxInstructionTemplateRec {
    String              name;
    uInt                code;     // an ptxInstructionCode, see generated ptxInstructions.h

    ptxInstructionFeature features;

    uInt                nrofInstrTypes;
    ptxInstructionType  instrType      [ptxMAX_INSTR_ARGS];
    stdBitSet_t         instrTypeSizes [ptxMAX_INSTR_ARGS];

    uInt                nrofArguments;
    ptxArgumentType     argType        [ptxMAX_INSTR_ARGS];

    uInt                followMap      [ptxMAX_INSTR_ARGS];
} ptxInstructionTemplateRec;

typedef struct ptxToSourceLineInfoMap {
    stdMap_t labelMap;
    stdMap_t inlinedLocMap;
    stdRangeMap_t instructionMap;
    stdRangeMap_t functionPtxLineRangeMap;
} ptxToSourceLineInfoMap ;


typedef struct ptxInstructionRec {
    uInt64                  virtualAddress;
    uInt64                  stmtId;
    ptxExpression           guard;
    ptxExpression           predicateOutput;
    ptxInstructionTemplate  tmplate;        // 'template' reserved keyword in C++
    ptxStorageClass*        storage;        // for instructions that affect storage
    int                     nrofStorage;    // number of memspaces populated in storage[]
    ptxModifier             modifiers;      // depending on specific features
    ptxComparison           cmp;            // in case of CMP feature
    ptxOperator             postop;         // For instructions that have (post-)operations. ptxNOP when absent
    ptxExpression          *arguments;      // Instruction arguments (nrofArguments)
    ptxType                *type;           // Type specified for instruction (nrofInstrTypes)
    ptxCodeLocation         loc;            // source code reference
} ptxInstructionRec;

/*-------------------------------- Statements --------------------------------*/

typedef enum {
    ptxInstructionStatement,
    ptxPragmaStatement
} ptxStatementKind;

typedef struct ptxStatementRec {
    ptxStatementKind         kind;

    union {
        struct {
            ptxInstruction   instruction;
        } Instruction;
        
        struct {
            listXList       (pragmas);           // list of statement-level pragma strings
        } Pragma;
        
    } cases;
} ptxStatementRec;



/*------------------------ Incrementally Parsed Program ----------------------*/


Bool ptxIsInternalSource(ptxInstructionSource lwrInstrSrc);
Bool ptxIsExpandedInternally(ptxInstructionSource lwrInstrSrc);
Bool areRestrictedUseFeaturesAllowed(ptxParsingState globalState);


typedef struct ptxLineInfoRec {
    uInt   fileIndex;
    uInt   lineNo;
    uInt   linePos;
} ptxLineInfoRec;

typedef struct ptxCodeLocationRec {
    ptxLineInfoRec lwrLoc;
    String functionName;
    struct ptxCodeLocationRec *inlineAtLoc;
} ptxCodeLocationRec;

Pointer ptxCreateKeyFromLoc(Int file, Int line, Int pos);

typedef struct ptxResourceInfoRec {
    Bool isSharedMemUsed;
    Bool isTextureUsed;
    Bool isSurfaceUsed;
    Bool isSamplerUsed;
    Bool isConstUsed;
} ptxResourceInfoRec;

typedef enum {
    NOT_PARSED = 0,
    NEED_PARSE = 1,
    DONE_PARSE = 2
} macroUtilFuncParseState;

typedef struct {
    const unsigned long long*     funcBody;
    macroUtilFuncParseState parseState;
} macroUtilFuncData_t;

typedef struct printMacroExpansionInfoRec {
    stdVector_t                 expansionStack;
    stdXArray(uInt32, expansionLengthStack);
    stdXArray(uInt32, locationStack);
    uInt                        expansionLengthStackTop;
    uInt                        locationStackTop;
    int                         expansionLength;
    int                         nofExpansion;
    String                      insName;
    Bool                        expandNestedMacro;
    int                         prev_macro_stack_ptr;
    stdXArray(uInt32, parentStack);
} printMacroExpansionInfoRec;

typedef enum {
    GPU_ARCH,
    SUPPORT_FAST_DIVISION,
    NEED_VIDEO_EMULATION,
    MEMBAR_WITH_ILWALL,
    IS_MERLWRY,
    FORCE_ALIGNED_SYNC_INSTRS,  // only support textually aligned Cooperative Group sync instructions
    EXPAND_SYNC_INST_LATE,      // Enable late expansion of sync instructions
    LEGACY_BAR_WARP_WIDE_BEHAVIOR,
    FORCE_OUTLINED_WMMA,
    ENABLE_UNALIGNED_WMMA_INSTRS,
    DISABLE_SUPER_HMMA,
    USE_MMA884_EMULATION,
    EXPAND_TEX_INTO_SYSCALL,
    PTX_MAJOR_VERSION,
    PTX_MINOR_VERSION,
    ELWVARS_MAX = PTX_MINOR_VERSION
} macroElwVar;

#define MAX_MACRO_DEPTH 32      //allowing maximum nested macro-expansion depth of 32

typedef struct ptxParseInfoRec {
    ptxDeclarationScope lwrScope;
    ptxType             lwrType;
    ptxSymbol           lwrDecl;
    ptxStorageClass     lwrStorageClass;
    uInt                lwrLogAlignment;
    ptxCodeLocation     lwrSourceCodeLocation, prvSourceCodeLocation;
    ptxCodeLocation     prologueSourceLocation;  // source code location where
                                                 // instruction from prologue for
                                                 // current function should mapped to
    ptxCodeLocation     nextSourceCodeLocation;  // source code location of next ptx instruction
    Bool                isInlineFunc;
    Bool                lwrIsEntry;
    Bool                lwrIsPrototype;
    ptxSymbolTableEntry lwrFunc;               // current function being parsed
    Bool                sawIncompleteParam;
    stdSet_t            reservedMacroFormals;
    stdMap_t            reservedMacroAliases;
    ptxSymbolTable      lastUserSymbolTable;
    ptxDwarfSection     lwrPtxDwarfSection;
    ptxDwarfLine        lwrPtxDwarfLine;
    uInt64              lwrSymbolAttributes;
    Bool                isF64Allowed;
    int                 moduleScopeNumAbiParamReg;
    int                 moduleScopeRetAddrBeforeParams;
    int                 moduleScopeRetAddrReg;
    int                 moduleScopeRetAddrUReg;
    int                 moduleScopeRelRetAddrReg;
    stdList_t           moduleScopeScratchRRegs;
    stdList_t           moduleScopeScratchBRegs;
    int                 moduleScopeFirstParamReg;
    Bool                moduleScopeCoroutinePragma;

    Bool                noSyncWarningEmitted;
    int                 firstInstrSourcePos;
    int                 lastInstrSourcePos;

    uInt                nrofEvictPriorityMod;
    uInt                evictPriorityMods[ptxMAX_EVICTPRIORITY_MODS];
    uInt                nrofLevelColonMod;
    uInt                levelColonMods[ptxMAX_LEVEL_MODS];
    uInt                nrofInstructionTypes;
    uInt                nrofTypeMod;
    uInt                nrofBMMAOperations;
    ptxType             instructionType[ptxMAX_INSTR_ARGS];
    uInt                typeMod[ptxMAX_INSTR_ARGS];
    uInt                BMMAOperations[ptxMAX_INSTR_ARGS];
    uInt                matrixLayout[2];
    uInt                nrofMatrixLayout;
    Bool                instrUsesTypeMod;
    Bool                isLastSeelwectorType;

    // Variable to track usage of "double" in an instruction.
    // Scope of this variable is only for an instruction so should be 
    // reset once new instruction is being parsed.
    Bool doubleUse;
    Bool parsingDwarfData;

    ptxExpression     guard;
    ptxExpression     predicateOutput;
    uInt              nrofInstrMemspace;
    ptxStorageClass   storage[ptxMAX_INSTR_MEMSPACE];
    uInt              nrofArguments;
    ptxModifier       modifiers;

    ptxComparison     cmp;

    ptxOperator       postop;
    ptxOperatorClass  postopclass;

    // In each of these arrays it is assumed that the number of qualifiers
    // for any valid instruction is strictly less than the constant ptxMAX_INSTR_ARGS.
    ptxExpression     arguments      [ptxMAX_INSTR_ARGS];
    int               instruction_tcode;
    uInt              nr;

    cString           ptxfilename;
    uInt              numScopesOnLine;

    // Macro-Instruction specific info

    int nrInstrTypes, nrArgs;
    char *predOutStr;
    char *arg[ptxMAX_INSTR_ARGS];
    ptxTypeKind instrTypes[ptxMAX_INSTR_ARGS+1];


    // Inline-Function specific info

    int             nrInlineFuncInputArgs, nrInlineFuncRetArgs;
    char            *inlineInputArgName [ptxMAX_INLINE_FUNCTION_INPUT_ARGS];
    char            *inlineOutputArgName[ptxMAX_INLINE_FUNCTION_OUTPUT_ARGS];
    ptxExpression   inlineInputArg [ptxMAX_INLINE_FUNCTION_INPUT_ARGS];
    ptxExpression   inlineOutputArg[ptxMAX_INLINE_FUNCTION_OUTPUT_ARGS];


    //  Common information for macro processing

    macroElwVar     elwVars[ELWVARS_MAX+1];
    char            *guardStr;

    // information required in Lexxer 

    stdList_t pushedInputs;
    // OPTIX_HAND_EDIT adding these three members for incremental parsing
    stdList_t pushedInputLens;
    stdList_t pushedPending;
    String    pushedInput;
    // OPTIX_HAND_EDIT end
    size_t    pushedInputLen;
    stdList_t pushedSourcePos;
    stdList_t pushedObfuscators;

    /* YY_BUFFER_STATE*/ void *macro_stack[MAX_MACRO_DEPTH];
    int macro_stack_ptr;
    uInt32 ptxCount;
    Bool initLexState;
    Char ptxLookahead;

    // Variables from ptxIR that are required in parseData as well
    String                  target_arch;
    String                  version;
    Bool                    isTexModeIndependent;

    // Instruction template processing
    stdMap_t                stdTemplates;
    stdMap_t                extTemplates;
    stdMap_t*               deobfuscatedStringMapPtr;
} ptxParseInfoRec;

typedef struct ptxParsingStateRec {
    stdMemSpace_t           memSpace;
    gpuFeaturesProfile      gpuInfo;
    stdMap_t                preprocessorMacros;
    IAtomTable              *atoms;             // Reference to the Atom table
    stdMap_t                dwarfLiveRangeMap;  // This is a map from functionName->(map(labelName->list of ptxDwarfLiveRangeMapListNodeRec)) 
    void                    *scanner;           // Scanner to store Lex - Parse state
    String                  inputFileName;

    ptxSymbolTable          globalSymbols;
    ptxSymbolTable          macroSymbols;
    stdSet_t                parsedObjects;      // ptxSymbolTable, one for each parsed input file
    stdList_t               entryPointsIndicies;   // Indicies of entry points

    stdMap_t                locationLabels;     // when debugging info is generated: 
                                                //   maps location labels to sequence numbers and entry name
    stdMap_t                sassAddresses;      // when sass debugging info is generated: 
                                                //   maps location labels to SASS addresses and entry name
    stdMap_t                paramOffset;        // when debugging info/sass debugging is generated: 
                                                //   maps param symbols to offset and entry name; 
                                            
    stdMap_t                symbolNamesCnt;     // String    --> nr           for function
    stdMap_t                symbolNames;        // ptxSymbol --> String             ptxGetUniqueName
    stdMap_t                arrayInit;           // String --> ptxVariableInfo
    
    String                  version;
    String                  target_arch;
    // OPTIX_HAND_EDIT adding parsed versions of version and target_arch strings to avoid
    // having to call sscanf over and over again.
    uInt                    version_major;
    uInt                    version_minor;
    uInt                    target_arch_int;
    // to avoid the static global optix state inside the lexer, we add it to parsing state. But due to unknown local struct definition
    // we keep it here as a void* and cast it locally inside the lexer via (OptixInputState)globalState->optixInputState
    void                    *optixInputState;
    // OPTIX_HAND_EDIT end
    Bool                    isTexmodeUnified;
    Bool                    isVersionTargetMismatch; // True iff PTX .version is not supporting .target
    msgSourcePos_t          targetDirectivePos;      // PTX source location of '.target directive'
    uInt                    addr_size;          // Tri-state 0, 32, 64
    /* Tristate
     * 0 -> Can't infer address size
     *      Can happen when all memory accesses in load/store etc. are direct
     * 32 -> 32bit address size
     * 64 -> 64bit address size
     */
    uInt                    inferredAddressSize;
    uInt                    max_target;
    stdMap_t                target_opts;
    Bool                    warn_on_double_demotion;
    Bool                    warn_on_double_use;
    Bool                    usesModuleScopedRegOrLocalVars;  // disable ABI mode if legacy PTX contains module-scoped .reg or .local vars
    Bool                    setTexmodeIndependent;          // True if "texmode_independent" set through command line option
    String                  moduleScopedRegOrLocalVarName;
    Bool                    usesFuncPointer;
    Bool                    callsExternFunc;
    Bool                    usesFuncWithMultipleRets;
    Bool                    isEmptyUserPTX;
    String                  funcWithMultipleRetsName;
    Bool                    usesQueryInstruction;
    uInt                    nextUniqueFuncIndex;

    listXList               (pragmas);          // list of file-scoped pragma strings

    ptxToSourceLineInfoMap  ptxToSourceLine;

   /*
    * Dwarf information extracted from ptx input file:
    */
    stdMap_t                    dwarfFiles;     // index -> DebugIndexedFile
    stdMap_t                    internalDwarfLabel; // Labels defined in DWARF section -> section+offset
    listXList                  (dwarfBytes);
    listXList                  (dwarfSections); // list of ptxDwarfSection
    struct { listXList (l); }   dwarfLocations [ ptxMAXStorage ];
    
    /* MetaData information */
    ptxMetaDataSection          metadataSection;

    uInt64                      virtualSize    [ ptxMAXStorage ];
    char                        *deobfuscatedMacro;        // Holds deobfuscated macro body
    stdMap_t                    macroMap;
    stdMap_t                    inlineFuncsMap;
    Bool                        allowAllPtxFeatures;       // Internal option that disables per-feature PTX version checks.
    Bool                        allowNonSyncInstrs;        // Allows using non-sync SHFL/VOTE in user program
    Bool                        enablePtxDebug; // .target debug used
    Bool                        foundDebugInfo; // Debug information found
    Bool                        enableLineInfoGeneration; // .loc found for PTX with version 4.0 or higher
    msgSourceStructure_t        ptxBuiltInSourceStruct;   // Used for creating source positions while populating opaque and global symbol table
    msgSourceStructure_t        ptxFileSourceStruct;      // Used for creating source positions for parsed symbols
    uInt                        generatePrefetchSizeSeed; // Internal option used to initialize seed for random 
                                                          // prefetch size for sector promotion stress testing.
    ptxResourceInfoRec          ptxResourceInfo;          // Contains boolean variables to indicate status of use of shared m/m
                                                          // textures, surfaces, samplers and constants
    Bool                        parsingTTUBlock;          // Used to check non-TTU instructions are present in TTU block

    printMacroExpansionInfoRec  printMacroExpansionInfo;       // Macro Expansion Information
    ptxInstructionSource        lwrInstrSrc;

    ptxSymbolTable              objectSymbolTable;
    ptxSymbolTable              globalSymbolTable;
    ptxSymbolTable              lwrSymbolTable;
    ptxSymbolTable              macroSymbolTable;
    Bool                        parsingParameters;

    stdMap_t                    uniqueTypes;
    stdMap_t                    *funcIndexToSymbolMap;
    FILE                        *ptxin;
    Bool                        ptxStringInput;
    Bool                        ptxDebugInfo;
    Bool                        ptxDebugOneLineBB;
    stdObfuscationState         ptxObfuscation;
    uInt32                      ptxLength; /* length of obfuscated PTX string */
    Bool                        ptxAllowMacros;

    macroUtilFuncData_t *       utilFuncs;
    stdMap_t                    macroUtilFuncMap;
    int                         numMacroUtilFunc;
    listXList                   (nonMercFeaturesUsed);    // list of ptxNonMercFeature used in program.
    ptxParseData                 parseData; // data structures used in actual parsing of input.
} ptxParsingStateRec;

/*This info is used to generate relocator in .debug_info section*/
typedef struct ptxSymLocInfoRec {
    String entry;
    uInt offset;
    Bool isParam;
} ptxSymLocInfoRec;

typedef struct ptxParamVarSaveRec {
    char save[2];
} ptxParamVarSaveRec;

#define DWARF_LABEL_INDICATOR 32
typedef struct dwarfLinesRec {

    // we cannot used stdVector to store DWARF data/long constants because the size
    // of vector element is equal to size of Pointer which is not sufficient
    // to hold packed record/long constants on 32 bit system

    stdXArray(uInt64, dwarfLineBytes); // encoded value indicating presence of
                                       // labels and size with dwarfdata
    uInt dwarfLineBytesTop, longConstantArrTop; // Indicates the index of top
                                                // element of array
    stdXArray(uInt64, longConstantArr); // Used to store 64 bit constant


} dwarfLinesRec;

typedef struct ptxDwarfSectionRec {
    String              name;
    dwarfLinesRec       dwarfLines;         // Holds the array of packed dwarf elements
    uInt                size;               // section size
    uInt                totalBytesFilled;   // Indicates total number of bytes
                                            // filled in dwarfLines
    stdVector_t         labelVector;        // Used to store dwarf labels
    DwarfSectionType    sectionType;        // Dwarf section code as defined in lwdwarf.h
} ptxDwarfSectionRec;


typedef struct ptxMetaDataSectionRec {
    stdMap_t    metaDataNodeMap;
} ptxMetaDataSectionRec;

typedef struct ptxMetaDataNodesRec {
    uInt index;
    String nodeName;
    stdList_t metaDataValues;       // list of ptxMetaDataValue
} ptxMetaDataNodesRec;

typedef enum {
    ptxMetaDataValueInt,
    ptxMetaDataValueIndex,
    ptxMetaDataValueString
} ptxMetaDataKind;

typedef struct ptxMetaDataValueRec {
    ptxMetaDataKind metadataKind;
    union {
        uInt val;
        uInt metadataIndex;
        String str;
    } cases;
} ptxMetaDataValueRec;

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
 *                    state          (IO) Parsing state to receive result
 *                    debugInfo      (I)  True iff. debug info needs be generated
 *                    debugOneLineBB (I)  True iff. a basic block is needed per source line
 *                    lineInfo       (I)  True iff. line info needs to be generated
 * Function Result  : True iff. parsing succeeded
 */
  Bool ptxParseInputFile( String inputFileName, uInt32 obfuscationKey, ptxParsingState object, Bool debugInfo, Bool debugOneLineBB, Bool lineInfo );

// OPTIX_HAND_EDIT
/*
 * Function         : Incremental parse of assembly, directly from ptx string.
 *                    The result of parsing will be added as a new symbol table
 *                    to the parsedObjects set in the state parameter.
 *                    global symbol resolution will take place in order to
 *                    resolve 'extern' symbols to definitions in other parsed files.
 *                    Hence, this function also serves as a linker.
 * Parameters       : ident          (I)  identifier for error handling. start with '<' for non user input
 *                    ptx            (I)  ptx assembly program as string
 *                    obfuscationKey (I)  value by which the ptx string was obfuscated, or zero
 *                    state          (IO) Parsing state to receive result
 *                    debugInfo      (I)  True iff. debug info needs be generated
 *                    debugOneLineBB (I)  True iff. a basic block is needed per source line
 *                    lineInfo       (I)  True iff. line info needs to be generated
 *                    ptxStringLength(I)  length of ptx input
 *                    decrypter      (I)  Instance of OptiX 7 EncryptionManager
 *                    decryptionCB   (I)  Decryption callback
 * Function Result  : True iff. parsing succeeded
 */
  Bool ptxParseInputString( cString ident, String ptx, uInt32 obfuscationKey, ptxParsingState object, Bool debugInfo, Bool debugOneLineBB, Bool lineInfo, uInt32 ptxStringLength, void* decrypter, GenericCallback decryptionCB );

/*
 * Function           : ptxDwarfGetSectionPointer
 * Parameters         : ptxParsingState
 *                    : DwarfSectionType
 * Result             : pointer to the Dwarf section record matching the input dwarf section.
 */
ptxDwarfSection ptxDwarfGetSectionPointer(ptxParsingState, DwarfSectionType);

/*
 * Function         : Create new, empty parsed state, to incrementally
 *                    fill by repeated calls to ptxParseInputFile.
 *                  : ptxInfo                        (I)  void pointer of "ptxInfo" which should be passed to
 *                                                        AddExtraPreProcessorMacroFlags
 * Parameters       : gpuInfo                        (I)  spec of gpu to parse for
                    : AddExtraPreProcessorMacroFlags (I)  function to set extra preprocessor macro flag for 
                                                          expanding tesla intrinsics for ori/ori-debug
 * Function Result  : Fresh, empty ptx parsed state
 */
ptxParsingState ptxCreateEmptyState(void *ptxInfo,
                                    gpuFeaturesProfile gpuInfo,
                                    IAtomTable* atoms,
                                    stdMap_t* funcIndexToSymbolMap,
                                    void (*AddExtraPreProcessorMacroFlags)(ptxParsingState state, void *ptxInfo),
                                    uInt generatePrefetchSizeSeed,
                                    cString extDescFileName, cString extDescAsString, stdMap_t* deobfuscatedStringMapPtr);

/*
 * Function         : Close object, finalize parsing state
 * Parameters       : object   (I) object to close
 * Function Result  :
 */
void ptxCloseObject( ptxParsingState object );

/*
 * Function         : Discard object, with entire parsing state that it contains
 * Parameters       : state   (I) state to delete
 * Function Result  : 
 */
void ptxDeleteObject( ptxParsingState state );

/*
 * Function         : Initialize lexer state
 * Parameters       : None
 * Function Result  : 
 */
void initPtxLex( void );

/*
 * Function         : Clean lexer state
 * Parameters       : None
 * Function Result  : 
 */
void termPtxLex( void );


/*
 * Function         : Print parsed debug information to specified file,
 *                    and builds up state->locationLabels as a side effect.
 * Parameters       : files         (I) the input files that where parsed.
 *                    state         (I) state to print
 *                    f             (I) file to print to
 * Function Result  : 
 */
void ptxPrintDebugInfo( stdList_t files, ptxParsingState state, FILE *f );


/*
 * Function         : Print import/export symbol information in option format for fatbin command
 * Parameters       : state       (I) state to print
 *                    f           (I) file to print to
 * Function Result  : 
 */
void ptxPrintLinkInfo( ptxParsingState state, FILE *f );


/*--------------------------- Expression Printing ----------------------------*/

/*
 * Function         : Print ptx comparison operator to specified string.
 * Parameters       : c           (I) Comparison operator to print
 *                    s           (I) string to print to
 * Function Result  : 
 */
void ptxPrintComparison( ptxComparison c, stdString_t s );


/*
 * Function         : Print ptx operator to specified string.
 * Parameters       : o           (I) Operator to print
 *                    s           (I) string to print to
 * Function Result  : 
 */
void ptxPrintOperator( ptxOperator o, stdString_t s );


/*
 * Function         : Print ptx operator to specified string (in modifier form).
 * Parameters       : o           (I) Operator to print
 *                    s           (I) string to print to
 * Function Result  : 
 */
void ptxPrintOperatorAsModifier( ptxOperator o, stdString_t s );

/*
 * Function         : Print parsed expression to specified string.
 * Parameters       : e           (I) Expression to print
 *                    s           (I) string to print to
 * Function Result  : 
 */
void ptxPrintExpression( ptxExpression e, stdString_t s );


/*
 * Function         : Print ptx type to specified string.
 * Parameters       : t           (I) Type expression to print
 *                    s           (I) string to print to
 * Function Result  : 
 */
void ptxPrintType( stdMap_t* deobfuscatedStringMapPtr, ptxType t, stdString_t s );

/*
 * Function         : Return a unique name of the specified symbol that 
 *                    can be used in a global namespace, for instance in the
 *                    symbol map assembly file generated for debugging
 * Parameters       : state       (I) ptx object that defines the symbol
 *                    symbol      (I) Symbol to list
 * Function Result  : 
 */
String ptxGetUniqueName( ptxParsingState state, ptxSymbol symbol );

/*
 * Function         : It gives the pre-image of the uniquename generated by ptxGetUniqueName(),
                      essentially it is the reverse of the algo used in ptxGetUniqueName().
 * Parameters       : uname       (I) ptr to uniquename
 * Function Result  : 
 */
String ptxGetPreUniqueName(String uname);

/*
 * Function         : Iteratively parse all called macro util funcs
 * Parameters       : object   (I) parsing state to use
 * Function Result  :
 */
void ptxProcessMacroUtilFuncs( ptxParsingState object );

/*
 * Function         : Sets an unique func index (starting from '0') for defined functions
 * Parameters       : funcSym    (I) function symbol
 *                    parseState (I) parsing state which stores all parsing related information
 * Function Result  :
 */
void ptxSetUniqueFuncIndex(ptxSymbolTableEntry funcSym, ptxParsingState parseState);

/*
 * Function         : Returns total number of defined functions
 * Parameters       : parseState (I) parsing state which stores all parsing related information
 * Function Result  :
 */
uInt ptxGetDefinedFunctionCount(ptxParsingState parseState);

/*
 * Function         : Checks if any of the LWCA SASS directives are used for a function
 * Parameters       : function on which presence of LWCA SASS directive is to be checked
 * Function Result  : True if LWCA SASS directives are used on th function symbol
 *                    False, otherwise
 */
Bool usesLwdaSass(ptxSymbolTableEntry func);

/*
 * Function         : Checks if any of the Custom ABI pragmas are used for a function
 * Parameters       : function on which presence of Custom ABI pragmas is to be checked
 * Function Result  : True if Custom ABI pragmas are used on th function symbol
 *                    False, otherwise
 */
Bool usesLwstomABI(ptxSymbolTableEntry func);

/*
 * Function         : Checks if the function is decorated with coroutine pragma. Such function can suspend.
 * Parameters       : function on which presence of coroutine pragmas is to be checked
 * Function Result  : True if coroutine pragmas are used on th function symbol
 *                    False, otherwise
 */
Bool usesCoroutine(ptxSymbolTableEntry func);

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
                          uInt64* ScratchR191To128, uInt64* ScratchR255To192);
/*
 * Function         : Parses reg count value from abi_param_reg, call_abi_param_reg and local_maxnreg pragma
 * Parameters       : parseData, Pragma
 * Function Result  : register count
 */
int ptxGetPragmaValue(ptxParseData parseData, String pragma);

/*
 * Function         : Parses multiple values from a known pragma
 * Parameters       : Pragma - pragma to be parsed
 *                    sortFinalList - whether to sort the final list in ascending order
 *                    addDummyTerminator - whether to add a dummy element (-1) at the end of the result list
 * Function Result  : Pragma value as a list
 */
stdList_t ptxGetPragmaValueList(ptxParseData parseData, String pragma, Bool sortFinalList, Bool addDummyTerminator);

/*
 * Function         : Parses value of abi_param_reg pragma
 * Parameters       : (I) Pragma   - pragma to be parsed
 *                    (O) numReg   - number of registers to be used for parameters
 *                    (O) startReg - starting register to be used for parameters
 *                    (I) parseState - parsing state which stores all parsing related information
 */
void ptxGetAbiParamRegPragma(String pragma, int* numReg, int* startReg, ptxParsingState parseState);

/*
 * Function         : Returns the value of a known jetfire pragma.
 * Parameters       : Pragma - Jetfire pragma whose value to be parsed.
                      isPragmaValueInt - Flag helps to interpret the returned pragma-value.
                                         (pragma-value can be either of int or String)
                      pragmaValue - Integer pragma value if `isPragmaValueInt` is True.
 * Function Result  : pragma-value as a string.
 */
String ptxGetJetfirePragmaValue(String pragma, Bool *isPragmaValueInt, Int64 *pragmaValue, ptxParseData parseData);

/*
 * Function         : Parses and sanitizes value of a known jetfire pragma
 * Parameters       : Pragma - Jetfire pragma to be parsed and sanitize
                      sourcePos - Referring source location
 */
void ptxSanitizeJetfirePragmaValue(String pragma, msgSourcePos_t sourcePos, ptxParseData parseData);

/*
 * Function         : Parses and sanitizes value of a known sync pragma
 * Parameters       : Pragma - Sync pragma to be parsed and sanitize
                      sourcePos - Referring source location
 */
void ptxSanitizeSyncPragma(String pragma, msgSourcePos_t sourcePos);

/*
 * Function         : Used to check whether the input function is a alias
 * Parameters       : symEnt   (I) function symbol which is to be checked for its alias-ness
 * Function Result  : True if 'symEnt' is an alias symbol
 *                    False, otherwise
 */
Bool isAliasSymbol(ptxSymbolTableEntry symEnt);

/*
 * Function         : Used to query aliasee function for an alias function
 * Parameters       : symEnt   (I) function symbol whose aliasee is to be determined
 * Function Result  : aliasee function symbol if input function 'symEnt' is an alias
 *                    'symeEnt' function symbol if input function 'symEnt' is not an alias
 */
ptxSymbolTableEntry ptxResolveAliasSymbol(ptxSymbolTableEntry alias);

/* Utility Functions */

/*
 * Function         : Get the operation-modifier from MMA_OPMOD table corresponding to the POSTOP
 * Parameters       : ptxOperator     (I) Postop whose corresponding entry
 *                                        in MMA_OPMOD table is to be found
 * Function Result  : MMA_OPMOD table entry which corresponds to ptxOperator 'op'
 */
uInt colwertPostOpToMMAOperation(ptxOperator op);

/*
 * Function         : Get the type-modifier from TYPEMOD table corresponding to the ptxType
 * Parameters       : type     (I) PTX Type whose corresponding entry in TYPEMOD table is to be found
 * Function Result  : TYPEMOD table entry which corresponds to ptxType 'type'
 */
uInt ptxGetTypeModFromType(ptxType type);

/* Function         : Get the type from TYPEMOD
 * Parameters       : typeMod    (I) PTX TypeMod whose corresponding ptx thpe table is to be found
 * Function Result  : ptx Type which corresponds to typeMod
 */
ptxType ptxGetTypeFromTypeMod(uInt typeMod, ptxParsingState parseState);

/*
 * Function         : Recognize TypeMod from a given string
 * Parameters       : modNameStr     (I) Name of the TypeMod to be recognized
 * Function Result  : ptxTYPEmod enum value corresponding to the input string
 */
uInt recognizeTypeMod(ptxParseData parseData, cString modNameStr);

/*
 * Function         : Get the type's size in bits
 * Parameters       : type     (I) Type from TYPEMOD table whose size is to be queried
 * Function Result  : Size of 'type' in bits
 */
uInt ptxGetTypeModSize(uInt type);

/*
 * Function         : Get number of group represented by group modifier
 * Parameters       : group modifier
 * Function Result  : number of group represented
 */
uInt ptxGetNumOfGroups(uInt groupMod);

/*
 * Function         : Appends the feature to list ptxIR->nonMercFeaturesUsed, only if it's not in the list.
 * Parameters       : feature : ptxNonMercFeature whcih is to be added to the list.
 */
#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_COMPUTE_COMPILER_INTERNAL)
void ptxSetNonMercFeatureUsed(ptxParsingState parseState, ptxNonMercFeature feature);
#endif

#define FITS_IN_INT32(num) ((num) <= 0xFFFFFFFF)
Bool ptxVersionAtLeast( int major, int minor, ptxParsingState parseState );
int ptxGetLatestMajorVersion(void);
int ptxGetLatestMinorVersion(void);
Bool ptxIsSupportedIsaVersion(int version);

Bool isDirectCall(ptxExpression *arguments, uInt nrofArguments);
Bool isImmediate(ptxExpression Expr);
Int64 ptxGetImmediateIntVal(ptxExpression exp);

Bool isB1  (ptxType type);
Bool isB2  (ptxType type);
Bool isB4  (ptxType type);
Bool isB8  (ptxType type);
Bool isB16 (ptxType type);
Bool isB32 (ptxType type);
Bool isB64 (ptxType type);
Bool isB128(ptxType type);
Bool isS2  (ptxType type);
Bool isS4  (ptxType type);
Bool isS8  (ptxType type);
Bool isS16 (ptxType type);
Bool isS32 (ptxType type);
Bool isS64 (ptxType type);
Bool isU2  (ptxType type);
Bool isU4  (ptxType type);
Bool isU8  (ptxType type);
Bool isU16 (ptxType type);
Bool isU32 (ptxType type);
Bool isU64 (ptxType type);
Bool isI2  (ptxType type);
Bool isI4  (ptxType type);
Bool isI8  (ptxType type);
Bool isI16 (ptxType type);
Bool isI32 (ptxType type);
Bool isI64 (ptxType type);
Bool isE4M3 (ptxType type);
Bool isE5M2 (ptxType type);
Bool isE4M3x2 (ptxType type);
Bool isE5M2x2 (ptxType type);
Bool isF8   (ptxType type);
Bool isF8x2 (ptxType type);
Bool isF16 (ptxType type);
Bool isF16x2 (ptxType type);
Bool isF32 (ptxType type);
Bool isF64 (ptxType type);
Bool isBF16(ptxType type);
Bool isBF16x2(ptxType type);
Bool isTF32(ptxType type);
Bool isPRED(ptxType type);
Bool isTEXREF    (ptxType type);
Bool isSAMPLERREF(ptxType type);
Bool isSURFREF   (ptxType type);
Bool isArray (ptxType type);
Bool isBitType(ptxType type);
Bool isInteger(ptxType type);
Bool isFloat(ptxType type);
Bool isSignedInt(ptxType type);
Bool isIntegerKind(ptxTypeKind kind);
Bool isFloatKind(ptxTypeKind kind);
Bool isBitTypeKind(ptxTypeKind kind);
Bool areAllFourMatrixTypesF16(ptxType instructionType[ptxMAX_INSTR_ARGS], 
                          uInt nrofInstructionTypes);

typedef enum {
    ptxDirectCall,
    ptxCallWithTargetList,
    ptxCallWithCallPrototype
} ptxCallInstrType;

/*
* TODO: This can be used more widely for eg within: isMMA(), checkMMA().
* Depending upon such usage, may need to extend the definition.
*/

typedef enum {
    ptxMMASubTypeHMMA,
    ptxMMASubTypeSparseHMMA,
    ptxMMASubTypeIMMA,
    ptxMMASubTypeSparseIMMA,
    ptxMMASubTypeSubByteIMMA,
    ptxMMASubTypeSubByteSparseIMMA,
    ptxMMASubTypeBMMA,
    ptxMMASubTypeDMMA,
    ptxMMASubTypeNonStdFp,
    ptxMMASubTypeSparseNonStdFp,
    ptxMMASubTypeWMMAFp,
    ptxMMASubTypeWMMADoubleFp,
    ptxMMASubTypeWMMANonStdFp,
    ptxMMASubTypeWMMANonFp,
    ptxMMASubTypeFpLoad,
    ptxMMASubTypeFpStore,
    ptxMMASubTypeNonStdFpLoad,
    ptxMMASubTypeDoubleFpLoad,
    ptxMMASubTypeDoubleFpStore,
    ptxMMASubTypeNonFpLoad,
    ptxMMASubTypeNonFpStore,
    ptxMMASubTypeQGMMA,
    ptxMMASubTypeHGMMA,
    ptxMMASubTypeIGMMA,
    ptxMMASubTypeBGMMA,
    ptxMMASubTypeUnknown
} ptxMMASubType;

#if (LWCFG(GLOBAL_CHIP_T194) || LWCFG(GLOBAL_GPU_IMPL_GV11B) || LWCFG(GLOBAL_ARCH_TURING)) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_63)
Bool isIMMA(ptxParsingState gblState, uInt tcode, uInt nrofInstructionTypes,
            ptxType instructionType[ptxMAX_INSTR_ARGS], 
            ptxModifier modifiers, Bool useTypeMod);
Bool isDenseIMMA(ptxParsingState gblState, uInt tcode, uInt nrofInstructionTypes,
                 ptxType instructionType[ptxMAX_INSTR_ARGS], 
                 ptxModifier modifiers, Bool useTypeMod);
#endif

#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_63)
Bool isBMMA(ptxParsingState gblState, uInt tcode, uInt nrofInstructionTypes,
            ptxType instructionType[ptxMAX_INSTR_ARGS],
            ptxModifier modifiers, Bool useTypeMod);
#endif

#if LWCFG(GLOBAL_ARCH_VOLTA) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_60)
Bool isHMMA(uInt tcode, uInt nrofInstructionTypes,
            ptxType instructionType[ptxMAX_INSTR_ARGS]);
#endif

#if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_70)
Bool isSparseIMMA(ptxParsingState gblState, uInt tcode, uInt nrofInstructionTypes,  
                  ptxType instructionType[ptxMAX_INSTR_ARGS],
                  ptxModifier modifiers, Bool useTypeMod);
#endif

#if LWCFG(GLOBAL_ARCH_HOPPER) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
Bool isHGMMA(uInt tcode, uInt nrofInstructionTypes,
            ptxType instructionType[ptxMAX_INSTR_ARGS]);
Bool isQGMMA(uInt tcode, uInt nrofInstructionTypes,
             ptxType instructionType[ptxMAX_INSTR_ARGS]);
Bool isIGMMA(ptxParsingState gblState, uInt tcode, uInt nrofInstructionTypes,
    ptxType instructionType[ptxMAX_INSTR_ARGS], ptxModifier modifiers, Bool useTypeMod);
Bool isBGMMA(ptxParsingState gblState, uInt tcode, uInt nrofInstructionTypes,
    ptxType instructionType[ptxMAX_INSTR_ARGS], ptxModifier modifiers, Bool useTypeMod);
#endif

#if LWCFG(GLOBAL_ARCH_ADA) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
Bool isWideLoadStoreInstr(uInt tcode, ptxType type, ptxModifier modifiers);
#endif // ada, internal

Bool isSrcSizePresentForCopyInstruction(uInt tcode, ptxExpression *args, uInt nargs, Bool hasMemDesc);
Bool isIgnoreSrcPresentForCopyInstruction(uInt tcode, ptxExpression *args, uInt nargs);
ptxExpression getIgnoreSrcArgForCopyInstr(uInt tcode, ptxExpression *args, uInt nargs);

Bool isDenseIMMAWithExplicitTypes(uInt shape);

Bool ptxIsParameterizedAuxSymbol(ptxSymbolTableEntry ptxsymEnt);

#if     defined(__cplusplus)
}
#endif 

#endif /* ptxIR_INCLUDED */
