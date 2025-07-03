#ifndef ptxIR_FWD_INCLUDED
#define ptxIR_FWD_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ptxSymbolRec              *ptxSymbol;
typedef struct ptxTypeRec                *ptxType;
typedef struct ptxExpressionRec          *ptxExpression;
typedef struct ptxSymbolTableEntryRec    *ptxSymbolTableEntry;
typedef struct ptxSymbolTableRec         *ptxSymbolTable;
typedef struct ptxInstructionTemplateRec *ptxInstructionTemplate;
typedef struct ptxInstructionRec         *ptxInstruction;
typedef struct ptxStatementRec           *ptxStatement;
typedef struct ptxParsingStateRec        *ptxParsingState;
typedef struct ptxInitializerRec         *ptxInitializer;
typedef struct ptxCodeLocationRec        *ptxCodeLocation;

#if     defined(__cplusplus)
}
#endif 

#endif /* ptxIR_FWD_INCLUDED */
