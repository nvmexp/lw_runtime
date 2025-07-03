/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2010-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef ptxMacroUtils_INCLUDED
#define ptxMacroUtils_INCLUDED

#include "ptxIR.h"

#ifdef __cplusplus
extern "C" {
#endif


/* ****************** API Functions for pasrer and ptxas *************************** */

void initMacroState(ptxParseData parseData);
void freeMacroState(ptxParseData parseData, int nrofArguments);
void initMacroElwVar(ptxParseData parseData, macroElwVar evar, int value);

void initMacroInstrTypes(ptxParseData parseData, ptxType instrType[], int n);
void initMacroInstrGuard(ptxParseData parseData, String s);
void initMacroInstrArgs(ptxParseData parseData, String s, int index);
void initMacroNumInstrArgs(ptxParseData parseData, uInt nrofArguments);
void initMacroPredicateOutput(ptxParseData parseData, String s);

void initMacroProfileTesla(ptxParsingState ptxIR);
void initMacroProfileFermi(ptxParsingState ptxIR);

void initMacroUtilFuncMap(ptxParsingState ptxIR);
void deleteMacroUtilFuncMap();
int  getMacroUtilFuncIndex(String name);
int  getNumMacroUtilFunc(void);
Bool IsMacroFunc(String name, ptxParsingState parseState);
Bool IsUniqueMacroFunc(ptxSymbolTableEntry symEnt, ptxParsingState parseState);
Bool IsNonUniqueMacroFunc(ptxSymbolTableEntry symEnt, ptxParsingState parseState);

/* ----------  Inline-Function Utilities ---------- */
void initInlineFunctionArgs(ptxParseData parseData, uInt nrofIArguments, uInt nrofOArguments,
                            String funcName, msgSourcePos_t sourcePos);
void setInlineFunctionInputArg(ptxParseData parseData, int index, String s, ptxExpression e);
void setInlineFunctionOutputArg(ptxParseData parseData, int index, String s, ptxExpression e);
void freeInlineFunctionArgs(ptxParseData parseData);


/* ****************** API Functions for macro processor *************************** */

int get_BYTE(ptxParseData parseData);
String getBYTEAsString(int byte);
String get_strBYTE(ptxParseData parseData);

int get_KEEPREF(ptxParseData parseData);
String getKEEPREFAsString(stdMap_t* deobfuscatedStringMapPtr, int keepref);
String get_strKEEPREF(ptxParseData parseData);

int get_NOATEXIT(ptxParseData parseData);
String getNOATEXITAsString(stdMap_t* deobfuscatedStringMapPtr, int noatexit);
String get_strNOATEXIT(ptxParseData parseData);

int get_POSTOP(ptxParseData parseData);
String getPOSTOPAsString(int postop);
String get_strPOSTOP(ptxParseData parseData);
String get_strBOP(ptxParseData parseData);
String get_str_POSTOP(ptxParseData parseData);

int get_LAYOUT(ptxParseData parseData);
int get_ALAYOUT(ptxParseData parseData);
int get_BLAYOUT(ptxParseData parseData);
String getLAYOUTAsString(int layout);
String get_strLAYOUT(ptxParseData parseData);
String get_strALAYOUT(ptxParseData parseData);
String get_strBLAYOUT(ptxParseData parseData);
String get_str_LAYOUT(ptxParseData parseData);
String get_str_ALAYOUT(ptxParseData parseData);
String get_str_BLAYOUT(ptxParseData parseData);
String get_strLAYOUT_TRANS(ptxParseData parseData);

int get_DESC(ptxParseData parseData);
String get_strDESC(ptxParseData parseData);
String getDESCAsString(stdMap_t* deobfuscatedStringMapPtr, int desc);
String get_str_DESC(ptxParseData parseData);
int get_DESC_ARG(ptxParseData parseData);
String get_strDESC_ARG(ptxParseData parseData);
String get_strOPTIONAL_DESC_ARG(ptxParseData parseData);
String get_strARGS(ptxParseData parseData);

// Sparsity related functions
int get_SPFORMAT(ptxParseData parseData);
int get_SPARSITY(ptxParseData parseData);
String get_strSPARSITY(ptxParseData parseData);
String get_strSPFORMAT(ptxParseData parseData);
String getSPARSITYAsString(int s);
String getSPFORMATAsString(stdMap_t* deobfuscatedStringMapPtr, int s);

int get_SHAPE(ptxParseData parseData);
String getSHAPEAsString(stdMap_t* deobfuscatedStringMapPtr, int shape);
String get_strSHAPE(ptxParseData parseData);
String get_str_SHAPE(ptxParseData parseData);

String get_strLDM(ptxParseData parseData);

String get_strCTYPE(ptxParseData parseData);
String get_str_CTYPE(ptxParseData parseData);
String get_strDTYPE(ptxParseData parseData);
String get_str_DTYPE(ptxParseData parseData);

int get_COMPARE(ptxParseData parseData);
String getCOMPAREAsString(int compare);
String get_strCOMPARE(ptxParseData parseData);

int get_APRX(ptxParseData parseData);
String getAPRXAsString(int aprx);
String get_strAPRX(ptxParseData parseData);

int get_RELU(ptxParseData parseData);
String getRELUAsString(int relu);
String get_strRELU(ptxParseData parseData);

int get_NANMODE(ptxParseData parseData);
String getNANMODEAsString(int nan);
String get_strNANMODE(ptxParseData parseData);

int get_XORSIGN(ptxParseData parseData);
String getXORSIGNAsString(int xorsign);
String get_strXORSIGN(ptxParseData parseData);

int get_TRANSA(ptxParseData parseData);
String getTRANSAAsString(stdMap_t* deobfuscatedStringMapPtr, int nan);
String get_strTRANSA(ptxParseData parseData);

int get_NEGA(ptxParseData parseData);
String getNEGAAsString(stdMap_t* deobfuscatedStringMapPtr, int nan);
String get_strNEGA(ptxParseData parseData);

int get_TRANSB(ptxParseData parseData);
String getTRANSBAsString(stdMap_t* deobfuscatedStringMapPtr, int nan);
String get_strTRANSB(ptxParseData parseData);

int get_NEGB(ptxParseData parseData);
String getNEGBAsString(stdMap_t* deobfuscatedStringMapPtr, int nan);
String get_strNEGB(ptxParseData parseData);

int get_IGNOREC(ptxParseData parseData);
String getIGNORECAsString(stdMap_t* deobfuscatedStringMapPtr, int nan);
String get_strIGNOREC(ptxParseData parseData);

int get_ADDRTYPE(ptxParseData parseData);
String getADDRTYPEAsString(stdMap_t* deobfuscatedStringMapPtr, int addrtype);
String get_strADDRTYPE(ptxParseData parseData);

int get_FTZ(ptxParseData parseData);
String getFTZAsString(int order);
String get_strFTZ(ptxParseData parseData);

int get_NOFTZ(ptxParseData parseData);
String getNOFTZAsString(int order);
String get_strNOFTZ(ptxParseData parseData);

int get_OOB(ptxParseData parseData);
String getOOBAsString(stdMap_t* deobfuscatedStringMapPtr, int oob);
String get_strOOB(ptxParseData parseData);

int get_SAT(ptxParseData parseData);
String getSATAsString(int sat);
String get_strSAT(ptxParseData parseData);
String get_str_SAT(ptxParseData parseData);

int get_SATF(ptxParseData parseData);
String getSATFAsString(int satf);
String get_strSATF(ptxParseData parseData);
String get_str_SATF(ptxParseData parseData);

int get_SYNC(ptxParseData parseData);
String getSYNCAsString(int sync);
String get_strSYNC(ptxParseData parseData);

int get_NOINC(ptxParseData parseData);
String getNOINCAsString(int nocomplete);
String get_strNOINC(ptxParseData parseData);

int get_NOCOMPLETE(ptxParseData parseData);
String getNOCOMPLETEAsString(int nocomplete);
String get_strNOCOMPLETE(ptxParseData parseData);

int get_NOSLEEP(ptxParseData parseData);
String getNOSLEEPAsString(stdMap_t* deobfuscatedStringMapPtr, int nosleep);
String get_strNOSLEEP(ptxParseData parseData);

int get_SHAREDSCOPE(ptxParseData parseData);
String getSHAREDSCOPEAsString(stdMap_t* deobfuscatedStringMapPtr, int mod);
String get_strSHAREDSCOPE(ptxParseData parseData);

String get_strOPTIONAL_CP_MBARRIER_ARVCNT_ARG(ptxParseData parseData);

int get_TRANS(ptxParseData parseData);
String getTRANSAsString(int trans);
String get_strTRANS(ptxParseData parseData);

int get_EXPAND(ptxParseData parseData);
String getEXPANDAsString(stdMap_t* deobfuscatedStringMapPtr, int expand);
String get_strEXPAND(ptxParseData parseData);

int get_EXCLUSIVE(ptxParseData parseData);
String getEXCLUSIVEAsString(stdMap_t* deobfuscatedStringMapPtr, int mod);
String get_strEXCLUSIVE(ptxParseData parseData);

int get_NUM(ptxParseData parseData);
String getNUMAsString(int num);
String get_strNUM(ptxParseData parseData);

int get_GROUP(ptxParseData parseData);
String getGROUPAsString(stdMap_t* deobfuscatedStringMapPtr, int grp);
String get_strGROUP(ptxParseData parseData);

int get_SEQ(ptxParseData parseData);
String getSEQAsString(stdMap_t* deobfuscatedStringMapPtr, int SEQ);
String get_strSEQ(ptxParseData parseData);

int get_TTUSLOT(ptxParseData parseData);
String getTTUSLOTAsString(stdMap_t* deobfuscatedStringMapPtr, int ttuslot);
String get_strTTUSLOT(ptxParseData parseData);

int get_TTUMOD(ptxParseData parseData);
String getTTUMODAsString(stdMap_t* deobfuscatedStringMapPtr, int ttumod);
String get_strTTUMOD(ptxParseData parseData);

String getTYPEMODAsString(stdMap_t* deobfuscatedStringMapPtr, int typeMod);
String getTYPEMODAsStringRaw(stdMap_t* deobfuscatedStringMapPtr, int typeMod);

int get_EXPANDED_FORMAT(ptxParseData parseData);
String getEXPANDED_FORMATAsString(stdMap_t* deobfuscatedStringMapPtr, int num);
String get_strEXPANDED_FORMAT(ptxParseData parseData);

int get_COMPRESSED_FORMAT(ptxParseData parseData);
String getCOMPRESSED_FORMATAsString(stdMap_t* deobfuscatedStringMapPtr, int num);
String get_strCOMPRESSED_FORMAT(ptxParseData parseData);

int get_BAR(ptxParseData parseData);
String getBARAsString(int bar);
String get_strBAR(ptxParseData parseData);

int get_ALIGN(ptxParseData parseData);
String getALIGNAsString(int align);
String get_strALIGN(ptxParseData parseData);

int get_ATYPE(ptxParseData parseData);
String getATYPEAsString(stdMap_t* deobfuscatedStringMapPtr, int ATYPE);
String get_strATYPE(ptxParseData parseData);
String get_str_ATYPE(ptxParseData parseData);

int get_MMA_OP(ptxParseData parseData, int op);
String getMMA_OPAsString(int mma_op);
String get_strMMA_OP(ptxParseData parseData, int op);

String get_strWMMA_B_TYPE(ptxParseData parseData);
String get_strWMMA_A_TYPE(ptxParseData parseData);

int get_BTYPE(ptxParseData parseData);
String getBTYPEAsString(stdMap_t* deobfuscatedStringMapPtr, int BTYPE);
String get_strBTYPE(ptxParseData parseData);
int get_IS_SUB_BYTE_WMMA_LOAD(ptxParseData parseData);
int get_IS_SUB_BYTE_WMMA_MMA(ptxParseData parseData);
int get_IS_BIT_WMMA_LOAD(ptxParseData parseData);
int get_IS_BIT_WMMA_MMA(ptxParseData parseData);

int get_IS_NON_STANDARD_FP_WMMA_LOAD(ptxParseData parseData);
int get_IS_NON_STANDARD_FP_WMMA_MMA(ptxParseData parseData);
int get_IS_ATYPE_TF32(ptxParseData parseData);
int get_IS_ATYPE_BF16x2(ptxParseData parseData);
int get_IS_ATYPE_BF16(ptxParseData parseData);
int get_IS_BTYPE_BF16(ptxParseData parseData);

int get_IS_NON_STANDARD_MMA(ptxParseData parseData);
int get_IS_WMMA_TF32_MMA(ptxParseData parseData);
int get_IS_SUB_BYTE_MMA(ptxParseData parseData);

int get_THREADS(ptxParseData parseData);
String getTHREADSAsString(int threads);
String get_strTHREADS(ptxParseData parseData);

int get_THREADGROUP(ptxParseData parseData);
String getTHREADGROUPAsString(stdMap_t* deobfuscatedStringMapPtr, int threadgroup);
String get_strTHREADGROUP(ptxParseData parseData);

int get_CC(ptxParseData parseData);
String getCCAsString(int cc);
String get_strCC(ptxParseData parseData);

int get_SHAMT(ptxParseData parseData);
String getSHAMTAsString(int shamt);
String get_strSHAMT(ptxParseData parseData);

int get_SCOPE(ptxParseData parseData);
String getSCOPEAsString(int scope);
String get_strSCOPE(ptxParseData parseData);

int get_LEVEL(ptxParseData parseData);
String getLEVELAsString(int level);
String get_strLEVEL(ptxParseData parseData);

int get_EVICTPRIORITY(ptxParseData parseData);
String get_EVICTPRIORITYAsString(int evic);
String get_strEVICTPRIORITY(ptxParseData parseData);

int get_LEVELEVICTPRIORITY(ptxParseData parseData);
String get_LEVELEVICTPRIORITYAsString(int evic);
String get_strLEVELEVICTPRIORITY(ptxParseData parseData);

int get_L2EVICTPRIORITY(ptxParseData parseData);
String get_L2EVICTPRIORITYAsString(int evic);
String get_strL2EVICTPRIORITY(ptxParseData parseData);

int get_PREFETCHSIZE(ptxParseData parseData);
String get_strPREFETCHSIZE(ptxParseData parseData);
String get_PREFETCHSIZEAsString(stdMap_t* deobfuscatedStringMapPtr, int prefetchsize);

int get_CACHEPREFETCH(ptxParseData parseData);
String get_strCACHEPREFETCH(ptxParseData parseData);
String get_CACHEPREFETCHAsString(stdMap_t* deobfuscatedStringMapPtr, int cacheprefetch);

String get_strSTORAGE(ptxParseData parseData);
String get_str_STORAGE(ptxParseData parseData);
String get_strSTORAGES(ptxParseData parseData, int i);
int get_IS_GENERIC_STORAGE(ptxParseData parseData);

int get_CACHEOP(ptxParseData parseData);
String getCACHEOPAsString(stdMap_t* deobfuscatedStringMapPtr, int cacheop);
String get_strCACHEOP(ptxParseData parseData);

int get_NC(ptxParseData parseData);
String getNCAsString(int nc);
String get_strNC(ptxParseData parseData);

int get_ORDER(ptxParseData parseData);
String getORDERAsString(stdMap_t* deobfuscatedStringMapPtr, int order);
String get_strORDER(ptxParseData parseData);

int get_RAND(ptxParseData parseData);
String getRANDAsString(stdMap_t* deobfuscatedStringMapPtr, int rand);
String get_strRAND(ptxParseData parseData);

int get_ROUND(ptxParseData parseData);
String getROUNDAsString(int round);
String get_strROUND(ptxParseData parseData);

int get_TESTP(ptxParseData parseData);
String getTESTPAsString(int testp);
String get_strTESTP(ptxParseData parseData);

int get_FLOW(ptxParseData parseData);
String getFLOWAsString(int flow);
String get_strFLOW(ptxParseData parseData);

int get_BRANCH(ptxParseData parseData);
String getBRANCHAsString(stdMap_t* deobfuscatedStringMapPtr, int branch);
String get_strBRANCH(ptxParseData parseData);

int get_TEXTURE(ptxParseData parseData);
String getTEXTUREAsString(int texture);
String get_strTEXTURE(ptxParseData parseData);
String get_str_TEXTURE(ptxParseData parseData);
String get_strARG_VECTOR(ptxParseData parseData, uInt instrArgid);

int get_TENSORDIM(ptxParseData parseData);
String getTENSORDIMAsString(int tensorDim);
String get_strTENSORDIM(ptxParseData parseData);

int get_IM2COL(ptxParseData parseData);
String getIM2COLAsString(stdMap_t* deobfuscatedStringMapPtr, int im2col);
String get_strIM2COL(ptxParseData parseData);

int get_PACKEDOFF(ptxParseData parseData);
String getPACKEDOFFAsString(stdMap_t* deobfuscatedStringMapPtr, int packedOff);
String get_strPACKEDOFF(ptxParseData parseData);

int get_MULTICAST(ptxParseData parseData);
String getMULTICASTAsString(stdMap_t* deobfuscatedStringMapPtr, int multicast);
String get_strMULTICAST(ptxParseData parseData);

int get_FOOTPRINT(ptxParseData parseData);
String getFOOTPRINTAsString(stdMap_t* deobfuscatedStringMapPtr, int footprint);
String get_strFOOTPRINT(ptxParseData parseData);

int get_MBARRIER(ptxParseData parseData);
String getMBARRIERAsString(int mbarrier);
String get_strMBARRIER(ptxParseData parseData);

int get_COARSE(ptxParseData parseData);
String getCOARSEAsString(stdMap_t* deobfuscatedStringMapPtr, int coarse);
String get_strCOARSE(ptxParseData parseData);

int get_COMPONENT(ptxParseData parseData);
String getCOMPONENTAsString(int component);
String get_strCOMPONENT(ptxParseData parseData);

int get_QUERY(ptxParseData parseData);
String getQUERYAsString(int query);
String get_strQUERY(ptxParseData parseData);

int get_CLAMP(ptxParseData parseData);
String getCLAMPAsString(int clamp);
String get_strCLAMP(ptxParseData parseData);

int get_SHR(ptxParseData parseData);
String getSHRAsString(int shr);
String get_strSHR(ptxParseData parseData);

int get_VMAD(ptxParseData parseData);
String getVMADAsString(int vmad);
String get_strVMAD(ptxParseData parseData);

int get_PRMT(ptxParseData parseData);
String getPRMTAsString(int prmt);
String get_strPRMT(ptxParseData parseData);

int get_SHFL(ptxParseData parseData);
String get_strSHFL(ptxParseData parseData);
String getSHFLAsString(int shfl);
String get_str_SHFL(ptxParseData parseData);

int get_ENDIS(ptxParseData parseData);
String getENDISAsString(stdMap_t* deobfuscatedStringMapPtr, int endis);
String get_strENDIS(ptxParseData parseData);

int get_UNIFORM(ptxParseData parseData);
String getUNIFORMAsString(int uniform);
String get_strUNIFORM(ptxParseData parseData);

int get_VECTOR(ptxParseData parseData);
String getVECTORAsString(int vector);
String get_strVECTOR(ptxParseData parseData);

int get_VOTE(ptxParseData parseData);
String get_strVOTE(ptxParseData parseData);
String getVOTEAsString(int vote);
String get_str_VOTE(ptxParseData parseData);

int get_TYPES(ptxParseData parseData, int index);
String get_strTYPES(ptxParseData parseData);
String get_strTYPE(ptxParseData parseData, int index);
String get_str_TYPE(ptxParseData parseData, int index);

String getTYPEAsString(stdMap_t* deobfuscatedStringMapPtr, ptxType type);
String getTypeEnumAsString(stdMap_t* deobfuscatedStringMapPtr, ptxTypeKind type);

int get_GUARD(ptxParseData parseData);
String get_strGUARD(ptxParseData parseData);

int get_PRED(ptxParseData parseData);
String get_strPRED(ptxParseData parseData);
String get_strPRED_NEG(ptxParseData parseData);
String get_strArg(ptxParseData parseData, int index);
int get_HAS_SINK_DEST(ptxParseData parseData);

int get_NUMARGS(ptxParseData parseData);

int get_PTX_VERSION(ptxParseData parseData);

int get_PTX_TARGET(ptxParseData parseData);

int get_PREDICATE_OUTPUT(ptxParseData parseData);
String get_strPREDICATE_OUTPUT(ptxParseData parseData);
String get_strOPTIONAL_PREDICATE_OUTPUT(ptxParseData parseData);
String get_strIS_PREDICATE_OUTPUT(ptxParseData parseData);

int get_MacroElw(ptxParseData parseData, macroElwVar evar);
Bool isMacroElwEqual(ptxParseData parseData, macroElwVar evar, String str);

String get_strADDR_BASE(ptxParseData parseData, int index);
String get_strADDR_OFFSET(ptxParseData parseData, int index);
int get_IsSourceAddrArgInRegisterSpace(ptxParseData parseData);
int get_IsSourceAddrArgI32(ptxParseData parseData);
int get_IsDestAddrArgI32(ptxParseData parseData);
int get_IsDestAddrArgInRegisterSpace(ptxParseData parseData);
int get_IsWiderSrcForCvt(ptxParseData parseData);
int get_IsWiderDstForCvt(ptxParseData parseData);

String get_strVectorComponent_DST(ptxParseData parseData, int index);
String get_strWMMAStoreValue(ptxParseData parseData, int index);
String get_strWMMAVectorComponent_SRCA(ptxParseData parseData, int index);
String get_strWMMAVectorComponent_SRCB(ptxParseData parseData, int index);
String get_strWMMAVectorComponent_SRCC(ptxParseData parseData, int index);
int get_IsVabsdiff4NativelySupported(ptxParseData parseData);

int get_IS_TEX_INSTR_EMULATED(ptxParseData parseData);
int get_IS_SYSCALL_req(ptxParseData parseData);
int get_IS_TEXMODE_INDEPENDENT(ptxParseData parseData);
int get_HAS_OFFSET_ARG(ptxParseData parseData);
int get_HAS_DEPTH_COMPARE_ARG(ptxParseData parseData);
String get_strIS_TEXMODE_INDEPENDENT(ptxParseData parseData);
String get_strHAS_OFFSET_ARG(ptxParseData parseData);
String get_strHAS_DEPTH_COMPARE_ARG(ptxParseData parseData);
int get_BAR_IMM(ptxParseData parseData);
String get_strBAR_IMM(ptxParseData parseData);

String get_strVideoSelector_2SIMD(ptxParseData parseData, int index);
String get_strVideoSelector_4SIMD(ptxParseData parseData, int index);
String get_strVideoOperand(ptxParseData parseData, int index);
int    get_VIDEOSELECTOR(ptxParseData parseData, int index, int subindex, int simdWidth);
String get_strSIMD2PermuteControl(ptxParseData parseData, int index);

String get_strIfSIMD2DestHasByte(ptxParseData parseData, int byteno);
String get_strIfSIMD4DestHasByte(ptxParseData parseData, int byteno);

String get_strSIMD4PermuteControl(ptxParseData parseData, int index);
String get_strVideoPermCtrlToSelect(ptxParseData parseData, int index);
String get_strVideoPermCtrlToSelectAndRightShift(ptxParseData parseData, int index);
void scheduleMacroUtilFuncForParsing(String name, ptxParsingState parseState);
Bool getPendingMacroUtilFuncList(stdList_t* funcList, ptxParsingState parseState);
void initMacroUtilFuncParseState(ptxParsingState parseState);

int get_ABS(ptxParseData parseData);
String getABSAsString(int ABS);
String get_strABS(ptxParseData parseData);

int get_CACHEHINT(ptxParseData parseData);
String getCACHEHINTAsString(int CACHEHINT);
String get_strCACHEHINT(ptxParseData parseData);

int get_HITPRIORITY(ptxParseData parseData);
String getHITPRIORITYAsString(int HITPRIORITY);
String get_strHITPRIORITY(ptxParseData parseData);

int get_MISSPRIORITY(ptxParseData parseData);
String getMISSPRIORITYAsString(int MISSPRIORITY);
String get_strMISSPRIORITY(ptxParseData parseData);

String get_strENUM_FOR_HITPRIORITY(ptxParseData parseData);
String get_strENUM_FOR_MISSPRIORITY(ptxParseData parseData);

int get_PROXYKIND(ptxParseData parseData);
String getPROXYKINDAsString(int PROXYKIND);
String get_strPROXYKIND(ptxParseData parseData);


/* ----------  Inline-Function Utilities ---------- */

String get_strIArg(ptxParseData parseData, int index);
String get_strOArg(ptxParseData parseData, int index);

int getConstantValueOfInputArg(ptxParseData parseData, uInt n);
ptxExpressionKind getExpressionKindForArg(ptxParseData parseData, uInt n, Bool isRetArg);

String __deObfuscate(stdMap_t* deobfuscatedStringMapPtr, cString encodeName);
#ifdef __cplusplus
}
#endif

#endif
