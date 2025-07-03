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

#include "ptxConstructors.h"
#include "ptxMacroUtils.h"
#include "ptxIR.h"
#include "ptxObfuscatedIRdefs.h"
#include "ptxInstructions.h"
#include "ptxparseMessageDefs.h"
#include "ctArch.h"

// OPTIX_HAND_EDIT: Patched parser with this extra include to get ptxMsgTooManyArgsInlineFunc. Should not 
// need to touch files from the parser, but no idea where this symbol would come from otherwise...
#include "ptxparseMessageDefs.h"






#ifndef ALEN
#define ALEN(X) ((int) ((sizeof(X) / sizeof((X)[0]))))
#endif

/* ****************** API Functions for macro processor *************************** */

static ptxTypeKind getPtxTypeEnum(ptxType type)
{
    int t;

    if(isB8(type)) {
        t = ptxTypeB8;
    } else if(isB16(type)) {
        t = ptxTypeB16;
    } else if(isB32(type)) {
        t = ptxTypeB32;
    } else if(isB64(type)) {
        t = ptxTypeB64;
    } else if(isB128(type)) {
        t = ptxTypeB128;
    } else if(isS8(type)) {
        t = ptxTypeS8;
    } else if(isS16(type)) {
        t = ptxTypeS16;
    } else if(isS32(type)) {
        t = ptxTypeS32;
    } else if(isS64(type)) {
        t = ptxTypeS64;
    } else if(isU8(type)) {
        t = ptxTypeU8;
    } else if(isU16(type)) {
        t = ptxTypeU16;
    } else if(isU32(type)) {
        t = ptxTypeU32;
    } else if(isU64(type)) {
        t = ptxTypeU64;
    } else if(isE4M3x2(type)) {
        t = ptxTypeE4M3x2;
    } else if(isE5M2x2(type)) {
        t = ptxTypeE5M2x2;
    } else if(isF16(type)) {
        t = ptxTypeF16;
    } else if(isF16x2(type)) {
        t = ptxTypeF16x2;
    } else if(isBF16(type)) {
        t = ptxTypeBF16;
    } else if(isBF16x2(type)) {
        t = ptxTypeBF16x2;
    } else if(isF32(type)) {
        t = ptxTypeF32;
    } else if(isF64(type)) {
        t = ptxTypeF64;
    } else if (isPRED(type)) {
        t = ptxTypePred;
    } else {
        stdASSERT(False, ("Unexpected type"));
        t = ptxNOTYPE;
    }

    return t;
}

int get_POSTOP(ptxParseData parseData) { return parseData->postop; }
String getPOSTOPAsString(int postop)
{ 
    static char *const strPOSTOP[] = { PTX_OPERATOR_TABLE(GET_NAME) };
    return strPOSTOP[postop]; 
}
String get_strPOSTOP(ptxParseData parseData) { return getPOSTOPAsString(get_POSTOP(parseData)); }
String get_str_POSTOP(ptxParseData parseData)
{
    static char *const strPOSTOP[] = { PTX_OPERATOR_TABLE(GET_NAME) };
    String temp;
    temp = stdMALLOC(strlen(strPOSTOP[get_POSTOP(parseData)]) + 1);
    strcpy(temp, strPOSTOP[get_POSTOP(parseData)]);
    if (temp[0] != '.') {
        stdASSERT(False, ("Unexpected postop for instruction"));
        return temp;
    }
    temp[0] = '_';
    return temp;
}

String get_strBOP(ptxParseData parseData) { return get_strPOSTOP(parseData); }

int get_COMPARE(ptxParseData parseData) { return parseData->cmp; }
String getCOMPAREAsString(int cmp)
{
    static char *const strCOMPARE[] = { PTX_COMPARE_TABLE(GET_NAME) };
    return strCOMPARE[cmp];
}
String get_strCOMPARE(ptxParseData parseData) { return getCOMPAREAsString(get_COMPARE(parseData)); }

int get_APRX(ptxParseData parseData) { return parseData->modifiers.APRX; }
String getAPRXAsString(int aprx)
{
    static char *const strAPRX[] = { PTX_APRXMOD_TABLE(GET_NAME) };
    return strAPRX[aprx];
}
String get_strAPRX(ptxParseData parseData) { return getAPRXAsString(get_APRX(parseData)); }

int get_RELU(ptxParseData parseData) { return parseData->modifiers.RELU; }
String getRELUAsString(int relu)
{
    static char *const strRELU[] = { PTX_RELUMOD_TABLE(GET_NAME) };
    return strRELU[relu];
}
String get_strRELU(ptxParseData parseData) { return getRELUAsString(get_RELU(parseData)); }

int get_NANMODE(ptxParseData parseData) { return parseData->modifiers.NANMODE; }
String getNANMODEAsString(int nan)
{
    static char *const strNANMODE[] = { PTX_NANMODEMOD_TABLE(GET_NAME) };
    return strNANMODE[nan];
}
String get_strNANMODE(ptxParseData parseData) { return getNANMODEAsString(get_NANMODE(parseData)); }

int get_XORSIGN(ptxParseData parseData) { return parseData->modifiers.XORSIGN; }
String getXORSIGNAsString(int xorsign)
{
    static char *const strXORSIGN[] = { PTX_XORSIGNMOD_TABLE(GET_NAME) };
    return strXORSIGN[xorsign];
}
String get_strXORSIGN(ptxParseData parseData) { return getXORSIGNAsString(get_XORSIGN(parseData)); }

int get_TRANSA(ptxParseData parseData) { return parseData->modifiers.TRANSA; }
String getTRANSAAsString(stdMap_t* deobfuscatedStringMapPtr, int transa)
{
    static char *const strTRANSA[] = { PTX_TRANSAMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strTRANSA[transa]);
}
String get_strTRANSA(ptxParseData parseData) { return getTRANSAAsString(parseData->deobfuscatedStringMapPtr, get_TRANSA(parseData)); }

int get_NEGA(ptxParseData parseData) { return parseData->modifiers.NEGA; }
String getNEGAAsString(stdMap_t* deobfuscatedStringMapPtr, int nega)
{
    static char *const strNEGA[] = { PTX_NEGAMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strNEGA[nega]);
}
String get_strNEGA(ptxParseData parseData) { return getNEGAAsString(parseData->deobfuscatedStringMapPtr, get_NEGA(parseData)); }

int get_NEGB(ptxParseData parseData) { return parseData->modifiers.NEGB; }
String getNEGBAsString(stdMap_t* deobfuscatedStringMapPtr, int negb)
{
    static char *const strNEGB[] = { PTX_NEGBMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strNEGB[negb]);
}
String get_strNEGB(ptxParseData parseData) { return getNEGBAsString(parseData->deobfuscatedStringMapPtr, get_NEGB(parseData)); }

int get_TRANSB(ptxParseData parseData) { return parseData->modifiers.TRANSB; }
String getTRANSBAsString(stdMap_t* deobfuscatedStringMapPtr, int transb)
{
    static char *const strTRANSB[] = { PTX_TRANSBMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strTRANSB[transb]);
}
String get_strTRANSB(ptxParseData parseData) { return getTRANSBAsString(parseData->deobfuscatedStringMapPtr, get_TRANSB(parseData)); }

int get_IGNOREC(ptxParseData parseData) { return parseData->modifiers.IGNOREC; }
String getIGNORECAsString(stdMap_t* deobfuscatedStringMapPtr, int ignorec)
{
    static char *const strIGNOREC[] = { PTX_IGNORECMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strIGNOREC[ignorec]);
}
String get_strIGNOREC(ptxParseData parseData) { return getIGNORECAsString(parseData->deobfuscatedStringMapPtr, get_IGNOREC(parseData)); }

int get_ADDRTYPE(ptxParseData parseData) { return parseData->modifiers.ADDRTYPE; }
String getADDRTYPEAsString(stdMap_t* deobfuscatedStringMapPtr, int addrtype)
{
    static char *const strADDRTYPE[] = { PTX_ADDRTYPEMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strADDRTYPE[addrtype]);
}

String get_strADDRTYPE(ptxParseData parseData) { return getADDRTYPEAsString(parseData->deobfuscatedStringMapPtr, get_ADDRTYPE(parseData)); }

int get_FTZ(ptxParseData parseData) { return parseData->modifiers.FTZ; }
String getFTZAsString(int ftz)
{
    static char *const strFTZ[] = { PTX_FTZMOD_TABLE(GET_NAME) };
    return strFTZ[ftz];
}
String get_strFTZ(ptxParseData parseData) { return getFTZAsString(get_FTZ(parseData)); }

int get_NOFTZ(ptxParseData parseData) { return parseData->modifiers.NOFTZ; }
String getNOFTZAsString(int noftz)
{
    static char *const strNOFTZ[] = { PTX_NOFTZMOD_TABLE(GET_NAME) };
    return strNOFTZ[noftz];
}
String get_strNOFTZ(ptxParseData parseData) { return getNOFTZAsString(get_NOFTZ(parseData)); }

int get_OOB(ptxParseData parseData) { return parseData->modifiers.OOB; }
String getOOBAsString(stdMap_t* deobfuscatedStringMapPtr, int oob)
{
    static char *const strOOB[] = { PTX_OOBMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strOOB[oob]);
}
String get_strOOB(ptxParseData parseData) { return getOOBAsString(parseData->deobfuscatedStringMapPtr, get_OOB(parseData)); }

int get_SAT(ptxParseData parseData) { return parseData->modifiers.SAT; }
String get_str_SAT(ptxParseData parseData)
{
    static char *const strSAT[] = { PTX_SATMOD_TABLE_RAW(GET_NAME, "_") };
    return strSAT[get_SAT(parseData)];
}
String getSATAsString(int sat)
{
    static char *const strSAT[] = { PTX_SATMOD_TABLE(GET_NAME) };
    return strSAT[sat];
}
String get_strSAT(ptxParseData parseData) { return getSATAsString(get_SAT(parseData)); }

int get_SATF(ptxParseData parseData) { return parseData->modifiers.SATF; }
String get_str_SATF(ptxParseData parseData)
{
    static char *const strSATF[] = { PTX_SATFMOD_TABLE_RAW(GET_NAME, "_") };
    return strSATF[get_SATF(parseData)];
}
String getSATFAsString(int satf)
{
    static char *const strSATF[] = { PTX_SATFMOD_TABLE(GET_NAME) };
    return strSATF[satf];
}
String get_strSATF(ptxParseData parseData) { return getSATFAsString(get_SATF(parseData)); }

int get_SYNC(ptxParseData parseData) { return parseData->modifiers.SYNC; }
String getSYNCAsString(int sync)
{
    static char *const strSYNC[] = { PTX_SYNCMOD_TABLE(GET_NAME) };
    return strSYNC[sync];
}
String get_strSYNC(ptxParseData parseData) { return getSYNCAsString(get_SYNC(parseData)); }

int get_NOINC(ptxParseData parseData) { return parseData->modifiers.NOINC; }
String getNOINCAsString(int noinc)
{
    static char *const strNOINC[] = { PTX_NOINCMOD_TABLE(GET_NAME) };
    return strNOINC[noinc];
}
String get_strNOINC(ptxParseData parseData) { return getNOINCAsString(get_NOINC(parseData)); }

int get_NOCOMPLETE(ptxParseData parseData) { return parseData->modifiers.NOCOMPLETE; }
String getNOCOMPLETEAsString(int nocomplete)
{
    static char *const strNOCOMPLETE[] = { PTX_NOCOMPLETEMOD_TABLE(GET_NAME) };
    return strNOCOMPLETE[nocomplete];
}
String get_strNOCOMPLETE(ptxParseData parseData) { return getNOCOMPLETEAsString(get_NOCOMPLETE(parseData)); }

int get_NOSLEEP(ptxParseData parseData) { return parseData->modifiers.NOSLEEP; }
String getNOSLEEPAsString(stdMap_t* deobfuscatedStringMapPtr, int nosleep)
{
    static char *const strNOSLEEP[] = { PTX_NOSLEEPMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strNOSLEEP[nosleep]);
}
String get_strNOSLEEP(ptxParseData parseData) { return getNOSLEEPAsString(parseData->deobfuscatedStringMapPtr, get_NOSLEEP(parseData)); }

int get_SHAREDSCOPE(ptxParseData parseData) { return parseData->modifiers.SHAREDSCOPE; }
String getSHAREDSCOPEAsString(stdMap_t* deobfuscatedStringMapPtr, int mod)
{
    static char *const strSHAREDSCOPE[] = { PTX_SHAREDSCOPEMOD_TABLE(GET_NAME) };
    stdASSERT(mod < ALEN(strSHAREDSCOPE), ("Modifier value out of range"));
    return __deObfuscate(deobfuscatedStringMapPtr, strSHAREDSCOPE[mod]);
}
String get_strSHAREDSCOPE(ptxParseData parseData) { return getSHAREDSCOPEAsString(parseData->deobfuscatedStringMapPtr, get_SHAREDSCOPE(parseData)); }

String get_strOPTIONAL_CP_MBARRIER_ARVCNT_ARG(ptxParseData parseData)
{
    if (parseData->nrArgs == 2)
        return stdCONCATSTRING(" , ", parseData->arg[1]);

    return "";
}

int get_NUM(ptxParseData parseData) { return parseData->modifiers.NUM; }
String get_strNUM(ptxParseData parseData)
{
    return getNUMAsString(get_NUM(parseData));
}
String getNUMAsString(int num)
{
    static char *const strNUM[] = { PTX_NUMMOD_TABLE(GET_NAME) };
    return strNUM[num];
}

int get_GROUP(ptxParseData parseData) { return parseData->modifiers.GROUP; }
String get_strGROUP(ptxParseData parseData)
{
    return getGROUPAsString(parseData->deobfuscatedStringMapPtr, get_GROUP(parseData));
}
String getGROUPAsString(stdMap_t* deobfuscatedStringMapPtr, int grp)
{
    static char *const strGrp[] = { PTX_GROUPMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strGrp[grp]);
}

int get_SEQ(ptxParseData parseData) { return parseData->modifiers.SEQ; }
String get_strSEQ(ptxParseData parseData)
{
    return getSEQAsString(parseData->deobfuscatedStringMapPtr, get_SEQ(parseData));
}
String getSEQAsString(stdMap_t* deobfuscatedStringMapPtr, int mod)
{
    static char *const strSEQ[] = { PTX_SEQMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strSEQ[mod]);
}

int get_TTUSLOT(ptxParseData parseData) { return parseData->modifiers.TTUSLOT; }
String getTTUSLOTAsString(stdMap_t* deobfuscatedStringMapPtr, int ttuslot)
{
    static char *const strTTUSLOT[] = { PTX_TTUSLOTMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strTTUSLOT[ttuslot]);
}
String get_strTTUSLOT(ptxParseData parseData) { return getTTUSLOTAsString(parseData->deobfuscatedStringMapPtr, get_TTUSLOT(parseData)); }

int get_TTUMOD(ptxParseData parseData) { return parseData->modifiers.TTU; }
String getTTUMODAsString(stdMap_t* deobfuscatedStringMapPtr, int ttu)
{
    static char *const strTTU[] = { PTX_TTUMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strTTU[ttu]);
}
String get_strTTUMOD(ptxParseData parseData) { return getTTUMODAsString(parseData->deobfuscatedStringMapPtr, get_TTUMOD(parseData)); }

String getTYPEMODAsStringRaw(stdMap_t* deobfuscatedStringMapPtr, int typeMod)
{
    static char *const strTYPE[] = { PTX_TYPEMOD_TABLE(GET_NAME) };
    stdASSERT(typeMod < ALEN(strTYPE), ("Modifier value out of range"));
    if (typeMod == ptxTYPE_b4_MOD) {
        return __deObfuscate(deobfuscatedStringMapPtr, strTYPE[typeMod]);
    }
    return strTYPE[typeMod];
}

String getTYPEMODAsString(stdMap_t* deobfuscatedStringMapPtr, int typeMod)
{
    return getTYPEMODAsStringRaw(deobfuscatedStringMapPtr, typeMod);
}

int get_COMPRESSED_FORMAT(ptxParseData parseData) { return parseData->modifiers.COMPRESSED_FORMAT; }
String getCOMPRESSED_FORMATAsString(stdMap_t* deobfuscatedStringMapPtr, int COMPRESSED_FORMAT)
{
    return getTYPEMODAsString(deobfuscatedStringMapPtr, COMPRESSED_FORMAT);
}
String get_strCOMPRESSED_FORMAT(ptxParseData parseData)
{
    return getCOMPRESSED_FORMATAsString(parseData->deobfuscatedStringMapPtr, get_COMPRESSED_FORMAT(parseData));
}

int get_EXPANDED_FORMAT(ptxParseData parseData) { return parseData->modifiers.EXPANDED_FORMAT; }
String getEXPANDED_FORMATAsString(stdMap_t* deobfuscatedStringMapPtr, int EXPANDED_FORMAT)
{
    return getTYPEMODAsString(deobfuscatedStringMapPtr, EXPANDED_FORMAT);
}
String get_strEXPANDED_FORMAT(ptxParseData parseData)
{
    return getEXPANDED_FORMATAsString(parseData->deobfuscatedStringMapPtr, get_EXPANDED_FORMAT(parseData));
}

int get_ATYPE(ptxParseData parseData) { return parseData->modifiers.ATYPE; }
String getATYPEAsString(stdMap_t* deobfuscatedStringMapPtr, int ATYPE)
{
    return getTYPEMODAsString(deobfuscatedStringMapPtr, ATYPE);
}
String get_strATYPE(ptxParseData parseData)
{
    return getATYPEAsString(parseData->deobfuscatedStringMapPtr, get_ATYPE(parseData));
}

int get_ABS(ptxParseData parseData) { return parseData->modifiers.ABS; }
String get_strABS(ptxParseData parseData)
{
    return getABSAsString(get_ABS(parseData));
}
String getABSAsString(int mod)
{
    static char *const strABS[] = { PTX_ABSMOD_TABLE(GET_NAME) };
    return strABS[mod];
}

int get_CACHEHINT(ptxParseData parseData) { return parseData->modifiers.CACHEHINT; }
String get_strCACHEHINT(ptxParseData parseData)
{
    return getCACHEHINTAsString(get_CACHEHINT(parseData));
}
String getCACHEHINTAsString(int mod)
{
    static char *const strCACHEHINT[] = { PTX_CACHEHINTMOD_TABLE(GET_NAME) };
    return strCACHEHINT[mod];
}

int get_HITPRIORITY(ptxParseData parseData) { return parseData->modifiers.HITPRIORITY; }
String get_strHITPRIORITY(ptxParseData parseData)
{
    return getHITPRIORITYAsString(get_HITPRIORITY(parseData));
}
String getHITPRIORITYAsString(int mod)
{
    return get_EVICTPRIORITYAsString(mod);
}

int get_MISSPRIORITY(ptxParseData parseData) { return parseData->modifiers.MISSPRIORITY; }
String get_strMISSPRIORITY(ptxParseData parseData)
{
    return getMISSPRIORITYAsString(get_MISSPRIORITY(parseData));
}
String getMISSPRIORITYAsString(int mod)
{
    return get_EVICTPRIORITYAsString(mod);
}

String get_strENUM_FOR_HITPRIORITY(ptxParseData parseData)
{
    int mod = get_HITPRIORITY(parseData);

    if (mod == ptxEVICTFIRST_MOD) {
        return "1";
    }
    if (mod == ptxEVICTLAST_MOD) {
        return "2";
    }
    if (mod == ptxEVICTNORMAL_MOD) {
        return "3";
    }
    if (mod == ptxEVICTUNCHANGED_MOD) {
        return "0";
    }

    stdASSERT( False, ("Unexpected primary_priority") );
    return "0";
}
String get_strENUM_FOR_MISSPRIORITY(ptxParseData parseData)
{
    int mod = get_MISSPRIORITY(parseData);

    if (mod == ptxEVICTFIRST_MOD) {
        return "1";
    }
    if (mod == ptxEVICTUNCHANGED_MOD || mod == ptxNOEVICTPRIORITY_MOD) {
        return "0";
    }

    stdASSERT( False, ("Unexpected secondary_priority") );
    return "0";
}

int get_PROXYKIND(ptxParseData parseData) { return parseData->modifiers.PROXYKIND; }
String get_strPROXYKIND(ptxParseData parseData)
{
    return getPROXYKINDAsString(get_PROXYKIND(parseData));
}
String getPROXYKINDAsString(int mod)
{
    static char *const strPROXYKIND[] = { PTX_PROXYKINDMOD_TABLE(GET_NAME) };
    return strPROXYKIND[mod];
}

// NOTE: Since __deObfuscate() may return the same deobfuscated string for
//       every invocation with the same input string, the returned string needs
//       to be copied before modifying.
String get_str_ATYPE(ptxParseData parseData) {
    String typeModStr = stdCOPYSTRING(get_strATYPE(parseData));
    if (typeModStr[0] == '.') {
        typeModStr[0] = '_';
    }
    return typeModStr;
}

int get_MMA_OP(ptxParseData parseData, int op) { return op == 1 ? parseData->modifiers.MMA_OP1 :
                                          parseData->modifiers.MMA_OP2; }
String getMMA_OPAsString(int mma_op)
{
    static char *const mmaOp[] = { PTX_MMA_OPMOD_TABLE(GET_NAME) };
    return mmaOp[mma_op];
}
String get_strMMA_OP(ptxParseData parseData, int op)
{
    return getMMA_OPAsString(get_MMA_OP(parseData, op));
}

int get_BTYPE(ptxParseData parseData) { return parseData->modifiers.BTYPE; }
String getBTYPEAsString(stdMap_t* deobfuscatedStringMapPtr, int BTYPE)
{
    return getTYPEMODAsString(deobfuscatedStringMapPtr, BTYPE);
}
String get_strBTYPE(ptxParseData parseData)
{
    return getBTYPEAsString(parseData->deobfuscatedStringMapPtr, get_BTYPE(parseData));
}

int get_IS_WMMA_TF32_MMA(ptxParseData parseData)
{
    stdASSERT( ptxIsWMMAInstr(parseData->instruction_tcode), ("wmma instruction expected") );
    return ((get_ATYPE(parseData) == ptxTYPE_TF32_MOD) &&
            (get_BTYPE(parseData) == ptxTYPE_TF32_MOD));
}

int get_IS_NON_STANDARD_MMA(ptxParseData parseData)
{
    stdASSERT( parseData->instruction_tcode == ptx_mma_Instr || parseData->instruction_tcode == ptx__mma_Instr, ("mma instruction expected") );
    return ((get_ATYPE(parseData) == ptxTYPE_BF16_MOD) ||
            (get_ATYPE(parseData) == ptxTYPE_TF32_MOD)) &&
            (get_ATYPE(parseData) == get_BTYPE(parseData));
}

int get_IS_SUB_BYTE_MMA(ptxParseData parseData)
{
    stdASSERT( parseData->instruction_tcode == ptx_mma_Instr || parseData->instruction_tcode == ptx__mma_Instr, ("mma instruction expected") );
    return (get_ATYPE(parseData) == ptxTYPE_u4_MOD || get_ATYPE(parseData) == ptxTYPE_s4_MOD) &&
           (get_BTYPE(parseData) == ptxTYPE_u4_MOD || get_BTYPE(parseData) == ptxTYPE_s4_MOD);
}

// This routine is used in MACRO processing to determine sub_byte_wmma
int get_IS_SUB_BYTE_WMMA_LOAD(ptxParseData parseData)
{
    stdASSERT( ptxIsWMMAInstr(parseData->instruction_tcode), ("wmma instruction expected") );
    return (get_ATYPE(parseData) == ptxTYPE_u4_MOD || get_ATYPE(parseData) == ptxTYPE_s4_MOD) ||
           (get_BTYPE(parseData) == ptxTYPE_u4_MOD || get_BTYPE(parseData) == ptxTYPE_s4_MOD);
}

int get_IS_SUB_BYTE_WMMA_MMA(ptxParseData parseData)
{
    stdASSERT( ptxIsWMMAInstr(parseData->instruction_tcode), ("wmma instruction expected") );
    return (get_ATYPE(parseData) == ptxTYPE_u4_MOD && get_BTYPE(parseData) == ptxTYPE_u4_MOD) ||
           (get_ATYPE(parseData) == ptxTYPE_s4_MOD && get_BTYPE(parseData) == ptxTYPE_s4_MOD);
}

int get_IS_BIT_WMMA_LOAD(ptxParseData parseData)
{
    stdASSERT( ptxIsWMMAInstr(parseData->instruction_tcode), ("wmma instruction expected") );
    return (get_ATYPE(parseData) == ptxTYPE_b1_MOD || get_BTYPE(parseData) == ptxTYPE_b1_MOD);
}

int get_IS_BIT_WMMA_MMA(ptxParseData parseData)
{
    stdASSERT( ptxIsWMMAInstr(parseData->instruction_tcode), ("wmma instruction expected") );
    return (get_ATYPE(parseData) == ptxTYPE_b1_MOD && get_BTYPE(parseData) == ptxTYPE_b1_MOD);
}

int get_IS_NON_STANDARD_FP_WMMA_LOAD(ptxParseData parseData)
{
    stdASSERT( ptxIsWMMAInstr(parseData->instruction_tcode), ("wmma instruction expected") );
    return (get_ATYPE(parseData) == ptxTYPE_BF16_MOD);
}

int get_IS_NON_STANDARD_FP_WMMA_MMA(ptxParseData parseData)
{
    stdASSERT( ptxIsWMMAInstr(parseData->instruction_tcode), ("wmma instruction expected") );
    return (get_ATYPE(parseData) == ptxTYPE_BF16_MOD) &&
           (get_BTYPE(parseData) == ptxTYPE_BF16_MOD);
}

int get_IS_ATYPE_TF32(ptxParseData parseData)
{
    return (get_ATYPE(parseData) == ptxTYPE_TF32_MOD);
}
int get_IS_ATYPE_BF16(ptxParseData parseData)
{
    return (get_ATYPE(parseData) == ptxTYPE_BF16_MOD);
}
int get_IS_ATYPE_BF16x2(ptxParseData parseData)
{
    return (get_ATYPE(parseData) == ptxTYPE_BF16x2_MOD);
}
int get_IS_BTYPE_BF16(ptxParseData parseData)
{
    return (get_BTYPE(parseData) == ptxTYPE_BF16_MOD);
}

String get_strWMMA_A_TYPE(ptxParseData parseData)
{
    switch (parseData->modifiers.SHAPE) {
    case ptxSHAPE_080832_MOD:
    case ptxSHAPE_0808128_MOD:
        return get_strATYPE(parseData);
    case ptxSHAPE_161616_MOD:
    case ptxSHAPE_320816_MOD:
    case ptxSHAPE_083216_MOD:
        if (parseData->nrInstrTypes == 0) {
        // Non standard FP WMMA is present
            return  get_strATYPE(parseData);
        }
    // When only 2 types are specified WMMA ATYPE is implicitly assumed
        return parseData->nrInstrTypes  == 4 ? get_strTYPE(parseData, 1) : "";
    default:
        return "";
    }
}

String get_strWMMA_B_TYPE(ptxParseData parseData)
{
    switch (parseData->modifiers.SHAPE) {
    case ptxSHAPE_080832_MOD:
    case ptxSHAPE_0808128_MOD:
        return get_strATYPE(parseData);
    case ptxSHAPE_161616_MOD:
    case ptxSHAPE_320816_MOD:
    case ptxSHAPE_083216_MOD:
    // When only 2 types are specified WMMA BTYPE is implicitly assumed
        return parseData->nrInstrTypes == 4 ? get_strTYPE(parseData, 2) : "";
    default:
        return "";
    }
}

int get_TRANS(ptxParseData parseData) { return parseData->modifiers.TRANS; }
String getTRANSAsString(int trans)
{
    static char *const strTRANS[] = { PTX_TRANSMOD_TABLE(GET_NAME) };
    return strTRANS[trans];
}
String get_strTRANS(ptxParseData parseData) { return getTRANSAsString(get_TRANS(parseData)); }

int get_EXPAND(ptxParseData parseData) { return parseData->modifiers.EXPAND; }
String getEXPANDAsString(stdMap_t* deobfuscatedStringMapPtr, int expand)
{
    static char *const strEXPAND[] = { PTX_EXPANDMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strEXPAND[expand]);
}
String get_strEXPAND(ptxParseData parseData) { return getEXPANDAsString(parseData->deobfuscatedStringMapPtr, get_EXPAND(parseData)); }

int get_EXCLUSIVE(ptxParseData parseData) { return parseData->modifiers.EXCLUSIVE; }
String getEXCLUSIVEAsString(stdMap_t* deobfuscatedStringMapPtr, int mod)
{
    static char *const strEXCLUSIVE[] = { PTX_EXCLUSIVEMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strEXCLUSIVE[mod]);
}
String get_strEXCLUSIVE(ptxParseData parseData)
{
    return getEXCLUSIVEAsString(parseData->deobfuscatedStringMapPtr, get_EXCLUSIVE(parseData));
}

int get_BAR(ptxParseData parseData) { return parseData->modifiers.BAR; }
String getBARAsString(int bar)
{
    static char *const strBAR[] = { PTX_BARMOD_TABLE(GET_NAME) };
    stdASSERT(bar < ALEN(strBAR), ("Modifier value out of range"));
    return strBAR[bar];
}
String get_strBAR(ptxParseData parseData)
{
    return getBARAsString(get_BAR(parseData));
}

int get_LAYOUT(ptxParseData parseData) { return parseData->modifiers.ALAYOUT; }
int get_ALAYOUT(ptxParseData parseData) { return parseData->modifiers.ALAYOUT; }
int get_BLAYOUT(ptxParseData parseData) { return parseData->modifiers.BLAYOUT; }
String getLAYOUTAsString(int layout)
{
    static char *const strLAYOUT[] = { PTX_LAYOUTMOD_TABLE(GET_NAME) };
    stdASSERT(layout < ALEN(strLAYOUT), ("Modifier value out of range"));
    return strLAYOUT[layout];
}
String get_strLAYOUT(ptxParseData parseData) { return getLAYOUTAsString(get_LAYOUT(parseData)); }
String get_strALAYOUT(ptxParseData parseData) { return getLAYOUTAsString(get_ALAYOUT(parseData)); }
String get_strBLAYOUT(ptxParseData parseData) { return getLAYOUTAsString(get_BLAYOUT(parseData)); }

static String getLAYOUTAsEscapedString(int layout)
{
    static char *const strLAYOUT[] = { PTX_LAYOUTMOD_TABLE_RAW(GET_NAME, "_") };
    stdASSERT(layout < ALEN(strLAYOUT), ("Modifier value out of range"));
    return strLAYOUT[layout];
}
String get_str_LAYOUT(ptxParseData parseData) { return getLAYOUTAsEscapedString(get_LAYOUT(parseData)); }
String get_str_ALAYOUT(ptxParseData parseData) { return getLAYOUTAsEscapedString(get_ALAYOUT(parseData)); }
String get_str_BLAYOUT(ptxParseData parseData) { return getLAYOUTAsEscapedString(get_BLAYOUT(parseData)); }

String get_strLAYOUT_TRANS(ptxParseData parseData)
{
    int layout = get_LAYOUT(parseData);
    if (layout) {
        return (layout == ptxLAYOUT_ROW_MOD) ? getLAYOUTAsString(ptxLAYOUT_COL_MOD)
                                             : getLAYOUTAsString(ptxLAYOUT_ROW_MOD);
    }
    return "";
}

int get_DESC(ptxParseData parseData) { return parseData->modifiers.DESC; }
String get_strDESC(ptxParseData parseData)
{
    return getDESCAsString(parseData->deobfuscatedStringMapPtr, get_DESC(parseData));
}
String getDESCAsString(stdMap_t* deobfuscatedStringMapPtr, int desc)
{
    static char *const strDESC[] = { PTX_DESCMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strDESC[desc]);
}

// NOTE: Since __deObfuscate() may return the same deobfuscated string for
//       every invocation with the same input string, the returned string needs
//       to be copied before modifying.
String get_str_DESC(ptxParseData parseData)
{
    String descStr = stdCOPYSTRING(get_strDESC(parseData));

    if (descStr[0] == '\0') {
        return descStr;
    }
    // Similar to get_str_{SHAPE,TYPE}, get_strDESC(parseData) returns a copied desc string
    stdASSERT(descStr[0] == '.', ("Unexpected desc string"));
    descStr[0] = '_';
    return descStr;
}
int get_DESC_ARG(ptxParseData parseData)
{
    if (!ptxInstrSupportsMemDesc(parseData->instruction_tcode))
        return 0;

    return get_DESC(parseData);
}

String get_strDESC_ARG(ptxParseData parseData)
{
    stdString_t str = stringNEW();
    if (!get_DESC_ARG(parseData))
        return "";

    ptxPrintExpression(parseData->arguments[parseData->nrArgs - 1], str);
    return stringStripToBuf(str);
}

String get_strOPTIONAL_DESC_ARG(ptxParseData parseData)
{
    String descArg = get_strDESC_ARG(parseData);
    if (descArg[0] != '\0')
        return stdCONCATSTRING(" , ", descArg);

    return "";
}

String get_strARGS(ptxParseData parseData)
{
    stdString_t argStr = stringNEW();
    int i;
    for(i = 0; i < parseData->nrArgs; i++) {
        ptxPrintExpression(parseData->arguments[i], argStr);
        if (i != parseData->nrArgs - 1) {
            stringAddBuf(argStr, " , ");
        }
    }
    return stringStripToBuf(argStr);
}

// Sparsity related functions
String get_strSPARSITY(ptxParseData parseData) { return getSPARSITYAsString(get_SPARSITY(parseData)); }
String get_strSPFORMAT(ptxParseData parseData) { return getSPFORMATAsString(parseData->deobfuscatedStringMapPtr, get_SPFORMAT(parseData)); }
int get_SPFORMAT(ptxParseData parseData) { return parseData->modifiers.SPFORMAT; }
int get_SPARSITY(ptxParseData parseData) { return parseData->modifiers.SPARSITY; }

String getSPARSITYAsString(int s)
{
    static char *const strSPARSITY[] = { PTX_SPARSITYMOD_TABLE(GET_NAME) };
    stdASSERT(s < ALEN(strSPARSITY), ("Modifier value out of range"));
    return strSPARSITY[s];
}

String getSPFORMATAsString(stdMap_t* deobfuscatedStringMapPtr, int s)
{
    static char *const strSPFORMAT[] = { PTX_SPFORMATMOD_TABLE(GET_NAME) };
    stdASSERT(s < ALEN(strSPFORMAT), ("Modifier value out of range"));
    return __deObfuscate(deobfuscatedStringMapPtr, strSPFORMAT[s]);
}

int get_SHAPE(ptxParseData parseData) { return parseData->modifiers.SHAPE; }
String getSHAPEAsString(stdMap_t* deobfuscatedStringMapPtr, int shape)
{
    static char *const strSHAPE[] = { PTX_SHAPEMOD_TABLE(GET_NAME) };
    stdASSERT(shape < ALEN(strSHAPE), ("Modifier value out of range"));
    return __deObfuscate(deobfuscatedStringMapPtr, strSHAPE[shape]);
}
String get_strSHAPE(ptxParseData parseData)
{
    return getSHAPEAsString(parseData->deobfuscatedStringMapPtr, get_SHAPE(parseData));
}

// NOTE: Since __deObfuscate() may return the same deobfuscated string for
//       every invocation with the same input string, the returned string needs
//       to be copied before modifying.
String get_str_SHAPE(ptxParseData parseData)
{
    String shapeStr = stdCOPYSTRING(get_strSHAPE(parseData));
    // In original string first character is '.'
    // We need to manually replace it with '_' to generate _{shape} string
    stdASSERT(shapeStr[0] == '.', ("Unexpected shape string"));
    shapeStr[0] = '_';
    return shapeStr;
}

/*
|       | 8x32x16 | 32x8x16 | 16x16x16 | 8x8x32 | 8x8x128 |  8x8x4  |
|-------+---------+---------+----------|--------|---------|---------|
| a.row |      16 |      16 |       16 |     32 |     128 |       4 |
| a.col |       8 |      32 |       16 |     NA |      NA |       8 |
| b.row |      32 |       8 |       16 |     NA |      NA |       8 |
| b.col |      16 |      16 |       16 |     32 |     128 |       4 |
| c.row |      32 |       8 |       16 |      8 |       8 |       8 |
| c.col |       8 |      32 |       16 |      8 |       8 |       8 |
| d.row |      32 |       8 |       16 |      8 |       8 |       8 |
| d.col |       8 |      32 |       16 |      8 |       8 |       8 |
*/

// TODO: Rewrite this function to make more readable
String get_strLDM(ptxParseData parseData)
{

#if LWCFG(GLOBAL_ARCH_VOLTA) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_60)
    switch (parseData->modifiers.SHAPE) {
#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_63)
    case ptxSHAPE_080832_MOD:
    case ptxSHAPE_0808128_MOD:
#endif
#if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_70)
    case ptxSHAPE_080804_MOD:
    case ptxSHAPE_161608_MOD:
#endif
    case ptxSHAPE_161616_MOD:
    case ptxSHAPE_320816_MOD:
    case ptxSHAPE_083216_MOD:
        break;
    default:
        stdASSERT(False, ("Unexpected shape for WMMA"));
        return "0";
    }

    if (get_NUMARGS(parseData) == 4 ||
        (get_NUMARGS(parseData) == 3 && !parseData->modifiers.DESC))
    {
        return get_strArg(parseData, 2);
    }

#if LWCFG(GLOBAL_ARCH_TURING) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_63)
    if (get_SHAPE(parseData) == ptxSHAPE_080832_MOD) {
        if (parseData->instruction_tcode == ptx_wmma_load_a_Instr) {
            stdASSERT(get_LAYOUT(parseData) == ptxLAYOUT_ROW_MOD,
                      ("Unexpected layout for Sub Byte WMMA"));
            return "32";
        }
        if (parseData->instruction_tcode == ptx_wmma_load_b_Instr) {
            stdASSERT(get_LAYOUT(parseData) == ptxLAYOUT_COL_MOD,
                      ("Unexpected layout for Sub Byte WMMA"));
            return "32";
        }
        if (parseData->instruction_tcode == ptx_wmma_load_c_Instr ||
            parseData->instruction_tcode == ptx_wmma_store_d_Instr)
        {
            return "8";
        } else {
            stdASSERT(False, ("LDM not required outside of wmma"));
            return "0";
        }
    }

    if (get_SHAPE(parseData) == ptxSHAPE_0808128_MOD) {
        if (parseData->instruction_tcode == ptx_wmma_load_a_Instr) {
            stdASSERT(get_LAYOUT(parseData) == ptxLAYOUT_ROW_MOD,
                      ("Unexpected layout for Bit WMMA"));
            return "128";
        }
        if (parseData->instruction_tcode == ptx_wmma_load_b_Instr) {
            stdASSERT(get_LAYOUT(parseData) == ptxLAYOUT_COL_MOD,
                      ("Unexpected layout for Bit WMMA"));
            return "128";
        }
        if (parseData->instruction_tcode == ptx_wmma_load_c_Instr ||
            parseData->instruction_tcode == ptx_wmma_store_d_Instr)
        {
            return "8";
        } else {
            stdASSERT(False, ("LDM not required outside of wmma"));
            return "0";
        }
    }

#endif

#if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_VERSION_70)
    if (get_SHAPE(parseData) == ptxSHAPE_080804_MOD) {
        if (parseData->instruction_tcode == ptx_wmma_load_a_Instr) {
            if (get_LAYOUT(parseData) == ptxLAYOUT_ROW_MOD) {
                return "4";
            } else {
                stdASSERT(get_LAYOUT(parseData) == ptxLAYOUT_COL_MOD,
                          ("Unexpected layout for FP64 type WMMA"));
                return "8";
            }
        }
        if (parseData->instruction_tcode == ptx_wmma_load_b_Instr) {
            if (get_LAYOUT(parseData) == ptxLAYOUT_ROW_MOD) {
                return "8";
            } else {
                stdASSERT(get_LAYOUT(parseData) == ptxLAYOUT_COL_MOD,
                          ("Unexpected layout for FP64 type WMMA"));
                return "4";
            }
        }
        if (parseData->instruction_tcode == ptx_wmma_load_c_Instr ||
            parseData->instruction_tcode == ptx_wmma_store_d_Instr)
        {
            return "8";
        } else {
            stdASSERT(False, ("LDM not required outside of wmma"));
            return "0";
        }
    }
    if (get_SHAPE(parseData) == ptxSHAPE_161608_MOD) {
        if (parseData->instruction_tcode == ptx_wmma_load_a_Instr) {
            if (get_LAYOUT(parseData) == ptxLAYOUT_ROW_MOD) {
                return "8";
            } else {
                stdASSERT(get_LAYOUT(parseData) == ptxLAYOUT_COL_MOD,
                          ("Unexpected layout for .tf32 type WMMA"));
                return "16";
            }
        }
        if (parseData->instruction_tcode == ptx_wmma_load_b_Instr) {
            if (get_LAYOUT(parseData) == ptxLAYOUT_ROW_MOD) {
                return "16";
            } else {
                stdASSERT(get_LAYOUT(parseData) == ptxLAYOUT_COL_MOD,
                          ("Unexpected layout for .tf32 type WMMA"));
                return "8";
            }
        }
        if (parseData->instruction_tcode == ptx_wmma_load_c_Instr ||
            parseData->instruction_tcode == ptx_wmma_store_d_Instr)
        {
            return "16";
        } else {
            stdASSERT(False, ("LDM not required outside of wmma"));
            return "0";
        }
    }
#endif // Ampere && ISA_70

    if (get_SHAPE(parseData) == ptxSHAPE_161616_MOD) return "16";

    if (parseData->instruction_tcode == ptx_wmma_load_a_Instr
        && get_LAYOUT(parseData) == ptxLAYOUT_ROW_MOD)
    {
        return "16";
    }

    if (parseData->instruction_tcode == ptx_wmma_load_b_Instr
        && get_LAYOUT(parseData) == ptxLAYOUT_COL_MOD)
    {
        return "16";
    }

    if (get_SHAPE(parseData) == ptxSHAPE_083216_MOD) {
        if (get_LAYOUT(parseData) == ptxLAYOUT_ROW_MOD) return "32";
        return "8";
    }

    if (get_LAYOUT(parseData) == ptxLAYOUT_COL_MOD) return "32";
    return "8";
#else
    stdASSERT(False, ("LDM not required outside of wmma"));
    return "0";
#endif
}

String get_strCTYPE(ptxParseData parseData) {return get_strTYPE(parseData, 1);}
String get_strDTYPE(ptxParseData parseData) {return get_strTYPE(parseData, 0);}

String get_str_CTYPE(ptxParseData parseData) {return get_str_TYPE(parseData, 1); }
String get_str_DTYPE(ptxParseData parseData) {return get_str_TYPE(parseData, 0); }

int get_ALIGN(ptxParseData parseData) { return parseData->modifiers.ALIGN; }
String getALIGNAsString(int align)
{
    static char *const strALIGN[] = { PTX_ALIGNMOD_TABLE(GET_NAME) };
    return strALIGN[align];
}
String get_strALIGN(ptxParseData parseData)
{
    return getALIGNAsString(get_ALIGN(parseData));
}

int get_THREADS(ptxParseData parseData) { return parseData->modifiers.THREADS; }
String getTHREADSAsString(int threads)
{
    static char *const strTHREADS[] = { PTX_THREADSMOD_TABLE(GET_NAME) };
    stdASSERT(threads < ALEN(strTHREADS), ("Modifier value out of range"));
    return strTHREADS[threads];
}
String get_strTHREADS(ptxParseData parseData)
{
    return getTHREADSAsString(get_THREADS(parseData));
}

int get_THREADGROUP(ptxParseData parseData) { return parseData->modifiers.THREADGROUP; }
String getTHREADGROUPAsString(stdMap_t* deobfuscatedStringMapPtr, int threadgroup)
{
    static char *const strTHREADGROUP[] = { PTX_THREADGROUPMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strTHREADGROUP[threadgroup]);
}
String get_strTHREADGROUP(ptxParseData parseData)
{
    return getTHREADGROUPAsString(parseData->deobfuscatedStringMapPtr, get_THREADGROUP(parseData));
}

int get_CC(ptxParseData parseData) { return parseData->modifiers.CC; }
String getCCAsString(int cc)
{
    static char *const strCC[] = { PTX_CCMOD_TABLE(GET_NAME) };
    return strCC[cc];
}
String get_strCC(ptxParseData parseData) { return getCCAsString(get_CC(parseData)); }

int get_SHAMT(ptxParseData parseData) { return parseData->modifiers.SHAMT; }
String getSHAMTAsString(int shamt)
{
    static char *const strSHAMT[] = { PTX_SHAMTMOD_TABLE(GET_NAME) };
    return strSHAMT[shamt];
}
String get_strSHAMT(ptxParseData parseData) { return getSHAMTAsString(get_SHAMT(parseData)); }

int get_SCOPE(ptxParseData parseData) { return parseData->modifiers.SCOPE; }
String getSCOPEAsString(int scope)
{
    static char *const strSCOPE[] = { PTX_SCOPEMOD_TABLE(GET_NAME) };
    stdASSERT(scope < ALEN(strSCOPE), ("Modifier value out of range"));
    return strSCOPE[scope];
}
String get_strSCOPE(ptxParseData parseData) { return getSCOPEAsString(get_SCOPE(parseData)); }

int get_LEVEL(ptxParseData parseData) { return parseData->modifiers.LEVEL; }
String getLEVELAsString(int level)
{
    static char *const strLEVEL[] = { PTX_LEVELMOD_TABLE(GET_NAME) };
    stdASSERT(level < ALEN(strLEVEL), ("Modifier value out of range"));
    return strLEVEL[level];
}
String get_strLEVEL(ptxParseData parseData) { return getLEVELAsString(get_LEVEL(parseData)); }

int get_EVICTPRIORITY(ptxParseData parseData) { return parseData->modifiers.EVICTPRIORITY; }
String get_EVICTPRIORITYAsString(int evic) {
    static char *const strEVIC[] = { PTX_EVICTPRIORITYMOD_TABLE(GET_NAME) };
    stdASSERT(evic < ALEN(strEVIC), ("Modifier value out of range"));
    return strEVIC[evic];
}
String get_strEVICTPRIORITY(ptxParseData parseData)
{
    return get_EVICTPRIORITYAsString(get_EVICTPRIORITY(parseData) );
}

int get_LEVELEVICTPRIORITY(ptxParseData parseData) { return parseData->modifiers.LEVELEVICTPRIORITY; }
String get_LEVELEVICTPRIORITYAsString(int evic) {
    static char *const strLEVELEVIC[] = { PTX_LEVELEVICTPRIORITYMOD_TABLE(GET_NAME) };
    stdASSERT(evic < ALEN(strLEVELEVIC), ("Modifier value out of range"));
    return strLEVELEVIC[evic];
}
String get_strLEVELEVICTPRIORITY(ptxParseData parseData)
{
    return get_LEVELEVICTPRIORITYAsString(get_LEVELEVICTPRIORITY(parseData) );
}

int get_L2EVICTPRIORITY(ptxParseData parseData) { return parseData->modifiers.L2EVICTPRIORITY; }
String get_L2EVICTPRIORITYAsString(int evic) {
    return get_LEVELEVICTPRIORITYAsString(evic);
}
String get_strL2EVICTPRIORITY(ptxParseData parseData)
{
    return get_L2EVICTPRIORITYAsString(get_L2EVICTPRIORITY(parseData));
}

int get_PREFETCHSIZE(ptxParseData parseData) { return parseData->modifiers.PREFETCHSIZE; }
String get_strPREFETCHSIZE(ptxParseData parseData)
{
    return get_PREFETCHSIZEAsString(parseData->deobfuscatedStringMapPtr, get_PREFETCHSIZE(parseData));
}
String get_PREFETCHSIZEAsString(stdMap_t* deobfuscatedStringMapPtr, int prefetchsize)
{
    static char *const strPREFETCHSIZE[] = { PTX_PREFETCHSIZEMOD_TABLE(GET_NAME) };
    stdASSERT(prefetchsize < ALEN(strPREFETCHSIZE), ("Modifier value out of range"));
    return __deObfuscate(deobfuscatedStringMapPtr, strPREFETCHSIZE[prefetchsize]);
}

int get_CACHEPREFETCH(ptxParseData parseData) { return parseData->modifiers.CACHEPREFETCH; }
String get_strCACHEPREFETCH(ptxParseData parseData)
{
    return get_CACHEPREFETCHAsString(parseData->deobfuscatedStringMapPtr, get_CACHEPREFETCH(parseData));
}
String get_CACHEPREFETCHAsString(stdMap_t* deobfuscatedStringMapPtr, int cacheprefetch)
{
    static char *const strCACHEPREFETCH[] = { PTX_CACHEPREFETCHMOD_TABLE(GET_NAME) };
    stdASSERT(cacheprefetch < ALEN(strCACHEPREFETCH), ("Modifier value out of range"));
    return __deObfuscate(deobfuscatedStringMapPtr, strCACHEPREFETCH[cacheprefetch]);
}

// Note that this function returns an empty string for generic
// storage. This may not be suitable for generating error messages in
// the parser. But the empty string is important in macro expansions.
String get_strSTORAGE(ptxParseData parseData)
{
   return  get_strSTORAGES(parseData, 0);
}

String get_strSTORAGES(ptxParseData parseData, int i)
{
#define __ptxKindNameMacro(x,y,z) z,
    static char *const strSTORAGE[] =
        { "",          // ptxUNSPECIFIEDStorage
          ptxStorageKindIterate(__ptxKindNameMacro)
          ".undefined" // ptxMAXStorage
        };
#undef __ptxKindNameMacro
    // OPTIX_HAND_EDIT Fix signed/unsigned comparison warnings
    if ((uInt)i > parseData->nrofInstrMemspace - 1) {
        return "";
    }

    return strSTORAGE[parseData->storage[i].kind];
}


// Note that this function returns an empty string for generic
// storage. This may not be suitable for generating error messages in
// the parser. But the empty string is important in macro expansions.
String get_str_STORAGE(ptxParseData parseData)
{
#define __ptxKindNameMacro(x,y,z) z,
    static char *const strSTORAGE[] =
        { "",          // ptxUNSPECIFIEDStorage
          ptxStorageKindIterateRaw(__ptxKindNameMacro, "_")
          "_undefined" // ptxMAXStorage
        };
#undef __ptxKindNameMacro

    // TODO: add support to access entire "storage[]"
    return strSTORAGE[parseData->storage[0].kind];
}

int get_IS_GENERIC_STORAGE(ptxParseData parseData)
{
    // TODO: add support to access entire "storage[]"
    return parseData->storage[0].kind ==  ptxGenericStorage;
}

int get_CACHEOP(ptxParseData parseData) { return parseData->modifiers.CACHEOP; }
String getCACHEOPAsString(stdMap_t* deobfuscatedStringMapPtr, int cacheop)
{
    static char *const strCACHEOP[] = { PTX_CACHEOPMOD_TABLE(GET_NAME) };
    stdASSERT(cacheop < ALEN(strCACHEOP), ("Modifier value out of range"));
    if (cacheop == ptxILW_MOD || cacheop == ptxILWALL_MOD) {
        return __deObfuscate(deobfuscatedStringMapPtr, strCACHEOP[cacheop]);
    }
    return strCACHEOP[cacheop];
}
String get_strCACHEOP(ptxParseData parseData) { return getCACHEOPAsString(parseData->deobfuscatedStringMapPtr, get_CACHEOP(parseData)); }

int get_ORDER(ptxParseData parseData) { return parseData->modifiers.ORDER; }
String getORDERAsString(stdMap_t* deobfuscatedStringMapPtr, int order)
{
    static char *const strORDER[] = { PTX_ORDERMOD_TABLE(GET_NAME) };
    stdASSERT(order < ALEN(strORDER), ("Modifier value out of range"));
    if (order == ptxMMIO_MOD) {
        return __deObfuscate(deobfuscatedStringMapPtr, strORDER[order]);
    }
    return strORDER[order];
}
String get_strORDER(ptxParseData parseData)
{
    return getORDERAsString(parseData->deobfuscatedStringMapPtr, get_ORDER(parseData));
}

int get_KEEPREF(ptxParseData parseData) { return parseData->modifiers.KEEPREF; }
String getKEEPREFAsString(stdMap_t* deobfuscatedStringMapPtr, int keepref)
{
    static char *const strKEEPREF[] = { PTX_KEEPREFMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strKEEPREF[keepref]);
}
String get_strKEEPREF(ptxParseData parseData)
{
    return getKEEPREFAsString(parseData->deobfuscatedStringMapPtr, get_KEEPREF(parseData));
}

int get_NOATEXIT(ptxParseData parseData) { return parseData->modifiers.NOATEXIT; }
String getNOATEXITAsString(stdMap_t* deobfuscatedStringMapPtr, int noatexit)
{
    static char *const strNOATEXIT[] = { PTX_NOATEXITMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strNOATEXIT[noatexit]);
}
String get_strNOATEXIT(ptxParseData parseData)
{
    return getNOATEXITAsString(parseData->deobfuscatedStringMapPtr, get_NOATEXIT(parseData));
}

int get_NC(ptxParseData parseData) { return parseData->modifiers.NC; }
String getNCAsString(int nc)
{
    static char *const strNC[] = { PTX_NCMOD_TABLE(GET_NAME) };
    return strNC[nc];
}
String get_strNC(ptxParseData parseData) { return getNCAsString(get_NC(parseData)); }

int get_RAND(ptxParseData parseData) { return parseData->modifiers.RAND; }
String getRANDAsString(stdMap_t* deobfuscatedStringMapPtr, int rand)
{
    static char *const strRAND[] = { PTX_RANDMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strRAND[rand]);
}
String get_strRAND(ptxParseData parseData) { return getRANDAsString(parseData->deobfuscatedStringMapPtr, get_RAND(parseData)); }

int get_ROUND(ptxParseData parseData) { return parseData->modifiers.ROUND; }
String getROUNDAsString(int round)
{
    static char *const strROUND[] = { PTX_ROUNDMOD_TABLE(GET_NAME) };
    stdASSERT(round < ALEN(strROUND), ("Modifier value out of range"));
    return strROUND[round];
}
String get_strROUND(ptxParseData parseData) { return getROUNDAsString(get_ROUND(parseData)); }

int get_TESTP(ptxParseData parseData) { return parseData->modifiers.TESTP; }
String getTESTPAsString(int testp)
{
    static char *const strTESTP[] = { PTX_TESTPMOD_TABLE(GET_NAME) };
    stdASSERT(testp < ALEN(strTESTP), ("Modifier value out of range"));
    return strTESTP[testp];
}
String get_strTESTP(ptxParseData parseData) { return getTESTPAsString(get_TESTP(parseData)); }

int get_FLOW(ptxParseData parseData) { return parseData->modifiers.FLOW; }
String getFLOWAsString(int flow)
{
    static char *const strFLOW[] = { PTX_FLOWMOD_TABLE(GET_NAME) };
    stdASSERT(flow < ALEN(strFLOW), ("Modifier value out of range"));
    return strFLOW[flow];
}
String get_strFLOW(ptxParseData parseData) { return getFLOWAsString(get_FLOW(parseData)); }

int get_BRANCH(ptxParseData parseData) { return parseData->modifiers.BRANCH; }
String getBRANCHAsString(stdMap_t* deobfuscatedStringMapPtr, int branch)
{
    static char *const strBRANCH[] = { PTX_BRANCHMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strBRANCH[branch]);
}
String get_strBRANCH(ptxParseData parseData)
{
    return getBRANCHAsString(parseData->deobfuscatedStringMapPtr, get_BRANCH(parseData));
}

int get_TEXTURE(ptxParseData parseData) { return parseData->modifiers.TEXTURE; }
String getTEXTUREAsString(int texture)
{
    static char *const strTEXTURE[] = { PTX_TEXTUREMOD_TABLE(GET_NAME) };
    stdASSERT(texture < ALEN(strTEXTURE), ("Modifier value out of range"));
    return strTEXTURE[texture];
}
String get_strTEXTURE(ptxParseData parseData) { return getTEXTUREAsString(get_TEXTURE(parseData)); }
String get_str_TEXTURE(ptxParseData parseData) 
{
    String texMod =  stdCOPYSTRING(getTEXTUREAsString(get_TEXTURE(parseData)));
    // In original string first character is '.'
    // We need manually replace it with '_' to generate _{texture} string
    stdASSERT(texMod[0] == '.', ("Unexpected type string"));
    texMod[0] = '_';
    return texMod;
}

int get_TENSORDIM(ptxParseData parseData) { return parseData->modifiers.TENSORDIM; }
String getTENSORDIMAsString(int tensorDim)
{
    // Get (generic) Dimension table for lookup. There is no table for TENSORDIM.
    static char *const strTENSORDIM[] = { PTX_DIMMOD_TABLE(GET_NAME) };
    stdASSERT(tensorDim < ALEN(strTENSORDIM), ("Modifier value out of range"));
    return strTENSORDIM[tensorDim];
}
String get_strTENSORDIM(ptxParseData parseData) { return getTENSORDIMAsString(get_TENSORDIM(parseData)); }

int get_IM2COL(ptxParseData parseData) { return parseData->modifiers.IM2COL; }
String getIM2COLAsString(stdMap_t* deobfuscatedStringMapPtr, int im2col)
{
    static char *const strIM2COL[] = { PTX_IM2COLMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strIM2COL[im2col]);
}
String get_strIM2COL(ptxParseData parseData) { return getIM2COLAsString(parseData->deobfuscatedStringMapPtr, get_IM2COL(parseData)); }

int get_PACKEDOFF(ptxParseData parseData) { return parseData->modifiers.PACKEDOFF; }
String getPACKEDOFFAsString(stdMap_t* deobfuscatedStringMapPtr, int packedOff)
{
    static char *const strPACKEDOFF[] = { PTX_PACKEDOFFMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strPACKEDOFF[packedOff]);
}
String get_strPACKEDOFF(ptxParseData parseData) { return getPACKEDOFFAsString(parseData->deobfuscatedStringMapPtr, get_PACKEDOFF(parseData)); }

int get_MULTICAST(ptxParseData parseData) { return parseData->modifiers.MULTICAST; }
String getMULTICASTAsString(stdMap_t* deobfuscatedStringMapPtr, int multicast)
{
    static char *const strMULTICAST[] = { PTX_MULTICASTMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strMULTICAST[multicast]);
}
String get_strMULTICAST(ptxParseData parseData) { return getMULTICASTAsString(parseData->deobfuscatedStringMapPtr, get_MULTICAST(parseData)); }

int get_MBARRIER(ptxParseData parseData) { return parseData->modifiers.MBARRIER; }
String getMBARRIERAsString(int mbarrier)
{
    static char *const strMBARRIER[] = { PTX_MBARRIERMOD_TABLE(GET_NAME) };
    return strMBARRIER[mbarrier];
}
String get_strMBARRIER(ptxParseData parseData) { return getMBARRIERAsString(get_MBARRIER(parseData)); }

int get_FOOTPRINT(ptxParseData parseData) { return parseData->modifiers.FOOTPRINT; }
String getFOOTPRINTAsString(stdMap_t* deobfuscatedStringMapPtr, int footprint)
{
    static char *const strFOOTPRINT[] = { PTX_FOOTPRINTMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strFOOTPRINT[footprint]);
}
String get_strFOOTPRINT(ptxParseData parseData) { return getFOOTPRINTAsString(parseData->deobfuscatedStringMapPtr, get_FOOTPRINT(parseData)); }

int get_COARSE(ptxParseData parseData) { return parseData->modifiers.COARSE; }
String getCOARSEAsString(stdMap_t* deobfuscatedStringMapPtr, int coarse)
{
    static char *const strCOARSE[] = { PTX_COARSEMOD_TABLE(GET_NAME) };
    return __deObfuscate(deobfuscatedStringMapPtr, strCOARSE[coarse]);
}
String get_strCOARSE(ptxParseData parseData) { return getCOARSEAsString(parseData->deobfuscatedStringMapPtr, get_COARSE(parseData)); }

int get_COMPONENT(ptxParseData parseData) { return parseData->modifiers.COMPONENT; }
String getCOMPONENTAsString(int component)
{
    static char *const strCOMPONENT[] = { PTX_COMPONENTMOD_TABLE(GET_NAME) };
    stdASSERT(component < ALEN(strCOMPONENT), ("Modifier value out of range"));
    return strCOMPONENT[component];
}
String get_strCOMPONENT(ptxParseData parseData) { return getCOMPONENTAsString(get_COMPONENT(parseData)); }

int get_QUERY(ptxParseData parseData) { return parseData->modifiers.QUERY; }
String getQUERYAsString(int query)
{
    static char *const strQUERY[] = { PTX_QUERYMOD_TABLE(GET_NAME) };
    return strQUERY[query];
}
String get_strQUERY(ptxParseData parseData) { return getQUERYAsString(get_QUERY(parseData)); }

int get_CLAMP(ptxParseData parseData) { return parseData->modifiers.CLAMP; }
String getCLAMPAsString(int clamp)
{
    static char *const strCLAMP[] = { PTX_CLAMPMOD_TABLE(GET_NAME) };
    stdASSERT(clamp < ALEN(strCLAMP), ("Modifier value out of range"));
    return strCLAMP[clamp];
}
String get_strCLAMP(ptxParseData parseData) { return getCLAMPAsString(get_CLAMP(parseData)); }

int get_SHR(ptxParseData parseData) { return parseData->modifiers.SHR; }
String getSHRAsString(int shr)
{
    static char *const strSHR[] = { PTX_SHRMOD_TABLE(GET_NAME) };
    stdASSERT(shr < ALEN(strSHR), ("Modifier value out of range"));
    return strSHR[shr];
}
String get_strSHR(ptxParseData parseData) { return getSHRAsString(get_SHR(parseData)); }

int get_VMAD(ptxParseData parseData) { return parseData->modifiers.VMAD; }
String getVMADAsString(int vmad)
{
    static char *const strVMAD[] = { PTX_VMADMOD_TABLE(GET_NAME) };
    return strVMAD[vmad];
}
String get_strVMAD(ptxParseData parseData) { return getVMADAsString(get_VMAD(parseData)); }

int get_PRMT(ptxParseData parseData) { return parseData->modifiers.PRMT; }
String getPRMTAsString(int prmt)
{
    static char *const strPRMT[] = { PTX_PRMTMOD_TABLE(GET_NAME) };
    stdASSERT(prmt < ALEN(strPRMT), ("Modifier value out of range"));
    return strPRMT[prmt];
}
String get_strPRMT(ptxParseData parseData) { return getPRMTAsString(get_PRMT(parseData)); }


int get_SHFL(ptxParseData parseData) { return parseData->modifiers.SHFL; }
String getSHFLAsString(int shfl)
{
    static char *const strSHFL[] = { PTX_SHFLMOD_TABLE(GET_NAME) };
    stdASSERT(shfl < 5, ("Modifier value out of range"));
    return strSHFL[shfl];
}
String get_strSHFL(ptxParseData parseData) { return getSHFLAsString(get_SHFL(parseData)); }
String get_str_SHFL(ptxParseData parseData)
{
    static char *const strSHFL[] = { PTX_SHFLMOD_TABLE_RAW(GET_NAME, "_") };
    int shfl = get_SHFL(parseData);
    stdASSERT(shfl < 5, ("Modifier value out of range"));
    return strSHFL[shfl];
}

int get_ENDIS(ptxParseData parseData) { return parseData->modifiers.ENDIS; }
String getENDISAsString(stdMap_t* deobfuscatedStringMapPtr, int endis)
{
    static char *const strENDIS[] = { PTX_ENDISMOD_TABLE(GET_NAME) };
    stdASSERT(endis < 3, ("Modifier value out of range"));
    return __deObfuscate(deobfuscatedStringMapPtr, strENDIS[endis]);
}
String get_strENDIS(ptxParseData parseData) { return getENDISAsString(parseData->deobfuscatedStringMapPtr, get_ENDIS(parseData)); }

int get_UNIFORM(ptxParseData parseData) { return parseData->modifiers.UNIFORM; }
String getUNIFORMAsString(int uniform)
{
    static char *const strUNIFORM[] = { PTX_UNIFORMMOD_TABLE(GET_NAME) };
    return strUNIFORM[uniform];
}
String get_strUNIFORM(ptxParseData parseData) { return getUNIFORMAsString(get_UNIFORM(parseData)); }

int get_VECTOR(ptxParseData parseData) { return parseData->modifiers.VECTOR; }
String getVECTORAsString(int vector)
{
    static char *const strVECTOR[] = { PTX_VECTORMOD_TABLE(GET_NAME) };
    return strVECTOR[vector];
}
String get_strVECTOR(ptxParseData parseData) { return getVECTORAsString(get_VECTOR(parseData)); }

int get_VOTE(ptxParseData parseData) { return parseData->modifiers.VOTE; }
String getVOTEAsString(int vote)
{
    static char *const strVOTE[] = { PTX_VOTEMOD_TABLE(GET_NAME) };
    stdASSERT(vote < ALEN(strVOTE), ("Modifier value out of range"));
    return strVOTE[vote];
}
String get_strVOTE(ptxParseData parseData) { return getVOTEAsString(get_VOTE(parseData)); }
String get_str_VOTE(ptxParseData parseData)
{
    static char *const strVOTE[] = { PTX_VOTEMOD_TABLE_RAW(GET_NAME, "_") };
    return strVOTE[get_VOTE(parseData)];
}

int get_GUARD(ptxParseData parseData) { return parseData->guard != NULL; }
String get_strGUARD(ptxParseData parseData) { return parseData->guardStr; }

int get_PRED(ptxParseData parseData) { return parseData->guard != NULL; }
String get_strPRED(ptxParseData parseData)
{
    String temp;
    if(get_PRED(parseData)) {
        temp = stdMALLOC(strlen(parseData->guardStr) + 1 + 1);
        strcpy(temp, "@");
        strcpy(temp + 1, parseData->guardStr);
        return temp;
    } else {
        return "";
    }
}
String get_strPRED_NEG(ptxParseData parseData)
{
    String temp;
    int tPos = 0, gPos = 0;
    if(get_PRED(parseData)) {
        temp = stdMALLOC(strlen(parseData->guardStr) + 2 + 1);
        strcpy(temp + tPos, "@");
        tPos++;
        if (parseData->guardStr[0] == '!') {
            gPos++;
        } else {
            strcpy(temp + tPos, "!");
            tPos++;
        }
        strcpy(temp + tPos, parseData->guardStr + gPos);
        return temp;
    } else {
        return "";
    }
}

int get_PTX_VERSION(ptxParseData parseData)
{
    int minor, major;
    sscanf(getPtxVersionString(parseData), "%d.%d", &major, &minor);
    return major * 10 + minor;
}

int get_PTX_TARGET(ptxParseData parseData)
{
  return (int) ctParseArchVersion(getTargetArchString(parseData));
}

String get_strArg(ptxParseData parseData, int index) { return parseData->arg[index]; }

int get_HAS_SINK_DEST(ptxParseData parseData)
{
    return (parseData->arguments[0]->kind == ptxSinkExpression);
}

String get_strIArg(ptxParseData parseData, int index) { return parseData->inlineInputArgName[index]; }
String get_strOArg(ptxParseData parseData, int index) { return parseData->inlineOutputArgName[index]; }

int get_NUMARGS(ptxParseData parseData) { return parseData->nrArgs; }

int get_PREDICATE_OUTPUT(ptxParseData parseData) { return parseData->predicateOutput != NULL; }
String get_strPREDICATE_OUTPUT(ptxParseData parseData) { return parseData->predOutStr; }
String get_strOPTIONAL_PREDICATE_OUTPUT(ptxParseData parseData)
{
    if (get_PREDICATE_OUTPUT(parseData))
        return stdCONCATSTRING(" | ", parseData->predOutStr);

    return "";
}
String get_strIS_PREDICATE_OUTPUT(ptxParseData parseData)
{
    return  get_PREDICATE_OUTPUT(parseData) ? "_pred" : ""; 
}

int get_IS_TEXMODE_INDEPENDENT(ptxParseData parseData)
{ 
    return parseData->isTexModeIndependent ? 1 : 0;
}

String get_strIS_TEXMODE_INDEPENDENT(ptxParseData parseData)
{ 
    return parseData->isTexModeIndependent ? "_texmode_independent" : "";
}

int get_IS_TEX_INSTR_EMULATED(ptxParseData parseData)
{
    if (get_HAS_DEPTH_COMPARE_ARG(parseData)) {
        return 1;
    } else if (parseData->modifiers.TEXTURE == ptxLWBE_MOD  || 
               parseData->modifiers.TEXTURE == ptxALWBE_MOD ||
               parseData->modifiers.TEXTURE == ptx3D_MOD)
    {
        return 1;
    } else {
        return 0;
    }
}
/*
* Function         : Get argument number of co-ordinate vector in texture/surface instruction.
* Parameters       : ptxInstruction code
* Function Result  : argument number
*/
static int ptxTexSurfGetCoordArgNumberParseData(ptxParseData parseData, uInt code)
{
    stdASSERT( ptxIsTextureInstr(code) || ptxIsSurfaceInstr(code), ("texture/surface instruction expected") );

    if (ptxIsTextureInstr(code)){
        return parseData->isTexModeIndependent  ? 3 : 2;
    } else if(code == ptx_sust_b_Instr || code == ptx_sust_p_Instr || code == ptx_sured_b_Instr || code == ptx_sured_p_Instr) {
        return 1;
    } else {
        return 2;
    }
}

// NOTE: Following *ParseData functions are duplicates of corresponding function in ptxIR.h
//       which accepts parseState. We need to redefine functions here as we don't have access
//       of parseState.
//
// TODO: Update functions in ptxIR.h to accept only required filed instead of complete parseData


/*
* Function         : Get minimum number of arguments in texture/surface instruction
* Parameters       : ptxInstruction code
* Function Result  : minimum number of arguments based on independent/unified mode
*/

static uInt ptxTexGetMinNumberOfArgsParseData(ptxParseData parseData, uInt code)
{
    int texCordPos = ptxTexSurfGetCoordArgNumberParseData(parseData, code);
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

int get_HAS_OFFSET_ARG(ptxParseData parseData)
{
    uInt minArgs = ptxTexGetMinNumberOfArgsParseData(parseData, parseData->instruction_tcode);
    stdASSERT( ptxIsTextureInstr(parseData->instruction_tcode), ("texture instruction expected") );

    if (parseData->modifiers.FOOTPRINT) {
        return 0;
    }
 
    if (parseData->nrArgs > (int)minArgs) {
        return parseData->arguments[minArgs]->type->kind == ptxVectorType ? 1 : 0;
    } else {
        return 0;
    }
}

String get_strHAS_OFFSET_ARG(ptxParseData parseData)
{
    uInt minArgs = ptxTexGetMinNumberOfArgsParseData(parseData, parseData->instruction_tcode);
    stdASSERT( ptxIsTextureInstr(parseData->instruction_tcode), ("texture instruction expected") );

    if (parseData->modifiers.FOOTPRINT) {
        return "";
    }
 
    if (parseData->nrArgs > minArgs) {
        return parseData->arguments[minArgs]->type->kind == ptxVectorType ? "_offset" : "";
    } else {
        return "";
    }
}

/* Generate vector .vX string as per size of argument vector */
String get_strARG_VECTOR(ptxParseData parseData, uInt instrArgid)
{
    if (instrArgid > parseData->nrArgs) {
        stdASSERT(False, ("Incorrect argument id"));
        return "";
    } 
    if (parseData->arguments[instrArgid]->type->kind == ptxVectorType) {
        switch (parseData->arguments[instrArgid]->type->cases.Vector.N) {
        case 2:
            return ".v2";
        case 4:
            return ".v4";
        default:
            return "";
        }
    }
        return "";
}
/*
 * Function         : Check if tex,tld4 instruction has an explicit sampler
 * Function Result  : True iff texture instruction has an explicit sampler
 */

int get_HAS_DEPTH_COMPARE_ARG(ptxParseData parseData)
{
    uInt minArgs = ptxTexGetMinNumberOfArgsParseData(parseData, parseData->instruction_tcode);
    ptxType type = parseData->arguments[parseData->nrArgs - 1]->type; 
    ptxExpressionKind exprKind = parseData->arguments[parseData->nrArgs - 1]->kind;
    stdASSERT( ptxIsTextureInstr(parseData->instruction_tcode), ("texture instruction expected") );
 
    if (parseData->modifiers.FOOTPRINT) {
        return 0;
    }

    // OPTIX_HAND_EDIT Fix signed/unsigned comparison warnings
    if (parseData->nrArgs > (int)minArgs) {
        return (isF32(type) || isB32(type) || exprKind == ptxFloatConstantExpression) ? 1 : 0;
    } else {
        return 0;
    }
}

String get_strHAS_DEPTH_COMPARE_ARG(ptxParseData parseData)
{
    uInt minArgs = ptxTexGetMinNumberOfArgsParseData(parseData, parseData->instruction_tcode);
    ptxType type = parseData->arguments[parseData->nrArgs - 1]->type; 
    ptxExpressionKind exprKind = parseData->arguments[parseData->nrArgs - 1]->kind;
    stdASSERT( ptxIsTextureInstr(parseData->instruction_tcode), ("texture instruction expected") );
 
    if (parseData->modifiers.FOOTPRINT) {
        return "";
    }

    if (parseData->nrArgs > minArgs) {
        return (isF32(type) || isB32(type) || exprKind == ptxFloatConstantExpression) ? "_depth_compare" : "";
    } else {
        return "";
    }
}

// return -1 if argument not immediate, otherwise return barrier number
int get_BAR_IMM(ptxParseData parseData)
{
    stdASSERT(ptxIsBarOrBarrierInstr(parseData->instruction_tcode), ("barrier instruction expected"));
    if (parseData->arguments[0]->kind == ptxIntConstantExpression) {
        return parseData->arguments[0]->cases.IntConstant.i;
    } else {
        return -1;
    }
}
String get_strBAR_IMM(ptxParseData parseData)
{
    static char *const strBAR_IMM[] = { "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15" };
    int barNum = get_BAR_IMM(parseData);
    stdASSERT(ptxIsBarOrBarrierInstr(parseData->instruction_tcode), ("barrier instruction expected"));
    stdASSERT(-1 <= barNum && barNum < 16, ("Illegal barrier number, bar.sync %d", barNum));
    return strBAR_IMM[barNum+1];
}

/*
 * Input            : 1. selector string of the form 0xDDDD/0xDD/0xD e.g. 0x3120
 *                       (Note there should not be .h/.b in string)
 *                    2. index of the argument
 *                    3. simdWidth
 * Output           : True if selectorString is valid, False otherwise
 * Comments         : In order to check the validity of selectorString using following function,
 *                    the string must be in the form of "0xDDDD" (in case of SIMD4) or 
 *                    "0xDD" (in case of SIMD2) where D = {0, 1, ... ,7} or 0xX (in case of scalar video instructions)
 *                    where X = {0, 1, 2, 3}
 */                    
static Bool isVideoSelectorStringValid(String selectorString, int index, int simdWidth)
{
    int i, maxAllowedSelector;
    Bool isValid = True;

    switch(simdWidth) {
    case 1: 
        if (!strcmp(selectorString, "")) {
            isValid &= True;        // "" is default selector for scalar video instructions
        } else {
            // Check if proper SIMD selector is provided.
            if (strlen(selectorString) != simdWidth + 2) 
                isValid &= False;

            maxAllowedSelector = selectorString[2] == 'h' ? 1 : 3;
            for (i = 2; i < strlen(selectorString); i++) {
                if (!(selectorString[i] - '0' >= 0 && selectorString[i] - '0' <= maxAllowedSelector))
                    isValid &= False;
            }
        }
        stdASSERT(isValid, ("Illegal SIMD%d selector value", simdWidth));
        break;    
    case 2:
    case 4:
        maxAllowedSelector =  2 * simdWidth - 1; 
        // Check if byte/half-word selector digit is greater than 0
        // and less than  2 * simdWidth - 1.
        for (i = 2; i < strlen(selectorString); i++) {
            if (!((selectorString[i] - '0') >= 0 && (selectorString[i] - '0') <= maxAllowedSelector))
                isValid &= False;
        }
        stdASSERT(isValid, ("Illegal SIMD%d selector value", simdWidth));

        if (index != 0) {
            if (simdWidth + 2 != strlen(selectorString)) {
                // Check selector string length
                stdASSERT(strlen(selectorString) == simdWidth + 2, ("Incorrect length of SIMD%d selector value", simdWidth));
                isValid &= False;
            }
        } else {
            for (i = 2; i < strlen(selectorString) - 1; i++) { 
                if (selectorString[i + 1] >= selectorString[i]) {
                    isValid &= False;
                }
            }
            stdASSERT(isValid, ("Incorrect SIMD%d destination mask", simdWidth));
        }
        break;
    } // end switch
    return isValid;
}

// Return default selector as per specification.
// This function will allocate string on heap
// and it is client's responsibility to free it.
static String getDefaultSelector(int index, int simdWidth)
{
    switch(simdWidth * 10 + index) {
    case 10 :                                    // SIMD1 mask for all
    case 11 :                                    // operands is ""
    case 12 : return stdCOPYSTRING("");
    case 20 : return stdCOPYSTRING("0x10");      // SIMD2, Mask
    case 21 : return stdCOPYSTRING("0x10");      // SIMD2, 1st Inputs default
    case 22 : return stdCOPYSTRING("0x32");      // SIMD2, 2nd Inputs default
    case 40 : return stdCOPYSTRING("0x3210");    // SIMD4, Mask
    case 41 : return stdCOPYSTRING("0x3210");    // SIMD4, 1st Inputs default
    case 42 : return stdCOPYSTRING("0x7654");    // SIMD4, 2nd Inputs defaultx1
    default:
        stdASSERT(False, ("Invalid index value"));
        return stdCOPYSTRING("");
    }
}

static String getSubControl(int digit)
{
    switch(digit) {
    case 0 : return "10";
    case 1 : return "32";
    case 2 : return "54";
    case 3 : return "76";
    default:
        stdASSERT(False, ("Invalid parameter"));
        return "";
    }
    
}

// The following function returns permute controls
// required for selecting operands in scalar video instruction.
// Note, 
// 1. The permute controls are meant to be used in PRMT instruction.
// 2. These permute controls will select the byte/half-word
//    based on "cases.VideoSelect->selector[0]" and 
//    mask off other bytes/half-words.
String get_strVideoPermCtrlToSelect(ptxParseData parseData, int index)
{
    ptxExpression expr = parseData->arguments[index];

    // Considering prmt.b32 d, a, b, c;
    // bytes are numbered as {b, a} = {{b7, b6, b5, b4}, {b3, b2, b1, b0}}
    // The following selectors assume "b" as third operand of video instruction
    // and "a" as result of intermediate operation i.e. result before merging.
    // Note, intermediate result is present in 0th byte so 
    // permutation control is seleting "0"th byte and not others.

    if (expr->kind == ptxVideoSelectExpression) {
        switch (expr->cases.VideoSelect->selector[0]) {
        case ptxCOMP_NONE: return "0x3210";
        case ptxCOMP_H0:   return "0x7610";
        case ptxCOMP_H1:   return "0x3254";
        case ptxCOMP_B0:   return "0x7650";
        case ptxCOMP_B1:   return "0x7604";
        case ptxCOMP_B2:   return "0x7054";
        case ptxCOMP_B3:   return "0x0654";
        default:
            stdASSERT(False, ("Invalid index"));
            return "";
        }
    } else if (expr->kind == ptxSymbolExpression || expr->kind == ptxVectorSelectExpression || expr->kind == ptxIntConstantExpression) {
        // This case will be hit if operands do not have selector specified at source level.
        return "0x3210";
    } else {
        stdASSERT(0, ("Unexpected expression kind"));
        return "";
    }
}

// The following function returns permute controls
// required for selecting and right shifting operands in 
// scalar video instruction. 
// Note, 
// 1. The permute controls are meant to be used in PRMT instruction.
// 2. These permute controls will select the byte/half-word
//    based on "cases.VideoSelect->selector[0]",  
//    mask off other bytes/half-words, and do right shift.
String get_strVideoPermCtrlToSelectAndRightShift(ptxParseData parseData, int index)
{
    ptxExpression expr = parseData->arguments[index];

    if (expr->kind == ptxVideoSelectExpression) {
        switch (expr->cases.VideoSelect->selector[0]){
        case ptxCOMP_NONE: return "0x3210";
        case ptxCOMP_H0:   return "0x7610";
        case ptxCOMP_H1:   return "0x5432";
        case ptxCOMP_B0:   return "0x7650";
        case ptxCOMP_B1:   return "0x7651";
        case ptxCOMP_B2:   return "0x7652";
        case ptxCOMP_B3:   return "0x7653";
        default:
            stdASSERT(False, ("Invalid index"));
            return "";
        }
    } else if (expr->kind == ptxSymbolExpression || expr->kind == ptxVectorSelectExpression || expr->kind == ptxIntConstantExpression) {
        // This case will be hit if operands do not have selector specified at source level.
        return "0x3210";
    } else {
        stdASSERT(0, ("Unexpected expression kind"));
        return "";
    }
}

// Following function prefixes "0x" to the input string.
String addHexPrefix(String s)
{
    String result = stdMALLOC(10);
    result[0] = '\0';
    strcat(result, "0x");
    strcat(result, s);
    return result;
}

// Following function strips out .h/.b and 0x 
// from selector strings and returns bare numbers.
String getSelectorString(ptxParseData parseData, int index, int simdWidth)
{
    char *pos;
    String instrOperand, selectorString, temp, defaultSelector; 

    instrOperand    = parseData->arg[index];
    selectorString  = stdMALLOC(5);
    temp            = stdMALLOC(10);
    defaultSelector = NULL;   

    pos = strstr(instrOperand, ".");
    if (pos != NULL) {
        sscanf(instrOperand, "%*[^.].%*[h|b]%s", selectorString);
    } else {
        defaultSelector = getDefaultSelector(index, simdWidth);
        sscanf(defaultSelector, "0x%s", selectorString);
        stdFREE(defaultSelector);
    }

    temp[0] = '\0';
    strcat(temp, "0x");
    strcat(temp, selectorString);

    if (ptxIsVideoSIMD2Instr(parseData->instruction_tcode) || ptxIsVideoSIMD4Instr(parseData->instruction_tcode))
        isVideoSelectorStringValid(temp, index, simdWidth);

    stdFREE(temp);
    return selectorString;
}

// Following function returns permutation controls 
// required for properly selecting half-words.
String get_strSIMD2PermuteControl(ptxParseData parseData, int index)
{
    String selectorString, temp, permCtrl;
    int i, hashTable[2] = {0, 0};
  
    selectorString = NULL;
    permCtrl       = NULL; 
    temp           = stdMALLOC(10); 
    temp[0]        = '\0';
    selectorString = getSelectorString(parseData, index, 2);
    
    // If index=0 and there is only one half-word no provided in mask 
    // then use corresponding half-word from last operand.

   if (index == 0) {
       for (i = 0; i < strlen(selectorString); i++)
           hashTable[selectorString[i] - '0'] = 1;

       // Traverse in reverse direction, as in video instructions 
       // destination mask can occur only in descending order.
       // If hash table entry is 1 then corresponding byte is requested 
       // else take +2nd byte from last operand.
                                       
       for (i = 1; i >= 0; i--) {
           if (hashTable[i] == 1) {
               strcat(temp, getSubControl(i));
           } else {
               strcat(temp, getSubControl(i + 2));
           }  
       } 
    } else {   
        // else colwert to what is required by .prmt instruction. 
        for (i = 0; i < 2; i++)
            strcat(temp, getSubControl(selectorString[i] - '0'));
    }
    
    stdFREE(selectorString);
    permCtrl = addHexPrefix(temp);
    stdFREE(temp);
    return permCtrl;
}

// Following function returns permutation controls 
// required for properly selecting bytes.
String get_strSIMD4PermuteControl(ptxParseData parseData, int index)
{
    String result, temp, selectorString, permCtrl;
    int i, hashTable[] = {0, 0, 0, 0};                                    // A simple hash table to record the oclwrance of byte/half-word number.

    result         = NULL;
    temp           = NULL;
    permCtrl       = NULL;
    selectorString = getSelectorString(parseData, index, 4);
    if (index != 0) {                                                 // If index!=0 then simply colwert to hex and return.
        result = addHexPrefix(selectorString);
        stdFREE(selectorString);
        return result;
    } else {
        temp   = stdMALLOC(10);
        result = stdMALLOC(10);

        result[0] = '\0';

        for (i = 0; i < strlen(selectorString); i++)
            hashTable[selectorString[i] - '0'] = 1;                   // Record in hash table.
        // Traverse in reverse direction, as in SIMD4 video instructions 
        // destination mask can occur only in descending order.
        // If hash table entry is 1 then corresponding byte is requested 
        // else take +4th byte from last operand.

        for (i = 3; i >= 0; i--) {
            if (hashTable[i] == 1)
                sprintf(temp, "%d", i);
            else
                sprintf(temp, "%d", i+4);
            strcat(result, temp);
        }

        stdFREE(temp);
        stdFREE(selectorString);
        permCtrl = addHexPrefix(result);
        stdFREE(result);
        return permCtrl;
    }
}

// Following two functions return "1" if "byteno" half-word needs 
// to be added in merge operation of video instructions.
String get_strIfSIMD2DestHasByte(ptxParseData parseData, int byteno)
{
    String selectorString, needToAdd;
    int i;

    needToAdd      = "0";
    selectorString = getSelectorString(parseData, 0, 2);

    for (i = 0; i < strlen(selectorString); i++) {
        if ((selectorString[i] - '0') == byteno) {
            needToAdd = "1";
            break;
        }
    }

    stdFREE(selectorString);
    return needToAdd;
}

// Following function return "1" if "byteno" byte needs 
// to be added in merge operation of video instructions.
String get_strIfSIMD4DestHasByte(ptxParseData parseData, int byteno)
{
    String selectorString, needToAdd;
    int i;

    needToAdd      = "0";
    selectorString = getSelectorString(parseData, 0, 4);

    for (i = 0; i < strlen(selectorString); i++) {
        if ((selectorString[i] - '0') == byteno) {
            needToAdd = "1";
            break;
        }
    }

    stdFREE(selectorString);
    return needToAdd;
}

static String extractVideoSelectorFromOperand(ptxParseData parseData, int index)
{
    String instrOperand = parseData->arg[index], selector = stdMALLOC(7), selectorValue = stdMALLOC(10);
    int n;

    n = sscanf(instrOperand, "%*[^.].%*[h|b]%s", selectorValue);
    if (n == 1) {
        sprintf(selector, "0x%s", selectorValue);
    } else {
        selector[0] = 0;
    }

    stdFREE(selectorValue);
    return selector;
}

String get_strVideoOperand(ptxParseData parseData, int index)
{
   String instrOperand = parseData->arg[index];
   String operand = stdMALLOC(strlen(instrOperand));

   sscanf(instrOperand, "%[^.].", operand);
   return operand;
}

// Basic purpose of this function is to return 
// enum value corresponding to the input selector
// i.e. return ptxCOMP_H0 for .h0, ptxCOMP_B0 for .b0 and so on.
int get_VIDEOSELECTOR(ptxParseData parseData, int index, int subindex, int simdWidth)
{
    String selector;
    char temp[7];
    int  n, offset, minComponent, paramsScanned;
    String instrOperand;
    static const int videoSelectors[] = { PTX_VIDEOSELECTOR_TABLE(GET_ENUM) };

    selector      = NULL;
    offset        = 0;
    minComponent  = 0;
    paramsScanned = 0;
    instrOperand  = parseData->arg[index];

    stdASSERT(subindex < simdWidth, ("subindex cannot be less than SIMD width"));
    stdASSERT(ptxGetVideoInstrSIMDWidth(parseData->instruction_tcode) == simdWidth, 
             (" simdWidth is wrong, provided %d, expected %d\n", simdWidth, ptxGetVideoInstrSIMDWidth(parseData->instruction_tcode)));

    // We have to handle following cases here.
    // 1. When Scalar video instructions have default selectors i.e. ""
    // 2. When scalar video instructions have user specified selectors.
    // 3. When non-scalar video instructions have default selectors i.e. 0x3210/0x7654.
    // 4. When non-scalar video instructions have  user specified selectors.
    selector = extractVideoSelectorFromOperand(parseData, index);
    n = strlen(selector);

    if (n == 0) { // Default modifiers
        // Free "selector"
        stdFREE(selector);
        selector = getDefaultSelector(index, simdWidth);
        n = strlen(selector);
        // To handle case 3 from the above comment. 
        minComponent = (simdWidth == 2) ? ptxCOMP_H0 : ptxCOMP_B0;
    }

    // Check if selectors are correct.
    isVideoSelectorStringValid(selector, index, simdWidth);

    // subindicies n-1, n-2 would refer to "0" "x" in selector string.
    // Return ptxCOMP_NONE in case selector string A is compared against selector
    // string B, where A is suffix of B and strlen (A) < strlen(B).
    if (subindex >= n - 2) {
        stdFREE(selector); 
        return ptxCOMP_NONE;
    }    

    // To handle case 2 and 4 from the above comment.
    paramsScanned = sscanf(instrOperand, "%*[^.].%s", temp);

    if (paramsScanned == 1) {
        switch(temp[0]) {
        case 'h':  minComponent = ptxCOMP_H0; break;
        case 'b':  minComponent = ptxCOMP_B0; break;
        default :  stdASSERT(0, ("Wrong selector specifier\n")); break;
        }
    } else if (simdWidth == 1) {
        stdFREE(selector);
        // To handle case 1 from the above comment.
        return ptxCOMP_NONE;
    }

    offset = selector[(n - 1) - subindex] - '0';

    stdFREE(selector);
    return videoSelectors[minComponent + offset];
}

int get_TYPES(ptxParseData parseData, int index) {return parseData->instrTypes[index]; }
String get_strTYPES(ptxParseData parseData)
{
    String temp;
    int len = 0, i;

    for(i = 0; i < parseData->nrInstrTypes; i++)
        len += strlen(get_strTYPE(parseData, i));

    temp = stdMALLOC(len + 1);
    len = 0;
    for(i = 0; i < parseData->nrInstrTypes; i++) {
        strcpy(temp + len, get_strTYPE(parseData, i));
        len += strlen(temp + len);
    }
    temp[len] = '\0';
    return temp;
}

static Bool isStringAssociatedWithTypeKind(ptxTypeKind typeKind)
{
    switch(typeKind) {
    case ptxTypeB1:
    case ptxTypeB2:
    case ptxTypeB4:
    case ptxTypeB8:
    case ptxTypeB16:
    case ptxTypeB32:
    case ptxTypeB64:
    case ptxTypeB128:
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
    case ptxTypeF16:
    case ptxTypeF16x2:
    case ptxTypeF32:
    case ptxTypeF64:
    case ptxTypeBF16:
    case ptxTypeBF16x2:
    case ptxTypeTF32:
    case ptxTypePred:
        return True;
    default:
        return False;
    }
}

String getTypeEnumAsString(stdMap_t* deobfuscatedStringMapPtr, ptxTypeKind typeKind)
{
    static char *const strTYPES[] = { PTX_TYPES_TABLE(GET_NAME) };
    stdASSERT(typeKind < ALEN(strTYPES), ("Modifier value out of range"));
    if (typeKind == ptxTypeB128   ||
        typeKind == ptxTypeE4M3   ||
        typeKind == ptxTypeE5M2   ||
        typeKind == ptxTypeE4M3x2 ||
        typeKind == ptxTypeE5M2x2)
    {
        return __deObfuscate(deobfuscatedStringMapPtr, strTYPES[typeKind]);
    }
    return strTYPES[typeKind];
}

String getTYPEAsString(stdMap_t* deobfuscatedStringMapPtr, ptxType type)
{
    return getTypeEnumAsString(deobfuscatedStringMapPtr, type->kind);
}

String get_strTYPE(ptxParseData parseData, int index)
{
    return getTypeEnumAsString(parseData->deobfuscatedStringMapPtr, get_TYPES(parseData, index));
}

// NOTE: Since __deObfuscate() may return the same deobfuscated string for
//       every invocation with the same input string, the returned string needs
//       to be copied before modifying.
String get_str_TYPE(ptxParseData parseData, int index)
{
    static char *const strTYPES[] = { PTX_TYPES_TABLE(GET_NAME) };
    char* typeStr;
    int type = get_TYPES(parseData, index);

    stdASSERT(type < ALEN(strTYPES), ("Modifier value out of range"));
    if (type == ptxTypeB128) {
        typeStr = __deObfuscate(parseData->deobfuscatedStringMapPtr, strTYPES[type]);
        if (typeStr == strTYPES[type]) {
            stdASSERT(False, ("__deObfuscate routine is expected to return new string copy"));
            typeStr = stdCOPYSTRING(strTYPES[type]);
        }
    } else {
        typeStr = stdCOPYSTRING(strTYPES[type]);
    }
    // In original string first character is '.'
    // We need manually replace it with '_' to generate _{type} string
    stdASSERT(typeStr[0] == '.', ("Unexpected type string"));
    typeStr[0] = '_';
    return typeStr;
}

static String get_strWMMAScalarComponent(ptxExpression expr, int index)
{
    stdList_t l;
    int count;

    count = 0;
    for (l = expr->cases.Vector.elements; l; l = l->tail) {
        if (count == index) {
            stdString_t s = stringNEW();

            ptxPrintExpression(l->head, s);
            return stringStripToBuf(s);
        }
        count++;
    }
    return NULL;
} 

String get_strWMMAVectorComponent_SRCA(ptxParseData parseData, int index)
{
    String s;
    ptxExpression expr = parseData->arguments[1];

    stdASSERT(expr && expr->kind == ptxVectorExpression, ("vector expression expected"));

    if ((s = get_strWMMAScalarComponent(expr, index))) {
        return s;
    } else {
        stdASSERT(0, ("invalid index"));
        return "";
    }
}

String get_strWMMAVectorComponent_SRCB(ptxParseData parseData, int index)
{
    String s;
    ptxExpression expr = parseData->arguments[2];

    stdASSERT(expr && expr->kind == ptxVectorExpression, ("vector expression expected"));

    if ((s = get_strWMMAScalarComponent(expr, index))) {
        return s;
    } else {
        stdASSERT(0, ("invalid index"));
        return "";
    }
}

String get_strWMMAVectorComponent_SRCC(ptxParseData parseData, int index)
{
    String s;
    ptxExpression expr = parseData->arguments[3];

    stdASSERT(expr && expr->kind == ptxVectorExpression,
              ("vector expression expected"));

    if ((s = get_strWMMAScalarComponent(expr, index))) {
        return s;
    } else {
        stdASSERT(0, ("invalid index"));
        return "";
    }
}

String get_strADDR_BASE(ptxParseData parseData, int index)
{
    ptxSymbolTableEntry symbol = ptxGetSymEntFromExpr(parseData->arguments[index]);
    stdString_t s = stringNEW();
    stringAddBuf(s, symbol->symbol->unMangledName);
    return stringStripToBuf(s);
}

String get_strADDR_OFFSET(ptxParseData parseData, int index)
{
    ptxExpression expr = parseData->arguments[index];
    stdString_t s = stringNEW();
    stdASSERT(expr && expr->kind == ptxAddressRefExpression,
              ("expected address ref"));

    expr = expr->cases.AddressRef.arg;
    if (expr->kind == ptxBinaryExpression) {
        // base + offset
        expr = expr->cases.Binary->right;
        ptxPrintExpression(expr, s);
    } else {
        stringAddFormat(s, "%d", 0);
    }
    return stringStripToBuf(s);
}

int get_IsSourceAddrArgInRegisterSpace(ptxParseData parseData)
{
    ptxSymbolTableEntry symbol = ptxGetSymEntFromExpr(parseData->arguments[1]);
    return ptxIsRegisterStorage(symbol->storage) ? 1 : 0;
}

int get_IsSourceAddrArgI32(ptxParseData parseData)
{
    ptxSymbolTableEntry symbol = ptxGetSymEntFromExpr(parseData->arguments[1]);
    return (isI32(symbol->symbol->type) || isB32(symbol->symbol->type)) ? 1 : 0;
}

int get_IsDestAddrArgInRegisterSpace(ptxParseData parseData)
{
    ptxSymbolTableEntry symbol = ptxGetSymEntFromExpr(parseData->arguments[0]);
    return ptxIsRegisterStorage(symbol->storage) ? 1 : 0;
}

int get_IsDestAddrArgI32(ptxParseData parseData)
{
    ptxSymbolTableEntry symbol = ptxGetSymEntFromExpr(parseData->arguments[0]);
    return (isI32(symbol->symbol->type) || isB32(symbol->symbol->type)) ? 1 : 0;
}

int get_IsWiderSrcForCvt(ptxParseData parseData)
{
    stdASSERT( parseData->instruction_tcode == ptx_cvt_Instr, ("cvt instruction expected") );
    return isB64(parseData->arguments[1]->type) ? 1 : 0;
}

int get_IsWiderDstForCvt(ptxParseData parseData)
{
    stdASSERT( parseData->instruction_tcode == ptx_cvt_Instr, ("cvt instruction expected") );
    return isB64(parseData->arguments[0]->type) ? 1 : 0;
}

String get_strVectorComponent_DST(ptxParseData parseData, int index)
{
    stdList_t l;
    int count;
    ptxExpression expr = parseData->arguments[0];

    stdASSERT(expr && expr->kind == ptxVectorExpression, ("vector expression expected"));

    count = 0;
    for (l = expr->cases.Vector.elements; l; l = l->tail) {
        if (count == index) {
            stdString_t s = stringNEW();

            ptxPrintExpression(l->head, s);
            return stringStripToBuf(s);
        }
        count++;
    }
    stdASSERT(0, ("invalid index"));
    return "";
}

String get_strWMMAStoreValue(ptxParseData parseData, int index)
{
    stdList_t l;
    int count;
    ptxExpression expr = parseData->arguments[1];

    stdASSERT(expr && expr->kind == ptxVectorExpression, ("vector expression expected"));

    count = 0;
    for (l = expr->cases.Vector.elements; l; l = l->tail) {
        if (count == index) {
            stdString_t s = stringNEW();

            ptxPrintExpression(l->head, s);
            return stringStripToBuf(s);
        }
        count++;
    }
    stdASSERT(0, ("invalid index"));
    return "";
}

int get_MacroElw(ptxParseData parseData, macroElwVar evar)
{
    return parseData->elwVars[evar];
}

Bool isMacroElwEqual(ptxParseData parseData, macroElwVar evar, String str)
{
    uInt arch;

    if(evar == GPU_ARCH) {
        sscanf(str, "%*[^0-9]%u", &arch);
        return arch == parseData->elwVars[evar];
    }
    return False;
}

static Bool isVideoSelectorDefault(ptxParseData parseData)
{
    Bool isDefaultMask, isDefaultASel, isDefaultBSel;
    isDefaultMask = (get_VIDEOSELECTOR(parseData, 0, 0, 4) == ptxCOMP_B0) &&
                    (get_VIDEOSELECTOR(parseData, 0, 1, 4) == ptxCOMP_B1) &&
                    (get_VIDEOSELECTOR(parseData, 0, 2, 4) == ptxCOMP_B2) &&
                    (get_VIDEOSELECTOR(parseData, 0, 3, 4) == ptxCOMP_B3);
    isDefaultASel = (get_VIDEOSELECTOR(parseData, 1, 0, 4) == ptxCOMP_B0) &&
                    (get_VIDEOSELECTOR(parseData, 1, 1, 4) == ptxCOMP_B1) &&
                    (get_VIDEOSELECTOR(parseData, 1, 2, 4) == ptxCOMP_B2) &&
                    (get_VIDEOSELECTOR(parseData, 1, 3, 4) == ptxCOMP_B3);
    isDefaultBSel = (get_VIDEOSELECTOR(parseData, 2, 0, 4) == ptxCOMP_B4) &&
                    (get_VIDEOSELECTOR(parseData, 2, 1, 4) == ptxCOMP_B5) &&
                    (get_VIDEOSELECTOR(parseData, 2, 2, 4) == ptxCOMP_B6) &&
                    (get_VIDEOSELECTOR(parseData, 2, 3, 4) == ptxCOMP_B7);
    return isDefaultMask && isDefaultASel && isDefaultBSel;
}

int get_IsVabsdiff4NativelySupported(ptxParseData parseData)
{
    ptxTypeKind type0, type1, type2;
    int arch;

    stdASSERT( parseData->instruction_tcode == ptx_vabsdiff4_Instr, ("vabsdiff4 instruction expected") );

    // Do not emulate for below Maxwell architectures
    if (!get_MacroElw(parseData, NEED_VIDEO_EMULATION))
        return 1;

    type0 = get_TYPES(parseData, 0);
    type1 = get_TYPES(parseData, 1);
    type2 = get_TYPES(parseData, 2);
    arch  = get_MacroElw(parseData, GPU_ARCH);

    // Maxwell and Pascal support below variants of vabsdiff4 natively. 
    if (type0 == ptxTypeU32 &&
        get_SAT(parseData) != ptxSAT_MOD &&
        isVideoSelectorDefault(parseData))
    {
        if (arch < 70)
            return 1;
        
        // For Volta+ native VABSDIFF4 sass is supported if type1 == type2.
        if (type1 == type2)
            return 1;
        
    }
    return 0;
}

/* ****************** API Functions for pasrer and ptxas *************************** */

void initMacroState(ptxParseData parseData)
{
    parseData->nrInstrTypes = parseData->nrArgs = parseData->nrInlineFuncInputArgs = parseData->nrInlineFuncRetArgs = 0;
    stdMEMCLEAR(&parseData->guardStr);
    stdMEMCLEAR(&parseData->predOutStr);
    stdMEMCLEAR(&parseData->arg);
    stdMEMCLEAR(&parseData->instrTypes);
}

void initMacroElwVar(ptxParseData parseData, macroElwVar evar, int value)
{
    parseData->elwVars[evar] = value;
}

void initMacroInstrTypes(ptxParseData parseData, ptxType instrType[], int n)
{
    int i;
    ptxType type;

    parseData->nrInstrTypes = n;

    for(i = 0; i < n; i++) {
        type = instrType[i];
        parseData->instrTypes[i] = getPtxTypeEnum(type);
    }
}

void initMacroNumInstrArgs(ptxParseData parseData, uInt nrofArguments)
{
    parseData->nrArgs = nrofArguments;
}

void initInlineFunctionArgs(ptxParseData parseData, uInt nrofIArguments, uInt nrofOArguments, 
                            String funcName, msgSourcePos_t sourcePos)
{
    stdCHECK_WITH_POS(nrofIArguments < ptxMAX_INLINE_FUNCTION_INPUT_ARGS, 
                                (ptxMsgTooManyArgsInlineFunc, sourcePos,
                                   "input", nrofIArguments, funcName,
                                   ptxMAX_INLINE_FUNCTION_INPUT_ARGS));

    stdCHECK_WITH_POS(nrofOArguments < ptxMAX_INLINE_FUNCTION_OUTPUT_ARGS, 
                                (ptxMsgTooManyArgsInlineFunc, sourcePos,
                                   "return", nrofOArguments, funcName,
                                   ptxMAX_INLINE_FUNCTION_OUTPUT_ARGS));

    parseData->nrInlineFuncInputArgs = nrofIArguments;
    parseData->nrInlineFuncRetArgs   = nrofOArguments;
}

void initMacroInstrArgs(ptxParseData parseData, String s, int index)
{
    parseData->arg[index] = (String) stdMALLOC(strlen(s) + 1);
    strcpy(parseData->arg[index], s);
}

void setInlineFunctionInputArg(ptxParseData parseData, int index, String s, ptxExpression e)
{
    parseData->inlineInputArgName[index] = (String) stdMALLOC(strlen(s) + 1);
    strcpy(parseData->inlineInputArgName[index], s);
    parseData->inlineInputArg[index] = e;
}

void setInlineFunctionOutputArg(ptxParseData parseData, int index, String s, ptxExpression e)
{
    parseData->inlineOutputArgName[index] = (String) stdMALLOC(strlen(s) + 1);
    strcpy(parseData->inlineOutputArgName[index], s);
    parseData->inlineOutputArg[index] = e;
}

void initMacroInstrGuard(ptxParseData parseData, String s)
{
    parseData->guardStr = (String) stdMALLOC(strlen(s) + 1);
    strcpy(parseData->guardStr, s);
}

void initMacroPredicateOutput(ptxParseData parseData, String s)
{
    parseData->predOutStr = (String) stdMALLOC(strlen(s) + 1);
    strcpy(parseData->predOutStr, s);
}

void initMacroUtilFuncParseState(ptxParsingState parseState)
{
   int i;

   for (i=0; i < parseState->numMacroUtilFunc; i++)
      parseState->utilFuncs[i].parseState = NOT_PARSED;
}

void scheduleMacroUtilFuncForParsing(String name, ptxParsingState parseState)
{
    int idx = (int) (long long) mapApply(parseState->macroUtilFuncMap, name);

    if (idx == 0) // NULL entry
       return;

    if (parseState->utilFuncs[idx].parseState != DONE_PARSE) { 
       parseState->utilFuncs[idx].parseState = NEED_PARSE;
    }
}

Bool getPendingMacroUtilFuncList(stdList_t* funcList, ptxParsingState parseState)
{
   int i;
   Bool addedFunc = False;

   for (i=1; i < parseState->numMacroUtilFunc; i++)
   {
       if (parseState->utilFuncs[i].parseState == NEED_PARSE)
       {
           listAddTo((String)(parseState->utilFuncs[i].funcBody), funcList);
           parseState->utilFuncs[i].parseState = DONE_PARSE;
           addedFunc = True;
       } 
   }
   return addedFunc;
}

void freeMacroState(ptxParseData parseData, int nrofArguments)
{
    // OPTIX_HAND_EDIT Fix signed/unsigned comparison warnings
    int i;
    for (i = 0; i < nrofArguments; i++) {
        stdFREE(parseData->arg[i]);
    }
    stdFREE(parseData->guardStr);
    stdFREE(parseData->predOutStr);
}

void freeInlineFunctionArgs(ptxParseData parseData)
{
    // OPTIX_HAND_EDIT Fix signed/unsigned comparison warnings
    int i;
    for (i = 0; i < parseData->nrInlineFuncInputArgs; i++) {
        stdFREE(parseData->inlineInputArgName[i]);
        parseData->inlineInputArg[i] = NULL;
    }
    for (i = 0; i < parseData->nrInlineFuncRetArgs; i++) {
        stdFREE(parseData->inlineOutputArgName[i]);
        parseData->inlineOutputArg[i] = NULL;
    }
    stdFREE(parseData->guardStr);
}

Bool IsMacroFunc(String name, ptxParsingState parseState)
{
    return mapIsDefined(parseState->macroUtilFuncMap, name);
}

Bool IsUniqueMacroFunc(ptxSymbolTableEntry symEnt, ptxParsingState parseState)
{
    return symEnt->aux->isUnique && IsMacroFunc(symEnt->symbol->name, parseState);
}

Bool IsNonUniqueMacroFunc(ptxSymbolTableEntry symEnt, ptxParsingState parseState) {
    return IsMacroFunc(symEnt->symbol->name, parseState) && !IsUniqueMacroFunc(symEnt, parseState);
}

ptxExpressionKind getExpressionKindForArg(ptxParseData parseData, uInt n, Bool isRetArg)
{
    // OPTIX_HAND_EDIT Fix signed/unsigned comparison warnings
    if (isRetArg) {
        stdASSERT((int)n < parseData->nrInlineFuncRetArgs, ("Accessing invalid return argument index"));
        return parseData->inlineOutputArg[n]->kind;
    } else {
        stdASSERT((int)n < parseData->nrInlineFuncInputArgs, ("Accessing invalid input argument index"));
        return parseData->inlineInputArg[n]->kind;
    }
}

int getConstantValueOfInputArg(ptxParseData parseData, uInt n)
{
    ptxExpression argN = parseData->inlineInputArg[n];
    stdASSERT(argN->kind == ptxIntConstantExpression, ("Integer constant expected"));
    return argN->cases.IntConstant.i;
}

/*  The ROT13 technique has following special property:
 *  `string == obfuscate ( obfuscate ( string ) )`
 *  To obtain original string, it is sufficient to again obfuscate the already
 *  obfuscated string. In other words, __deObfuscate function need not do
 *  anything additional to what __obfuscate (or encoding) does.
 *
 *  The corresponding __obfuscate is defined in the `obfuscateSensitiveStrings`
 *  perl script. Changes to the definition of the following function needs to be
 *  propagated in the `encode_string` function (defined in the above perl script).
 */
String __deObfuscate(stdMap_t* deobfuscatedStringMapPtr, cString encodeName)
{
    int length, i;
    String origName;
    
    stdASSERT(deobfuscatedStringMapPtr != NULL, ("deobfuscatedStringMapPtr is null"));
    origName = (String) mapApply(*deobfuscatedStringMapPtr, (Pointer)encodeName);
    if (origName) {
        return origName;
    }
    length = strlen(encodeName);
    origName = stdMALLOC(length + 1);
    for (i = 0; i < length; i++) {
        char c = encodeName[i];
        // callwlate ROT13 of character.
        if ((c >= 'A' && c <= 'M') || (c >= 'a' && c <= 'm')) {
            origName[i] = c + 13;
        } else if ((c >= 'N' && c <= 'Z') || (c >= 'n' && c <= 'z')) {
            origName[i] = c - 13;
        } else {
            origName[i] = encodeName[i];
        }
    }
    origName[i] = '\0';
    mapDefine(*deobfuscatedStringMapPtr, (Pointer)encodeName, origName);
    return origName;
}
