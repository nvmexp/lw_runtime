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
 *  Module name              : ptx.h
 *
 *  Description              :
 *
 */

#ifndef ptx_INCLUDED
#define ptx_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "ptxIR.h"
#include "ptxConstructors.h"
#include "ptxInstructionTemplates.h"
#include "stdLocal.h"
#include "stdObfuscate.h"

#ifdef __cplusplus
extern "C" {
#endif

/*------------------------------ Definitions ---------------------------------*/

extern cString               ptxfilename;
void             ptxerror(void* scanner, ptxParsingState ptxIR, String s);
void             ptxcleanup(void);
#define YY_DECL  int ptxlex(YYSTYPE * yylval_param, void *yyscanner,  ptxParsingState gblState)
void             ptxInitLexState(ptxParsingState parseState);
void             ptxDestroyLexState(ptxParsingState parseState);
int              ptxparse(void *scanner, ptxParsingState parseState);
int              ptxget_lineno (void *yyscanner );
void             ptxset_lineno (int line_number ,void *yyscanner);
msgSourcePos_t   ptxsLwrPos(ptxParsingState gblState);
msgSourcePos_t   ptxUserPos(ptxParsingState parseState, msgSourcePos_t *pos);
// OPTIX_HAND_EDIT
void             ptxPushInput(ptxParsingState parseState, String s, uInt32 strLength, stdObfuscationState state, cString newFileName, uInt newLineNo, void* decrypter, GenericCallback decryptionCB);
void             ptxPushMacroBody(String s, String fileName, uInt lineNo, ptxParsingState parseState, void *scanner);
String           ptxReadMacroBody(ptxParsingState gblState);
void             ptxInitScanner(ptxParsingState gblState);
void             ptxDefinePreprocessorMacro(ptxParsingState gblState, String name, String value);
Bool             isPtxUserInput(cString fileName);

/*
 * Function        : Used to initialize the ptxfilename and ptxlineno and add to include stack of ptxFileSourceStruct
 * Parameters      : gblState     (I) Current ptx parsing state. 
 *                   newFileName  (I) name of a file which is about to be parsed
 *                   newLineNo    (I) line number of the file from which parsing would start
 *                                    This is always 1.
 * Note            : Affects ptxlineno and ptxfilename as a side effect
 */
void             ptxInitLwrrentSourcePos(ptxParsingState gblState, String newFileName, uInt newLineNo);

/*
 * Function        : Used to push a new input file/string for parsing and add to include stack of ptxFileSourceStruct
 * Parameters      : gblState     (I) Current ptx parsing state. 
 *                 : newFileName  (I) name of a file / desc. of the string which is about to be parsed
 *                   newLineNo    (I) line number of the file from which parsing would start
 * Note            : Affects ptxlineno and ptxfilename as a side effect
 */
void             ptxPushSourcePos(ptxParsingState gblState, cString newFileName, uInt newLineNo);

#if     defined(__cplusplus)
}
#endif

#endif
