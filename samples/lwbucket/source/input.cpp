 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: input.cpp                                                         *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  bool            isConstant(PULONG64 pValue, PULONG64 pMask, PSTR pString);
static  bool            isMasmConstant(PULONG64 pValue, PULONG64 pMask, PSTR pString);
static  bool            isCppConstant(PULONG64 pValue, PULONG64 pMask, PSTR pString);
static  ULONG64         constantValue(CString sPrefix, CString sConstant, CString sPostfix, ULONG ulRadix);
static  ULONG64         constantMask(CString sPrefix, CString sConstant, CString sPostfix, ULONG ulRadix);

//******************************************************************************
//
// Locals
//
//******************************************************************************
static  CInputHook      s_inputHook;

static  regex_t         s_reMasmHex     = {0};
static  regex_t         s_reMasmDec     = {0};
static  regex_t         s_reMasmOct     = {0};
static  regex_t         s_reMasmUnk     = {0};

static  regex_t         s_reCpp         = {0};
static  regex_t         s_reFile        = {0};

static  CString         s_sReplacementPolicy[] = {
/* 0x0 */                                         "lru",    // Least recently used (LRU) replacement policy string
/* 0x1 */                                         "mru",    // Most recently used (MRU) replacement policy string
                                                 };

static  CString         s_sWritePolicy[] = {
/* 0x0 */                                   "writeback",    // Write back write policy string
/* 0x1 */                                   "writethru",    // Write thru write policy string
                                           };

static  CString         s_sMissPolicy[] = {
/* 0x0 */                                  "writeallocate", // Write allocate miss policy string
/* 0x1 */                                  "writearound",   // Write around miss policy string
                                          };

//******************************************************************************

void
expressionInput
(
    PULONG64            pValue,
    PULONG64            pMask,
    PSTR                pString
)
{
    assert(pValue != NULL);
    assert(pMask != NULL);
    assert(pString != NULL);

    // Check to see if the expression is not a valid constant
    if (!isConstant(pValue, pMask, pString))
    {
        // Get the argument value (In the current radix) [Force mask to unmasked]
        *pValue = getExpression(pString);
        *pMask  = 0xffffffffffffffff;
    }

} // expressionInput

//******************************************************************************

void
booleanInput
(
    PULONG64            pValue,
    PULONG64            pMask,
    PSTR                pString
)
{
    CString             sRegularExpression;
    ULONG64             ulBoolealwalue;
    ULONG64             ulBooleanMask;

    // Check to see if the input string is a regular expression
    if (isRegularExpression(pString))
    {
        // Get the regular expression string from the input string
        sRegularExpression = getRegularExpression(pString);

        // Try to get the boolean value from the given regular expression
        ulBoolealwalue = boolealwalue(sRegularExpression);
        if (ulBoolealwalue == ILWALID_BOOLEAN_VALUE)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Unmatched regular expression '%s'",
                             sRegularExpression);
        }
        // Force boolean mask value
        ulBooleanMask = 0xffffffffffffffff;
    }
    else    // Input is not a (forced) regular expression
    {
        // Catch any expression errors
        try
        {
            // Get the boolean input value and mask
            expressionInput(&ulBoolealwalue, &ulBooleanMask, pString);
        }
        catch (CException& eError)
        {
            // Check for an invalid parameter exception
            if (eError.hResult() == E_ILWALIDARG)
            {
                // Expression error, try and treat string as a regular expression
                sRegularExpression = pString;

                // Try to get the boolean value from the regular expression
                ulBoolealwalue = boolealwalue(sRegularExpression);
                if (ulBoolealwalue == ILWALID_BOOLEAN_VALUE)
                {
                    throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                     ": Invalid expression '%s'",
                                     pString);
                }
                // Force boolean mask value
                ulBooleanMask = 0xffffffffffffffff;
            }
            else    // Other expression error
            {
                throw;
            }
        }
    }
    // Include the new boolean value and mask in the existing input value and mask
    *pValue |= ulBoolealwalue;
    *pMask  &= ulBooleanMask;

} // booleanInput

//******************************************************************************

bool
isConstant
(
    PULONG64            pValue,
    PULONG64            pMask,
    PSTR                pString
)
{
    ULONG               ulExpressionSyntax;
    HRESULT             hResult;
    bool                bConstant = false;

    // Try to get the current expression syntax
    hResult = GetExpressionSyntax(&ulExpressionSyntax);
    if (SUCCEEDED(hResult))
    {
        // Check for MASM vs C++ expression syntax        
        if (ulExpressionSyntax == DEBUG_EXPR_MASM)
        {
            // Check for a valid MASM constant
            bConstant = isMasmConstant(pValue, pMask, pString);
        }
        else if (ulExpressionSyntax == DEBUG_EXPR_CPLUSPLUS)
        {
            // Check for a valid C++ constant
            bConstant = isCppConstant(pValue, pMask, pString);
        }
        else    // Unknown expression syntax
        {
            // Throw exception indicating unknown expression syntax
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Unknown expression syntax!");
        }
    }
    else    // Unable to get current expression syntax
    {
        // Throw exception indicating error getting expression syntax
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Unable to get current expression syntax!");
    }
    return bConstant;

} // isConstant

//******************************************************************************

bool
isMasmConstant
(
    PULONG64            pValue,
    PULONG64            pMask,
    PSTR                pString
)
{
    regex_t            *pRegEx;
    regmatch_t          reMatch[25];
    CString             sSign("");
    CString             sPrefix("");
    CString             sPostfix("");
    CString             sNumber("");
    CString             sConstant("");
    ULONG               ulBase;
    ULONG               ulRadix;
    int                 reResult;
    bool                bMasmConstant = false;

    // Expression parsing affected by current radix, get current radix
    GetRadix(&ulRadix);

    // Get the right MASM regular expression based on current radix
    switch(ulRadix)
    {
        case 16:                        // Base 16 - Hexadecimal

            // Setup hexadecimal MASM regular expression
            pRegEx = &s_reMasmHex;
            break;

        case 10:                        // Base 10 - Decimal

            // Setup decimal MASM regular expression
            pRegEx = &s_reMasmDec;
            break;

        case 8:                         // Base 8 - Octal

            // Setup octal MASM regular expression
            pRegEx = &s_reMasmOct;
            break;

        default:                        // Unknown base

            // Warn about unknown radix and complete MASM regular expression (Only allow forced radix values)
            dPrintf("Unknown radix value (%d)!\n", ulRadix);

            // Setup unknown radix MASM regular expression
            pRegEx = &s_reMasmUnk;
            break;
    }
    // Check to see if string is a MASM constant
    reResult = regexec(pRegEx, pString, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Default to no pre/postfix and current radix
        sPrefix   = "";
        sPostfix  = "";
        sConstant = pString;
        ulBase    = ulRadix;

        // Try to find the correct radix for the constant
        if (reMatch[MASM_HEX_PREFIX].rm_so != -1)
        {
            // Prefix 0x specifying hexadecimal
            sPrefix   = subExpression(pString, reMatch, MASM_HEX_PREFIX);
            sConstant = subExpression(pString, reMatch, MASM_HEX_CONSTANT);
            sPostfix  = "";
            ulBase    = 16;
        }
        else if (reMatch[MASM_HEXA_POSTFIX].rm_so != -1)
        {
            // Postfix h specifying hexadecimal
            sPrefix   = "";
            sConstant = subExpression(pString, reMatch, MASM_HEXA_CONSTANT);
            sPostfix  = subExpression(pString, reMatch, MASM_HEXA_POSTFIX);
            ulBase    = 16;
        }
        else if (reMatch[MASM_DEC_PREFIX].rm_so != -1)
        {
            // Prefix 0n specifying decimal
            sPrefix   = subExpression(pString, reMatch, MASM_DEC_PREFIX);
            sConstant = subExpression(pString, reMatch, MASM_DEC_CONSTANT);
            sPostfix  = "";
            ulBase    = 10;
        }
        else if (reMatch[MASM_OCT_PREFIX].rm_so != -1)
        {
            // Prefix 0t specifying octal
            sPrefix   = subExpression(pString, reMatch, MASM_OCT_PREFIX);
            sConstant = subExpression(pString, reMatch, MASM_OCT_CONSTANT);
            sPostfix  = "";
            ulBase    = 8;
        }
        else if (reMatch[MASM_BIN_PREFIX].rm_so != -1)
        {
            // Prefix 0y specifying binary
            sPrefix   = subExpression(pString, reMatch, MASM_BIN_PREFIX);
            sConstant = subExpression(pString, reMatch, MASM_BIN_CONSTANT);
            sPostfix  = "";
            ulBase    = 2;
        }
        // Try to get the constant value and mask for this MASM constant
        *pValue = constantValue(sPrefix, sConstant, sPostfix, ulBase);
        *pMask  = constantMask(sPrefix, sConstant, sPostfix, ulBase);

        // Indicate this is a masm constant
        bMasmConstant = true;
    }
    return bMasmConstant;

} // isMasmConstant

//******************************************************************************

bool
isCppConstant
(
    PULONG64            pValue,
    PULONG64            pMask,
    PSTR                pString
)
{
    regmatch_t          reMatch[25];
    CString             sSign("");
    CString             sPrefix("");
    CString             sPostfix("");
    CString             sNumber("");
    CString             sConstant("");
    ULONG               ulBase;
    ULONG               ulRadix;
    int                 reResult;
    bool                bCppConstant = false;

    // Expression parsing affected by current radix, get current radix
    ulRadix = GetRadix(&ulRadix);

    // Check to see if string is a C++ constant
    reResult = regexec(&s_reCpp, pString, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Default to no pre/postfix and current radix
        sPrefix   = "";
        sPostfix  = "";
        sConstant = pString;
        ulBase    = ulRadix;

        // Try to find the correct radix for the constant
        if (reMatch[CPP_HEX_PREFIX].rm_so != -1)
        {
            // Prefix 0x specifying hexadecimal
            sPrefix   = subExpression(pString, reMatch, CPP_HEX_PREFIX);
            sConstant = subExpression(pString, reMatch, CPP_HEX_CONSTANT);
            sPostfix  = "";
            ulBase    = 16;
        }
        else if (reMatch[CPP_OCT_PREFIX].rm_so != -1)
        {
            // Prefix 0 specifying octal
            sPrefix   = subExpression(pString, reMatch, CPP_OCT_PREFIX);
            sConstant = subExpression(pString, reMatch, CPP_OCT_CONSTANT);
            sPostfix  = "";
            ulBase    = 8;
        }
        else if (reMatch[CPP_DEC_POSTFIX].rm_so != -1)
        {
            // Postfix L/U/I64
            sPrefix   = "";
            sConstant = subExpression(pString, reMatch, CPP_DEC_CONSTANT);
            sPostfix  = subExpression(pString, reMatch, CPP_DEC_POSTFIX);
            ulBase    = 10;
        }
        // Try to get the constant value and mask for this MASM constant
        *pValue = constantValue(sPrefix, sConstant, sPostfix, ulBase);
        *pMask  = constantMask(sPrefix, sConstant, sPostfix, ulBase);

        // Indicate this is a C++ constant
        bCppConstant = true;
    }
    return bCppConstant;

} // isCppConstant

//******************************************************************************

static ULONG64
constantValue
(
    CString             sPrefix,
    CString             sConstant,
    CString             sPostfix,
    ULONG               ulRadix
)
{
    UNREFERENCED_PARAMETER(ulRadix);

    CString             sValue;
    int                 nLocation;
    ULONG64             ulValue;

    // Initialize value string to constant string
    sValue = sConstant;

    // Make sure value string is lower case
    sValue.lower();

    // Replace mask characters with 0 character
    nLocation = static_cast<int>(sValue.find('x', 0));
    while(nLocation != NOT_FOUND)
    {
        // Replace this mask character with 0
        sValue[nLocation] = '0';

        // Check for more oclwrances
        nLocation = static_cast<int>(sValue.find('x', nLocation));
    }
    // Generate the actual value string (Prefix, Constant, Postfix)
    sValue = sPrefix + sValue + sPostfix;

    // Get the constant value
    ulValue = getExpression(sValue);

    return ulValue;

} // constantValue

//******************************************************************************

static ULONG64
constantMask
(
    CString             sPrefix,
    CString             sConstant,
    CString             sPostfix,
    ULONG               ulRadix
)
{
    CString             sMask;
    char                cReplace;
    int                 nLocation;
    ULONG64             ulMask;

    // Initialize mask string to constant string
    sMask = sConstant;

    // Make sure mask string is lower case
    sMask.lower();

    // Compute mask character replacement based on radix
    switch(ulRadix)
    {
        case 16:        cReplace = 'f'; break;
        case 10:        cReplace = '9'; break;
        case 8:         cReplace = '7'; break;
        case 2:         cReplace = '1'; break;
        default:        cReplace = '0'; break;
    }
    // Replace characters for mask generation
    for (nLocation = 0; nLocation < static_cast<int>(sMask.length()); nLocation++)
    {
        if (sMask[nLocation] == 'x')
            sMask[nLocation] = cReplace;
        else
            sMask[nLocation] = '0';
    }
    // Generate the actual mask string (Prefix, Constant, Postfix)
    sMask = sPrefix + sMask + sPostfix;

    // Get the constant mask
    ulMask = ~getExpression(sMask);

    return ulMask;

} // constantMask

//******************************************************************************

void
verboseInput
(
    PULONG64            pValue,
    PULONG64            pMask,
    PSTR                pString
)
{
    CString             sRegularExpression;
    ULONG64             ulVerboseValue;
    ULONG64             ulVerboseMask;

    assert(pValue != NULL);
    assert(pMask != NULL);
    assert(pString != NULL);

    // Check to see if the input string is a regular expression
    if (isRegularExpression(pString))
    {
        // Get the regular expression string from the input string
        sRegularExpression = getRegularExpression(pString);

        // Try to get the verbose value from the given regular expression
        ulVerboseValue = verboseValue(sRegularExpression);
        if (ulVerboseValue == ILWALID_VERBOSE_VALUE)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Unmatched regular expression '%s'",
                             sRegularExpression);
        }
        // Force verbose mask value
        ulVerboseMask = 0xffffffffffffffff;
    }
    else    // Input is not a (forced) regular expression
    {
        // Catch any expression errors
        try
        {
            // Get the verbose input value and mask
            expressionInput(&ulVerboseValue, &ulVerboseMask, pString);
        }
        catch (CException& eError)
        {
            // Check for an invalid parameter exception
            if (eError.hResult() == E_ILWALIDARG)
            {
                // Expression error, try and treat string as a regular expression
                sRegularExpression = pString;

                // Try to get the verbose value from the regular expression
                ulVerboseValue = verboseValue(sRegularExpression);
                if (ulVerboseValue == ILWALID_VERBOSE_VALUE)
                {
                    throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                     ": Invalid expression '%s'",
                                     pString);
                }
                // Force verbose mask value
                ulVerboseMask = 0xffffffffffffffff;
            }
            else    // Other expression error
            {
                throw;
            }
        }
    }
    // Include the new verbose value and mask in the existing input value and mask
    *pValue |= ulVerboseValue;
    *pMask  &= ulVerboseMask;

} // verboseInput

//******************************************************************************

void
stringInput
(
    PULONG64            pValue,
    PULONG64            pMask,
    PSTR                pString
)
{
    UNREFERENCED_PARAMETER(pMask);

    CString             sStringExpression;

    assert(pValue != NULL);
    assert(pMask != NULL);
    assert(pString != NULL);

    // Check to see if the input string is a string expression
    if (isStringExpression(pString))
    {
        // Get the string expression string from the input string
        sStringExpression = getStringExpression(pString);
    }
    else    // Input is not a string expression
    {
        // Set string expression to the input string
        sStringExpression = CString(pString);
    }
    // Save pointer to string as argument value
    *pValue = reinterpret_cast<ULONG64>(pString);

} // stringInput

//******************************************************************************

void
fileInput
(
    PULONG64            pValue,
    PULONG64            pMask,
    PSTR                pString
)
{
    UNREFERENCED_PARAMETER(pMask);

    regmatch_t          reMatch[25];
    int                 reResult;
    CString             sStringExpression;

    assert(pValue != NULL);
    assert(pMask != NULL);
    assert(pString != NULL);

    // Check to see if the input string is a string expression
    if (isStringExpression(pString))
    {
        // Get the string expression string from the input string
        sStringExpression = getStringExpression(pString);
    }
    else    // Input is not a string expression
    {
        // Set string expression to the input string
        sStringExpression = CString(pString);
    }
    // Check to see if the given string appears to be a filename string
    reResult = regexec(&s_reFile, sStringExpression, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {    
        // Save pointer to filename string as argument value
        *pValue = reinterpret_cast<ULONG64>(pString);
    }
    else    // Input string doesn't appear to be a filename
    {
        // Throw exception indicating invalid filename syntax
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": %s doesn't appear to be a valid filename!",
                         sStringExpression);
    }

} // fileInput

//******************************************************************************

HRESULT
CInputHook::initialize
(
    const PULONG        pVersion,
    const PULONG        pFlags
)
{
    UNREFERENCED_PARAMETER(pVersion);
    UNREFERENCED_PARAMETER(pFlags);

    regex_t            *pRegEx = NULL;
    const char         *pRegExpr = NULL;
    int                 reResult;
    HRESULT             hResult = S_OK;

    assert(pVersion != NULL);
    assert(pFlags != NULL);

    // Try to compile MASM hexadecimal base regular expression
    reResult = regcomp(&s_reMasmHex, MASMEXPR HEXEXPR, REG_EXTENDED | REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Try to compile MASM decimal base regular expression
        reResult = regcomp(&s_reMasmDec, MASMEXPR DECEXPR, REG_EXTENDED | REG_ICASE);
        if (reResult == REG_NOERROR)
        {
            // Try to compile MASM octal base regular expression
            reResult = regcomp(&s_reMasmOct, MASMEXPR OCTEXPR, REG_EXTENDED | REG_ICASE);
            if (reResult == REG_NOERROR)
            {
                // Try to compile MASM unknown base regular expression
                reResult = regcomp(&s_reMasmUnk, MASMEXPR UNKEXPR, REG_EXTENDED | REG_ICASE);
                if (reResult == REG_NOERROR)
                {
                    // Try to compile C++ regular expression
                    reResult = regcomp(&s_reCpp, CPPEXPR, REG_EXTENDED | REG_ICASE);
                    if (reResult == REG_NOERROR)
                    {
                        // Try to compile file regular expression
                        reResult = regcomp(&s_reFile, FILEEXPR, REG_EXTENDED | REG_ICASE);
                        if (reResult != REG_NOERROR)
                        {
                            // Save regular expression error
                            pRegEx   = &s_reFile;
                            pRegExpr = FILEEXPR;
                        }
                    }
                    else    // Error compiling C++ regular expression
                    {
                        // Save regular expression error
                        pRegEx   = &s_reCpp;
                        pRegExpr = CPPEXPR;
                    }
                }
                else    // Error compiling MASM unknown base regular expression
                {
                    // Save regular expression error
                    pRegEx   = &s_reMasmUnk;
                    pRegExpr = MASMEXPR UNKEXPR;
                }
            }
            else    // Error compiling MASM octal base regular expresssion
            {
                // Save regular expression error
                pRegEx   = &s_reMasmOct;
                pRegExpr = MASMEXPR OCTEXPR;
            }
        }
        else    // Error compiling MASM decimal base regular expression
        {
            // Save regular expression error
            pRegEx   = &s_reMasmDec;
            pRegExpr = MASMEXPR DECEXPR;
        }
    }
    else    // Error compiling MASM hexadecimal base regular expression
    {
        // Save regular expression error
        pRegEx   = &s_reMasmHex;
        pRegExpr = MASMEXPR HEXEXPR;
    }
    // Check for any errors compiling the regular expressions
    if (reResult != REG_NOERROR)
    {
        // Display regular expression error to the user
        dPrintf("%s\n", regString(reResult, pRegEx, pRegExpr));

        // Free any allocated input resources
        uninitialize();

        // Indicate failure
        hResult = E_FAIL;
    }
    return hResult;

} // initialize

//******************************************************************************

void
CInputHook::uninitialize(void)
{
    // Free any allocated regular expressions
    regfree(&s_reFile);
    regfree(&s_reCpp);

    regfree(&s_reMasmUnk);
    regfree(&s_reMasmOct);
    regfree(&s_reMasmDec);
    regfree(&s_reMasmHex);

} // uninitialize

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
