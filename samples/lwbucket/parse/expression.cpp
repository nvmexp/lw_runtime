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
|*  Module: expression.cpp                                                    *|
|*                                                                            *|
 \****************************************************************************/
#include "parprecomp.h"

//******************************************************************************
//
//  Locals
//
//******************************************************************************
// Current input type value
static  ULONG                   s_ulInputType;                  // Current input type value

//******************************************************************************

ULONG64
getExpression
(
    PCSTR               pExpression,
    ULONG               ulBase,
    ULONG               ulDesiredType
)
{
    HRESULT             hResult;
    ULONG               ulRadix = 0;
    ULONG               ulInputType = 0;
    ULONG64             ulValue = 0;
    DEBUG_VALUE         Value;

    assert(pExpression != NULL);

    // Check for a requested base (Non-zero)
    if (ulBase != 0)
    {
        // Get the current radix value
        hResult = GetRadix(&ulRadix);
        if (FAILED(hResult))
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": Unable to get current radix");
        }
        // Set the requested radix (Base)
        hResult = SetRadix(ulBase);
        if (FAILED(hResult))
        {
            throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                             ": Unable to set current radix to %d",
                             ulBase);
        }
    }
    // Check for a desired input type
    if (ulDesiredType != DEBUG_VALUE_ILWALID)
    {
        // Set the desired input type (Returns original input type)
        ulInputType = setInputType(ulDesiredType);
    }
    // Try to evaluate the expression
    hResult = Evaluate(pExpression, getInputType(), &Value, NULL);
    if (FAILED(hResult))
    {
        // Restore the original radix value (If changed)
        if (ulBase != 0)
        {
            SetRadix(ulRadix);
        }
        // Restore the original input type (If changed)
        if (ulDesiredType != DEBUG_VALUE_ILWALID)
        {
            setInputType(ulInputType);
        }
        // Check for a ctrl-break from user (Otherwise throw expression exception)
        if (hResult != STATUS_CONTROL_BREAK_EXIT)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Invalid expression '%s'",
                             pExpression);
        }
    }
    else    // Got the expression value
    {
        // Get the expression value
        ulValue = Value.I64;
    }
    // Restore the original radix value (If changed)
    if (ulBase != 0)
    {
        SetRadix(ulRadix);
    }
    // Restore the original input type (If changed)
    if (ulDesiredType != DEBUG_VALUE_ILWALID)
    {
        setInputType(ulInputType);
    }
    return ulValue;

} // getExpression

//******************************************************************************

ULONG
getInputType()
{
    // Return the current input type value
    return s_ulInputType;

} // getInputType

//******************************************************************************

ULONG
setInputType
(
    ULONG               ulInputType
)
{
    ULONG               ulOriginalType = s_ulInputType;

    // Check for a valid input type (Only supports integral input types and natural [INVALID])
    switch(ulInputType)
    {
        case DEBUG_VALUE_ILWALID:               // Invalid (Natural) input type
        case DEBUG_VALUE_INT8:                  // 8-bit integral input type
        case DEBUG_VALUE_INT16:                 // 16-bit integral input type
        case DEBUG_VALUE_INT32:                 // 32-bit integral input type
        case DEBUG_VALUE_INT64:                 // 64-bit integral input type

            break;

        default:                                // Unknown/invalid input type

            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Invalid input type '%d'",
                             ulInputType);
    }
    // Set the new validated input type
    s_ulInputType = ulInputType;

    // Return the original input type value
    return ulOriginalType;

} // setInputType

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
