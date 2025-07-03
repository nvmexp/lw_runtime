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
|*  Module: input.h                                                           *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _INPUT_H
#define _INPUT_H

//******************************************************************************
//
//  Forwards
//
//******************************************************************************



//******************************************************************************
//
//  Regular Expressions
//
//******************************************************************************
#define MASMEXPR            "^(([+-]?)((0x){1}([a-f0-9x]+))|(([a-f0-9x]+)(h)?)|((0n){1}([0-9]+))|((0t){1}([0-7x]+))|((0y){1}([0-1x]+))"
#define HEXEXPR             "|([a-f0-9x]+))$"
#define DECEXPR             "|([0-9]+))$"
#define OCTEXPR             "|([0-7x]+))$"
#define UNKEXPR             ")$"
#define MASM_HEX_PREFIX     4
#define MASM_HEX_CONSTANT   5
#define MASM_HEXA_CONSTANT  7
#define MASM_HEXA_POSTFIX   8
#define MASM_DEC_PREFIX     10
#define MASM_DEC_CONSTANT   11
#define MASM_OCT_PREFIX     13
#define MASM_OCT_CONSTANT   14
#define MASM_BIN_PREFIX     16
#define MASM_BIN_CONSTANT   17

#define CPPEXPR             "^(([+-]?)((0x){1}([a-f0-9x]+))|((0){1}([0-7x]+))|([0-9]+)((l|u|i64)?))$"
#define CPP_HEX_PREFIX      4
#define CPP_HEX_CONSTANT    5
#define CPP_OCT_PREFIX      7
#define CPP_OCT_CONSTANT    8
#define CPP_DEC_CONSTANT    9
#define CPP_DEC_POSTFIX     11

//******************************************************************************
//
//  Input Hook Class
//
//******************************************************************************
class CInputHook : public CHook
{
public:
                        CInputHook() : CHook()  {};
virtual                ~CInputHook()            {};

        // Input hook methods
virtual HRESULT         initialize(const PULONG pVersion, const PULONG pFlags);
virtual void            uninitialize(void);

}; // class CInputHook

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  void            expressionInput(PULONG64 pValue, PULONG64 pMask, PSTR pString);
extern  void            booleanInput(PULONG64 pValue, PULONG64 pMask, PSTR pString);

extern  void            verboseInput(PULONG64 pValue, PULONG64 pMask, PSTR pString);

extern  void            stringInput(PULONG64 pValue, PULONG64 pMask, PSTR pString);
extern  void            fileInput(PULONG64 pValue, PULONG64 pMask, PSTR pString);

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _INPUT_H
