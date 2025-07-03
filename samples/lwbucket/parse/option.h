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
|*  Module: option.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OPTION_H
#define _OPTION_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
//  Options - All commands that share code *must* use the same options
enum Option
{
    UnknownOption = 0,                      // Unknown command option (0)
    DisableOption,                          // Disable command option
    DmlOption,                              // DML command option
    EnableOption,                           // Enable command option
    HelpOption,                             // Help command option
    QuietOption,                            // Quiet command option
    VerboseOption,                          // Verbose command option
    ExchangeOption,                         // Exchange command option
    QuestionOption,                         // Question/alternate help command option

    OptionCount                             // Number of command options

}; // Option

//******************************************************************************
//
// Function Pointers
//
//******************************************************************************
typedef HRESULT (*POPTION_ROUTINE)(int nOption, int nLastOption, int hasArg);
typedef void    (*POPTION_INPUT)(PULONG64 pValue, PULONG64 pMask, PSTR pString);

//******************************************************************************
//
// Structures
//
//******************************************************************************
typedef struct _OPTION_DATA
{
    int             hasArg;                 // Option has argument value
    POPTION_INPUT   pOptionInput;           // Option input processing routine
    POPTION_ROUTINE pOptionRoutine;         // Option custom processing routine

} OPTION_DATA, *POPTION_DATA;

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  void            initializeOptions();

extern  HRESULT         parseOptions(int argc, char **argv);

extern  void            setDefaultOption(LONG lOption, int arguments = no_argument, POPTION_ROUTINE pOptionRoutine = NULL);
extern  void            addOption(LONG lOption, int arguments);
extern  void            addOption(LONG lOption, int arguments, POPTION_INPUT pOptionInput);
extern  void            addOption(LONG lOption, int arguments, POPTION_ROUTINE pOptionRoutine);
extern  void            addOption(LONG lOption, int arguments, POPTION_INPUT pOptionInput, POPTION_ROUTINE pOptionRoutine);

extern  void            setSortType(LONG lOption, bool bSortOrder);

extern  CString         getCommandOptions();
extern  CString         getSearchOptions();
extern  CString         getSortOptions();

extern  ULONG64         commandValue(LONG lOption);
extern  PULONG64        commandAddress(LONG lOption);
extern  void            setCommandValue(LONG lOption, ULONG64 ulValue);

extern  ULONG64         searchValue(LONG lOption);
extern  PULONG64        searchAddress(LONG lOption);
extern  void            setSearchValue(LONG lOption, ULONG64 ulValue);

extern  ULONG64         maskValue(LONG lOption);
extern  PULONG64        maskAddress(LONG lOption);
extern  void            setMaskValue(LONG lOption, ULONG64 ulValue);

extern  ULONG64         countValue(LONG lOption);
extern  PULONG64        countAddress(LONG lOption);
extern  void            setCountValue(LONG lOption, ULONG64 ulValue);

extern  LONG            sortValue(LONG lSortCount);
extern  void            setSortValue(LONG lOption, LONG lValue);

extern  bool            sortOrder(LONG lSortCount);
extern  void            setSortOrder(LONG lSortCount, bool bValue);

extern  LONG            sortCount();
extern  void            setSortCount(LONG lSortCount);

extern  option          optionTable(LONG lOption);
extern  OPTION_DATA     optionData(LONG lOption);
extern  int             optionCount();
extern  const char*     optionName(LONG lOption);

extern  void            setOption(LONG lOption);
extern  void            resetOption(LONG lOption);
extern  bool            toggleOption(LONG lOption);
extern  bool            isOption(LONG lOption);

extern  void            setSearch(LONG lOption);
extern  void            resetSearch(LONG lOption);
extern  bool            toggleSearch(LONG lOption);
extern  bool            isSearch(LONG lOption);

extern  void            setVerbose(LONG lOption);
extern  void            resetVerbose(LONG lOption);
extern  bool            toggleVerbose(LONG lOption);
extern  bool            isVerbose(LONG lOption);

extern  void            setCount(LONG lOption);
extern  void            resetCount(LONG lOption);
extern  bool            toggleCount(LONG lOption);
extern  bool            isCount(LONG lOption);

extern  void            setSort(LONG lOption);
extern  void            resetSort(LONG lOption);
extern  bool            toggleSort(LONG lOption);
extern  bool            isSort(LONG lOption);

extern  HRESULT         processSetOption(int nOption, int nLastOption, int hasArg);
extern  HRESULT         processToggleOption(int nOption, int nLastOption, int hasArg);
extern  HRESULT         processSortOption(int nOption, int nLastOption, int hasArg);
extern  HRESULT         processSearchOption(int nOption, int nLastOption, int hasArg);
extern  HRESULT         processValueOption(int nOption, int nLastOption, int hasArg);
extern  HRESULT         processCountOption(int nOption, int nLastOption, int hasArg);
extern  HRESULT         processDataOption(int nOption, int nLastOption, int hasArg);

extern  HRESULT         processExchangeOption(int nOption, int nLastOption, int hasArg);
extern  HRESULT         processVerboseOption(int nOption, int nLastOption, int hasArg);

extern  HRESULT         checkTypeOption(int nOption);

//******************************************************************************
//
//  Inline Functions
//
//******************************************************************************
inline  void            setCommandValue(LONG lOption, POINTER ptrPointer)
                            { setCommandValue(lOption, ptrPointer.ptr()); }
inline  void            setSearchValue(LONG lOption, POINTER ptrPointer)
                            { setSearchValue(lOption, ptrPointer.ptr()); }
inline  void            setMaskValue(LONG lOption, POINTER ptrPointer)
                            { setMaskValue(lOption, ptrPointer.ptr()); }
inline  void            setCountValue(LONG lOption, POINTER ptrPointer)
                            { setCountValue(lOption, ptrPointer.ptr()); }

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OPTION_H
