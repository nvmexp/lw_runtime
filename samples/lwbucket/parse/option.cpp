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
|*  Module: option.cpp                                                        *|
|*                                                                            *|
 \****************************************************************************/
#include "parprecomp.h"

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  HRESULT         processOption();

//******************************************************************************
//
// Locals
//
//******************************************************************************
// Argument/Option Globals
static  bool            s_CommandOptions[OptionCount];      // Command options
static  ULONG64         s_CommandValues[OptionCount];       // Command option values
static  bool            s_SearchOptions[OptionCount];       // Search options
static  ULONG64         s_SearchValues[OptionCount];        // Search option values
static  ULONG64         s_MaskValues[OptionCount];          // Mask option values
static  bool            s_VerboseOptions[OptionCount];      // Verbose options
static  bool            s_CountOptions[OptionCount];        // Count options
static  ULONG64         s_CountValues[OptionCount];         // Count option values
static  LONG            s_SortCount;                        // Sort count
static  bool            s_SortOptions[OptionCount];         // Sort options
static  LONG            s_SortValues[OptionCount];          // Sort type values
static  bool            s_SortOrders[OptionCount];          // Sort order values

static option           s_OptionTable[OptionCount + 2];     // Command long option table
static OPTION_DATA      s_OptionData[OptionCount];          // Command option data table
static  int             s_nOption = 0;                      // Command option selected
static  int             s_nLastOption = 0;                  // Command last option selected
static int              s_OptionCount = 0;                  // Command option count
static char*            s_OptionName[OptionCount] = {
                                                     "",                // Unknown command option
                                                     "disable",         // Disable command option
                                                     "dml",             // DML command option
                                                     "enable",          // Enable command option
                                                     "help",            // Help command option
                                                     "quiet",           // Quiet command option
                                                     "verbose",         // Verbose command option
                                                     "xchg",            // Exchange command option
                                                     "?"                // Question/alternate help command option
                                                    };

//******************************************************************************

void
initializeOptions()
{
    // Initialize the option table
    memset(s_OptionTable, 0, sizeof(s_OptionTable));

    // Initialize the option data
    memset(s_OptionData, 0, sizeof(s_OptionData));

    // Reset the option count and sort count (First entry is always unknown option)
    s_OptionCount = 1;
    s_SortCount   = 0;

    // Reset current and last option to unknown option
    s_nOption     = UnknownOption;
    s_nLastOption = UnknownOption;

} // initializeOptions

//******************************************************************************

HRESULT
parseOptions
(
    int                 argc,
    char              **argv
)
{
    int                 nResult;
    HRESULT             hResult = S_OK;

    assert(argv != NULL);

    // Loop processing the arguments
    while ((nResult = getopt_long_only(argc, argv, "", &s_OptionTable[1], NULL)) != -1)
    {
        // Check for a valid parse result (0)
        if (nResult == 0)
        {
            // Call routine to process the next option
            hResult = processOption();
        }
        else    // Invalid parse result
        {
            // Display the unknown/ambiguous option
            dPrintf("Unknown/ambiguous/invalid option %s\n", argv[optind - 1]);

            // Set invalid argument error result
            hResult = E_ILWALIDARG;
        }
        // Check for error processing last option (Stop processing)
        if (FAILED(hResult))
        {
            break;
        }
        // Save the last option value (Only if not verbose option)
        if (s_nOption != VerboseOption)
        {
            s_nLastOption = s_nOption;
        }
    }
    return hResult;

} // parseOptions

//******************************************************************************

void
setDefaultOption
(
    LONG                lOption,
    int                 hasArg,
    POPTION_ROUTINE     pOptionRoutine
)
{
    // Set the requested default option value
    s_OptionTable[UnknownOption].val = lOption;

    // Set the default option data
    s_OptionData[UnknownOption].hasArg         = hasArg;
    s_OptionData[UnknownOption].pOptionRoutine = pOptionRoutine;

} // setDefaultOption

//******************************************************************************

void
addOption
(
    LONG                lOption,
    int                 hasArg
)
{
    // Call the actual addOption routine w/no input or processing routines
    addOption(lOption, hasArg, NULL, NULL);

} // addOption

//******************************************************************************

void
addOption
(
    LONG                lOption,
    int                 hasArg,
    POPTION_INPUT       pOptionInput
)
{
    // Call the actual addOption routine w/no processing routine
    addOption(lOption, hasArg, pOptionInput, NULL);

} // addOption

//******************************************************************************

void
addOption
(
    LONG                lOption,
    int                 hasArg,
    POPTION_ROUTINE     pOptionRoutine
)
{
    // Call the actual addOption routine w/no input routine
    addOption(lOption, hasArg, NULL, pOptionRoutine);

} // addOption

//******************************************************************************

void
addOption
(
    LONG                lOption,
    int                 hasArg,
    POPTION_INPUT       pOptionInput,
    POPTION_ROUTINE     pOptionRoutine
)
{
    // Check for a valid command option
    if (lOption < static_cast<LONG>(countof(s_OptionName)))
    {
        // Check for too many command options
        if (s_OptionCount < static_cast<LONG>(countof(s_OptionTable) - 1))
        {
            // Add the requested option to the option table
            s_OptionTable[s_OptionCount].name    = s_OptionName[lOption];
            s_OptionTable[s_OptionCount].has_arg = hasArg;
            s_OptionTable[s_OptionCount].flag    = &s_nOption;
            s_OptionTable[s_OptionCount].val     = lOption;

            // Save the command argument data
            s_OptionData[lOption].hasArg         = hasArg;
            s_OptionData[lOption].pOptionInput   = pOptionInput;
            s_OptionData[lOption].pOptionRoutine = pOptionRoutine;

            // Increment the option count
            s_OptionCount++;

            // Check for adding the help option (If so, also add alternate help option [question])
            if (lOption == HelpOption)
            {
                // Add the alternate help option (Question)
                s_OptionTable[s_OptionCount].name    = s_OptionName[QuestionOption];
                s_OptionTable[s_OptionCount].has_arg = hasArg;
                s_OptionTable[s_OptionCount].flag    = &s_nOption;
                s_OptionTable[s_OptionCount].val     = QuestionOption;

                // Save the command argument data
                s_OptionData[lOption].hasArg         = hasArg;
                s_OptionData[lOption].pOptionInput   = pOptionInput;
                s_OptionData[lOption].pOptionRoutine = pOptionRoutine;

                // Increment the option count
                s_OptionCount++;
            }
        }
        else    // Too many command options
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Too many command options (> %d)",
                             countof(s_OptionTable));
        }
    }
    else    // Invalid command option
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid command option (%d)",
                         lOption);
    }

} // addOption

//******************************************************************************

void
setSortType
(
    LONG                lOption,
    bool                bSortOrder
)
{
    // Check to see if this sort option not set already
    if (!isSort(lOption))
    {
        // Set the new sort option and type
        setSort(lOption);
        s_SortValues[s_SortCount] = lOption;
        s_SortOrders[s_SortCount] = bSortOrder;

        // Increment the sort count
        s_SortCount++;
    }

} // setSortType

//******************************************************************************

CString
getCommandOptions()
{
    LONG                lOption;
    CString             sCommandOptions("");

    // Loop thru all the possible options and get the command options
    for (lOption = 1; lOption < static_cast<LONG>(countof(s_CommandOptions)); lOption++)
    {
        // Check for next command option
        if (isOption(lOption))
        {
            // Check for next command option
            if (!sCommandOptions.empty())
            {
                sCommandOptions += " -";
            }
            else    // First command option
            {
                sCommandOptions += "-";
            }
            // Add the next selected command option name
            sCommandOptions += s_OptionName[lOption];
        }
    }
    return sCommandOptions;

} // getCommandOptions

//******************************************************************************

CString
getSearchOptions()
{
    LONG                lOption;
    CString             sValue(QWORD_PRINT_WIDTH);
    CString             sSearchOptions("");

    // Loop thru all the possible options and get the search options
    for (lOption = 1; lOption < static_cast<LONG>(countof(s_SearchOptions)); lOption++)
    {
        // Check for next search option
        if (isSearch(lOption))
        {
            // Check for next search option
            if (!sSearchOptions.empty())
            {
                sSearchOptions += " -";
            }
            else    // First search option
            {
                sSearchOptions += "-";
            }
            // Add the next search option name and value
            sSearchOptions += s_OptionName[lOption];
            sSearchOptions += "=";

            sValue.sprintf("0x%08I64x", searchValue(lOption));
            sSearchOptions += sValue;
        }
    }
    return sSearchOptions;

} // getSearchOptions

//******************************************************************************

CString
getSortOptions()
{
    LONG                lSort;
    LONG                lOption;
    CString             sSortOptions("");

    // Loop processing the sort options (In reverse order)
    for (lSort = 0; lSort < s_SortCount; lSort++)
    {
        // Get the next sort option
        lOption = s_SortValues[lSort];

        // Check for next sort option
        if (isSort(lOption))
        {
            // Check for next sort option
            if (!sSortOptions.empty())
            {
                sSortOptions += " -";
            }
            else    // First sort option
            {
                sSortOptions += "-";
            }
            // Add the next sort option
            sSortOptions += s_OptionName[lOption];

            // Check the sort order
            if (s_SortOrders[lSort] != 0)
            {
                // Add option to reverse sort
                sSortOptions += " -x";
            }
        }
    }
    return sSortOptions;

} // getSortOptions

//******************************************************************************

ULONG64
commandValue
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested command value
    return s_CommandValues[lOption];

} // commandValue

//******************************************************************************

PULONG64
commandAddress
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested command address
    return &s_CommandValues[lOption];

} // commandAddress

//******************************************************************************

void
setCommandValue
(
    LONG                lOption,
    ULONG64             ulValue
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested command value
    s_CommandValues[lOption] = ulValue;

} // setCommandValue

//******************************************************************************

ULONG64
searchValue
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested search value
    return s_SearchValues[lOption];

} // searchValue

//******************************************************************************

PULONG64
searchAddress
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested search address
    return &s_SearchValues[lOption];

} // searchAddress

//******************************************************************************

void
setSearchValue
(
    LONG                lOption,
    ULONG64             ulValue
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested search value
    s_SearchValues[lOption] = ulValue;

} // setSearchValue

//******************************************************************************

ULONG64
maskValue
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested mask value
    return s_MaskValues[lOption];

} // maskValue

//******************************************************************************

PULONG64
maskAddress
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested mask address
    return &s_MaskValues[lOption];

} // maskAddress

//******************************************************************************

void
setMaskValue
(
    LONG                lOption,
    ULONG64             ulValue
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested mask value
    s_MaskValues[lOption] = ulValue;

} // setMaskValue

//******************************************************************************

ULONG64
countValue
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested count value
    return s_CountValues[lOption];

} // countValue

//******************************************************************************

PULONG64
countAddress
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested count address
    return &s_CountValues[lOption];

} // countAddress

//******************************************************************************

void
setCountValue
(
    LONG                lOption,
    ULONG64             ulValue
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested count value
    s_CountValues[lOption] = ulValue;

} // setCountValue

//******************************************************************************

LONG
sortValue
(
    LONG                lSortCount
)
{
    // Check for valid sort count value
    if ((lSortCount < 0) || (lSortCount >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid sort count value %d!",
                         lSortCount);
    }
    // Return the requested sort value
    return s_SortValues[lSortCount];

} // sortValue

//******************************************************************************

void
setSortValue
(
    LONG                lSortCount,
    LONG                lValue
)
{
    // Check for valid sort count value
    if ((lSortCount < 0) || (lSortCount >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid sort count value %d!",
                         lSortCount);
    }
    // Set the requested sort value
    s_SortValues[lSortCount] = lValue;

} // setSortValue

//******************************************************************************

bool
sortOrder
(
    LONG                lSortCount
)
{
    // Check for valid sort count value
    if ((lSortCount < 0) || (lSortCount >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid sort count value %d!",
                         lSortCount);
    }
    // Return the requested sort order value
    return s_SortOrders[lSortCount];

} // sortOrder

//******************************************************************************

void
setSortOrder
(
    LONG                lSortCount,
    bool                bValue
)
{
    // Check for valid sort count value
    if ((lSortCount < 0) || (lSortCount >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid sort count value %d!",
                         lSortCount);
    }
    // Set the requested sort order value
    s_SortOrders[lSortCount] = bValue;

} // setSortOrder

//******************************************************************************

LONG
sortCount()
{
    // Return the sort count
    return s_SortCount;

} // sortCount

//******************************************************************************

void
setSortCount
(
    LONG                lSortCount
)
{
    // Check for valid sort count value
    if ((lSortCount < 0) || (lSortCount >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid sort count value %d!",
                         lSortCount);
    }
    // Set the sort count
    s_SortCount = lSortCount;

} // setSortCount

//******************************************************************************

option
optionTable
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the option
    return s_OptionTable[lOption];

} // optionTable

//******************************************************************************

OPTION_DATA
optionData
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the option data
    return s_OptionData[lOption];

} // optionData

//******************************************************************************

int
optionCount()
{
    // Return the option count
    return s_OptionCount;

} // optionCount

//******************************************************************************

const char*
optionName
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the option name
    return s_OptionName[lOption];

} // optionName

//******************************************************************************

void
setOption
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested command option
    s_CommandOptions[lOption] = true;

} // setOption

//******************************************************************************

void
resetOption
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Reset the requested command option
    s_CommandOptions[lOption] = false;

} // resetOption

//******************************************************************************

bool
toggleOption
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Toggle the requested command option
    s_CommandOptions[lOption] ^= true;

    return s_CommandOptions[lOption];

} // toggleOption

//******************************************************************************

bool
isOption
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested command option
    return s_CommandOptions[lOption];

} // isOption

//******************************************************************************

void
setSearch
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested search option
    s_SearchOptions[lOption] = true;

} // setSearch

//******************************************************************************

void
resetSearch
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Reset the requested search option
    s_SearchOptions[lOption] = false;

} // resetSearch

//******************************************************************************

bool
toggleSearch
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Toggle the requested search option
    s_SearchOptions[lOption] ^= true;

    return s_SearchOptions[lOption];

} // toggleSearch

//******************************************************************************

bool
isSearch
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested search option
    return s_SearchOptions[lOption];

} // isSearch

//******************************************************************************

void
setVerbose
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested verbose option
    s_VerboseOptions[lOption] = true;

} // setVerbose

//******************************************************************************

void
resetVerbose
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Reset the requested verbose option
    s_VerboseOptions[lOption] = false;

} // resetVerbose

//******************************************************************************

bool
toggleVerbose
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Toggle the requested verbose option
    s_VerboseOptions[lOption] ^= true;

    return s_VerboseOptions[lOption];

} // toggleVerbose

//******************************************************************************

bool
isVerbose
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested verbose option
    return s_VerboseOptions[lOption];

} // isVerbose

//******************************************************************************

void
setCount
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested count option
    s_CountOptions[lOption] = true;

} // setCount

//******************************************************************************

void
resetCount
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Reset the requested count option
    s_CountOptions[lOption] = false;

} // resetCount

//******************************************************************************

bool
toggleCount
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Toggle the requested count option
    s_CountOptions[lOption] ^= true;

    return s_CountOptions[lOption];

} // toggleCount

//******************************************************************************

bool
isCount
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested count option
    return s_CountOptions[lOption];

} // isCount

//******************************************************************************

void
setSort
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Set the requested sort option
    s_SortOptions[lOption] = true;

} // setSort

//******************************************************************************

void
resetSort
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Reset the requested sort option
    s_SortOptions[lOption] = false;

} // resetSort

//******************************************************************************

bool
toggleSort
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Toggle the requested sort option
    s_SortOptions[lOption] ^= true;

    return s_SortOptions[lOption];

} // toggleSort

//******************************************************************************

bool
isSort
(
    LONG                lOption
)
{
    // Check for valid option value
    if ((lOption < 0) || (lOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         lOption);
    }
    // Return the requested sort option
    return s_SortOptions[lOption];

} // isSort

//******************************************************************************

HRESULT
processOption()
{
    HRESULT             hResult = S_OK;

    // Switch on the next option
    switch(s_nOption)
    {
        //
        // If no argument is present set the option flag
        // If an argument is present then set the option flag
        // and retrieve the value in the current radix and save it
        // as the command value
        // If verbose is set then the option verbose is set
        //
        case EnableOption:                  // Enable option

            // Check for custom option processing
            if (s_OptionData[s_nOption].pOptionRoutine != NULL)
            {
                // Call the custom option processing routine
                hResult = (*s_OptionData[s_nOption].pOptionRoutine)(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            else    // No custom option processing routine
            {
                // Process these options as value options
                hResult = processValueOption(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            break;
        //
        // If there are sort options and the last sort option matches
        // last option specified then toggle the last sort order, otherwise
        // toggle the exchange option
        // If there are no sort options then toggle the exchange option
        //
        case ExchangeOption:                // Exchange option

            // Check for custom option processing
            if (s_OptionData[s_nOption].pOptionRoutine != NULL)
            {
                // Call the custom option processing routine
                hResult = (*s_OptionData[s_nOption].pOptionRoutine)(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            else    // No custom option processing routine
            {
                // Process this option as exchange option
                hResult = processExchangeOption(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            break;
        //
        // Simply sets the requested option
        //
        case HelpOption:                    // Help option

            // Check for custom option processing
            if (s_OptionData[s_nOption].pOptionRoutine != NULL)
            {
                // Call the custom option processing routine
                hResult = (*s_OptionData[s_nOption].pOptionRoutine)(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            else    // No custom option processing routine
            {
                // Process these options as set options
                hResult = processSetOption(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            break;
        //
        // Simply toggles the requested option
        //
        case DmlOption:                     // DML option
        case QuietOption:                   // Quiet option

            // Check for custom option processing
            if (s_OptionData[s_nOption].pOptionRoutine != NULL)
            {
                // Call the custom option processing routine
                hResult = (*s_OptionData[s_nOption].pOptionRoutine)(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            else    // No custom option processing routine
            {
                // Process these options as toggle options
                hResult = processToggleOption(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            break;

        //
        // The last option value is checked for a special value
        // (Unknown, Dml, Exchange, Number, or Quiet) if so then the
        // verbose option and the default option verbose is set
        // If the last option value was verbose then the verbose
        // option and the default option verbose is toggled
        // If last option is not a special value then the last option
        // verbose is toggled
        //
        case VerboseOption:                 // Verbose option

            // Check for custom option processing
            if (s_OptionData[s_nOption].pOptionRoutine != NULL)
            {
                // Call the custom option processing routine
                hResult = (*s_OptionData[s_nOption].pOptionRoutine)(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            else    // No custom option processing routine
            {
                // Process this option as verbose option
                hResult = processVerboseOption(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            break;
        //
        // Special case alternate help option (?), simply set the help option
        //
        case QuestionOption:                // Question/alternate help option

            // Check for custom option processing
            if (s_OptionData[s_nOption].pOptionRoutine != NULL)
            {
                // Call the custom option processing routine
                hResult = (*s_OptionData[s_nOption].pOptionRoutine)(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            else    // No custom option processing routine
            {
                // Process this option as set option (Alias help option)
                hResult = processSetOption(HelpOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            break;
        //
        // Basic default argument processing simply sets the option
        //
        default:                            // Default option processing

            // Check for custom option processing
            if (s_OptionData[s_nOption].pOptionRoutine != NULL)
            {
                // Call the custom option processing routine
                hResult = (*s_OptionData[s_nOption].pOptionRoutine)(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            else    // No custom option processing routine
            {
                // Process this option as set option
                hResult = processSetOption(s_nOption, s_nLastOption, s_OptionData[s_nOption].hasArg);
            }
            break;
    }
    return hResult;

} // processOption

//******************************************************************************

HRESULT
processSetOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    UNREFERENCED_PARAMETER(nLastOption);

    HRESULT             hResult = S_OK;

    // Check for valid option value
    if ((nOption < 0) || (nOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         nOption);
    }
    // Check for no arguments present
    if (hasArg == no_argument)
    {
        // Set option flag
        setOption(nOption);
    }
    else    // Arguments may be present
    {
        // Check for no optional argument (Set option)
        if (optarg == NULL)
        {
            // Set option flag
            setOption(nOption);
        }
        else    // Argument present
        {
            // Check for an option input routine
            if (s_OptionData[nOption].pOptionInput != NULL)
            {
                (*s_OptionData[nOption].pOptionInput)(searchAddress(nOption), maskAddress(nOption), optarg);
            }
            else    // No option input routine
            {
                // Get the argument value
                expressionInput(searchAddress(nOption), maskAddress(nOption), optarg);
            }
            // Set search option
            setSearch(nOption);
        }
    }
    // Set verbose if verbose option
    if (isOption(VerboseOption))
    {
        setVerbose(nOption);
    }
    return hResult;

} // processSetOption

//******************************************************************************

HRESULT
processToggleOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    UNREFERENCED_PARAMETER(nLastOption);

    HRESULT             hResult = S_OK;

    // Check for valid option value
    if ((nOption < 0) || (nOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         nOption);
    }
    // Check for no arguments present
    if (hasArg == no_argument)
    {
        // Toggle option flag
        toggleOption(nOption);
    }
    else    // Arguments may be present
    {
        // Check for no optional argument (Toggle option)
        if (optarg == NULL)
        {
            // Toggle option flag
            toggleOption(nOption);
        }
        else    // Argument present
        {
            // Check for an option input routine
            if (s_OptionData[nOption].pOptionInput != NULL)
            {
                (*s_OptionData[nOption].pOptionInput)(searchAddress(nOption), maskAddress(nOption), optarg);
            }
            else    // No option input routine
            {
                // Get the argument value
                expressionInput(searchAddress(nOption), maskAddress(nOption), optarg);
            }
            // Set search option
            setSearch(nOption);
        }
    }
    // Set verbose if verbose option
    if (isOption(VerboseOption))
    {
        setVerbose(nOption);
    }
    return hResult;

} // processToggleOption

//******************************************************************************

HRESULT
processSortOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    UNREFERENCED_PARAMETER(nLastOption);
    UNREFERENCED_PARAMETER(hasArg);

    HRESULT             hResult = S_OK;

    // Check for valid option value
    if ((nOption < 0) || (nOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         nOption);
    }
    // Check for no arguments present
    if (hasArg == no_argument)
    {
        // Set option flag
        setOption(nOption);
    }
    else    // Arguments may be present
    {
        // Check for no optional argument (Sort option)
        if (optarg == NULL)
        {
            // Set option sort type and order
            setSortType(nOption, isOption(ExchangeOption));
        }
        else    // Argument present
        {
            // Check for an option input routine
            if (s_OptionData[nOption].pOptionInput != NULL)
            {
                (*s_OptionData[nOption].pOptionInput)(searchAddress(nOption), maskAddress(nOption), optarg);
            }
            else    // No option input routine
            {
                // Get the argument value
                expressionInput(searchAddress(nOption), maskAddress(nOption), optarg);
            }
            // Set search option
            setSearch(nOption);
        }
    }
    // Set verbose if verbose option
    if (isOption(VerboseOption))
    {
        setVerbose(nOption);
    }
    return hResult;

} // processSortOption

//******************************************************************************

HRESULT
processSearchOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    UNREFERENCED_PARAMETER(nLastOption);

    HRESULT             hResult = S_OK;

    // Check for valid option value
    if ((nOption < 0) || (nOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         nOption);
    }
    // Check for no arguments present
    if (hasArg == no_argument)
    {
        // Set search option
        setSearch(nOption);
    }
    else    // Arguments may be present
    {
        // Check for no optional argument (Set search)
        if (optarg == NULL)
        {
            // Set search option
            setSearch(nOption);
        }
        else    // Argument present
        {
            // Check for an option input routine
            if (s_OptionData[nOption].pOptionInput != NULL)
            {
                (*s_OptionData[nOption].pOptionInput)(searchAddress(nOption), maskAddress(nOption), optarg);
            }
            else    // No option input routine
            {
                // Get the argument value
                expressionInput(searchAddress(nOption), maskAddress(nOption), optarg);
            }
            // Set search option
            setSearch(nOption);
        }
    }
    // Set verbose if verbose option
    if (isOption(VerboseOption))
    {
        setVerbose(nOption);
    }
    return hResult;

} // processSearchOption

//******************************************************************************

HRESULT
processValueOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    UNREFERENCED_PARAMETER(nLastOption);
    UNREFERENCED_PARAMETER(hasArg);

    HRESULT             hResult = S_OK;

    // Check for valid option value
    if ((nOption < 0) || (nOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         nOption);
    }
    // Check for no argument present
    if (optarg == NULL)
    {
        // Set option flag
        setOption(nOption);
    }
    else    // Argument present
    {
        // Check for no optional argument (Set option)
        if (optarg == NULL)
        {
            // Set option
            setOption(nOption);
        }
        else    // Argument present
        {
            // Check for an option input routine
            if (s_OptionData[nOption].pOptionInput != NULL)
            {
                (*s_OptionData[nOption].pOptionInput)(commandAddress(nOption), maskAddress(nOption), optarg);
            }
            else    // No option input routine
            {
                // Get the argument value
                expressionInput(commandAddress(nOption), maskAddress(nOption), optarg);
            }
            // Set option
            setOption(nOption);
        }
    }
    // Set verbose if verbose option
    if (isOption(VerboseOption))
    {
        setVerbose(nOption);
    }
    return hResult;

} // processValueOption

//******************************************************************************

HRESULT
processCountOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    UNREFERENCED_PARAMETER(nLastOption);
    UNREFERENCED_PARAMETER(hasArg);

    HRESULT             hResult = S_OK;

    // Check for valid option value
    if ((nOption < 0) || (nOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         nOption);
    }






    return hResult;

} // processCountOption

//******************************************************************************

HRESULT
processExchangeOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    UNREFERENCED_PARAMETER(nLastOption);
    UNREFERENCED_PARAMETER(hasArg);

    HRESULT             hResult = S_OK;

    // Check for valid option value
    if ((nOption < 0) || (nOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         nOption);
    }
    // Check for sort options
    if (s_SortCount != 0)
    {
        // Check for last sort option (Toggle if so)
        if (s_SortValues[s_SortCount - 1] == nLastOption)
        {
            s_SortOrders[s_SortCount - 1] ^= true;
        }
        else
        {
            toggleOption(nOption);
        }
    }
    else    // No sort options
    {
        // Toggle the option
        toggleOption(nOption);
    }
    return hResult;

} // processExchangeOption

//******************************************************************************

HRESULT
processVerboseOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    UNREFERENCED_PARAMETER(nLastOption);
    UNREFERENCED_PARAMETER(hasArg);

    HRESULT             hResult = S_OK;

    // Check for valid option value
    if ((nOption < 0) || (nOption >= OptionCount))
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid option value %d!",
                         nOption);
    }
    // Check for no optional argument (Verbose option)
    if (optarg == NULL)
    {
        // Switch on the last option
        switch(nLastOption)
        {
            case UnknownOption:             // No last option
            case DmlOption:                 // Dml option
            case ExchangeOption:            // Exchange option
            case QuietOption:               // Quiet option

                // Set the verbose and default option verbose
                setOption(VerboseOption);
                setVerbose(s_OptionTable[UnknownOption].val);

                break;

            case VerboseOption:             // Verbose option

                // Toggle the verbose option and default option verbose
                toggleOption(nOption);
                toggleVerbose(s_OptionTable[UnknownOption].val);

                break;

            default:

                // Toggle the last option verbose
                toggleVerbose(nLastOption);

                break;
        }
    }
    else    // Argument present (Verbose level)
    {
        // Check for an option input routine
        if (s_OptionData[nOption].pOptionInput != NULL)
        {
            (*s_OptionData[nOption].pOptionInput)(commandAddress(nOption), maskAddress(nOption), optarg);
        }
        else    // No option input routine
        {
            // Get the argument value
            verboseInput(commandAddress(nOption), maskAddress(nOption), optarg);
        }
    }
    return hResult;

} // processVerboseOption

//******************************************************************************

HRESULT
checkTypeOption
(
    int                 nOption
)
{
    HRESULT             hResult = S_OK;

    // Switch on the option
    switch(nOption)
    {
        case DisableOption:             // Disable option

            // If enable option is set, then reset it
            if (isOption(EnableOption))
            {
                dPrintf("Disable option overriding enable option\n");
                resetOption(EnableOption);
            }
            break;

        case EnableOption:              // Enable option

            // If disable option is set, then reset it
            if (isOption(DisableOption))
            {
                dPrintf("Enable option overriding disable option\n");
                resetOption(DisableOption);
            }
            break;

        default:                            // Unknown option

            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": Invalid type option (%d)",
                             nOption);

            break;
    }
    return hResult;

} // checkTypeOption

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
