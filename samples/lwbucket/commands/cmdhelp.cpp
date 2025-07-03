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
|*  Module: cmdhelp.cpp                                                       *|
|*                                                                            *|
 \****************************************************************************/
#include "cmdprecomp.h"

//******************************************************************************
//
//  cmd namespace
//
//******************************************************************************
namespace cmd
{

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  HRESULT         parseHelp(int argc, char **argv);

static  HRESULT         helpOption(int nOption, int nLastOption, int hasArg);

static  void            helpHelp();

//******************************************************************************
//
// Locals
//
//******************************************************************************
static  CHelp           s_helpHelp("help", helpHelp);

CHelp*                  s_pFirstHelp  = NULL;       // First help entry
CHelp*                  s_pLastHelp   = NULL;       // Last help entry
ULONG                   s_ulHelpCount = 0;          // Help count value
ULONG                   s_ulHelpWidth = 0;          // Help width value

//******************************************************************************
//
// help
//
// Display help information
//
//******************************************************************************

DEBUGGER_COMMAND
help
(
    PDEBUG_CLIENT       pDbgClient,
    PCSTR               args
)
{
    UNREFERENCED_PARAMETER(pDbgClient);

    int                 argc;
    char               *argv[MAX_ARGUMENTS];
    CString             sString(MAX_DBGPRINTF_STRING);
    CString             sOptions(MAX_COMMAND_STRING);
    CString             sRegularExpression;
    CString             sLink;
    CHelp*             *pHelpTable;
    CHelp              *pHelp = NULL;
    regex_t             reHelp = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulArg;
    ULONG               ulCommand;
    ULONG               ulHelpCount = 0;
    bool                bFound;
    HRESULT             hResult = S_OK;

    assert(pDbgClient != NULL);
    assert(args != NULL);

    try
    {
        // Setup the help command (No globals)
        CCommand Command(pDbgClient, args, &argc, argv, "help", NO_GLOBALS);

        // Call routine to parse help arguments/options
        hResult = parseHelp(argc, argv);
        if (SUCCEEDED(hResult))
        {
            // Check for help arguments (Specified help)
            if ((argc - optind) != 0)
            {
                // Loop through the given arguments (Displaying help)
                for (ulArg = optind; ulArg < static_cast<ULONG>(argc); ulArg++)
                {
                    // Default to no help found for this argument
                    bFound = false;

                    // Check to see if next argument is a regular expression
                    if (isRegularExpression(argv[ulArg]))
                    {
                        // Get the regular expression string from the input string
                        sRegularExpression = getRegularExpression(argv[ulArg]);
                    }
                    else    // Not a forced regular expression
                    {
                        // Toggle the verbose option (Assume verbose help)
                        toggleOption(VerboseOption);

                        // Treat the argument string as a regular expression
                        sRegularExpression = argv[ulArg];
                    }
                    // Try to compile the given string as a case insensitive regular expression
                    reResult = regcomp(&reHelp, sRegularExpression, REG_EXTENDED + REG_ICASE);
                    if (reResult == REG_NOERROR)
                    {
                        // Loop searching for the matching help command
                        pHelp = s_pFirstHelp;
                        while (pHelp != NULL)
                        {
                            // Compare the given type and next help type string
                            reResult = regexec(&reHelp, pHelp->helpString(), countof(reMatch), reMatch, 0);
                            if (reResult == REG_NOERROR)
                            {
                                // Check to see if this help command should be displayed
                                if (pHelp->helpDisplay(COMMAND_SPECIFIED))
                                {
                                    // Command match, indicate match found
                                    bFound = true;

                                    // Display help for the matching command
                                    pHelp->helpFunction()();

                                    // Check for an exact command match
                                    if ((strlen(pHelp->helpString()) == sRegularExpression.length()) &&
                                        (_stricmp(pHelp->helpString(), sRegularExpression) == 0))
                                    {
                                        // Exact command match, break out of search loop
                                        break;
                                    }
                                }
                            }
                            // Move to the next help entry
                            pHelp = pHelp->nextHelp();
                        }
                        // Check for no help found (Display warning)
                        if (!bFound)
                        {
                            dPrintf("No help found for command %s!\n", argv[ulArg]);
                        }
                    }
                    else    // Invalid regular expression
                    {
                        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                         regString(reResult, &reHelp, sRegularExpression));
                    }
                    // Check to see if last argument wasn't a regular expression
                    if (!isRegularExpression(argv[ulArg]))
                    {
                        // Toggle the verbose option (Reset original verbose value)
                        toggleOption(VerboseOption);
                    }
                }
            }
            else    // No specified help
            {
                // Check for verbose option
                if (isOption(VerboseOption))
                {
                    // Display verbose help title
                    dPrintf("Help for the lWpu Bucket debugger extension (%s)\n", LWEXT_MODULE_NAME);
                }
                else    // Not verbose
                {
                    // Display help title
                    dPrintf("Help for %s\n", LWEXT_MODULE_NAME);
                }
                // Check for any help commands available
                if (s_ulHelpCount != 0)
                {
                    // Try to allocation the help table
                    pHelpTable = new CHelp*[s_ulHelpCount];

                    // Loop loading the help table (Conditional help display)
                    pHelp = s_pFirstHelp;
                    for (ulCommand = 0; ulCommand < s_ulHelpCount; ulCommand++)
                    {
                        // Check to see if this help should be displayed
                        if (pHelp->helpDisplay())
                        {
                            // Set help table pointer and increment help count
                            pHelpTable[ulHelpCount++] = pHelp;
                        }
                        // Move on to the next help command
                        pHelp = pHelp->nextHelp();
                    }
                    // Check for help commands to display
                    if (ulHelpCount != 0)
                    {
                        // Sort the help commands (by name)
                        qsort(pHelpTable, ulHelpCount, sizeof(CHelp*), helpCompare);

                        // Loop displaying help for all the commands
                        for (ulCommand = 0; ulCommand < ulHelpCount; ulCommand++)
                        {
                            pHelpTable[ulCommand]->helpFunction()();
                        }
                    }
                    // Free the help table
                    delete[] pHelpTable;
                    pHelpTable = NULL;
                }
            }
        }
    }
    catch (CException& exception)
    {
        // Display exception message and return error
        exception.dPrintf();
        return exception.hResult();
    }
    return hResult;

} // help

//******************************************************************************

static HRESULT
parseHelp
(
    int                 argc, 
    char              **argv
)
{
    HRESULT             hResult = S_OK;

    assert(argv != NULL);

    // Set the default option (Verbose)
    setDefaultOption(VerboseOption);

    // Add the help command options
    addOption(DmlOption,        no_argument,        expressionInput,    helpOption);
    addOption(HelpOption,       no_argument,        expressionInput,    helpOption);
    addOption(VerboseOption,    optional_argument,  expressionInput,    helpOption);

    // Call routine to parse the help command options
    hResult = parseOptions(argc, argv);
    if (SUCCEEDED(hResult))
    {
        // Toggle DML setting if requested
        if (isOption(DmlOption))
        {
            dmlState(!dmlState());
        }
        // If help option is set, ignore other options and display help
        if (isOption(HelpOption))
        {
            // Display verbose help command help
            setOption(VerboseOption);
            helpHelp();

            // Set status to not execute the help command
            hResult = E_FAIL;
        }
    }
    else    // Invalid arguments/options
    {
        // Throw exception for invalid arguments
        throw CException(hResult, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid %s arguments/options, please see help",
                         argv[0]);
    }
    return hResult;

} // parseHelp

//******************************************************************************

static HRESULT
helpOption
(
    int                 nOption,
    int                 nLastOption,
    int                 hasArg
)
{
    HRESULT             hResult = S_OK;

    // Switch on the option value
    switch(nOption)
    {
        // Treat the following option(s) as toggle options
        case DmlOption:                         // DML option

            // Process this option as a toggle option
            hResult = processToggleOption(nOption, nLastOption, hasArg);

            break;

        // Treat the following option(s) as set options
        case HelpOption:                        // Help option

            // Process this option as a set options
            hResult = processSetOption(nOption, nLastOption, hasArg);

            break;

        // Treat the following option as verbose option
        case VerboseOption:                     // Verbose option

            // Process this option as a verbose option
            hResult = processVerboseOption(nOption, nLastOption, hasArg);

            break;

        default:                                // Unknown option

            // Display the unknown/ambiguous option
            dPrintf("Unknown/ambiguous option %s (%d)\n", optarg, nOption);

            // Set invalid argument error result
            hResult = E_ILWALIDARG;

            break;
    }
    return hResult;

} // helpOption

//******************************************************************************

void
helpHelp()
{
    // Display base help information (w/link for help help if DML)
    dPrintf("    %s", helpString("help"));
    dPrintf("         [options]         [command(s)]          - Display command help\n");

    // Check for verbose option
    if (isOption(VerboseOption))
    {
        // Display verbose help information
        dPrintf("                 -dml                                    - Toggle DML setting for this command\n");
        dPrintf("                 -help                                   - Display help command help\n");
        dPrintf("                 -verbose          [=level]              - Verbose display or level\n");
        dPrintf("                                   [command(s)]          - Optional command(s) to display help information for\n");
        dPrintf("\n");
        if (dmlState())
        {
            dPrintf("%s", startForeground(RED));
        }
        dPrintf("Note - This extension is only useful for OCA crash dump bucketing and has no\n");
        dPrintf("       commands of it's own other than this help and a version command!\n");
        if (dmlState())
        {
            dPrintf("%s", endForeground());
        }
        dPrintf("\n");
    }

} // helpHelp

//******************************************************************************

CHelp::CHelp
(
    const char         *pHelpString,
    PFN_HELP            pfnHelp
)
:   m_pPrevHelp(NULL),
    m_pNextHelp(NULL),
    m_pHelpString(pHelpString),
    m_pfnHelp(pfnHelp)
{
    assert(pHelpString != NULL);

    // Add this help to the help list
    addHelp(this);

} // CHelp

//******************************************************************************

CHelp::~CHelp()
{

} // ~CHelp

//******************************************************************************

void
CHelp::addHelp
(
    CHelp              *pHelp
)
{
    ULONG               ulWidth;

    assert(pHelp != NULL);

    // Get the width of this help string
    ulWidth = static_cast<ULONG>(strlen(pHelp->helpString()));

    // Check for first help
    if (s_pFirstHelp == NULL)
    {
        // Set first and last help to this help
        s_pFirstHelp = pHelp;
        s_pLastHelp  = pHelp;
    }
    else    // Adding new help to help list
    {
        // Add this help to the end of the help list
        pHelp->m_pPrevHelp = s_pLastHelp;
        pHelp->m_pNextHelp = NULL;

        s_pLastHelp->m_pNextHelp = pHelp;

        s_pLastHelp = pHelp;
    }
    // Check width of this help string vs maximum
    if (ulWidth > s_ulHelpWidth)
    {
        // Update the maxmimum help width
        s_ulHelpWidth = ulWidth;
    }
    // Increment the help count
    s_ulHelpCount++;

} // addHelp

//******************************************************************************

ULONG
CHelp::helpWidth() const
{
    // Just return the help width value
    return s_ulHelpWidth;

} // helpWidth

//******************************************************************************

bool
CHelp::helpDisplay
(
    bool                bSpecified
) const
{
    UNREFERENCED_PARAMETER(bSpecified);

    bool                bDisplay = true;

    return bDisplay;

} // helpDisplay

//******************************************************************************

CString
helpString
(
    const char         *pCommand
)
{
    CString             sCommand(MAX_COMMAND_STRING);
    CString             sHelp(MAX_COMMAND_STRING);

    assert(pCommand != NULL);

    // Check for DML help string requested    
    if (dmlState())
    {
        // Check for verbose option
        if (isOption(VerboseOption))
        {
            sCommand.sprintf("help");
        }
        else    // Non-verbose option
        {
            sCommand.sprintf("help %s", pCommand);
        }
        // Build the correct help string
        sHelp = exec(pCommand, buildLwExtCommand(sCommand));
    }
    else    // Plain text only
    {
        sHelp.sprintf("%s", pCommand);
    }
    return sHelp;

} // helpString

//******************************************************************************

CHelpUser::CHelpUser
(
    const char         *pHelpString,
    PFN_HELP            pfnHelp
)
:   CHelp(pHelpString, pfnHelp)
{
    assert(pHelpString != NULL);

} // CHelpUser

//******************************************************************************

CHelpUser::~CHelpUser()
{

} // ~CHelpUser

//******************************************************************************

bool
CHelpUser::helpDisplay
(
    bool                bSpecified
) const
{
    UNREFERENCED_PARAMETER(bSpecified);

    bool                bDisplay = false;

    // Only display help for this command if in user mode
    if (isUserMode())
    {
        // Indicate help should be displayed
        bDisplay = true;
    }
    return bDisplay;

} // helpDisplay

//******************************************************************************

CHelpKernel::CHelpKernel
(
    const char         *pHelpString,
    PFN_HELP            pfnHelp
)
:   CHelp(pHelpString, pfnHelp)
{
    assert(pHelpString != NULL);

} // CHelpKernel

//******************************************************************************

CHelpKernel::~CHelpKernel()
{

} // ~CHelpKernel

//******************************************************************************

bool
CHelpKernel::helpDisplay
(
    bool                bSpecified
) const
{
    UNREFERENCED_PARAMETER(bSpecified);

    bool                bDisplay = false;

    // Only display help for this command if in kernel mode
    if (isKernelMode())
    {
        // Indicate help should be displayed
        bDisplay = true;
    }
    return bDisplay;

} // helpDisplay

//******************************************************************************

CHelpDebug::CHelpDebug
(
    const char         *pHelpString,
    PFN_HELP            pfnHelp
)
:   CHelp(pHelpString, pfnHelp)
{
    assert(pHelpString != NULL);

} // CHelpDebug

//******************************************************************************

CHelpDebug::~CHelpDebug()
{

} // ~CHelpDebug

//******************************************************************************

bool
CHelpDebug::helpDisplay
(
    bool                bSpecified
) const
{
    UNREFERENCED_PARAMETER(bSpecified);

    bool                bDisplay = false;

#if DEBUG
    // Always display help if a debug build
    bDisplay = true;
#else
    // Only display help for this command if specified command help
    if (bSpecified)
    {
        bDisplay = true;
    }
#endif
    return bDisplay;

} // helpDisplay

//******************************************************************************

CHelpDump::CHelpDump
(
    const char         *pHelpString,
    PFN_HELP            pfnHelp
)
:   CHelp(pHelpString, pfnHelp)
{
    assert(pHelpString != NULL);

} // CHelpDump

//******************************************************************************

CHelpDump::~CHelpDump()
{

} // ~CHelpDump

//******************************************************************************

bool
CHelpDump::helpDisplay
(
    bool                bSpecified
) const
{
    UNREFERENCED_PARAMETER(bSpecified);

    bool                bDisplay = false;

    // Only display help for this command if in dump mode
    if (isDumpFile())
    {
        // Indicate help should be displayed
        bDisplay = true;
    }
    return bDisplay;

} // helpDisplay

} // cmd namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
