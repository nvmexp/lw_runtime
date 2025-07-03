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
|*  Module: cmdversion.cpp                                                    *|
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
static  HRESULT         parseVersion(int argc, char **argv);

static  HRESULT         versionOption(int nOption, int nLastOption, int hasArg);

static  void            versionHelp();

//******************************************************************************
//
// Locals
//
//******************************************************************************
static  CHelp           s_versionHelp("version", versionHelp);

//******************************************************************************
//
// version
//
// Display extension version information
//
//******************************************************************************

DEBUGGER_COMMAND
version
(
    PDEBUG_CLIENT       pDbgClient,
    PCSTR               args
)
{
    UNREFERENCED_PARAMETER(pDbgClient);

    int                 argc;
    char               *argv[MAX_ARGUMENTS];
    HRESULT             hResult = S_OK;

    assert(pDbgClient != NULL);
    assert(args != NULL);

    try
    {
        // Setup the version command (No globals)
        CCommand Command(pDbgClient, args, &argc, argv, "version", NO_GLOBALS);

        // Call routine to parse version arguments/options
        hResult = parseVersion(argc, argv);
        if (SUCCEEDED(hResult))
        {
            // Check for verbose option
            if (isOption(VerboseOption))
            {
                // Display debugger extension version information (Plus time/date stamp)
                dbgPrintf("%s debugger extension version %d.%d (Built %s)\n",
                          LWEXT_MODULE_NAME, LWEXT_MAJOR_VERSION, LWEXT_MINOR_VERSION, getTimeDateStamp());
            }
            else    // Non-verbose
            {
                // Display the debugger extension version information
                dbgPrintf("%s debugger extension version %d.%d\n",
                          LWEXT_MODULE_NAME, LWEXT_MAJOR_VERSION, LWEXT_MINOR_VERSION);
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

} // version

//******************************************************************************

static HRESULT
parseVersion
(
    int                 argc, 
    char                **argv
)
{
    HRESULT             hResult = S_OK;

    assert(argv != NULL);

    // Set the default option (Verbose)
    setDefaultOption(VerboseOption);

    // Add the version command options
    addOption(DmlOption,        no_argument,        expressionInput,    versionOption);
    addOption(HelpOption,       no_argument,        expressionInput,    versionOption);
    addOption(VerboseOption,    optional_argument,  expressionInput,    versionOption);

    // Call routine to parse the version command options
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
            // Display verbose version command help
            setOption(VerboseOption);
            versionHelp();

            // Set status to not execute the version command
            hResult = E_FAIL;
        }
        else    // No help option
        {
            // Make sure we have no arguments (This command takes none)
            if (optind < argc)
            {
                // Throw exception for invalid arguments
                throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                 ": %s command takes no arguments",
                                 argv[0]);
            }
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

} // parseVersion

//******************************************************************************

static HRESULT
versionOption
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

} // versionOption

//******************************************************************************

void
versionHelp()
{
    // Display base version help information (w/link for version help if DML)
    dPrintf("    %s", helpString("version"));
    dPrintf("      [options]                               - Display extension version information\n");

    // Check for verbose option
    if (isOption(VerboseOption))
    {
        // Display verbose version help information
        dPrintf("                 -dml                                    - Toggle DML setting for this command\n");
        dPrintf("                 -help                                   - Display version command help\n");
        dPrintf("                 -verbose          [=level]              - Verbose display or level\n");
        dPrintf("\n");
    }

} // versionHelp

//******************************************************************************

CString
getTimeDateStamp()
{
    char               *pTimeDateStamp;
    PIMAGE_FILE_HEADER  pImageFileHeader;
    CString             sTimeDateStamp("Unknown");

    // Try to get the image file header for this module (Has timestamp in it)
    pImageFileHeader = getImageFileHeader(getModuleHandle());
    if (pImageFileHeader != NULL)
    {
        // Try to colwert image file header timestamp to time/date string
        pTimeDateStamp = _ctime32(reinterpret_cast<__time32_t *>(&pImageFileHeader->TimeDateStamp));
        if (pTimeDateStamp != NULL)
        {
            // If time/date stamp contains a newline character, remove it
            if (pTimeDateStamp[24] == '\n')
            {
                pTimeDateStamp[24] = 0;
            }
            // Set the time/date stamp string
            sTimeDateStamp = pTimeDateStamp;
        }
    }
    // Return the extension time/date stamp
    return sTimeDateStamp;

} // getTimeDateStamp

//******************************************************************************
//
// Version
//
// Export to return extension version information
//
//******************************************************************************
EXPORT_ULONG
Version()
{
    ULONG               ulVersion = (LWEXT_MAJOR_VERSION << 16) + LWEXT_MINOR_VERSION;

    // Return the extension version information
    return ulVersion;

} // Version

} // cmd namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
