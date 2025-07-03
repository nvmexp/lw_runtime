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
|*  Module: parse.cpp                                                         *|
|*                                                                            *|
 \****************************************************************************/
#include "parprecomp.h"

//******************************************************************************
//
//  Locals
//
//******************************************************************************
static char s_aLiteral[]    = {SINGLE_QUOTE, DOUBLE_QUOTE};
static char s_aRegEx[]      = {SLASH, BACKSLASH};
static char s_aWhitespace[] = {BLANK, TAB};
static char s_aDelimiter[]  = {BLANK, TAB};
static char s_cTerminator   = EOS;
static char s_cOption       = DASH;
static char s_cDelimiter    = EQUAL;

static char s_cmdBuffer[MAX_COMMAND_SIZE];

//******************************************************************************

HRESULT
parseArguments
(
    char               *args,
    char               *pProgram,
    int                *argc,
    char              **argv
)
{
    char                cChar;
    char                cLiteral = 0;
    char                cRegEx = 0;
    int                 nLiteral;
    int                 nRegEx;
    int                 nDelimiter;
    int                 nWhitespace;
    int                 nInput;
    int                 nOutput;
    int                 nLength;
    int                 nArg;
    bool                bLiteral = false;
    bool                bRegEx = false;
    bool                bOption = false;
    bool                bArgument = false;
    HRESULT             hResult = S_OK;

    assert(args != NULL);
    assert(argc != NULL);
    assert(argv != NULL);

    // Initialize the getopt values
    optind = 0;
    opterr = 0;

    // Check for a program name given
    if (pProgram != NULL)
    {
        // Get the length of the program name
        nLength = static_cast<ULONG>(strlen(pProgram));

        // Initialize argument count to include program name (Used as argv array index)
        *argc = 1;

        // Make sure program name is not too long
        if (nLength < MAX_COMMAND_SIZE)
        {
            // Copy the program name into the command buffer
            strcpy(s_cmdBuffer, pProgram);

            // Initialize the current input and output indices
            nInput  = 0;
            nOutput = nLength + 1;
        }
        else    // Program name is too long
        {
            // Only copy what will fit into the command buffer
            strncpy(s_cmdBuffer, pProgram, (MAX_COMMAND_SIZE - 1));

            // Initialize the current input and output indices
            nInput  = 0;
            nOutput = MAX_COMMAND_SIZE;
        }
        // Set argv[0] to the program name
        argv[0] = &s_cmdBuffer[0];
    }
    else    // No program name given
    {
        // Initialize the argument count value (Used as argv array index)
        *argc = 0;

        // Initialize the current input and output indices
        nInput  = 0;
        nOutput = 0;
    }
    // Get the length of the argument string
    nLength = static_cast<int>(strlen(args));

    // Loop processing the arguments (Until end of input or output buffer)
    while ((nInput < nLength) && (nOutput < (MAX_COMMAND_SIZE - 1)))
    {
        // Get the next character in the input string
        cChar = args[nInput];

        // Check for in a literal string
        if (bLiteral)
        {
            // Check to see if this is the end of the literal string
            if (cChar == cLiteral)
            {
                // Indicate end of literal string
                bLiteral = false;
            }
            else    // Still in literal string
            {
                // Store literal string character
                s_cmdBuffer[nOutput++] = cChar;
            }
        }
        else    // Not in a literal string
        {
            // Check for in a regular expression
            if (bRegEx)
            {
                // Check to see if this is the end of the regular expression
                if (cChar == cRegEx)
                {
                    // Indicate end of regular expression
                    bRegEx = false;
                }
                // Store regular expression character
                s_cmdBuffer[nOutput++] = cChar;

                // Check for not already processing an argument
                if (!bArgument)
                {
                    // Save argument pointer and update argument count
                    argv[(*argc)++] = &s_cmdBuffer[nOutput];

                    // Set argument processing
                    bArgument = true;
                }
            }
            else    // Not in a regular expression
            {
                // Loop checking for a literal character
                for (nLiteral = 0; nLiteral < sizeof(s_aLiteral); nLiteral++)
                {
                    // Check next literal string character
                    if (cChar == s_aLiteral[nLiteral])
                    {
                        break;
                    }
                }
                // Check for start of a literal string
                if (nLiteral != sizeof(s_aLiteral))
                {
                    // Save the literal character and set in literal string
                    cLiteral = cChar;
                    bLiteral = true;

                    // Check for not already processing an argument
                    if (!bArgument)
                    {
                        // Save argument pointer and update argument count
                        argv[(*argc)++] = &s_cmdBuffer[nOutput];

                        // Set argument processing
                        bArgument = true;
                    }
                }
                else    // Not a literal delimiter (Normal character)
                {
                    // Loop checking for a regular expression character
                    for (nRegEx = 0; nRegEx < sizeof(s_aRegEx); nRegEx++)
                    {
                        // Check next regular expression character
                        if (cChar == s_aRegEx[nRegEx])
                        {
                            break;
                        }
                    }
                    // Check for start of a regular expression
                    if (nRegEx != sizeof(s_aRegEx))
                    {
                        // Save the regular expression character and set in regular expression
                        cRegEx = cChar;
                        bRegEx = true;

                        // Store regular expression character
                        s_cmdBuffer[nOutput++] = cChar;
                    }
                    else    // Not a regular expression (Normal character)
                    {
                        // Check for processing an argument
                        if (bArgument)
                        {
                            // Loop checking for a delimiter character
                            for (nDelimiter = 0; nDelimiter < sizeof(s_aDelimiter); nDelimiter++)
                            {
                                // Check next delimiter string character
                                if (cChar == s_aDelimiter[nDelimiter])
                                {
                                    break;
                                }
                            }
                            // Check for a delimiter character
                            if (nDelimiter != sizeof(s_aDelimiter))
                            {
                                // Terminate the argument string and argument/option processing
                                s_cmdBuffer[nOutput++] = s_cTerminator;
                                bArgument = false;
                                bOption   = false;

                                // Check for argument limit
                                if (*argc >= MAX_ARGUMENTS)
                                {
                                    break;
                                }
                            }
                            else    // Not a delimiter character
                            {
                                // Check for processing an option argument
                                if (bOption)
                                {
                                    // Check for delimiter to option value
                                    if (cChar == s_cDelimiter)
                                    {
                                        // Clear the option processing so value is not colwerted to lowercase (leave argument flag)
                                        bOption = false;
                                    }
                                    // Save the next option character (Case insensitive)
                                    s_cmdBuffer[nOutput++] = static_cast<char>(tolower(cChar));
                                }
                                else    // Regular argument
                                {
                                    // Save the next argument character
                                    s_cmdBuffer[nOutput++] = cChar;
                                }
                            }
                        }
                        else    // Not processing an argument (Scanning whitespace)
                        {
                            // Loop checking for a whitespace character
                            for (nWhitespace = 0; nWhitespace < sizeof(s_aWhitespace); nWhitespace++)
                            {
                                // Check next whitespace string character
                                if (cChar == s_aWhitespace[nWhitespace])
                                {
                                    break;
                                }
                            }
                            // Check for a non-whitespace character (Argument start)
                            if (nWhitespace == sizeof(s_aWhitespace))
                            {
                                // Save argument pointer and update argument count
                                argv[(*argc)++] = &s_cmdBuffer[nOutput];

                                // Loop checking for a delimiter character
                                for (nDelimiter = 0; nDelimiter < sizeof(s_aDelimiter); nDelimiter++)
                                {
                                    // Check next delimiter string character
                                    if (cChar == s_aDelimiter[nDelimiter])
                                    {
                                        break;
                                    }
                                }
                                // Check for a delimiter character (NULL argument)
                                if (nDelimiter != sizeof(s_aDelimiter))
                                {
                                    // Terminate the argument string (NULL argument)
                                    s_cmdBuffer[nOutput++] = s_cTerminator;
                                }
                                else    // Non-NULL argument
                                {
                                    // Check for option argument
                                    if (cChar == s_cOption)
                                    {
                                        // Save first option character and set option processing (Case insensitive)
                                        s_cmdBuffer[nOutput++] = static_cast<char>(tolower(cChar));

                                        // Set option argument processing
                                        bArgument = true;
                                        bOption   = true;
                                    }
                                    else    // Regular argument
                                    {
                                        // Save first argument character and set argument processing
                                        s_cmdBuffer[nOutput++] = cChar;

                                        // Set argument processing
                                        bArgument = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Increment to the next input character
        nInput++;
    }
    // Loop initializing the unused argument pointers
    for (nArg = *argc; nArg < MAX_ARGUMENTS; nArg++)
    {
        // Initialize the next argument pointer
        argv[nArg] = NULL;
    }
    // Check for processing argument when terminator hit
    if (bArgument)
    {
        // Terminate the final argument
        s_cmdBuffer[nOutput++] = s_cTerminator;
    }
    // Check for unmatched literal string
    if (bLiteral)
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Unmatched literal string!");
    }
    // Check for unmatched regular expression
    if (bRegEx)
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Unmatched regular expression!");
    }
    return hResult;

} // parseArguments

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
