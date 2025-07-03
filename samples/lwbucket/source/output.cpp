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
|*  Module: output.cpp                                                        *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

//******************************************************************************
//
// Locals
//
//******************************************************************************
// Debug output indentation control
static  int             s_dbgIndent    = 0;     // Current output indent value
static  bool            s_bNeedsIndent = true;  // Set if indention needed

// Flush output control
bool                    s_bFlush = false;       // Set if output flush requested

//******************************************************************************

void
dbgPrintf
(
    const char         *pszFormat,
    ...
)
{
    char                sBuffer[MAX_DBGPRINTF_STRING];
    char               *pString = sBuffer;
    char               *pBuffer = pString;
    va_list             va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Perform the printf into the buffer
    vsprintf(sBuffer, pszFormat, va);

    // End the variable argument list
    va_end(va);

    // Loop processing the next string in the output
    while (*pBuffer != '\0')
    {
        // Check for end of line (Time to print)
        if (*pBuffer == '\n')
        {
            // Terminate the string in the buffer
            *pBuffer = '\0';

            // Print the string (Indented if needed)
            if (s_bNeedsIndent && (s_dbgIndent != 0))
            {
                // Check for DML output enabled
                if (dmlState())
                {
                    ControlledOutput(DEBUG_OUTCTL_AMBIENT_DML, DEBUG_OUTPUT_NORMAL, "%*s%s\n", s_dbgIndent, "", pString);
                }
                else
                {
                    ControlledOutput(DEBUG_OUTCTL_AMBIENT_TEXT, DEBUG_OUTPUT_NORMAL, "%*s%s\n", s_dbgIndent, "", pString);
                }
            }
            else    // No indentation needed
            {
                // Check for DML output enabled
                if (dmlState())
                {
                    ControlledOutput(DEBUG_OUTCTL_AMBIENT_DML, DEBUG_OUTPUT_NORMAL, "%s\n", pString);
                }
                else
                {
                    ControlledOutput(DEBUG_OUTCTL_AMBIENT_TEXT, DEBUG_OUTPUT_NORMAL, "%s\n", pString);
                }
            }
            // Indicate indention is now needed
            s_bNeedsIndent = true;

            // Update string and buffer pointers
            pString = ++pBuffer;
        }
        else    // Not end of the line yet
        {
            pBuffer++;
        }
    }
    // Check for partial line
    if (*pString != '\0')
    {
        // Print the string (Indented if needed)
        if (s_bNeedsIndent && (s_dbgIndent != 0))
        {
            // Check for DML output enabled
            if (dmlState())
            {
                ControlledOutput(DEBUG_OUTCTL_AMBIENT_DML, DEBUG_OUTPUT_NORMAL, "%*s%s", s_dbgIndent, "", pString);
            }
            else
            {
                ControlledOutput(DEBUG_OUTCTL_AMBIENT_TEXT, DEBUG_OUTPUT_NORMAL, "%*s%s", s_dbgIndent, "", pString);
            }
        }
        else    // No indentation needed
        {
            // Check for DML output enabled
            if (dmlState())
            {
                ControlledOutput(DEBUG_OUTCTL_AMBIENT_DML, DEBUG_OUTPUT_NORMAL, "%s", pString);
            }
            else
            {
                ControlledOutput(DEBUG_OUTCTL_AMBIENT_TEXT, DEBUG_OUTPUT_NORMAL, "%s", pString);
            }
        }
        // Indicate indention is not needed
        s_bNeedsIndent = false;
    }
    // Check for output flush requested
    if (outputFlush())
    {
        // Make sure any output callbacks are flushed (Not necessarily dispatched yet)
        FlushCallbacks();

        // Dispatch all pending callbacks (Including output callbacks, i.e. flush, with enough timeout to complete)
        DispatchCallbacks(static_cast<ULONG>(dmllen(sBuffer)));
    }
    // Perform a status check to allow user break during output
    statusCheck();

} // dbgPrintf

//******************************************************************************

void
dPrintf
(
    const char         *pszFormat,
    ...
)
{
    char                sBuffer[MAX_DBGPRINTF_STRING];
    va_list             va;

    assert(pszFormat != NULL);

    // Setup start of the variable argument list
    va_start(va, pszFormat);

    // Perform the printf into the buffer
    vsprintf(sBuffer, pszFormat, va);

    // End the variable argument list
    va_end(va);

    // Check for DML output enabled
    if (dmlState())
    {
        ControlledOutput(DEBUG_OUTCTL_AMBIENT_DML, DEBUG_OUTPUT_NORMAL, "%s", sBuffer);
    }
    else
    {
        ControlledOutput(DEBUG_OUTCTL_AMBIENT_TEXT, DEBUG_OUTPUT_NORMAL, "%s", sBuffer);
    }
    // Check for output flush requested
    if (outputFlush())
    {
        // Make sure any output callbacks are flushed (Not necessarily dispatched yet)
        FlushCallbacks();

        // Dispatch all pending callbacks (Including output callbacks, i.e. flush, with enough timeout to complete)
        DispatchCallbacks(static_cast<ULONG>(dmllen(sBuffer)));
    }
    // Perform a status check to allow user break during output
    statusCheck();

} // dPrintf

//******************************************************************************

void
dbgIndent
(
    int                 indent
)
{
    // Update the debug indent value
    s_dbgIndent += indent;

} // dbgIndent

//******************************************************************************

void
dbgUnindent
(
    int                 indent
)
{
    // Update the debug indent value
    s_dbgIndent -= indent;

} // dbgUnindent

//******************************************************************************

int
dbgIndentation()
{
    // Return the current indentation value
    return s_dbgIndent;

} // dbgIndentation

//******************************************************************************

int
dbgIndentation
(
    int                 dbgNewIndent
)
{
    int                 dbgOldIndent = s_dbgIndent;

    // Set the new indentation value and return the old indentation value
    s_dbgIndent = dbgNewIndent;

    return dbgOldIndent;

} // dbgIndentation

//******************************************************************************

void
dbgResetIndent()
{
    // Reset the debug output indention values
    s_dbgIndent    = 0;
    s_bNeedsIndent = true;

} // dbgResetIndent

//******************************************************************************

bool
outputFlush()
{
    // Return the current flush value
    return s_bFlush;

} // outputFlush

//******************************************************************************

bool
outputFlush
(
    bool                bFlush
)
{
    bool                bLastFlush = s_bFlush;

    // Set the new flush value
    s_bFlush = bFlush;

    // Return the last flush value
    return bLastFlush;

} // outputFlush

//******************************************************************************

HRESULT
getCaptureBuffer
(
    char              **pBuffer,
    ULONG              *pSize
)
{
    HRESULT             hResult;

    assert(pBuffer != NULL);
    assert(pSize != NULL);

    // Clear buffer pointer and size
    *pBuffer = NULL;
    *pSize   = 0;

    // Get the size of the captured output
    hResult = GetCaptureSize(pSize);
    if (SUCCEEDED(hResult))
    {
        // Check for captured output present
        if (*pSize != 0)
        {
            // Try to allocate buffer to hold capture output
            *pBuffer = new char[*pSize];
            if (*pBuffer != NULL)
            {
                // Try to get the captured output
                hResult = GetCaptureOutput(*pBuffer);
                if (SUCCEEDED(hResult))
                {
                    // Clear the capture buffer
                    ClearCapture();
                }
                // Check capture output for Ctrl-C break
                if (strstr(*pBuffer, BREAK_STRING) != 0)
                {
                    // Throw break exception
                    throw CBreakException(STATUS_CONTROL_BREAK_EXIT);
                }
            }
        }
    }
    return hResult;

} // getCaptureBuffer

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
