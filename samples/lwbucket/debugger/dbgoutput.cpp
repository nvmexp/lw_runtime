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
|*  Module: dbgoutput.cpp                                                     *|
|*                                                                            *|
 \****************************************************************************/
#include "dbgprecomp.h"

//******************************************************************************
//
//  dbg namespace
//
//******************************************************************************
namespace dbg
{
//******************************************************************************
//
// Undefine any interface macros for routines that this code will declare
//
//******************************************************************************
#undef GetOutputState
#undef GetDisplayState
#undef GetCaptureState
#undef SetOutputState
#undef SetDisplayState
#undef SetCaptureState
#undef GetCaptureSize
#undef GetCaptureOutput
#undef ClearCapture

//******************************************************************************
//
//  CLwstomDebugOutputCallbacks Interface
//
//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::QueryInterface
(
    THIS_
    __in  REFIID        InterfaceId,
    __out PVOID*        Interface
)
{
    HRESULT             hResult = E_NOINTERFACE;

    assert(Interface != NULL);

    if (IsEqualIID(InterfaceId , __uuidof(IUnknown))               ||
        IsEqualIID(InterfaceId , __uuidof(IDebugOutputCallbacks))  ||
        IsEqualIID(InterfaceId , __uuidof(IDebugOutputCallbacks2)))
    {
        *Interface = (IDebugOutputCallbacks *)this;
        AddRef ( ) ;

        hResult = S_OK;
    }
    return hResult;

} // QueryInterface

//******************************************************************************

STDMETHODIMP_(ULONG)
CLwstomDebugOutputCallbacks::AddRef
(
    THIS_
)
{
    // Increment and return the new interface reference count
    return InterlockedIncrement(&m_lRefCount);

} // AddRef

//******************************************************************************

STDMETHODIMP_(ULONG)
CLwstomDebugOutputCallbacks::Release
(
    THIS_
)
{
    LONG                lRefCount;

    // Decrement the interface reference count
    lRefCount = InterlockedDecrement(&m_lRefCount);

    // Free the interface if no longer referenced
    if (lRefCount == 0)
    {
        delete this;
    }
    return lRefCount;

} // Release

//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::Output
(
    THIS_
    __in  ULONG         Mask,
    __in  PCSTR         Text
)
{
    bool                bProgressIndicator;
    bool                bProgressStatus;
    ULONG               ulLength;
    ULONG               ulNewBufferSize;
    char               *pNewCaptureBuffer;
    const char         *pProgressString;
    HRESULT             hResult = S_OK;

    assert(Text != NULL);

    // Get the current progress indicator and status
    bProgressIndicator = progressIndicator();
    bProgressStatus    = progressStatus();

    // Only need to check progress indicator if displaying output
    if (m_bDisplayState == TRUE)
    {
        // Check for non-progress output (Need to make sure progress indicator is off)
        if (bProgressStatus == NOT_IN_PROGRESS)
        {
            // Check to see if progress indicator is on (Needs to be off)
            if (bProgressIndicator == INDICATOR_ON)
            {
                // Get the current progress indicator clear string
                pProgressString = progressClearString();

                // Output progress string to original outputs callbacks (If present)
                if (m_pDebugOutputCallbacks != NULL)
                {
                    hResult = m_pDebugOutputCallbacks->Output(Mask, pProgressString);
                }
            }
        }
    }
    // Only capture non-progress indicator output
    if (bProgressStatus == NOT_IN_PROGRESS)
    {
        // Check for capturing output
        if (m_bCaptureState == TRUE)
        {
            // Get the length of the new output text
            ulLength = static_cast<ULONG>(strlen(Text));

            // Check for not enough room in capture buffer for new text
            if ((m_ulBufferOffset + ulLength) >= m_ulBufferSize)
            {
                // Loop computing the new required capture buffer size
                ulNewBufferSize = m_ulBufferSize * 2;
                while((m_ulBufferOffset + ulLength) >= ulNewBufferSize)
                {
                    // Double buffer size until we get enough space
                    ulNewBufferSize *= 2;
                }
                // Try to reallocate the capture buffer
                pNewCaptureBuffer = charptr(realloc(m_pCaptureBuffer, ulNewBufferSize));

                // Check for capture buffer reallocation
                if (pNewCaptureBuffer != NULL)
                {
                    // Setup the new capture buffer
                    m_pCaptureBuffer = pNewCaptureBuffer;
                    m_ulBufferSize   = ulNewBufferSize;
                }
                else    // Capture buffer reallocation failure
                {
                    // Indicate capture buffer reallocation failure (Don't copy text)
                    hResult  = E_OUTOFMEMORY;
                    ulLength = 0;
                }
            }
            // Capture new output text (If any and no reallocation failure)
            if (ulLength != 0)
            {
                strcpy(&m_pCaptureBuffer[m_ulBufferOffset], Text);
                m_ulBufferOffset += ulLength;
            }
        }
    }
    // Check for displaying output
    if (m_bDisplayState == TRUE)
    {
        // Output the new text to original output callbacks (If present)
        if (m_pDebugOutputCallbacks != NULL)
        {
            hResult = m_pDebugOutputCallbacks->Output(Mask, Text);
        }
        // Check for non-progress output (Need to re-enable progress indicator if necessary)
        if (bProgressStatus == NOT_IN_PROGRESS)
        {
            // Check to see if progress indicator is on (Needs to be re-enabled)
            if (bProgressIndicator == INDICATOR_ON)
            {
                // Get the current progress indicator string
                pProgressString = progressIndicatorString();

                // Output progress string to original output callbacks (If present)
                if (m_pDebugOutputCallbacks != NULL)
                {
                    hResult = m_pDebugOutputCallbacks->Output(Mask, pProgressString);
                }
            }
        }
    }
    return hResult;

} // Output

//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::GetInterestMask
(
    THIS_
    __out PULONG        Mask
)
{
    HRESULT             hResult = S_OK;

    assert(Mask != NULL);

    // Set the interest mask (Any format output + explicit flushes)
    *Mask = DEBUG_OUTCBI_ANY_FORMAT | DEBUG_OUTCBI_EXPLICIT_FLUSH;

    return hResult;

} // GetInterestMask

//******************************************************************************
    
STDMETHODIMP
CLwstomDebugOutputCallbacks::Output2
(
    THIS_
    __in ULONG          Which,
    __in ULONG          Flags,
    __in ULONG64        Arg,
    __in_opt PCWSTR     Text
)
{
    bool                bProgressIndicator;
    bool                bProgressStatus;
    ULONG               ulLength;
    ULONG               ulNewBufferSize;
    char               *pNewCaptureBuffer;
    const char         *pProgressString;
    WCHAR               progressString[80];
    CString             sString;
    CString             sStripped;
    HRESULT             hResult = S_OK;

    // Get the current progress indicator and status
    bProgressIndicator = progressIndicator();
    bProgressStatus    = progressStatus();

    // Only need to check progress indicator if displaying output
    if (m_bDisplayState == TRUE)
    {
        // Check for non-progress output (Need to make sure progress indicator is off)
        if (bProgressStatus == NOT_IN_PROGRESS)
        {
            // Check to see if progress indicator is on (Needs to be off)
            if (bProgressIndicator == INDICATOR_ON)
            {
                // Get the current progress indicator clear string
                pProgressString = progressClearString();

                // Colwert progress string to wide character (Including terminator)
                ulLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, pProgressString, -1, progressString, static_cast<int>(countof(progressString)));

                // Check for no colwersion errors
                if (ulLength != 0)
                {
                    // Output progress string to original outputs callbacks (If present)
                    if (m_pDebugOutputCallbacks2 != NULL)
                    {
                        hResult = m_pDebugOutputCallbacks2->Output2(Which, Flags, Arg, progressString);
                    }
                }
                else    // Colwersion errors
                {
                    // Get the colwersion error
                    hResult = GetLastError();
                }
            }
        }
    }
    // Set length to zero in case there is no text
    ulLength = 0;

    // Switch on the output2 type
    switch(Which)
    {
        case DEBUG_OUTCB_TEXT:              // Plain text output

            // Only capture non-progress indicator output
            if (bProgressStatus == NOT_IN_PROGRESS)
            {
                // Check for capturing output
                if (m_bCaptureState == TRUE)
                {
                    // Only capture if there is actual text
                    if (Text != NULL)
                    {
                        // Get the length of the new output text (ANSI for the capture, including terminator)
                        ulLength = WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, Text, -1, NULL, 0, NULL, NULL);

                        // Check for not enough room in capture buffer for new text
                        if ((m_ulBufferOffset + ulLength) >= m_ulBufferSize)
                        {
                            // Loop computing the new required capture buffer size
                            ulNewBufferSize = m_ulBufferSize * 2;
                            while((m_ulBufferOffset + ulLength) >= ulNewBufferSize)
                            {
                                // Double buffer size until we get enough space
                                ulNewBufferSize *= 2;
                            }
                            // Try to reallocate the capture buffer
                            pNewCaptureBuffer = charptr(realloc(m_pCaptureBuffer, ulNewBufferSize));

                            // Check for capture buffer reallocation
                            if (pNewCaptureBuffer != NULL)
                            {
                                // Setup the new capture buffer
                                m_pCaptureBuffer = pNewCaptureBuffer;
                                m_ulBufferSize   = ulNewBufferSize;
                            }
                            else    // Capture buffer reallocation failure
                            {
                                // Indicate capture buffer reallocation failure (Don't copy text)
                                hResult  = E_OUTOFMEMORY;
                                ulLength = 0;
                            }
                        }
                    }
                    // Capture new output text (If any and no reallocation failure)
                    if (ulLength != 0)
                    {
                        // Colwert output text to ANSI (Including NULL terminator)
                        ulLength = WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, Text, -1, &m_pCaptureBuffer[m_ulBufferOffset], ulLength, NULL, NULL);
                        if (ulLength != 0)
                        {
                            // Update the current buffer offset (Don't include terminator)
                            m_ulBufferOffset += ulLength - 1;
                        }
                        else    // Colwersion error
                        {
                            // Get the colwersion error
                            hResult = GetLastError();
                        }
                    }
                }
            }
            // Check for displaying output
            if (m_bDisplayState == TRUE)
            {
                // Check if original output interface interested in plain text
                if ((m_ulMask & DEBUG_OUTCBI_TEXT) != 0)
                {
                    // Output the new text to original outputs callbacks (If present)
                    if (m_pDebugOutputCallbacks2 != NULL)
                    {
                        hResult = m_pDebugOutputCallbacks2->Output2(Which, Flags, Arg, Text);
                    }
                }
            }
            break;

        case DEBUG_OUTCB_DML:               // DML output

            // Only capture non-progress indicator output
            if (bProgressStatus == NOT_IN_PROGRESS)
            {
                // Check for capturing output
                if (m_bCaptureState == TRUE)
                {
                    // Only capture if there is actual text
                    if (Text != NULL)
                    {
                        // Get the length of the new output text (ANSI for the capture, including terminator)
                        ulLength = WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, Text, -1, NULL, 0, NULL, NULL);

                        // Check for not enough room in capture buffer for new text
                        if ((m_ulBufferOffset + ulLength) >= m_ulBufferSize)
                        {
                            // Loop computing the new required capture buffer size
                            ulNewBufferSize = m_ulBufferSize * 2;
                            while((m_ulBufferOffset + ulLength) >= ulNewBufferSize)
                            {
                                // Double buffer size until we get enough space
                                ulNewBufferSize *= 2;
                            }
                            // Try to reallocate the capture buffer
                            pNewCaptureBuffer = charptr(realloc(m_pCaptureBuffer, ulNewBufferSize));

                            // Check for capture buffer reallocation
                            if (pNewCaptureBuffer != NULL)
                            {
                                // Setup the new capture buffer
                                m_pCaptureBuffer = pNewCaptureBuffer;
                                m_ulBufferSize   = ulNewBufferSize;
                            }
                            else    // Capture buffer reallocation failure
                            {
                                // Indicate capture buffer reallocation failure (Don't copy text)
                                hResult  = E_OUTOFMEMORY;
                                ulLength = 0;
                            }
                        }
                    }
                    // Capture new output text (If any and no reallocation failure)
                    if (ulLength != 0)
                    {
                        // Reserve enough space for the captured string (Including terminator)
                        sString.reserve(ulLength);

                        // Colwert output text to ANSI (Including NULL terminator)
                        ulLength = WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, Text, -1, sString.data(), ulLength, NULL, NULL);
                        if (ulLength != 0)
                        {
                            // Strip any DML from the captured text (get new length)
                            sStripped = dmlStrip(sString);
                            ulLength  = static_cast<ULONG>(sStripped.length());

                            // Copied stripped text to capture buffer (Including terminator)
                            strcpy(&m_pCaptureBuffer[m_ulBufferOffset], sStripped);

                            // Update the current buffer offset
                            m_ulBufferOffset += ulLength;
                        }
                        else    // Colwersion error
                        {
                            // Get the colwersion error
                            hResult = GetLastError();
                        }
                    }
                }
            }
            // Check for displaying output
            if (m_bDisplayState == TRUE)
            {
                // Check if original output interface interested in DML
                if ((m_ulMask & DEBUG_OUTCBI_DML) != 0)
                {
                    // Output the new text to original outputs callbacks (If present)
                    if (m_pDebugOutputCallbacks2 != NULL)
                    {
                        hResult = m_pDebugOutputCallbacks2->Output2(Which, Flags, Arg, Text);
                    }
                }
            }
            break;

        case DEBUG_OUTCB_EXPLICIT_FLUSH:    // Explicit flush

            // Check if original output interface interested in explicit flush
            if ((m_ulMask & DEBUG_OUTCBI_EXPLICIT_FLUSH) != 0)
            {
                // Check for displaying output
                if (m_bDisplayState == TRUE)
                {
                    // Send explicit flush to original outputs callbacks (If present)
                    if (m_pDebugOutputCallbacks2 != NULL)
                    {
                        hResult = m_pDebugOutputCallbacks2->Output2(Which, Flags, Arg, Text);
                    }
                }
            }
            break;

        default:                            // Unknown output type

            hResult = E_FAIL;

            break;
    }
    // Only need to check progress indicator if displaying output
    if (m_bDisplayState == TRUE)
    {
        // Check for non-progress output (Need to re-enable progress indicator if necessary)
        if (bProgressStatus == NOT_IN_PROGRESS)
        {
            // Check to see if progress indicator is on (Needs to be re-enabled)
            if (bProgressIndicator == INDICATOR_ON)
            {
                // Get the current progress indicator string
                pProgressString = progressIndicatorString();

                // Colwert progress string to wide character (Including terminator)
                ulLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, pProgressString, -1, progressString, static_cast<int>(countof(progressString)));

                // Terminate the progress string
                progressString[ulLength] = NULL;

                // Check for no colwersion errors
                if (ulLength != 0)
                {
                    // Output progress string to original outputs callbacks (If present)
                    if (m_pDebugOutputCallbacks2 != NULL)
                    {
                        hResult = m_pDebugOutputCallbacks2->Output2(Which, Flags, Arg, progressString);
                    }
                }
                else    // Colwersion error
                {
                    // Get the colwersion error
                    hResult = GetLastError();
                }
            }
        }
    }
    return hResult;

} // Output2

//******************************************************************************

STDMETHODIMP_(BOOL)
CLwstomDebugOutputCallbacks::GetDisplayState
(
    THIS
)
{
    // Return the current display output state
    return m_bDisplayState;

} // GetDisplayState

//******************************************************************************

STDMETHODIMP_(BOOL)
CLwstomDebugOutputCallbacks::GetCaptureState
(
    THIS
)
{
    // Return the current capture output state
    return m_bCaptureState;

} // GetCaptureState

//******************************************************************************

STDMETHODIMP_(BOOL)
CLwstomDebugOutputCallbacks::GetDmlState
(
    THIS
)
{
    // Return the current DML state
    return m_bDmlState;

} // GetDmlState

//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::SetDisplayState
(
    THIS_
    __in BOOL           bDisplayState
)
{
    HRESULT             hResult = S_OK;

    // Set the new display output state
    m_bDisplayState = bDisplayState;

    return hResult;

} // SetDisplayState

//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::SetCaptureState
(
    THIS_
    __in BOOL           bCaptureState
)
{
    HRESULT             hResult = S_OK;

    // Check new capture state to see if enabled
    if (bCaptureState == TRUE)
    {
        // Check for a capture buffer allocated
        if (m_pCaptureBuffer == NULL)
        {
            // Setup the new capture buffer size and try to allocate capture buffer
            m_ulBufferSize   = 1024;
            m_pCaptureBuffer = static_cast<char*>(malloc(m_ulBufferSize));

            // Check for capture buffer allocation
            if (m_pCaptureBuffer != NULL)
            {
                // Clear the capture buffer
                ClearCapture();
            }
            else    // Capture buffer allocation failure
            {
                // Indicate capture buffer allocation failure (Turn off capture)
                m_ulBufferSize = 0;
                hResult        = E_OUTOFMEMORY;
                bCaptureState  = FALSE;
            }
        }
    }
    // Set the new capture state
    m_bCaptureState = bCaptureState;

    return hResult;

} // SetCaptureState

//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::SetDmlState
(
    THIS_
    __in BOOL           bDmlState
)
{
    HRESULT             hResult = S_OK;

    // Set the new DML state
    m_bDmlState = bDmlState;

    // Check the new DML state and set the global DML state
    if (m_bDmlState)
    {
        // Add/enable the global DML engine option
        AddEngineOptions(DEBUG_ENGOPT_PREFER_DML);
    }
    else    // DML disabled
    {
        // Remove/disable the global DML engine option
        RemoveEngineOptions(DEBUG_ENGOPT_PREFER_DML);
    }
    return hResult;

} // SetDmlState

//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::GetCaptureSize
(
    THIS_
    __out PULONG        pSize
)
{
    HRESULT             hResult = S_OK;

    assert(pSize != NULL);

    // Get the current capture size (Buffer offset + 1) [Include terminator]
    *pSize = m_ulBufferOffset + 1;

    return hResult;

} // GetCaptureSize

//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::GetCaptureOutput
(
    THIS_
    __out PSTR          pBuffer
)
{
    HRESULT             hResult = S_OK;

    assert(pBuffer != NULL);

    // Copy the current capture buffer contents (If present)
    if (m_pCaptureBuffer != NULL)
    {
        strcpy(pBuffer, m_pCaptureBuffer);
    }
    return hResult;

} // GetCaptureOutput

//******************************************************************************

STDMETHODIMP
CLwstomDebugOutputCallbacks::ClearCapture
(
    THIS_
)
{
    HRESULT             hResult = S_OK;

    // Clear capture buffer contents (If allocated)
    if (m_pCaptureBuffer != NULL)
    {
        m_ulBufferOffset                   = 0;
        m_pCaptureBuffer[m_ulBufferOffset] = 0;
    }
    return hResult;

} // ClearCapture

//******************************************************************************

CLwstomDebugOutputCallbacks::CLwstomDebugOutputCallbacks()
:   m_lRefCount(0),
    m_ulMask(0),
    m_bOutputEnable(FALSE),
    m_bDisplayState(TRUE),
    m_bCaptureState(FALSE),
    m_bDmlState(TRUE),
    m_pCaptureBuffer(NULL),
    m_ulBufferOffset(0),
    m_ulBufferSize(0),
    m_pDebugOutputCallbacks(NULL),
    m_pDebugOutputCallbacks2(NULL)
{
    HRESULT             hResult;

    // Try to get existing debug output callbacks
    hResult = GetOutputCallbacks(&m_pDebugOutputCallbacks);
    if (SUCCEEDED(hResult))
    {
        // Check for output callbacks present (May be none)
        if (m_pDebugOutputCallbacks != NULL)
        {
            // Try to get the new debug output2 interface (May not be supported)
            m_pDebugOutputCallbacks->QueryInterface(__uuidof(IDebugOutputCallbacks2), pvoidptr(&m_pDebugOutputCallbacks2));

            // Check for new debug output2 interface supported (DML)
            if (m_pDebugOutputCallbacks2 != NULL)
            {
                // Get the interest mask for the debug output2 interface
                m_pDebugOutputCallbacks2->GetInterestMask(&m_ulMask);
            }
        }
        // Try to setup the custom debug output callbacks
        hResult = SetOutputCallbacks(funcptr(PDEBUG_OUTPUT_CALLBACKS, this));
    }

} // CLwstomDebugOutputCallbacks

//******************************************************************************

CLwstomDebugOutputCallbacks::~CLwstomDebugOutputCallbacks()
{
    // Free the capture buffer if still allocated
    if (m_pCaptureBuffer != NULL)
    {
        free(m_pCaptureBuffer);
        m_pCaptureBuffer = NULL;
    }
    // Restore the original output callbacks (ignore any errors)
    SetOutputCallbacks(m_pDebugOutputCallbacks);

} // ~CLwstomDebugOutputCallbacks

//******************************************************************************
//
// Custom Debug Output Interface wrappers
//
//******************************************************************************

BOOL
GetDisplayState()
{
    PLWSTOM_DEBUG_OUTPUT_CALLBACKS pDbgOutput = dbgOutput();
    BOOL                bDisplayState = FALSE;

    // Check for output interface
    if (pDbgOutput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetDisplayState debug output method
        bDisplayState = pDbgOutput->GetDisplayState();

        // Release the debug interface
        releaseDebugInterface();
    }
    return bDisplayState;

} // GetDisplayState

//******************************************************************************

BOOL
GetCaptureState()
{
    PLWSTOM_DEBUG_OUTPUT_CALLBACKS pDbgOutput = dbgOutput();
    BOOL                bCaptureState = FALSE;

    // Check for output interface
    if (pDbgOutput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetCaptureState debug output method
        bCaptureState = pDbgOutput->GetCaptureState();

        // Release the debug interface
        releaseDebugInterface();
    }
    return bCaptureState;

} // GetCaptureState

//******************************************************************************

HRESULT
SetDisplayState
(
    BOOL                bDisplayState
)
{
    PLWSTOM_DEBUG_OUTPUT_CALLBACKS pDbgOutput = dbgOutput();
    HRESULT             hResult = E_FAIL;

    // Check for output interface
    if (pDbgOutput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetDisplayState debug output method
        hResult = pDbgOutput->SetDisplayState(bDisplayState);

        // Release the debug interface
        releaseDebugInterface();
    }
    return breakCheck(hResult);

} // SetDisplayState

//******************************************************************************

HRESULT
SetCaptureState
(
    BOOL                bCaptureState
)
{
    PLWSTOM_DEBUG_OUTPUT_CALLBACKS pDbgOutput = dbgOutput();
    HRESULT             hResult = E_NOINTERFACE;

    // Check for output interface
    if (pDbgOutput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the SetCaptureState debug output method
        hResult = pDbgOutput->SetCaptureState(bCaptureState);

        // Release the debug interface
        releaseDebugInterface();
    }
    return breakCheck(hResult);

} // SetCaptureState

//******************************************************************************

HRESULT
GetCaptureSize
(
    PULONG              pSize
)
{
    PLWSTOM_DEBUG_OUTPUT_CALLBACKS pDbgOutput = dbgOutput();
    HRESULT             hResult = E_NOINTERFACE;

    assert(pSize != NULL);

    // Initialize the capture size (In case call below fails)
    *pSize = 0;

    // Check for output interface
    if (pDbgOutput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetCaptureSize debug output method
        hResult = pDbgOutput->GetCaptureSize(pSize);

        // Release the debug interface
        releaseDebugInterface();
    }
    return breakCheck(hResult);

} // GetCaptureSize

//******************************************************************************

HRESULT
GetCaptureOutput
(
    PSTR                pBuffer
)
{
    PLWSTOM_DEBUG_OUTPUT_CALLBACKS pDbgOutput = dbgOutput();
    HRESULT             hResult = E_NOINTERFACE;

    assert(pBuffer != NULL);

    // Check for output interface
    if (pDbgOutput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the GetCaptureOutput debug output method
        hResult = pDbgOutput->GetCaptureOutput(pBuffer);

        // Release the debug interface
        releaseDebugInterface();
    }
    return breakCheck(hResult);

} // GetCaptureOutput

//******************************************************************************

HRESULT
ClearCapture()
{
    PLWSTOM_DEBUG_OUTPUT_CALLBACKS pDbgOutput = dbgOutput();
    HRESULT             hResult = E_NOINTERFACE;

    // Check for output interface
    if (pDbgOutput != NULL)
    {
        // Acquire the debug interface
        acquireDebugInterface();

        // Call the ClearCapture debug output method
        hResult = pDbgOutput->ClearCapture();

        // Release the debug interface
        releaseDebugInterface();
    }
    return breakCheck(hResult);

} // ClearCapture

} // dbg namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
