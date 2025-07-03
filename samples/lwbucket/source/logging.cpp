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
|*  Module: logging.cpp                                                       *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

//******************************************************************************
//
// Locals
//
//******************************************************************************
static  CLoggingHook    s_loggingHook;

// Log Enum Helpers
CEnum           CLogging::m_dmInterfaceEnum               (&kmDriver(), "DmInterface");
CEnum           CLogging::m_lddmInterfaceEnum             (&kmDriver(), "LddmInterface");
CEnum           CLogging::m_dlLayerInterfaceEnum          (&kmDriver(), "DlLayerInterface");
CEnum           CLogging::m_rmInterfaceEnum               (&kmDriver(), "RmInterface");
CEnum           CLogging::m_cbInterfaceEnum               (&kmDriver(), "CbInterface");
CEnum           CLogging::m_ifaceInterfaceEnum            (&kmDriver(), "IfaceInterface");
CEnum           CLogging::m_agpInterfaceEnum              (&kmDriver(), "AgpInterface");
CEnum           CLogging::m_timedOpInterfaceEnum          (&kmDriver(), "TimedOpInterface");
CEnum           CLogging::m_clientArbSubtypeEnum          (&kmDriver(), "ClientArbitrationLogSubType");
CEnum           CLogging::m_gdiAccelOpTypeEnum            (&kmDriver(), "GDIHwAccelOpType");
CEnum           CLogging::m_bufferOperationEnum           (&kmDriver(), "BufferOperation");
CEnum           CLogging::m_idleOperationEnum             (&kmDriver(), "IdleOperation");
CEnum           CLogging::m_interruptEnum                 (&kmDriver(), "_DXGK_INTERRUPT_TYPE");
CEnum           CLogging::m_buildPagingBufferEnum         (&kmDriver(), "_DXGK_BUILDPAGINGBUFFER_OPERATION");

// Compiled regular expressions for matching known types
static  regex_t         s_reDmInterface      = {0};     // For matching DM interface type
static  regex_t         s_reLddmInterface    = {0};     // For matching LDDM interface type
static  regex_t         s_reDlInterface      = {0};     // For matching DL interface type
static  regex_t         s_reCbInterface      = {0};     // For matching CB interface type
static  regex_t         s_reIfaceInterface   = {0};     // For matching IFACE interface type
static  regex_t         s_reRmInterface      = {0};     // For matching RM interface type
static  regex_t         s_reAgpInterface     = {0};     // For matching AGP interface type
static  regex_t         s_reTimedOpInterface = {0};     // For matching timed operation interface type

static  regex_t         s_reClientArbType  = {0};       // For matching client arbitration type
static  regex_t         s_reGdiAccelOpType = {0};       // For matching GDI accelerated operation type

static  regex_t         s_reBufferOperation = {0};      // For matching buffer operation type
static  regex_t         s_reIdleOperation   = {0};      // For matching idle operation type
static  regex_t         s_reIsrNotify       = {0};      // For matching ISR notify type

static  regex_t         s_reBuildPagingBuffer = {0};    // For matching build paging buffer operation type

//******************************************************************************

HRESULT
CLoggingHook::initialize
(
    const PULONG        pVersion,
    const PULONG        pFlags
)
{
    UNREFERENCED_PARAMETER(pVersion);
    UNREFERENCED_PARAMETER(pFlags);

    int                 reResult;
    CString             sError(MAX_DBGPRINTF_STRING);
    HRESULT             hResult = S_OK;

    assert(pVersion != NULL);
    assert(pFlags != NULL);

    // Catch any regular expression failures
    try
    {
        // Try to compile the type matching regular expressions
        reResult = regcomp(&s_reDmInterface, DM_INTERFACE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reDmInterface, DM_INTERFACE));
        }
        reResult = regcomp(&s_reLddmInterface, LDDM_INTERFACE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reLddmInterface, LDDM_INTERFACE));
        }
        reResult = regcomp(&s_reDlInterface, DL_INTERFACE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reDlInterface, DL_INTERFACE));
        }
        reResult = regcomp(&s_reCbInterface, CB_INTERFACE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reCbInterface, CB_INTERFACE));
        }
        reResult = regcomp(&s_reIfaceInterface, IFACE_INTERFACE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reIfaceInterface, IFACE_INTERFACE));
        }
        reResult = regcomp(&s_reRmInterface, RM_INTERFACE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reRmInterface, RM_INTERFACE));
        }
        reResult = regcomp(&s_reAgpInterface, AGP_INTERFACE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reAgpInterface, AGP_INTERFACE));
        }
        reResult = regcomp(&s_reTimedOpInterface, TIMED_OP_INTERFACE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reTimedOpInterface, TIMED_OP_INTERFACE));
        }
        reResult = regcomp(&s_reClientArbType, CLIENT_ARB_TYPE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reClientArbType, CLIENT_ARB_TYPE));
        }
        reResult = regcomp(&s_reGdiAccelOpType, GDI_ACCEL_OP_TYPE, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reGdiAccelOpType, GDI_ACCEL_OP_TYPE));
        }
        reResult = regcomp(&s_reBufferOperation, BUFFER_OPERATION, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reBufferOperation, BUFFER_OPERATION));
        }
        reResult = regcomp(&s_reIdleOperation, IDLE_OPERATION, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reIdleOperation, IDLE_OPERATION));
        }
        reResult = regcomp(&s_reIsrNotify, ISR_NOTIFY, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reIsrNotify, ISR_NOTIFY));
        }
        reResult = regcomp(&s_reBuildPagingBuffer, BUILD_PAGING_BUFFER, REG_EXTENDED + REG_ICASE);
        if (reResult != REG_NOERROR)
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             regString(reResult, &s_reBuildPagingBuffer, BUILD_PAGING_BUFFER));
        }
    }
    catch (CException& exception)
    {
        UNREFERENCED_PARAMETER(exception);

        // Free any allocated logging resources
        uninitialize();

        throw;
    }
     return hResult;

} // initialize

//******************************************************************************

void
CLoggingHook::uninitialize(void)
{
    // Free the type matching regular expressions
    regfree(&s_reDmInterface);
    regfree(&s_reLddmInterface);
    regfree(&s_reDlInterface);
    regfree(&s_reCbInterface);
    regfree(&s_reRmInterface);
    regfree(&s_reIfaceInterface);
    regfree(&s_reAgpInterface);
    regfree(&s_reTimedOpInterface);

    regfree(&s_reClientArbType);
    regfree(&s_reGdiAccelOpType);

    regfree(&s_reBufferOperation);
    regfree(&s_reIdleOperation);
    regfree(&s_reIsrNotify);
    regfree(&s_reBuildPagingBuffer);

} // uninitialize

//******************************************************************************

bool
isDmInterfaceType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bDmInterfaceType = false;

    // Check given type for a DM interface type match
    reResult = regexec(&s_reDmInterface, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a DM interface type
        bDmInterfaceType = true;
    }
    return bDmInterfaceType;

} // isDmInterfaceType

//******************************************************************************

bool
isLddmInterfaceType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bLddmInterfaceType = false;

    // Check given type for a LDDM interface type match
    reResult = regexec(&s_reLddmInterface, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a LDDM interface type
        bLddmInterfaceType = true;
    }
    return bLddmInterfaceType;

} // isLddmInterfaceType

//******************************************************************************

bool
isDlInterfaceType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bDlInterfaceType = false;

    // Check given type for a DL interface type match
    reResult = regexec(&s_reDlInterface, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a DL interface type
        bDlInterfaceType = true;
    }
    return bDlInterfaceType;

} // isDlInterfaceType

//******************************************************************************

bool
isCbInterfaceType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bCbInterfaceType = false;

    // Check given type for a CB interface type match
    reResult = regexec(&s_reCbInterface, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a CB interface type
        bCbInterfaceType = true;
    }
    return bCbInterfaceType;

} // isCbInterfaceType

//******************************************************************************

bool
isIfaceInterfaceType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bIfaceInterfaceType = false;

    // Check given type for a IFACE interface type match
    reResult = regexec(&s_reIfaceInterface, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a IFACE interface type
        bIfaceInterfaceType = true;
    }
    return bIfaceInterfaceType;

} // isIfaceInterfaceType

//******************************************************************************

bool
isRmInterfaceType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bRmInterfaceType = false;

    // Check given type for a RM interface type match
    reResult = regexec(&s_reRmInterface, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a RM interface type
        bRmInterfaceType = true;
    }
    return bRmInterfaceType;

} // isRmInterfaceType

//******************************************************************************

bool
isAgpInterfaceType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bAgpInterfaceType = false;

    // Check given type for an AGP interface type match
    reResult = regexec(&s_reAgpInterface, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is an AGP interface type
        bAgpInterfaceType = true;
    }
    return bAgpInterfaceType;

} // isAgpInterfaceType

//******************************************************************************

bool
isTimedOpInterfaceType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bTimedOpInterfaceType = false;

    // Check given type for a TimedOp interface type match
    reResult = regexec(&s_reTimedOpInterface, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a TimedOp interface type
        bTimedOpInterfaceType = true;
    }
    return bTimedOpInterfaceType;

} // isTimedOpInterfaceType

//******************************************************************************

bool
isClientArbType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bClientArbType = false;

    // Check given type for a Client Arbitration type match
    reResult = regexec(&s_reClientArbType, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a Client Arbitration type
        bClientArbType = true;
    }
    return bClientArbType;

} // isClientArbType

//******************************************************************************

bool
isGdiAccelOpType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bGdiAccelOpType = false;

    // Check given type for a GDI Accelerated operation type match
    reResult = regexec(&s_reGdiAccelOpType, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a GDI Accelerated operation type
        bGdiAccelOpType = true;
    }
    return bGdiAccelOpType;

} // isGdiAccelOpType

//******************************************************************************

bool
isBufferOperationType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bBufferOperationType = false;

    // Check given type for a buffer operation type match
    reResult = regexec(&s_reBufferOperation, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a buffer operation type
        bBufferOperationType = true;
    }
    return bBufferOperationType;

} // isBufferOperationType

//******************************************************************************

bool
isIdleOperationType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bIdleOperationType = false;

    // Check given type for a idle operation type match
    reResult = regexec(&s_reIdleOperation, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a idle operation type
        bIdleOperationType = true;
    }
    return bIdleOperationType;

} // isIdleOperationType

//******************************************************************************

bool
isIsrNotifyType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bIsrNotifyType = false;

    // Check given type for an ISR notify type match
    reResult = regexec(&s_reIsrNotify, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is an ISR notify type
        bIsrNotifyType = true;
    }
    return bIsrNotifyType;

} // isIsrNotifyType

//******************************************************************************

bool
isBuildPagingBufferType
(
    const char         *pType
)
{
    regmatch_t          reMatch[10];
    int                 reResult;
    bool                bBuildPagingBufferType = false;

    // Check given type for a build paging buffer type match
    reResult = regexec(&s_reBuildPagingBuffer, pType, countof(reMatch), reMatch, 0);
    if (reResult == REG_NOERROR)
    {
        // Indicate this is a build paging buffer type
        bBuildPagingBufferType = true;
    }
    return bBuildPagingBufferType;

} // isBuildPagingBufferType

//******************************************************************************

const CValue*
dmInterfaceValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known DM interfaces for a matching string
        if (CLogging::dmInterfaceEnum().isPresent())
        {
            // Loop checking the DM interface subtype values
            for (ulValue = 0; ulValue < CLogging::dmInterfaceEnum().values(); ulValue++)
            {
                // Get the next DM interface value
                pValue = CLogging::dmInterfaceEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next DM interface name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching DM interface name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // dmInterfaceValue

//******************************************************************************

const CValue*
lddmInterfaceValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known LDDM interfaces for a matching string
        if (CLogging::lddmInterfaceEnum().isPresent())
        {
            // Loop checking the LDDM interface subtype values
            for (ulValue = 0; ulValue < CLogging::lddmInterfaceEnum().values(); ulValue++)
            {
                // Get the next LDDM interface value
                pValue = CLogging::lddmInterfaceEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next LDDM interface name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching LDDM interface name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // lddmInterfaceValue

//******************************************************************************

const CValue*
dlInterfaceValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known DL interfaces for a matching string
        if (CLogging::dlLayerInterfaceEnum().isPresent())
        {
            // Loop checking the DL interface subtype values
            for (ulValue = 0; ulValue < CLogging::dlLayerInterfaceEnum().values(); ulValue++)
            {
                // Get the next DL interface value
                pValue = CLogging::dlLayerInterfaceEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next DL interface name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching DL interface name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // dlInterfaceValue

//******************************************************************************

const CValue*
cbInterfaceValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known CB interfaces for a matching string
        if (CLogging::cbInterfaceEnum().isPresent())
        {
            // Loop checking the CB interface subtype values
            for (ulValue = 0; ulValue < CLogging::cbInterfaceEnum().values(); ulValue++)
            {
                // Get the next CB interface value
                pValue = CLogging::cbInterfaceEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next CB interface name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching CB interface name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // cbInterfaceValue

//******************************************************************************

const CValue*
ifaceInterfaceValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known IFACE interfaces for a matching string
        if (CLogging::ifaceInterfaceEnum().isPresent())
        {
            // Loop checking the IFACE interface subtype values
            for (ulValue = 0; ulValue < CLogging::ifaceInterfaceEnum().values(); ulValue++)
            {
                // Get the next IFACE interface value
                pValue = CLogging::ifaceInterfaceEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next IFACE interface name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching IFACE interface name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // ifaceInterfaceValue

//******************************************************************************

const CValue*
rmInterfaceValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known RM interfaces for a matching string
        if (CLogging::rmInterfaceEnum().isPresent())
        {
            // Loop checking the RM interface subtype values
            for (ulValue = 0; ulValue < CLogging::rmInterfaceEnum().values(); ulValue++)
            {
                // Get the next RM interface value
                pValue = CLogging::rmInterfaceEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next RM interface name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching RM interface name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // rmInterfaceValue

//******************************************************************************

const CValue*
agpInterfaceValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known AGP interfaces for a matching string
        if (CLogging::agpInterfaceEnum().isPresent())
        {
            // Loop checking the AGP interface subtype values
            for (ulValue = 0; ulValue < CLogging::agpInterfaceEnum().values(); ulValue++)
            {
                // Get the next AGP interface value
                pValue = CLogging::agpInterfaceEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next AGP interface name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching AGP interface name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // agpInterfaceValue

//******************************************************************************

const CValue*
timedOpInterfaceValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known TimedOp interfaces for a matching string
        if (CLogging::timedOpInterfaceEnum().isPresent())
        {
            // Loop checking the TimedOp interface subtype values
            for (ulValue = 0; ulValue < CLogging::timedOpInterfaceEnum().values(); ulValue++)
            {
                // Get the next TimedOp interface value
                pValue = CLogging::timedOpInterfaceEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next TimedOp interface name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching TimedOp interface name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // timedOpInterfaceValue

//******************************************************************************

const CValue*
clientArbSubtypeValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known Client Arbitration subtypes for a matching string
        if (CLogging::clientArbSubtypeEnum().isPresent())
        {
            // Loop checking the Client Arbitration subtype values
            for (ulValue = 0; ulValue < CLogging::clientArbSubtypeEnum().values(); ulValue++)
            {
                // Get the next Client Arbitration subtype value
                pValue = CLogging::clientArbSubtypeEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next Client Arbitration subtype name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching Client Arbitration subtype name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // clientArbSubtypeValue

//******************************************************************************

const CValue*
gdiAccelOpTypeValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known GDI Accelerated operation types for a matching string
        if (CLogging::gdiAccelOpTypeEnum().isPresent())
        {
            // Loop checking the GDI Accelerated operation type values
            for (ulValue = 0; ulValue < CLogging::gdiAccelOpTypeEnum().values(); ulValue++)
            {
                // Get the next GDI Accelerated operation type value
                pValue = CLogging::gdiAccelOpTypeEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next GDI Accelerated operation type name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching GDI Accelerated operation type name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // gdiAccelOpTypeValue

//******************************************************************************

const CValue*
bufferOperatiolwalue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known buffer operations for a matching string
        if (CLogging::bufferOperationEnum().isPresent())
        {
            // Loop checking the buffer operation subtype values
            for (ulValue = 0; ulValue < CLogging::bufferOperationEnum().values(); ulValue++)
            {
                // Get the next buffer operation value
                pValue = CLogging::bufferOperationEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next buffer operation name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching buffer operation name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // bufferOperatiolwalue

//******************************************************************************

const CValue*
idleOperatiolwalue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known idle operations for a matching string
        if (CLogging::idleOperationEnum().isPresent())
        {
            // Loop checking the idle operation subtype values
            for (ulValue = 0; ulValue < CLogging::idleOperationEnum().values(); ulValue++)
            {
                // Get the next buffer operation value
                pValue = CLogging::idleOperationEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next idle operation name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching idle operation name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // idleOperatiolwalue

//******************************************************************************

const CValue*
isrNotifyValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known ISR notifies for a matching string
        if (CLogging::interruptEnum().isPresent())
        {
            // Loop checking the ISR notify subtype values
            for (ulValue = 0; ulValue < CLogging::interruptEnum().values(); ulValue++)
            {
                // Get the next ISR notify value
                pValue = CLogging::interruptEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next ISR notify name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching ISR notify name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // isrNotifyValue

//******************************************************************************

const CValue*
buildPagingBufferValue
(
    const char         *pString
)
{
    regex_t             reString = {0};
    regmatch_t          reMatch[10];
    int                 reResult;
    ULONG               ulValue;
    const CValue       *pValue = NULL;

    // Try to compile the given string as a case insensitive regular expression
    reResult = regcomp(&reString, pString, REG_EXTENDED + REG_ICASE);
    if (reResult == REG_NOERROR)
    {
        // Search all known build paging buffer for a matching string
        if (CLogging::buildPagingBufferEnum().isPresent())
        {
            // Loop checking the build paging buffer subtype values
            for (ulValue = 0; ulValue < CLogging::buildPagingBufferEnum().values(); ulValue++)
            {
                // Get the next build paging buffer value
                pValue = CLogging::buildPagingBufferEnum().value(ulValue);
                if (pValue != NULL)
                {
                    // Compare the given subtype and next build paging buffer name string
                    reResult = regexec(&reString, pValue->name(), countof(reMatch), reMatch, 0);
                    if (reResult == REG_NOERROR)
                    {
                        // Found matching build paging bufffer name (stop search)
                        break;
                    }
                    else    // Not a match
                    {
                        pValue = NULL;
                    }
                }
            }
        }
        // Free the compiled regular expression
        regfree(&reString);
    }
    else    // Invalid regular expression
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         regString(reResult, &reString, pString));
    }
    return pValue;

} // buildPagingBufferValue

//******************************************************************************

CString
dmInterfaceName
(
    ULONG64             ulValue
)
{
    CString             sDmInterface(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get DM interface name (if present)
        if (CLogging::dmInterfaceEnum().isPresent())
        {
            CLogging::dmInterfaceEnum().getConstantName(ulValue, sDmInterface.data(), static_cast<ULONG>(sDmInterface.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sDmInterface;

} // dmInterfaceName

//******************************************************************************

CString
lddmInterfaceName
(
    ULONG64             ulValue
)
{
    CString             sLddmInterface(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get LDDM interface name (if present)
        if (CLogging::lddmInterfaceEnum().isPresent())
        {
            CLogging::lddmInterfaceEnum().getConstantName(ulValue, sLddmInterface.data(), static_cast<ULONG>(sLddmInterface.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sLddmInterface;

} // lddmInterfaceName

//******************************************************************************

CString
dlInterfaceName
(
    ULONG64             ulValue
)
{
    CString             sDlInterface(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get DL interface name (if present)
        if (CLogging::dlLayerInterfaceEnum().isPresent())
        {
            CLogging::dlLayerInterfaceEnum().getConstantName(ulValue, sDlInterface.data(), static_cast<ULONG>(sDlInterface.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sDlInterface;

} // dlInterfaceName

//******************************************************************************

CString
cbInterfaceName
(
    ULONG64             ulValue
)
{
    CString             sCbInterface(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get CB interface name (if present)
        if (CLogging::cbInterfaceEnum().isPresent())
        {
            CLogging::cbInterfaceEnum().getConstantName(ulValue, sCbInterface.data(), static_cast<ULONG>(sCbInterface.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sCbInterface;

} // cbInterfaceName

//******************************************************************************

CString
ifaceInterfaceName
(
    ULONG64             ulValue
)
{
    CString             sIfaceInterface(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get IFACE interface name (if present)
        if (CLogging::ifaceInterfaceEnum().isPresent())
        {
            CLogging::ifaceInterfaceEnum().getConstantName(ulValue, sIfaceInterface.data(), static_cast<ULONG>(sIfaceInterface.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sIfaceInterface;

} // ifaceInterfaceName

//******************************************************************************

CString
rmInterfaceName
(
    ULONG64             ulValue
)
{
    CString             sRmInterface(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get RM interface name (if present)
        if (CLogging::rmInterfaceEnum().isPresent())
        {
            CLogging::rmInterfaceEnum().getConstantName(ulValue, sRmInterface.data(), static_cast<ULONG>(sRmInterface.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sRmInterface;

} // rmInterfaceName

//******************************************************************************

CString
agpInterfaceName
(
    ULONG64             ulValue
)
{
    CString             sAgpInterface(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get AGP interface name (if present)
        if (CLogging::agpInterfaceEnum().isPresent())
        {
            CLogging::agpInterfaceEnum().getConstantName(ulValue, sAgpInterface.data(), static_cast<ULONG>(sAgpInterface.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sAgpInterface;

} // agpInterfaceName

//******************************************************************************

CString
timedOpInterfaceName
(
    ULONG64             ulValue
)
{
    CString             sTimedOpInterface(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get TimedOp interface name (if present)
        if (CLogging::timedOpInterfaceEnum().isPresent())
        {
            CLogging::timedOpInterfaceEnum().getConstantName(ulValue, sTimedOpInterface.data(), static_cast<ULONG>(sTimedOpInterface.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sTimedOpInterface;

} // timedOpInterfaceName

//******************************************************************************

CString
clientArbSubtypeName
(
    ULONG64             ulValue
)
{
    CString             sClientArbSubtype(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get Client Arbitration subtype name (if present)
        if (CLogging::clientArbSubtypeEnum().isPresent())
        {
            CLogging::clientArbSubtypeEnum().getConstantName(ulValue, sClientArbSubtype.data(), static_cast<ULONG>(sClientArbSubtype.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sClientArbSubtype;

} // clientArbTypeName

//******************************************************************************

CString
gdiAccelOpTypeName
(
    ULONG64             ulValue
)
{
    CString             sGdiAccelOpType(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get GDI Accelerated operation type name (if present)
        if (CLogging::gdiAccelOpTypeEnum().isPresent())
        {
            CLogging::gdiAccelOpTypeEnum().getConstantName(ulValue, sGdiAccelOpType.data(), static_cast<ULONG>(sGdiAccelOpType.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sGdiAccelOpType;

} // gdiAccelOpTypeName

//******************************************************************************

CString
bufferOperationName
(
    ULONG64             ulValue
)
{
    CString             sBufferOperation(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get buffer operation name (if present)
        if (CLogging::bufferOperationEnum().isPresent())
        {
            CLogging::bufferOperationEnum().getConstantName(ulValue, sBufferOperation.data(), static_cast<ULONG>(sBufferOperation.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sBufferOperation;

} // bufferOperationName

//******************************************************************************

CString
idleOperationName
(
    ULONG64             ulValue
)
{
    CString             sIdleOperation(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get idle operation name (if present)
        if (CLogging::idleOperationEnum().isPresent())
        {
            CLogging::idleOperationEnum().getConstantName(ulValue, sIdleOperation.data(), static_cast<ULONG>(sIdleOperation.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sIdleOperation;

} // idleOperationName

//******************************************************************************

CString
isrNotifyName
(
    ULONG64             ulValue
)
{
    CString             sIsrNotify(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get ISR notify name (if present)
        if (CLogging::interruptEnum().isPresent())
        {
            CLogging::interruptEnum().getConstantName(ulValue, sIsrNotify.data(), static_cast<ULONG>(sIsrNotify.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sIsrNotify;

} // isrNotifyName

//******************************************************************************

CString
buildPagingBufferName
(
    ULONG64             ulValue
)
{
    CString             sBuildPagingBuffer(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Try to get build paging buffer name (if present)
        if (CLogging::buildPagingBufferEnum().isPresent())
        {
            CLogging::buildPagingBufferEnum().getConstantName(ulValue, sBuildPagingBuffer.data(), static_cast<ULONG>(sBuildPagingBuffer.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sBuildPagingBuffer;

} // buildPagingBufferName

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
