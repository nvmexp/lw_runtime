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
|*  Module: progress.cpp                                                      *|
|*                                                                            *|
 \****************************************************************************/
#include "../include/progress.h"

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  void            stepReset(void* pContext, const void* pInfo);
static  void            stepUpdate(void* pContext);
static  const char*     stepString(void* pContext);

static  void            integerReset(void* pContext, const void* pInfo);
static  void            integerUpdate(void* pContext);
static  const char*     integerString(void* pContext);

static  void            floatReset(void* pContext, const void* pInfo);
static  void            floatUpdate(void* pContext);
static  const char*     floatString(void* pContext);

static  void            barReset(void* pContext, const void* pInfo);
static  void            barUpdate(void* pContext);
static  const char*     barString(void* pContext);

static  const void*     progressInfo();
static  const void*     progressInfo(const void* pProgressInfo);

static  void            updateClear(const char* pString = NULL);

//******************************************************************************
//
// Locals
//
//******************************************************************************
// Spinner progress indicator
static  char*               s_pSpinnerSteps[]   = {
                                                   "|",
                                                   "/",
                                                   "-",
                                                   "\\",
                                                  };
static  STEP_INDICATOR      s_SpinnerIndicator  = {static_cast<ULONG>(countof(s_pSpinnerSteps)), s_pSpinnerSteps};
static  STEP_CONTEXT        s_SpinnerContext    = {0, &s_SpinnerIndicator};
static  PROGRESS_INDICATOR  s_Spinner           = {stepReset, stepUpdate, stepString, &s_SpinnerContext};

// Cylon progress indicator
static  char*               s_pCylonSteps[]     = {
                                                   "|",
                                                   " |",
                                                   "  |",
                                                   " |",
                                                  };
static  STEP_INDICATOR      s_CylonIndicator    = {static_cast<ULONG>(countof(s_pCylonSteps)), s_pCylonSteps};
static  STEP_CONTEXT        s_CylonContext      = {0, &s_CylonIndicator};
static  PROGRESS_INDICATOR  s_Cylon             = {stepReset, stepUpdate, stepString, &s_CylonContext};

// Metronome progress indicator
static  char*               s_pMetronomeSteps[] = {
                                                   "-",
                                                   "\\",
                                                   "|",
                                                   "/",
                                                   "-",
                                                   "/",
                                                   "|",
                                                   "\\",
                                                  };
static  STEP_INDICATOR      s_MetronomeIndicator= {static_cast<ULONG>(countof(s_pMetronomeSteps)), s_pMetronomeSteps};
static  STEP_CONTEXT        s_MetronomeContext  = {0, &s_MetronomeIndicator};
static  PROGRESS_INDICATOR  s_Metronome         = {stepReset, stepUpdate, stepString, &s_MetronomeContext};

// Blink progress indicator
static  char*               s_pBlinkSteps[]     = {
                                                   "",
                                                   ".",
                                                   "*",
                                                   ".",
                                                  };
static  STEP_INDICATOR      s_BlinkIndicator    = {static_cast<ULONG>(countof(s_pBlinkSteps)), s_pBlinkSteps};
static  STEP_CONTEXT        s_BlinkContext      = {0, &s_BlinkIndicator};
static  PROGRESS_INDICATOR  s_Blink             = {stepReset, stepUpdate, stepString, &s_BlinkContext};

// Train progress indicator
static  char*               s_pTrainSteps[]     = {
                                                   "",
                                                   ".",
                                                   "..",
                                                   "...",
                                                   " ...",
                                                   "  ...",
                                                   "   ..",
                                                   "    .",
                                                  };
static  STEP_INDICATOR      s_TrainIndicator    = {static_cast<ULONG>(countof(s_pTrainSteps)), s_pTrainSteps};
static  STEP_CONTEXT        s_TrainContext      = {0, &s_TrainIndicator};
static  PROGRESS_INDICATOR  s_Train             = {stepReset, stepUpdate, stepString, &s_TrainContext};

// Arrow progress indicator
static  char*               s_pArrowSteps[]     = {
                                                   "",
                                                   ">",
                                                   ".>",
                                                   "..>",
                                                   " ..>",
                                                   "  ..>",
                                                   "   ..",
                                                   "    .",
                                                  };
static  STEP_INDICATOR      s_ArrowIndicator    = {static_cast<ULONG>(countof(s_pArrowSteps)), s_pArrowSteps};
static  STEP_CONTEXT        s_ArrowContext      = {0, &s_ArrowIndicator};
static  PROGRESS_INDICATOR  s_Arrow             = {stepReset, stepUpdate, stepString, &s_ArrowContext};

// Worm progress indicator
static  char*               s_pWormSteps[]      = {
                                                   "",
                                                   ".",
                                                   "..",
                                                   "^.",
                                                   "...",
                                                   ".^.",
                                                   " ...",
                                                   " .^.",
                                                   "  ...",
                                                   "  .^.",
                                                   "   ..",
                                                   "   .^",
                                                   "    .",
                                                  };
static  STEP_INDICATOR      s_WormIndicator     = {static_cast<ULONG>(countof(s_pWormSteps)), s_pWormSteps};
static  STEP_CONTEXT        s_WormContext       = {0, &s_WormIndicator};
static  PROGRESS_INDICATOR  s_Worm              = {stepReset, stepUpdate, stepString, &s_WormContext};

// Pong progress indicator
static  char*               s_pPongSteps[]      = {
                                                   "|*  |",
                                                   "| * |",
                                                   "|  *|",
                                                   "| * |",
                                                  };
static  STEP_INDICATOR      s_PongIndicator     = {static_cast<ULONG>(countof(s_pPongSteps)), s_pPongSteps};
static  STEP_CONTEXT        s_PongContext       = {0, &s_PongIndicator};
static  PROGRESS_INDICATOR  s_Pong              = {stepReset, stepUpdate, stepString, &s_PongContext};

// Integer progress indicator
static  const char          s_IntegerFormat[]   = "%3d%%";
static  INTEGER_CONTEXT     s_IntegerContext    = {0, s_IntegerFormat, ""};
static  PROGRESS_INDICATOR  s_Integer           = {integerReset, integerUpdate, integerString, &s_IntegerContext};

// Float progress indicator
static  const char          s_FloatFormat[]     = "%6.2f%%";
static  FLOAT_CONTEXT       s_FloatContext      = {0.0, s_FloatFormat, ""};
static  PROGRESS_INDICATOR  s_Float             = {floatReset, floatUpdate, floatString, &s_FloatContext};

// Bar progress indicator
static  BAR_CONTEXT         s_BarContext        = {0.0, DEFAULT_BAR_LENGTH, ""};
static  PROGRESS_INDICATOR  s_Bar               = {barReset, barUpdate, barString, &s_BarContext};

// Progress indicators table
static  PPROGRESS_INDICATOR s_ProgressTable[] = {&s_Spinner,
                                                 &s_Cylon,
                                                 &s_Metronome,
                                                 &s_Blink,
                                                 &s_Train,
                                                 &s_Arrow,
                                                 &s_Worm,
                                                 &s_Pong,
                                                 &s_Integer,
                                                 &s_Float,
                                                 &s_Bar,
                                                };

// Progress indicator variables
static  char                s_sClearString[MAX_COMMAND_STRING];
static  bool                s_bProgressState     = true;                // Current progress indicator state (Enabled/Disabled)
static  bool                s_bProgressIndicator = false;               // Current progress indicator state (On/Off)
static  bool                s_bProgressStatus    = false;               // Current progress status (True = in progress)

static  const char*         s_pProgressColor  = NULL;                   // Current progress color (DML)
static  ULONG               s_ulProgressTime  = DEFAULT_PROGRESS_TIME;  // Current progress update time (Ticks/ms)
static  ULONG               s_ulProgressStyle = DEFAULT_PROGRESS_STYLE; // Current progress indicator style

static  float               s_fProgressPercentage = 0.0;                // Lwrrrent progress percentage
static  const void*         s_pProgressInfo       = NULL;               // Current progress information

static  DWORD               s_dwTickCount = GetTickCount();             // Last progress tick count

static  PPROGRESS_INDICATOR s_pIndicator = s_ProgressTable[s_ulProgressStyle];

//******************************************************************************

CProgressState::CProgressState()
:   m_bProgressIndicator(false),
    m_pProgressColor(NULL),
    m_ulProgressTime(0),
    m_ulProgressStyle(0),
    m_fProgressPercentage(0.0),
    m_pProgressInfo(NULL)
{
    // Update the progress state (Initial)
    update();

} // CProgressState

//******************************************************************************

CProgressState::~CProgressState()
{
    // Restore the progress state
    restore();

} // ~CProgressState

//******************************************************************************

void
CProgressState::update()
{
    // Update the progress state
    m_bProgressIndicator  = progressIndicator();
    m_pProgressColor      = progressColor();
    m_ulProgressTime      = progressTime();
    m_ulProgressStyle     = progressStyle();
    m_fProgressPercentage = progressPercentage();
    m_pProgressInfo       = progressInfo();

} // update

//******************************************************************************

void
CProgressState::restore()
{
    // Restore the progress state (if necessary)
    if (progressInfo() != m_pProgressInfo)
    {
        progressInfo(m_pProgressInfo);
    }
    if (progressPercentage() != m_fProgressPercentage)
    {
        progressPercentage(m_fProgressPercentage);
    }
    if (progressIndicator() != m_bProgressIndicator)
    {
        progressIndicator(m_bProgressIndicator);
    }
    if (progressTime() != m_ulProgressTime)
    {
        progressTime(m_ulProgressTime);
    }
    if (progressColor() != m_pProgressColor)
    {
        progressColor(m_pProgressColor);
    }
    if (progressStyle() != m_ulProgressStyle)
    {
        progressStyle(m_ulProgressStyle, m_pProgressInfo);
    }

} // restore

//******************************************************************************

CProgressStatus::CProgressStatus()
{
    assert(s_bProgressStatus == false);

    // Indicate progress update in progress
    s_bProgressStatus = true;

} // CProgressStatus

//******************************************************************************

CProgressStatus::~CProgressStatus()
{
    assert(s_bProgressStatus == true);

    // Indicate progress update no longer in progress
    s_bProgressStatus = false;

} // ~CProgressStatus

//******************************************************************************

bool
progressState()
{
    // Return the progress indicator state (Enabled/Disabled)
    return s_bProgressState;

} // progressState

//******************************************************************************

bool
progressState
(
    bool                bProgressState
)
{
    bool                bLastProgressState = s_bProgressState;

    // Update the progress indicator state
    s_bProgressState = bProgressState;

    // Return the last progress indicator state (Enabled/Disabled)
    return bLastProgressState;

} // progressState

//******************************************************************************

bool
progressIndicator()
{
    // Return the current progress indicator value (On/Off)
    return s_bProgressIndicator;

} // progressIndicator

//******************************************************************************

bool
progressIndicator
(
    bool                bProgressIndicator,
    bool                bProgressUpdate
)
{
    CProgressStatus     progressStatus;
    const char         *pProgressString;
    bool                bLastProgressIndicator = s_bProgressIndicator;

    // Check for turning progress indicator On vs. Off
    if (bProgressIndicator == INDICATOR_ON)
    {
        // Request to turn on progress indicator (Only if lwrrently off)
        if (s_bProgressIndicator == INDICATOR_OFF)
        {
            // Get the progress string for the progress indicator
            pProgressString = s_pIndicator->pfnProgressString(s_pIndicator->pProgressContext);

            // Check for DML output enabled
            if (dmlState())
            {
                ControlledOutput(DEBUG_OUTCTL_ALL_CLIENTS | DEBUG_OUTCTL_DML | DEBUG_OUTCTL_NOT_LOGGED, DEBUG_OUTPUT_NORMAL, "%s", pProgressString);
            }
            else
            {
                ControlledOutput(DEBUG_OUTCTL_ALL_CLIENTS | DEBUG_OUTCTL_NOT_LOGGED, DEBUG_OUTPUT_NORMAL, "%s", pProgressString);
            }
            // Update the progress clear string
            updateClear(pProgressString);

            // Update the progress indicator value (On)
            s_bProgressIndicator = INDICATOR_ON;
        }
        // Update the progress tick count
        s_dwTickCount = GetTickCount();
    }
    else    // Turning progress indicator off
    {
        // Request to turn off progress indicator (Only if lwrrently on)
        if (s_bProgressIndicator == INDICATOR_ON)
        {
            // Check for DML output enabled
            if (dmlState())
            {
                ControlledOutput(DEBUG_OUTCTL_ALL_CLIENTS | DEBUG_OUTCTL_DML | DEBUG_OUTCTL_NOT_LOGGED, DEBUG_OUTPUT_NORMAL, "%s", s_sClearString);
            }
            else
            {
                ControlledOutput(DEBUG_OUTCTL_ALL_CLIENTS | DEBUG_OUTCTL_NOT_LOGGED, DEBUG_OUTPUT_NORMAL, "%s", s_sClearString);
            }
            // Update the progress clear string
            updateClear();

            // Check for progress update request (Update progress for on/off)
            if (bProgressUpdate)
            {
                // Update the progress indicator
                s_pIndicator->pfnProgressUpdate(s_pIndicator->pProgressContext);
            }
            // Update the progress indicator value (Off)
            s_bProgressIndicator = INDICATOR_OFF;
        }
    }
    // Return the last progress indicator value (On/Off)
    return bLastProgressIndicator;

} // progressIndicator

//******************************************************************************

ULONG
progressStyle()
{
    // Return the current progress style
    return s_ulProgressStyle;

} // progressStyle

//******************************************************************************

ULONG
progressStyle
(
    ULONG               ulProgressStyle,
    const void         *pProgressInfo
)
{
    bool                bProgressIndicator;
    ULONG               ulLastProgressStyle = s_ulProgressStyle;

    // Check for a valid progress style
    if (ulProgressStyle < countof(s_ProgressTable))
    {
        // Check for progress style change
        if (s_ulProgressStyle != ulProgressStyle)
        {
            // Make sure progress indicator is off when changing styles
            bProgressIndicator = progressIndicator(INDICATOR_OFF, false);

            // Set the new progress indicator style (and style information)
            s_ulProgressStyle = ulProgressStyle;
            s_pProgressInfo   = pProgressInfo;

            // Set the new progress indicator
            s_pIndicator = s_ProgressTable[s_ulProgressStyle];

            // Reset the new progress indicator (w/user information)
            s_pIndicator->pfnProgressReset(s_pIndicator->pProgressContext, s_pProgressInfo);

            // Restore the original progress indicator state
            progressIndicator(bProgressIndicator, false);
        }
        else    // Same progress style
        {
            // Reset the new progress indicator (w/user information)
            s_pIndicator->pfnProgressReset(s_pIndicator->pProgressContext, s_pProgressInfo);
        }
    }
    else    // Invalid progress style
    {
        throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid progress indicator style (%d >= %d)",
                         ulProgressStyle, countof(s_ProgressTable));
    }
    // Return the last progress style
    return ulLastProgressStyle;

} // progressStyle

//******************************************************************************

ULONG
progressStyles()
{
    // Return the number of progress styles
    return static_cast<ULONG>(countof(s_ProgressTable));

} // progressStyles

//******************************************************************************

const char*
progressColor()
{
    // Return the progress color (May be NULL)
    return s_pProgressColor;

} // progressColor

//******************************************************************************

const char*
progressColor
(
    const char         *pProgressColor
)
{
    bool                bProgressIndicator;
    const char         *pLastProgressColor = s_pProgressColor;

    // Check for possible progress color change
    if (s_pProgressColor != pProgressColor)
    {
        // Make sure progress indicator is off when changing colors
        bProgressIndicator = progressIndicator(INDICATOR_OFF, false);

        // Set the new progress color value (May be NULL)
        s_pProgressColor = pProgressColor;

        // Restore the original progress indicator state
        progressIndicator(bProgressIndicator, false);
    }
    // Return the last progress color
    return pLastProgressColor;

} // progressColor

//******************************************************************************

ULONG
progressTime()
{
    // Return the progress update time (Ticks)
    return s_ulProgressTime;

} // progressTime

//******************************************************************************

ULONG
progressTime
(
    ULONG               ulProgressTime
)
{
    ULONG               ulLastProgressTime = s_ulProgressTime;

    // Update the progress update time (Ticks)
    s_ulProgressTime = ulProgressTime;

    // Return the last progress update time
    return ulLastProgressTime;

} // progressTime

//******************************************************************************

float
progressPercentage()
{
    // Return the progress percentage
    return s_fProgressPercentage;

} // progressPercentage

//******************************************************************************

float
progressPercentage
(
    float               fProgressPercentage
)
{
    float               fLastProgressPercentage = s_fProgressPercentage;

    // Update the progress percentage
    s_fProgressPercentage = fProgressPercentage;

    // Return the last progress percentage
    return fLastProgressPercentage;

} // progressPercentage

//******************************************************************************

const void*
progressInfo()
{
    // Return the progress information (May be NULL)
    return s_pProgressInfo;

} // progressInfo

//******************************************************************************

const void*
progressInfo
(
    const void         *pProgressInfo
)
{
    const void         *pLastProgressInfo = s_pProgressInfo;

    // Update the progress information (May be NULL)
    s_pProgressInfo = pProgressInfo;

    // Return the last progress information
    return pLastProgressInfo;

} // progressInfo

//******************************************************************************

bool
progressUpdate
(
    bool                bUpdate
)
{
    CProgressStatus     progressStatus;
    const char         *pProgressString;
    DWORD               dwTickCount;

    // Check for progress indicator on (Only need update if on)
    if (s_bProgressIndicator == INDICATOR_ON)
    {
        // Get the current tick count
        dwTickCount = GetTickCount();

        // Check for no forced update
        if (!bUpdate)
        {
            // Check for time to update
            if (s_dwTickCount < dwTickCount)
            {
                if ((dwTickCount - s_dwTickCount) >= s_ulProgressTime)
                {
                    // Indicate time for an update
                    bUpdate = true;
                }
            }
            else    // Timer wrap
            {
                if ((s_dwTickCount - dwTickCount) >= s_ulProgressTime)
                {
                    // Indicate time for an update
                    bUpdate = true;
                }
            }
        }
        // Check for update needed
        if (bUpdate)
        {
            // Check for DML output enabled
            if (dmlState())
            {
                ControlledOutput(DEBUG_OUTCTL_ALL_CLIENTS | DEBUG_OUTCTL_DML | DEBUG_OUTCTL_NOT_LOGGED, DEBUG_OUTPUT_NORMAL, "%s", s_sClearString);
            }
            else
            {
                ControlledOutput(DEBUG_OUTCTL_ALL_CLIENTS | DEBUG_OUTCTL_NOT_LOGGED, DEBUG_OUTPUT_NORMAL, "%s", s_sClearString);
            }
            // Update the progress clear string
            updateClear();

            // Update the current progress indicator
            s_pIndicator->pfnProgressUpdate(s_pIndicator->pProgressContext);

            // Get updated progress string for the progress indicator
            pProgressString = s_pIndicator->pfnProgressString(s_pIndicator->pProgressContext);

            // Check for DML output enabled
            if (dmlState())
            {
                ControlledOutput(DEBUG_OUTCTL_ALL_CLIENTS | DEBUG_OUTCTL_DML | DEBUG_OUTCTL_NOT_LOGGED, DEBUG_OUTPUT_NORMAL, "%s", pProgressString);
            }
            else
            {
                ControlledOutput(DEBUG_OUTCTL_ALL_CLIENTS | DEBUG_OUTCTL_NOT_LOGGED, DEBUG_OUTPUT_NORMAL, "%s", pProgressString);
            }
            // Update the progress clear string
            updateClear(pProgressString);

            // Update the progress tick count
            s_dwTickCount = dwTickCount;
        }
    }
    return bUpdate;

} // progressUpdate

//******************************************************************************

bool
progressStatus()
{
    // Return the current progress status (True = in process)
    return s_bProgressStatus;

} // progressStatus

//******************************************************************************

bool
progressCheck()
{
    bool                bUpdate = false;

    // Check for progress update already in process
    if (!progressStatus())
    {
        // No update already in progress, perform progress update
        bUpdate = progressUpdate();
    }
    return bUpdate;

} // progressCheck

//******************************************************************************

const char*
progressIndicatorString()
{
    // Return the current progress indicator step string
    return s_pIndicator->pfnProgressString(s_pIndicator->pProgressContext);

} // progressIndicatorString

//******************************************************************************

const char*
progressClearString()
{
    // Return the current progress indicator clear string
    return s_sClearString;

} // progressClearString

//******************************************************************************

void
progressReset()
{
    // Reset the progress indicator state
    s_bProgressIndicator = false;
    s_bProgressStatus    = false;

    // Reset default progress indicator values
    s_pProgressColor      = NULL;
    s_ulProgressTime      = DEFAULT_PROGRESS_TIME;
    s_ulProgressStyle     = DEFAULT_PROGRESS_STYLE;

    // Reset internal progress state
    s_fProgressPercentage = 0.0;
    s_pProgressInfo       = NULL;
    s_dwTickCount         = GetTickCount();

    // Setup the default progress indicator
    s_pIndicator = s_ProgressTable[s_ulProgressStyle];

    // Reset default progress indicator
    s_pIndicator->pfnProgressReset(s_pIndicator->pProgressContext, s_pProgressInfo);

} // progressReset

//******************************************************************************

static void
stepReset
(
    void               *pContext,
    const void         *pInfo
)
{
    UNREFERENCED_PARAMETER(pInfo);

    STEP_CONTEXT       *pStepContext = reinterpret_cast<PSTEP_CONTEXT>(pContext);

    // Reset the current step value
    pStepContext->ulLwrrentStep = 0;

    // Clear the step string
    pStepContext->sString[0] = EOS;

} // stepReset

//******************************************************************************

static void
stepUpdate
(
    void               *pContext
)
{
    STEP_CONTEXT       *pStepContext = reinterpret_cast<PSTEP_CONTEXT>(pContext);
    const char         *pStepString;

    assert(pContext != NULL);

    // Increment the current step value (To get to next step)
    pStepContext->ulLwrrentStep = (pStepContext->ulLwrrentStep + 1) % pStepContext->pStepIndicator->ulStepCount;

    // Get the new step progress string
    pStepString = pStepContext->pStepIndicator->pProgressSteps[pStepContext->ulLwrrentStep];

    // Check for using DML (Color progress indicator)
    if (s_pProgressColor != NULL)
    {
        // Update the step progress string
        sprintf(pStepContext->sString, "%s", DML(foreground(pStepString, s_pProgressColor)));
    }
    else    // No progress indicator color
    {
        // Update the step progress string
        sprintf(pStepContext->sString, "%s", pStepString);
    }

} // stepUpdate

//******************************************************************************

static const char*
stepString
(
    void               *pContext
)
{
    STEP_CONTEXT       *pStepContext = reinterpret_cast<PSTEP_CONTEXT>(pContext);

    assert(pContext != NULL);

    // Return step progress string
    return pStepContext->sString;

} // stepString

//******************************************************************************

static void
integerReset
(
    void               *pContext,
    const void         *pInfo
)
{
    INTEGER_CONTEXT    *pIntegerContext = reinterpret_cast<PINTEGER_CONTEXT>(pContext);
    const char         *pFormat = reinterpret_cast<const char*>(pInfo);

    assert(pContext != NULL);

    // Check for integer format given (Information parameter)
    if (pFormat != NULL)
    {
        // Setup the requested integer format
        pIntegerContext->pFormat = pFormat;
    }
    else    // No integer format (Use default)
    {
        pIntegerContext->pFormat = s_IntegerFormat;
    }
    // Reset the percentage value
    pIntegerContext->ulPercentage = static_cast<ULONG>(s_fProgressPercentage + 0.5);

    // Clear the integer string
    pIntegerContext->sString[0] = EOS;

} // integerReset

//******************************************************************************

static void
integerUpdate
(
    void               *pContext
)
{
    INTEGER_CONTEXT    *pIntegerContext = reinterpret_cast<PINTEGER_CONTEXT>(pContext);
    CString             sString(MAX_COMMAND_STRING);

    assert(pContext != NULL);

    // Update the integer percentage value
    pIntegerContext->ulPercentage = static_cast<ULONG>(s_fProgressPercentage + 0.5);

    // Check for using DML (Color progress indicator)
    if (s_pProgressColor != NULL)
    {
        // Update the integer progress string
        sString.sprintf(pIntegerContext->pFormat, pIntegerContext->ulPercentage);
        sprintf(pIntegerContext->sString, "%s", DML(foreground(sString, s_pProgressColor)));
    }
    else    // No progress indicator color
    {
        // Update the integer progress string
        sprintf(pIntegerContext->sString, pIntegerContext->pFormat, pIntegerContext->ulPercentage);
    }

} // integerUpdate

//******************************************************************************

static const char*
integerString
(
    void               *pContext
)
{
    INTEGER_CONTEXT    *pIntegerContext = reinterpret_cast<PINTEGER_CONTEXT>(pContext);

    assert(pContext != NULL);

    // Return integer progress string
    return pIntegerContext->sString;

} // integerString

//******************************************************************************

static void
floatReset
(
    void               *pContext,
    const void         *pInfo
)
{
    FLOAT_CONTEXT      *pFloatContext = reinterpret_cast<PFLOAT_CONTEXT>(pContext);
    const char         *pFormat = reinterpret_cast<const char*>(pInfo);

    assert(pContext != NULL);

    // Check for float format given (Information parameter)
    if (pFormat != NULL)
    {
        // Setup the requested float format
        pFloatContext->pFormat = pFormat;
    }
    else    // No float format (Use default)
    {
        pFloatContext->pFormat = s_FloatFormat;
    }
    // Reset the percentage value
    pFloatContext->fPercentage = s_fProgressPercentage;

    // Clear the float string
    pFloatContext->sString[0] = EOS;

} // floatReset

//******************************************************************************

static void
floatUpdate
(
    void               *pContext
)
{
    FLOAT_CONTEXT      *pFloatContext = reinterpret_cast<PFLOAT_CONTEXT>(pContext);
    CString             sString(MAX_COMMAND_STRING);

    assert(pContext != NULL);

    // Update the float percentage value
    pFloatContext->fPercentage = s_fProgressPercentage;

    // Check for using DML (Color progress indicator)
    if (s_pProgressColor != NULL)
    {
        // Update the float progress string
        sString.sprintf(pFloatContext->pFormat, pFloatContext->fPercentage);
        sprintf(pFloatContext->sString, "%s", DML(foreground(sString, s_pProgressColor)));
    }
    else    // No progress indicator color
    {
        // Update the float progress string
        sprintf(pFloatContext->sString, pFloatContext->pFormat, pFloatContext->fPercentage);
    }

} // floatUpdate

//******************************************************************************

static const char*
floatString
(
    void               *pContext
)
{
    FLOAT_CONTEXT      *pFloatContext = reinterpret_cast<PFLOAT_CONTEXT>(pContext);

    assert(pContext != NULL);

    // Return float progress string
    return pFloatContext->sString;

} // floatString

//******************************************************************************

static void
barReset
(
    void               *pContext,
    const void         *pProgressInfo
)
{
    BAR_CONTEXT        *pBarContext = reinterpret_cast<PBAR_CONTEXT>(pContext);
    ULONG               ulLength = static_cast<ULONG>(reinterpret_cast<ULONG_PTR>(pProgressInfo));

    assert(pContext != NULL);

    // Check for bar length given (Information parameter)
    if (ulLength != 0)
    {
        // Setup the requested bar length
        pBarContext->ulLength = ulLength;
    }
    else    // No bar length (Use default)
    {
        pBarContext->ulLength = DEFAULT_BAR_LENGTH;
    }
    // Reset the percentage value
    pBarContext->fPercentage = 0.0;

    // Clear the bar string
    pBarContext->sString[0] = EOS;

} // barReset

//******************************************************************************

static void
barUpdate
(
    void               *pContext
)
{
    BAR_CONTEXT        *pBarContext = reinterpret_cast<PBAR_CONTEXT>(pContext);
    CString             sString(MAX_COMMAND_STRING);
    CString             sChar(MAX_COMMAND_STRING);
    ULONG               ulLength;

    assert(pContext != NULL);

    // Update the bar percentage value
    pBarContext->fPercentage = s_fProgressPercentage;

    // Compute bar length based on percentage value
    ulLength = static_cast<ULONG>((static_cast<float>(pBarContext->ulLength) * (s_fProgressPercentage / 100.0f)) + 0.5);

    // Build bar string using bar length
    sString.append("|");

    sChar.erase();
    sChar.fill('*', ulLength);
    sString.append(sChar);

    sChar.erase();
    sChar.fill(' ', pBarContext->ulLength - ulLength);
    sString.append(sChar);

    sString.append("|");

    // Check for using DML (Color progress indicator)
    if (s_pProgressColor != NULL)
    {
        // Update the bar progress string
        sprintf(pBarContext->sString, "%s", DML(foreground(sString, s_pProgressColor)));
    }
    else    // No progress indicator color
    {
        // Update the bar progress string
        sprintf(pBarContext->sString, "%s", STR(sString));
    }

} // barUpdate

//******************************************************************************

static const char*
barString
(
    void               *pContext
)
{
    BAR_CONTEXT        *pBarContext = reinterpret_cast<PBAR_CONTEXT>(pContext);

    assert(pContext != NULL);

    // Return bar progress string
    return pBarContext->sString;

} // barString

//******************************************************************************

static void
updateClear
(
    const char         *pString
)
{
    size_t              length = 0;

    // Check for progress string
    if (pString)
    {
        // Get the length of the progress string
        length = dmllen(pString);

        // Make sure the new progress string is not too long
        assert(length < countof(s_sClearString));

        // Update the progress clear string
        memset(s_sClearString, BACKSPACE, length);
    }
    // Terminate the progress clear string
    s_sClearString[length] = EOS;

} // updateClear

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
