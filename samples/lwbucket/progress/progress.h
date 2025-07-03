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
|*  Module: progress.h                                                        *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _PROGRESS_H
#define _PROGRESS_H

//******************************************************************************
//
// Constants
//
//******************************************************************************
// Progress indicator values
#define INDICATOR_OFF           false           // Progress indicator off
#define INDICATOR_ON            true            // Progress indicator on

// Progress style values
#define SPINNER_STYLE           0               // Spinner style progress indicator
#define CYLON_STYLE             1               // Cylon style progress indicator
#define METRONOME_STYLE         2               // Metronome style progress indicator
#define BLINK_STYLE             3               // Blink style progress indicator
#define TRAIN_STYLE             4               // Train style progress indicator
#define ARROW_STYLE             5               // Arrow style progress indicator
#define WORM_STYLE              6               // Worm style progress indicator
#define PONG_STYLE              7               // Pong style progress indicator
#define INTEGER_STYLE           8               // Integer style progress indicator
#define FLOAT_STYLE             9               // Float style progress indicator
#define BAR_STYLE               10              // Bar style progress indicator

// Progress defaults
#define DEFAULT_PROGRESS_STYLE  SPINNER_STYLE   // Default progress style
#define DEFAULT_PROGRESS_TIME   100             // Default progress time (100 ticks/ms)

// Bar progress indicator defaults
#define DEFAULT_BAR_LENGTH      25              // Default bar progress indicator length

// Progress status values
#define IN_PROGRESS             true            // Update in progress
#define NOT_IN_PROGRESS         false           // Update not in progress

//******************************************************************************
//
//  Type Definitions
//
//******************************************************************************
typedef void        (*PFN_PROGRESS_RESET)(void* pContext, const void* pInfo);
typedef void        (*PFN_PROGRESS_UPDATE)(void* pContext);
typedef const char* (*PFN_PROGRESS_STRING)(void* pContext);

//******************************************************************************
//
//  Structures
//
//******************************************************************************
typedef struct _STEP_INDICATOR
{
    ULONG               ulStepCount;
    char              **pProgressSteps;

} STEP_INDICATOR, *PSTEP_INDICATOR;

typedef struct _STEP_CONTEXT
{
    ULONG               ulLwrrentStep;
    STEP_INDICATOR     *pStepIndicator;
    char                sString[MAX_COMMAND_STRING];

} STEP_CONTEXT, *PSTEP_CONTEXT;

typedef struct _INTEGER_CONTEXT
{
    ULONG               ulPercentage;
    const char*         pFormat;
    char                sString[MAX_COMMAND_STRING];

} INTEGER_CONTEXT, *PINTEGER_CONTEXT;

typedef struct _FLOAT_CONTEXT
{
    float               fPercentage;
    const char*         pFormat;
    char                sString[MAX_COMMAND_STRING];

} FLOAT_CONTEXT, *PFLOAT_CONTEXT;

typedef struct _BAR_CONTEXT
{
    float               fPercentage;
    ULONG               ulLength;
    char                sString[MAX_COMMAND_STRING];

} BAR_CONTEXT, *PBAR_CONTEXT;

typedef struct _PROGRESS_INDICATOR
{
    PFN_PROGRESS_RESET  pfnProgressReset;
    PFN_PROGRESS_UPDATE pfnProgressUpdate;
    PFN_PROGRESS_STRING pfnProgressString;
    void               *pProgressContext;

} PROGRESS_INDICATOR, *PPROGRESS_INDICATOR;

//******************************************************************************
//
// Class CProgressState
//
//******************************************************************************
class CProgressState
{
private:
        bool            m_bProgressIndicator;   // Progress indicator state
const   char*           m_pProgressColor;       // Progress color
        ULONG           m_ulProgressTime;       // Progress update time
        ULONG           m_ulProgressStyle;      // Progress style
        float           m_fProgressPercentage;  // Progress percentage
const   void*           m_pProgressInfo;        // Progress information

public:
                        CProgressState();
                       ~CProgressState();

        void            update();
        void            restore();

}; // CProgressState

//******************************************************************************
//
// Class CProgressStatus
//
//******************************************************************************
class CProgressStatus
{
public:
                        CProgressStatus();
                       ~CProgressStatus();

}; // CProgressStatus

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  bool            progressState();
extern  bool            progressState(bool bProgressState);
extern  bool            progressIndicator();
extern  bool            progressIndicator(bool bProgressIndicator, bool bProgressUpdate = true);
extern  ULONG           progressStyle();
extern  ULONG           progressStyle(ULONG ulProgressStyle, const void* pInfo = NULL);
extern  ULONG           progressStyles();
extern  const char*     progressColor();
extern  const char*     progressColor(const char* pProgressColor);
extern  ULONG           progressTime();
extern  ULONG           progressTime(ULONG ulProgressTime);
extern  float           progressPercentage();
extern  float           progressPercentage(float fProgressPercentage);
extern  bool            progressUpdate(bool bUpdate = false);
extern  bool            progressStatus();
extern  bool            progressCheck();

extern  const char*     progressIndicatorString();
extern  const char*     progressClearString();

extern  void            progressReset();

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _PROGRESS_H
