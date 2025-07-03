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
|*  Module: dml.h                                                             *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _DML_H
#define _DML_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
// Define the DML special characters
#define DML_AMPERSAND               '&'             // DML ampersand special character
#define DML_LESS_THAN               '<'             // DML less than special character
#define DML_GREATER_THAN            '>'             // DML greater than special character
#define DML_QUOTE                   '"'             // DML quote special character

// Define the DML escape strings
#define DML_AMPERSAND_ESCAPE        "&amp;"         // DML ampersand escape string
#define DML_LESS_THAN_ESCAPE        "&lt;"          // DML less than escape string
#define DML_GREATER_THAN_ESCAPE     "&gt;"          // DML greater than escape string
#define DML_QUOTE_ESCAPE            "&quot;"        // DML quote escape string

// Define the DML color values (Based on text types)
#define USER_COMMAND_BACK           "uwbg"          // White
#define USER_COMMAND_FORE           "uwfg"          // Black
#define INTERNAL_PROTOCOL_BACK      "ikdbg"         // White
#define INTERNAL_PROTOCOL_FORE      "ikdfg"         // Black
#define INTERNAL_REMOTING_BACK      "irembg"        // White
#define INTERNAL_REMOTING_FORE      "iremfg"        // Black
#define INTERNAL_BREAKPOINT_BACK    "ibpbg"         // White
#define INTERNAL_BREAKPOINT_FORE    "ibpfg"         // Black
#define INTERNAL_EVENT_BACK         "ievtbg"        // White
#define INTERNAL_EVENT_FORE         "ievtfg"        // Black
#define SYMBOL_MESSAGE_BACK         "symbg"         // White
#define SYMBOL_MESSAGE_FORE         "symfg"         // Black
#define DEBUGEE_PROMPT_BACK         "dbgpbg"        // White
#define DEBUGEE_PROMPT_FORE         "dbgpfg"        // Black
#define DEBUGEE_LEVEL_BACK          "dbgbg"         // White
#define DEBUGEE_LEVEL_FORE          "dbgfg"         // Black
#define EXTENSION_WARNING_BACK      "extbg"         // White
#define EXTENSION_WARNING_FORE      "extfg"         // Black
#define PROMPT_REGISTERS_BACK       "promptregbg"   // White
#define PROMPT_REGISTERS_FORE       "promptregfg"   // Black
#define PROMPT_LEVEL_BACK           "promptbg"      // White
#define PROMPT_LEVEL_FORE           "promptfg"      // Black
#define VERBOSE_LEVEL_BACK          "verbbg"        // White
#define VERBOSE_LEVEL_FORE          "verbfg"        // Black
#define WARNING_LEVEL_BACK          "warnbg"        // White
#define WARNING_LEVEL_FORE          "warnfg"        // Black
#define ERROR_LEVEL_BACK            "errbg"         // White
#define ERROR_LEVEL_FORE            "errfg"         // Black
#define NORMAL_LEVEL_BACK           "normbg"        // White
#define NORMAL_LEVEL_FORE           "normfg"        // Black
#define SUBDUED_TEXT_BACK           "subbg"         // Dark gray
#define SUBDUED_TEXT_FORE           "subfg"         // Light gray
#define EMPHASIZED_TEXT_BACK        "emphbg"        // White
#define EMPHASIZED_TEXT_FORE        "emphfg"        // Light blue
#define SOURCE_ANNOTATION           "srcannot"      // Blue
#define SOURCE_SPECIAL_IDENTIFIER   "srcspid"       // Light blue
#define SOURCE_DIRECTIVE            "srcdrct"       // Light blue
#define SOURCE_COMMENT              "srccmnt"       // Green
#define SOURCE_MATCHING_PAIR        "srcpair"       // Black
#define SOURCE_KEYWORD              "srckw"         // Light blue
#define SOURCE_IDENTIFIER           "srcid"         // Black
#define SOURCE_STRING_CONSTANT      "srcstr"        // Red
#define SOURCE_CHARACTER_CONSTANT   "srcchar"       // Red
#define SOURCE_NUMERIC_CONSTANT     "srcnum"        // Black
#define USER_SELECTED_BACK          "uslbg"         // White
#define USER_SELECTED_FORE          "uslfg"         // Black
#define SECONDARY_LINE_BACK         "slbg"          // Teal
#define SECONDARY_LINE_FORE         "slfg"          // White
#define DISABLED_WINDOW             "dwbg"          // Gray
#define CHANGED_DATA                "changed"       // Light red
#define DISABLED_BREAKPOINT_BACK    "dbpbg"         // Yellow
#define DISABLED_BREAKPOINT_FORE    "dbpfg"         // White
#define ENABLED_BREAKPOINT_BACK     "ebpbg"         // Light Red
#define ENABLED_BREAKPOINT_FORE     "ebpfg"         // White
#define BREAKPOINT_LWRRENT_BACK     "bpbg"          // Purple
#define BREAKPOINT_LWRRENT_FORE     "bpfg"          // White
#define LWRRENT_LINE_BACK           "clbg"          // Medium blue
#define LWRRENT_LINE_FORE           "clfg"          // White
#define BACKGROUND                  "wbg"           // White
#define FOREGROUND                  "wfg"           // Black

// Define the foreground color values (Backgrounds not lwrrently used)
#define RED                         "srcstr"        // Source string (Red)
#define GREEN                       "srccmnt"       // Source comment (Green)
#define BLUE                        "srcannot"      // Source annotation (Blue)
#define BLACK                       "srcid"         // Source identifier (Black)
#define GRAY                        "subfg"         // Subdued text (Gray)
#define WHITE                       "slfg"          // Secondary line (White)

// Define the maximum DML stack level
#define MAX_DML_LEVEL           8               // Maximum DML stacking level

//******************************************************************************
//
//  Macros
//
//******************************************************************************
#define DML(string)             ((string).str())
#define VARDML(width, string)   ((width) + dmlDelta((string))), DML(string)

//******************************************************************************
//
// Class CDmlState
//
//******************************************************************************
class CDmlState
{
private:
        bool            m_bDmlState;            // DML state

public:
                        CDmlState();
                       ~CDmlState();

        void            update();
        void            restore();

}; // CDmlState

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  CString         startColor(const CString& sForeground, const CString& sBackground);
extern  CString         endColor();
extern  CString         color(const CString& sString, const CString& sForeground, const CString& sBackground);

extern  CString         startForeground(const CString& sForeground);
extern  CString         endForeground();
extern  CString         foreground(const CString& sString, const CString& sForeground);

extern  CString         startBackground(const CString& sBackground);
extern  CString         endBackground();
extern  CString         background(const CString& sString, const CString& sBackground);

extern  CString         startBold();
extern  CString         endBold();
extern  CString         bold(const CString& sString);

extern  CString         startItalic();
extern  CString         endItalic();
extern  CString         italic(const CString& sString);

extern  CString         startUnderline();
extern  CString         endUnderline();
extern  CString         underline(const CString& sString);

extern  CString         startExec(const CString& sCommand);
extern  CString         endExec();
extern  CString         exec(const CString& sString, const CString& sCommand);

extern  CString         startLink(const CString& sName, const char *pSection = NULL);
extern  CString         endLink();
extern  CString         link(const CString& sString, const CString& sName, const char *pSection = NULL);

extern  CString         startSection(const CString& sSection, const char *pName = NULL);
extern  CString         endSection();
extern  CString         section(const CString& sString, const CString& sSection, const char *pName = NULL);

extern  CString         startLinkCmd(const CString& sCommand, const CString& sName, const char *pSection = NULL);
extern  CString         endLinkCmd();
extern  CString         linkCmd(const CString& sString, const CString& sCommand, const CString& sName, const char *pSection = NULL);

extern  size_t          dmllen(const CString& sString);
extern  size_t          dmlDelta(const CString& sString);

extern  CString         dmlStrip(const CString& sString);
extern  CString         dmlEscape(const CString& sString);

extern  void            dmlReset();

extern  bool            dmlState();
extern  bool            dmlState(bool bDmlState);

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _DML_H
