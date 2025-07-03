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
|*  Module: dml.cpp                                                           *|
|*                                                                            *|
 \****************************************************************************/
#include "../include/dml.h"

//******************************************************************************
//
// Forwards
//
//******************************************************************************
static  void            pushForeground(const CString& sForeground);
static  void            popForeground();
static  void            pushBackground(const CString& sBackground);
static  void            popBackground();

//******************************************************************************
//
// Locals
//
//******************************************************************************
static  bool            s_bBold = false;                // Current bold state
static  bool            s_bItalic = false;              // Current italic state
static  bool            s_bUnderline = false;           // Current underline state

static  CString         s_sForeground;                  // Current foreground color
static  CString         s_aForeground[MAX_DML_LEVEL];   // Foreground color stack
static  ULONG           s_ulForeground = 0;             // Current foreground level
static  CString         s_sBackground;                  // Current background color
static  CString         s_aBackground[MAX_DML_LEVEL];   // Background color stack
static  ULONG           s_ulBackground = 0;             // Current background level

static  bool            s_bDmlState = true;             // Current DML state

//******************************************************************************

CString
startColor
(
    const CString&      sForeground,
    const CString&      sBackground
)
{
    CString             sColor;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Push current foreground/background colors onto the stack
        pushForeground(sForeground);
        pushBackground(sBackground);

        // Build and return the DML start color string
        sColor.sprintf("<col fg=\"%s\" bg=\"%s\">", DML(sForeground), DML(sBackground));
    }
    return sColor;

} // startColor

//******************************************************************************

CString
endColor()
{
    CString             sColor;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Pop foreground/background colors from the stack
        popForeground();
        popBackground();

        // Check for current foreground color
        if (!s_sForeground.empty())
        {
            // Check for current background color
            if (!s_sBackground.empty())
            {
                // Set end color string (w/foreground & background reset)
                sColor.sprintf("</col><col fg=\"%s\"><col bg=\"%s\">", DML(s_sForeground), DML(s_sBackground));
            }
            else    // No current background color
            {
                // Set end color string (w/foreground reset)
                sColor.sprintf("<col fg=\"%s\">", DML(s_sForeground));
            }
        }
        else    // No current foreground color
        {
            // Check for current background color
            if (!s_sBackground.empty())
            {
                // Set end color string (w/background reset)
                sColor.sprintf("</col><col bg=\"%s\">", DML(s_sBackground));
            }
            else    // No current background color
            {
                // Set end color string
                sColor.sprintf("</col>");
            }
        }
    }
    // Return the DML end color string
    return sColor;

} // endColor

//******************************************************************************

CString
color
(
    const CString&      sString,
    const CString&      sForeground,
    const CString&      sBackground
)
{
    CString             sColor;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for current foreground color
        if (!s_sForeground.empty())
        {
            // Check for current background color
            if (!s_sBackground.empty())
            {
                // Set color string (w/foreground & background reset)
                sColor.sprintf("<col fg=\"%s\" bg=\"%s\">%s<col fg=\"%s\" bg=\"%s\">",
                               DML(sForeground),   DML(sBackground), STR(sString),
                               DML(s_sForeground), DML(s_sBackground));
            }
            else    // No current background color
            {
                // Set color string (w/foreground reset)
                sColor.sprintf("<col fg=\"%s\" bg=\"%s\">%s</col><col fg=\"%s\">",
                               DML(sForeground), DML(sBackground), STR(sString),
                               DML(s_sForeground));
            }
        }
        else    // No current foreground color
        {
            // Check for current background color
            if (!s_sBackground.empty())
            {
                // Set color string (w/background reset)
                sColor.sprintf("<col fg=\"%s\" bg=\"%s\">%s</col><col bg=\"%s\">",
                               DML(sForeground), DML(sBackground), STR(sString),
                               DML(s_sBackground));
            }
            else    // No current background color
            {
                // Set color string
                sColor.sprintf("<col fg=\"%s\" bg=\"%s\">%s</col>",
                               DML(sForeground), DML(sBackground), STR(sString));
            }
        }
    }
    else    // Plain text only
    {
        // Just set color string to input string
        sColor = sString;
    }
    return sColor;

} // color

//******************************************************************************

CString
startForeground
(
    const CString&      sForeground
)
{
    CString             sColor;

    // Only generate string if DML enabled
    if (dmlState())
    {
        // Push current foreground color onto the stack (Set new foreground color)
        pushForeground(sForeground);

        // Build and return the DML start foreground color string
        sColor.sprintf("<col fg=\"%s\">", DML(sForeground));
    }
    return sColor;

} // startForeground

//******************************************************************************

CString
endForeground()
{
    CString             sColor;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Pop foreground color from the stack
        popForeground();

        // Check for current foreground color
        if (!s_sForeground.empty())
        {
            // Set end foreground string (w/foreground reset)
            sColor.sprintf("<col fg=\"%s\">", DML(s_sForeground));
        }
        else    // No current foreground color
        {
            // Check for current background color
            if (!s_sBackground.empty())
            {
                // Set end foreground string (w/background reset)
                sColor.sprintf("</col><col bg=\"%s\">", DML(s_sBackground));
            }
            else    // No current background color
            {
                // Set end foreground string
                sColor.sprintf("</col>");
            }
        }
    }
    // Return the DML end foreground color string
    return sColor;

} // endForeground

//******************************************************************************

CString
foreground
(
    const CString&      sString,
    const CString&      sForeground
)
{
    CString             sColor;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for a current foreground color
        if (!s_sForeground.empty())
        {
            // Set DML foreground color string (w/foreground reset)
            sColor.sprintf("<col fg=\"%s\">%s<col fg=\"%s\">", DML(sForeground), STR(sString), DML(s_sForeground));
        }
        else    // No current foreground color
        {
            // Check for a current background color
            if (!s_sBackground.empty())
            {
                // Set DML foreground color string (w/background reset)
                sColor.sprintf("<col fg=\"%s\">%s</col><col bg=\"%s\">", DML(sForeground), STR(sString), DML(s_sBackground));
            }
            else    // No current background color
            {
                // Set DML foreground color string
                sColor.sprintf("<col fg=\"%s\">%s</col>", DML(sForeground), STR(sString));
            }
        }
    }
    else    // Plain text only
    {
        // Just set color string to input string
        sColor = sString;
    }
    return sColor;

} // foreground

//******************************************************************************

CString
startBackground
(
    const CString&      sBackground
)
{
    CString             sColor;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Push current background color onto the stack (Set new background color)
        pushBackground(sBackground);

        // Build and return the DML start background color string
        sColor.sprintf("<col bg=\"%s\">", DML(sBackground));
    }
    return sColor;

} // startBackground

//******************************************************************************

CString
endBackground()
{
    CString             sColor;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Pop background color from the stack
        popBackground();

        // Check for current background color
        if (!s_sBackground.empty())
        {
            // Set end background string (w/background reset)
            sColor.sprintf("<col bg=\"%s\">", DML(s_sBackground));
        }
        else    // No current background color
        {
            // Check for current foreground color
            if (!s_sForeground.empty())
            {
                // Set end background string (w/foreground reset)
                sColor.sprintf("</col><col fg=\"%s\">", DML(s_sForeground));
            }
            else    // No current foreground color
            {
                // Set end background string
                sColor.sprintf("</col>");
            }
        }
    }
    // Return the DML end background color string
    return sColor;

} // endBackground

//******************************************************************************

CString
background
(
    const CString&      sString,
    const CString&      sBackground
)
{
    CString             sColor;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for a current background color
        if (!s_sBackground.empty())
        {
            // Set DML background color string (w/background reset)
            sColor.sprintf("<col bg=\"%s\">%s<col bg=\"%s\">", DML(sBackground), STR(sString), DML(s_sBackground));
        }
        else    // No current background color
        {
            // Check for a current foreground color
            if (!s_sForeground.empty())
            {
                // Set DML background color string (w/foreground reset)
                sColor.sprintf("<col bg=\"%s\">%s</col><col fg=\"%s\">", DML(sBackground), STR(sString), DML(s_sForeground));
            }
            else    // No current foreground color
            {
                // Set DML background color string
                sColor.sprintf("<col bg=\"%s\">%s</col>", DML(sBackground), STR(sString));
            }
        }
    }
    else    // Plain text only
    {
        // Just set color string to input string
        sColor = sString;
    }
    return sColor;

} // background

//******************************************************************************

CString
startBold()
{
    CString             sBold;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Update the bold state
        s_bBold = true;

        // Build the DML start bold string
        sBold = "<b>";
    }
    // Return the DML start bold string
    return sBold;

} // startBold

//******************************************************************************

CString
endBold()
{
    CString             sBold;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Update the bold state
        s_bBold = false;

        // Build the DML end bold string
        sBold = "</b>";
    }
    // Return the DML end bold string
    return sBold;

} // endBold

//******************************************************************************

CString
bold
(
    const CString&      sString
)
{
    CString             sBold;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build and return the DML bold string
        sBold.sprintf("<b>%s</b>", STR(sString));
    }
    else    // Plain text only
    {
        // Just set bold string to input string
        sBold = sString;
    }
    return sBold;

} // bold

//******************************************************************************

CString
startItalic()
{
    CString             sItalic;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Update the italic state
        s_bItalic = true;

        // Build the DML start italic string
        sItalic = "<i>";
    }
    // Return the DML start italic string
    return sItalic;

} // startItalic

//******************************************************************************

CString
endItalic()
{
    CString             sItalic;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Update the italic state
        s_bItalic = false;

        // Build the DML end italic string
        sItalic = "</i>";
    }
    // Return the DML end italic string
    return sItalic;

} // endItalic

//******************************************************************************

CString
italic
(
    const CString&      sString
)
{
    CString             sItalic;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build and return the DML italic string
        sItalic.sprintf("<i>%s</i>", STR(sString));
    }
    else    // Plain text only
    {
        // Just set italic string to input string
        sItalic = sString;
    }
    return sItalic;

} // italic

//******************************************************************************

CString
startUnderline()
{
    CString             sUnderline;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Update the underline state
        s_bUnderline = true;

        // Build the DML start underline string
        sUnderline = "<u>";
    }
    // Return the DML start underline string
    return sUnderline;

} // startUnderline

//******************************************************************************

CString
endUnderline()
{
    CString             sUnderline;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Update the underline state
        s_bUnderline = false;

        // Build the DML end underline string
        sUnderline = "</u>";
    }
    // Return the DML end underline string
    return sUnderline;

} // endUnderline

//******************************************************************************

CString
underline
(
    const CString&      sString
)
{
    CString             sUnderline;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build and return the DML underline string
        sUnderline.sprintf("<u>%s</u>", STR(sString));
    }
    else    // Plain text only
    {
        // Just set underline string to input string
        sUnderline = sString;
    }
    return sUnderline;

} // underline

//******************************************************************************

CString
startExec
(
    const CString&      sCommand
)
{
    CString             sExec;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build and return the DML start exec string
        sExec.sprintf("<exec cmd=\"%s\">", STR(sCommand));
    }
    return sExec;

} // startExec

//******************************************************************************

CString
endExec()
{
    CString             sExec;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build the DML end exec string
        sExec = "</exec>";
    }
    // Return the DML end exec string
    return sExec;

} // endExec

//******************************************************************************

CString
exec
(
    const CString&      sString,
    const CString&      sCommand
)
{
    CString             sExec;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build and return the DML exec string
        sExec.sprintf("<exec cmd=\"%s\">%s</exec>", STR(sCommand), STR(sString));
    }
    else    // Plain text only
    {
        // Just set exec string to input string
        sExec = sString;
    }
    return sExec;

} // exec

//******************************************************************************

CString
startLink
(
    const CString&      sName,
    const char         *pSection
)
{
    CString             sLink;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for a section given
        if (pSection != NULL)
        {
            // Build the DML start link string
            sLink.sprintf("<link name=\"%s\" section=\"%s\">", STR(sName), pSection);
        }
        else    // No section given
        {
            // Build the DML start link string
            sLink.sprintf("<link name=\"%s\">", STR(sName));
        }
    }
    // Return the start link string
    return sLink;

} // startLink

//******************************************************************************

CString
endLink()
{
    CString             sLink;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build the DML end link string
        sLink = "</link>";
    }
    // Return the DML end link string
    return sLink;

} // endLink

//******************************************************************************

CString
link
(
    const CString&      sString,
    const CString&      sName,
    const char         *pSection
)
{
    CString             sLink;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for a section given
        if (pSection != NULL)
        {
            // Build the DML link string
            sLink.sprintf("<link name=\"%s\" section=\"%s\">%s</link>", STR(sName), pSection, STR(sString));
        }
        else    // No section given
        {
            // Build the DML link string
            sLink.sprintf("<link name=\"%s\">%s</link>", STR(sName), STR(sString));
        }
    }
    else    // Plain text only
    {
        // Just set link string to input string
        sLink = sString;
    }
    // Return the link string
    return sLink;

} // link

//******************************************************************************

CString
startSection
(
    const CString&      sSection,
    const char         *pName
)
{
    CString             sLink;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for a name given
        if (pName != NULL)
        {
            // Build the DML start section string
            sLink.sprintf("<link name=\"%s\" section=\"%s\">", pName, STR(sSection));
        }
        else    // No name given
        {
            // Build the DML start section string
            sLink.sprintf("<link section=\"%s\">", STR(sSection));
        }
    }
    // Return the start section string
    return sLink;

} // startSection

//******************************************************************************

CString
endSection()
{
    CString             sLink;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build the DML end section string
        sLink = "</link>";
    }
    // Return the DML end section string
    return sLink;

} // endSection

//******************************************************************************

CString
section
(
    const CString&      sString,
    const CString&      sSection,
    const char         *pName
)
{
    CString             sLink;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for a name given
        if (pName != NULL)
        {
            // Build the DML start section string
            sLink.sprintf("<link name=\"%s\" section=\"%s\">%s</link>", pName, STR(sSection), STR(sString));
        }
        else    // No name given
        {
            // Build the DML link string
            sLink.sprintf("<link section=\"%s\">%s</link>", STR(sSection), STR(sString));
        }
    }
    else    // Plain text only
    {
        // Just set link string to input string
        sLink = sString;
    }
    // Return the link string
    return sLink;

} // section

//******************************************************************************

CString
startLinkCmd
(
    const CString&      sCommand,
    const CString&      sName,
    const char         *pSection
)
{
    CString             sLink(MAX_DBGPRINTF_STRING);

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for a section given
        if (pSection != NULL)
        {
            // Build the DML start link string
            sLink.sprintf("<link cmd=\"%s\" name=\"%s\" section=\"%s\">", STR(sCommand), STR(sName), pSection);
        }
        else    // No section given
        {
            // Build the DML start link string
            sLink.sprintf("<link cmd=\"%s\" name=\"%s\">", STR(sCommand), STR(sName));
        }
    }
    // Return the start link string
    return sLink;

} // startLinkCmd

//******************************************************************************

CString
endLinkCmd()
{
    CString             sLink;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Build the DML end link string
        sLink = "</link>";
    }
    // Return the DML end link string
    return sLink;

} // endLinkCmd

//******************************************************************************

CString
linkCmd
(
    const CString&      sString,
    const CString&      sCommand,
    const CString&      sName,
    const char         *pSection
)
{
    CString             sLink;

    // Only generate DML string if DML enabled
    if (dmlState())
    {
        // Check for a section given
        if (pSection != NULL)
        {
            // Build the DML link string
            sLink.sprintf("<link cmd=\"%s\" name=\"%s\" section=\"%s\">%s</link>", STR(sCommand), STR(sName), pSection, STR(sString));
        }
        else    // No section given
        {
            // Build the DML link string
            sLink.sprintf("<link cmd=\"%s\" name=\"%s\">%s</link>", STR(sCommand), STR(sName), STR(sString));
        }
    }
    else    // Plain text only
    {
        // Just set link string to input string
        sLink = sString;
    }
    // Return the link string
    return sLink;

} // linkCmd

//******************************************************************************

size_t
dmllen
(
    const CString&      sString
)
{
    size_t              location;
    size_t              special;
    size_t              size;
    size_t              start;
    size_t              end;
    size_t              position = 0;
    size_t              length   = 0;

    // Only callwlate DML length if DML enabled
    if (dmlState())
    {
        // Search string for DML sections
        while ((location = sString.find(DML_LESS_THAN, position)) != NOT_FOUND)
        {
            // Add characters before DML section to string length
            length += location - position;

            // Set starting location of string
            start = position;

            // Search for "special" DML sequences
            while ((special = sString.find(DML_AMPERSAND, start)) != NOT_FOUND)
            {
                // Check for a valid special location
                if (special < location)
                {
                    // Callwlate the number of characters before DML start
                    size = location - special - 1;

                    // Check for any DML "escaped" characters
                    if ((size >= sizeof(DML_LESS_THAN_ESCAPE)) && (_strnicmp(sString.str(special), DML_LESS_THAN_ESCAPE, sizeof(DML_LESS_THAN_ESCAPE)) == 0))
                    {
                        // Adjust length for DML "escaped" less than sequence
                        length -= sizeof(DML_LESS_THAN_ESCAPE) - 1;

                        // Update start past the escape sequence
                        start += sizeof(DML_LESS_THAN_ESCAPE);
                    }
                    else if ((size >= sizeof(DML_GREATER_THAN_ESCAPE)) && (_strnicmp(sString.str(special), DML_GREATER_THAN_ESCAPE, sizeof(DML_GREATER_THAN_ESCAPE)) == 0))
                    {
                        // Adjust length for DML "escaped" greater than sequence
                        length -= sizeof(DML_GREATER_THAN_ESCAPE) - 1;

                        // Update start past the escape sequence
                        start += sizeof(DML_GREATER_THAN_ESCAPE);
                    }
                    else if ((size >= sizeof(DML_AMPERSAND_ESCAPE)) && (_strnicmp(sString.str(special), DML_AMPERSAND_ESCAPE, sizeof(DML_AMPERSAND_ESCAPE)) == 0))
                    {
                        // Adjust length for DML "escaped" ampersand sequence
                        length -= sizeof(DML_AMPERSAND_ESCAPE) - 1;

                        // Update start past the escape sequence
                        start += sizeof(DML_AMPERSAND_ESCAPE);
                    }
                    else if ((size >= sizeof(DML_QUOTE_ESCAPE)) && (_strnicmp(sString.str(special), DML_QUOTE_ESCAPE, sizeof(DML_QUOTE_ESCAPE)) == 0))
                    {
                        // Adjust length for DML "escaped" quote sequence
                        length -= sizeof(DML_QUOTE_ESCAPE) - 1;

                        // Update start past the escape sequence
                        start += sizeof(DML_QUOTE_ESCAPE);
                    }
                    else    // Not an "escaped" DML character
                    {
                        // Update start past the special character
                        start = special + 1;
                    }
                    // Check for end of string to check
                    if (start >= location)
                    {
                        // Stop searching for "escaped" sequences
                        break;
                    }
                }
                else    // Invalid special location
                {
                    // We can just stop the search here
                    break;
                }
            }
            // Set starting location of DML section
            start = location + 1;

            // Try to find end of the DML section
            end = sString.find(DML_GREATER_THAN, start);
            if (end != NOT_FOUND)
            {
                // Update position to past the DML section (Don't include in length)
                position = end + 1;
            }
            else    // Invalid DML section
            {
                throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                 ": Invalid DML section");
            }
        }
        // Add any remaining characters after DML sections to string length
        length += sString.size() - position;

        // Set starting location of string
        start = position;

        // Search for "special" DML sequences
        while ((special = sString.find(DML_AMPERSAND, start)) != NOT_FOUND)
        {
            // Check for a valid special location
            if (special < location)
            {
                // Callwlate the number of characters before DML start
                size = location - special - 1;

                // Check for any DML "escaped" characters
                if ((size >= sizeof(DML_LESS_THAN_ESCAPE)) && (_strnicmp(sString.str(special), DML_LESS_THAN_ESCAPE, sizeof(DML_LESS_THAN_ESCAPE)) == 0))
                {
                    // Adjust length for DML "escaped" less than sequence
                    length -= sizeof(DML_LESS_THAN_ESCAPE) - 1;

                    // Update start past the escape sequence
                    start += sizeof(DML_LESS_THAN_ESCAPE);
                }
                else if ((size >= sizeof(DML_GREATER_THAN_ESCAPE)) && (_strnicmp(sString.str(special), DML_GREATER_THAN_ESCAPE, sizeof(DML_GREATER_THAN_ESCAPE)) == 0))
                {
                    // Adjust length for DML "escaped" greater than sequence
                    length -= sizeof(DML_GREATER_THAN_ESCAPE) - 1;

                    // Update start past the escape sequence
                    start += sizeof(DML_GREATER_THAN_ESCAPE);
                }
                else if ((size >= sizeof(DML_AMPERSAND_ESCAPE)) && (_strnicmp(sString.str(special), DML_AMPERSAND_ESCAPE, sizeof(DML_AMPERSAND_ESCAPE)) == 0))
                {
                    // Adjust length for DML "escaped" ampersand sequence
                    length -= sizeof(DML_AMPERSAND_ESCAPE) - 1;

                    // Update start past the escape sequence
                    start += sizeof(DML_AMPERSAND_ESCAPE);
                }
                else if ((size >= sizeof(DML_QUOTE_ESCAPE)) && (_strnicmp(sString.str(special), DML_QUOTE_ESCAPE, sizeof(DML_QUOTE_ESCAPE)) == 0))
                {
                    // Adjust length for DML "escaped" quote sequence
                    length -= sizeof(DML_QUOTE_ESCAPE) - 1;

                    // Update start past the escape sequence
                    start += sizeof(DML_QUOTE_ESCAPE);
                }
                else    // Not an "escaped" DML character
                {
                    // Update start past the special character
                    start = special + 1;
                }
                // Check for end of string to check
                if (start >= location)
                {
                    // Stop searching for "escaped" sequences
                    break;
                }
            }
            else    // Invalid special location
            {
                // We can just stop the search here
                break;
            }
        }
    }
    else    // Plain text only
    {
        // Set length to string length
        length = sString.length();
    }
    return length;

} // dmllen

//******************************************************************************

size_t
dmlDelta
(
    const CString&      sString
)
{
    size_t              delta = 0;

    // Only return DML delta if DML enabled
    if (dmlState())
    {
        // Callwlate delta between strlen and dmllen
        delta = strlen(sString) - dmllen(sString);
    }
    return delta;

} // dmlDelta

//******************************************************************************

CString
dmlStrip
(
    const CString&      sString
)
{
    size_t              location;
    size_t              start;
    size_t              end;
    size_t              position = 0;
    size_t              length   = 0;
    CString             sStripped(sString);

    // Only strip DML if DML enabled
    if (dmlState())
    {
        // Search string for DML sections
        while ((location = sStripped.find(DML_LESS_THAN, position)) != NOT_FOUND)
        {
            // Set starting location of DML section
            start = location;

            // Try to find end of the DML section
            end = sStripped.find(DML_GREATER_THAN, start);
            if (end != NOT_FOUND)
            {
                // Compute the length of the DML to strip
                length = end - start + 1;

                // Erase the DML from the string
                sStripped.erase(start, length);

                // Update position to where the DML started
                position = start;
            }
            else    // Invalid DML section
            {
                throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                 ": Invalid DML section");
            }
        }
    }
    return sStripped;

} // dmlStrip

//******************************************************************************

CString
dmlEscape
(
    const CString&      sString
)
{
    size_t              location;
    size_t              position;
    CString             sEscaped(sString);

    // Only DML escape string if DML enabled
    if (dmlState())
    {
        // Search string for DML ampersand characters (and replace if found)
        position = 0;
        while ((location = sEscaped.find(DML_AMPERSAND, position)) != NOT_FOUND)
        {
            // Replace DML ampersand character with escape sequence
            sEscaped.replace(location, 1, DML_AMPERSAND_ESCAPE);

            // Update position to where replacement was made
            position = location + sizeof(DML_AMPERSAND_ESCAPE);

            // Make sure we are not at the end of the string
            if (position > sEscaped.size())
            {
                // Terminate the search
                break;
            }
        }
        // Search string for DML less than characters (and replace if found)
        position = 0;
        while ((location = sEscaped.find(DML_LESS_THAN, position)) != NOT_FOUND)
        {
            // Replace DML less than character with escape sequence
            sEscaped.replace(location, 1, DML_LESS_THAN_ESCAPE);

            // Update position to where replacement was made
            position = location + sizeof(DML_LESS_THAN_ESCAPE);

            // Make sure we are not at the end of the string
            if (position > sEscaped.size())
            {
                // Terminate the search
                break;
            }
        }
        // Search string for DML greater than characters (and replace if found)
        position = 0;
        while ((location = sEscaped.find(DML_GREATER_THAN, position)) != NOT_FOUND)
        {
            // Replace DML greater than character with escape sequence
            sEscaped.replace(location, 1, DML_GREATER_THAN_ESCAPE);

            // Update position to where replacement was made
            position = location + sizeof(DML_GREATER_THAN_ESCAPE);

            // Make sure we are not at the end of the string
            if (position > sEscaped.size())
            {
                // Terminate the search
                break;
            }
        }
        // Search string for DML quote characters (and replace if found)
        position = 0;
        while ((location = sEscaped.find(DML_QUOTE, position)) != NOT_FOUND)
        {
            // Replace DML quote character with escape sequence
            sEscaped.replace(location, 1, DML_QUOTE_ESCAPE);

            // Update position to where replacement was made
            position = location + sizeof(DML_QUOTE_ESCAPE);

            // Make sure we are not at the end of the string
            if (position > sEscaped.size())
            {
                // Terminate the search
                break;
            }
        }
    }
    return sEscaped;

} // dmlEscape

//******************************************************************************

void
dmlReset()
{
    // Only perform DML reset if DML enabled
    if (dmlState())
    {
        // Check for foreground or background colors on the stack
        if (s_ulForeground || s_ulBackground)
        {
            // Pop off all the foreground colors
            while (s_ulForeground != 0)
            {
                popForeground();
            }
            // Pop off all the background colors
            while (s_ulBackground != 0)
            {
                popBackground();
            }
            // Reset the foreground/background colors
            dPrintf("</col>");
        }
        // Check for bold left on
        if (s_bBold)
        {
            // Clear the bold state
            s_bBold = false;
            dPrintf("</b>");
        }
        // Check for italic left on
        if (s_bItalic)
        {
            // Clear the italic state
            s_bItalic = false;
            dPrintf("</i>");
        }
        // Check for underline left on
        if (s_bUnderline)
        {
            // Clear the underline state
            s_bUnderline = false;
            dPrintf("</u>");
        }
    }

} // dmlReset

//******************************************************************************

static void
pushForeground
(
    const CString&      sForeground
)
{
    // Check for foreground stack overflow
    if (s_ulForeground < MAX_DML_LEVEL)
    {
        // Push current foreground color onto the stack (and set new color)
        s_aForeground[s_ulForeground] = s_sForeground;
        s_sForeground                 = sForeground;

        // Increment foreground stack level
        s_ulForeground++;
    }
    else    // Foreground stack overflow
    {
        throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                         ": DML foreground stack overflow");
    }

} // pushForeground

//******************************************************************************

static void
popForeground()
{
    // Check for foreground stack underflow
    if (s_ulForeground != 0)
    {
        // Decrement foreground stack level
        s_ulForeground--;

        // Pop current foreground color from the stack (and clear stack entry)
        s_sForeground = s_aForeground[s_ulForeground];
        s_aForeground[s_ulForeground].clear();
    }
    else    // Foreground stack underflow
    {
        throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                         ": DML foreground stack underflow");
    }

} // popForeground

//******************************************************************************

static void
pushBackground
(
    const CString&      sBackground
)
{
    // Check for background stack overflow
    if (s_ulBackground < MAX_DML_LEVEL)
    {
        // Push current background color onto the stack (and set new color)
        s_aBackground[s_ulBackground] = s_sBackground;
        s_sBackground                 = sBackground;

        // Increment background stack level
        s_ulBackground++;
    }
    else    // Background stack overflow
    {
        throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                         ": DML background stack overflow");
    }

} // pushBackground

//******************************************************************************

static void
popBackground()
{
    // Check for background stack underflow
    if (s_ulBackground != 0)
    {
        // Decrement background stack level
        s_ulBackground--;

        // Pop current background color from the stack (and clear stack entry)
        s_sBackground = s_aBackground[s_ulBackground];
        s_aBackground[s_ulBackground].clear();
    }
    else    // Background stack underflow
    {
        throw CException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                         ": DML background stack underflow");
    }

} // popBackground

//******************************************************************************

bool
dmlState()
{
    // Return the current DML state
    return s_bDmlState;

} // dmlState

//******************************************************************************

bool
dmlState
(
    bool                bDmlState
)
{
    bool                bLastDmlState = s_bDmlState;

    // Set the new DML state
    s_bDmlState = bDmlState;

    // Check the new DML state and set the global DML state
    if (s_bDmlState)
    {
        // Add/enable the global DML engine option
        AddEngineOptions(DEBUG_ENGOPT_PREFER_DML);
    }
    else    // DML disabled
    {
        // Remove/disable the global DML engine option
        RemoveEngineOptions(DEBUG_ENGOPT_PREFER_DML);
    }
    // Return the last DML state
    return bLastDmlState;

} // dmlState

//******************************************************************************

CDmlState::CDmlState()
:   m_bDmlState(false)
{
    // Update the DML state (Initial)
    update();

} // CDmlState

//******************************************************************************

CDmlState::~CDmlState()
{
    // Restore the DML state
    restore();

} // ~CDmlState

//******************************************************************************

void
CDmlState::update()
{
    // Update the DML state
    m_bDmlState = dmlState();

} // update

//******************************************************************************

void
CDmlState::restore()
{
    // Restore the DML state (if necessary)
    if (dmlState() != m_bDmlState)
    {
        dmlState(m_bDmlState);
    }

} // restore

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
