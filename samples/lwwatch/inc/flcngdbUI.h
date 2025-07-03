/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcngdbUI.h
 * @brief flcngdb UI header file. 
 *
 *  */
#ifndef _FLCNGDBUI_H_
#define _FLCNGDBUI_H_

#include <windows.h>
#include <stdlib.h>
#include "Richedit.h"

// width and height of the code window
#define WINDOW_HEIGHT 750
#define WINDOW_WIDTH 900

#ifdef __cplusplus
extern "C" {    // so compile will not name mangle
#endif

// window handle of the main window
static HWND flcngdbUiHwnd = 0;
// window handle of the richEdit
static HWND flcngdbUiRichEdit;

// thread handle of the UI thread
static HANDLE flcngdbUiH;

// data structure for changing text background
static CHARFORMAT2 flcngdbUiCformat;

// for holding the filename that should be loaded into the window
static char flcngdbUiFileToLoad[512];

// line number to scroll to and highlight
static int flcngdbUiLineNum;

// flag to indicate that the editor has been created
static BOOL flcngdbUibHasEditor;

// WinAPI, window message callback
LRESULT CALLBACK flcngdbUiWndProcFlcngdb(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
// WinAPI, thread
DWORD WINAPI flcngdbUiThread(LPVOID lpParam);

// internal functions
void flcngdbUiLoadContent();

// utility functions for lwwatch to use
void flcngdbUiCreateFlcngdbWindow();
void flcngdbUiWaitForWindowCreation();
void flcngdbUiCloseFlcngdbWindow();
void flcngdbUiLoadFileFlcngdbWindow(const char* pFilename);
void flcngdbUiCenterOnLineFlcngdbWindow(const int lineNum);

#ifdef __cplusplus
}
#endif

#endif /* _FLCNGDBUI_H_ */

