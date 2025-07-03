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
 * @file  flcngdbUI.cpp
 * @brief UI Source window implemenataion on WinDbg
 *
 *  */
#include <fstream>
#include <iostream>
#include <string>

#ifdef WIN32

#include "flcngdbUI.h"
#include "flcngdbTypes.h"

using namespace std;

extern "C"
{
    LRESULT CALLBACK flcngdbUiWndProcFlcngdb(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        RECT r;

        switch(message)
        {
        case WM_CREATE:
            // make the source window always on top
            SetWindowPos(hWnd, HWND_TOPMOST, 0, 0, 1, 1, SWP_NOSIZE | SWP_DRAWFRAME | SWP_NOACTIVATE);
            GetClientRect(hWnd, &r);

            // create a RichEdit v1 text box with ID 1
            flcngdbUiRichEdit = CreateWindow(TEXT("RichEdit"), TEXT("Source code window"),
                WS_VISIBLE | WS_CHILD | WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | WS_HSCROLL,
                0, 0, r.right, r.bottom, hWnd, (HMENU) 1, NULL, NULL);
            if(flcngdbUiRichEdit != NULL)
                flcngdbUibHasEditor = true;

            // disable line wrap
            SendMessage(flcngdbUiRichEdit, EM_SETTARGETDEVICE, NULL, 1);

            break;

        default:
            break;
        }

        // default windows message handler for all the messages we did not handle
        // above
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    DWORD WINAPI flcngdbUiThread(LPVOID lpParam)
    {
        LoadLibrary(TEXT("Riched32.dll"));

        // stuff needed for creating a Windows window
        WNDCLASS wndClass;
        MSG msg;
        LPVOID lpMsgBuf;
        BOOL hasMsg;
        static bool registeredWindow = false;

        if(!registeredWindow) {
            // regsiter the window class
            memset (&wndClass, 0, sizeof(wndClass));
            wndClass.cbWndExtra    = 4;
            wndClass.hLwrsor       = LoadLwrsor( NULL, IDC_ARROW );
            wndClass.hIcon         = LoadIcon(GetModuleHandle(NULL), MAKEINTRESOURCE(IDI_APPLICATION));
            wndClass.hInstance     = GetModuleHandle(NULL);
            wndClass.lpfnWndProc   = flcngdbUiWndProcFlcngdb;
            wndClass.lpszClassName = TEXT("FlcngdbCodeWindow");
            wndClass.style         = CS_HREDRAW | CS_VREDRAW;

            // translate windows error messages if any oclwred
            if(!RegisterClass(&wndClass))
            {
                wndClass.lpfnWndProc = NULL;
                FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                    FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                    NULL,
                    GetLastError(),
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                    (LPTSTR) &lpMsgBuf,
                    0,
                    NULL);

                MessageBoxA(NULL, (LPCTSTR)lpMsgBuf, "Error", MB_OK | MB_ICONINFORMATION);
                // Free the buffer.
                LocalFree(lpMsgBuf);

                return 0;
            }
            registeredWindow = true;
        }

        // create the window
        flcngdbUiHwnd = CreateWindow(TEXT("FlcngdbCodeWindow"), TEXT("Flcngdb Source Window"),
            WS_BORDER | WS_CAPTION | WS_POPUP,
            50, 50, WINDOW_WIDTH, WINDOW_HEIGHT, NULL, NULL, wndClass.hInstance, 0);
        if(!flcngdbUiHwnd)
        {
            MessageBoxA( NULL, "Invalid hWnd", "Error", MB_OK | MB_ICONINFORMATION );
            if (wndClass.lpfnWndProc) {
                UnregisterClass(wndClass.lpszClassName, wndClass.hInstance);
            }
            return 1;
        }

        // display the window
        ShowWindow(flcngdbUiHwnd, SW_NORMAL);
        UpdateWindow(flcngdbUiHwnd);

        while(1) {
            // process Window messages
            hasMsg = GetMessage(&msg, NULL, 0, 0);
            if(hasMsg)
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
    }

    void flcngdbUiCreateFlcngdbWindow()
    {
        // variables required for thread
        DWORD uiId;
        LPVOID lpMsgBuf;

        flcngdbUibHasEditor = FALSE;

        // open a window to display source code. This has to be recreate each time we
        // re-enter the debugger (otherwise it would freeze anyway since the message queue
        // isnt being processed when this debugger isnt running)
        flcngdbUiH = CreateThread(NULL, 0, flcngdbUiThread, NULL, 0, &uiId);
        if(flcngdbUiH == NULL) {
            FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL,
                GetLastError(),
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                (LPTSTR) &lpMsgBuf,
                0,
                NULL);

            MessageBoxA( NULL, (LPCTSTR)lpMsgBuf, "Error", MB_OK | MB_ICONINFORMATION );
            // Free the buffer.
            LocalFree(lpMsgBuf);
        }
    }

    void flcngdbUiCloseFlcngdbWindow()
    {
        // TODO:
        // If this UI thread expands to anything more than display this has to be
        // edited so that the thread closes itself rather than just simply being
        // terminated
        if(flcngdbUiH != NULL)
            TerminateThread(flcngdbUiH, 0);
    }

    void flcngdbUiLoadFileFlcngdbWindow(const char* filename)
    {
        if(strcmp(filename, flcngdbUiFileToLoad) != 0)
        {
            strcpy(flcngdbUiFileToLoad, filename);

            int lineCount = 1;
            string content;
            string line;

            ifstream inFile(flcngdbUiFileToLoad);
            if(inFile.is_open())
            {
                while(inFile.good())
                {
                    getline(inFile, line);
                    content.append(line).append("\n");
                    lineCount++;
                }
            }

            // load string
            SendMessage(flcngdbUiRichEdit, WM_SETTEXT, NULL, (LPARAM) content.c_str());

            // wait for the file to show up on screen
            while(1)
            {
                if(SendMessage(flcngdbUiRichEdit, EM_GETLINECOUNT, 0, 0) >= lineCount)
                    break;
            }
        }
    }

    // Go to the source line number indicated, highlight the line and center the line
    // on screen
    void flcngdbUiCenterOnLineFlcngdbWindow(const int n)
    {
        static int lastStartCharOfLine = 0;
        static int lastLineLen = 0;
        int startCharOfLine;
        int lineLen;
        int topVisibleLine;
        int thisLine;
        int bottomVisibleLine;
        int scrollAmount;
        int scrollDirection;
        RECT r;
        POINTL p;

        // find the starting character position of the current
        startCharOfLine = (int) SendMessage(flcngdbUiRichEdit, EM_LINEINDEX, n, 0);

        // find the line length
        lineLen = (int) SendMessage(flcngdbUiRichEdit, EM_LINELENGTH, startCharOfLine, 0);

        // restore the old line
        SendMessage(flcngdbUiRichEdit, EM_SETSEL, lastStartCharOfLine, lastStartCharOfLine+lastLineLen);
        flcngdbUiCformat.dwMask = CFM_BACKCOLOR | CFM_COLOR;
        flcngdbUiCformat.cbSize = sizeof(flcngdbUiCformat);
        flcngdbUiCformat.crBackColor = RGB(255,255,255);
        SendMessage(flcngdbUiRichEdit, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM) &flcngdbUiCformat);
        SendMessage(flcngdbUiRichEdit, EM_SCROLLCARET, 0, 0);

        // select the line
        SendMessage(flcngdbUiRichEdit, EM_SETSEL, startCharOfLine, startCharOfLine+lineLen);

        // apply style
        flcngdbUiCformat.dwMask = CFM_BACKCOLOR | CFM_COLOR;
        flcngdbUiCformat.cbSize = sizeof(flcngdbUiCformat);
        flcngdbUiCformat.crBackColor = RGB(255,0,0);
        SendMessage(flcngdbUiRichEdit, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM) &flcngdbUiCformat);
        SendMessage(flcngdbUiRichEdit, EM_SCROLLCARET, 0, 0);

        // store the last values for restoring
        lastStartCharOfLine = startCharOfLine;
        lastLineLen = lineLen;

        // get the line number of the top visible line
        topVisibleLine = (int) SendMessage(flcngdbUiRichEdit, EM_GETFIRSTVISIBLELINE, 0, 0);

        // line number of this line
        thisLine = (int) SendMessage(flcngdbUiRichEdit, EM_LINEFROMCHAR, startCharOfLine, 0);

        // line number of the bottom visible line, this has to be done by generating
        // a drawing rectable and then using EM_CHARFROMPOS with the bottom most
        // pixel position of the editor window
        SendMessage(flcngdbUiRichEdit, EM_GETRECT, 0, (LPARAM) &r);
        p.x = r.right;
        p.y = r.bottom;
        bottomVisibleLine = (int) SendMessage(flcngdbUiRichEdit, EM_CHARFROMPOS, 0, (LPARAM) &p);
        bottomVisibleLine = (int) SendMessage(flcngdbUiRichEdit, EM_LINEFROMCHAR, bottomVisibleLine, 0);

        // We want thisLine to be the center of the screen.
        // bottomLine - topLine / 2 is the target line
        scrollAmount = thisLine - ((bottomVisibleLine - topVisibleLine)/2) - topVisibleLine;


        // change the scroll direction
        if (scrollAmount > 0)
            scrollDirection = SB_LINEDOWN;
        else
            scrollDirection = SB_LINEUP;

        // scroll lines
        for(int i = 0; i<abs(scrollAmount); i++)
        {
            SendMessage(flcngdbUiRichEdit, WM_VSCROLL, scrollDirection, 0);
        }
    }

    void flcngdbUiWaitForWindowCreation()
    {
        while(!flcngdbUibHasEditor)
        {
            //osPerfDelay(100 * 1000);
            Sleep(100);
        }
    }
}

#endif //WIN32

