/***************************************************************************\
* Copyright 1993-1999 LWPU, Corporation.  All rights reserved.            *
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO       *
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY  *
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.                *
*                                                                           *
*  Portions: Copyright (C) 1995 Microsoft Corporation.  All Rights Reserved.*
*                                                                           *
* Module: LwViewer.h                                                        *
*                                                                           *
*****************************************************************************
*                                                                           *
* History:                                                                  *
*       Andrei Osnovich    10/04/2000  Created                              *
*                                                                           *
\***************************************************************************/

#ifndef _LWVIEWER_H_
#define _LWVIEWER_H_

//Stereo image viewing mode
#define FULL_SCREEN         0
#define WINDOWED            1
#define WINDOWED_FIT        2
#define SCALE_TO_FIT        4   //Full screen only

// Flags for Display(...) and ViewDisplay(...)
#define INTERNAL_MESSAGING  0x00000000
#define EXTERNAL_MESSAGING  0x00000001
#define DISPLAY_ANAGLYPH 0x00000002
#define FRAMES_TIMED 0x80000000
#define SECONDS_PER_FRAME(x) (FRAMES_TIMED | ((x & 0x7F) << 24))
#define SECONDS_IN_FLAGS(x) (x & FRAMES_TIMED ? ((x & 0x7f000000) >> 24) : 0)

// Return values from Display(...) and ViewDisplay(...)
#define IV_QUIT 0
#define IV_NEXTIMAGE 1
#define IV_PREVIMAGE 2
#define IV_TRYNEXTIMAGE 3
#define IV_MOVEIMAGE 4

typedef class CStereoImageViewer
{
    /*
    * methods
    */
public:
    virtual DWORD Display (LPVOID pImage, DWORD dwWidth, DWORD dwHeight, DWORD dwBPP, DWORD dwViewMode, DWORD dwFlags);
    virtual DWORD Display (char *filename, DWORD dwViewMode, DWORD dwBPP, DWORD dwFlags);
    virtual DWORD WINAPI DestroyStereoImageViewer(void);

    virtual DWORD ViewSetup(HWND hWnd, DWORD dwWidth, DWORD dwHeight, DWORD dwBPP);

    virtual DWORD ViewDisplay(LPVOID pImage, DWORD dwWidth, DWORD dwHeight, DWORD dwBPP, DWORD dwViewMode, DWORD dwFlags, DWORD dwLeftEyeMask = 0xFFFFFFFF, DWORD dwRightEyeMask = 0xFFFFFFFF);
    virtual DWORD ViewDisplay(char *filename, DWORD dwViewMode, DWORD dwFlags, DWORD dwLeftEyeMask = 0xFFFFFFFF, DWORD dwRightEyeMask = 0xFFFFFFFF);

    virtual DWORD ViewTakedown();

    CStereoImageViewer();
    ~CStereoImageViewer();
protected:
    IDirect3D9* m_pD3D;
    IDirect3DDevice9* m_pD3DDev;
    IDirect3DSurface9* m_pImageSurf;
    D3DPRESENT_PARAMETERS m_PresentParams;

    HWND m_hWnd;
    RECT m_rWindowRect;

    DWORD m_dwWidth, m_dwHeight, m_dwBPP;
    DWORD m_dwImageWidth, m_dwImageHeight, m_dwImageBPP;

    HRESULT Clear(void);
    HRESULT ITakedown();
    HRESULT Render(RECT *rSource, RECT *rDest);
} CSTEREOIMAGEVIEWER, *LPCSTEREOIMAGEVIEWER;


extern DWORD WINAPI CreateStereoImageViewer(LPCSTEREOIMAGEVIEWER &pStereoImageViewer);

extern ULONG dumpImage(char* filename, int bpp, int width, int height, int pitch, UINT8* pAddr);

#endif _LWVIEWER_H_
