// dxvahdmfc.h : main header file for the dxvahdmfc application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CdxvahdmfcApp:
// See dxvahdmfc.cpp for the implementation of this class
//

class CdxvahdmfcApp : public CWinApp
{
public:
	CdxvahdmfcApp();


// Overrides
public:
	virtual BOOL InitInstance();

// Implementation
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CdxvahdmfcApp theApp;