/*
 * Copyright (c) 2015-2020, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <LlgdTestUtil.h>
#include <LlgdTestUtilDisplay.h>

#include <stdio.h>

#if !defined(WIN_INTERFACE_LWSTOM)
#define WIN_INTERFACE_LWSTOM
#endif

#include "lwn_win.h"

namespace llgd_lwn
{
void DisplayUtil::Initialize()
{
    LwnWin *lwnWin = LwnWin::GetInstance();
    if (lwnWin == NULL) {
        LlgdErr("%s", "Cannot obtain window interface. Be sure to add LW_WINSYS=null for headless display.\n");
        return;
    }

    int windowWidth  = 640;
    int windowHeight = 480;
    _nativeWindow = lwnWin->CreateWindow("LWN", windowWidth, windowHeight);

    if (!_nativeWindow) {
        LlgdErr("%s", "Cannot create window.\n");
    }
}

void DisplayUtil::Finalize()
{
    LwnWin *lwnWin = LwnWin::GetInstance();
    if (lwnWin == NULL) {
        LlgdErr("%s", "Cannot obtain window interface.\n");
        return;
    }

    lwnWin->DestroyWindow(_nativeWindow);
}
} // llgd_lwn
