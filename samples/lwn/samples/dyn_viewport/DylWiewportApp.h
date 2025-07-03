/*
* Copyright (c) 2016 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#ifndef _DYLWIEWPORT_APP
#define _DYLWIEWPORT_APP

#include <lwn/lwn_Cpp.h>

class DynamicViewport;

class DylwiewportApp
{
public:
    explicit DylwiewportApp(int argc, char** argv, bool debugEnabled);
    ~DylwiewportApp();

    bool init(LWNnativeWindow* win);
    bool display();

private:

    int             m_numLoops;
    int             m_frame;
    int             m_idx;
    const int       m_numTests;
    bool            m_useOriginTopLeft;
    bool            m_adjustCropRect;
    bool            m_useLwstomRect;
    const bool      m_debugEnabled;

    static const int m_switchFrame = 20;

    lwn::Device     m_device;
    lwn::Rectangle  m_lwstomRect;

    DynamicViewport *m_dv;
};

#endif
