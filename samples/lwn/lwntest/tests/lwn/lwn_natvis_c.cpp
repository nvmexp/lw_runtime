/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwn/lwn.h"
#include "cppogtest.h"
#include "lwn_utils_noapi.h"
#include <vector>

#include <lwnUtil/g_lwnObjectList.h>

// disable ogtest.h hack that breaks placement new
#ifdef new
#undef new
#endif

class LWNNatVisTestC
{
public:
    OGTEST_CppMethods();
};

lwString LWNNatVisTestC::getDescription()
{
    return "Creates each C API object that has NatVis entries.\n"
           "Test requires manual inspection of each API type using the\n"
           "visual studio debugger. Set a breakpoint at the end of\n"
           "doGraphics and open the locals tab.\n";
}

int LWNNatVisTestC::isSupported()
{
    return true;
}

void LWNNatVisTestC::initGraphics()
{
    lwnDefaultInitGraphics();
}

void LWNNatVisTestC::doGraphics()
{
    size_t bytes = sizeof(NatVisObjectsC) + sizeof(void*);
    std::vector<unsigned char> memZeroes, memOnes;
    memZeroes.resize(bytes, '\0');
    memOnes.resize(bytes, '\xff');
    NatVisObjectsC* objectsZeroes = new((void*)&memZeroes[0]) NatVisObjectsC;
    NatVisObjectsC* objectsOnes = new((void*)&memOnes[0]) NatVisObjectsC;
    (void)objectsZeroes;
    (void)objectsOnes;

    // Set a breakpoint here
    objectsZeroes->~NatVisObjectsC();
    objectsOnes->~NatVisObjectsC();

    LWNTestClearAndFinish(LWNTEST_COLOR_PASS);
    return;
}

void LWNNatVisTestC::exitGraphics()
{
    lwnDefaultExitGraphics();
}

OGTEST_CppTest(LWNNatVisTestC, lwn_natvis_c, );
