/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwn/lwn_Cpp.h"
#include "cppogtest.h"
#include "lwn_utils_noapi.h"
#include <vector>

#include <lwnUtil/g_lwnObjectListCpp.h>

// disable ogtest.h hack that breaks placement new
#ifdef new
#undef new
#endif

class LWNNatVisTestCPP
{
public:
    OGTEST_CppMethods();
};

lwString LWNNatVisTestCPP::getDescription()
{
    return "Creates each CPP API object that has NatVis entries.\n"
           "Test requires manual inspection of each API type using the\n"
           "visual studio debugger. Set a breakpoint at the end of\n"
           "doGraphics and open the locals tab.\n";
}

int LWNNatVisTestCPP::isSupported()
{
    return true;
}

void LWNNatVisTestCPP::initGraphics()
{
    lwnDefaultInitGraphics();
}

void LWNNatVisTestCPP::doGraphics()
{
    size_t bytes = sizeof(NatVisObjectsCPP) + sizeof(void*);
    std::vector<unsigned char> memZeroes, memOnes;
    memZeroes.resize(bytes, '\0');
    memOnes.resize(bytes, '\xff');
    NatVisObjectsCPP* objectsZeroes = new((void*)&memZeroes[0]) NatVisObjectsCPP;
    NatVisObjectsCPP* objectsOnes = new((void*)&memOnes[0]) NatVisObjectsCPP;
    (void)objectsZeroes;
    (void)objectsOnes;

    // Set a breakpoint here
    objectsZeroes->~NatVisObjectsCPP();
    objectsOnes->~NatVisObjectsCPP();
    
    LWNTestClearAndFinish(LWNTEST_COLOR_PASS);
    return;
}

void LWNNatVisTestCPP::exitGraphics()
{
    lwnDefaultExitGraphics();
}

OGTEST_CppTest(LWNNatVisTestCPP, lwn_natvis_cpp, );
