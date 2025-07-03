// This code contains LWPU Confidential Information and is disclosed to you
// under a form of LWPU software license agreement provided separately to you.
//
// Notice
// LWPU Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from LWPU Corporation is strictly prohibited.
//
// ALL LWPU DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". LWPU MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, LWPU Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of LWPU Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// LWPU Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// LWPU Corporation.
//
// Copyright 2015 LWPU Corporation. All rights reserved.

#pragma once

#if defined(__ANDROID__) || defined(__GNUC__)
    #ifdef ANSEL_SDK_EXPORTS
    #define ANSEL_SDK_API extern "C" __attribute__ ((visibility ("default")))
    #define ANSEL_SDK_CLASS_API  __attribute__ ((visibility ("default")))
    #else
    #define ANSEL_SDK_API 
    #define ANSEL_SDK_CLASS_API
    #endif
#else
    #ifdef ANSEL_SDK_EXPORTS
    #define ANSEL_SDK_API extern "C" __declspec(dllexport)
    #define ANSEL_SDK_CLASS_API __declspec(dllexport)
    #else
    #define ANSEL_SDK_API extern "C" __declspec(dllimport)
    #define ANSEL_SDK_CLASS_API __declspec(dllimport)
    #endif
#endif
