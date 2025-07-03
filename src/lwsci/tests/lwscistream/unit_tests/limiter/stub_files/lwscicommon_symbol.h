/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCICOMMON_SYMBOL_H
#define INCLUDED_LWSCICOMMON_SYMBOL_H

#ifdef __cplusplus
namespace LwSciCommon
{
/**
 * \brief definition of NULL pointer
 *
 */
constexpr auto NULL_PTR = nullptr;
}
#else // __cplusplus

/**
 * \brief definition of NULL pointer
 *
 */
#define NULL_PTR NULL

#endif // __cplusplus

#endif // INCLUDED_LWSCICOMMON_SYMBOL_H
