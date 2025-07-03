/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2018 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _LW_KERNEL_INTERFACE_API_H
#define _LW_KERNEL_INTERFACE_API_H
/**************************************************************************************************************
*
*    File:  lw-kernel-interface-api.h
*
*    Description:
*        Defines the LW API related macros. 
*
**************************************************************************************************************/

#if LWOS_IS_UNIX && LWCPU_IS_X86_64 && defined(__use_altstack__)
#define LW_API_CALL __attribute__((altstack(0)))
#else
#define LW_API_CALL
#endif

#endif /* _LW_KERNEL_INTERFACE_API_H */
