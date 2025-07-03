/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __readpixels_h
#define __readpixels_h

void CreateFramebufferBuffer(void);
void DestroyFramebufferBuffer(void);
void GetFramebufferData(unsigned char *out);

#endif // __readpixels_h

