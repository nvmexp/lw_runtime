/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwplatform_h_
#define __lwplatform_h_

#ifdef __cplusplus
extern "C" {
#endif

int lwplatform_setupWindow(int w, int h);
void lwplatform_mainLoop(void);
void lwplatform_swapBuffers(void);

void* lwplatform_getWindowHandle(void);

void lwplatform_getProc(char const * procname);

void lwplatform_displayFunc(void(*callback)(void));
void lwplatform_keyboardFunc(void(*callback)(unsigned char, int, int));


#ifdef __cplusplus
}
#endif

#endif //__lwplatform_h_
