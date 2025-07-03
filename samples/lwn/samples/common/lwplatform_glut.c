/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwplatform.h"

#ifdef __ANDROID__
#include "GL/freeglut.h"
#elif _WIN32 || __linux__
#include "GL/glut.h"
#else
#include "noglut.h"
#include <stdlib.h>
#endif

#include <string.h>
#include <assert.h>

#ifdef _MSC_VER
// Force the LWpu GPU on Optimus systems
_declspec(dllexport) unsigned long LwOptimusEnablement = 0x00000001;
#endif

void __idleFunc(void);


int lwplatform_setupWindow(int w, int h)
{
    const char *dummy_argv[] = { "lwnsample", NULL };
    int   dummy_argc = 1;

    glutInit(&dummy_argc, (char **)dummy_argv);

#ifdef __ANDROID__
    /* Some features of LWN (temporarily) depend on having a modern GL context
    * bound in order to function correctly. This dependency will be removed in
    * the future, but for now, bind a GL 4.5 context at init. */
    eglBindAPI(EGL_OPENGL_API);
    glutInitContextVersion(4, 5);
#endif
    glutInitDisplayString("double depth rgba stencil");
    glutInitWindowSize(w, h);
    glutCreateWindow((char *)dummy_argv[0]);

    glutIdleFunc(__idleFunc);

#ifdef __ANDROID__
    // Make sure glue isn't stripped.
    app_dummy();
#endif

    //TODO errors
    return 1;
}

void lwplatform_mainLoop(void){
    glutMainLoop();
}

void lwplatform_displayFunc(void(*callback)(void)){
    glutDisplayFunc(callback);
}

void lwplatform_keyboardFunc(void(*callback)(unsigned char, int, int)){
    glutKeyboardFunc(callback);
}

void lwplatform_swapBuffers(void){
    glutSwapBuffers();
}

void* lwplatform_getWindowHandle()
{
    return NULL;
}

void __idleFunc(void){
    //always redisplay
    glutPostRedisplay();
}
