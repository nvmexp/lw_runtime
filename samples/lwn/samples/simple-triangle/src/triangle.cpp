/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/
/////////////////////////////////////////////////////////////////////////////
//
// Draws a color-interpolated triangle.
// Changes the background color to show its liveliness.
//
//////////////////////////////////////////////////////////////////////////////

#include <stdarg.h>
#include <demo.h>
#include "lwnUtil/lwnUtil_AlignedStorageImpl.h"
#include "lwnUtil/lwnUtil_PoolAllocatorImpl.h"
#include "lwnUtil/lwnUtil_GlslcHelperImpl.h"
#include <lwwinsys_app.h>

////////////////////////////////////////////////////
//
// Assets data, types and interface for demos
//
////////////////////////////////////////////////////

static const s32 LOOPS_INFINITE = -1;

static s32 s_numLoops = LOOPS_INFINITE;

// Position for a single triangle.
static const DEMO_F32x3 TRIANGLE_POSITION_DATA[] =
{
   {{{-0.75f,  0.75f, -0.5f}}},
   {{{ 0.75f,  0.75f, -0.5f}}},
   {{{ 0.00f, -0.75f, -0.5f}}}
};

// Color for a single triangle.
static const DEMO_F32x3 TRIANGLE_COLOR_DATA[] =
{
    {{{1.0f,  0.0f,  0.0f}}},
    {{{0.0f,  1.0f,  0.0f}}},
    {{{0.0f,  0.0f,  1.0f}}}
};

// Number of vertex
static const u32 TRIANGLE_VERTEX_NUM = 3;

static const char *s_vsString = 
							"attribute vec4 a_position;"
							"attribute vec4 a_color;\n"
							""
							"varying vec4 v_color;"
							""
							"void main()"
							"{"
							"    gl_Position = a_position;"
							"    v_color = a_color;"
							"}";

static const char *s_psString = 
							"varying vec4 v_color;"
							""
							"void main()"
							"{"
							"    gl_FragColor = v_color;"
							"}";

static DEMOGfxVertexData s_posData;
static DEMOGfxVertexData s_colorData;

static DEMOGfxShader s_shader;

////////////////////////////////////////////////////
//
// App Functions
//
////////////////////////////////////////////////////

LWNnativeWindow s_nativeWindow;

bool appInit(int argc, char **argv, LWNnativeWindow nativeWindow)
{
    for (int i = 1; i < argc; ++i)  {
        if (strcmp(argv[i], "-n") == 0 && (i + 1) < argc) {
            s_numLoops = atol(argv[i + 1]);
        }
    }

    s_nativeWindow = nativeWindow;;
    DEMOGfxInit(argc, argv, s_nativeWindow);

	// Set up shader
	DEMOGfxCreateShaders(&s_shader, s_vsString, s_psString);

    // Set up shader vertex attributes
	DEMOGfxShaderAttributeData posAttrib	 = {0, LWN_FORMAT_RGB32F, 0, 0, sizeof(DEMO_F32x3)};
	DEMOGfxShaderAttributeData colorAttrib   = {1, LWN_FORMAT_RGB32F, 0, 1, sizeof(DEMO_F32x3)};	
	
	DEMOGfxSetShaderAttribute(&s_shader, &posAttrib);
	DEMOGfxSetShaderAttribute(&s_shader, &colorAttrib);

	// Set up vertex buffer
	DEMOGfxCreateVertexBuffer(&s_posData,   sizeof(TRIANGLE_POSITION_DATA));
	DEMOGfxCreateVertexBuffer(&s_colorData, sizeof(TRIANGLE_COLOR_DATA));

    // Fill vertex buffer with vertex data
	DEMOGfxSetVertexBuffer(&s_posData,   TRIANGLE_POSITION_DATA, 0, sizeof(TRIANGLE_POSITION_DATA));
	DEMOGfxSetVertexBuffer(&s_colorData, TRIANGLE_COLOR_DATA,    0, sizeof(TRIANGLE_COLOR_DATA));

    // Bind vertex buffers
	DEMOGfxBindVertexBuffer(&s_posData,   0, 0, sizeof(TRIANGLE_POSITION_DATA));
	DEMOGfxBindVertexBuffer(&s_colorData, 1, 0, sizeof(TRIANGLE_COLOR_DATA));

    DEMOGfxSetShaders(&s_shader);

	return true;
}

void appReshape(int w, int h)
{

}

bool appDisplay()
{
	static f32 s_grey = 0.0f;

    s_grey += 0.01f;
    if (s_grey > 1.0) s_grey = 0.0f;

	DEMOGfxBeforeRender();

	// Clear buffers
	DEMOGfxClearColor(s_grey, s_grey, s_grey, 1.0f);
    DEMOGfxClearDepthStencil(1.0f, 0);

	// Draw
    DEMOGfxDrawArrays(LWN_DRAW_PRIMITIVE_TRIANGLES, 0, TRIANGLE_VERTEX_NUM);

	DEMOGfxDoneRender();

    if (s_numLoops != LOOPS_INFINITE) {
        --s_numLoops;
    }

    return (s_numLoops != 0);
}

void appShutdown(void)
{
	DEMOGfxReleaseVertexBuffer(&s_posData);
	DEMOGfxReleaseVertexBuffer(&s_colorData);
    DEMOGfxReleaseShaders(&s_shader);
    DEMOGfxShutdown();
}
