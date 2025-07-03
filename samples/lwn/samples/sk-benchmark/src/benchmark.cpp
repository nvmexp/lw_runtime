/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/
/////////////////////////////////////////////////////////////////////////////
//
// Loads and draws model data to mesure performance
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

#define NUM_PATTERN 4
#define NUM_BLEND_PATTERN 10

#define LOOPS_INFINITE	(-1)

enum TestMode
{
    LOW_QUALITY = 0,
    NORMAL_QUALITY,
    HIGH_QUALITY,
    NUM_MODE,
};

enum BlendMode
{
    NO_BLEND = 0,
    LOW_BLEND,
    HIGH_BLEND,
    NUM_BLEND_MODE,
};

static TestMode s_meshMode = HIGH_QUALITY;
static TestMode s_texMode = HIGH_QUALITY;
static BlendMode s_blendMode = HIGH_BLEND;

//---------------------------------------------------------------------------*
//  Model Data
//---------------------------------------------------------------------------*/

#include "../data/assets/geometries/elephantPosition.dat.h"
#include "../data/assets/geometries/elephantNormal.dat.h"
#include "../data/assets/geometries/elephantTexCoord.dat.h"
#include "../data/assets/geometries/elephantIdx.dat.h"
#include "../data/assets/geometries/teapotPosition.dat.h"
#include "../data/assets/geometries/teapotNormal.dat.h"
#include "../data/assets/geometries/teapotTexCoord.dat.h"
#include "../data/assets/geometries/teapotIdx.dat.h"
#include "../data/assets/geometries/torusPosition.dat.h"
#include "../data/assets/geometries/torusNormal.dat.h"
#include "../data/assets/geometries/torusTexCoord.dat.h"
#include "../data/assets/geometries/torusIdx.dat.h"
#include "../data/assets/geometries/spherePosition.dat.h"
#include "../data/assets/geometries/sphereNormal.dat.h"
#include "../data/assets/geometries/sphereTexCoord.dat.h"
#include "../data/assets/geometries/sphereIdx.dat.h"

#define NUM_MODELS 4
static u8* MODEL_FILES[NUM_MODELS][4] =
{
    {elephantPosition_dat,
     elephantNormal_dat,
     elephantTexCoord_dat,
     elephantIdx_dat},
    {teapotPosition_dat,
     teapotNormal_dat,
     teapotTexCoord_dat,
     teapotIdx_dat},
    {torusPosition_dat,
     torusNormal_dat,
     torusTexCoord_dat,
     torusIdx_dat},
    {spherePosition_dat,
     sphereNormal_dat,
     sphereTexCoord_dat,
     sphereIdx_dat}
};
static u32 MODEL_FILES_LEN[NUM_MODELS][4] =
{
    {elephantPosition_dat_len,
     elephantNormal_dat_len,
     elephantTexCoord_dat_len,
     elephantIdx_dat_len},
    {teapotPosition_dat_len,
     teapotNormal_dat_len,
     teapotTexCoord_dat_len,
     teapotIdx_dat_len},
    {torusPosition_dat_len,
     torusNormal_dat_len,
     torusTexCoord_dat_len,
     torusIdx_dat_len},
    {spherePosition_dat_len,
     sphereNormal_dat_len,
     sphereTexCoord_dat_len,
     sphereIdx_dat_len}
};

struct Texture
{
    Texture(u32 _width, u32 _height) : width(_width), height(_height)
    {
        size = width * height * 4;
    }

    u32* data;
    u32 width;
    u32 height;
    u32 size;
};

Texture checkerboard_256x256(256, 256);
Texture checkerboard_512x512(512, 512);
Texture checkerboard_1024x1024(1024, 1024);

// [TODO] Support BC1 textures
#define NUM_IMAGES 12
static Texture* IMAGE_FILES[NUM_IMAGES] =
{
    &checkerboard_256x256,
    &checkerboard_256x256,
    &checkerboard_256x256,
    &checkerboard_512x512,
    &checkerboard_512x512,
    &checkerboard_512x512,
    &checkerboard_512x512,
    &checkerboard_1024x1024,
    &checkerboard_1024x1024,
    &checkerboard_1024x1024,
    &checkerboard_1024x1024,
    &checkerboard_1024x1024,
};

#define NUM_IDX_PLANE 4

static const DEMO_F32x3 s_planePos[NUM_IDX_PLANE] =
    {{{{-100.0f, -1000.0, -1000.0f}}},
     {{{-100.0f, -1000.0,  1000.0f}}},
     {{{-100.0f,  1000.0,  1000.0f}}},
     {{{-100.0f,  1000.0, -1000.0f}}}};

static const DEMO_F32x3 s_planeNormal[NUM_IDX_PLANE] =
    {{{{0.0f,  1.0,  1.0f}}},
     {{{0.0f,  1.0,  1.0f}}},
     {{{0.0f,  1.0,  1.0f}}},
     {{{0.0f,  1.0,  1.0f}}}};

static const DEMO_F32x2 s_planeTexCoord[NUM_IDX_PLANE] =
    {{{{-5.0f, -5.0f}}},
     {{{-5.0f,  5.0f}}},
     {{{ 5.0f,  5.0f}}},
     {{{ 5.0f, -5.0f}}}};

static const u32 s_planeIndex[NUM_IDX_PLANE] = {0, 1, 2, 3};

#define TEX_PLANE_WIDTH  2
#define TEX_PLANE_HEIGHT 2
static u8  s_planeTexData[TEX_PLANE_WIDTH*TEX_PLANE_HEIGHT] = {0, 1, 1, 0};

//---------------------------------------------------------------------------*
// Shaders
//---------------------------------------------------------------------------*/

// Shader for Models

static DEMOGfxShader s_shader;

static const char *s_vsStringLq =
                            "#version 440 compatibility\n"
                            "layout(location = 0) attribute vec3 a_position;"
                            "layout(location = 1) attribute vec3 a_normal;"
                            "layout(location = 2) attribute vec2 a_texCoord;"
                            ""
                            "layout(binding = 0) uniform BlockVS {"
                            "    mat4 u_modelMtx;"
                            "    mat4 u_viewMtx;"
                            "    mat4 u_projMtx;"
                            "};"
                            ""
                            "out vec3 v_normalVec;"
                            "out vec2 v_texCoord1;"
                               "out vec2 v_texCoord2;"
                            ""
                            "void main()"
                            "{"
                                "v_normalVec = a_normal;"
                                ""
                                "v_texCoord1 = a_texCoord;"
                                "v_texCoord2 = a_texCoord * 2.0;"
                                ""
                                "gl_Position = vec4(a_position, 1.0) * u_modelMtx * u_viewMtx * u_projMtx;"
                            "}";

static const char *s_psStringLq =
                            "#version 440 core\n"
                            "in vec3 v_normalVec;"
                            ""
                            "in vec2 v_texCoord1;"
                            "in vec2 v_texCoord2;"
                            ""
                            "out vec4 out_color;"
                            ""
                            "layout(binding = 0) uniform sampler2D s_texture1;"
                            "layout(binding = 1) uniform sampler2D s_texture2;"
                            ""
                            "void main()"
                            "{"
                                "out_color  = (vec4(1.0) - texture2D( s_texture1, v_texCoord1 ));"
                                "out_color *= (vec4(1.0) - texture2D( s_texture2, v_texCoord2 ));"
                                "out_color.rgb *= v_normalVec;"
                                "out_color.a = 0.2;"
                            "}";

static const char *s_vsStringNq =
                            "#version 440 compatibility\n"
                            "layout(location = 0) attribute vec3 a_position;"
                            "layout(location = 1) attribute vec3 a_normal;"
                            "layout(location = 2) attribute vec2 a_texCoord;"
                            ""
                            "layout(binding = 0) uniform BlockVS {"
                            "    mat4 u_modelMtx;"
                            "    mat4 u_viewMtx;"
                            "    mat4 u_projMtx;"
                            "};"
                            ""
                            "out vec3 v_normalVec;"
                            "out vec2 v_texCoord1;"
                               "out vec2 v_texCoord2;"
                               "out vec2 v_texCoord3;"
                            ""
                            "void main()"
                            "{"
                                "v_normalVec = a_normal;"
                                ""
                                "v_texCoord1 = a_texCoord;"
                                "v_texCoord2 = a_texCoord * 2.0;"
                                "v_texCoord3 = a_texCoord * 4.0;"
                                ""
                                "gl_Position = vec4(a_position, 1.0) * u_modelMtx * u_viewMtx * u_projMtx;"
                            "}";

static const char *s_psStringNq =
                            "#version 440 core\n"
                            "in vec3 v_normalVec;"
                            ""
                            "in vec2 v_texCoord1;"
                            "in vec2 v_texCoord2;"
                            "in vec2 v_texCoord3;"
                            ""
                            "out vec4 out_color;"
                            ""
                            "layout(binding = 0) uniform sampler2D s_texture1;"
                            "layout(binding = 1) uniform sampler2D s_texture2;"
                            "layout(binding = 2) uniform sampler2D s_texture3;"
                            ""
                            "void main()"
                            "{"
                                "out_color  = (vec4(1.0) - texture2D( s_texture1, v_texCoord1 ));"
                                "out_color *= (vec4(1.0) - texture2D( s_texture2, v_texCoord2 ));"
                                "out_color *= (vec4(1.0) - texture2D( s_texture3, v_texCoord3 ));"
                                "out_color.rgb *= v_normalVec;"
                                "out_color.a = 0.2;"
                            "}";

static const char *s_vsStringHq =
                            "#version 440 compatibility\n"
                            "layout(location = 0) attribute vec3 a_position;"
                            "layout(location = 1) attribute vec3 a_normal;"
                            "layout(location = 2) attribute vec2 a_texCoord;"
                            ""
                            "layout(binding = 0) uniform BlockVS {"
                            "    mat4 u_modelMtx;"
                            "    mat4 u_viewMtx;"
                            "    mat4 u_projMtx;"
                            "};"
                            ""
                            "out vec3 v_normalVec;"
                            "out vec2 v_texCoord1;"
                               "out vec2 v_texCoord2;"
                               "out vec2 v_texCoord3;"
                            "out vec2 v_texCoord4;"
                            ""
                            "void main()"
                            "{"
                                "v_normalVec = a_normal;"
                                ""
                                "v_texCoord1 = a_texCoord;"
                                "v_texCoord2 = a_texCoord * 2.0;"
                                "v_texCoord3 = a_texCoord * 4.0;"
                                "v_texCoord4 = a_texCoord * 8.0;"
                                ""
                                "gl_Position = vec4(a_position, 1.0) * u_modelMtx * u_viewMtx * u_projMtx;"
                            "}";

static const char *s_psStringHq =
                            "#version 440 core\n"
                            "in vec3 v_normalVec;"
                            ""
                            "in vec2 v_texCoord1;"
                            "in vec2 v_texCoord2;"
                            "in vec2 v_texCoord3;"
                            "in vec2 v_texCoord4;"
                            ""
                            "out vec4 out_color;"
                            ""
                            "layout(binding = 0) uniform sampler2D s_texture1;"
                            "layout(binding = 1) uniform sampler2D s_texture2;"
                            "layout(binding = 2) uniform sampler2D s_texture3;"
                            "layout(binding = 3) uniform sampler2D s_texture4;"
                            ""
                            "void main()"
                            "{"
                                "out_color  = (vec4(1.0) - texture2D( s_texture1, v_texCoord1 ));"
                                "out_color *= (vec4(1.0) - texture2D( s_texture2, v_texCoord2 ));"
                                "out_color *= (vec4(1.0) - texture2D( s_texture3, v_texCoord3 ));"
                                "out_color *= (vec4(1.0) - texture2D( s_texture4, v_texCoord4 ));"
                                "out_color.rgb *= v_normalVec;"
                                "out_color.a = 0.2;"
                            "}";

// Shader for Back Plane

static const char *s_vsStringPlane =
                            "#version 440 compatibility\n"
                            "layout(location = 0) attribute vec3 a_position;"
                            "layout(location = 1) attribute vec3 a_normal;"
                            "layout(location = 2) attribute vec2 a_texCoord;"
                            "layout(binding = 0) uniform BlockVS {"
                            "    mat4 u_modelMtx;"
                            "    mat4 u_viewMtx;"
                            "    mat4 u_projMtx;"
                            "};"
                            ""
                            "out vec2 v_texCoord;"
                            ""
                            "void main()"
                            "{"
                                "v_texCoord = a_texCoord;"
                                ""
                                "gl_Position = vec4(a_position, 1.0) * u_modelMtx * u_viewMtx * u_projMtx;"
                            "}";

static const char *s_psStringPlane =
                            "#version 440 core\n"
                            ""
                            "in vec2 v_texCoord;"
                            ""
                            "out vec4 out_color;"
                            ""
                            "layout(binding = 0) uniform sampler2D s_texture;"
                            ""
                            "void main()"
                            "{"
                                "out_color.rgb = vec3(0.4, 0.6, 0.9);"
                                "out_color.a   = max(texture2D( s_texture, v_texCoord ).r, 0.5);"
                            "}";

// Matricies
static Mtx44 s_modelMtx44 = {{0}};
static Mtx44 s_identityMtx44 = {{0}};
static Mtx44 s_viewMtx44  = {{0}};
static Mtx44 s_projMtx44  = {{0}};

// Globals
static s32 s_numLoops = LOOPS_INFINITE;
static u32 s_frames   = 0;
static u32 s_modelId  = 0;
static u32 s_imageId  = 0;
static f32 s_yrad     = 0.0f;
static f32 s_xrad     = 0.0f;

#define MAX_RING 30
#define MAX_MODELS 256
static u32 s_numLwrRing = 7;
static const u32 s_numMaxRing[MAX_RING] = {
    1, 4, 8, 12, 16, 20, 24, 28, 32, 36 ,40, 44, 48, 52, 56, 60,
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116};

#define CAMERA_ROTATION_THRESHOLD (80.0f / 180.0f * 3.14159265f)
static f32 s_cameraDistance = 1000.0f;
static f32 s_cameraAngle = CAMERA_ROTATION_THRESHOLD;
static f32 s_objDistance = 80.0f;

#define STICK_SCALE 2000.0f
#define INC_ROTX  (f32)6.0e-3
#define INC_ROTY  (f32)3.0e-3

static u32 s_meshPattern[NUM_MODE][NUM_PATTERN]
    = {{2, 3, 2, 3},
       {1, 0, 2, 3},
       {1, 0, 1, 0}};
static u32 s_texPattern[NUM_MODE][NUM_PATTERN]
    = {{0, 1, 2, 3},
       {4, 5, 6, 7},
       {8, 9,10,11}};

static int s_blendPatterns[NUM_BLEND_MODE][NUM_BLEND_PATTERN]
    = {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
       {0, -1, -1, -1, -1, -1, -1, 1, -1, -1},
       {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}};

// Data file buffers
static DEMO_U8*  s_pTexture[NUM_IMAGES];
static DEMO_F32* s_pVertex[NUM_MODELS];
static DEMO_F32* s_pNormal[NUM_MODELS];
static DEMO_F32* s_pTexCoord[NUM_MODELS];
static DEMO_U32* s_pIndex[NUM_MODELS];

// Data size (Model + Plane)
static u32 s_indexNum[NUM_MODELS + 1]     = {0};
static u32 s_vertexSize[NUM_MODELS + 1]   = {0};
static u32 s_normalSize[NUM_MODELS + 1]   = {0};
static u32 s_texCoordSize[NUM_MODELS + 1] = {0};

// Data structures
static DEMOGfxVertexData  s_vertexData[NUM_MODELS + 1];
static DEMOGfxIndexData   s_indexData[NUM_MODELS + 1];

static DEMOGfxUniformData s_uniformDataVS[MAX_MODELS];
static DEMOGfxTexture     s_texture[NUM_IMAGES+1];

// Plane Shader
static DEMOGfxShader s_planeShader;

////////////////////////////////////////////////////
//
// Prototypes
//
////////////////////////////////////////////////////

static void ShaderInit(void);
static void VertexDataInit(void);
static void CameraInit(Mtx44 results_projMtx44, Mtx44 results_viewMtx44);
static void TextureDataInit(void);
static void UniformDataInit(void);
static int  SceneInit(void);
static int  SceneDraw(void);
static void SceneShutdown(void);
static void ModelRotateTick(Mtx44 resultMtx44);
static void AnimTick(void);

////////////////////////////////////////////////////
//
// Functions
//
////////////////////////////////////////////////////

static void ShaderInit(void)
{
    // Set up shader
    switch (s_texMode)
    {
    case LOW_QUALITY:
        DEMOGfxCreateShaders(&s_shader, s_vsStringLq, s_psStringLq);
        break;
    case NORMAL_QUALITY:
        DEMOGfxCreateShaders(&s_shader, s_vsStringNq, s_psStringNq);
        break;
    case HIGH_QUALITY:
        DEMOGfxCreateShaders(&s_shader, s_vsStringHq, s_psStringHq);
        break;
    default:
        assert(0);
        break;
    }

    // Set up vertex attributes
    DEMOGfxShaderAttributeData posAttrib      = {0, LWN_FORMAT_RGB32F, 0, 0, sizeof(DEMO_F32x3)};
    DEMOGfxShaderAttributeData normalAttrib   = {1, LWN_FORMAT_RGB32F, 0, 1, sizeof(DEMO_F32x3)};
    DEMOGfxShaderAttributeData texCoordAttrib = {2, LWN_FORMAT_RG32F, 0, 2, sizeof(DEMO_F32x2)};

    DEMOGfxSetShaderAttribute(&s_shader, &posAttrib);
    DEMOGfxSetShaderAttribute(&s_shader, &normalAttrib);
    DEMOGfxSetShaderAttribute(&s_shader, &texCoordAttrib);

    // Set up shader
    DEMOGfxCreateShaders(&s_planeShader, s_vsStringPlane, s_psStringPlane);
    DEMOGfxSetShaderAttribute(&s_planeShader, &posAttrib);
    DEMOGfxSetShaderAttribute(&s_planeShader, &normalAttrib);
    DEMOGfxSetShaderAttribute(&s_planeShader, &texCoordAttrib);
}

static void VertexDataInit(void)
{
    // Set up vertex data
    u32 indexSize = 0;
    u32 i = 0;

    for(i = 0; i < NUM_MODELS; i++)
    {
        s_pVertex[i]      = (DEMO_F32*)MODEL_FILES[i][0];
        s_vertexSize[i]   = MODEL_FILES_LEN[i][0];
        s_pNormal[i]      = (DEMO_F32*)MODEL_FILES[i][1];
        s_normalSize[i]   = MODEL_FILES_LEN[i][1];
        s_pTexCoord[i]    = (DEMO_F32*)MODEL_FILES[i][2];
        s_texCoordSize[i] = MODEL_FILES_LEN[i][2];
        s_pIndex[i]       = (DEMO_U32*)MODEL_FILES[i][3];
        indexSize         = MODEL_FILES_LEN[i][3];

        s_indexNum[i] = indexSize / sizeof(DEMO_U32);

        // Create vertex buffer
        DEMOGfxCreateVertexBuffer(&s_vertexData[i], s_vertexSize[i] + s_normalSize[i] + s_texCoordSize[i]);

        u32 normalOffset   = s_vertexSize[i];
        u32 texCoordOffset = s_vertexSize[i] + s_normalSize[i];

        // Fill vertex buffer with vertex data
        DEMOGfxSetVertexBuffer(&s_vertexData[i], s_pVertex[i],   0,                  s_vertexSize[i]);
        DEMOGfxSetVertexBuffer(&s_vertexData[i], s_pNormal[i],   normalOffset,    s_normalSize[i]);
        DEMOGfxSetVertexBuffer(&s_vertexData[i], s_pTexCoord[i], texCoordOffset , s_texCoordSize[i]);

        // Set up index buffer
        DEMOGfxCreateIndexBuffer(&s_indexData[i], indexSize);

        // Fill index buffer with index data
        DEMOGfxSetIndexBuffer(&s_indexData[i], s_pIndex[i], 0, indexSize);
    }

    //
    // For Plane
    //
    s_vertexSize[NUM_MODELS]   = sizeof(s_planePos);
    s_normalSize[NUM_MODELS]   = sizeof(s_planeNormal);
    s_texCoordSize[NUM_MODELS] = sizeof(s_planeTexCoord);

    // Create vertex buffer
    DEMOGfxCreateVertexBuffer(&s_vertexData[NUM_MODELS],  s_vertexSize[NUM_MODELS] + s_normalSize[NUM_MODELS] + s_texCoordSize[NUM_MODELS]);

    // Fill vertex buffer with vertex data
    DEMOGfxSetVertexBuffer(&s_vertexData[NUM_MODELS], s_planePos,                            0,                             s_vertexSize[NUM_MODELS]);
    DEMOGfxSetVertexBuffer(&s_vertexData[NUM_MODELS], s_planeNormal,   s_vertexSize[NUM_MODELS],                            s_normalSize[NUM_MODELS]);
    DEMOGfxSetVertexBuffer(&s_vertexData[NUM_MODELS], s_planeTexCoord, s_vertexSize[NUM_MODELS] + s_normalSize[NUM_MODELS], s_texCoordSize[NUM_MODELS]);

    // Set up index buffer
    DEMOGfxCreateIndexBuffer(&s_indexData[NUM_MODELS], sizeof(s_planeIndex));

    // Fill index buffer with index data
    DEMOGfxSetIndexBuffer(&s_indexData[NUM_MODELS], s_planeIndex, 0, sizeof(s_planeIndex));

     s_indexNum[NUM_MODELS] = sizeof(s_planeIndex) / sizeof(DEMO_U32);
}

// Init function for setting projection matrix
static void CameraInit(Mtx44 results_projMtx44, Mtx44 results_viewMtx44)
{
    // row major matricies
    Mtx   lookAtMtx34;

    Vec     up = {0.0f,  1.0f, 0.0f};
    Vec  objPt = {0.0f, 0.0f, 0.0f};
    Vec camLoc = {1000.0f, 0.0f, 0.0f};

    f32   pers = 35.0f;
    f32 aspect = (f32)DEMOGfxOffscreenWidth / (f32)DEMOGfxOffscreenHeight;
    f32  znear = 50.0f;
    f32   zfar = 2000.0f;


    //s_cameraAngle += (f32)padInfo.stickY / STICK_SCALE;

    s_cameraAngle = CAMERA_ROTATION_THRESHOLD;

    if(s_cameraAngle > CAMERA_ROTATION_THRESHOLD)
    {
        s_cameraAngle = CAMERA_ROTATION_THRESHOLD;
    }

    if(s_cameraAngle < -CAMERA_ROTATION_THRESHOLD)
    {
        s_cameraAngle = -CAMERA_ROTATION_THRESHOLD;
    }

    camLoc.x = s_cameraDistance*cosf(s_cameraAngle);
    camLoc.y = s_cameraDistance*sinf(s_cameraAngle);

    if(camLoc.x < 0.0f)
    {
        up.y = -1.0f;
    }

    // Compute perspective matrix
    MTXPerspective(results_projMtx44, pers, aspect, znear, zfar);

    // Compute lookAt matrix
    MTXLookAt(lookAtMtx34, &camLoc, &up, &objPt);
    MTX34To44(lookAtMtx34, results_viewMtx44);
}

// Update Animation
static void AnimTick(void)
{
    s_frames++;
}

// Update Model function for setting model matrix
static void ModelRotateTick(Mtx44 resultMtx44)
{
   // row major matrix
    Mtx  rotXMtx34;
    Mtx  rotYMtx34;
    Mtx  modelMtx34;

    // Compute rotation matrix
    MTXRotRad(rotYMtx34, 'y', s_yrad * 2);
    MTXRotRad(rotXMtx34, 'x', s_xrad * 2);

    // Compute model matrix
    MTXConcat(rotXMtx34, rotYMtx34, modelMtx34);
    MTX34To44(modelMtx34, resultMtx44);

    if(s_yrad + INC_ROTY > 1.0e+6)
    {
        s_yrad = -1.0e+6;
    }

    if(s_xrad + INC_ROTX > 1.0e+6)
    {
        s_xrad = -1.0e+6;
    }

    s_yrad += INC_ROTY;
    s_xrad += INC_ROTX;
}

// Update Model function for setting model matrix
static void ModelTransformTick(Mtx44 resultMtx44, f32 offsetX, f32 offsetY, f32 offsetZ)
{
    resultMtx44[0][3] = offsetX;
    resultMtx44[1][3] = offsetY;
    resultMtx44[2][3] = offsetZ;
}

// Make 32 pixel blocks
static void CheckerboardInit(Texture* texture)
{
    texture->data = (u32*)DEMOAlloc(texture->size);

    for (u32 x = 0; x < texture->width; ++x)
    {
        for (u32 y = 0; y < texture->height; ++y)
        {
            texture->data[x + y * texture->width] = 0xFF000000 + 0x00FFFFFF * ((x / 32 + y / 32) & 1);
        }
    }
}

static void TextureDataInit(void)
{
    CheckerboardInit(&checkerboard_256x256);
    CheckerboardInit(&checkerboard_512x512);
    CheckerboardInit(&checkerboard_1024x1024);

    // Load Texture data
    for(u32 i = 0; i < NUM_IMAGES; i++)
    {
        s_pTexture[i] = (DEMO_U8*)IMAGE_FILES[i]->data;

        // Set up texture buffer
        DEMOGfxCreateTextureBuffer(&s_texture[i], LWN_TEXTURE_TARGET_2D, 1, LWN_FORMAT_RGBA8, IMAGE_FILES[i]->width, IMAGE_FILES[i]->height, 1, 0, IMAGE_FILES[i]->size);

        // Fill texture buffer with texture data
        DEMOGfxSetTextureBuffer(&s_texture[i], 0, (const void*)s_pTexture[i], 0, 0, 0, 0, IMAGE_FILES[i]->width, IMAGE_FILES[i]->height, 1, IMAGE_FILES[i]->size);
    }

    // For Plane
    // Set up texture buffer
    DEMOGfxCreateTextureBuffer(&s_texture[NUM_IMAGES], LWN_TEXTURE_TARGET_2D, 1, LWN_FORMAT_R8, TEX_PLANE_WIDTH, TEX_PLANE_HEIGHT, 1, 0, sizeof(s_planeTexData));

    // Fill texture buffer with texture data
    DEMOGfxSetTextureBuffer(&s_texture[NUM_IMAGES], 0, (const void*)s_planeTexData, 0, 0, 0, 0, TEX_PLANE_WIDTH, TEX_PLANE_HEIGHT, 1, sizeof(s_planeTexData));
}

static void UniformDataInit(void)
{
    // Set up camera
    CameraInit(s_projMtx44, s_viewMtx44);

    // Set up model mtx
    MTX44Identity(s_modelMtx44);
    MTX44Identity(s_identityMtx44);

    for(u32 i=0; i < MAX_MODELS; i++)
    {
        DEMOGfxCreateUniformBuffer(&s_uniformDataVS[i], sizeof(Mtx44) * 3);

        // Fill buffer with constant uniform data
        DEMOGfxSetUniformBuffer(&s_uniformDataVS[i], s_modelMtx44, 0,                 sizeof(Mtx44));
        DEMOGfxSetUniformBuffer(&s_uniformDataVS[i], s_viewMtx44,  sizeof(Mtx44),     sizeof(Mtx44));
        DEMOGfxSetUniformBuffer(&s_uniformDataVS[i], s_projMtx44,  sizeof(Mtx44) * 2, sizeof(Mtx44));
    }
}

// The init function for the rendering portions of this app
static int SceneInit(void)
{
    //DEMOPrintf("Mesh Quality: %i  Texture Quality: %i\n", s_meshMode, s_texMode);

    // Init Shader
    ShaderInit();

    // Init Vertex Data
    VertexDataInit();

    // Init Texture Data
    TextureDataInit();

    // Init Uniform Data
    UniformDataInit();

    return 1;
}

static void SceneShutdown(void)
{
    for(u32 i = 0; i < NUM_MODELS + 1; i++)
    {
        DEMOGfxReleaseVertexBuffer(&s_vertexData[i]);
        DEMOGfxReleaseIndexBuffer(&s_indexData[i]);
    }

    for(u32 i = 0; i < MAX_MODELS; i++)
    {
        DEMOGfxReleaseUniformBuffer(&s_uniformDataVS[i]);
    }

    for(u32 i = 0; i < NUM_IMAGES; i++)
    {
        DEMOGfxReleaseTextureBuffer(&s_texture[i]);
    }

    DEMOFree(checkerboard_256x256.data);
    DEMOFree(checkerboard_512x512.data);
    DEMOFree(checkerboard_1024x1024.data);

    DEMOGfxReleaseShaders(&s_shader);
    DEMOGfxReleaseShaders(&s_planeShader);

    DEMOGfxReleaseTextureBuffer(&s_texture[NUM_IMAGES]);

}

// The draw function for the rendering portions of this app
static int SceneDraw(void)
{
    u32 uboId = 0;
    u32 texId = 0;
    u32 i,j = 0;
    u32 lwrBlend = 0;

    DEMOGfxBeforeRender();

    // Clear buffers
    DEMOGfxClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    DEMOGfxClearDepthStencil(1.0f, 0);

    // Bind the plane pipeline
    DEMOGfxSetShaders(&s_planeShader);

    // Fill uniform buffer with uniform data
    DEMOGfxSetUniformBuffer(&s_uniformDataVS[uboId], s_identityMtx44, 0, sizeof(Mtx44));

    // Bind uniform buffer
    DEMOGfxBindUniformBuffer(&s_uniformDataVS[uboId], LWN_SHADER_STAGE_VERTEX, 0, 0, sizeof(Mtx44) * 3);

    // Bind vertex buffers for plane
    u32 normalOffset   = s_vertexSize[NUM_MODELS];
    u32 texCoordOffset = s_vertexSize[NUM_MODELS] + s_normalSize[NUM_MODELS];

    DEMOGfxBindVertexBuffer(&s_vertexData[NUM_MODELS], 0,            0, s_vertexSize[NUM_MODELS]);
    DEMOGfxBindVertexBuffer(&s_vertexData[NUM_MODELS], 1, normalOffset, s_normalSize[NUM_MODELS]);
    DEMOGfxBindVertexBuffer(&s_vertexData[NUM_MODELS], 2, texCoordOffset, s_texCoordSize[NUM_MODELS]);

    // Bind texture
    DEMOGfxBindTextureBuffer(&s_texture[NUM_IMAGES], LWN_SHADER_STAGE_FRAGMENT, 0);

    // Draw Plane
    DEMOGfxDrawElements(&s_indexData[NUM_MODELS], LWN_DRAW_PRIMITIVE_QUADS, LWN_INDEX_TYPE_UNSIGNED_INT, s_indexNum[NUM_MODELS], 0);

    uboId++;

    // Set the shaders
    DEMOGfxSetShaders(&s_shader);

    // Update Model Matrix Uniform
    ModelRotateTick(s_modelMtx44);

    for(i=0; i<s_numLwrRing; i++)
    {
        f32 radius = i * s_objDistance;

        for(j=0; j<s_numMaxRing[i]; j++)
        {
            f32 angle = 360.0f / s_numMaxRing[i] * j;
            f32 x = radius * cosf(angle / 180.0f * 3.1415f);
            f32 y = radius * sinf(angle / 180.0f * 3.1415f);

            Mtx44 copyMtx;
            MTX44Copy(s_modelMtx44, copyMtx);

            // Update Model Matrix Uniform
            ModelTransformTick(copyMtx,  0.0f, y, x);

            // Enable Blend sometimes
            if (s_blendPatterns[s_blendMode][lwrBlend % NUM_BLEND_PATTERN] >= 0)
                DEMOGfxSetColorControl(0, 0, s_blendPatterns[s_blendMode][lwrBlend % NUM_BLEND_PATTERN], LWN_LOGIC_OP_COPY);
            ++lwrBlend;

            // Fill uniform buffer with uniform data
            DEMOGfxSetUniformBuffer(&s_uniformDataVS[uboId], copyMtx, 0, sizeof(Mtx44));

            // Bind uniform buffer
            DEMOGfxBindUniformBuffer(&s_uniformDataVS[uboId], LWN_SHADER_STAGE_VERTEX, 0, 0, sizeof(Mtx44) * 3);

            uboId++;

            if(uboId > MAX_MODELS - 1)
            {
                goto out;
            }

            // Bind texture buffer
            s_imageId = s_texPattern[s_texMode][texId % NUM_PATTERN];
            DEMOGfxBindTextureBuffer(&s_texture[s_imageId], LWN_SHADER_STAGE_FRAGMENT, 0);
            texId++;

            s_imageId = s_texPattern[s_texMode][texId % NUM_PATTERN];
            DEMOGfxBindTextureBuffer(&s_texture[s_imageId], LWN_SHADER_STAGE_FRAGMENT, 1);
            texId++;

            if((NORMAL_QUALITY == s_texMode) || (HIGH_QUALITY == s_texMode))
            {
                s_imageId = s_texPattern[s_texMode][texId % NUM_PATTERN];
                DEMOGfxBindTextureBuffer(&s_texture[s_imageId], LWN_SHADER_STAGE_FRAGMENT, 2);
                texId++;
            }

            if(HIGH_QUALITY == s_texMode)
            {
                s_imageId = s_texPattern[s_texMode][texId % NUM_PATTERN];
                DEMOGfxBindTextureBuffer(&s_texture[s_imageId], LWN_SHADER_STAGE_FRAGMENT, 3);
                texId++;
            }

            // Bind vertex buffers
            s_modelId = s_meshPattern[s_meshMode][i % NUM_PATTERN];

            u32 normalOffset   = s_vertexSize[s_modelId];
            u32 texCoordOffset = s_vertexSize[s_modelId] + s_normalSize[s_modelId];

            DEMOGfxBindVertexBuffer(&s_vertexData[s_modelId], 0,            0, s_vertexSize[s_modelId]);
            DEMOGfxBindVertexBuffer(&s_vertexData[s_modelId], 1, normalOffset, s_normalSize[s_modelId]);
            DEMOGfxBindVertexBuffer(&s_vertexData[s_modelId], 2, texCoordOffset, s_texCoordSize[s_modelId]);

            // Draw
            DEMOGfxDrawElements(&s_indexData[s_modelId], LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT, s_indexNum[s_modelId], 0);
        }
    }
out:
    //PrintInfo();
    DEMOGfxDoneRender();

    // Update Animation
    AnimTick();

    return 1;
}

////////////////////////////////////////////////////
//
// App Functions
//
////////////////////////////////////////////////////

struct Test
{
    int numRings;
    bool useMiniObjects;
    bool useCommandBuffer;
    bool useCBTransient;
    TestMode meshQuality;
    TestMode textureQuality;
    BlendMode blendMode;
};

int lwrTest = 0;
Test testList[] = {
    {4, true, false, false, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, true, false, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, true, true, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, false, false, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, true, false, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, true, true, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, false, false, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, true, false, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, true, true, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, false, false, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, true, false, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, true, true, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, false, false, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, true, false, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, true, true, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {4, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {4, true, false, false, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, true, false, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, true, true, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, false, false, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, true, false, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, true, true, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, false, false, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, true, false, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, true, true, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, false, false, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, true, false, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, true, true, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, false, false, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, true, false, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, true, true, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, false, false, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, true, false, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, true, true, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {7, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {7, true, false, false, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, true, false, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {7, true, true, true, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, false, false, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, true, false, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, true, true, LOW_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, false, false, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, true, false, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, true, true, LOW_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, false, false, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, true, false, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, true, true, LOW_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, false, false, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, true, false, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, true, true, NORMAL_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, false, false, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, true, false, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, true, true, HIGH_QUALITY, LOW_QUALITY, NO_BLEND},
    {10, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, NO_BLEND},
    {10, true, false, false, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, true, false, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {10, true, true, true, HIGH_QUALITY, HIGH_QUALITY, NO_BLEND},
    {4, true, false, false, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, true, false, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, true, true, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, false, false, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, true, false, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, true, true, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, false, false, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, true, false, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, true, true, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, false, false, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, true, false, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, true, true, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, false, false, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, true, false, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, true, true, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {4, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {4, true, false, false, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, true, false, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, true, true, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, false, false, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, true, false, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, true, true, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, false, false, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, true, false, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, true, true, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, false, false, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, true, false, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, true, true, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, false, false, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, true, false, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, true, true, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, false, false, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, true, false, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, true, true, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {7, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {7, true, false, false, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, true, false, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {7, true, true, true, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, false, false, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, true, false, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, true, true, LOW_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, false, false, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, true, false, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, true, true, LOW_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, false, false, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, true, false, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, true, true, LOW_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, false, false, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, true, false, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, true, true, NORMAL_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, false, false, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, true, false, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, true, true, HIGH_QUALITY, LOW_QUALITY, LOW_BLEND},
    {10, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, LOW_BLEND},
    {10, true, false, false, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, true, false, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {10, true, true, true, HIGH_QUALITY, HIGH_QUALITY, LOW_BLEND},
    {4, true, false, false, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, true, false, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, true, true, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, false, false, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, true, false, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, true, true, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, false, false, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {4, true, true, false, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {4, true, true, true, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {4, true, false, false, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, true, false, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, true, true, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {4, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {4, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {4, true, false, false, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, true, false, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, true, true, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {4, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {4, true, false, false, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {4, true, true, false, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {4, true, true, true, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, false, false, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, true, false, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, true, true, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, false, false, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, true, false, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, true, true, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, false, false, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, true, false, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, true, true, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, false, false, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, true, false, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, true, true, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, false, false, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, true, false, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, true, true, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {7, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {7, true, false, false, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, true, false, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {7, true, true, true, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, false, false, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, true, false, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, true, true, LOW_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, false, false, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, true, false, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, true, true, LOW_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, false, false, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, true, false, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, true, true, LOW_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, false, false, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, true, false, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, true, true, NORMAL_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, false, false, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, true, false, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, true, true, NORMAL_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, false, false, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, true, false, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, true, true, NORMAL_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, false, false, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, true, false, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, true, true, HIGH_QUALITY, LOW_QUALITY, HIGH_BLEND},
    {10, true, false, false, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, true, false, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, true, true, HIGH_QUALITY, NORMAL_QUALITY, HIGH_BLEND},
    {10, true, false, false, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, true, false, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
    {10, true, true, true, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND},
};
const int NUM_TESTS = sizeof(testList) / sizeof(*testList);

Test baseTest = {10, true, true, true, HIGH_QUALITY, HIGH_QUALITY, HIGH_BLEND};

static void SetTestParameters(const Test& test)
{
    //DEMOGfxSetUseLWNMiniObjects(test.useMiniObjects);
    DEMOGfxSetUseCommandBuffer(test.useCommandBuffer);
    DEMOGfxSetCommandBufferTransient(test.useCBTransient);
    s_numLwrRing = test.numRings;
    s_meshMode = test.meshQuality;
    s_texMode  = test.textureQuality;
    s_blendMode  = test.blendMode;
}

LWNnativeWindow s_nativeWindow;

bool appInit(int argc, char **argv, LWNnativeWindow nativeWindow)
{
#ifdef ENABLE_PERF
    lwrTest = 0;
    SetTestParameters(testList[lwrTest % NUM_TESTS]);
#else
    SetTestParameters(baseTest);
#endif
    int numPresentBuffers = 2;
    int presentInterval = 1;

    for (int i = 1; i < argc; ++i)  {
        if (strcmp(argv[i], "-n") == 0 && (i + 1) < argc) {
            s_numLoops = atol(argv[i + 1]);
        }
        if (strcmp(argv[i], "--num-buffers") == 0 && (i+1) < argc) {
            numPresentBuffers = atol(argv[i+1]);
        }
        if (strcmp(argv[i], "--present-interval") == 0 && (i+1) < argc) {
            presentInterval = atol(argv[i+1]);
        }
    }

    DEMOInit();
    s_nativeWindow = nativeWindow;
    DEMOGfxInit(argc, argv, s_nativeWindow, numPresentBuffers, presentInterval);

    SceneInit();

    return true;
}

void appReshape(int w, int h)
{

}

void appShutdown(void)
{
    SceneShutdown();

    //DEMOFontShutdown();
    DEMOGfxShutdown();
    DEMOShutdown();
}

bool appDisplay(void)
{
    SceneDraw();

#ifdef ENABLE_PERF
    if (s_frames == 2000)
    {
        DEMOGfxPrintLWNPerformace();
        s_frames = 0;

        if (s_numLoops != LOOPS_INFINITE) {
            --s_numLoops;
        }

#ifdef ENABLE_REINIT
        SceneShutdown();
        DEMOGfxShutdown();

        ++lwrTest;
        SetTestParameters(testList[lwrTest % NUM_TESTS]);

        DEMOGfxInit(0, NULL, s_nativeWindow);
        SceneInit();
#endif
    }
#endif
    // return true if we have more frames to render
    return (s_numLoops != 0);
}
