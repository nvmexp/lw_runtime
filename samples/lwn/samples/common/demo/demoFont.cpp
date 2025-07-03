/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/
////===========================================================================
///  demoFont.cpp
///
///     This is font code for the demo library.
///
////===========================================================================

#if defined(_MSC_VER)
#define _CRT_SELWRE_NO_WARNINGS
#endif

#include <stdarg.h>
#include <stdio.h>

#include <demo.h>
#include <demoFontData.h>

// --------------------------------------------------------------------------
//  Macro definitions
// --------------------------------------------------------------------------
// Max Buffer Size for vsnprintf_s
#define FONT_MAX_BUFSIZ 512

static const char *s_vsString  =  
                            "#version 440 compatibility\n"
                            "layout(location = 0) attribute vec3 a_Position;"
                            "layout(location = 1) attribute vec2 a_TexCoord;"
                            ""
                            "layout(binding = 0) uniform BlockVS {"
                            "   vec2 u_Scale;"
                            "};"
                            "varying vec2  v_TexCoord;"
                            ""
                            "void main()"
                            "{"
                            "  gl_Position.x  = a_Position.x * u_Scale.x - 1;"
                            "  gl_Position.y  = a_Position.y * u_Scale.y + 1;"
                            "  gl_Position.z  = a_Position.z;" 
                            "  gl_Position.w  = 1.0f;" 
                            "  v_TexCoord  = a_TexCoord;"
                            "}";

static const char *s_psString = 
                            "#version 440 compatibility\n"
                            "layout(binding = 0) uniform sampler2D u_CharTex;"
                            "varying vec2  v_TexCoord;                       "
                            ""
                            "layout(binding = 0) uniform BlockPS {"
                            "   vec4 u_Color;"
                            "};"
                            ""
                            "void main()                                    "
                            "{                                              "
                            "   vec4 texColor = texture2D(u_CharTex, v_TexCoord);\n"                            
                            "   texColor.a    = smoothstep(0.0f, 1.0f, texColor.r);"
                            "   texColor  = vec4(1.0f, 1.0f, 1.0f, texColor.a);                 "
                            "   gl_FragColor  = u_Color * texColor;                  "
                            "}";

static DEMOGfxShader s_shader;

// Simple structure that contains all the data
// we need to know about characters in the font texture
struct CharContainer_t
{
    u32   id;
    f32 minS;
    f32 minT;
    f32 maxS;
    f32 maxT;
    f32 minX;
    f32 minY;
    f32 maxX;
    f32 maxY;
    f32 xAdvance;
};

struct FontData_t
{
    // Contains all the data we need to render each character
    CharContainer_t* pCharDataBuffer;

    // Font texture height
    f32 fontTextureWidth;

    // Font texture width
    f32 fontTextureHeight;

    // Font channel (RGBA)
    u32 channel;

    // The number of characters stored in the raw char data
    u32 numCharData;

    // The offset height of grid
    f32 gridOffsetY;

    // The offset width of grid
    f32 gridOffsetX;

    // Pointer to Font Image Data
    u8* pFontImageData;

    // includes pointer to buffer containing the Texture image Data
    DEMOGfxTexture    texture;

};

// DEMO Font Context State
static DEMOGfxContextState s_contextState;

// P: Proportional, F: Fixed mono_space
static FontData_t s_fontP, s_fontF;


u32 gDEMOFontNumLines = 0;
#define MAX_NUM_LINES 100
static u32 s_stringSize[MAX_NUM_LINES];
static DEMOGfxVertexData s_fontPosData[MAX_NUM_LINES];
static DEMOGfxVertexData s_fontTcData[MAX_NUM_LINES];
static DEMOGfxUniformData s_fontUniformDataVS[MAX_NUM_LINES];
static DEMOGfxUniformData s_fontUniformDataPS[MAX_NUM_LINES];
static DEMOGfxTexture* s_fontTexture[MAX_NUM_LINES];
static DEMOGfxIndexData  s_fontIndexData;

static DEMOFontFontData s_font;
static BOOL s_proportional=FALSE;
static BOOL s_enabled=FALSE;

// Max number of unique characters defined in each font
#define MAX_NUM_CHARS 100
static CharContainer_t s_charDataBufferP[MAX_NUM_CHARS];
static CharContainer_t s_charDataBufferF[MAX_NUM_CHARS];

//--------------------------------------------------------------------------
//   Forward references
//--------------------------------------------------------------------------
static FontData_t* GetLwrrentFont(void);
static void UpdateScale(DEMOFontFontData* pFont, f32 scaleX, f32 scaleY);
static void GetFontCharData(FontData_t* pFont,
                            const u32* pFontHeader,
                            const u32* pCharData,
                            const u8* pFontImageData);
static void InitFontTexture(FontData_t* pFont);
static void InitShader(void);
static u32  BSearchIndex(FontData_t* pFont, u32 id);

extern LWNdevice    s_device;

// --------------------------------------------------------------------------
//  Init
// --------------------------------------------------------------------------
void DEMOFontInit()
{
    DEMOGfxContextState* pOldContextState = DEMOGfxGetContextState();
    DEMOGfxInitContextState(&s_contextState);

    DEMOGfxSetColorControl(0, 0, LWN_TRUE, LWN_LOGIC_OP_COPY);
    DEMOGfxSetBlendControl(0, LWN_BLEND_FUNC_SRC_ALPHA, LWN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, LWN_BLEND_EQUATION_ADD,
                              LWN_BLEND_FUNC_SRC_ALPHA, LWN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, LWN_BLEND_EQUATION_ADD);
    DEMOGfxSetDepthControl(LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_DEPTH_FUNC_LESS);
    
    // Get Font char data
    GetFontCharData(&s_fontF, s_FontHeaderF, (u32*)s_CharDataF, s_FontImageDataF);

    GetFontCharData(&s_fontP, s_FontHeaderP, (u32*)s_CharDataP, s_FontImageDataP);
               
    // Initialise Shader
    InitShader();
     
    // Initialize Font Texture
    InitFontTexture(&s_fontP);
    InitFontTexture(&s_fontF);

    // Set up index buffer
    DEMOGfxCreateIndexBuffer(&s_fontIndexData, sizeof(DEMO_U16) * FONT_MAX_BUFSIZ * 6);

    // Initialize index buffer
    static u16 indexBuf[FONT_MAX_BUFSIZ * 6];
    
    // 2 triangles per char = 6 indices
    u32 elements = FONT_MAX_BUFSIZ * 6;

    for(u32 i = 0; i < elements; ++i)
    {
        u16 remap[] = { 0, 1, 3, 3, 0, 2 };
        indexBuf[i] =(u16)(i / 6 * 4 + remap[i%6]);
    }
    DEMOGfxSetIndexBuffer(&s_fontIndexData, indexBuf, 0, sizeof(indexBuf));

    // Initialize Proportional Font
    DEMOFontSetSpacing(1);
    DEMOFontSetGridSize(60, 24);
    DEMOFontSetColor(1.0f, 1.0f, 1.0f, 1.0f);
    DEMOFontSetZValue(-1.0f);
    
    s_enabled = TRUE;

    DEMOGfxSetContextState(pOldContextState);
}

// --------------------------------------------------------------------------
//  ShutDown
// --------------------------------------------------------------------------
void DEMOFontShutdown(void)
{
    // Release buffers
    DEMOGfxReleaseTextureBuffer(&s_fontP.texture);
    DEMOGfxReleaseTextureBuffer(&s_fontF.texture);
    DEMOGfxReleaseIndexBuffer(&s_fontIndexData);

    for(int i = 0; i < MAX_NUM_LINES; i++)
    {
        DEMOGfxReleaseVertexBuffer(&s_fontPosData[i]);
        DEMOGfxReleaseVertexBuffer(&s_fontTcData[i]);

        DEMOGfxReleaseUniformBuffer(&s_fontUniformDataVS[i]);
        DEMOGfxReleaseUniformBuffer(&s_fontUniformDataPS[i]);
    }
}

// --------------------------------------------------------------------------
//  Enable
// --------------------------------------------------------------------------
void DEMOFontDrawEnable(BOOL enable)
{
    s_enabled = enable;
}

BOOL DEMOFontIsEnable(void)
{
    return s_enabled;
}

void DEMOFontDoneRender(void)
{
    DEMOGfxContextState *pOldContextState = DEMOGfxGetContextState();
    DEMOGfxSetContextState(&s_contextState);

    for(u32 i = 0; i < gDEMOFontNumLines; i++)
    {
        // Bind texture buffer
        DEMOGfxBindTextureBuffer(s_fontTexture[i], LWN_SHADER_STAGE_FRAGMENT, 0);

        // Bind vertex buffers
        DEMOGfxBindVertexBuffer(&s_fontPosData[i],   0, 0, sizeof(DEMO_F32x3) * 4 * s_stringSize[i]);
        DEMOGfxBindVertexBuffer(&s_fontTcData[i],    1, 0, sizeof(DEMO_F32x2) * 4 * s_stringSize[i]);

        // Bind uniform buffer
        DEMOGfxBindUniformBuffer(&s_fontUniformDataVS[i], LWN_SHADER_STAGE_VERTEX,   0, 0, sizeof(DEMO_F32x2));
        DEMOGfxBindUniformBuffer(&s_fontUniformDataPS[i], LWN_SHADER_STAGE_FRAGMENT, 0, 0, sizeof(DEMO_F32x4));

        // Draw
        DEMOGfxDrawElements(&s_fontIndexData, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_SHORT, s_stringSize[i] * 6, 0);
    }

    // Restore state
    DEMOGfxSetContextState(pOldContextState);

    gDEMOFontNumLines = 0;
}

// --------------------------------------------------------------------------
//  Printf
// --------------------------------------------------------------------------
void DEMOFontPrintf(f32 column, f32 line, const char* pFmt, ... )
{
    char str[FONT_MAX_BUFSIZ];
    va_list args;
    s32 stringSize;
    // Don't draw if fonts are disabled
    if (!s_enabled) return;


    // Get string
    va_start(args, pFmt);
    stringSize = vsnprintf( str, FONT_MAX_BUFSIZ, pFmt, args );

    // Assert for over string size
    if ( stringSize < 0 )
    {
        DEMOAssert(!"String is too long\n");
    }

    va_end(args);

    DEMOFontPuts( column, line, str );

}

// Updates all the vertex buffers with the data needed to render this font
static void UpdateVertexBuffers(const char* pStr, u32 strLength, f32 x, f32 y)
{
    u32 i = 0;
    f32 lwrsorOffset = 0;
    f32 gridOffset = 0;

    static f32 posBuf[FONT_MAX_BUFSIZ * 3 * 4];
    static f32 tcBuf[FONT_MAX_BUFSIZ * 2 * 4];

    DEMOFontFontData* pFontData = &s_font;
    FontData_t* pFont = GetLwrrentFont();

    while (i < strLength)
    {
        for(; i < strLength; i++) 
        {
            // Get character id
            u32 id = (u32) pStr[i];
            u32 index = 0;

            // Set index offset of buffers
            u32 posSt = i*12;
            u32 texSt = i*8;

            // Check "\n"
            if(id == 10)
            {
                gridOffset  -= (f32)pFont->gridOffsetY;
                lwrsorOffset = 0;
                //index = 0;
                i++;
                break;
            }
            else if(id >= 32 && id <= 127)
            {
                // Get index of character id
                index = BSearchIndex(pFont,id);
            }

            // Set Vertex position
            posBuf[posSt+0] = (f32)(pFont->pCharDataBuffer[index].minX + lwrsorOffset + x);
            posBuf[posSt+3] = (f32)posBuf[posSt+0];
            posBuf[posSt+6] = (f32)(pFont->pCharDataBuffer[index].maxX + lwrsorOffset + x);
            posBuf[posSt+9] = (f32)posBuf[posSt+6];

            lwrsorOffset += pFont->pCharDataBuffer[index].xAdvance;

            posBuf[posSt+1] = (f32)(pFont->pCharDataBuffer[index].minY + gridOffset + y);
            posBuf[posSt+4] = (f32)(pFont->pCharDataBuffer[index].maxY + gridOffset + y);
            posBuf[posSt+7] = (f32)posBuf[posSt+1];
            posBuf[posSt+10] = (f32)posBuf[posSt+4];

            posBuf[posSt+2] = (f32)pFontData->depth;
            posBuf[posSt+5] = (f32)pFontData->depth;
            posBuf[posSt+8] = (f32)pFontData->depth;
            posBuf[posSt+11] = (f32)pFontData->depth;

            // Set Texture coordinate
            tcBuf[texSt+0] = (f32)pFont->pCharDataBuffer[index].minS;
            tcBuf[texSt+2] = (f32)pFont->pCharDataBuffer[index].minS;
            tcBuf[texSt+4] = (f32)pFont->pCharDataBuffer[index].maxS;
            tcBuf[texSt+6] = (f32)pFont->pCharDataBuffer[index].maxS;

            tcBuf[texSt+1] = (f32)pFont->pCharDataBuffer[index].minT;
            tcBuf[texSt+3] = (f32)pFont->pCharDataBuffer[index].maxT;
            tcBuf[texSt+5] = (f32)pFont->pCharDataBuffer[index].minT;
            tcBuf[texSt+7] = (f32)pFont->pCharDataBuffer[index].maxT;
        }
        //i++;
    }

    // Fill vertex buffer with vertex data
    DEMOGfxSetVertexBuffer(&s_fontPosData[gDEMOFontNumLines],   posBuf, 0, sizeof(DEMO_F32x3) * 4 * strLength);
    DEMOGfxSetVertexBuffer(&s_fontTcData[gDEMOFontNumLines],    tcBuf,  0, sizeof(DEMO_F32x2) * 4 * strLength);

    s_stringSize[gDEMOFontNumLines] = strLength;
}

static void UpdateTextureBuffer(DEMOGfxTexture* pTexture)
{
    s_fontTexture[gDEMOFontNumLines] = pTexture;
}

static void UpdateUniformBuffers(f32* pMagScale, f32* pColor)
{
    DEMOGfxSetUniformBuffer(&s_fontUniformDataVS[gDEMOFontNumLines], pMagScale,   0, sizeof(DEMO_F32x2));
    DEMOGfxSetUniformBuffer(&s_fontUniformDataPS[gDEMOFontNumLines], pColor,         0, sizeof(DEMO_F32x4));
}

void DEMOFontPuts(f32 column, f32 line, const char* pStr)
{
    u32 stringLength;
    f32 offsetX;
    f32 offsetY;
    FontData_t* pFont = GetLwrrentFont();
    DEMOFontFontData* pFontData = &s_font;

    //DEMOPrintf("str = %s\n", pStr);

    // Don't draw if fonts are disabled
    if (!s_enabled) return;

    // Check the initialize
    
    DEMOAssert(pStr &&
        "Need to initialize pStr.\n");
    DEMOAssert(pFont->pCharDataBuffer &&
        "Need to call DEMOFontInit(). Before this function.\n");
    
    stringLength = (u32)(strlen((const char *)pStr));

    // Calc offsets
    offsetX = (f32)(column * pFont->gridOffsetX);
    offsetY = -(f32)(line * pFont->gridOffsetY);

    // Update Texture Buffer
    UpdateTextureBuffer(&pFont->texture);

    // Update Vertex Buffers
    UpdateVertexBuffers(pStr, stringLength, offsetX, offsetY - pFont->gridOffsetY);

    // Update Uniform Buffers
    UpdateUniformBuffers(pFontData->charMagScale, pFontData->color);

    ++gDEMOFontNumLines;
}

// --------------------------------------------------------------------------
//  Update
// --------------------------------------------------------------------------
void DEMOFontSetSpacing(BOOL proportional)
{
    // Check the initialize
    DEMOAssert(s_fontP.pCharDataBuffer &&
        "Need to call DEMOFontInit(). Before this function.\n");
        
    // Update proportional boolean
    s_proportional = proportional;
    
    // Switch which scale we use
    if (proportional)
        s_font.charMagScale = s_font.charMagScaleP;
    else
        s_font.charMagScale = s_font.charMagScaleF;
}

void DEMOFontSetGridSize(f32 xGrid, f32 yGrid)
{
    // Check the initialize
    DEMOAssert(s_fontP.pCharDataBuffer &&
        "Need to call DEMOFontInit(). Before this function.\n");

    // Update scale
    UpdateScale(&s_font, xGrid, yGrid);
}

void DEMOFontSetColor(f32 r, f32 g, f32 b, f32 a)
{
    // Check the initialize
    DEMOAssert(s_fontP.pCharDataBuffer &&
        "Need to call DEMOFontInit(). Before this function.\n");

    // Update color
    s_font.color[0] = r;
    s_font.color[1] = g;
    s_font.color[2] = b;
    s_font.color[3] = a;
}

void DEMOFontSetZValue(f32 zValue)
{
    // Check the initialize
    DEMOAssert(s_fontP.pCharDataBuffer &&
        "Need to call DEMOFontInit(). Before this function.\n");

    // Update depth value
    s_font.depth = zValue;
}

// --------------------------------------------------------------------------
//  Getter
// --------------------------------------------------------------------------

void DEMOFontGetCharSize(f32 *pCharWidth, f32 *pCharHeight)
{
    FontData_t* pFont = GetLwrrentFont();

    // Get size of font
    *pCharWidth   = pFont->gridOffsetX;
    *pCharHeight  = pFont->gridOffsetY;
}

// --------------------------------------------------------------------------
//  Private Functions
// --------------------------------------------------------------------------
/// Check Proportional boolean and return Font handle
static FontData_t* GetLwrrentFont(void)
{
    if(s_proportional)
    {
        return &s_fontP;
    }
    else
    {
        return &s_fontF;
    }
}

/// Update scale
static void UpdateScale(DEMOFontFontData* pFont, f32 scaleX, f32 scaleY)
{
    // Callwlate char scale
    pFont->charMagScaleF[0] = 2.0f/(scaleX * s_fontF.gridOffsetX);
    pFont->charMagScaleF[1] = 2.0f/(scaleY * s_fontF.gridOffsetY);
    
    pFont->charMagScaleP[0] = 2.0f/(scaleX * s_fontP.gridOffsetX);
    pFont->charMagScaleP[1] = 2.0f/(scaleY * s_fontP.gridOffsetY);
}

/// Get Font char data from demoFontData.h
static void GetFontCharData(FontData_t* pFont,
                            const u32* pFontHeader,
                            const u32* pCharData,
                            const u8* pFontImageData)
{
    u32 i;
    f32 lineHeight;
    f32 maxCharWidth = 0.0f;
    f32 maxCharHeight = 0.0f;

    // Check Font Texture Data
    DEMOAssert(pFontHeader != NULL && pCharData != NULL &&
        pFontImageData != NULL && "No texture data.\n");

    // Check char data buffer
    if (pFont->pCharDataBuffer)
    {
        // Skip Data Initialization
        return;
    }

    // Set Font Texture Data Information
    pFont->fontTextureWidth  = (f32)pFontHeader[0];
    pFont->fontTextureHeight = (f32)pFontHeader[1];
    pFont->channel           = (u32)pFontHeader[2];
    lineHeight               = (f32)pFontHeader[4];
    pFont->numCharData       = pFontHeader[5];
    pFont->gridOffsetX       = 0;
    pFont->gridOffsetY       = 0;

    if(pFont == &s_fontP)
    {
        pFont->pCharDataBuffer = s_charDataBufferP;
    }
    else
    {
        pFont->pCharDataBuffer = s_charDataBufferF;
    }

    // Check the max number
    DEMOAssert(pFont->numCharData <= MAX_NUM_CHARS &&
        "Font has over the max number of characters.\n");

    // Format of data is: id, x, y, width, height, xOffset, yOffset, xAdvance
    for(i = 0; i < pFont->numCharData; ++i)
    {
        u32 id         = pCharData[i*8 + 0];
        u32 x          = pCharData[i*8 + 1];
        u32 y          = pCharData[i*8 + 2];
        u32 w          = pCharData[i*8 + 3];
        u32 h          = pCharData[i*8 + 4];
        s32 xOffset    = (s32)pCharData[i*8 + 5];        
        s32 yOffset    = (s32)pCharData[i*8 + 6];
        u32 xAdvance   = pCharData[i*8 + 7];
        f32 charHeight = (f32)(h + yOffset);

        pFont->pCharDataBuffer[i].id       = id;
        pFont->pCharDataBuffer[i].minS     = (f32)x / pFont->fontTextureWidth;
        pFont->pCharDataBuffer[i].minT     = (f32)(pFont->fontTextureHeight - h - y) / pFont->fontTextureHeight;
        pFont->pCharDataBuffer[i].maxS     = (f32)(x + w) / pFont->fontTextureWidth;
        pFont->pCharDataBuffer[i].maxT     = (f32)(pFont->fontTextureHeight  - y) / pFont->fontTextureHeight;
        pFont->pCharDataBuffer[i].minX     = (f32)xOffset;
        pFont->pCharDataBuffer[i].minY     = (f32)(lineHeight - yOffset - h);
        pFont->pCharDataBuffer[i].maxX     = (f32)(xOffset + w);
        pFont->pCharDataBuffer[i].maxY     = (f32)(lineHeight - yOffset);
        pFont->pCharDataBuffer[i].xAdvance = (f32)xAdvance;

        // Set max height of char in GL cordinate space
        if(charHeight > maxCharHeight)
        {
            maxCharHeight = charHeight;
        }

        if(pFont->pCharDataBuffer[i].xAdvance > maxCharWidth)
        {
            maxCharWidth = (f32)pFont->pCharDataBuffer[i].xAdvance;
        }
    }

    // Set grid offsetX
    pFont->gridOffsetX = maxCharWidth;

    // Set grid offsetY
    pFont->gridOffsetY = maxCharHeight;

    // Set pointer of Font Image Data
    pFont->pFontImageData = (u8*)pFontImageData;
}

/// Init Font Texture

static void InitFontTexture(FontData_t* pFont)
{
    // Texture setup
    u32 width   =(u32)pFont->fontTextureWidth;
    u32 height  =(u32)pFont->fontTextureHeight;
    u32 depth   = 1;
    u32 channel =(u32)pFont->channel;
    u32 size    =width*height*depth*channel;

    // Set up texture buffer
    DEMOGfxCreateTextureBuffer(&pFont->texture, LWN_TEXTURE_TARGET_2D, 1, LWN_FORMAT_R8, width, height, depth, 0, size);

    // Fill texture buffer with texture data
    DEMOGfxSetTextureBuffer(&pFont->texture, 0, (const void*)pFont->pFontImageData, 0, 0, 0, 0, width, height, depth, size);
}

/// Initialize Font shader
static void InitShader(void)
{
    // Set up shader
    DEMOGfxCreateShaders(&s_shader, s_vsString, s_psString);

    // Set up vertex attributes
    DEMOGfxShaderAttributeData posAttrib     = {0, LWN_FORMAT_RGB32F, 0, 0, sizeof(DEMO_F32x3)};
    DEMOGfxShaderAttributeData tcAttrib    = {1, LWN_FORMAT_RG32F, 0, 1, sizeof(DEMO_F32x2)}; 

    DEMOGfxSetShaderAttribute(&s_shader, &posAttrib);
    DEMOGfxSetShaderAttribute(&s_shader, &tcAttrib);

    for(int i = 0; i < MAX_NUM_LINES; i++)
    {
        // Set up vertex buffer
        DEMOGfxCreateVertexBuffer(&s_fontPosData[i],   sizeof(DEMO_F32x3) * 4 * FONT_MAX_BUFSIZ);
        DEMOGfxCreateVertexBuffer(&s_fontTcData[i],    sizeof(DEMO_F32x2) * 4 * FONT_MAX_BUFSIZ);

        // Init uniform buffer
        DEMOGfxCreateUniformBuffer(&s_fontUniformDataVS[i], sizeof(DEMO_F32x2) + sizeof(DEMO_F32x3));
        DEMOGfxCreateUniformBuffer(&s_fontUniformDataPS[i], sizeof(DEMO_F32x4));
    }

    DEMOGfxSetShaders(&s_shader);
}


// Search index of character id
static u32 BSearchIndex(FontData_t* pFont, u32 id)
{
    u32 first;
    u32 last;
    u32 index = 0;

    // quick check for sanely-ordered fonts
    if (id >= 32 && id < 32+pFont->numCharData &&
        pFont->pCharDataBuffer[id-32].id == id )
        return id-32;

    first = 0;
    last  = pFont->numCharData-1;

    while(first <= last)
    {
        u32 mid = first + (last - first) / 2;

        if( pFont->pCharDataBuffer[mid].id < id )
        {
            first = mid + 1;
        }
        else if( id < pFont->pCharDataBuffer[mid].id )
        {
            last = mid - 1;
        }
        else
        {
            index = mid;
            break;
        }
    }
    return index;
}
