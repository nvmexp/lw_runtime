/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/
// -----------------------------------------------------------------------------
//  demoFont.h
//
// -----------------------------------------------------------------------------

#ifndef __DEMO_FONT_H_
#define __DEMO_FONT_H_

#include <types.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup demoFont
/// @{

/// \brief Structure used to define the font data for an instance of DEMOFont
typedef struct
{
    /// \brief Mag Scale X,Y of characters
    f32 *charMagScale;

    /// \brief Mag Scale X,Y of characters (proportional)
    f32 charMagScaleF[2]; // xxx should it be [4]?
    
    /// \brief Mag Scale X,Y of characters (fixed)
    f32 charMagScaleP[2]; // xxx should it be [4]?
    
    /// \brief Depth value
    f32 depth;

    /// \brief Color
    f32 color[4];

} DEMOFontFontData;

/// \brief Initializes the font data and font shader program
///
/// Sets up demo font context state and font texture object.
/// Creates the font shader program, Sets up font initial values.
//  Sets up initial grid = 60x24(proportional), 80x24(mono-spacing), color = (1.0f,1.0f,1.0f,1.0f), zValue=-1.0f, proportional=TRUE.
/// At default, DEMOFont uses DEMOColorBuffer and viewport which is created in DEMOGfxInit.
/// If you want to change the buffer, need to call DEMOFontSetTarget after init.
///
void DEMOFontInit(void);

/// \brief Deletes all the font data
void DEMOFontShutdown(void);

/// \brief Indicates the font is done ready to be rendered
void DEMOFontDoneRender(void);

/// \brief Set view port for demo font.
///
/// Set view port for demo font.
/// Need to call this API after calling DEMOFontSetContextState.
///
/// \param xOrig orig x point of viewport
/// \param yOrig orig y point of viewport
/// \param width width of viewport
/// \param height height of viewport
inline void DEMOFontSetViewport(f32 xOrig, f32 yOrig, f32 width, f32 height)
{
    //GX2SetViewport(xOrig, yOrig, width, height, 0.0f, 1.0f);
    //GX2SetScissor(xOrig, yOrig, width, height);
}

/// \brief Set color buffer and viewport for demo font
///
/// Set color buffer and view port for demo font.
/// Need to call this API after calling DEMOFontSetContextState.
///
/// \param colorBuffer Ptr to color buffer structure to set.
//inline void DEMOFontSetTarget(GX2ColorBuffer* colorBuffer)
//{
//    DEMOFontSetViewport(0.0,
//                        0.0,
//                        colorBuffer->surface.width,
//                        colorBuffer->surface.height);
//
//    GX2SetColorBuffer(colorBuffer, GX2_RENDER_TARGET_0);
//}

/// \brief Enable/disable fonts from drawing
///
/// \param enable If true, fonts will draw. If false, they will not draw.
void DEMOFontDrawEnable(BOOL enable);

/// \brief Check Enable/disable fonts from drawing
BOOL DEMOFontIsEnable(void);

/// \brief Output the string into Screen by character units step
///
/// Need to call this API after DEMOFontSetContextState.
///
/// \param column Column position of the string by character units step (0 is left side of Screen)
/// \param line Line position of the string by character units step (0 is the top of Screen)
/// \param pFmt Pointer to a null-terminated string including format specification (equivalent to C's standard output function)
void DEMOFontPrintf(f32 column, f32 line, const char* pFmt, ... );

/// \brief Output the pre-formatted string into Screen by character units step
///
/// Need to call this API after DEMOFontSetContextState.
///
/// \param column Column position of the string by character units step (0 is left side of Screen)
/// \param line Line position of the string by character units step (0 is the top of Screen)
/// \param pStr Pointer to a null-terminated string
void DEMOFontPuts(f32 column, f32 line, const char* pStr);

/// \brief Set the font color
///
/// \param r The red color for the font
/// \param g The green color for the font
/// \param b The blue color for the font
/// \param a The alpha color for the font
void DEMOFontSetColor(f32 r, f32 g, f32 b, f32 a);

/// \brief Set the grid size
///
/// \param xGrid The number of fixed-width columns
///  \param yGrid The number of fixed-width lines
void DEMOFontSetGridSize(f32 xGrid, f32 yGrid);

/// \brief Set the font spacing type
///
/// \param proportional True means to use proportional spacing, false means mono spacing
void DEMOFontSetSpacing(BOOL proportional);

/// \brief Set the Z value
///
///    \param zValue  -1.0 = near plane, 1.0 = far plane
void DEMOFontSetZValue(f32 zValue);

/// \brief Get the maximum texel size of the character.
///
/// \param pCharWidth Pointer to the width texel size of character cell
/// \param pCharHeight Pointer to the height texel size of character cell
void DEMOFontGetCharSize(f32 *pCharWidth, f32 *pCharHeight);

/// @}

#ifdef __cplusplus
}
#endif

#endif // __DEMO_FONT_H_

