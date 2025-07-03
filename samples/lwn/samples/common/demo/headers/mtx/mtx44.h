/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/


/*---------------------------------------------------------------------------*
    Matrix-Vector Library : 4x4 matrix extension
 *---------------------------------------------------------------------------*/

#ifndef __MTX44EXT_H__
#define __MTX44EXT_H__


#include <mtx.h>


#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------------------------------------------------*
    Default function binding configuration
 *---------------------------------------------------------------------------*/
// [Binding Rule]
//
// "MTX_USE_ASM" -> When this flag is specified, it uses PS* (Paired-Single
//                 assembler code) functions for non-prefixed function calls.
// "MTX_USE_PS" -> When this flag is specified, it uses PS* (Paired-Single
//                 intrinsics code) functions for non-prefixed function calls.
// "MTX_USE_C " -> When this flag is specified, it uses C_* (C code) functions
//                 for non-prefixed function calls.
//
// The first binding specified in priority order listed will be used
// If nothing is specified, refers ASM* functions

#if ( !defined(MTX_USE_ASM) && !defined(MTX_USE_PS) && !defined(MTX_USE_C))
#define MTX_USE_ASM
#endif

/*---------------------------------------------------------------------------*
    4x4 GENERAL MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{
 
// C version
 
/// \brief Sets a 4x4 Identity Matrix.
///
/// \param m Matrix to set
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44Identity         ( Mtx44 m );
 
/// \brief Copy a 4x4 Matrix.
///
/// \note This is safe for src == dst.
///
/// \param src Source matrix
/// \param dst Destination matrix
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44Copy             ( MTX_CONST Mtx44 src, Mtx44 dst );
 
/// \brief Concatenate two 4x4 Matrix.
/// The order of operations is:
///
/// AB = A x B
///
/// \note Safe if ab == a == b.
/// \note In the C version, an extra MTXCopy operation oclwrs if ab == a or 
/// ab == b.
///
/// \param a Left matrix
/// \param b Right matrix
/// \param ab Destination matrix
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44Concat           ( MTX_CONST Mtx44 a, MTX_CONST Mtx44 b, Mtx44 ab );
 
/// \brief Compute a 4x4 transpose matrix.
///
/// \param src Source matrix
/// \param xPose Destination (transposed) matrix
///
/// \note It is ok if src == xPose.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44Transpose        ( MTX_CONST Mtx44 src, Mtx44 xPose );
 
/// \brief Computes a fast ilwerse of a 4x4 Matrix.
/// Uses the Gauss-Jordan method (with partial pivoting)
///
/// \param src Source matrix
/// \param ilw Destination (ilwerse) matrix
/// \return 0 if src is not ilwertible, 1 on success.
///
/// \note It is ok if src == ilw.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
u32  C_MTX44Ilwerse          ( MTX_CONST Mtx44 src, Mtx44 ilw );

/// @}

// PS intrinsics version
void PSMTX44Identity         ( Mtx44 m );
void PSMTX44Copy             ( MTX_CONST Mtx44 src, Mtx44 dst );
void PSMTX44Concat           ( MTX_CONST Mtx44 a, MTX_CONST Mtx44 b, Mtx44 ab );
void PSMTX44Transpose        ( MTX_CONST Mtx44 src, Mtx44 xPose );

// PS assembler version
void ASM_MTX44Identity       ( Mtx44 m );
void ASM_MTX44Copy           ( MTX_CONST Mtx44 src, Mtx44 dst );
void ASM_MTX44Concat         ( MTX_CONST Mtx44 a, MTX_CONST Mtx44 b, Mtx44 ab );
void ASM_MTX44Transpose      ( MTX_CONST Mtx44 src, Mtx44 xPose );

#define MTX44Ilwerse            C_MTX44Ilwerse

// Bindings
#ifdef MTX_USE_ASM
#define MTX44Concat             ASM_MTX44Concat
#define MTX44Identity           ASM_MTX44Identity
#define MTX44Copy               ASM_MTX44Copy
#define MTX44Transpose          ASM_MTX44Transpose
#else
#ifdef MTX_USE_PS
#define MTX44Concat             PSMTX44Concat
#define MTX44Identity           PSMTX44Identity
#define MTX44Copy               PSMTX44Copy
#define MTX44Transpose          PSMTX44Transpose
#else // MTX_USE_C
#define MTX44Concat             C_MTX44Concat
#define MTX44Identity           C_MTX44Identity
#define MTX44Copy               C_MTX44Copy
#define MTX44Transpose          C_MTX44Transpose
#endif
#endif

/*---------------------------------------------------------------------------*
    MODEL MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{

// C version

/// \brief Sets a translation 4x4 matrix
///
/// \param m Matrix to be set
/// \param xT x component of translation
/// \param yT y component of translation
/// \param zT z component of translation
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44Trans            ( Mtx44 m, f32 xT, f32 yT, f32 zT );

/// \brief Apply a translation to a 4x4 matrix.
/// This function is equivalent to \ref MTX44Trans + \ref MTX44Concat.
///
/// \note This is safe for the case where src == dst.
///
/// \param src Matrix to multiply the translation by
/// \param dst Resulting matrix from concatenation
/// \param xT x component of translation
/// \param yT y component of translation
/// \param zT z component of translation
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44TransApply       ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xT, f32 yT, f32 zT );

/// \brief Sets a scale 4x4 matrix
///
/// \param m Matrix to be set
/// \param xS x component of scale
/// \param yS y component of scale
/// \param zS z component of scale
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44Scale            ( Mtx44 m, f32 xS, f32 yS, f32 zS );

/// \brief Apply a scale to a 4x4 matrix.
/// This function is equivalent to \ref MTX44Scale + \ref MTX44Concat.
///
/// \note This is safe for the case where src == dst.
///
/// \param src Matrix to multiply the scale by
/// \param dst Resulting matrix from concatenation
/// \param xS x component of scale
/// \param yS y component of scale
/// \param zS z component of scale
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44ScaleApply       ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xS, f32 yS, f32 zS );

/// \brief Sets a rotation 4x4 matrix about one of the X, Y or Z axes.
///
/// \note Counter clockwise rotation is positive.
///
/// \param m Matrix to be set
/// \param axis Principal axis of rotation. Must be 'X', 'x', 'Y', 'y', 'Z', or 'z'.
/// \param rad Rotation angle in radians
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44RotRad           ( Mtx44 m, char axis, f32 rad );

/// \brief Sets a rotation 4x4 matrix about one of the X, Y, or Z axes from specified trig ratios.
///
/// \note Counter clockwise rotation is positive.
///
/// \param m Matrix to be set
/// \param axis Principal axis of rotation. Must be 'X', 'x', 'Y', 'y', 'Z', or 'z'.
/// \param sinA Sine of rotation angle
/// \param cosA Cosine of rotation angle
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44RotTrig          ( Mtx44 m, char axis, f32 sinA, f32 cosA );

/// \brief Sets a rotation 4x4 matrix about an arbitrary axis.
///
/// \note Counter clockwise rotation is positive.
///
/// \param m Matrix to be set
/// \param axis Pointer to a vector containing the (x, y, z) axis components
/// \param rad Rotation angle in radians
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44RotAxisRad       ( Mtx44 m, const Vec *axis, f32 rad);

/// @}

// PS intrinsics version
void PSMTX44Trans            ( Mtx44 m, f32 xT, f32 yT, f32 zT );
void PSMTX44TransApply       ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xT, f32 yT, f32 zT );
void PSMTX44Scale            ( Mtx44 m, f32 xS, f32 yS, f32 zS );
void PSMTX44ScaleApply       ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xS, f32 yS, f32 zS );

void PSMTX44RotRad           ( Mtx44 m, char axis, f32 rad );
void PSMTX44RotTrig          ( Mtx44 m, char axis, f32 sinA, f32 cosA );
void PSMTX44RotAxisRad       ( Mtx44 m, const Vec *axis, f32 rad);

// PS assembler version
void ASM_MTX44Scale          ( Mtx44 m, f32 xS, f32 yS, f32 zS );
void ASM_MTX44ScaleApply     ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xS, f32 yS, f32 zS );
void ASM_MTX44Trans          ( Mtx44 m, f32 xT, f32 yT, f32 zT );
void ASM_MTX44TransApply     ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xT, f32 yT, f32 zT );
void ASM_MTX44RotTrig        ( Mtx44 m, char axis, f32 sinA, f32 cosA );
void ASM_MTX44RotRad         ( Mtx44 m, char axis, f32 rad );
void ASM_MTX44RotAxisRad     ( Mtx44 m, const Vec *axis, f32 rad );

// Bindings
#ifdef MTX_USE_ASM
#define MTX44RotRad             ASM_MTX44RotRad
#define MTX44RotTrig            ASM_MTX44RotTrig
#define MTX44Trans              ASM_MTX44Trans
#define MTX44TransApply         ASM_MTX44TransApply
#define MTX44Scale              ASM_MTX44Scale
#define MTX44ScaleApply         ASM_MTX44ScaleApply
#define MTX44RotAxisRad         ASM_MTX44RotAxisRad
#else
#ifdef MTX_USE_PS
#define MTX44RotRad             PSMTX44RotRad
#define MTX44RotTrig            PSMTX44RotTrig
#define MTX44Trans              PSMTX44Trans
#define MTX44TransApply         PSMTX44TransApply
#define MTX44Scale              PSMTX44Scale
#define MTX44ScaleApply         PSMTX44ScaleApply
#define MTX44RotAxisRad         PSMTX44RotAxisRad
#else // MTX_USE_C
#define MTX44RotRad             C_MTX44RotRad
#define MTX44RotTrig            C_MTX44RotTrig
#define MTX44Trans              C_MTX44Trans
#define MTX44TransApply         C_MTX44TransApply
#define MTX44Scale              C_MTX44Scale
#define MTX44ScaleApply         C_MTX44ScaleApply
#define MTX44RotAxisRad         C_MTX44RotAxisRad
#endif
#endif

/*---------------------------------------------------------------------------*
    MATRIX-VECTOR SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{
 
// C version

/// \brief Multiplies a vector by a 4x4 matrix
///
/// dst = m x src
///
/// \note It is safe for src == dst.
///
/// \param m Matrix to multiply by
/// \param src Source vector of multiply
/// \param dst Resulting vector from multiply
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44MultVec          ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst );

/// \brief Multiplies an array of vectors by a 4x4 matrix.
///
/// \note It is safe for srcBase == dstBase.
///
/// \warning This function cannot check for array overflow.
///
/// \param m Matrix to multiply by
/// \param srcBase Source vector array
/// \param dstBase Resulting vector array
/// \param count Number of vectors in srcBase and dstBase arrays.
///
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44MultVecArray     ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase, u32 count );

/// \brief Multiply a vector by a 4x4 Scaling and Rotating matrix
/// \note It is assumed that the 4th row/column are (0, 0, 0, 1).
///
/// This is equivalent to:
///
/// dst = m x src
///
/// \note It is safe for src == dst.
///
/// \param m Matrix to multiply by
/// \param src Source vector for multiply
/// \param dst Resulting vector from multiply
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44MultVecSR        ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst );

/// \brief Multiply an array of vector by a 4x4 Scaling and Rotating matrix
/// \note It is assumed that the 4th row/column are (0, 0, 0, 1).
///
/// This is equivalent to:
///
/// dstBase[i] = m x srcBase[i]
///
/// \note It is safe for srcBase == dstBase.
///
/// \warning This function cannot check for array overflow
///
/// \param m Matrix to multiply by
/// \param srcBase Source vector array
/// \param dstBase Resulting vector array
/// \param count Number of vectors in srcBase and dstBase
///
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX44MultVecArraySR   ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase, u32 count );

/// @}

// PS intrinsics version
void PSMTX44MultVec          ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst );
void PSMTX44MultVecArray     ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase, u32 count );
void PSMTX44MultVecSR        ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst );
void PSMTX44MultVecArraySR   ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase, u32 count );

// PS assembler version
void ASM_MTX44MultVec        ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst );
void ASM_MTX44MultVecArray   ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase,  u32 count );
void ASM_MTX44MultVecArraySR ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase,  u32 count );
void ASM_MTX44MultVecSR      ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst );

// Bindings
#ifdef MTX_USE_ASM
#define MTX44MultVec            ASM_MTX44MultVec
#define MTX44MultVecArray       ASM_MTX44MultVecArray
#define MTX44MultVecSR          ASM_MTX44MultVecSR
#define MTX44MultVecArraySR     ASM_MTX44MultVecArraySR
#else
#ifdef MTX_USE_PS
#define MTX44MultVec            PSMTX44MultVec
#define MTX44MultVecArray       PSMTX44MultVecArray
#define MTX44MultVecSR          PSMTX44MultVecSR
#define MTX44MultVecArraySR     PSMTX44MultVecArraySR
#else // MTX_USE_C
#define MTX44MultVec            C_MTX44MultVec
#define MTX44MultVecArray       C_MTX44MultVecArray
#define MTX44MultVecSR          C_MTX44MultVecSR
#define MTX44MultVecArraySR     C_MTX44MultVecArraySR
#endif
#endif

/*---------------------------------------------------------------------------*
    MATRIX COLWERSION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{
 
 
/// \brief Colwert a 3x4 matrix to a 4x4 matrix.
/// The fourth row is set to (0, 0, 0, 1).
///
/// \param src Source 3x4 matrix
/// \param dst Destination 4x4 matrix
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_MTX34To44  ( MTX_CONST Mtx src, Mtx44 dst );

/// @}

void PSMTX34To44  ( MTX_CONST Mtx src, Mtx44 dst );
void ASM_MTX34To44( MTX_CONST Mtx src, Mtx44 dst );

// Bindings
#ifdef MTX_USE_ASM
#define MTX34To44            ASM_MTX34To44
#else
#ifdef MTX_USE_PS
#define MTX34To44            PSMTX34To44
#else // MTX_USE_C
#define MTX34To44            C_MTX34To44
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif // __MTX44EXT_H__

/*===========================================================================*/
