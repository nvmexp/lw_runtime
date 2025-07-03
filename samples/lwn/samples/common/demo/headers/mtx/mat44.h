/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/


/*---------------------------------------------------------------------------*
    Matrix-Vector Library : 4x4 Matrix (Structure Version)
 *---------------------------------------------------------------------------*/

#ifndef __MAT44_H__
#define __MAT44_H__

#include <cafe/mat.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup MAT
/// @{

/*---------------------------------------------------------------------------*
    4x4 GENERAL MATRIX SECTION
 *---------------------------------------------------------------------------*/

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
static inline void MAT44Identity       ( Mat44 *m )
    { MTX44Identity       ( (Mtx44Ptr)(m->mtx) ); }

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
static inline void MAT44Copy           ( const Mat44 *src, Mat44 *dst )
    { MTX44Copy           ( (Mtx44Ptr)(src->mtx), (Mtx44Ptr)(dst->mtx) ); }

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
static inline void MAT44Concat         ( const Mat44 *a, const Mat44 *b, Mat44 *ab )
    { MTX44Concat         ( (Mtx44Ptr)(a->mtx), (Mtx44Ptr)(b->mtx), (Mtx44Ptr)(ab->mtx) ); }

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
static inline void MAT44Transpose      ( const Mat44 *src, Mat44 *xPose )
    { MTX44Transpose      ( (Mtx44Ptr)(src->mtx), (Mtx44Ptr)(xPose->mtx) ); }

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
static inline u32  MAT44Ilwerse        ( const Mat44 *src, Mat44 *ilw )
    { return MTX44Ilwerse        ( (Mtx44Ptr)(src->mtx), (Mtx44Ptr)(ilw->mtx) ); }

/*-------------------------------------------------------------------------*
    MODEL MATRIX SECTION
 *-------------------------------------------------------------------------*/

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
static inline void MAT44Trans          ( Mat44 *m, f32 xT, f32 yT, f32 zT )
    { MTX44Trans          ( (Mtx44Ptr)(m->mtx), xT, yT, zT ); }

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
static inline void MAT44TransApply     ( const Mat44 *src, Mat44 *dst, f32 xT, f32 yT, f32 zT )
    { MTX44TransApply     ( (Mtx44Ptr)(src->mtx), (Mtx44Ptr)(dst->mtx), xT, yT, zT ); }

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
static inline void MAT44Scale          ( Mat44 *m, f32 xS, f32 yS, f32 zS )
    { MTX44Scale          ( (Mtx44Ptr)(m->mtx), xS, yS, zS ); }

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
static inline void MAT44ScaleApply     ( const Mat44 *src, Mat44 *dst, f32 xS, f32 yS, f32 zS )
    { MTX44ScaleApply     ( (Mtx44Ptr)(src->mtx), (Mtx44Ptr)(dst->mtx), xS, yS, zS ); }


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
static inline void MAT44RotRad         ( Mat44 *m, char axis, f32 rad )
    { MTX44RotRad         ( (Mtx44Ptr)(m->mtx), axis, rad ); }

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
static inline void MAT44RotTrig        ( Mat44 *m, char axis, f32 sinA, f32 cosA )
    { MTX44RotTrig        ( (Mtx44Ptr)(m->mtx), axis, sinA, cosA ); }

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
static inline void MAT44RotAxisRad     ( Mat44 *m, const Vec *axis, f32 rad)
    { MTX44RotAxisRad     ( (Mtx44Ptr)(m->mtx), axis, rad); }

/*-------------------------------------------------------------------------*
    MATRIX-VECTOR SECTION
 *-------------------------------------------------------------------------*/

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
static inline void MAT44MultVec        ( const Mat44 *m, const Vec *src, Vec *dst )
    { MTX44MultVec        ( (Mtx44Ptr)(m->mtx), src, dst ); }

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
static inline void MAT44MultVecArray   ( const Mat44 *m, const Vec *srcBase, Vec *dstBase, u32 count )
    { MTX44MultVecArray   ( (Mtx44Ptr)(m->mtx), srcBase, dstBase, count ); }

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
static inline void MAT44MultVecSR      ( const Mat44 *m, const Vec *src, Vec *dst )
    { MTX44MultVecSR      ( (Mtx44Ptr)(m->mtx), src, dst ); }

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
static inline void MAT44MultVecArraySR ( const Mat44 *m, const Vec *srcBase, Vec *dstBase, u32 count )
    { MTX44MultVecArraySR ( (Mtx44Ptr)(m->mtx), srcBase, dstBase, count ); }

/*---------------------------------------------------------------------------*
    MATRIX COLWERSION
 *---------------------------------------------------------------------------*/

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
static inline void MAT34To44( const Mat34 *src, Mat44 *dst )
    { MTX34To44( (MtxPtr)(src->mtx), (Mtx44Ptr)(dst->mtx) ); }

/// @}

#ifdef __cplusplus
}
#endif

#endif // __MAT44_H__

/*===========================================================================*/
