/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

/*---------------------------------------------------------------------------*
    Matrix-Vector Library (Structure Version)
 *---------------------------------------------------------------------------*/

#ifndef __MATVEC_H__
#define __MATVEC_H__

#include <cafe/mtx.h>
#include <cafe/mtx/mtxGeoTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup MAT
/// @{

/*---------------------------------------------------------------------------*
    GENERAL MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// \brief Set a 3x4 matrix to the identity.
///
/// \param m Matrix to be set.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34Identity    ( Mat34 *m )
    { MTXIdentity    ( (MtxPtr)(m->mtx) ); }

/// \brief Copies the contents of one 3x4 matrix into another
///
/// \param src Source matrix for the copy.
/// \param dst Destination matrix for the copy.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34Copy        ( const Mat34 *src, Mat34 *dst )
    { MTXCopy        ( (MtxPtr)(src->mtx), (MtxPtr)(dst->mtx) ); }

/// \brief Concatenates two 3x4 matrices
///
/// Order of operations is A x B = AB.
/// This function can handle the case when ab == a == b.
///
/// \param a First matrix to concatentate
/// \param b Second matrix to concatentate
/// \param ab Resulting matrix from concatentate
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34Concat      ( const Mat34 *a, const Mat34 *b, Mat34 *ab )
    { MTXConcat      ( (MtxPtr)(a->mtx), (MtxPtr)(b->mtx), (MtxPtr)(ab->mtx) ); }

/// \brief Concatentates a 3x4 matrix to an array of 3x4 matrices.
/// The order of operations is A x B(array) = AB(array)
/// This routine is equivalent to:
///
/// dstBase[i] = A x srcBase[i] for all i = 0 to count - 1
///
/// \param a first matrix for concatenation
/// \param srcBase array base of second matrix for concatenation
/// \param dstBase array base of resulting matrix from concatenation
/// \param count number of matrices in srcBase and dstBase arrays
///
/// \warning This routine cannot check for array overflow.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34ConcatArray ( const Mat34 *a, const Mat34 *srcBase, Mat34 *dstBase, u32 count )
    { MTXConcatArray ( (MtxPtr)(a->mtx), (Mtx *)(void *)(srcBase->mtx), (Mtx *)(&dstBase->mtx), count ); }

/// \brief Computes the transpose of a 3x4 matrix.
///
/// \note It is safe for src == xPose
///
/// \warning If the matrix is a 3x4 matrix, the fourth column (translation
/// component) is lost and becomes (0, 0, 0). This function is intended for
/// use in computing an ilwerse-transpose matrix to transform normals for 
/// lighting. In this case, the loss of the translation component doesn't
/// matter.
///
/// \param src Source matrix
/// \param xPose Destination (transposed) matrix
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34Transpose   ( const Mat34 *src, Mat34 *xPose )
    { MTXTranspose   ( (MtxPtr)(src->mtx), (MtxPtr)(xPose->mtx) ); }

/// \brief Computes a fast ilwerse of a 3x4 matrix.
/// This algorith works for matrices with a fourth row of (0, 0, 0, 1).
///
/// For a matrix:
///
/// M = | A  C |
///     | 0  1 |
///
/// Where A is the upper 3x3 submatrix and C is a 1x3 column vector:
///
/// ILW(M) = | ilw(A)   ilw(A)*(-C) |
///          |   0          1       |
///
/// \note It is safe for src == ilw.
///
/// \param src Source matrix
/// \param ilw Destination (ilwerse) matrix
/// \return 0 if src is not ilwertible, 1 on success.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline u32  MAT34Ilwerse     ( const Mat34 *src, Mat34 *ilw )
    { return MTXIlwerse     ( (MtxPtr)(src->mtx), (MtxPtr)(ilw->mtx) ); }

/// \brief Computes a fast ilwerse-transpose of a 3x4 matrix.
///
/// This algorithm works for matrices with a fourth row of (0, 0, 0, 1).
/// Commonly used for callwlating normal transform matrices.
///
/// This function is equivalent to the combination of two functions
/// \ref MTXIlwerse + \ref MTXTranspose
///
/// \note It is safe to call this function if src == ilwX.
///
/// \param src Source matrix
/// \param ilwX Destination (ilwerse-transpose) matrix.
/// \return 0 if src is not ilwertible, 1 on success.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline u32  MAT34IlwXpose    ( const Mat34 *src, Mat34 *ilwX )
    { return MTXIlwXpose    ( (MtxPtr)(src->mtx), (MtxPtr)(ilwX->mtx) ); }

/*---------------------------------------------------------------------------*
    MATRIX-VECTOR SECTION
 *---------------------------------------------------------------------------*/

/// \brief Multiplies a vector by a 3x4 matrix
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
static inline void MAT34MultVec        ( const Mat34 *m, const Vec *src, Vec *dst )
    { MTXMultVec        ( (MtxPtr)(m->mtx), src, dst ); }

/// \brief Multiplies an array of vectors by a 3x4 matrix.
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
static inline void MAT34MultVecArray   ( const Mat34 *m, const Vec *srcBase, Vec *dstBase, u32 count )
    { MTXMultVecArray   ( (MtxPtr)(m->mtx), srcBase, dstBase, count ); }

/// \brief Multiply a vector by a 3x4 Scaling and Rotating matrix
/// \note It is assumed that the 4th column (translation) is 0.
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
static inline void MAT34MultVecSR      ( const Mat34 *m, const Vec *src, Vec *dst )
    { MTXMultVecSR      ( (MtxPtr)(m->mtx), src, dst ); }

/// \brief Multiply an array of vector by a 3x4 Scaling and Rotating matrix
/// \note It is assumed that the 4th column (translation) is 0.
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
static inline void MAT34MultVecArraySR ( const Mat34 *m, const Vec *srcBase, Vec *dstBase, u32 count )
    { MTXMultVecArraySR ( (MtxPtr)(m->mtx), srcBase, dstBase, count ); }

/*---------------------------------------------------------------------------*
    MODEL MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// \brief Sets a rotation 3x4 matrix from a quaternion.
///
/// \param m Matrix to be set.
/// \param q Pointer to Quaternion
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34Quat       ( Mat34 *m, const Quaternion *q )
    { MTXQuat       ( (MtxPtr)(m->mtx), q ); }

/// \brief Reflect a rotation 3x4 matrix with respect to a plane.
///
/// \param m Matrix to be set.
/// \param p point on the plane
/// \param n normal of the plane
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34Reflect    ( Mat34 *m, const Vec *p, const Vec *n )
    { MTXReflect    ( (MtxPtr)(m->mtx), p, n ); }

/// \brief Sets a translation 3x4 matrix
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
static inline void MAT34Trans      ( Mat34 *m, f32 xT, f32 yT, f32 zT )
    { MTXTrans      ( (MtxPtr)(m->mtx), xT, yT, zT ); }

/// \brief Apply a translation to a 3x4 matrix.
/// This function is equivalent to \ref MTXTrans + \ref MTXConcat.
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
static inline void MAT34TransApply ( const Mat34 *src, Mat34 *dst, f32 xT, f32 yT, f32 zT )
    { MTXTransApply ( (MtxPtr)(src->mtx), (MtxPtr)(dst->mtx), xT, yT, zT ); }

/// \brief Sets a scale 3x4 matrix
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
static inline void MAT34Scale      ( Mat34 *m, f32 xS, f32 yS, f32 zS )
    { MTXScale      ( (MtxPtr)(m->mtx), xS, yS, zS ); }

/// \brief Apply a scale to a 3x4 matrix.
/// This function is equivalent to \ref MTXScale + \ref MTXConcat.
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
static inline void MAT34ScaleApply ( const Mat34 *src, Mat34 *dst, f32 xS, f32 yS, f32 zS )
    { MTXScaleApply ( (MtxPtr)(src->mtx), (MtxPtr)(dst->mtx), xS, yS, zS ); }


/// \brief Sets a rotation 3x4 matrix about one of the X, Y or Z axes.
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
static inline void MAT34RotRad     ( Mat34 *m, char axis, f32 rad )
    { MTXRotRad     ( (MtxPtr)(m->mtx), axis, rad ); }

/// \brief Sets a rotation 3x4 matrix about one of the X, Y, or Z axes from specified trig ratios.
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
static inline void MAT34RotTrig    ( Mat34 *m, char axis, f32 sinA, f32 cosA )
    { MTXRotTrig    ( (MtxPtr)(m->mtx), axis, sinA, cosA ); }

/// \brief Sets a rotation 3x4 matrix about an arbitrary axis.
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
static inline void MAT34RotAxisRad ( Mat34 *m, const Vec *axis, f32 rad )
    { MTXRotAxisRad ( (MtxPtr)(m->mtx), axis, rad ); }

/// \brief Sets a rotation 3x4 matrix about one of the X, Y or Z axes.
///
/// \note Counter clockwise rotation is positive.
///
/// \param m Matrix to be set
/// \param axis Principal axis of rotation. Must be 'X', 'x', 'Y', 'y', 'Z', or 'z'.
/// \param deg Rotation angle in degrees
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
#define MAT34RotDeg( m, axis, deg ) \
    MAT34RotRad( m, axis, MTXDegToRad(deg) )

/// \brief Sets a rotation 3x4 matrix about an arbitrary axis.
///
/// \note Counter clockwise rotation is positive.
///
/// \param m Matrix to be set
/// \param axis Pointer to a vector containing the (x, y, z) axis components
/// \param deg Rotation angle in degrees
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
#define MAT34RotAxisDeg( m, axis, deg ) \
    MAT34RotAxisRad( m, axis, MTXDegToRad(deg) )

/*---------------------------------------------------------------------------*
    VIEW MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// \brief Compute a 3x4 matrix to transform points to camera coordinates.
///
/// \param m Matrix to be set
/// \param camPos Camera position
/// \param camUp Camera 'up' direction
/// \param target Camera aim point
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34LookAt ( Mat34         *m,
                                 const Point3d *camPos,
                                 const Vec     *camUp,
                                 const Point3d *target )
    { MTXLookAt( (MtxPtr)(m->mtx), camPos, camUp, target ); }

/*---------------------------------------------------------------------------*
    PROJECTION MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// \brief Compute a 4x4 perspective projection matrix from a specified view volume.
///
/// \param m 4x4 Matrix to be set
/// \param t Top coordinate of the viewing volume at the near clipping plane
/// \param b Bottom coordinate of the viewing volume at the near clipping plane
/// \param l Left coordinate of the viewing volume at the near clipping plane
/// \param r Right coordinate of the viewing volume at the near clipping plane
/// \param n Positive distance from camera to the near clipping plane
/// \param f Positive distance from camera to the far clipping plane
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT44Frustum     ( Mat44 *m, f32 t, f32 b, f32 l, f32 r, f32 n, f32 f )
    { MTXFrustum     ( (Mtx44Ptr)(m->mtx), t, b, l, r, n, f ); }

/// \brief Compute a 4x4 perspective projection matrix from the field of view and aspect ratio.
///
/// \param m 4x4 Matrix to be set
/// \param fovY Total field of view in degrees in the YZ plane
/// \param aspect Ratio of view window width:height (X / Y)
/// \param n Positive distance from camera to the near clipping plane
/// \param f Positive distance from camera to the far clipping plane
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT44Perspective ( Mat44 *m, f32 fovY, f32 aspect, f32 n, f32 f )
    { MTXPerspective ( (Mtx44Ptr)(m->mtx), fovY, aspect, n, f ); }

/// \brief Compute a 4x4 orthographic projection matrix.
///
/// \param m 4x4 Matrix to be set
/// \param t Top coordinate of the parallel view volume.
/// \param b Bottom coordinate of the parallel view volume.
/// \param l Left coordinate of the parallel view volume.
/// \param r Right coordinate of the parallel view volume.
/// \param n Positive distance from camera to the near clipping plane
/// \param f Positive distance from camera to the far clipping plane
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT44Ortho       ( Mat44 *m, f32 t, f32 b, f32 l, f32 r, f32 n, f32 f )
    { MTXOrtho       ( (Mtx44Ptr)(m->mtx), t, b, l, r, n, f ); }

/*---------------------------------------------------------------------------*
    TEXTURE PROJECTION MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// \brief Compute a 3x4 perspective projection matrix for texture projection.
///
/// \param m Matrix to be set
/// \param t Top coordinate of the viewing volume at the near clipping plane
/// \param b Bottom coordinate of the viewing volume at the near clipping plane
/// \param l Left coordinate of the viewing volume at the near clipping plane
/// \param r Right coordinate of the viewing volume at the near clipping plane
/// \param n Positive distance from camera to the near clipping plane
/// \param scaleS Scale in the S direction for projected coordinates (usually 0.5)
/// \param scaleT Scale in the T direction for projected coordinates (usually 0.5)
/// \param transS Translate in the S direction for projected coordinates (usually 0.5)
/// \param transT Translate in the T direction for projected coordinates (usually 0.5)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34LightFrustum     ( Mat34 *m, f32 t, f32 b, f32 l, f32 r, f32 n,
                                           f32 scaleS, f32 scaleT, f32 transS,
                                           f32 transT )
    { MTXLightFrustum( (MtxPtr)(m->mtx), t, b, l, r, n, scaleS, scaleT, transS, transT ); }

/// \brief Compute a 3x4 perspective projection matrix from field of view and aspect ratio for texture projection.
///
/// \param m Matrix to be set
/// \param fovY Total field of view in degrees in the YZ plane
/// \param aspect Ratio of view window width:height (X / Y)
/// \param scaleS Scale in the S direction for projected coordinates (usually 0.5)
/// \param scaleT Scale in the T direction for projected coordinates (usually 0.5)
/// \param transS Translate in the S direction for projected coordinates (usually 0.5)
/// \param transT Translate in the T direction for projected coordinates (usually 0.5)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34LightPerspective ( Mat34 *m, f32 fovY, f32 aspect, f32 scaleS,
                                           f32 scaleT, f32 transS, f32 transT )
    { MTXLightPerspective( (MtxPtr)(m->mtx), fovY, aspect, scaleS, scaleT, transS, transT ); }

/// \brief Compute a 3x4 orthographic projection matrix for texture projection.
///
/// \param m Matrix to be set
/// \param t Top coordinate of the viewing volume at the near clipping plane
/// \param b Bottom coordinate of the viewing volume at the near clipping plane
/// \param l Left coordinate of the viewing volume at the near clipping plane
/// \param r Right coordinate of the viewing volume at the near clipping plane
/// \param scaleS Scale in the S direction for projected coordinates (usually 0.5)
/// \param scaleT Scale in the T direction for projected coordinates (usually 0.5)
/// \param transS Translate in the S direction for projected coordinates (usually 0.5)
/// \param transT Translate in the T direction for projected coordinates (usually 0.5)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void MAT34LightOrtho       ( Mat34 *m, f32 t, f32 b, f32 l, f32 r, f32 scaleS,
                                           f32 scaleT, f32 transS, f32 transT )
    { MTXLightOrtho( (MtxPtr)(m->mtx), t, b, l, r, scaleS, scaleT, transS, transT ); }

/*---------------------------------------------------------------------------*
    SPECIAL PURPOSE MATRIX SECTION
 *---------------------------------------------------------------------------*/
/// \brief Creates a reordered (column-major) matrix from a row-major matrix.
/// This is useful for getting better performance for the MTXRO* functions.
///
/// \warning It is not safe to have src == dst.
///
/// \param src Source matrix
/// \param dest Destination matrix, note type is ROMtx.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void PSMAT34Reorder        ( const Mat34 *src, Mat43 *dest )
    { MTXReorder        ( (MtxPtr)(src->mtx), (ROMtxPtr)(dest->mtx) ); }

/// \brief Multiplies an array of vectors by a reordered matrix.
///
/// \note It is ok for source == destination.
/// \note Number of vertices transformed cannot be less than 2.
///
/// \param m Reordered matrix
/// \param srcBase Start of source vector array
/// \param dstBase Start of the resulting vector array
/// \param count Number of vectors in srcBase and dstBase arrays. Count must be greater than 2.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void PSMAT43MultVecArray ( const Mat43 *m, const Vec *srcBase, Vec *dstBase, u32 count )
    { MTXROMultVecArray ( (ROMtxPtr)(m->mtx), srcBase, dstBase, count ); }


/*---------------------------------------------------------------------------*
    MATRIX STACK SECTION
 *---------------------------------------------------------------------------*/
/// \brief Initializes a matrix stack size and stack ptr from a previously allocated stack
/// This resets the stack pointer to NULL(empty) and updates the stack size.
///
/// \note The stack (array) memory must have been previously allocated. Use \ref MTXAllocStack and \ref MTXFreeStack to create/destroy the stack.
///
/// \param sPtr Pointer to \ref Mat34Stack structure to be initialized
/// \param numMat34 Number of matrices in the stack
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline void     MAT34InitStack    ( Mat34Stack *sPtr, u32 numMat34 )
    { MTXInitStack    ( (MtxStackPtr)(void *)(sPtr), numMat34 ); }

/// \brief Copy a matrix to stack pointer + 1.
/// Increment the stack pointer.
///
/// \param sPtr Pointer to Mat34Stack structure
/// \param m Matrix to copy into (stack pointer + 1) location
/// \return Returns the resulting stack pointer
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline Mat34 *  MAT34Push         ( Mat34Stack *sPtr, const Mat34 *m )
    { return (Mat34 *)(void *) MTXPush         ( (MtxStackPtr)(void *)(sPtr), (MtxPtr)(m->mtx) ); }

/// \brief Concatenate a matrix with the current top of the stack push
/// the resulting matrix onto the stack.
/// This is intended for use in building forward transformations, so 
/// concatentation is post-order:
///
/// (top of stack + 1) = (top of stack x m);
///
/// \param sPtr Pointer to Mat34Stack structure
/// \param m Matrix to concatenate with stack pointer and push to (stack pointer + 1)
/// \return Returns the resulting stack pointer
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline Mat34 *  MAT34PushFwd      ( Mat34Stack *sPtr, const Mat34 *m )
    { return (Mat34 *)(void *) MTXPushFwd      ( (MtxStackPtr)(void *)(sPtr), (MtxPtr)(m->mtx) ); }

/// \brief Concatenate the ilwerse of a matrix with the top of the stack
/// and push the resulting matrix onto the stack.
/// This is intended for building ilwerse transformations so concatenation
/// is pre-order:
///
/// (top of stack + 1) = (m x top of stack);
///
/// \note m is not modified by this function.
///
/// \param sPtr Pointer to Mat34Stack structure
/// \param m  Matrix to ilwerse-concatenate with stack pointer and push to (stack pointer + 1)
/// \return Returns the resulting stack pointer
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline Mat34 *  MAT34PushIlw      ( Mat34Stack *sPtr, const Mat34 *m )
    { return (Mat34 *)(void *) MTXPushIlw      ( (MtxStackPtr)(void *)(sPtr), (MtxPtr)(m->mtx) ); }

/// \brief Concatenate the ilwerse-transpose of a matrix with the top of the
/// stack and push the resulting matrix onto the stack.
/// This is intended for building ilwerse-transpose matrix for forward
/// transformations of normals, so concatenation is post-order:
///
/// (top of stack + 1) = (top of stack x m);
///
/// \param sPtr Pointer to Mat34Stack structure
/// \param m  Matrix to ilwerse-concatenate with stack pointer and push to (stack pointer + 1)
/// \return Returns the resulting stack pointer
///
/// \par Usage
/// \note m is not modified by this function.
///
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline Mat34 *  MAT34PushIlwXpose ( Mat34Stack *sPtr, const Mat34 *m )
    { return (Mat34 *)(void *) MTXPushIlwXpose ( (MtxStackPtr)(void *)(sPtr), (MtxPtr)(m->mtx) ); }

/// \brief Decrement the stack pointer.
///
/// \param sPtr Pointer to Mat34Stack structure
/// \return Returns the stack pointer.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline Mat34 *  MAT34Pop          ( Mat34Stack *sPtr )
    { return (Mat34 *)(void *) MTXPop          ( (MtxStackPtr)(void *)(sPtr) ); }

/// \brief Return the stack pointer.
///
/// \param sPtr Pointer to Mat34Stack structure
/// \return Returns the current stack pointer
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
static inline Mat34 *  MAT34GetStackPtr  ( const Mat34Stack *sPtr )
    { return (Mat34 *)(void *) MTXGetStackPtr  ( (MtxStackPtr)(void *)(sPtr) ); }

/// \brief Macro to create a matrix stack.
/// \note This allocates using MEMAllocFromDefaultHeap. This can be modified
/// by the user.
///
/// \param sPtr Pointer to Mat34Stack structure
/// \param numMat34 Number of \ref Mtx structures to allocate for the stack.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \notthreadsafe \userheap \devonly \enddonotcall
///
#define MAT34AllocStack( sPtr, numMat34 ) (  ((Mat34Stack *)(sPtr))->stackBase = (Mat34 *)MEMAllocFromDefaultHeap( ( (numMat34) * sizeof(Mat34) ) )  )

/// \brief Macro to free a matrix stack.
/// \note This allocates using MEMFreeToDefaultHeap. This can be modified
/// by the user.
///
/// \param sPtr Pointer to Mat34Stack structure
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \notthreadsafe \userheap \devonly \enddonotcall
///
#define MAT34FreeStack( sPtr )    (  MEMFreeToDefaultHeap( (void*)( ((Mat34Stack *)(sPtr))->stackBase ) )  )

/*---------------------------------------------------------------------------*/

/// @}

#ifdef __cplusplus
}
#endif

#endif // __MATVEC_H__

/*===========================================================================*/

