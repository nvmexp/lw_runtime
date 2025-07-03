/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

/// @defgroup VEC VEC
/// @ingroup MTXL

/// @defgroup QUAT QUAT
/// @ingroup MTXL

/// @defgroup MTX MTX
/// @ingroup MTXL

/*---------------------------------------------------------------------------*
    Matrix-Vector Library
 *---------------------------------------------------------------------------*/

#ifndef __MTXVEC_H__
#define __MTXVEC_H__

#if defined(CAFE)
#include <cafe/os.h>
#else
#include <types.h>
#ifdef _DEBUG
#ifndef ASSERTMSG
#define ASSERTMSG(exp, msg)                                     \
    (void) ((exp) ||                                            \
            (printf("'%s' at %s:%d", msg, __FILE__, __LINE__)))
#endif
#else   // _DEBUG
#ifndef ASSERTMSG
#define ASSERTMSG(exp, msg)                                     ((void) 0)
#endif

#endif   // _DEBUG
#define MTX_USE_C
#endif

#include <mtx/mtxGeoTypes.h>

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
    Macro definitions
 *---------------------------------------------------------------------------*/

// MtxPtr offset to access next Mtx of an array
#define MTX_PTR_OFFSET      3

// Mtx44Ptr offset to access next Mtx44 of an array
#define MTX44_PTR_OFFSET    4

// "const" doesn't really work correctly for 2-dimensional mtx types
// If that changes, then redefine this as "const" (without quotes):
#define MTX_CONST

/// @addtogroup MTX
/// @{

// Degree <--> radian colwersion macros
/// \brief Colwert Degrees to Radians.
/// \param a Degrees
/// \return Radians
///
#define MTXDegToRad(a)   ( (a) *  0.01745329252f )

/// \brief Colwert Radians to Degrees.
/// \param a Radians
/// \return Degrees
///
#define MTXRadToDeg(a)   ( (a) * 57.29577951f )

/// \brief Matrix-element-referencing macro.
/// Insulates user from changes from row-major to column-major and vice-versa.
/// Fully dolwments which index is row, which index is column.
///
/// \param m Matrix (\ref Mtx or \ref Mtx44)
/// \param r Row
/// \param c Column
/// \return Value of matrix at given row/column
///
#define MTXRowCol(m,r,c) ((m)[(r)][(c)])

/*---------------------------------------------------------------------------*
    Typedefs
 *---------------------------------------------------------------------------*/

/// Matrix stack for the \ref Mtx type.
typedef struct _MtxStack
{

    u32    numMtx; ///< Size of the matrix stack.
    MtxPtr stackBase; ///< Base pointer of the matrix stack.
    MtxPtr stackPtr; ///< Current stack pointer. NULL means an empty stack.

} MtxStack, *MtxStackPtr;

/// @}

/*---------------------------------------------------------------------------*
    GENERAL MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{

// C version

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
void    C_MTXIdentity           ( Mtx m );

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
void    C_MTXCopy               ( MTX_CONST Mtx src, Mtx dst );

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
void    C_MTXConcat             ( MTX_CONST Mtx a, MTX_CONST Mtx b, Mtx ab );

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
void    C_MTXConcatArray        ( MTX_CONST Mtx a, MTX_CONST Mtx* srcBase, Mtx* dstBase, u32 count );

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
void    C_MTXTranspose          ( MTX_CONST Mtx src, Mtx xPose );

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
u32     C_MTXIlwerse            ( MTX_CONST Mtx src, Mtx ilw );

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
u32     C_MTXIlwXpose           ( MTX_CONST Mtx src, Mtx ilwX );

/// @}

// PS intrinsics version
void    PSMTXIdentity           ( Mtx m );
void    PSMTXCopy               ( MTX_CONST Mtx src, Mtx dst );
void    PSMTXConcat             ( MTX_CONST Mtx a, MTX_CONST Mtx b, Mtx ab );
void    PSMTXConcatArray        ( MTX_CONST Mtx a, MTX_CONST Mtx* srcBase, Mtx* dstBase, u32 count );
void    PSMTXTranspose          ( MTX_CONST Mtx src, Mtx xPose );
u32     PSMTXIlwerse            ( MTX_CONST Mtx src, Mtx ilw );
u32     PSMTXIlwXpose           ( MTX_CONST Mtx src, Mtx ilwX );

// PS assembler version
void    ASM_MTXIdentity         ( Mtx m);
void    ASM_MTXCopy             ( MTX_CONST Mtx src, Mtx dst );
void    ASM_MTXConcat           ( MTX_CONST Mtx mA, MTX_CONST Mtx mB, Mtx mAB );
void    ASM_MTXConcatArray      ( MTX_CONST Mtx a, MTX_CONST Mtx* srcBase, Mtx* dstBase, u32 count );
void    ASM_MTXTranspose        ( MTX_CONST Mtx src, Mtx xPose );
u32     ASM_MTXIlwerse          ( MTX_CONST Mtx src, Mtx ilw );
u32     ASM_MTXIlwXpose         ( MTX_CONST Mtx src, Mtx ilwX );

// Bindings
#ifdef MTX_USE_ASM
#define MTXIdentity             ASM_MTXIdentity
#define MTXCopy                 ASM_MTXCopy
#define MTXConcat               ASM_MTXConcat
#define MTXConcatArray          ASM_MTXConcatArray
#define MTXTranspose            ASM_MTXTranspose
#define MTXIlwerse              ASM_MTXIlwerse
#define MTXIlwXpose             ASM_MTXIlwXpose
#else
#ifdef MTX_USE_PS
#define MTXIdentity             PSMTXIdentity
#define MTXCopy                 PSMTXCopy
#define MTXConcat               PSMTXConcat
#define MTXConcatArray          PSMTXConcatArray
#define MTXTranspose            PSMTXTranspose
#define MTXIlwerse              PSMTXIlwerse
#define MTXIlwXpose             PSMTXIlwXpose
#else // MTX_USE_C
#define MTXIdentity             C_MTXIdentity
#define MTXCopy                 C_MTXCopy
#define MTXConcat               C_MTXConcat
#define MTXConcatArray          C_MTXConcatArray
#define MTXTranspose            C_MTXTranspose
#define MTXIlwerse              C_MTXIlwerse
#define MTXIlwXpose             C_MTXIlwXpose
#endif
#endif

/*---------------------------------------------------------------------------*
    MATRIX-VECTOR SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{
 
// C version

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
void    C_MTXMultVec            ( MTX_CONST Mtx m, const Vec *src, Vec *dst );

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
void    C_MTXMultVecArray       ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count );

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
void    C_MTXMultVecSR          ( MTX_CONST Mtx m, const Vec *src, Vec *dst );

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
void    C_MTXMultVecArraySR     ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count );

/// @}

// PS intrisics version
void    PSMTXMultVec            ( MTX_CONST Mtx m, const Vec *src, Vec *dst );
void    PSMTXMultVecArray       ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count );
void    PSMTXMultVecSR          ( MTX_CONST Mtx m, const Vec *src, Vec *dst );
void    PSMTXMultVecArraySR     ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count );

// PS assembler version
void   ASM_MTXMultVec           ( MTX_CONST Mtx m, const Vec *src, Vec *dst );
void   ASM_MTXMultVecArray      ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count );
void   ASM_MTXMultVecArraySR    ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count );
void   ASM_MTXMultVecSR         ( MTX_CONST Mtx m, const Vec *src, Vec *dst );

// Bindings
#ifdef MTX_USE_ASM
#define MTXMultVec              ASM_MTXMultVec
#define MTXMultVecArray         ASM_MTXMultVecArray
#define MTXMultVecSR            ASM_MTXMultVecSR
#define MTXMultVecArraySR       ASM_MTXMultVecArraySR
#else
#ifdef MTX_USE_PS
#define MTXMultVec              PSMTXMultVec
#define MTXMultVecArray         PSMTXMultVecArray
#define MTXMultVecSR            PSMTXMultVecSR
#define MTXMultVecArraySR       PSMTXMultVecArraySR
#else // MTX_USE_C
#define MTXMultVec              C_MTXMultVec
#define MTXMultVecArray         C_MTXMultVecArray
#define MTXMultVecSR            C_MTXMultVecSR
#define MTXMultVecArraySR       C_MTXMultVecArraySR
#endif
#endif

/*---------------------------------------------------------------------------*
    MODEL MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{

// C version

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
void    C_MTXQuat             ( Mtx m, const Quaternion *q );

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
void    C_MTXReflect          ( Mtx m, const Vec *p, const Vec *n );

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
void    C_MTXTrans            ( Mtx m, f32 xT, f32 yT, f32 zT );

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
void    C_MTXTransApply       ( MTX_CONST Mtx src, Mtx dst, f32 xT, f32 yT, f32 zT );

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
void    C_MTXScale            ( Mtx m, f32 xS, f32 yS, f32 zS );

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
void    C_MTXScaleApply       ( MTX_CONST Mtx src, Mtx dst, f32 xS, f32 yS, f32 zS );


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
void    C_MTXRotRad           ( Mtx m, char axis, f32 rad );

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
void    C_MTXRotTrig          ( Mtx m, char axis, f32 sinA, f32 cosA );

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
void    C_MTXRotAxisRad       ( Mtx m, const Vec *axis, f32 rad );

/// @}

// PS intrisics version
void    PSMTXQuat             ( Mtx m, const Quaternion *q );
void    PSMTXReflect          ( Mtx m, const Vec *p, const Vec *n );

void    PSMTXTrans            ( Mtx m, f32 xT, f32 yT, f32 zT );
void    PSMTXTransApply( Mtx src, Mtx dst, f32 xT, f32 yT, f32 zT );
void    PSMTXScale            ( Mtx m, f32 xS, f32 yS, f32 zS );
void    PSMTXScaleApply       ( MTX_CONST Mtx src, Mtx dst, f32 xS, f32 yS, f32 zS );

void    PSMTXRotRad           ( Mtx m, char axis, f32 rad );
void    PSMTXRotTrig          ( Mtx m, char axis, f32 sinA, f32 cosA );
void    PSMTXRotAxisRad       ( Mtx m, const Vec *axis, f32 rad );

// PS assembler version
void   ASM_MTXQuat            ( Mtx m, const Quaternion *q );
void   ASM_MTXReflect         ( Mtx m, const Vec *p, const Vec *n );

void   ASM_MTXTrans           ( Mtx m, f32 xT, f32 yT, f32 zT );
void   ASM_MTXTransApply      ( MTX_CONST Mtx src, Mtx dst, f32 xT, f32 yT, f32 zT );
void   ASM_MTXScale           ( Mtx m, f32 xS, f32 yS, f32 zS );
void   ASM_MTXScaleApply      ( MTX_CONST Mtx src, Mtx dst, f32 xS, f32 yS, f32 zS );

void   ASM_MTXRotRad          ( Mtx m, char axis, f32 rad );
void   ASM_MTXRotTrig         ( Mtx m, char axis, f32 sinA, f32 cosA );
void   ASM_MTXRotAxisRad      ( Mtx m, const Vec *axis, f32 rad );

// Bindings

#ifdef MTX_USE_ASM
#define MTXTrans                ASM_MTXTrans
#define MTXTransApply           ASM_MTXTransApply
#define MTXQuat                 ASM_MTXQuat
#define MTXReflect              ASM_MTXReflect
#define MTXScale                ASM_MTXScale
#define MTXScaleApply           ASM_MTXScaleApply
#define MTXRotRad               ASM_MTXRotRad
#define MTXRotTrig              ASM_MTXRotTrig
#define MTXRotDeg( m, axis, deg ) \
    ASM_MTXRotRad( m, axis, MTXDegToRad(deg) )
#define MTXRotAxisRad           ASM_MTXRotAxisRad
#define MTXRotAxisDeg( m, axis, deg ) \
    ASM_MTXRotAxisRad( m, axis, MTXDegToRad(deg) )
#else
#ifdef MTX_USE_PS
#define MTXTrans                PSMTXTrans
#define MTXTransApply           PSMTXTransApply
#define MTXQuat                 PSMTXQuat
#define MTXReflect              PSMTXReflect
#define MTXScale                PSMTXScale
#define MTXScaleApply           PSMTXScaleApply
#define MTXRotRad               PSMTXRotRad
#define MTXRotTrig              PSMTXRotTrig
#define MTXRotDeg( m, axis, deg ) \
    PSMTXRotRad( m, axis, MTXDegToRad(deg) )
#define MTXRotAxisRad           PSMTXRotAxisRad
#define MTXRotAxisDeg( m, axis, deg ) \
    PSMTXRotAxisRad( m, axis, MTXDegToRad(deg) )

#else // MTX_USE_C
#define MTXTrans                C_MTXTrans
#define MTXTransApply           C_MTXTransApply
#define MTXQuat                 C_MTXQuat
#define MTXReflect              C_MTXReflect
#define MTXScale                C_MTXScale
#define MTXScaleApply           C_MTXScaleApply
#define MTXRotRad               C_MTXRotRad
#define MTXRotTrig              C_MTXRotTrig
#define MTXRotDeg( m, axis, deg ) \
    C_MTXRotRad( m, axis, MTXDegToRad(deg) )
#define MTXRotAxisRad           C_MTXRotAxisRad
#define MTXRotAxisDeg( m, axis, deg ) \
    C_MTXRotAxisRad( m, axis, MTXDegToRad(deg) )

#endif
#endif

// Obsolete. Don't use this if possible.
#define MTXRotAxis              MTXRotAxisDeg


/*---------------------------------------------------------------------------*
    VIEW MATRIX SECTION
 *---------------------------------------------------------------------------*/
 
/// @addtogroup MTX
/// @{
 
// C version only so far

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
void    C_MTXLookAt         ( Mtx            m,
                              const Point3d *camPos,
                              const Vec     *camUp,
                              const Point3d *target );

/// @}

// Bindings
#define MTXLookAt               C_MTXLookAt

/*---------------------------------------------------------------------------*
    PROJECTION MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{
 
// C version only so far

/// \brief Compute a 4x4 perspective projection matrix from a specified view volume.
///
/// \param m 4x4 Matrix to be set
/// \param t Top coordinate of the viewing volume at the near clipping plane
/// \param b Bottom coordinate of the viewing volume at the near clipping plane
/// \param lf Left coordinate of the viewing volume at the near clipping plane
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
void    C_MTXFrustum        ( Mtx44 m, f32 t, f32 b, f32 lf, f32 r, f32 n, f32 f );
 
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
void    C_MTXPerspective    ( Mtx44 m, f32 fovY, f32 aspect, f32 n, f32 f );
 
/// \brief Compute a 4x4 orthographic projection matrix.
///
/// \param m 4x4 Matrix to be set
/// \param t Top coordinate of the parallel view volume.
/// \param b Bottom coordinate of the parallel view volume.
/// \param lf Left coordinate of the parallel view volume.
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
void    C_MTXOrtho          ( Mtx44 m, f32 t, f32 b, f32 lf, f32 r, f32 n, f32 f );

/// @}

// Bindings
#define MTXFrustum              C_MTXFrustum
#define MTXPerspective          C_MTXPerspective
#define MTXOrtho                C_MTXOrtho

/*---------------------------------------------------------------------------*
    TEXTURE PROJECTION MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{

// C version only so far
 
/// \brief Compute a 3x4 perspective projection matrix for texture projection.
///
/// \param m Matrix to be set
/// \param t Top coordinate of the viewing volume at the near clipping plane
/// \param b Bottom coordinate of the viewing volume at the near clipping plane
/// \param lf Left coordinate of the viewing volume at the near clipping plane
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
void    C_MTXLightFrustum       ( Mtx m, f32 t, f32 b, f32 lf, f32 r, f32 n,
                                  f32 scaleS, f32 scaleT, f32 transS,
                                  f32 transT );

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
void    C_MTXLightPerspective   ( Mtx m, f32 fovY, f32 aspect, f32 scaleS,
                                  f32 scaleT, f32 transS, f32 transT );

/// \brief Compute a 3x4 orthographic projection matrix for texture projection.
///
/// \param m Matrix to be set
/// \param t Top coordinate of the viewing volume at the near clipping plane
/// \param b Bottom coordinate of the viewing volume at the near clipping plane
/// \param lf Left coordinate of the viewing volume at the near clipping plane
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
void    C_MTXLightOrtho         ( Mtx m, f32 t, f32 b, f32 lf, f32 r, f32 scaleS,
                                  f32 scaleT, f32 transS, f32 transT );

/// @}

// Bindings
#define MTXLightFrustum         C_MTXLightFrustum
#define MTXLightPerspective     C_MTXLightPerspective
#define MTXLightOrtho           C_MTXLightOrtho

/*---------------------------------------------------------------------------*
    VECTOR SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup VEC
/// @{

// C version
 
/// \brief Add two vectors.
///
/// \note It is safe for ab == a == b
///
/// \param a First vector
/// \param b Second vector
/// \param ab Resulting vector (a + b)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void  C_VECAdd              ( const Vec *a, const Vec *b, Vec *ab );
 
/// \brief Subtract two vectors.
///
/// \note It is safe for a_b == a == b
///
/// \param a First vector
/// \param b Second vector
/// \param a_b Resulting vector (a - b)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void  C_VECSubtract         ( const Vec *a, const Vec *b, Vec *a_b );
 
/// \brief Scale a vector using a scalar.
///
/// \note It is safe for src == dst.
///
/// \param src Unscaled source vector
/// \param dst Scaled resultant vector (src * scale)
/// \param scale Scaling factor
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void  C_VECScale            ( const Vec *src, Vec *dst, f32 scale );
 
/// \brief Normalize a vector.
///
/// \note It is safe for src == unit.
///
/// \param src Non-unit source vector
/// \param unit Resulting unit vector (src / src magnitude)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void  C_VECNormalize        ( const Vec *src, Vec *unit );
 
/// \brief Compute the square of the magnitude of a vector.
///
/// \param v Source vector
/// \return Square magnitude of the vector
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
f32   C_VECSquareMag        ( const Vec *v );
 
/// \brief Compute the magnitude of a vector.
///
/// \param v Source vector
/// \return Magnitude of the vector
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
f32   C_VECMag              ( const Vec *v );
 
/// \brief Compute the dot product of two vectors.
///
/// \note Input vectors do not have to be normalized.
/// \note Input vectors are not normalized in the function.
///
/// \warning If direct cosine computation of the angle between a and b is desired, a and b should be normalized prior to calling \ref VECDotProduct.
///
/// \param a First vector
/// \param b Second vector
/// \result Dot product of the two vectors
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
f32   C_VECDotProduct       ( const Vec *a, const Vec *b );
 
/// \brief Compute the cross product of two vectors.
///
/// \note It is safe for axb == a == b
/// \note Input vectors do not have to be normalized.
/// \note Input vectors are not normalized in the function.
///
/// \param a First vector
/// \param b Second vector
/// \param axb Resulting cross product vector (a x b)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void  C_VECCrossProduct     ( const Vec *a, const Vec *b, Vec *axb );
 
/// \brief Returns the square of the distance between vectors a and b.
/// Distance can be callwlated using the square root of the returned value.
///
/// \note It is safe for ab == a == b
///
/// \param a First vector
/// \param b Second vector
/// \return Square distance between the two vectors
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
f32   C_VECSquareDistance   ( const Vec *a, const Vec *b );
 
/// \brief Returns the distance between vectors a and b.
///
/// \note It is safe for ab == a == b
///
/// \param a First vector
/// \param b Second vector
/// \return Distance between the two vectors
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
f32   C_VECDistance         ( const Vec *a, const Vec *b );
 
/// \brief Reflect a vector about a normal to a surface.
///
/// \note It is safe for dst == src.
/// \note This function normalizes src and normal vectors.
///
/// \param src Incident vector
/// \param normal Normal to surface
/// \param dst Normalized reflected vector
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void  C_VECReflect          ( const Vec *src, const Vec *normal, Vec *dst );
 
/// \brief Compute the vector halfway between two vectors.
/// This is intended for use in computing spelwlar highlights.
///
/// \note It is safe for half == a == b
/// \note Input vectors do not have to be normalized.
///
/// \param a First vector. This must point FROM the light source (tail) TO the surface (head).
/// \param b Second vector. This must point FROM the viewer (tail) TO the surface (head).
/// \param half Resulting normalized 'half-angle' vector.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void  C_VECHalfAngle        ( const Vec *a, const Vec *b, Vec *half );

/// @}

// PS intrisics version
void  PSVECAdd              ( const Vec *a, const Vec *b, Vec *ab );
void  PSVECSubtract         ( const Vec *a, const Vec *b, Vec *a_b );
void  PSVECScale            ( const Vec *src, Vec *dst, f32 scale );
void  PSVECNormalize        ( const Vec *src, Vec *unit );
f32   PSVECSquareMag        ( const Vec *v );
f32   PSVECMag              ( const Vec *v );
f32   PSVECDotProduct       ( const Vec *a, const Vec *b );
void  PSVECCrossProduct     ( const Vec *a, const Vec *b, Vec *axb );
f32   PSVECSquareDistance   ( const Vec *a, const Vec *b );
f32   PSVECDistance         ( const Vec *a, const Vec *b );

// PS assembler version
void  ASM_VECAdd            ( const Vec *vec1, const Vec *vec2, Vec *dst );
void  ASM_VECSubtract       ( const Vec *vec1, const Vec *vec2, Vec *dst );
void  ASM_VECScale          ( const Vec *src, Vec *dst, f32 mult );
void  ASM_VECNormalize      ( const Vec *src, Vec *unit );
f32   ASM_VECSquareMag      ( const Vec *vec1 );
f32   ASM_VECMag            ( const Vec *v );
f32   ASM_VECDotProduct     ( const Vec *a, const Vec *b );
void  ASM_VECCrossProduct   ( const Vec *vec1, const Vec *vec2, Vec *dst );
f32   ASM_VECSquareDistance ( const Vec* a, const Vec* b );
f32   ASM_VECDistance       ( const Vec *a, const Vec *b );

// Bindings
#ifdef MTX_USE_ASM
#define VECNormalize            ASM_VECNormalize
#define VECDotProduct           ASM_VECDotProduct
#define VECAdd                  ASM_VECAdd
#define VECSubtract             ASM_VECSubtract
#define VECScale                ASM_VECScale
#define VECSquareMag            ASM_VECSquareMag
#define VECMag                  ASM_VECMag
#define VECCrossProduct         ASM_VECCrossProduct
#define VECSquareDistance       ASM_VECSquareDistance
#define VECDistance             ASM_VECDistance
#else
#ifdef MTX_USE_PS
#define VECNormalize            PSVECNormalize
#define VECDotProduct           PSVECDotProduct
#define VECAdd                  PSVECAdd
#define VECSubtract             PSVECSubtract
#define VECScale                PSVECScale
#define VECSquareMag            PSVECSquareMag
#define VECMag                  PSVECMag
#define VECCrossProduct         PSVECCrossProduct
#define VECSquareDistance       PSVECSquareDistance
#define VECDistance             PSVECDistance
#else // MTX_USE_C
#define VECNormalize            C_VECNormalize
#define VECDotProduct           C_VECDotProduct
#define VECAdd                  C_VECAdd
#define VECSubtract             C_VECSubtract
#define VECScale                C_VECScale
#define VECSquareMag            C_VECSquareMag
#define VECMag                  C_VECMag
#define VECCrossProduct         C_VECCrossProduct
#define VECSquareDistance       C_VECSquareDistance
#define VECDistance             C_VECDistance
#endif
#endif

#define VECReflect              C_VECReflect
#define VECHalfAngle            C_VECHalfAngle

/*---------------------------------------------------------------------------*
    QUATERNION SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup QUAT
/// @{
 
/// \brief Quaternion epsilon comparison used for compare against 0.0f.
///
#define QUAT_EPSILON        0.00001F

// C version
 
/// \brief Returns the sum of two quaternions.
///
/// \note It is safe for p == q == r
///
/// \param p First quaternion
/// \param q Second quaternion
/// \param r Resulting quaternion (p + q)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATAdd              ( const Quaternion *p, const Quaternion *q, Quaternion *r );
 
/// \brief Returns the difference of two quaternions p-q.
///
/// \note It is safe for p == q == r
///
/// \param p First quaternion
/// \param q Second quaternion
/// \param r Resulting quaternion (p - q)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATSubtract         ( const Quaternion *p, const Quaternion *q, Quaternion *r );
 
/// \brief Returns the product of two quaternions.
/// The order of multiplication is important. (p*q != q*p)
///
/// \note It is safe for p == q == pq
///
/// \param p Left quaternion
/// \param q Right quaternion
/// \param pq Resulting quaternion product (p * q)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATMultiply         ( const Quaternion *p, const Quaternion *q, Quaternion *pq );
 
/// \brief Returns the ratio of two quaternions.
/// Creates a result r = p/q such that q*r = p (order of multiplication is important).
///
/// \note It is safe for p == q == r
///
/// \param p Left quaternion
/// \param q Right quaternion
/// \param r Resulting quaternion ratio (p / q)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATDivide           ( const Quaternion *p, const Quaternion *q, Quaternion *r );
 
/// \brief Scales a quaternion.
///
/// \note It is safe for q == r
///
/// \param q Quaternion
/// \param r Resulting scaled quaternion
/// \param scale Scaling factor
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATScale            ( const Quaternion *q, Quaternion *r, f32 scale );
 
/// \brief Returns the dot product of two quaternions.
///
/// \param p First quaternion
/// \param q Second quaternion
/// \return Dot product of the two quaternions
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
f32  C_QUATDotProduct       ( const Quaternion *p, const Quaternion *q );
 
/// \brief Normalizes a quaternion.
///
/// \note It is safe for src == unit
/// \warning If using MTX_USE_C and the magnitude of the quaternion is < \ref QUAT_EPSILON then the resulting quaternion is 0.
///
/// \param src Source quaternion
/// \param unit Resulting unit quaternion
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATNormalize        ( const Quaternion *src, Quaternion *unit );
 
/// \brief Returns the ilwerse of a quaternion.
///
/// \param src Source quaternion
/// \param ilw Resulting ilwerse quaternion
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATIlwerse          ( const Quaternion *src, Quaternion *ilw );
 
/// \brief Exponentiate quaternion (where q.w == 0).
///
/// \note It is safe for q == r
///
/// \param q Pure quaternion
/// \param r Resulting exponentiated quaternion (an unit quaternion)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATExp              ( const Quaternion *q, Quaternion *r );
 
/// \brief Returns the natural logarithm of a UNIT quaternion.
///
/// \note It is safe for q == r
///
/// \param q Unit quaternion
/// \param r Resulting logarithm quaternion (a pure quaternion)
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATLogN             ( const Quaternion *q, Quaternion *r );

/// \brief Modify q so it is on the same side of the hypersphere as qto
///
/// \note It is safe for p == q == r
///
/// \param q Quaternion
/// \param qto Quaternion to be close to
/// \param r Resulting modified quaternion
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATMakeClosest      ( const Quaternion *q, const Quaternion *qto, Quaternion *r );
 
/// \brief Returns the sum of two quaternions.
///
/// \note It is safe for p == q == r
///
/// \param r Resultng rotation quaternion
/// \param axis Rotation axis
/// \param rad Rotation angle in radians
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATRotAxisRad       ( Quaternion *r, const Vec *axis, f32 rad );
 
/// \brief Colwerts a matrix to a unit quaternion.
///
/// \note It is safe for p == q == r
///
/// \param r Resulting quaternion
/// \param m Input matrix
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATMtx              ( Quaternion *r, MTX_CONST Mtx m );

/// \brief Linear interpolation between two quaternions.
///
/// \note It is safe for p == q == r
///
/// \param p First quaternion
/// \param q Second quaternion
/// \param r Resulting quaternion (q*t + (1 - t) * p)
/// \param t Interpolation parameter
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATLerp             ( const Quaternion *p, const Quaternion *q, Quaternion *r, f32 t );
 
/// \brief Spherical linear interpolation of two quaternions
///
/// \note It is safe for p == q == r
///
/// \param p First quaternion
/// \param q Second quaternion
/// \param r Resulting interpolated quaternion (p + q)
/// \param t Interpolation parameter
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATSlerp            ( const Quaternion *p, const Quaternion *q, Quaternion *r, f32 t );
 
/// \brief Spherical cubic quadrangle interpolation of two quaternions with derrived inner-quadrangle quaternions.
/// This will be used with the function \ref QUATCompA.
///
/// \note It is safe for p == q == r
///
/// \param p First quaternion
/// \param a Derrived inner-quadrangle quaternion
/// \param b Derrived inner-quadrangle quaternion
/// \param q Second quaternion
/// \param r Resulting quaternion (p + q)
/// \param t Interpolation value
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATSquad            ( const Quaternion *p, const Quaternion *a, const Quaternion *b,
                              const Quaternion *q, Quaternion *r, f32 t );
 
/// \brief Compute a, the term used in Boehm-type interpolation.
///
/// a[n] = q[n] * qexp(-(1/4) * (logN(qilw(q[n])*q[n+1]) + logN(qilw(q[n])*q[n-1])))
///
/// \note This is safe for the case where qprev == q == qnext == a.
///
/// \param qprev Previous quaternion
/// \param q Current quaternion
/// \param qnext Next quaternion
/// \param a Resulting quaternion A
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void C_QUATCompA            ( const Quaternion *qprev, const Quaternion *q,
                              const Quaternion *qnext, Quaternion *a );

/// @}
                              
// PS intrisics version
void PSQUATAdd              ( const Quaternion *p, const Quaternion *q, Quaternion *r );
void PSQUATSubtract         ( const Quaternion *p, const Quaternion *q, Quaternion *r );
void PSQUATMultiply         ( const Quaternion *p, const Quaternion *q, Quaternion *pq );
void PSQUATDivide           ( const Quaternion *p, const Quaternion *q, Quaternion *r );
void PSQUATScale            ( const Quaternion *q, Quaternion *r, f32 scale );
f32  PSQUATDotProduct       ( const Quaternion *p, const Quaternion *q );
void PSQUATNormalize        ( const Quaternion *src, Quaternion *unit );
void PSQUATIlwerse          ( const Quaternion *src, Quaternion *ilw );

// PS assembler version
void ASM_QUATAdd            ( const Quaternion *p, const Quaternion *q, Quaternion *r );
void ASM_QUATDivide         ( const Quaternion *p, const Quaternion *q, Quaternion *r );
f32  ASM_QUATDotProduct     ( const Quaternion *p, const Quaternion *q );
void ASM_QUATIlwerse        ( const Quaternion *src, Quaternion *ilw );
void ASM_QUATMultiply       ( const Quaternion *p, const Quaternion *q, Quaternion *pq );
void ASM_QUATNormalize      ( const Quaternion *src, Quaternion *unit );
void ASM_QUATScale          ( const Quaternion *q, Quaternion *r, f32 scale );
void ASM_QUATSubtract       ( const Quaternion *p, const Quaternion *q, Quaternion *r );

// Bindings
#ifdef MTX_USE_ASM
#define QUATAdd                 ASM_QUATAdd
#define QUATSubtract            ASM_QUATSubtract
#define QUATMultiply            ASM_QUATMultiply
#define QUATDivide              ASM_QUATDivide
#define QUATScale               ASM_QUATScale
#define QUATDotProduct          ASM_QUATDotProduct
#define QUATNormalize           ASM_QUATNormalize
#define QUATIlwerse             ASM_QUATIlwerse
#else
#ifdef MTX_USE_PS
#define QUATAdd                 PSQUATAdd
#define QUATSubtract            PSQUATSubtract
#define QUATMultiply            PSQUATMultiply
#define QUATDivide              PSQUATDivide
#define QUATScale               PSQUATScale
#define QUATDotProduct          PSQUATDotProduct
#define QUATNormalize           PSQUATNormalize
#define QUATIlwerse             PSQUATIlwerse
#else // MTX_USE_C
#define QUATAdd                 C_QUATAdd
#define QUATSubtract            C_QUATSubtract
#define QUATMultiply            C_QUATMultiply
#define QUATDivide              C_QUATDivide
#define QUATScale               C_QUATScale
#define QUATDotProduct          C_QUATDotProduct
#define QUATNormalize           C_QUATNormalize
#define QUATIlwerse             C_QUATIlwerse
#endif
#endif

#define QUATExp                 C_QUATExp
#define QUATLogN                C_QUATLogN
#define QUATMakeClosest         C_QUATMakeClosest
#define QUATRotAxisRad          C_QUATRotAxisRad
#define QUATMtx                 C_QUATMtx
#define QUATLerp                C_QUATLerp
#define QUATSlerp               C_QUATSlerp
#define QUATSquad               C_QUATSquad
#define QUATCompA               C_QUATCompA

/*---------------------------------------------------------------------------*
    MATRIX STACK SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{

 
/// \brief Initializes a matrix stack size and stack ptr from a previously allocated stack
/// This resets the stack pointer to NULL(empty) and updates the stack size.
///
/// \note The stack (array) memory must have been previously allocated. Use \ref MTXAllocStack and \ref MTXFreeStack to create/destroy the stack.
///
/// \param sPtr Pointer to \ref MtxStack structure to be initialized
/// \param numMtx Number of matrices in the stack
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
void    MTXInitStack          ( MtxStack *sPtr, u32 numMtx );
 
/// \brief Copy a matrix to stack pointer + 1.
/// Increment the stack pointer.
///
/// \param sPtr Pointer to MtxStack structure
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
MtxPtr  MTXPush               ( MtxStack *sPtr, MTX_CONST Mtx m );
 
/// \brief Concatenate a matrix with the current top of the stack push
/// the resulting matrix onto the stack.
/// This is intended for use in building forward transformations, so 
/// concatentation is post-order:
///
/// (top of stack + 1) = (top of stack x m);
///
/// \param sPtr Pointer to MtxStack structure
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
MtxPtr  MTXPushFwd            ( MtxStack *sPtr, MTX_CONST Mtx m );
 
/// \brief Concatenate the ilwerse of a matrix with the top of the stack
/// and push the resulting matrix onto the stack.
/// This is intended for building ilwerse transformations so concatenation
/// is pre-order:
///
/// (top of stack + 1) = (m x top of stack);
///
/// \note m is not modified by this function.
///
/// \param sPtr Pointer to MtxStack structure
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
MtxPtr  MTXPushIlw            ( MtxStack *sPtr, MTX_CONST Mtx m );
 
/// \brief Concatenate the ilwerse-transpose of a matrix with the top of the
/// stack and push the resulting matrix onto the stack.
/// This is intended for building ilwerse-transpose matrix for forward
/// transformations of normals, so concatenation is post-order:
///
/// (top of stack + 1) = (top of stack x m);
///
/// \param sPtr Pointer to MtxStack structure
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
MtxPtr  MTXPushIlwXpose       ( MtxStack *sPtr, MTX_CONST Mtx m );
 
/// \brief Decrement the stack pointer.
///
/// \param sPtr Pointer to MtxStack structure
/// \return Returns the stack pointer.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
MtxPtr  MTXPop                ( MtxStack *sPtr );
 
/// \brief Return the stack pointer.
///
/// \param sPtr Pointer to MtxStack structure
/// \return Returns the current stack pointer
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \threadsafe \devonly \enddonotcall
///
MtxPtr  MTXGetStackPtr        ( const MtxStack *sPtr );

/// \brief Macro to create a matrix stack.
/// \note This allocates using MEMAllocFromDefaultHeap. This can be modified
/// by the user.
///
/// \param sPtr Pointer to MtxStack structure
/// \param numMtx Number of \ref Mtx structures to allocate for the stack.
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \notthreadsafe \userheap \devonly \enddonotcall
///
#define MTXAllocStack( sPtr, numMtx ) (  ((MtxStackPtr)(sPtr))->stackBase = (MtxPtr)MEMAllocFromDefaultHeap( ( (numMtx) * sizeof(Mtx) ) )  )

/// \brief Macro to free a matrix stack.
/// \note This allocates using MEMFreeToDefaultHeap. This can be modified
/// by the user.
///
/// \param sPtr Pointer to MtxStack structure
///
/// \par Usage
///  - Add "#define MTX_USE_ASM" prior to including the mtx header to use PPC Assembly based version
///  - Add "#define MTX_USE_PS" prior to including the mtx header to use PPC Paired-Single instruction based version
///  - Add "#define MTX_USE_C" prior to including the mtx header to use C implementation (default). This is useful for debugging.
///
/// \donotcall \notthreadsafe \userheap \devonly \enddonotcall
///
#define MTXFreeStack( sPtr )    (  MEMFreeToDefaultHeap( (void*)( ((MtxStackPtr)(sPtr))->stackBase ) )  )

/// @}

/*---------------------------------------------------------------------------*
    SPECIAL PURPOSE MATRIX SECTION
 *---------------------------------------------------------------------------*/

/// @addtogroup MTX
/// @{
 
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
void C_MTXReorder       ( MTX_CONST Mtx src, ROMtx dest );
 
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
void C_MTXROMultVecArray( MTX_CONST ROMtx m,  const Vec *srcBase, Vec *dstBase, u32 count );

/// @}

void PSMTXReorder       ( MTX_CONST Mtx src, ROMtx dest );
void ASM_MTXReorder     ( MTX_CONST Mtx src, ROMtx dest );

// Bindings
#ifdef MTX_USE_ASM
#define MTXReorder            ASM_MTXReorder
#else
#ifdef MTX_USE_PS
#define MTXReorder            PSMTXReorder
#else
#define MTXReorder            C_MTXReorder
#endif
#endif

#define MTXROMultVecArray     C_MTXROMultVecArray

/*---------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif // __MTXVEC_H__

/*===========================================================================*/

