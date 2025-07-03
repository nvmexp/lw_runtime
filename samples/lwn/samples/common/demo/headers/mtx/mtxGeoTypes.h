/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

#ifndef __GEOTYPES_H__
#define __GEOTYPES_H__

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup VEC
/// @{

// ---------- Vector types:

/// \brief 3D vector, point.
typedef struct
{
	f32 x; ///< X Component
        f32 y; ///< Y Component
        f32 z; ///< Z Component

} Vec, *VecPtr, Point3d, *Point3dPtr;

/// \brief 2D vector, point.
typedef struct Vec2{
    f32 x; ///< X component
    f32 y; ///< Y component
} Vec2;

/// \brief Signed 16bit 3D vector.
typedef struct
{

    s16 x; ///< X component
    s16 y; ///< Y component
    s16 z; ///< Z component

}S16Vec, *S16VecPtr;

/// @}

/// @addtogroup QUAT
/// @{

/// \brief Quaternion.
typedef struct
{
        /// x, y, and z components
	f32 x, y, z, w;

} Quaternion, *QuaternionPtr, Qtrn, *QtrnPtr;

/// @}

/// @addtogroup MTX
/// @{

// ---------- Array versions of matrix:

/// 3x4 matrix.
typedef f32 Mtx[3][4];

/// 3x4 matrix pointer.
typedef f32 (*MtxPtr)[4];


/// 4x3 reordered matrix.
/// This is used in specially optimized functions that utilize the
/// paired-single unit efficiently.
typedef f32 ROMtx[4][3];

/// 4x3 reordered matrix pointer.
/// This is used in specially optimized functions that utilize the
/// paired-single unit efficiently.
typedef f32 (*ROMtxPtr)[3];


/// \brief 4x4 matrix
/// Used for projection matrix
typedef f32 Mtx44[4][4];

/// \brief 4x4 matrix pointer
/// Used for projection matrix
typedef f32 (*Mtx44Ptr)[4];

/// @}

// Matrix-Vector Library (Structure Version)

// Note: pointer types are omitted intentionally due to const issue.
// (Ie, "const MatPtr foo" is not the same as "const Mat *foo".)

// Avoid warning about anonymous members in types below
#ifdef __ghs__
#pragma ghs nowarning 619
#pragma ghs nowarning 620
#endif

/// @addtogroup MAT
/// @{

/// \brief 3x4 matrix
/// This provides an overloaded matrix structure to allow for multiple
/// access methods.
typedef struct _Mat34
{
    union
    {
        struct 
        {
            f32 _00, _01, _02, _03;
            f32 _10, _11, _12, _13;
            f32 _20, _21, _22, _23;
        };
        f32 m[3][4];
        f32 a[12];
        Mtx mtx;
    };
} Mat34;

/// \brief 4x3 reordered matrix
/// This provides an overloaded matrix structure to allow for multiple
/// access methods.
typedef struct _Mat43
{
    union
    {
        struct
        {
            f32 _00, _01, _02;
            f32 _10, _11, _12;
            f32 _20, _21, _22;
            f32 _30, _31, _32;
        };
        f32 m[4][3];
        f32 a[12];
        ROMtx mtx;
    };
} Mat43;

/// \brief 4x4 matrix
/// Used for projection matrix
/// This provides an overloaded matrix structure to allow for multiple
/// access methods.
typedef struct _Mat44
{
    union
    {
        struct 
        {
            f32 _00, _01, _02, _03;
            f32 _10, _11, _12, _13;
            f32 _20, _21, _22, _23;
            f32 _30, _31, _32, _33;
        };
        f32 m[4][4];
        f32 a[16];
        Mtx44 mtx;
    };
} Mat44;

#ifdef __ghs__
#pragma ghs endnowarning
#pragma ghs endnowarning
#endif

/// 3x4 Matrix stack for the \ref Mat34 type.
typedef struct _Mat34Stack
{

    u32    numMtx; ///< Size of the matrix stack.
    Mat34 *stackBase; ///< Base pointer of the stack
    Mat34 *stackPtr; ///< Current stack pointer. NULL if stack is empty.

} Mat34Stack;

/// @}

#ifdef __cplusplus
}
#endif

#endif  // __GEOTYPES_H__


