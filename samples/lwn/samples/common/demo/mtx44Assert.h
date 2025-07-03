/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/



#ifndef __MTX44ASSERT_H__
#define __MTX44ASSERT_H__



#ifdef __cplusplus
extern "C" {
#endif


//---------------------------------------------------------------------------------

// ASSERT messages for 4x4 matrix extensions.

//  4x4 GENERAL MATRIX SECTION
#define MTX44_IDENTITY_1        "MTX44Identity():  NULL Mtx44 'm' "

#define MTX44_COPY_1            "MTX44Copy():  NULL Mtx44Ptr 'src' "
#define MTX44_COPY_2            "MTX44Copy():  NULL Mtx44Ptr 'dst' "

#define MTX44_CONCAT_1          "MTX44Concat():  NULL Mtx44Ptr 'a'  "
#define MTX44_CONCAT_2          "MTX44Concat():  NULL Mtx44Ptr 'b'  "
#define MTX44_CONCAT_3          "MTX44Concat():  NULL Mtx44Ptr 'ab' "

#define MTX44_TRANSPOSE_1       "MTX44Transpose():  NULL Mtx44Ptr 'src' "
#define MTX44_TRANSPOSE_2       "MTX44Transpose():  NULL Mtx44Ptr 'xPose' "

#define MTX44_ILWERSE_1         "MTX44Ilwerse():  NULL Mtx44Ptr 'src' "
#define MTX44_ILWERSE_2         "MTX44Ilwerse():  NULL Mtx44Ptr 'ilw' "

//  MODEL  SECTION
#define MTX44_ROTRAD_1          "MTX44RotRad():  NULL Mtx44Ptr 'm' "

#define MTX44_ROTTRIG_1         "MTX44RotTrig():  NULL Mtx44Ptr 'm' "
#define MTX44_ROTTRIG_2         "MTX44RotTrig():  invalid 'axis' value "

#define MTX44_ROTAXIS_1         "MTX44RotAxisRad():  NULL Mtx44Ptr 'm' "
#define MTX44_ROTAXIS_2         "MTX44RotAxisRad():  NULL VecPtr 'axis' "

#define MTX44_TRANS_1           "MTX44Trans():  NULL Mtx44Ptr 'm' "

#define MTX44_TRANSAPPLY_1      "MTX44TransApply(): NULL Mtx44Ptr 'src' "
#define MTX44_TRANSAPPLY_2      "MTX44TransApply(): NULL Mtx44Ptr 'dst' "
#define MTX44_SCALE_1           "MTX44Scale():  NULL Mtx44Ptr 'm' "

#define MTX44_SCALEAPPLY_1      "MTX44ScaleApply(): NULL Mtx44Ptr 'src' "
#define MTX44_SCALEAPPLY_2      "MTX44ScaleApply(): NULL Mtx44Ptr 'dst' "


//  MATRIX_VECTOR SECTION
#define MTX44_MULTVEC_1         "MTX44MultVec():  NULL Mtx44Ptr 'm' "
#define MTX44_MULTVEC_2         "MTX44MultVec():  NULL VecPtr 'src' "
#define MTX44_MULTVEC_3         "MTX44MultVec():  NULL VecPtr 'dst' "

#define MTX44_MULTVECARRAY_1    "MTX44MultVecArray():  NULL Mtx44Ptr 'm' "
#define MTX44_MULTVECARRAY_2    "MTX44MultVecArray():  NULL VecPtr 'srcBase' "
#define MTX44_MULTVECARRAY_3    "MTX44MultVecArray():  NULL VecPtr 'dstBase' "

#define MTX44_MULTVECSR_1       "MTX44MultVecSR():  NULL Mtx44Ptr 'm' "
#define MTX44_MULTVECSR_2       "MTX44MultVecSR():  NULL VecPtr 'src' "
#define MTX44_MULTVECSR_3       "MTX44MultVecSR():  NULL VecPtr 'dst' "

#define MTX44_MULTVECARRAYSR_1  "MTX44MultVecArraySR():  NULL Mtx44Ptr 'm' "
#define MTX44_MULTVECARRAYSR_2  "MTX44MultVecArraySR():  NULL VecPtr 'srcBase' "
#define MTX44_MULTVECARRAYSR_3  "MTX44MultVecArraySR():  NULL VecPtr 'dstBase' "



//---------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // __MTX44ASSERT_H__

