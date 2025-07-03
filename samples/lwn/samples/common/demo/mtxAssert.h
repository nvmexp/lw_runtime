/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/


#ifndef __MTXASSERT_H__
#define __MTXASSERT_H__


#ifdef __cplusplus
extern "C" {
#endif


//---------------------------------------------------------------------------------

//     GENERAL   SECTION

#define	MTX_IDENTITY_1			"MtxIdentity():  NULL Mtx 'm' "

#define	MTX_COPY_1				"MTXCopy():  NULL MtxPtr 'src' "
#define MTX_COPY_2				"MTXCopy():  NULL MtxPtr 'dst' "

#define	MTX_CONCAT_1			"MTXConcat():  NULL MtxPtr 'a'  "
#define	MTX_CONCAT_2			"MTXConcat():  NULL MtxPtr 'b'  "
#define	MTX_CONCAT_3			"MTXConcat():  NULL MtxPtr 'ab' "

#define	MTX_TRANSPOSE_1			"MTXTranspose():  NULL MtxPtr 'src' "
#define	MTX_TRANSPOSE_2			"MTXTranspose():  NULL MtxPtr 'xPose' "

#define	MTX_ILWERSE_1			"MTXIlwerse():  NULL MtxPtr 'src' "
#define	MTX_ILWERSE_2			"MTXIlwerse():  NULL MtxPtr 'ilw' "

#define MTX_ILWXPOSE_1          "MTXIlwXpose(): NULL MtxPtr 'src' "
#define MTX_ILWXPOSE_2          "MTXIlwXpose(): NULL MtxPtr 'ilwX' "


//  MATRIX_VECTOR SECTION

#define	MTX_MULTVEC_1			"MTXMultVec():  NULL MtxPtr 'm' "
#define	MTX_MULTVEC_2			"MTXMultVec():  NULL VecPtr 'src' "
#define	MTX_MULTVEC_3			"MTXMultVec():  NULL VecPtr 'dst' "

#define	MTX_MULTVECARRAY_1		"MTXMultVecArray():  NULL MtxPtr 'm' "
#define	MTX_MULTVECARRAY_2		"MTXMultVecArray():  NULL VecPtr 'srcBase' "
#define	MTX_MULTVECARRAY_3		"MTXMultVecArray():  NULL VecPtr 'dstBase' "
#define MTX_MULTVECARRAY_4      "MTXMultVecArray():  count must be greater than 1."


#define	MTX_MULTVECSR_1	        "MTXMultVecSR():  NULL MtxPtr 'm' "
#define	MTX_MULTVECSR_2	        "MTXMultVecSR():  NULL VecPtr 'src' "
#define	MTX_MULTVECSR_3         "MTXMultVecSR():  NULL VecPtr 'dst' "

#define	MTX_MULTVECARRAYSR_1	"MTXMultVecArraySR():  NULL MtxPtr 'm' "
#define	MTX_MULTVECARRAYSR_2	"MTXMultVecArraySR():  NULL VecPtr 'srcBase' "
#define	MTX_MULTVECARRAYSR_3	"MTXMultVecArraySR():  NULL VecPtr 'dstBase' "
#define MTX_MULTVECARRAYSR_4    "MTXMultVecArraySR():  count must be greater than 1."


//  MODEL  SECTION

#define	MTX_ROTRAD_1			"MTXRotRad():  NULL MtxPtr 'm' "

#define	MTX_ROTTRIG_1			"MTXRotTrig():  NULL MtxPtr 'm' "
#define	MTX_ROTTRIG_2			"MTXRotTrig():  invalid 'axis' value "

#define	MTX_ROTAXIS_1			"MTXRotAxisRad():  NULL MtxPtr 'm' "
#define	MTX_ROTAXIS_2			"MTXRotAxisRad():  NULL VecPtr 'axis' "

#define	MTX_TRANS_1				"MTXTrans():  NULL MtxPtr 'm' "

#define MTX_TRANSAPPLY_1        "MTXTransApply(): NULL MtxPtr 'src' "
#define MTX_TRANSAPPLY_2        "MTXTransApply(): NULL MtxPtr 'dst' "

#define	MTX_SCALE_1				"MTXScale():  NULL MtxPtr 'm' "

#define MTX_SCALEAPPLY_1        "MTXScaleApply(): NULL MtxPtr 'src' "
#define MTX_SCALEAPPLY_2        "MTXScaleApply(): NULL MtxPtr 'dst' "

#define	MTX_QUAT_1				"MTXQuat():  NULL MtxPtr 'm' "
#define	MTX_QUAT_2				"MTXQuat():  NULL QuaternionPtr 'q' "
#define	MTX_QUAT_3				"MTXQuat():  zero-value quaternion "



//  VIEW   SECTION

#define	MTX_LOOKAT_1			"MTXLookAt():  NULL MtxPtr 'm' "
#define	MTX_LOOKAT_2			"MTXLookAt():  NULL VecPtr 'camPos' "
#define	MTX_LOOKAT_3			"MTXLookAt():  NULL VecPtr 'camUp' "
#define	MTX_LOOKAT_4			"MTXLookAt():  NULL Point3dPtr 'target' "



//  PROJECTION   SECTION

#define	MTX_FRUSTUM_1			"MTXFrustum():  NULL Mtx44Ptr 'm' "
#define	MTX_FRUSTUM_2			"MTXFrustum():  't' and 'b' clipping planes are equal "
#define	MTX_FRUSTUM_3			"MTXFrustum():  'l' and 'r' clipping planes are equal "
#define	MTX_FRUSTUM_4			"MTXFrustum():  'n' and 'f' clipping planes are equal "

#define	MTX_PERSPECTIVE_1		"MTXPerspective():  NULL Mtx44Ptr 'm' "
#define	MTX_PERSPECTIVE_2		"MTXPerspective():  'fovY' out of range "
#define	MTX_PERSPECTIVE_3		"MTXPerspective():  'aspect' is 0 "

#define	MTX_ORTHO_1				"MTXOrtho():  NULL Mtx44Ptr 'm' "
#define	MTX_ORTHO_2				"MTXOrtho():  't' and 'b' clipping planes are equal "
#define	MTX_ORTHO_3				"MTXOrtho():  'l' and 'r' clipping planes are equal "
#define	MTX_ORTHO_4				"MTXOrtho():  'n' and 'f' clipping planes are equal "



//  STACK   SECTION

#define	MTX_INITSTACK_1			"MTXInitStack():  NULL MtxStackPtr 'sPtr' "
#define	MTX_INITSTACK_2			"MTXInitStack():  'sPtr' contains a NULL ptr to stack memory "
#define	MTX_INITSTACK_3			"MTXInitStack():  'numMtx' is 0 "

#define	MTX_PUSH_1				"MTXPush():  NULL MtxStackPtr 'sPtr' "
#define	MTX_PUSH_2				"MTXPush():  'sPtr' contains a NULL ptr to stack memory "
#define	MTX_PUSH_3				"MTXPush():  NULL MtxPtr 'm' "
#define	MTX_PUSH_4				"MTXPush():  stack overflow "

#define	MTX_PUSHFWD_1			"MTXPushFwd():  NULL MtxStackPtr 'sPtr' "
#define	MTX_PUSHFWD_2			"MTXPushFwd():  'sPtr' contains a NULL ptr to stack memory "
#define	MTX_PUSHFWD_3			"MTXPushFwd():  NULL MtxPtr 'm' "
#define	MTX_PUSHFWD_4			"MTXPushFwd():  stack overflow"

#define	MTX_PUSHILW_1			"MTXPushIlw():  NULL MtxStackPtr 'sPtr' "
#define	MTX_PUSHILW_2			"MTXPushIlw():  'sPtr' contains a NULL ptr to stack memory "
#define	MTX_PUSHILW_3			"MTXPushIlw():  NULL MtxPtr 'm' "
#define	MTX_PUSHILW_4			"MTXPushIlw():  stack overflow"

#define	MTX_PUSHILWXPOSE_1		"MTXPushIlwXpose():  NULL MtxStackPtr 'sPtr' "
#define	MTX_PUSHILWXPOSE_2		"MTXPushIlwXpose():  'sPtr' contains a NULL ptr to stack memory "
#define	MTX_PUSHILWXPOSE_3		"MTXPushIlwXpose():  NULL MtxPtr 'm' "
#define	MTX_PUSHILWXPOSE_4		"MTXPushIlwXpose():  stack overflow "

#define	MTX_POP_1				"MTXPop():  NULL MtxStackPtr 'sPtr' "
#define MTX_POP_2				"MTXPop():  'sPtr' contains a NULL ptr to stack memory "

#define	MTX_GETSTACKPTR_1		"MTXGetStackPtr():  NULL MtxStackPtr 'sPtr' "
#define	MTX_GETSTACKPTR_2		"MTXGetStackPtr():  'sPtr' contains a NULL ptr to stack memory "



//  VECTOR   SECTION

#define	VEC_ADD_1				"VECAdd():  NULL VecPtr 'a' "
#define	VEC_ADD_2				"VECAdd():  NULL VecPtr 'b' "
#define	VEC_ADD_3				"VECAdd():  NULL VecPtr 'ab' "

#define	VEC_SUBTRACT_1			"VECSubtract():  NULL VecPtr 'a' "
#define	VEC_SUBTRACT_2			"VECSubtract():  NULL VecPtr 'b' "
#define	VEC_SUBTRACT_3			"VECSubtract():  NULL VecPtr 'a_b' "

#define	VEC_SCALE_1				"VECScale():  NULL VecPtr 'src' "
#define	VEC_SCALE_2				"VECScale():  NULL VecPtr 'dst' "

#define	VEC_NORMALIZE_1			"VECNormalize():  NULL VecPtr 'src' "
#define	VEC_NORMALIZE_2			"VECNormalize():  NULL VecPtr 'unit' "
#define	VEC_NORMALIZE_3			"VECNormalize():  zero magnitude vector "

#define	VEC_MAG_1				"VECMag():  NULL VecPtr 'v' "


#define	VEC_REFLECT_1			"VECReflect():  NULL VecPtr 'src' "
#define	VEC_REFLECT_2			"VECReflect():  NULL VecPtr 'normal' "
#define	VEC_REFLECT_3			"VECReflect():  NULL VecPtr 'dst' "

#define	VEC_DOTPRODUCT_1		"VECDotProduct():  NULL VecPtr 'a' "
#define	VEC_DOTPRODUCT_2		"VECDotProduct():  NULL VecPtr 'b' "

#define	VEC_CROSSPRODUCT_1		"VECCrossProduct():  NULL VecPtr 'a' "
#define	VEC_CROSSPRODUCT_2		"VECCrossProduct():  NULL VecPtr 'b' "
#define	VEC_CROSSPRODUCT_3		"VECCrossProduct():  NULL VecPtr 'axb' "

#define	VEC_HALFANGLE_1			"VECHalfAngle():  NULL VecPtr 'a' "
#define	VEC_HALFANGLE_2			"VECHalfAngle():  NULL VecPtr 'b' "
#define	VEC_HALFANGLE_3			"VECHalfAngle():  NULL VecPtr 'half' "


//  QUATERNION SECTION

#define	QUAT_ADD_1              "QUATAdd():  NULL QuaternionPtr 'p' "
#define	QUAT_ADD_2              "QUATAdd():  NULL QuaternionPtr 'q' "
#define	QUAT_ADD_3              "QUATAdd():  NULL QuaternionPtr 'r' "

#define	QUAT_SUBTRACT_1         "QUATSubtract():  NULL QuaternionPtr 'p' "
#define	QUAT_SUBTRACT_2         "QUATSubtract():  NULL QuaternionPtr 'q' "
#define	QUAT_SUBTRACT_3         "QUATSubtract():  NULL QuaternionPtr 'r' "

#define	QUAT_MULTIPLY_1         "QUATMultiply():  NULL QuaternionPtr 'p' "
#define	QUAT_MULTIPLY_2         "QUATMultiply():  NULL QuaternionPtr 'q' "
#define	QUAT_MULTIPLY_3         "QUATMultiply():  NULL QuaternionPtr 'pq' "

#define	QUAT_DIVIDE_1           "QUATDivide():  NULL QuaternionPtr 'p' "
#define	QUAT_DIVIDE_2           "QUATDivide():  NULL QuaternionPtr 'q' "
#define	QUAT_DIVIDE_3           "QUATDivide():  NULL QuaternionPtr 'r' "

#define	QUAT_SCALE_1            "QUATScale():  NULL QuaternionPtr 'q' "
#define	QUAT_SCALE_2            "QUATScale():  NULL QuaternionPtr 'r' "

#define	QUAT_DOTPRODUCT_1       "QUATDotProduct():  NULL QuaternionPtr 'p' "
#define	QUAT_DOTPRODUCT_2       "QUATDotProduct():  NULL QuaternionPtr 'q' "

#define	QUAT_NORMALIZE_1        "QUATNormalize():  NULL QuaternionPtr 'src' "
#define	QUAT_NORMALIZE_2        "QUATNormalize():  NULL QuaternionPtr 'unit' "

#define	QUAT_ILWERSE_1          "QUATIlwerse():  NULL QuaternionPtr 'src' "
#define	QUAT_ILWERSE_2          "QUATIlwerse():  NULL QuaternionPtr 'ilw' "

#define	QUAT_EXP_1              "QUATExp():  NULL QuaternionPtr 'q' "
#define	QUAT_EXP_2              "QUATExp():  NULL QuaternionPtr 'r' "
#define	QUAT_EXP_3              "QUATExp():  'q' is not a pure quaternion. "

#define	QUAT_LOGN_1             "QUATLogN():  NULL QuaternionPtr 'q' "
#define	QUAT_LOGN_2             "QUATLogN():  NULL QuaternionPtr 'r' "
#define	QUAT_LOGN_3             "QUATLogN():  'q' is not a unit quaternion. "

#define	QUAT_MAKECLOSEST_1      "QUATMakeClosest():  NULL QuaternionPtr 'q' "
#define	QUAT_MAKECLOSEST_2      "QUATMakeClosest():  NULL QuaternionPtr 'qto' "
#define	QUAT_MAKECLOSEST_3      "QUATMakeClosest():  NULL QuaternionPtr 'r' "

#define	QUAT_ROTAXISRAD_1       "QUATRotAxisRad():  NULL QuaternionPtr 'r' "
#define	QUAT_ROTAXISRAD_2       "QUATRotAxisRad():  NULL VecPtr 'axis' "

#define	QUAT_MTX_1              "QUATMtx():  NULL QuaternionPtr 'r' "
#define	QUAT_MTX_2              "QUATMtx():  NULL MtxPtr 'm' "

#define	QUAT_LERP_1             "QUATLerp():  NULL QuaternionPtr 'p' "
#define	QUAT_LERP_2             "QUATLerp():  NULL QuaternionPtr 'q' "
#define	QUAT_LERP_3             "QUATLerp():  NULL QuaternionPtr 'r' "

#define	QUAT_SLERP_1            "QUATSlerp():  NULL QuaternionPtr 'p' "
#define	QUAT_SLERP_2            "QUATSlerp():  NULL QuaternionPtr 'q' "
#define	QUAT_SLERP_3            "QUATSlerp():  NULL QuaternionPtr 'r' "

#define	QUAT_SQUAD_1            "QUATSquad():  NULL QuaternionPtr 'p' "
#define	QUAT_SQUAD_2            "QUATSquad():  NULL QuaternionPtr 'a' "
#define	QUAT_SQUAD_3            "QUATSquad():  NULL QuaternionPtr 'b' "
#define	QUAT_SQUAD_4            "QUATSquad():  NULL QuaternionPtr 'q' "
#define	QUAT_SQUAD_5            "QUATSquad():  NULL QuaternionPtr 'r' "

#define	QUAT_COMPA_1            "QUATCompA():  NULL QuaternionPtr 'qprev' "
#define	QUAT_COMPA_2            "QUATCompA():  NULL QuaternionPtr 'q' "
#define	QUAT_COMPA_3            "QUATCompA():  NULL QuaternionPtr 'qnext' "
#define	QUAT_COMPA_4            "QUATCompA():  NULL QuaternionPtr 'a' "


//	Texture Projection Section

#define	MTX_LIGHT_FRUSTUM_1		"MTXLightFrustum():  NULL MtxPtr 'm' "
#define	MTX_LIGHT_FRUSTUM_2		"MTXLightFrustum():  't' and 'b' clipping planes are equal "
#define	MTX_LIGHT_FRUSTUM_3		"MTXLightFrustum():  'l' and 'r' clipping planes are equal "
#define	MTX_LIGHT_FRUSTUM_4		"MTXLightFrustum():  'n' and 'f' clipping planes are equal "

#define	MTX_LIGHT_PERSPECTIVE_1	"MTXLightPerspective():  NULL MtxPtr 'm' "
#define	MTX_LIGHT_PERSPECTIVE_2	"MTXLightPerspective():  'fovY' out of range "
#define	MTX_LIGHT_PERSPECTIVE_3	"MTXLightPerspective():  'aspect' is 0 "

#define	MTX_LIGHT_ORTHO_1		"MTXLightOrtho():  NULL MtxPtr 'm' "
#define	MTX_LIGHT_ORTHO_2		"MTXLightOrtho():  't' and 'b' clipping planes are equal "
#define	MTX_LIGHT_ORTHO_3		"MTXLightOrtho():  'l' and 'r' clipping planes are equal "

//---------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // __MTXASSERT_H__

