/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

#include <math.h>
#include <stdio.h>
#include <mtx.h>
#include <mtx/mtx44.h>
#include "mtxAssert.h"
#include "mtx44Assert.h"

/*---------------------------------------------------------------------*
    Constants
 *---------------------------------------------------------------------*/
static const f32x2 c00 = {0.0F, 0.0F};
static const f32x2 c01 = {0.0F, 1.0F};
static const f32x2 c10 = {1.0F, 0.0F};
static const f32x2 c11 = {1.0F, 1.0F};
//static const f32x2 c22 = {2.0F, 2.0F};
static const f32x2 c33 = {3.0F, 3.0F};
static const f32x2 c0505 = {0.5F, 0.5F};

/*---------------------------------------------------------------------*


                             PROJECTION SECTION


 *---------------------------------------------------------------------*/

/*---------------------------------------------------------------------*

Name:           MTXFrustum

Description:    compute a 4x4 perspective projection matrix from a
                specified view volume.


Arguments:      m        4x4 matrix to be set

                t        top coord. of view volume at the near clipping plane

                b        bottom coord of view volume at the near clipping plane

                lf       left coord. of view volume at near clipping plane

                r        right coord. of view volume at near clipping plane

                n        positive distance from camera to near clipping plane

                f        positive distance from camera to far clipping plane


Return:         none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXFrustum ( Mtx44 m, f32 t, f32 b, f32 lf, f32 r, f32 n, f32 f )
{
    f32 tmp;

    ASSERTMSG( (m != 0),  MTX_FRUSTUM_1     );
    ASSERTMSG( (t != b),  MTX_FRUSTUM_2     );
    ASSERTMSG( (lf != r), MTX_FRUSTUM_3     );
    ASSERTMSG( (n != f),  MTX_FRUSTUM_4     );

    tmp     =  1.0f / (r - lf);
    m[0][0] =  (2*n) * tmp;
    m[0][1] =  0.0f;
    m[0][2] =  (r + lf) * tmp;
    m[0][3] =  0.0f;

    tmp     =  1.0f / (t - b);
    m[1][0] =  0.0f;
    m[1][1] =  (2*n) * tmp;
    m[1][2] =  (t + b) * tmp;
    m[1][3] =  0.0f;

    m[2][0] =  0.0f;
    m[2][1] =  0.0f;

    tmp     =  1.0f / (f - n);

    // scale z to (-w, w) range 
    m[2][2] = -(f + n) * tmp;
    m[2][3] = -(2*f*n) * tmp;

    m[3][0] =  0.0f;
    m[3][1] =  0.0f;
    m[3][2] = -1.0f;
    m[3][3] =  0.0f;
}

/*---------------------------------------------------------------------*

Name:           MTXPerspective

Description:    compute a 4x4 perspective projection matrix from
                field of view and aspect ratio.


Arguments:      m       4x4 matrix to be set

                fovy    total field of view in in degrees in the YZ plane

                aspect  ratio of view window width:height (X / Y)

                n       positive distance from camera to near clipping plane

                f       positive distance from camera to far clipping plane


Return:         none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXPerspective ( Mtx44 m, f32 fovY, f32 aspect, f32 n, f32 f )
{
    f32 angle;
    f32 cot;
    f32 tmp;

    ASSERTMSG( (m != 0),                             MTX_PERSPECTIVE_1    );
    ASSERTMSG( ( (fovY > 0.0) && ( fovY < 180.0) ),  MTX_PERSPECTIVE_2    );
    ASSERTMSG( (aspect != 0),                        MTX_PERSPECTIVE_3    );

    // find the cotangent of half the (YZ) field of view
    angle = fovY * 0.5f;
    angle = MTXDegToRad( angle );

    cot = 1.0f / tanf(angle);

    m[0][0] =  cot / aspect;
    m[0][1] =  0.0f;
    m[0][2] =  0.0f;
    m[0][3] =  0.0f;

    m[1][0] =  0.0f;
    m[1][1] =   cot;
    m[1][2] =  0.0f;
    m[1][3] =  0.0f;

    m[2][0] =  0.0f;
    m[2][1] =  0.0f;

    tmp     = 1.0f / (f - n);

    // scale z to (-w, +w) range
    m[2][2] = -(f + n) * tmp;
    m[2][3] = -(2*f*n) * tmp;

    m[3][0] =  0.0f;
    m[3][1] =  0.0f;
    m[3][2] = -1.0f;
    m[3][3] =  0.0f;
}

/*---------------------------------------------------------------------*

Name:           MTXOrtho

Description:    compute a 4x4 orthographic projection matrix.


Arguments:      m        4x4 matrix to be set

                t        top coord. of parallel view volume

                b        bottom coord of parallel view volume

                lf       left coord. of parallel view volume

                r        right coord. of parallel view volume

                n        positive distance from camera to near clipping plane

                f        positive distance from camera to far clipping plane


Return:         none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXOrtho ( Mtx44 m, f32 t, f32 b, f32 lf, f32 r, f32 n, f32 f )
{
    f32 tmp;

    ASSERTMSG( (m != 0),  MTX_ORTHO_1  );
    ASSERTMSG( (t != b),  MTX_ORTHO_2  );
    ASSERTMSG( (lf != r), MTX_ORTHO_3  );
    ASSERTMSG( (n != f),  MTX_ORTHO_4  );

    tmp     =  1.0f / (r - lf);
    m[0][0] =  2.0f * tmp;
    m[0][1] =  0.0f;
    m[0][2] =  0.0f;
    m[0][3] = -(r + lf) * tmp;

    tmp     =  1.0f / (t - b);
    m[1][0] =  0.0f;
    m[1][1] =  2.0f * tmp;
    m[1][2] =  0.0f;
    m[1][3] = -(t + b) * tmp;

    m[2][0] =  0.0f;
    m[2][1] =  0.0f;

    tmp     =  1.0f / (f - n);

    // scale z to (-1, 1) range
    m[2][2] = -2.0f * tmp;
    m[2][3] = -(f + n) * tmp;

    m[3][0] =  0.0f;
    m[3][1] =  0.0f;
    m[3][2] =  0.0f;
    m[3][3] =  1.0f;
}

/*---------------------------------------------------------------------*


                             GENERAL SECTION


 *---------------------------------------------------------------------*/

/* NOTE: Prototypes for these functions are defined in "mtx44ext.h".   */

/*---------------------------------------------------------------------*
Name:           MTX44Identity

Description:    sets a matrix to identity

Arguments:      m :  matrix to be set

Return:         none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44Identity( Mtx44 m )
{
    ASSERTMSG( (m != 0), MTX44_IDENTITY_1 );

    m[0][0] = 1.0f; m[0][1] = 0.0f; m[0][2] = 0.0f; m[0][3] = 0.0f;
    m[1][0] = 0.0f; m[1][1] = 1.0f; m[1][2] = 0.0f; m[1][3] = 0.0f;
    m[2][0] = 0.0f; m[2][1] = 0.0f; m[2][2] = 1.0f; m[2][3] = 0.0f;
    m[3][0] = 0.0f; m[3][1] = 0.0f; m[3][2] = 0.0f; m[3][3] = 1.0f;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/

void PSMTX44Identity( register Mtx44 m )
{
    __PSQ_ST(m, c10, 0, 0);
    __PSQ_STX(m,  8, c00, 0, 0);
    __PSQ_STX(m, 16, c01, 0, 0);
    __PSQ_STX(m, 24, c00, 0, 0);
    __PSQ_STX(m, 32, c00, 0, 0);
    __PSQ_STX(m, 40, c10, 0, 0);
    __PSQ_STX(m, 48, c00, 0, 0);
    __PSQ_STX(m, 56, c01, 0, 0);
}
#endif

/*---------------------------------------------------------------------*
Name:           MTX44Copy

Description:    copies the contents of one matrix into another

Arguments:      src        source matrix for copy
                dst        destination matrix for copy


Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44Copy( MTX_CONST Mtx44 src, Mtx44 dst )
{
    ASSERTMSG( (src != 0) , MTX44_COPY_1 );
    ASSERTMSG( (dst != 0) , MTX44_COPY_2 );

    if( src == dst )
    {
        return;
    }

    dst[0][0] = src[0][0]; dst[0][1] = src[0][1]; dst[0][2] = src[0][2]; dst[0][3] = src[0][3];
    dst[1][0] = src[1][0]; dst[1][1] = src[1][1]; dst[1][2] = src[1][2]; dst[1][3] = src[1][3];
    dst[2][0] = src[2][0]; dst[2][1] = src[2][1]; dst[2][2] = src[2][2]; dst[2][3] = src[2][3];
    dst[3][0] = src[3][0]; dst[3][1] = src[3][1]; dst[3][2] = src[3][2]; dst[3][3] = src[3][3];
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX44Copy( MTX_CONST Mtx44 src, Mtx44 dst )
{
    f32x2 fp1;

    //psq_l       fp1,  0(src), 0, 0;
    fp1 = __PSQ_L(src, 0, 0);

    //psq_st      fp1,  0(dst), 0, 0;
    __PSQ_ST(dst, fp1, 0, 0);

    //psq_l       fp1,  8(src), 0, 0;
    fp1 = __PSQ_LX(src, 8, 0, 0);

    //psq_st      fp1,  8(dst), 0, 0;
    __PSQ_STX(dst, 8, fp1, 0, 0);

    //psq_l       fp1, 16(src), 0, 0;
    fp1 = __PSQ_LX(src, 16, 0, 0);

    //psq_st      fp1, 16(dst), 0, 0;
    __PSQ_STX(dst, 16, fp1, 0, 0);

    //psq_l       fp1, 24(src), 0, 0;
    fp1 = __PSQ_LX(src, 24, 0, 0);

    //psq_st      fp1, 24(dst), 0, 0;
    __PSQ_STX(dst, 24, fp1, 0, 0);

    //psq_l       fp1, 32(src), 0, 0;
    fp1 = __PSQ_LX(src, 32, 0, 0);

    //psq_st      fp1, 32(dst), 0, 0;
    __PSQ_STX(dst, 32, fp1, 0, 0);

    //psq_l       fp1, 40(src), 0, 0;
    fp1 = __PSQ_LX(src, 40, 0, 0);

    //psq_st      fp1, 40(dst), 0, 0;
    __PSQ_STX(dst, 40, fp1, 0, 0);

    //psq_l       fp1, 48(src), 0, 0;
    fp1 = __PSQ_LX(src, 48, 0, 0);

    //psq_st      fp1, 48(dst), 0, 0;
    __PSQ_STX(dst, 48, fp1, 0, 0);

    //psq_l       fp1, 56(src), 0, 0;
    fp1 = __PSQ_LX(src, 56, 0, 0);

    //psq_st      fp1, 56(dst), 0, 0;
    __PSQ_STX(dst, 56, fp1, 0, 0);
}
#endif

/*---------------------------------------------------------------------*
Name:           MTX44Concat

Description:    concatenates two matrices.
                order of operation is A x B = AB.
                ok for any of ab == a == b.

                saves a MTXCopy operation if ab != to a or b.

Arguments:      a        first matrix for concat.
                b        second matrix for concat.
                ab       resultant matrix from concat.

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44Concat( MTX_CONST Mtx44 a, MTX_CONST Mtx44 b, Mtx44 ab )
{
    Mtx44       mTmp;
    Mtx44Ptr    m;

    ASSERTMSG( (a  != 0), MTX44_CONCAT_1 );
    ASSERTMSG( (b  != 0), MTX44_CONCAT_2 );
    ASSERTMSG( (ab != 0), MTX44_CONCAT_3 );

    if( (ab == a) || (ab == b) )
    {
        m = mTmp;
    }
    else
    {
        m = ab;
    }

    // compute (a x b) -> m

    m[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0] + a[0][3]*b[3][0];
    m[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1] + a[0][3]*b[3][1];
    m[0][2] = a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2] + a[0][3]*b[3][2];
    m[0][3] = a[0][0]*b[0][3] + a[0][1]*b[1][3] + a[0][2]*b[2][3] + a[0][3]*b[3][3];

    m[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0] + a[1][3]*b[3][0];
    m[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1] + a[1][3]*b[3][1];
    m[1][2] = a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2] + a[1][3]*b[3][2];
    m[1][3] = a[1][0]*b[0][3] + a[1][1]*b[1][3] + a[1][2]*b[2][3] + a[1][3]*b[3][3];

    m[2][0] = a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0] + a[2][3]*b[3][0];
    m[2][1] = a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1] + a[2][3]*b[3][1];
    m[2][2] = a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2] + a[2][3]*b[3][2];
    m[2][3] = a[2][0]*b[0][3] + a[2][1]*b[1][3] + a[2][2]*b[2][3] + a[2][3]*b[3][3];

    m[3][0] = a[3][0]*b[0][0] + a[3][1]*b[1][0] + a[3][2]*b[2][0] + a[3][3]*b[3][0];
    m[3][1] = a[3][0]*b[0][1] + a[3][1]*b[1][1] + a[3][2]*b[2][1] + a[3][3]*b[3][1];
    m[3][2] = a[3][0]*b[0][2] + a[3][1]*b[1][2] + a[3][2]*b[2][2] + a[3][3]*b[3][2];
    m[3][3] = a[3][0]*b[0][3] + a[3][1]*b[1][3] + a[3][2]*b[2][3] + a[3][3]*b[3][3];

    // overwrite a or b if needed
    if(m == mTmp)
    {
        C_MTX44Copy( *((MTX_CONST Mtx44 *)&mTmp), ab );
    }
}


#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/

void PSMTX44Concat( MTX_CONST Mtx44 a, MTX_CONST Mtx44 b, Mtx44 ab )
{
    f32x2 fp0, fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12, fp13;

    //psq_l       fp0 ,  0(a), 0, 0;          // a00,a01
    //fp0[0] = a[0][0];
    //fp0[1] = a[0][1];
    fp0 = __PSQ_L(a, 0, 0);

    //psq_l       fp2 ,  0(b), 0, 0;          // b00,b01
    //fp2[0] = b[0][0];
    //fp2[1] = b[0][1];
    fp2 = __PSQ_L(b, 0, 0);

    //ps_muls0    fp6 ,   fp2,  fp0;          // b00a00,b01a00
    fp6 = __PS_MULS0(fp2, fp0);

    //psq_l       fp3 , 16(b), 0, 0;          // b10,b11
    //fp3[0] = b[1][0];
    //fp3[1] = b[1][1];
    fp3 = __PSQ_LX(b, 16, 0, 0);

    //psq_l       fp4 , 32(b), 0, 0;          // b20,b21
    //fp4[0] = b[2][0];
    //fp4[1] = b[2][1];
    fp4 = __PSQ_LX(b, 32, 0, 0);

    //ps_madds1   fp6 ,   fp3,  fp0,  fp6;    // b00a00+b10a01,b01a00+b11a01
    fp6 = __PS_MADDS1(fp3, fp0, fp6);

    //psq_l       fp1 ,  8(a), 0, 0;          // a02,a03
    //fp1[0] = a[0][2];
    //fp1[1] = a[0][3];
    fp1 = __PSQ_LX(a,  8, 0, 0);

    //psq_l       fp5 , 48(b), 0, 0;          // b30,b31
    //fp5[0] = b[3][0];
    //fp5[1] = b[3][1];
    fp5 = __PSQ_LX(b, 48, 0, 0);

    // b00a00+b10a01+b20a02,b01a00+b11a01+b21a02
    //ps_madds0   fp6 ,   fp4,  fp1,  fp6;
    fp6 = __PS_MADDS0(fp4, fp1, fp6);

    //psq_l       fp0 , 16(a), 0, 0;          // a10,a11
    //fp0[0] = a[1][0];
    //fp0[1] = a[1][1];
    fp0 = __PSQ_LX(a,  16, 0, 0);

    // b00a00+b10a01+b20a02+b30a03,b01a00+b11a01+b21a02+b31a03
    //ps_madds1   fp6 ,   fp5,  fp1,  fp6;
    fp6 = __PS_MADDS1(fp5, fp1, fp6);

    //psq_l       fp1 , 24(a), 0, 0;          // a12,a13
    //fp1[0] = a[1][2];
    //fp1[1] = a[1][3];
    fp1 = __PSQ_LX(a,  24, 0, 0);

    //ps_muls0    fp8 ,   fp2,  fp0;          // b00a10,b01a10
    fp8 = __PS_MULS0(fp2, fp0);

    //ps_madds1   fp8 ,   fp3,  fp0,  fp8;    // b00a10+b10a11,b01a11+b11a11
    fp8 = __PS_MADDS1(fp3, fp0, fp8);

    //psq_l       fp0 , 32(a), 0, 0;          // a20,a21
    //fp0[0] = a[2][0];
    //fp0[1] = a[2][1];
    fp0 = __PSQ_LX(a,  32, 0, 0);

    // b00a10+b10a11+b20a12,b01a11+b11a11+b21a12
    //ps_madds0   fp8 ,   fp4,  fp1,  fp8;
    fp8 = __PS_MADDS0(fp4, fp1, fp8);

    // b00a10+b10a11+b20a12+b30a13,b01a10+b11a11+b21a12+b31a13
    //ps_madds1   fp8 ,   fp5,  fp1,  fp8;
    fp8 = __PS_MADDS1(fp5, fp1, fp8);

    //psq_l       fp1 , 40(a), 0, 0;          // a22,a23
    //fp1[0] = a[2][2];
    //fp1[1] = a[2][3];
    fp1 = __PSQ_LX(a, 40, 0, 0);

    //ps_muls0    fp10,   fp2,  fp0;          // b00a20,b01a20
    fp10 = __PS_MULS0(fp2, fp0);

    //ps_madds1   fp10,   fp3,  fp0, fp10;    // b00a20+b10a21,b01a20+b11a21
    fp10 = __PS_MADDS1(fp3, fp0, fp10);

    //psq_l       fp0 , 48(a), 0, 0;          // a30,a31
    //fp0[0] = a[3][0];
    //fp0[1] = a[3][1];
    fp0 = __PSQ_LX(a, 48, 0, 0);

    // b00a20+b10a21+b20a22,b01a20+b11a21+b21a22
    //ps_madds0   fp10,   fp4,  fp1, fp10;
    fp10 = __PS_MADDS0(fp4, fp1, fp10);

    // b00a20+b10a21+b20a22+b30a23,b01a20+b11a21+b21a22+b31a23
    //ps_madds1   fp10,   fp5,  fp1, fp10;
    fp10 = __PS_MADDS1(fp5, fp1, fp10);

    //psq_l       fp1 , 56(a), 0, 0;          // a32,a33
    //fp1[0] = a[3][2];
    //fp1[1] = a[3][3];
    fp1 = __PSQ_LX(a,  56, 0, 0);

    //ps_muls0    fp12,   fp2,  fp0;          // b00a30,b01a30
    fp12 = __PS_MULS0(fp2, fp0);

    //psq_l       fp2 ,  8(b), 0, 0;          // b02,b03
    //fp2[0] = b[0][2];
    //fp2[1] = b[0][3];
    fp2 = __PSQ_LX(b,  8, 0, 0);

    //ps_madds1   fp12,   fp3,  fp0, fp12;    // b00a30+b10a31,b01a30+b11a31
    fp12 = __PS_MADDS1(fp3, fp0, fp12);

    //psq_l       fp0 ,  0(a), 0, 0;          // a00,a01
    //fp0[0] = a[0][0];
    //fp0[1] = a[0][1];
    fp0 = __PSQ_LX(a,  0, 0, 0);

    // b00a30+b10a31+b20a32,b01a30+b11a31+b21a32
    //ps_madds0   fp12,   fp4,  fp1, fp12;
    fp12 = __PS_MADDS0(fp4, fp1, fp12);

    //psq_l       fp3 , 24(b), 0, 0;          // b12,b13
    //fp3[0] = b[1][2];
    //fp3[1] = b[1][3];
    fp3 = __PSQ_LX(b,  24, 0, 0);

    // b00a30+b10a31+b20a32+b30a33,b01a30+b11a31+b21a32+b31a33
    //ps_madds1   fp12,   fp5,  fp1, fp12;
    fp12 = __PS_MADDS1(fp5, fp1, fp12);

    //psq_l       fp1 ,  8(a), 0, 0;          // a02,a03
    //fp1[0] = a[0][2];
    //fp1[1] = a[0][3];
    fp1 = __PSQ_LX(a,  8, 0, 0);

    //ps_muls0    fp7 ,   fp2,  fp0;          // b02a00,b03a00
    fp7 = __PS_MULS0(fp2, fp0);

    //psq_l       fp4 , 40(b), 0, 0;          // b22,b23
    //fp4[0] = b[2][2];
    //fp4[1] = b[2][3];
    fp4 = __PSQ_LX(b,  40, 0, 0);

    //ps_madds1   fp7 ,   fp3,  fp0, fp7;     // b02a00+b12a01,b03a00+b13a01
    fp7 = __PS_MADDS1(fp3, fp0, fp7);

    //psq_l       fp5 , 56(b), 0, 0;          // b32,b33
    //fp5[0] = b[3][2];
    //fp5[1] = b[3][3];
    fp5 = __PSQ_LX(b, 56, 0, 0);

    // b02a00+b12a01+b22a02,b03a00+b13a01+b23a02
    //ps_madds0   fp7 ,   fp4,  fp1, fp7;
    fp7 = __PS_MADDS0(fp4, fp1, fp7);

    //psq_l       fp0 , 16(a), 0, 0;          // a10,a11
    //fp0[0] = a[1][0];
    //fp0[1] = a[1][1];
    fp0 = __PSQ_LX(a, 16, 0, 0);

    // b02a00+b12a01+b22a02+b32a03,b03a00+b13a01+b23a02+b33a03
    //ps_madds1   fp7 ,   fp5,  fp1, fp7;
    fp7 = __PS_MADDS1(fp5, fp1, fp7);

    //psq_l       fp1 , 24(a), 0, 0;          // a12,a13
    //fp1[0] = a[1][2];
    //fp1[1] = a[1][3];
    fp1 = __PSQ_LX(a,  24, 0, 0);

    //ps_muls0    fp9 ,   fp2,  fp0;          // b02a10,b03a10
    fp9 = __PS_MULS0(fp2, fp0);

    //psq_st      fp6 , 0(ab), 0, 0;          // ab00,ab01
    //ab[0][0] = fp6[0];
    //ab[0][1] = fp6[1];
    __PSQ_STX(ab, 0, fp6, 0, 0);

    //ps_madds1   fp9 ,   fp3,  fp0, fp9;     // b02a10+b12a11,b03a10+b13a11
    fp9 = __PS_MADDS1(fp3, fp0, fp9);

    //psq_l       fp0 , 32(a), 0, 0;          // a20,a21
    //fp0[0] = a[2][0];
    //fp0[1] = a[2][1];
    fp0 = __PSQ_LX(a, 32, 0, 0);

    // b02a10+b12a11+b22a12,b03a10+b13a11+b23a12
    //ps_madds0   fp9,    fp4,  fp1, fp9;
    fp9 = __PS_MADDS0(fp4, fp1, fp9);

    //psq_st      fp8 ,16(ab), 0, 0;          // ab10,ab11
    //ab[1][0] = fp8[0];
    //ab[1][1] = fp8[1];
    __PSQ_STX(ab, 16, fp8, 0, 0);

    // b02a10+b12a11+b22a12+b32a13,b03a10+b13a11+b23a12+b33a13
    //ps_madds1   fp9 ,   fp5,  fp1, fp9;
    fp9 = __PS_MADDS1(fp5, fp1, fp9);

    //psq_l       fp1 , 40(a), 0, 0;          // a22,a23
    //fp1[0] = a[2][2];
    //fp1[1] = a[2][3];
    fp1 = __PSQ_LX(a, 40, 0, 0);

    //ps_muls0    fp11,   fp2,  fp0;          // b02a20,b03a20
    fp11 = __PS_MULS0(fp2, fp0);

    //psq_st      fp10,32(ab), 0, 0;          // ab20,ab21
    //ab[2][0] = fp10[0];
    //ab[2][1] = fp10[1];
    __PSQ_STX(ab, 32, fp10, 0, 0);

    //ps_madds1   fp11,   fp3,  fp0, fp11;    // b02a20+b12a21,b03a20+b13a21
    fp11 = __PS_MADDS1(fp3, fp0, fp11);

    //psq_l       fp0 , 48(a), 0, 0;          // a30,a31
    //fp0[0] = a[3][0];
    //fp0[1] = a[3][1];
    fp0 = __PSQ_LX(a, 48, 0, 0);

    // b02a20+b12a21+b22a22,b03a20+b13a21+b23a22
    //ps_madds0   fp11,   fp4,  fp1, fp11;
    fp11 = __PS_MADDS0(fp4, fp1, fp11);

    //psq_st      fp12,48(ab), 0, 0;          // ab30,ab31
    //ab[3][0] = fp12[0];
    //ab[3][1] = fp12[1];
    __PSQ_STX(ab, 48, fp12, 0, 0);

    // b02a20+b12a21+b22a22+b32a23,b03a20+b13a21+b23a22+b33a23
    //ps_madds1   fp11,   fp5,  fp1, fp11;
    fp11 = __PS_MADDS1(fp5, fp1, fp11);

    //psq_l       fp1,  56(a), 0, 0;          // a32,a33
    //fp1[0] = a[3][2];
    //fp1[1] = a[3][3];
    fp1 = __PSQ_LX(a, 56, 0, 0);

    //ps_muls0    fp13,   fp2,  fp0;          // b02a30,b03a30
    fp13 = __PS_MULS0(fp2, fp0);

    //psq_st      fp7 , 8(ab), 0, 0;          // ab02,ab03
    //ab[0][2] = fp7[0];
    //ab[0][3] = fp7[1];
    __PSQ_STX(ab, 8, fp7, 0, 0);

    //ps_madds1   fp13,   fp3,  fp0, fp13;    // b02a30+b12a31,b03a30+b13a31
    fp13 = __PS_MADDS1(fp3, fp0, fp13);

    //psq_st      fp9 ,24(ab), 0, 0;          // ab12,ab13
    //ab[1][2] = fp9[0];
    //ab[1][3] = fp9[1];
    __PSQ_STX(ab, 24, fp9, 0, 0);

    // b02a30+b12a31+b22a32,b03a30+b13a31+b23a32
    //ps_madds0   fp13,   fp4,  fp1, fp13;
    fp13 = __PS_MADDS0(fp4, fp1, fp13);

    //psq_st      fp11,40(ab), 0, 0;          // ab22,ab23
    //ab[2][2] = fp11[0];
    //ab[2][3] = fp11[1];
    __PSQ_STX(ab, 40, fp11, 0, 0);

    // b02a30+b12a31+b22a32+b32a33,b03a30+b13a31+b23a32+b33a33
    //ps_madds1   fp13,   fp5,  fp1, fp13;
    fp13 = __PS_MADDS1(fp5, fp1, fp13);

    //psq_st      fp13,56(ab), 0, 0;          // ab32,ab33
    //ab[3][2] = fp13[0];
    //ab[3][3] = fp13[1];
    __PSQ_STX(ab, 56, fp13, 0, 0);
}
#endif


/*---------------------------------------------------------------------*
Name:           MTX44Transpose

Description:    computes the transpose of a matrix.

Arguments:      src       source matrix.
                xPose     destination (transposed) matrix.
                          ok if src == xPose.

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44Transpose ( MTX_CONST Mtx44 src, Mtx44 xPose )
{
    Mtx44       mTmp;
    Mtx44Ptr    m;

    ASSERTMSG( (src   != 0), MTX44_TRANSPOSE_1  );
    ASSERTMSG( (xPose != 0), MTX44_TRANSPOSE_2  );

    if(src == xPose)
    {
        m = mTmp;
    }
    else
    {
        m = xPose;
    }

    m[0][0] = src[0][0];    m[0][1] = src[1][0];    m[0][2] = src[2][0];    m[0][3] = src[3][0];
    m[1][0] = src[0][1];    m[1][1] = src[1][1];    m[1][2] = src[2][1];    m[1][3] = src[3][1];
    m[2][0] = src[0][2];    m[2][1] = src[1][2];    m[2][2] = src[2][2];    m[2][3] = src[3][2];
    m[3][0] = src[0][3];    m[3][1] = src[1][3];    m[3][2] = src[2][3];    m[3][3] = src[3][3];

    // copy back if needed
    if( m == mTmp )
    {
        C_MTX44Copy( *((MTX_CONST Mtx44 *)&mTmp), xPose );
    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX44Transpose ( MTX_CONST Mtx44 src, Mtx44 xPose )
{
    f32x2 fp0, fp1, fp2, fp3, fp4, fp5;

    //psq_l       fp0,  0(src), 0, 0;     // fp0 <= s00,s01
    fp0 = __PSQ_L(src, 0, 0);

    //psq_l       fp1, 16(src), 0, 0;     // fp1 <= s10,s11
    fp1 = __PSQ_LX(src, 16, 0, 0);

    //ps_merge00  fp4, fp0, fp1;              // fp4 <= t00,t10
    fp4 = __PS_MERGE00(fp0, fp1);

    //psq_l       fp2,  8(src), 0, 0;     // fp2 <= s02,s03
    fp2 = __PSQ_LX(src, 8, 0, 0);

    //psq_st      fp4,  0(xPose), 0, 0;
    __PSQ_ST(xPose, fp4, 0, 0);

    //ps_merge11  fp5, fp0, fp1;              // fp5 <= t01,t11
    fp5 = __PS_MERGE11(fp0, fp1);

    //psq_l       fp3, 24(src), 0, 0;     // fp3 <= s12,s13
    fp3 = __PSQ_LX(src, 24, 0, 0);

    //psq_st      fp5, 16(xPose), 0, 0;
    __PSQ_STX(xPose, 16, fp5, 0, 0);

    //ps_merge00  fp4, fp2, fp3;              // fp4 <= t02,t12
    fp4 = __PS_MERGE00(fp2, fp3);

    //psq_l       fp0, 32(src), 0, 0;     // fp0 <= s20,s21
    fp0 = __PSQ_LX(src, 32, 0, 0);

    //psq_st      fp4, 32(xPose), 0, 0;
    __PSQ_STX(xPose, 32, fp4, 0, 0);

    //ps_merge11  fp5, fp2, fp3;              // fp5 <= t03,t13
    fp5 = __PS_MERGE11(fp2, fp3);

    //psq_l       fp1, 48(src), 0, 0;     // fp1 <= s30,s31
    fp1 = __PSQ_LX(src, 48, 0, 0);

    //psq_st      fp5, 48(xPose), 0, 0;
    __PSQ_STX(xPose, 48, fp5, 0, 0);

    //ps_merge00  fp4, fp0, fp1;              // fp4 <= t20,t30
    fp4 = __PS_MERGE00(fp0, fp1);

    //psq_l       fp2, 40(src), 0, 0;     // fp2 <= s22,s23
    fp2 = __PSQ_LX(src, 40, 0, 0);

    //psq_st      fp4,  8(xPose), 0, 0;
    __PSQ_STX(xPose, 8, fp4, 0, 0);

    //ps_merge11  fp5, fp0, fp1;              // fp5 <= t21,t31
    fp5 = __PS_MERGE11(fp0, fp1);

    //psq_l       fp3, 56(src), 0, 0;     // fp2 <= s32,s33
    fp3 = __PSQ_LX(src, 56, 0, 0);

    //psq_st      fp5, 24(xPose), 0, 0;
    __PSQ_STX(xPose, 24, fp5, 0, 0);

    //ps_merge00  fp4, fp2, fp3;              // fp4 <= s22,s32
    fp4 = __PS_MERGE00(fp2, fp3);

    //psq_st      fp4, 40(xPose), 0, 0;
    __PSQ_STX(xPose, 40, fp4, 0, 0);

    //ps_merge11  fp5, fp2, fp3;              // fp5 <= s23,s33
    fp5 = __PS_MERGE11(fp2, fp3);

    //psq_st      fp5, 56(xPose), 0, 0;
    __PSQ_STX(xPose, 56, fp5, 0, 0);
}
#endif

/*---------------------------------------------------------------------*
Name:           MTX44Ilwerse

Description:    computes a fast ilwerse of a matrix.
                uses Gauss-Jordan(with partial pivoting)

Arguments:      src       source matrix.
                ilw       destination (ilwerse) matrix.
                          ok if src == ilw.

Return:         0 if src is not ilwertible.
                1 on success.
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version only
 *---------------------------------------------------------------------*/
#define NUM         4
#define SWAPF(a,b)  { f32 tmp; tmp = (a); (a) = (b); (b)=tmp; }

u32 C_MTX44Ilwerse( MTX_CONST Mtx44 src, Mtx44 ilw )
{
    Mtx44       gjm;
    s32         i, j, k;
    f32         w;

    ASSERTMSG( (src != 0), MTX44_ILWERSE_1 );
    ASSERTMSG( (ilw != 0), MTX44_ILWERSE_2 );

    C_MTX44Copy(src, gjm);
    C_MTX44Identity(ilw);

    for ( i = 0 ; i < NUM ; ++i )
    {
        f32 max = 0.0f;
        s32 swp = i;

        // ---- partial pivoting -----
        for( k = i ; k < NUM ; k++ )
        {
            f32 ftmp;
            ftmp = fabsf(gjm[k][i]);
            if ( ftmp > max )
            {
                max = ftmp;
                swp = k;
            }
        }

        // check singular matrix
        //(or can't solve ilwerse matrix with this algorithm)
        if ( max == 0.0f )
        {
            return 0;
        }

        // swap row
        if( swp != i )
        {
            for ( k = 0 ; k < NUM ; k++ )
            {
                SWAPF(gjm[i][k], gjm[swp][k]);
                SWAPF(ilw[i][k], ilw[swp][k]);
            }
        }

        // ---- pivoting end ----

        w = 1.0F / gjm[i][i];
        for ( j = 0 ; j < NUM ; ++j )
        {
            gjm[i][j] *= w;
            ilw[i][j] *= w;
        }

        for ( k = 0 ; k < NUM ; ++k )
        {
            if ( k == i )
                continue;

            w = gjm[k][i];
            for ( j = 0 ; j < NUM ; ++j )
            {
                gjm[k][j] -= gjm[i][j] * w;
                ilw[k][j] -= ilw[i][j] * w;
            }
        }
    }

    return 1;
}

#undef SWAPF
#undef NUM

/*---------------------------------------------------------------------*


                             MODEL SECTION


 *---------------------------------------------------------------------*/

/* NOTE: Prototypes for these functions are defined in "mtx44ext.h".   */

/*---------------------------------------------------------------------*
Name:           MTX44Trans

Description:    sets a translation matrix.

Arguments:       m        matrix to be set
                xT        x component of translation.
                yT        y component of translation.
                zT        z component of translation.

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44Trans ( Mtx44 m, f32 xT, f32 yT, f32 zT )
{
    ASSERTMSG( (m != 0), MTX44_TRANS_1 );

    m[0][0] = 1.0f;     m[0][1] = 0.0f;  m[0][2] = 0.0f;  m[0][3] =  xT;
    m[1][0] = 0.0f;     m[1][1] = 1.0f;  m[1][2] = 0.0f;  m[1][3] =  yT;
    m[2][0] = 0.0f;     m[2][1] = 0.0f;  m[2][2] = 1.0f;  m[2][3] =  zT;
    m[3][0] = 0.0f;     m[3][1] = 0.0f;  m[3][2] = 0.0f;  m[3][3] =  1.0f;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX44Trans( Mtx44 m, f32 xT, f32 yT, f32 zT )
{
    f32x2 xT2 = {0.0F, xT};
    f32x2 yT2 = {0.0F, yT};
    f32x2 zT2 = {1.0F, zT};
    __PSQ_ST(m, c10, 0, 0);
    __PSQ_STX(m,  8, xT2, 0, 0);
    __PSQ_STX(m, 16, c01, 0, 0);
    __PSQ_STX(m, 24, yT2, 0, 0);
    __PSQ_STX(m, 32, c00, 0, 0);
    __PSQ_STX(m, 40, zT2, 0, 0);
    __PSQ_STX(m, 48, c00, 0, 0);
    __PSQ_STX(m, 56, c01, 0, 0);
}
#endif

/*---------------------------------------------------------------------*
Name:           MTX44TransApply

Description:    This function performs the operation equivalent to
                MTXTrans + MTXConcat.

Arguments:      src       matrix to be operated.
                dst       resultant matrix from concat.
                xT        x component of translation.
                yT        y component of translation.
                zT        z component of translation.

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44TransApply ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xT, f32 yT, f32 zT )
{
    ASSERTMSG( (src != 0), MTX44_TRANSAPPLY_1 );
    ASSERTMSG( (dst != 0), MTX44_TRANSAPPLY_1 );

    if ( src != dst )
    {
        dst[0][0] = src[0][0];    dst[0][1] = src[0][1];    dst[0][2] = src[0][2];
        dst[1][0] = src[1][0];    dst[1][1] = src[1][1];    dst[1][2] = src[1][2];
        dst[2][0] = src[2][0];    dst[2][1] = src[2][1];    dst[2][2] = src[2][2];
        dst[3][0] = src[3][0];    dst[3][1] = src[3][1];    dst[3][2] = src[3][2];
        dst[3][3] = src[3][3];
    }

    dst[0][3] = src[0][3] + xT;
    dst[1][3] = src[1][3] + yT;
    dst[2][3] = src[2][3] + zT;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX44TransApply ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xT, f32 yT, f32 zT )
{
    f32x2 fp4, fp5, fp6, fp7, fp8; //fp0, fp1, fp2, fp3, 
    f32x2 xT2 = {xT, 0.0F};
    f32x2 yT2 = {yT, 0.0F};
    f32x2 zT2 = {zT, 0.0F};

    //psq_l       fp4, 0(src),     0, 0;
    fp4 = __PSQ_L(src, 0, 0);

    //frsp        xT, xT;                         // to make sure xS = single precision
    //psq_l       fp5, 8(src),     0, 0;
    fp5 = __PSQ_LX(src, 8, 0, 0);

    //frsp        yT, yT;                         // to make sure yS = single precision
    //psq_l       fp6, 16(src),    0, 0;
    fp6 = __PSQ_LX(src, 16, 0, 0);

    //frsp        zT, zT;                         // to make sure zS = single precision
    //psq_l       fp7, 24(src),    0, 0;
    fp7 = __PSQ_LX(src, 24, 0, 0);

    //psq_st      fp4, 0(dst),     0, 0;
    __PSQ_ST(dst, fp4, 0, 0);

    //ps_sum1     fp5, xT, fp5, fp5;
    fp5 = __PS_SUM1(xT2, fp5, fp5);

    //psq_l       fp4, 40(src),    0, 0;
    fp4 = __PSQ_LX(src, 40, 0, 0);

    //psq_st      fp6, 16(dst),    0, 0;
    __PSQ_STX(dst, 16, fp6, 0, 0);

    //ps_sum1     fp7, yT, fp7, fp7;
    fp7 = __PS_SUM1(yT2, fp7, fp7);

    //psq_l       fp8, 32(src),    0, 0;
    fp8 = __PSQ_LX(src, 32, 0, 0);

    //psq_st      fp5, 8(dst),     0, 0;
    __PSQ_STX(dst, 8, fp5, 0, 0);

    //ps_sum1     fp4, zT, fp4, fp4;
    fp4 = __PS_SUM1(zT2, fp4, fp4);

    //psq_st      fp7, 24(dst),    0, 0;
    __PSQ_STX(dst, 24, fp7, 0, 0);

    //psq_st      fp8, 32(dst),    0, 0;
    __PSQ_STX(dst, 32, fp8, 0, 0);

    //psq_l       fp5, 48(src),    0, 0;
    fp5 = __PSQ_LX(src, 48, 0, 0);

    //psq_l       fp6, 56(src),    0, 0;
    fp6 = __PSQ_LX(src, 56, 0, 0);

    //psq_st      fp4, 40(dst),    0, 0;
    __PSQ_STX(dst, 40, fp4, 0, 0);

    //psq_st      fp5, 48(dst),    0, 0;
    __PSQ_STX(dst, 48, fp5, 0, 0);

    //psq_st      fp6, 56(dst),    0, 0;
    __PSQ_STX(dst, 56, fp6, 0, 0);
}
#endif

/*---------------------------------------------------------------------*
Name:            MTX44Scale

Description:     sets a scaling matrix.

Arguments:       m        matrix to be set
                xS        x scale factor.
                yS        y scale factor.
                zS        z scale factor.

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44Scale ( Mtx44 m, f32 xS, f32 yS, f32 zS )
{
    ASSERTMSG( (m != 0), MTX44_SCALE_1 );

    m[0][0] = xS;      m[0][1] = 0.0f;  m[0][2] = 0.0f;  m[0][3] = 0.0f;
    m[1][0] = 0.0f;    m[1][1] = yS;    m[1][2] = 0.0f;  m[1][3] = 0.0f;
    m[2][0] = 0.0f;    m[2][1] = 0.0f;  m[2][2] = zS;    m[2][3] = 0.0f;
    m[3][0] = 0.0f;    m[3][1] = 0.0f;  m[3][2] = 0.0f;  m[3][3] = 1.0f;
}


#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX44Scale( Mtx44 m, f32 xS, f32 yS, f32 zS )
{
    f32x2 xS2 = {xS,   0.0F};
    f32x2 yS2 = {0.0F, yS};
    f32x2 zS2 = {zS, 0.0F};

    __PSQ_ST(m, xS2, 0, 0);
    __PSQ_STX(m,  8, c00, 0, 0);
    __PSQ_STX(m, 16, yS2, 0, 0);
    __PSQ_STX(m, 24, c00, 0, 0);
    __PSQ_STX(m, 32, c00, 0, 0);
    __PSQ_STX(m, 40, zS2, 0, 0);
    __PSQ_STX(m, 48, c00, 0, 0);
    __PSQ_STX(m, 56, c01, 0, 0);
}
#endif

/*---------------------------------------------------------------------*
Name:           MTX44ScaleApply

Description:    This function performs the operation equivalent to
                MTXScale + MTXConcat

Arguments:      src       matrix to be operated.
                dst       resultant matrix from concat.
                xS        x scale factor.
                yS        y scale factor.
                zS        z scale factor.

Return:         none
*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44ScaleApply ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xS, f32 yS, f32 zS )
{
    ASSERTMSG( (src != 0), MTX44_SCALEAPPLY_1 );
    ASSERTMSG( (dst != 0), MTX44_SCALEAPPLY_2 );

    dst[0][0] = src[0][0] * xS;     dst[0][1] = src[0][1] * xS;
    dst[0][2] = src[0][2] * xS;     dst[0][3] = src[0][3] * xS;

    dst[1][0] = src[1][0] * yS;     dst[1][1] = src[1][1] * yS;
    dst[1][2] = src[1][2] * yS;     dst[1][3] = src[1][3] * yS;

    dst[2][0] = src[2][0] * zS;     dst[2][1] = src[2][1] * zS;
    dst[2][2] = src[2][2] * zS;     dst[2][3] = src[2][3] * zS;

    dst[3][0] = src[3][0] ; dst[3][1] = src[3][1];
    dst[3][2] = src[3][2] ; dst[3][3] = src[3][3];
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/

void PSMTX44ScaleApply ( MTX_CONST Mtx44 src, Mtx44 dst, f32 xS, f32 yS, f32 zS )
{
    f32x2 fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11; //fp0, fp1, fp2, fp3, 
    f32x2 xS2 = {xS, xS};
    f32x2 yS2 = {yS, yS};
    f32x2 zS2 = {zS, zS};

    //psq_l       fp4,     0(src), 0, 0;          // fp4 <- src00,src01
    //fp4[0] = src[0][0];
    //fp4[1] = src[0][1];
    fp4 = __PSQ_L(src, 0, 0);

    //frsp        xS, xS;                         // to make sure xS = single precision
    //psq_l       fp5,     8(src), 0, 0;          // fp5 <- src02,src03
    //fp5[0] = src[0][2];
    //fp5[1] = src[0][3];
    fp5 = __PSQ_LX(src,  8, 0, 0);

    //frsp        yS, yS;                         // to make sure yS = single precision
    //psq_l       fp6,    16(src), 0, 0;          // fp6 <- src10,src11
    //fp6[0] = src[1][0];
    //fp6[1] = src[1][1];
    fp6 = __PSQ_LX(src,  16, 0, 0);

    //ps_muls0    fp4,    fp4, xS;                // fp4 <- src00*xS,src01*xS
    fp4 = __PS_MULS0(fp4, xS2);

    //psq_l       fp7,    24(src), 0, 0;          // fp7 <- src12,src13
    //fp7[0] = src[1][2];
    //fp7[1] = src[1][3];
    fp7 = __PSQ_LX(src,  24, 0, 0);

    //ps_muls0    fp5,    fp5, xS;                // fp5 <- src02*xS,src03*xS
    fp5 = __PS_MULS0(fp5, xS2);

    //psq_l       fp8,    32(src), 0, 0;          // fp8 <- src20,src21
    //fp8[0] = src[2][0];
    //fp8[1] = src[2][1];
    fp8 = __PSQ_LX(src,  32, 0, 0);

    //frsp        zS, zS;                         // to make sure zS = single precision
    //psq_st      fp4,     0(dst), 0, 0;          // dst00,dst01
    //dst[0][0] = fp4[0];
    //dst[0][1] = fp4[1];
    __PSQ_ST(dst, fp4, 0, 0);

    //ps_muls0    fp6,    fp6, yS;                // fp6 <- src10*yS,src11*yS
    fp6 = __PS_MULS0(fp6, yS2);

    //psq_l       fp9,    40(src), 0, 0;          // fp9 <- src22,src23
    //fp9[0] = src[2][2];
    //fp9[1] = src[2][3];
    fp9 = __PSQ_LX(src, 40, 0, 0);

    //psq_st      fp5,     8(dst), 0, 0;          // dst02,dst03
    //dst[0][2] = fp5[0];
    //dst[0][3] = fp5[1];
    __PSQ_STX(dst, 8, fp5, 0, 0);

    //ps_muls0    fp7,    fp7, yS;                // fp7 <- src12*yS,src13*yS
    fp7 = __PS_MULS0(fp7, yS2);

    //psq_l       fp10,   48(src), 0, 0;          // fp10 <- src30src31
    //fp10[0] = src[3][0];
    //fp10[1] = src[3][1];
    fp10 = __PSQ_LX(src,  48, 0, 0);

    //psq_st      fp6,    16(dst), 0, 0;          // dst10,dst11
    //dst[1][0] = fp6[0];
    //dst[1][1] = fp6[1];
    __PSQ_STX(dst, 16, fp6, 0, 0);

    //ps_muls0    fp8,    fp8, zS;                // fp8 <- src20*zS,src21*zS
    fp8 = __PS_MULS0(fp8, zS2);

    //psq_l       fp11,   56(src), 0, 0;          // fp11 <- src32,src33
    //fp11[0] = src[3][2];
    //fp11[1] = src[3][3];
    fp11 = __PSQ_LX(src,  56, 0, 0);

    //psq_st      fp7,    24(dst), 0, 0;          // dst12,dst13
    //dst[1][2] = fp7[0];
    //dst[1][3] = fp7[1];
    __PSQ_STX(dst, 24, fp7, 0, 0);

    //ps_muls0    fp9,    fp9, zS;                // fp9 <- src22*zS,src23*zS
    fp9 = __PS_MULS0(fp9, zS2);

    //psq_st      fp8,    32(dst), 0, 0;          // dst20,dst21
    //dst[2][0] = fp8[0];
    //dst[2][1] = fp8[1];
    __PSQ_STX(dst, 32, fp8, 0, 0);

    //psq_st      fp9,    40(dst), 0, 0;          // dst22,dst23
    //dst[2][2] = fp9[0];
    //dst[2][3] = fp9[1];
    __PSQ_STX(dst, 40, fp9, 0, 0);

    //psq_st      fp10,   48(dst), 0, 0;          // dst30,dst31
    //dst[3][0] = fp10[0];
    //dst[3][1] = fp10[1];
    __PSQ_STX(dst, 48, fp10, 0, 0);

    //psq_st      fp11,   56(dst), 0, 0;          // dst32,dst33
    //dst[3][2] = fp11[0];
    //dst[3][3] = fp11[1];
    __PSQ_STX(dst, 56, fp11, 0, 0);
}
#endif


/*---------------------------------------------------------------------*
Name:           MTX44RotRad

Description:    sets a rotation matrix about one of the X, Y or Z axes

Arguments:      m       matrix to be set
                axis    major axis about which to rotate.
                        axis is passed in as a character.
                        it must be one of 'X', 'x', 'Y', 'y', 'Z', 'z'
                deg     rotation angle in radians.
                        note:  counter-clockwise rotation is positive.

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44RotRad ( Mtx44 m, char axis, f32 rad )
{
    f32 sinA, cosA;

    ASSERTMSG( (m != 0), MTX44_ROTRAD_1 );

    // verification of "axis" will occur in MTXRotTrig

    sinA = sinf(rad);
    cosA = cosf(rad);

    C_MTX44RotTrig( m, axis, sinA, cosA );
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/

void PSMTX44RotRad ( Mtx44 m, char axis, f32 rad )
{
    f32 sinA, cosA;

    sinA = sinf(rad);
    cosA = cosf(rad);

    PSMTX44RotTrig( m, axis, sinA, cosA );
}
#endif

/*---------------------------------------------------------------------*
Name:           MTX44RotTrig

Arguments:      m       matrix to be set
                axis    major axis about which to rotate.
                        axis is passed in as a character.
                        It must be one of 'X', 'x', 'Y', 'y', 'Z', 'z'
                sinA    sine of rotation angle.
                cosA    cosine of rotation angle.
                        note:  counter-clockwise rotation is positive.

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44RotTrig ( Mtx44 m, char axis, f32 sinA, f32 cosA )
{
    ASSERTMSG( (m != 0), MTX44_ROTTRIG_1 );

    axis |= 0x20;
    switch(axis)
    {

    case 'x':
        m[0][0] =  1.0f;  m[0][1] =  0.0f;    m[0][2] =  0.0f;  m[0][3] = 0.0f;
        m[1][0] =  0.0f;  m[1][1] =  cosA;    m[1][2] = -sinA;  m[1][3] = 0.0f;
        m[2][0] =  0.0f;  m[2][1] =  sinA;    m[2][2] =  cosA;  m[2][3] = 0.0f;
        m[3][0] =  0.0f;  m[3][1] =  0.0f;    m[3][2] =  0.0f;  m[3][3] = 1.0f;
        break;

    case 'y':
        m[0][0] =  cosA;  m[0][1] =  0.0f;    m[0][2] =  sinA;  m[0][3] = 0.0f;
        m[1][0] =  0.0f;  m[1][1] =  1.0f;    m[1][2] =  0.0f;  m[1][3] = 0.0f;
        m[2][0] = -sinA;  m[2][1] =  0.0f;    m[2][2] =  cosA;  m[2][3] = 0.0f;
        m[3][0] =  0.0f;  m[3][1] =  0.0f;    m[3][2] =  0.0f;  m[3][3] = 1.0f;
        break;

    case 'z':
        m[0][0] =  cosA;  m[0][1] = -sinA;    m[0][2] =  0.0f;  m[0][3] = 0.0f;
        m[1][0] =  sinA;  m[1][1] =  cosA;    m[1][2] =  0.0f;  m[1][3] = 0.0f;
        m[2][0] =  0.0f;  m[2][1] =  0.0f;    m[2][2] =  1.0f;  m[2][3] = 0.0f;
        m[3][0] =  0.0f;  m[3][1] =  0.0f;    m[3][2] =  0.0f;  m[3][3] = 1.0f;
        break;

    default:
        ASSERTMSG( 0, MTX44_ROTTRIG_2 );
        break;
    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX44RotTrig( Mtx44  m, char axis, f32 sinA, f32 cosA )
{
    f32x2 ftmp0, ftmp1, ftmp4; //ftmp2, ftmp3, 
    f32x2 sinA10 = {sinA, 0.0F};
    f32x2 cosA10 = {cosA, 0.0F};

    switch(axis)
    {
    case 'x':
    case 'X':
        //psq_st      c_one,   0(m), 1, 0;        // m00 <= 1.0
        __PSQ_ST(m, c11, 1, 0);

        //psq_st      c_zero,  4(m), 0, 0;        // m01,m02 <= 0.0,0.0
        __PSQ_STX(m, 4, c00, 0, 0);

        //ps_neg      ftmp0, sinA;                // ftmp0 <= -sinA
        ftmp0 = __PS_NEG(sinA10);

        //psq_st      c_zero, 12(m), 0, 0;        // m03,m10 <= 0.0,0.0
        __PSQ_STX(m, 12, c00, 0, 0);

        //ps_merge00  ftmp1, sinA, cosA;          // ftmp1 <= sinA,cosA
        ftmp1 = __PS_MERGE00(sinA10, cosA10);

        //psq_st      c_zero, 28(m), 0, 0;        // m13,m20 <= 0.0,0.0
        __PSQ_STX(m, 12, c00, 0, 0);

        //ps_merge00  ftmp0, cosA, ftmp0;         // ftmp0 <= cosA,-sinA
        ftmp0 = __PS_MERGE00(cosA10, ftmp0);

        //psq_st      c_zero, 44(m), 0, 0;        // m23,m30 <= 0.0,0.0
        __PSQ_STX(m, 44, c00, 0, 0);

        //psq_st      c_zero, 52(m), 0, 0;        // m23,m30 <= 0.0,0.0
        __PSQ_STX(m, 52, c00, 0, 0);

        //psq_st      ftmp1,  36(m), 0, 0;        // m21,m22 <= sinA,cosA
        __PSQ_STX(m, 36, ftmp1, 0, 0);

        //psq_st      ftmp0,  20(m), 0, 0;        // m11,m12 <= cosA,-sinA
        __PSQ_STX(m, 20, ftmp0, 0, 0);

        //psq_st      c_one,  60(m), 1, 0;        // m33 <= 0.0
        __PSQ_STX(m, 60, c11, 1, 0);

        break;

    case 'y':
    case 'Y':

        //psq_st      c_zero, 48(m), 0, 0;        // m30,m31 <= 0.0,0.0
        __PSQ_STX(m, 48, c00, 0, 0);

        //ps_neg      ftmp0, sinA;                // ftmp0 <= -sinA,0.0
        ftmp0 = __PS_NEG(sinA10);

        //psq_st      c_zero, 24(m), 0, 0;        // m12,m13 <= 0.0,0.0
        __PSQ_STX(m, 24, c00, 0, 0);

        //psq_st      cosA10,   0(m), 0, 0;        // m00,m01 <= cosA,0.0
        __PSQ_ST(m, cosA10, 0, 0);

        //psq_st      c01,  16(m), 0, 0;        // m10,m11 <= 0.0,1.0
        __PSQ_STX(m, 16, c01, 0, 0);

        //psq_st      sinA10,   8(m), 0, 0;        // m02,m03 <= sinA,0.0
        __PSQ_STX(m, 8, sinA10, 0, 0);

        //psq_st      ftmp0,  32(m), 0, 0;        // m20,m21 <= -sinA,0.0
        __PSQ_STX(m, 32, ftmp0, 0, 0);

        //psq_st      cosA10,  40(m), 0, 0;        // m22,m23 <= cosA,0.0
        __PSQ_STX(m, 40, cosA10, 0, 0);

        //psq_st      c01,  56(m), 0, 0;        // m32,m33 <= 0.0,1.0
        __PSQ_STX(m, 56, c01, 0, 0);

        break;

    case 'z':
    case 'Z':
        //psq_st      c_zero,  8(m), 0, 0;        // m02,m03 <= 0.0,0.0
        __PSQ_STX(m, 8, c00, 0, 0);

        //ps_neg      ftmp0, sinA;                // ftmp0 <= -sinA
        ftmp0 = __PS_NEG(sinA10);

        //psq_st      c_zero, 24(m), 0, 0;        // m12,m13 <= 0.0,0.0
        __PSQ_STX(m, 24, c00, 0, 0);

        //ps_merge00  ftmp1, sinA, cosA;          // ftmp1 <= sinA,cosA
        ftmp1 = __PS_MERGE00(sinA10, cosA10);

        //psq_st      c_zero, 32(m), 0, 0;        // m20,m21 <= 0.0,0.0
        __PSQ_STX(m, 32, c00, 0, 0);

        //psq_st      c_zero, 48(m), 0, 0;        // m30,m31 <= 0.0,0.0
        __PSQ_STX(m, 48, c00, 0, 0);

        //psq_st      ftmp1,  16(m), 0, 0;        // m10,m11 <= sinA,cosA
        __PSQ_STX(m, 16, ftmp1, 0, 0);

        //ps_merge00  ftmp4, cosA, ftmp0;         // ftmp4 <= cosA, -sinA
        ftmp4 = __PS_MERGE00(cosA10, ftmp0);

        //psq_st      ftmp2,  40(m), 0, 0;        // m22,m23 <= 1.0,0.0
        __PSQ_STX(m, 40, c10, 0, 0);

        //psq_st      ftmp3,  56(m), 0, 0;        // m32,m33 <= 0.0,1.0
        __PSQ_STX(m, 56, c01, 0, 0);

        //psq_st      ftmp4,   0(m), 0, 0;        // m00,m00 <= cosA,-sinA
        __PSQ_ST(m, ftmp4, 0, 0);

        break;

    default:
        ASSERTMSG( 0, MTX44_ROTTRIG_2 );
        break;
    }
}
#endif

/*---------------------------------------------------------------------*
Name:           C_MTX44RotAxisRad
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44RotAxisRad( Mtx44 m, const Vec *axis, f32 rad )
{
    Vec vN;
    f32 s, c;             // sinTheta, cosTheta
    f32 t;                // ( 1 - cosTheta )
    f32 x, y, z;          // x, y, z components of normalized axis
    f32 xSq, ySq, zSq;    // x, y, z squared

    ASSERTMSG( (m    != 0), MTX44_ROTAXIS_1  );
    ASSERTMSG( (axis != 0), MTX44_ROTAXIS_2  );

    s = sinf(rad);
    c = cosf(rad);
    t = 1.0f - c;

    C_VECNormalize( axis, &vN );

    x = vN.x;
    y = vN.y;
    z = vN.z;

    xSq = x * x;
    ySq = y * y;
    zSq = z * z;

    m[0][0] = ( t * xSq )   + ( c );
    m[0][1] = ( t * x * y ) - ( s * z );
    m[0][2] = ( t * x * z ) + ( s * y );
    m[0][3] =    0.0f;

    m[1][0] = ( t * x * y ) + ( s * z );
    m[1][1] = ( t * ySq )   + ( c );
    m[1][2] = ( t * y * z ) - ( s * x );
    m[1][3] =    0.0f;

    m[2][0] = ( t * x * z ) - ( s * y );
    m[2][1] = ( t * y * z ) + ( s * x );
    m[2][2] = ( t * zSq )   + ( c );
    m[2][3] =    0.0f;

    m[3][0] = 0.0f;
    m[3][1] = 0.0f;
    m[3][2] = 0.0f;
    m[3][3] = 1.0f;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/

static void __PSMTX44RotAxisRadInternal(
    Mtx44  m,
    const Vec   *axis,
    f32    sT,
    f32    cT )
{
    f32x2    tT2;
    f32x2    sT2 = {sT, sT};
    f32x2    cT2 = {cT, cT};
    f32x2    tmp0, tmp1, tmp2, tmp3, tmp4;
    f32x2    tmp5, tmp6, tmp7, tmp8, tmp9;

    // tmp0 = [x][y] : LOAD
    //psq_l       tmp0, 0(axis), 0, 0
    //tmp0[0] = axis->x;
    //tmp0[1] = axis->y;
    tmp0 = __PSQ_LX(axis,  0, 0, 0);

    // tmp1 = [z][z] : LOAD
    //lfs         tmp1, 8(axis)
    tmp1[0] = axis->z;
    tmp1[1] = axis->z;

    // tmp2 = [x*x][y*y]
    //ps_mul      tmp2, tmp0, tmp0
    tmp2 = __PS_MUL(tmp0, tmp0);

    // tmp7 = [1.0F]
    //fadds       tmp7, tmp9, tmp9

    // tmp3 = [x*x+z*z][y*y+z*z]
    //ps_madd     tmp3, tmp1, tmp1, tmp2
    tmp3 = __PS_MADD(tmp1, tmp1, tmp2);

    // fc0 = [0.0F]
    //fsubs       fc0, tmp9, tmp9

    // tmp4 = [S = x*x+y*y+z*z][z]
    //ps_sum0     tmp4, tmp3, tmp1, tmp2
    tmp4 = __PS_SUM0(tmp3, tmp1, tmp2);

    // tT = 1.0F - cT
    //fsubs       tT, tmp7, cT
    tT2 = __PS_SUB(c11, cT2);

    // tmp5 = [1.0/sqrt(S)] :estimation[E]
    //frsqrte     tmp5, tmp4
    tmp5 = __PS_RSQRTE(tmp4);

    // tmp7 = [0][1]
    //ps_merge00  tmp7, fc0, tmp7
    tmp7 = __PS_MERGE00(c00, c11);

    // Newton-Rapson refinement step
    // E' = E/2(3.0 - E*E*S)
    //fmuls       tmp2, tmp5, tmp5            // E*E
    tmp2 = __PS_MUL(tmp5, tmp5);

    //fmuls       tmp3, tmp5, tmp9            // E/2
    tmp3 = __PS_MUL(tmp5, c0505);

    // fc0 [m30=0][m31=0] : STORE
    //psq_st      fc0, 48(m), 0, 0
    //m[3][0] = 0.0F;
    //m[3][1] = 0.0F;
    __PSQ_STX(m, 48, c00, 0, 0);

    //fnmsubs     tmp2, tmp2, tmp4, tmp8      // (3-E*E*S)
    tmp2 = __PS_NMSUB(tmp2, tmp4, c33);

    //fmuls       tmp5, tmp2, tmp3            // (E/2)(3-E*E*S)
    tmp5 = __PS_MUL(tmp2, tmp3);

    // tmp7 [m32=0][m33=1] : STORE
    //psq_st      tmp7, 56(m), 0, 0
    //m[3][2] = 0.0F;
    //m[3][3] = 1.0F;
    __PSQ_STX(m, 56, tmp7, 0, 0);

    // cT = [c][c]
    //ps_merge00  cT, cT, cT

    // tmp0 = [nx = x/sqrt(S)][ny = y/sqrt(S)]
    //ps_muls0    tmp0, tmp0, tmp5
    tmp0 = __PS_MULS0(tmp0, tmp5);

    // tmp1 = [nz = z/sqrt(S)][nz = z/sqrt(S)]
    //ps_muls0    tmp1, tmp1, tmp5
    tmp1 = __PS_MULS0(tmp1, tmp5);

    // tmp4 = [t*nx][t*ny]
    //ps_muls0    tmp4, tmp0, tT
    tmp4 = __PS_MULS0(tmp0, tT2);

    // tmp9 = [s*nx][s*ny]
    //ps_muls0    tmp9, tmp0, sT
    tmp9 = __PS_MULS0(tmp0, sT2);

    // tmp5 = [t*nz][t*nz]
    //ps_muls0    tmp5, tmp1, tT
    tmp5 = __PS_MULS0(tmp1, tT2);

    // tmp3 = [t*nx*ny][t*ny*ny]
    //ps_muls1    tmp3, tmp4, tmp0
    tmp3 = __PS_MULS1(tmp4, tmp0);

    // tmp2 = [t*nx*nx][t*ny*nx]
    //ps_muls0    tmp2, tmp4, tmp0
    tmp2 = __PS_MULS0(tmp4, tmp0);

    // tmp4 = [t*nx*nz][t*ny*nz]
    //ps_muls0    tmp4, tmp4, tmp1
    tmp4 = __PS_MULS0(tmp4, tmp1);

    // tmp6 = [t*nx*nx-s*nz][t*ny*ny-s*nz]
    //fnmsubs     tmp6, tmp1, sT, tmp2
    tmp6 = __PS_NMSUB(tmp1, sT2, tmp2);

    // tmp7 = [t*nx*ny+s*nz][t*ny*ny+s*nz]
    //fmadds      tmp7, tmp1, sT, tmp3
    tmp7 = __PS_MADD(tmp1, sT2, tmp3);

    // tmp0 = [-s*nx][-s*ny]
    //ps_neg      tmp0, tmp9
    tmp0 = __PS_NEG(tmp9);

    // tmp8 = [t*nx*nz+s*ny][0] == [m02][m03]
    //ps_sum0     tmp8, tmp4, fc0, tmp9
    tmp8 = __PS_SUM0(tmp4, c00, tmp9);

    // tmp2 = [t*nx*nx+c][t*nx*ny-s*nz] == [m00][m01]
    //ps_sum0     tmp2, tmp2, tmp6, cT
    tmp2 = __PS_SUM0(tmp2, tmp6, cT2);

    // tmp3 = [t*nx*ny+s*nz][t*ny*ny+c] == [m10][m11]
    //ps_sum1     tmp3, cT, tmp7, tmp3
    tmp3 = __PS_SUM1(cT2, tmp7, tmp3);

    // tmp6 = [t*ny*nz-s*nx][0] == [m12][m13]
    //ps_sum0     tmp6, tmp0, fc0 ,tmp4
    tmp6 = __PS_SUM0(tmp0, c00, tmp4);

    // tmp8 [m02][m03] : STORE
    //psq_st      tmp8, 8(m), 0, 0
    //m[0][2] = tmp8[0];
    //m[0][3] = tmp8[1];
    __PSQ_STX(m, 8, tmp8, 0, 0);

    // tmp0 = [t*nx*nz-s*ny][t*ny*nz]
    //ps_sum0     tmp0, tmp4, tmp4, tmp0
    tmp0 = __PS_SUM0(tmp4, tmp4, tmp0);

    // tmp2 [m00][m01] : STORE
    //psq_st      tmp2, 0(m), 0, 0
    //m[0][0] = tmp2[0];
    //m[0][1] = tmp2[1];
    __PSQ_STX(m, 0, tmp2, 0, 0);

    // tmp5 = [t*nz*nz][t*nz*nz]
    //ps_muls0    tmp5, tmp5, tmp1
    tmp5 = __PS_MULS0(tmp5, tmp1);

    // tmp3 [m10][m11] : STORE
    //psq_st      tmp3, 16(m), 0, 0
    //m[1][0] = tmp3[0];
    //m[1][1] = tmp3[1];
    __PSQ_STX(m, 16, tmp3, 0, 0);

    // tmp4 = [t*nx*nz-s*ny][t*ny*nz+s*nx] == [m20][m21]
    //ps_sum1     tmp4, tmp9, tmp0, tmp4
    tmp4 = __PS_SUM1(tmp9, tmp0, tmp4);

    // tmp6 [m12][m13] : STORE
    //psq_st      tmp6, 24(m), 0, 0
    //m[1][2] = tmp6[0];
    //m[1][3] = tmp6[1];
    __PSQ_STX(m, 24, tmp6, 0, 0);

    // tmp5 = [t*nz*nz+c][0]   == [m22][m23]
    //ps_sum0     tmp5, tmp5, fc0, cT
    tmp5 = __PS_SUM0(tmp5, c00, cT2);

    // tmp4 [m20][m21] : STORE
    //psq_st      tmp4, 32(m), 0, 0
    //m[2][0] = tmp4[0];
    //m[2][1] = tmp4[1];
    __PSQ_STX(m, 32, tmp4, 0, 0);

    // tmp5 [m22][m23] : STORE
    //psq_st      tmp5, 40(m), 0, 0
    //m[2][2] = tmp5[0];
    //m[2][3] = tmp5[1];
    __PSQ_STX(m, 40, tmp5, 0, 0);

}

void PSMTX44RotAxisRad( Mtx44 m, const Vec *axis, f32 rad )
{
    f32     sinT, cosT;

    sinT = sinf(rad);
    cosT = cosf(rad);

    __PSMTX44RotAxisRadInternal(m, axis, sinT, cosT);
}
#endif


/*---------------------------------------------------------------------------*
    MATRIX COLWERSION
 *---------------------------------------------------------------------------*/
void C_MTX34To44 ( MTX_CONST Mtx src, Mtx44 dst)
{
    dst[0][0] = src[0][0];    dst[0][1] = src[0][1];    dst[0][2] = src[0][2];    dst[0][3] = src[0][3];
    dst[1][0] = src[1][0];    dst[1][1] = src[1][1];    dst[1][2] = src[1][2];    dst[1][3] = src[1][3];
    dst[2][0] = src[2][0];    dst[2][1] = src[2][1];    dst[2][2] = src[2][2];    dst[2][3] = src[2][3];
    dst[3][0] = 0.0f;         dst[3][1] = 0.0f;         dst[3][2] = 0.0f;         dst[3][3] = 1.0f;
}


#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single assembler version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX34To44( MTX_CONST Mtx src, Mtx44 dst )
{
    f32x2 fp1;

    //psq_l       fp1,  0(src), 0, 0;
    fp1 = __PSQ_L(src, 0, 0);

    //psq_st      fp1,  0(dst), 0, 0;
    __PSQ_ST(dst, fp1, 0, 0);

    //psq_l       fp1,  8(src), 0, 0;
    fp1 = __PSQ_LX(src, 8, 0, 0);

    //psq_st      fp1,  8(dst), 0, 0;
    __PSQ_STX(dst, 8, fp1, 0, 0);

    //psq_l       fp1, 16(src), 0, 0;
    fp1 = __PSQ_LX(src, 16, 0, 0);

    //psq_st      fp1, 16(dst), 0, 0;
    __PSQ_STX(dst, 16, fp1, 0, 0);

    //psq_l       fp1, 24(src), 0, 0;
    fp1 = __PSQ_LX(src, 24, 0, 0);

    //psq_st      fp1, 24(dst), 0, 0;
    __PSQ_STX(dst, 24, fp1, 0, 0);

    //psq_l       fp1, 32(src), 0, 0;
    fp1 = __PSQ_LX(src, 32, 0, 0);

    //psq_st      fp1, 32(dst), 0, 0;
    __PSQ_STX(dst, 32, fp1, 0, 0);

    //psq_l       fp1, 40(src), 0, 0;
    fp1 = __PSQ_LX(src, 40, 0, 0);

    //psq_st      fp1, 40(dst), 0, 0;
    __PSQ_STX(dst, 40, fp1, 0, 0);

    //psq_st      c00, 48(dst), 0, 0;
    __PSQ_STX(dst, 48, c00, 0, 0);

    //psq_st      c01, 56(dst), 0, 0;
    __PSQ_STX(dst, 56, c01, 0, 0);
}

/*===========================================================================*/


extern void _ASM_MTX44RotAxisRadInternal(Mtx m, const Vec *axis, f32 sT, f32 cT);

void ASM_MTX44RotAxisRad(Mtx44 m, const Vec *axis, f32 rad) {
    f32     sinT, cosT;

    sinT = sinf(rad);
    cosT = cosf(rad);

    _ASM_MTX44RotAxisRadInternal(m, axis, sinT, cosT);
}

void ASM_MTX44RotRad ( Mtx44 m, char axis, f32 rad )
{
    f32 sinA, cosA;

    sinA = sinf(rad);
    cosA = cosf(rad);

    ASM_MTX44RotTrig( m, axis, sinA, cosA );
}
#endif
