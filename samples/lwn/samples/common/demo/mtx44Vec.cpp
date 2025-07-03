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

                             MODEL SECTION

 *---------------------------------------------------------------------*/
/* NOTE: Prototypes for these functions are defined in "mtx44ext.h".   */

/*---------------------------------------------------------------------*
Name:           MTX44MultVec

Description:    multiplies a vector by a matrix.
                m x src = dst.

Arguments:      m         matrix.
                src       source vector for multiply.
                dst       resultant vector from multiply.
                note:      ok if src == dst.

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44MultVec ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst )
{
    Vec vTmp;
    f32 w;

    ASSERTMSG( (m   != 0), MTX44_MULTVEC_1 );
    ASSERTMSG( (src != 0), MTX44_MULTVEC_2 );
    ASSERTMSG( (dst != 0), MTX44_MULTVEC_3 );

    // a Vec has a 4th implicit 'w' coordinate of 1
    vTmp.x = m[0][0]*src->x + m[0][1]*src->y + m[0][2]*src->z + m[0][3];
    vTmp.y = m[1][0]*src->x + m[1][1]*src->y + m[1][2]*src->z + m[1][3];
    vTmp.z = m[2][0]*src->x + m[2][1]*src->y + m[2][2]*src->z + m[2][3];
    w      = m[3][0]*src->x + m[3][1]*src->y + m[3][2]*src->z + m[3][3];
    w = 1.0f/w;

    // copy back
    dst->x = vTmp.x * w;
    dst->y = vTmp.y * w;
    dst->z = vTmp.z * w;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that NO error checking is performed.
 *---------------------------------------------------------------------*/
void PSMTX44MultVec ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst )
{
    f32x2 fp0, fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp12, fp13; //fp10, fp11, 

    //psq_l       fp0, 0(src),    0, 0;       // fp0 <-src.x, src.y
    //fp0[0] = src->x;
    //fp0[1] = src->y;
    fp0 = __PSQ_LX(src, 0, 0, 0);

    //psq_l       fp2, 48(m),     0, 0;
    //fp2[0] = m[3][0];
    //fp2[1] = m[3][1];
    fp2 = __PSQ_LX(m, 48, 0, 0);

    //psq_l       fp1, 8(src),    1, 0;       // fp1 <-src.z, 1.0
    //fp1[0] = src->z;
    //fp1[1] = 1.0F;
    fp1 = __PSQ_LX(src, 8, 1, 0);

    //ps_mul      fp4, fp0, fp2;
    fp4 = __PS_MUL(fp0, fp2);

    //psq_l       fp3, 56(m),     0, 0;
    //fp3[0] = m[3][2];
    //fp3[1] = m[3][3];
    fp3 = __PSQ_LX(m, 56, 0, 0);

    //ps_madd     fp5, fp1, fp3, fp4;
    fp5 = __PS_MADD(fp1, fp3, fp4);

    //ps_merge11  fp12, fp1, fp1;             // fp12 = 1.0, 1.0
    fp12 = __PS_MERGE11(fp1, fp1);

    //ps_sum0     fp13, fp5, fp5, fp5;        // fp3 <-  w
    fp13 = __PS_SUM0(fp5, fp5, fp5);

    //psq_l       fp4, 0(m),      0, 0;
    //fp4[0] = m[0][0];
    //fp4[1] = m[0][1];
    fp4 = __PSQ_LX(m, 0, 0, 0);

    //ps_merge00  fp13, fp13, fp13;
    fp13 = __PS_MERGE00(fp13, fp13);

    //psq_l       fp5, 8(m),      0, 0;
    //fp5[0] = m[0][2];
    //fp5[1] = m[0][3];
    fp5 = __PSQ_LX(m, 8, 0, 0);

    //ps_div      fp13, fp12, fp13;           // fp13 <- 1/w
    fp13 = __PS_DIV(fp12, fp13);

    //psq_l       fp6, 16(m),     0, 0;
    //fp6[0] = m[1][0];
    //fp6[1] = m[1][1];
    fp6 = __PSQ_LX(m, 16, 0, 0);

    //psq_l       fp7, 24(m),     0, 0;
    //fp7[0] = m[1][2];
    //fp7[1] = m[1][3];
    fp7 = __PSQ_LX(m, 24, 0, 0);

    //psq_l       fp8, 32(m),     0, 0;
    //fp8[0] = m[2][0];
    //fp8[1] = m[2][1];
    fp8 = __PSQ_LX(m, 32, 0, 0);

    //psq_l       fp9, 40(m),     0, 0;
    //fp9[0] = m[2][2];
    //fp9[1] = m[2][3];
    fp9 = __PSQ_LX(m, 40, 0, 0);

    //ps_mul      fp4, fp0, fp4;
    fp4 = __PS_MUL(fp0, fp4);

    //ps_madd     fp2, fp1, fp5, fp4;
    fp2 = __PS_MADD(fp1, fp5, fp4);

    //ps_mul      fp6, fp0, fp6;
    fp6 = __PS_MUL(fp0, fp6);

    //ps_madd     fp3, fp1, fp7, fp6;
    fp3 = __PS_MADD(fp1, fp7, fp6);

    //ps_mul      fp8, fp0, fp8;
    fp8 = __PS_MUL(fp0, fp8);

    //ps_sum0     fp2, fp2, fp2, fp2;         // fp2 <- dst.x, --
    fp2 = __PS_SUM0(fp2, fp2, fp2);

    //ps_madd     fp9, fp1, fp9, fp8;
    fp9 = __PS_MADD(fp1, fp9, fp8);

    //ps_sum1     fp2, fp3, fp2, fp3;         // fp2 <- dst.x, dst.y
    fp2 = __PS_SUM1(fp3, fp2, fp3);

    //ps_sum0     fp3, fp9, fp9, fp9;
    fp3 = __PS_SUM0(fp9, fp9, fp9);

    //ps_mul      fp2, fp2, fp13;
    fp2 = __PS_MUL(fp2, fp13);

    //psq_st      fp2, 0(dst),    0, 0;
    //dst->x = fp2[0];
    //dst->y = fp2[1];
    __PSQ_STX(dst, 0, fp2, 0, 0);

    //ps_mul      fp3, fp3, fp13;
    fp3 = __PS_MUL(fp3, fp13);

    //psq_st      fp3, 8(dst),    1, 0;
    //dst->z = fp3[0];
    __PSQ_STX(dst, 8, fp3, 1, 0);
}
#endif

/*---------------------------------------------------------------------*
Name:           MTX44MultVecArray

Description:    multiplies an array of vectors by a matrix.


Arguments:      m         matrix.
                srcBase   start of source vector array.
                dstBase   start of resultant vector array.
                note:     ok if srcBase == dstBase.
                count     number of vectors in srcBase, dstBase arrays
                          note:      cannot check for array overflow

Return:         none
 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTX44MultVecArray ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase, u32 count )
{
    u32 i;
    Vec vTmp;
    f32 w;

    ASSERTMSG( (m       != 0),    MTX44_MULTVECARRAY_1    );
    ASSERTMSG( (srcBase != 0),    MTX44_MULTVECARRAY_2    );
    ASSERTMSG( (dstBase != 0),    MTX44_MULTVECARRAY_3    );

    for(i=0; i< count; i++)
    {
        // Vec has a 4th implicit 'w' coordinate of 1
        vTmp.x = m[0][0]*srcBase->x + m[0][1]*srcBase->y + m[0][2]*srcBase->z + m[0][3];
        vTmp.y = m[1][0]*srcBase->x + m[1][1]*srcBase->y + m[1][2]*srcBase->z + m[1][3];
        vTmp.z = m[2][0]*srcBase->x + m[2][1]*srcBase->y + m[2][2]*srcBase->z + m[2][3];
        w      = m[3][0]*srcBase->x + m[3][1]*srcBase->y + m[3][2]*srcBase->z + m[3][3];
        w = 1.0f/w;

        // copy back
        dstBase->x = vTmp.x * w;
        dstBase->y = vTmp.y * w;
        dstBase->z = vTmp.z * w;

        srcBase++;
        dstBase++;
    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that NO error checking is performed.
 *---------------------------------------------------------------------*/
void PSMTX44MultVecArray ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase, u32 count )
{
    u32 i;

    for(i=0; i< count; i++)
    {
        PSMTX44MultVec(m, srcBase, dstBase);

        srcBase++;
        dstBase++;
    }
}
#endif


/*---------------------------------------------------------------------*
Name:         MTX44MultVecSR

Description:  multiplies a vector by a matrix 3x3 (Scaling and Rotation)
              component.

              m x src = dst.

Arguments:    m       matrix.
              src     source vector for multiply.
              dst     resultant vector from multiply.
              note:   ok if src == dst.

Return:       none
 *---------------------------------------------------------------------*/
void C_MTX44MultVecSR ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst )
{
    Vec vTmp;

    ASSERTMSG( (m   != 0), MTX44_MULTVECSR_1 );
    ASSERTMSG( (src != 0), MTX44_MULTVECSR_2 );
    ASSERTMSG( (dst != 0), MTX44_MULTVECSR_3 );

    // a Vec has a 4th implicit 'w' coordinate of 1
    vTmp.x = m[0][0]*src->x + m[0][1]*src->y + m[0][2]*src->z;
    vTmp.y = m[1][0]*src->x + m[1][1]*src->y + m[1][2]*src->z;
    vTmp.z = m[2][0]*src->x + m[2][1]*src->y + m[2][2]*src->z;

    // copy back
    dst->x = vTmp.x;
    dst->y = vTmp.y;
    dst->z = vTmp.z;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX44MultVecSR ( MTX_CONST Mtx44 m, const Vec *src, Vec *dst )
{
    f32x2 fp0, fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12, fp13;

    //psq_l   fp0, 0(m), 0, 0    // m[0][0], m[0][1] GQR0 = 0
    //fp0[0] = m[0][0];
    //fp0[1] = m[0][1];
    fp0 = __PSQ_LX(m, 0, 0, 0);

    // fp6 - x y
    //psq_l   fp6, 0(src), 0, 0
    //fp6[0] = src->x;
    //fp6[1] = src->y;
    fp6 = __PSQ_LX(src, 0, 0, 0);

    //psq_l   fp2, 16(m), 0, 0   // m[1][0], m[1][1]
    //fp2[0] = m[1][0];
    //fp2[1] = m[1][1];
    fp2 = __PSQ_LX(m, 16, 0, 0);

    // fp8 = m00x m01y // next X
    //ps_mul  fp8, fp0, fp6
    fp8 = __PS_MUL(fp0, fp6);

    //psq_l   fp4, 32(m), 0, 0   // m[2][0], m[2][1]
    //fp4[0] = m[2][0];
    //fp4[1] = m[2][1];
    fp4 = __PSQ_LX(m, 32, 0, 0);

    // fp10 = m10x m11y // next Y
    //ps_mul  fp10, fp2, fp6
    fp10 = __PS_MUL(fp2, fp6);

    //psq_l   fp7, 8(src), 1, 0   // fp7 - z,1.0
    //fp7[0] = src->z;
    //fp7[1] = 1.0F;
    fp7 = __PSQ_LX(src, 8, 1, 0);

    // fp12 = m20x m21y // next Z
    //ps_mul  fp12, fp4, fp6
    fp12 = __PS_MUL(fp4, fp6);

    //psq_l   fp3, 24(m), 0, 0   // m[1][2], m[1][3]
    //fp3[0] = m[1][2];
    //fp3[1] = m[1][3];
    fp3 = __PSQ_LX(m, 24, 0, 0);

    //ps_sum0 fp8, fp8, fp8, fp8
    fp8 = __PS_SUM0(fp8, fp8, fp8);

    //psq_l   fp5, 40(m), 0, 0   // m[2][2], m[2][3]
    //fp5[0] = m[2][2];
    //fp5[1] = m[2][3];
    fp5 = __PSQ_LX(m, 40, 0, 0);

    //ps_sum0 fp10, fp10, fp10, fp10
    fp10 = __PS_SUM0(fp10, fp10, fp10);

    //psq_l   fp1,  8(m), 0, 0    // m[0][2], m[0][3]
    //fp1[0] = m[0][2];
    //fp1[1] = m[0][3];
    fp1 = __PSQ_LX(m, 8, 0, 0);

    //ps_sum0 fp12, fp12, fp12, fp12
    fp12 = __PS_SUM0(fp12, fp12, fp12);

    //ps_madd fp9, fp1, fp7, fp8
    fp9 = __PS_MADD(fp1, fp7, fp8);

    //psq_st  fp9,  0(dst), 1, 0      // store X
    //dst->x = fp9[0];
    __PSQ_STX(dst, 0, fp9, 1, 0);

    //ps_madd fp11, fp3, fp7, fp10
    fp11 = __PS_MADD(fp3, fp7, fp10);

    //psq_st  fp11, 4(dst), 1, 0      // store Y
    //dst->y = fp11[0];
    __PSQ_STX(dst, 4, fp11, 1, 0);

    //ps_madd fp13, fp5, fp7, fp12
    fp13 = __PS_MADD(fp5, fp7, fp12);

    //psq_st  fp13, 8(dst), 1, 0      //  sore Z
    //dst->z = fp13[0];
    __PSQ_STX(dst, 8, fp13, 1, 0);
}
#endif

/*---------------------------------------------------------------------*
Name:           MTX44MultVecArraySR

Description:    multiplies an array of vectors by a matrix 3x3
                (Scaling and Rotation) component.

Arguments:      m        matrix.
                srcBase  start of source vector array.
                dstBase  start of resultant vector array.
                note:    ok if srcBase == dstBase.

                count    number of vectors in srcBase, dstBase arrays
                note:    cannot check for array overflow

Return:         none
 *---------------------------------------------------------------------*/
void C_MTX44MultVecArraySR ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase, u32 count )
{
    u32 i;
    Vec vTmp;

    ASSERTMSG( (m       != 0), MTX44_MULTVECARRAYSR_1 );
    ASSERTMSG( (srcBase != 0), MTX44_MULTVECARRAYSR_2 );
    ASSERTMSG( (dstBase != 0), MTX44_MULTVECARRAYSR_3 );

    for ( i = 0; i < count; i ++ )
    {
        // Vec has a 4th implicit 'w' coordinate of 1
        vTmp.x = m[0][0]*srcBase->x + m[0][1]*srcBase->y + m[0][2]*srcBase->z;
        vTmp.y = m[1][0]*srcBase->x + m[1][1]*srcBase->y + m[1][2]*srcBase->z;
        vTmp.z = m[2][0]*srcBase->x + m[2][1]*srcBase->y + m[2][2]*srcBase->z;

        // copy back
        dstBase->x = vTmp.x;
        dstBase->y = vTmp.y;
        dstBase->z = vTmp.z;

        srcBase++;
        dstBase++;
    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTX44MultVecArraySR ( MTX_CONST Mtx44 m, const Vec *srcBase, Vec *dstBase, u32 count )
{
    u32 i;

    for ( i = 0; i < count; i ++ )
    {
        PSMTX44MultVecSR(m, srcBase, dstBase);
        srcBase++;
        dstBase++;
    }
}
#endif


/*===========================================================================*/
