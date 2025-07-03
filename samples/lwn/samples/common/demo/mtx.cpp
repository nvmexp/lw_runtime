/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

#include <math.h>
#include <stdio.h>
#include <mtx.h>
#include "mtxAssert.h"

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


                            GENERAL SECTION


*---------------------------------------------------------------------*/


/*---------------------------------------------------------------------*

Name:           MTXIdentity

Description:    sets a matrix to identity

Arguments:      m :  matrix to be set

Return:         none

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXIdentity ( Mtx m )
{
    ASSERTMSG( (m != 0), MTX_IDENTITY_1 );

    m[0][0] = 1.0f;     m[0][1] = 0.0f;  m[0][2] = 0.0f;  m[0][3] = 0.0f;
    m[1][0] = 0.0f;     m[1][1] = 1.0f;  m[1][2] = 0.0f;  m[1][3] = 0.0f;
    m[2][0] = 0.0f;     m[2][1] = 0.0f;  m[2][2] = 1.0f;  m[2][3] = 0.0f;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXIdentity( Mtx m )
{

    //psq_st      c00, 8(m),   0, 0     // m[0][2], m[0][3]
    __PSQ_STX(m, 8, c00, 0, 0);

    //psq_st      c00, 24(m),  0, 0     // m[1][2], m[1][3]
    __PSQ_STX(m, 24, c00, 0, 0);

    //psq_st      c00, 32(m),  0, 0     // m[2][0], m[2][1]
    __PSQ_STX(m, 32, c00, 0, 0);

    //psq_st      c01,   16(m),  0, 0     // m[1][0], m[1][1]
    __PSQ_STX(m, 16, c01, 0, 0);

    //psq_st      c10,   0(m),   0, 0     // m[0][0], m[0][1]
    __PSQ_STX(m, 0, c10, 0, 0);

    //psq_st      c10,   40(m),  0, 0     // m[2][2], m[2][3]
    __PSQ_STX(m, 40, c10, 0, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXCopy

Description:    copies the contents of one matrix into another

Arguments:      src        source matrix for copy
                dst        destination matrix for copy

Return:         none

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXCopy ( MTX_CONST Mtx src, Mtx dst )
{
    ASSERTMSG( (src != 0) , MTX_COPY_1 );
    ASSERTMSG( (dst != 0) , MTX_COPY_2 );

    if( src == dst )
    {
        return;
    }

    dst[0][0] = src[0][0];    dst[0][1] = src[0][1];    dst[0][2] = src[0][2];    dst[0][3] = src[0][3];
    dst[1][0] = src[1][0];    dst[1][1] = src[1][1];    dst[1][2] = src[1][2];    dst[1][3] = src[1][3];
    dst[2][0] = src[2][0];    dst[2][1] = src[2][1];    dst[2][2] = src[2][2];    dst[2][3] = src[2][3];
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXCopy(MTX_CONST Mtx src, Mtx dst )
{
    f32x2 fp0, fp1, fp2, fp3, fp4, fp5;

    //psq_l       fp0, 0(src),   0, 0
    fp0 = __PSQ_L(src, 0, 0);

    //psq_st      fp0, 0(dst),   0, 0
    __PSQ_ST(dst, fp0, 0, 0);

    //psq_l       fp1, 8(src),   0, 0
    fp1 = __PSQ_LX(src, 8, 0, 0);

    //psq_st      fp1, 8(dst),   0, 0
    __PSQ_STX(dst, 8, fp1, 0, 0);

    //psq_l       fp2, 16(src),  0, 0
    fp2 = __PSQ_LX(src, 16, 0, 0);

    //psq_st      fp2, 16(dst),  0, 0
    __PSQ_STX(dst, 16, fp2, 0, 0);

    //psq_l       fp3, 24(src),  0, 0
    fp3 = __PSQ_LX(src, 24, 0, 0);

    //psq_st      fp3, 24(dst),  0, 0
    __PSQ_STX(dst, 24, fp3, 0, 0);

    //psq_l       fp4, 32(src),  0, 0
    fp4 = __PSQ_LX(src, 32, 0, 0);

    //psq_st      fp4, 32(dst),  0, 0
    __PSQ_STX(dst, 32, fp4, 0, 0);

    //psq_l       fp5, 40(src),  0, 0
    fp5 = __PSQ_LX(src, 40, 0, 0);

    //psq_st      fp5, 40(dst),  0, 0
    __PSQ_STX(dst, 40, fp5, 0, 0);

}
#endif

/*---------------------------------------------------------------------*

Name:           MTXConcat

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
void C_MTXConcat ( MTX_CONST Mtx a, MTX_CONST Mtx b, Mtx ab )
{
    Mtx mTmp;
    MtxPtr m;

    ASSERTMSG( (a  != 0), MTX_CONCAT_1 );
    ASSERTMSG( (b  != 0), MTX_CONCAT_2 );
    ASSERTMSG( (ab != 0), MTX_CONCAT_3 );

    if( (ab == a) || (ab == b) )
    {
        m = mTmp;
    }

    else
    {
        m = ab;
    }

    // compute (a x b) -> m

    m[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0];
    m[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1];
    m[0][2] = a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2];
    m[0][3] = a[0][0]*b[0][3] + a[0][1]*b[1][3] + a[0][2]*b[2][3] + a[0][3];

    m[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0];
    m[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1];
    m[1][2] = a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2];
    m[1][3] = a[1][0]*b[0][3] + a[1][1]*b[1][3] + a[1][2]*b[2][3] + a[1][3];

    m[2][0] = a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0];
    m[2][1] = a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1];
    m[2][2] = a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2];
    m[2][3] = a[2][0]*b[0][3] + a[2][1]*b[1][3] + a[2][2]*b[2][3] + a[2][3];

    // overwrite a or b if needed
    if(m == mTmp)
    {
        C_MTXCopy( *((MTX_CONST Mtx *)&mTmp), ab );
    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXConcat ( MTX_CONST Mtx a, MTX_CONST Mtx b, Mtx ab )
{
    f32x2 A00_A01 = __PSQ_L(a, 0, 0);
    f32x2 A02_A03;
    f32x2 A10_A11;
    f32x2 A12_A13;
    f32x2 A20_A21;
    f32x2 A22_A23;
    f32x2 B00_B01 = __PSQ_L(b, 0, 0);
    f32x2 B02_B03 = __PSQ_LX(b,  8, 0, 0);
    f32x2 B10_B11 = __PSQ_LX(b, 16, 0, 0);
    f32x2 B12_B13;
    f32x2 B20_B21;
    f32x2 B22_B23;

    f32x2 D00_D01;
    f32x2 D02_D03;
    f32x2 D10_D11;
    f32x2 D12_D13;
    f32x2 D20_D21;
    f32x2 D22_D23;

    // D00_D01 = b00a00 , b01a00
    D00_D01 = __PS_MULS0( B00_B01, A00_A01);
    A10_A11 = __PSQ_LX(a, 16, 0, 0);

    // D02_D03 = b02a00 , b03a00
    D02_D03 = __PS_MULS0( B02_B03, A00_A01);

    // D10_D11 = a10b00 , a10b01
    D10_D11 = __PS_MULS0( B00_B01, A10_A11);
    B12_B13 = __PSQ_LX(b, 24, 0, 0);

    // D12_D13 = a10b02 , a10b03
    D12_D13 = __PS_MULS0( B02_B03, A10_A11);
    A02_A03 = __PSQ_LX(a,  8, 0, 0);

    // D00_D01 = b10a01 + b00a00 , b11a01 + b01a00
    D00_D01 = __PS_MADDS1( B10_B11, A00_A01, D00_D01);
    A12_A13 = __PSQ_LX(a, 24, 0, 0);

    // D10_D11 = a10b00 + a11b10 , a10b01 + a11b11
    D10_D11 =  __PS_MADDS1( B10_B11, A10_A11, D10_D11);
    B20_B21 = __PSQ_LX(b, 32, 0, 0);

    // D02_D03 = b12a01 + b02a00 , b13a01 + b03a00
    D02_D03 =  __PS_MADDS1( B12_B13, A00_A01, D02_D03);
    B22_B23 = __PSQ_LX(b, 40, 0, 0);

    // D12_D13 = a10b02 + a11b12, a10b03+a11b13
    D12_D13 =  __PS_MADDS1( B12_B13, A10_A11, D12_D13);

    A20_A21 = __PSQ_LX(a, 32, 0, 0);
    A22_A23 = __PSQ_LX(a, 40, 0, 0);

    // D00_D01 = b20a02 + b10a01 + b00a00 , b21a02 + b11a01 + b01a00
    D00_D01 =  __PS_MADDS0( B20_B21, A02_A03, D00_D01); // m00, m01 computed

    // D02_D03 = b12a01 + b02a00 + b22a02 , b13a01 + b03a00 + b23a02
    D02_D03 =  __PS_MADDS0( B22_B23, A02_A03, D02_D03);

    // D10_D11 = a10b00 + a11b10 +a12b20, a10b01 + a11b11 + a12b21
    D10_D11 =  __PS_MADDS0( B20_B21, A12_A13, D10_D11); // m10, m11 computed

    // D12_D13 = a10b02 + a11b12 + a12b22, a10b03+a11b13 + a12b23 + a13
    D12_D13 =  __PS_MADDS0( B22_B23, A12_A13, D12_D13);

    // store m00m01
    __PSQ_ST(ab, D00_D01, 0, 0);

    // D20_D21 = a20b00, a20b01
    D20_D21 = __PS_MULS0( B00_B01, A20_A21);

    // get a03 from fp1 and add to D02_D03
    D02_D03 =  __PS_MADDS1( c01, A02_A03, D02_D03); // m02, m03 computed

    // D22_D23 = a20b02, a20b03
    D22_D23 = __PS_MULS0( B02_B03, A20_A21);

    // store m10m11
    __PSQ_STX(ab, 16, D10_D11, 0, 0);

    // get a13 from fp3 and add to D12_D13
    D12_D13 =  __PS_MADDS1( c01, A12_A13, D12_D13); // m12, m13 computed

    // store m02m03
    __PSQ_STX(ab, 8, D02_D03, 0, 0);

    // D20_D21 = a20b00 + a21b10, a20b01 + a21b11
    D20_D21 =  __PS_MADDS1( B10_B11, A20_A21, D20_D21);

    // D22_D23 = a20b02 + a21b12, a20b03 + a21b13
    D22_D23 =  __PS_MADDS1( B12_B13, A20_A21, D22_D23);

    // D20_D21 = a20b00 + a21b10 + a22b20, a20b01 + a21b11 + a22b21
    D20_D21 =  __PS_MADDS0( B20_B21, A22_A23, D20_D21);

    // store m12m13
    __PSQ_STX(ab, 24, D12_D13, 0, 0);

    // D22_D23 = a20b02 + a21b12 + a22b22, a20b03 + a21b13 + a22b23 + a23
    D22_D23 =  __PS_MADDS0( B22_B23, A22_A23, D22_D23);

    // store m20m21

    __PSQ_STX(ab, 32, D20_D21, 0, 0);

    // get a23 from fp5 and add to fp17
    D22_D23 =  __PS_MADDS1( c01, A22_A23, D22_D23);

    // store m22m23
    __PSQ_STX(ab, 40, D22_D23, 0, 0);

}
#endif

/*---------------------------------------------------------------------*

Name:           MTXConcatArray

Description:    concatenates a matrix to an array of matrices.
                order of operation is A x B(array) = AB(array).

Arguments:      a        first matrix for concat.
                srcBase  array base of second matrix for concat.
                dstBase  array base of resultant matrix from concat.
                count    number of matrices in srcBase, dstBase arrays.

                note:      cannot check for array overflow

Return:         none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXConcatArray ( MTX_CONST Mtx a, MTX_CONST Mtx* srcBase, Mtx* dstBase, u32 count )
{
    u32 i;

    ASSERTMSG( (a       != 0), "MTXConcatArray(): NULL MtxPtr 'a' " );
    ASSERTMSG( (srcBase != 0), "MTXConcatArray(): NULL MtxPtr 'srcBase' " );
    ASSERTMSG( (dstBase != 0), "MTXConcatArray(): NULL MtxPtr 'dstBase' " );
    ASSERTMSG( (count > 1),    "MTXConcatArray(): count must be greater than 1." );

    for ( i = 0 ; i < count ; i++ )
    {
        C_MTXConcat(a, *srcBase, *dstBase);

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
void PSMTXConcatArray (
    MTX_CONST Mtx  a,
    MTX_CONST Mtx* srcBase,
    Mtx* dstBase,
    u32  count )
{

    int i;

    for ( i = 0 ; i < count ; i++ )
    {
        PSMTXConcat(a, *srcBase, *dstBase);

        srcBase++;
        dstBase++;
    }
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXTranspose

Description:    computes the transpose of a matrix.
                As matrices are 3x4, fourth column (translation component) is
                lost and becomes (0,0,0).

                This function is intended for use in computing an
                ilwerse-transpose matrix to transform normals for lighting.
                In this case, lost translation component doesn't matter.

Arguments:      src       source matrix.
                xPose     destination (transposed) matrix.
                          ok if src == xPose.

Return:         none

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXTranspose ( MTX_CONST Mtx src, Mtx xPose )
{
    Mtx mTmp;
    MtxPtr m;

    ASSERTMSG( (src   != 0), MTX_TRANSPOSE_1  );
    ASSERTMSG( (xPose != 0), MTX_TRANSPOSE_2  );

    if(src == xPose)
    {
        m = mTmp;
    }
    else
    {
        m = xPose;
    }

    m[0][0] = src[0][0];   m[0][1] = src[1][0];      m[0][2] = src[2][0];     m[0][3] = 0.0f;
    m[1][0] = src[0][1];   m[1][1] = src[1][1];      m[1][2] = src[2][1];     m[1][3] = 0.0f;
    m[2][0] = src[0][2];   m[2][1] = src[1][2];      m[2][2] = src[2][2];     m[2][3] = 0.0f;

    // copy back if needed
    if( m == mTmp )
    {
        C_MTXCopy( *((MTX_CONST Mtx *)&mTmp), xPose );
    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXTranspose ( MTX_CONST Mtx src, Mtx xPose )
{
    f32x2 row0a, row1a, row0b, row1b;
    f32x2 trns0, trns1, trns2;

    //psq_l       row0a, 0(src),  0, 0    // [0][0], [0][1]
    row0a = __PSQ_L(src, 0, 0);

    //psq_l       row1a, 16(src), 0, 0    // [1][0], [1][1]
    row1a = __PSQ_LX(src, 16, 0, 0);

    //ps_merge00  trns0, row0a, row1a     // [0][0], [1][0]
    trns0 = __PS_MERGE00(row0a, row1a);

    //psq_l       row0b, 8(src),  1, 0    // [0][2], 1
    row0b = __PSQ_LX(src, 8, 1, 0);

    //ps_merge11  trns1, row0a, row1a     // [0][1], [1][1]
    trns1 = __PS_MERGE11(row0a, row1a);

    //psq_l       row1b, 24(src), 1, 0    // [1][2], 1
    row1b = __PSQ_LX(src, 24, 1, 0);

    //psq_st      trns0, 0(xPose),  0, 0  // [0][0], [1][0] -> [0][0], [0][1]
    __PSQ_ST(xPose, trns0, 0, 0);

    //psq_l       row0a, 32(src), 0, 0    // [2][0], [2][1]
    row0a = __PSQ_LX(src, 32, 0, 0);

    //ps_merge00  trns2, row0b, row1b     // [0][2], [1][2]
    trns2 = __PS_MERGE00(row0b, row1b);

    //psq_st      trns1, 16(xPose), 0, 0  // [0][1], [1][1] -> [1][0], [1][1]
    __PSQ_STX(xPose, 16, trns1, 0, 0);

    //ps_merge00  trns0, row0a, c00       // [2][0], 0
    trns0 = __PS_MERGE00(row0a, c00);

    //psq_st      trns2, 32(xPose), 0, 0  // [0][2], [1][2] -> [2][0], [2][1]
    __PSQ_STX(xPose, 32, trns2, 0, 0);

    //ps_merge10  trns1, row0a, c00       // [2][1], 0
    trns1 = __PS_MERGE10(row0a, c00);

    //psq_st      trns0, 8(xPose),  0, 0  // [2][0], 0 -> [0][2], [0][3]
    __PSQ_STX(xPose, 8, trns0, 0, 0);

    //lfs         row0b, 40(src)          // [2][2]
    row0b = __PSQ_LX(src, 40, 1, 0);

    //psq_st      trns1, 24(xPose), 0, 0  // [2][1], 0 -> [1][2], [1][3]
    __PSQ_STX(xPose, 24, trns1, 0, 0);

    //stfs        row0b, 40(xPose)        // [2][2] -> [2][2]
   __PSQ_STX(xPose, 40, row0b, 1, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXIlwerse

Description:    computes a fast ilwerse of a matrix.
                this algorithm works for matrices with a fourth row of
                (0,0,0,1).

                for a matrix
                M =  |     A      C      |  where A is the upper 3x3 submatrix,
                     |     0      1      |        C is a 1x3 column vector

                ILW(M)     =    |  ilw(A)      (ilw(A))*(-C)    |
                                |     0               1         |

Arguments:      src       source matrix.
                ilw       destination (ilwerse) matrix.
                          ok if src == ilw.

Return:         0 if src is not ilwertible.
                1 on success.

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
u32 C_MTXIlwerse ( MTX_CONST Mtx src, Mtx ilw )
{
    Mtx mTmp;
    MtxPtr m;
    f32 det;

    ASSERTMSG( (src != 0), MTX_ILWERSE_1 );
    ASSERTMSG( (ilw != 0), MTX_ILWERSE_2 );

    if( src == ilw )
    {
        m = mTmp;
    }
    else
    {
        m = ilw;
    }

    // compute the determinant of the upper 3x3 submatrix
    det =   src[0][0]*src[1][1]*src[2][2] + src[0][1]*src[1][2]*src[2][0] + src[0][2]*src[1][0]*src[2][1]
          - src[2][0]*src[1][1]*src[0][2] - src[1][0]*src[0][1]*src[2][2] - src[0][0]*src[2][1]*src[1][2];

    // check if matrix is singular
    if( det == 0.0f )
    {
        return 0;
    }

    // compute the ilwerse of the upper submatrix:

    // find the transposed matrix of cofactors of the upper submatrix
    // and multiply by (1/det)

    det = 1.0f / det;

    m[0][0] =  (src[1][1]*src[2][2] - src[2][1]*src[1][2]) * det;
    m[0][1] = -(src[0][1]*src[2][2] - src[2][1]*src[0][2]) * det;
    m[0][2] =  (src[0][1]*src[1][2] - src[1][1]*src[0][2]) * det;

    m[1][0] = -(src[1][0]*src[2][2] - src[2][0]*src[1][2]) * det;
    m[1][1] =  (src[0][0]*src[2][2] - src[2][0]*src[0][2]) * det;
    m[1][2] = -(src[0][0]*src[1][2] - src[1][0]*src[0][2]) * det;

    m[2][0] =  (src[1][0]*src[2][1] - src[2][0]*src[1][1]) * det;
    m[2][1] = -(src[0][0]*src[2][1] - src[2][0]*src[0][1]) * det;
    m[2][2] =  (src[0][0]*src[1][1] - src[1][0]*src[0][1]) * det;

    // compute (ilwA)*(-C)
    m[0][3] = -m[0][0]*src[0][3] - m[0][1]*src[1][3] - m[0][2]*src[2][3];
    m[1][3] = -m[1][0]*src[0][3] - m[1][1]*src[1][3] - m[1][2]*src[2][3];
    m[2][3] = -m[2][0]*src[0][3] - m[2][1]*src[1][3] - m[2][2]*src[2][3];

    // copy back if needed
    if( m == mTmp )
    {
        C_MTXCopy( *((MTX_CONST Mtx *)&mTmp),ilw );
    }

    return 1;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
            Results may be a little bit different from the C version
            because it doesn't perform exactly same callwlation.
 *---------------------------------------------------------------------*/
u32 PSMTXIlwerse ( MTX_CONST Mtx src, Mtx ilw )
{

    f32x2 fp0;
    f32x2 fp1;
    f32x2 fp2;
    f32x2 fp3;
    f32x2 fp4;
    f32x2 fp5;

    f32x2 fp6;
    f32x2 fp7;
    f32x2 fp8;
    f32x2 fp9;
    f32x2 fp10;
    f32x2 fp11;
    f32x2 fp12;
    f32x2 fp13;

    // fp0 [ 00 ][ 1.0F ] : Load
    fp0 = __PSQ_LX(src, 0, 1, 0);

    // fp1 [ 01 ][ 02 ]   : Load
    fp1 = __PSQ_LX(src, 4, 0, 0);

    // fp2 [ 10 ][ 1.0F ] : Load
    fp2 = __PSQ_LX(src, 16, 1, 0);

    // fp6 [ 02 ][ 00 ]
    fp6 = __PS_MERGE10(fp1, fp0);

    // fp3 [ 11 ][ 12 ]   : Load
    fp3 = __PSQ_LX(src, 20, 0, 0);

    // fp4 [ 20 ][ 1.0F ] : Load
    fp4 = __PSQ_LX(src, 32, 1, 0);

    // fp7 [ 12 ][ 10 ]
    fp7 = __PS_MERGE10(fp3, fp2);

    // fp5 [ 21 ][ 22 ]   : Load
    fp5 = __PSQ_LX(src, 36, 0, 0);

    // fp11[ 11*02 ][ 00*12 ]
    fp11 = __PS_MUL(fp3, fp6);

    // fp8 [ 22 ][ 20 ]
    fp8 = __PS_MERGE10(fp5, fp4);

    // fp13[ 21*12 ][ 10*22 ]
    fp13 = __PS_MUL(fp5, fp7);

    // fp11[ 01*12 - 11*02 ][ 10*02 - 00*12 ]
    fp11 = __PS_MSUB(fp1, fp7, fp11);

    // fp12[ 01*22 ][ 20*02 ]
    fp12 = __PS_MUL(fp1, fp8);

    // fp13[ 11*22 - 21*12 ][ 20*12 - 10*22 ]
    fp13 = __PS_MSUB(fp3, fp8, fp13);

    // fp10[ 20*11 ][ N/A ]
    fp10 = __PS_MUL(fp3, fp4);

    // fp12[ 21*02 - 01*22 ][ 00*22 - 20*02 ]
    fp12 = __PS_MSUB(fp5, fp6, fp12);

    // fp7 [ 00*(11*22-21*12) ][ N/A ]
    fp7  = __PS_MUL(fp0, fp13);

    // fp9 [ 00*21 ][ N/A ]
    fp9  = __PS_MUL(fp0, fp5);

    // fp8 [ 10*01 ][ N/A ]
    fp8  = __PS_MUL(fp1, fp2);

    // fp7 [ 00*(11*22-21*12) + 10*(21*02-01*22) ][ N/A ]
    fp7 = __PS_MADD(fp2, fp12, fp7);

    // fp6 [ 0.0F ][ 0.0F ]
    fp6 = __PS_SUB(fp6, fp6);

    // fp10[ 10*21 - 20*11 ][ N/A ]
    fp10 = __PS_MSUB(fp2, fp5, fp10);

    // fp7 [ 00*(11*22-21*12) + 10*(21*02-01*22) + 20*(01*12-11*02) ][ N/A ] : det
    fp7 = __PS_MADD(fp4, fp11, fp7);

    // fp9 [ 20*01 - 00*21 ][ N/A ]
    fp9 = __PS_MSUB(fp1, fp4, fp9);

    // fp8 [ 00*11 - 10*01 ][ N/A ]
    fp8 = __PS_MSUB(fp0, fp3, fp8);

    // check if matrix is singular
    if( fp7[0] == 0.0f && fp7[1] == 0.0f)
    {
        return 0;
    }

    // compute the ilwerse of the upper submatrix:

    // find the transposed matrix of cofactors of the upper submatrix
    // and multiply by (1/det)

    // fp0 [ 1/det ][ N/A ]
    fp0 = __PS_RES(fp7);

    // Newton's approximation
    // Refinement : ( E = est. of 1/K ) -> ( E' = ( 2 - K * E ) * E )
    fp6 = __PS_ADD(fp0, fp0);
    fp5 = __PS_MUL(fp7, fp0);
    fp0 = __PS_NMSUB(fp0, fp5, fp6);

    // fp1 [ 03 ][ 03 ] : Load
    fp1[0] = src[0][3];
    fp1[1] = src[0][3];

    // fp13[ ( 11*22 - 21*12 ) * rdet ][ ( 20*12 - 10*22 ) * rdet ] : i[0][0], i[1][0]
    fp13 = __PS_MULS0(fp13, fp0);

    // fp2 [ 13 ][ 13 ] : Load
    fp2[0] = src[1][3];
    fp2[1] = src[1][3];

    // fp12[ ( 21*02 - 01*22 ) * rdet ][ ( 00*22 - 20*02 ) * rdet ] : i[0][1], i[1][1]
    fp12 = __PS_MULS0(fp12, fp0);

    // fp3 [ 23 ][ 23 ] : Load
    fp3[0] = src[2][3];
    fp3[1] = src[2][3];

    // fp11[ ( 01*12 - 11*02 ) * rdet ][ ( 10*02 - 00*12 ) * rdet ] : i[0][2], i[1][2]
    fp11 = __PS_MULS0(fp11, fp0);

    // fp5 [ i00 ][ i01 ]
    fp5 = __PS_MERGE00(fp13, fp12);

    // fp4 [ i10 ][ i11 ]
    fp4 = __PS_MERGE11(fp13, fp12);

    // fp6 [ i00*03 ][ i10*03 ]
    fp6 = __PS_MUL(fp13, fp1);

    // [ i00 ][ i01 ] : Store fp5   -> free(fp5[ i00 ][ i01 ])
    //ilw[0][0] = fp5[0];
    //ilw[0][1] = fp5[1];
    __PSQ_STX(ilw, 0, fp5, 0, 0);

    // [ i10 ][ i11 ] : Store fp4   -> free(fp4[ i10 ][ i11 ])
    //ilw[1][0] = fp4[0];
    //ilw[1][1] = fp4[1];
    __PSQ_STX(ilw, 16, fp4, 0, 0);

    // fp10[ ( 10*21 - 20*11 ) * rdet ] : i[2][0]
    fp10  = __PS_MULS0(fp10, fp0);

    // fp9 [ ( 20*01 - 00*21 ) * rdet ] : i[2][1]
    fp9  = __PS_MULS0(fp9,  fp0);

    // fp6 [ i00*03+i01*13 ][ i10*03+i11*13 ]
    fp6 = __PS_MADD(fp12, fp2, fp6);

    // [ i20 ] : Store fp10
    //ilw[2][0] = fp10[0];
    __PSQ_STX(ilw, 32, fp10, 1, 0);

    // fp8 [ ( 00*11 - 10*01 ) * rdet ] : i[2][2]
    fp8 = __PS_MULS0(fp8,  fp0);

    // fp6 [ -i00*03-i01*13-i02*23 ][ -i10*03-i11*13-i12*23 ] : i[0][3], i[1][3]
    fp6 = __PS_NMADD(fp11, fp3, fp6);

    // [ i21 ] : Store fp9
    //ilw[2][1] = fp9[0];
    __PSQ_STX(ilw, 36, fp9, 1, 0);

    // fp7 [ i20*03 ][ N/A ]
    fp7 = __PS_MUL(fp10, fp1);

    // fp5 [ i02 ][ i03 ]
    fp5 = __PS_MERGE00(fp11, fp6);

    // [ i22 ] : Store fp8
    //ilw[2][2] = fp8[0];
    __PSQ_STX(ilw, 40, fp8, 1, 0);

    // fp7 [ i20*03+i21*13 ][ N/A ]
    fp7  = __PS_MADD(fp9,  fp2, fp7);

    // fp4 [ i12 ][ i13 ]
    fp4  = __PS_MERGE11(fp11, fp6);

    // [ i02 ][ i03 ] : Store fp5
    //ilw[0][2] = fp5[0];
    //ilw[0][3] = fp5[1];
    __PSQ_STX(ilw, 8, fp5, 0, 0);

    // fp7 [ -i20*03-i21*13-i22*23 ][ N/A ] : i[2][3]
    fp7 = __PS_NMADD(fp8,  fp3, fp7);

    // [ i12 ][ i13 ] : Store fp4
    //ilw[1][2] = fp4[0];
    //ilw[1][3] = fp4[1];
    __PSQ_STX(ilw, 24, fp4, 0, 0);

    // [ i23 ] : Store fp7
    //ilw[2][3] = fp7[0];
    __PSQ_STX(ilw, 44, fp7, 1, 0);

    return 1;
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXIlwXpose

Description:    computes a fast ilwerse-transpose of a matrix.
                this algorithm works for matrices with a fourth row of
                (0,0,0,1). Commonly used for callwlating normal transform
                matrices.

                This function is equivalent to the combination of
                two functions MTXIlwerse + MTXTranspose.

Arguments:      src       source matrix.
                ilwx      destination (ilwerse-transpose) matrix.
                          ok if src == ilwx.

Return:         0 if src is not ilwertible.
                1 on success.

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
u32 C_MTXIlwXpose ( MTX_CONST Mtx src, Mtx ilwX )
{
    Mtx mTmp;
    MtxPtr m;
    f32 det;

    ASSERTMSG( (src != 0), MTX_ILWXPOSE_1 );
    ASSERTMSG( (ilwX != 0), MTX_ILWXPOSE_2 );

    if( src == ilwX )
    {
        m = mTmp;
    }
    else
    {
        m = ilwX;
    }

    // compute the determinant of the upper 3x3 submatrix
    det =   src[0][0]*src[1][1]*src[2][2] + src[0][1]*src[1][2]*src[2][0] + src[0][2]*src[1][0]*src[2][1]
          - src[2][0]*src[1][1]*src[0][2] - src[1][0]*src[0][1]*src[2][2] - src[0][0]*src[2][1]*src[1][2];

    // check if matrix is singular
    if( det == 0.0f )
    {
        return 0;
    }

    // compute the ilwerse-transpose of the upper submatrix:

    // find the transposed matrix of cofactors of the upper submatrix
    // and multiply by (1/det)

    det = 1.0f / det;

    m[0][0] =  (src[1][1]*src[2][2] - src[2][1]*src[1][2]) * det;
    m[0][1] = -(src[1][0]*src[2][2] - src[2][0]*src[1][2]) * det;
    m[0][2] =  (src[1][0]*src[2][1] - src[2][0]*src[1][1]) * det;

    m[1][0] = -(src[0][1]*src[2][2] - src[2][1]*src[0][2]) * det;
    m[1][1] =  (src[0][0]*src[2][2] - src[2][0]*src[0][2]) * det;
    m[1][2] = -(src[0][0]*src[2][1] - src[2][0]*src[0][1]) * det;

    m[2][0] =  (src[0][1]*src[1][2] - src[1][1]*src[0][2]) * det;
    m[2][1] = -(src[0][0]*src[1][2] - src[1][0]*src[0][2]) * det;
    m[2][2] =  (src[0][0]*src[1][1] - src[1][0]*src[0][1]) * det;

    // the fourth columns should be all zero
    m[0][3] = 0.0F;
    m[1][3] = 0.0F;
    m[2][3] = 0.0F;

    // copy back if needed
    if( m == mTmp )
    {
        C_MTXCopy( *((MTX_CONST Mtx *)&mTmp),ilwX );
    }

    return 1;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
            Results may be a little bit different from the C version
            because it doesn't perform exactly same callwlation.
 *---------------------------------------------------------------------*/
u32 PSMTXIlwXpose ( MTX_CONST Mtx src, Mtx ilwX )
{
    f32x2 fp0;
    f32x2 fp1;
    f32x2 fp2;
    f32x2 fp3;
    f32x2 fp4;
    f32x2 fp5;

    f32x2 fp6;
    f32x2 fp7;
    f32x2 fp8;
    f32x2 fp9;
    f32x2 fp10;
    f32x2 fp11;
    f32x2 fp12;
    f32x2 fp13;

    // fp0 [ 00 ][ 1.0F ] : Load
    //fp0[0] = src[0][0];
    //fp0[1] = 1.0F;
    fp0 = __PSQ_LX(src, 0, 1, 0);

    // fp1 [ 01 ][ 02 ]   : Load
    //fp1[0] = src[0][1];
    //fp1[1] = src[0][2];
    fp1 = __PSQ_LX(src, 4, 0, 0);

    // fp2 [ 10 ][ 1.0F ] : Load
    //fp2[0] = src[1][0];
    //fp2[1] = 1.0F;
    fp2 = __PSQ_LX(src, 16, 1, 0);

    // fp6 [ 02 ][ 00 ]
    fp6 = __PS_MERGE10(fp1, fp0);

    // fp3 [ 11 ][ 12 ]   : Load
    //fp3[0] = src[1][1];
    //fp3[1] = src[1][2];
    fp3 = __PSQ_LX(src, 20, 0, 0);

    // fp4 [ 20 ][ 1.0F ] : Load
    //fp4[0] = src[2][0];
    //fp4[1] = 1.0F;
    fp4 = __PSQ_LX(src, 32, 1, 0);

    // fp7 [ 12 ][ 10 ]
    fp7 = __PS_MERGE10(fp3, fp2);

    // fp5 [ 21 ][ 22 ]   : Load
    //fp5[0] = src[2][1];
    //fp5[1] = src[2][2];
    fp5 = __PSQ_LX(src, 36, 0, 0);

    // fp11[ 11*02 ][ 00*12 ]
    fp11 = __PS_MUL(fp3, fp6);

    // fp8 [ 22 ][ 20 ]
    fp8 = __PS_MERGE10(fp5, fp4);

    // fp13[ 21*12 ][ 10*22 ]
    fp13 = __PS_MUL(fp5, fp7);

    // fp11[ 01*12 - 11*02 ][ 10*02 - 00*12 ]
    fp11 = __PS_MSUB(fp1, fp7, fp11);

    // fp12[ 01*22 ][ 20*02 ]
    fp12 = __PS_MUL(fp1, fp8);

    // fp13[ 11*22 - 21*12 ][ 20*12 - 10*22 ]
    fp13 = __PS_MSUB(fp3, fp8, fp13);

    // fp10[ 20*11 ][ N/A ]
    fp10 = __PS_MUL(fp3, fp4);

    // fp12[ 21*02 - 01*22 ][ 00*22 - 20*02 ]
    fp12 = __PS_MSUB(fp5, fp6, fp12);

    // fp7 [ 00*(11*22-21*12) ][ N/A ]
    fp7  = __PS_MUL(fp0, fp13);

    // fp9 [ 00*21 ][ N/A ]
    fp9  = __PS_MUL(fp0, fp5);

    // fp8 [ 10*01 ][ N/A ]
    fp8  = __PS_MUL(fp1, fp2);

    // fp7 [ 00*(11*22-21*12) + 10*(21*02-01*22) ][ N/A ]
    fp7 = __PS_MADD(fp2, fp12, fp7);

    // fp6 [ 0.0F ][ 0.0F ]
    fp6 = __PS_SUB(fp6, fp6);

    // fp10[ 10*21 - 20*11 ][ N/A ]
    fp10 = __PS_MSUB(fp2, fp5, fp10);

    // fp7 [ 00*(11*22-21*12) + 10*(21*02-01*22) + 20*(01*12-11*02) ][ N/A ] : det
    fp7 = __PS_MADD(fp4, fp11, fp7);

    // fp9 [ 20*01 - 00*21 ][ N/A ]
    fp9 = __PS_MSUB(fp1, fp4, fp9);

    // fp8 [ 00*11 - 10*01 ][ N/A ]
    fp8 = __PS_MSUB(fp0, fp3, fp8);

    // check if matrix is singular
    if( fp7[0] == 0.0f && fp7[1] == 0.0f)
    {
        return 0;
    }

    // compute the ilwerse-transpose of the upper submatrix:

    // find the transposed matrix of cofactors of the upper submatrix
    // and multiply by (1/det)

    // fp0 [ 1/det ][ N/A ]
    fp0 = __PS_RES(fp7);

    // [ ix03 ] : Store fp6
    ilwX[0][3] = fp6[0];

    // Newton's approximation
    // Refinement : ( E = est. of 1/K ) -> ( E' = ( 2 - K * E ) * E )
    fp4 = __PS_ADD(fp0, fp0);
    fp5 = __PS_MUL(fp7, fp0);

    // [ ix13 ] : Store fp6
    //ilwX[1][3] = fp6[0];
    __PSQ_STX(ilwX, 28, fp6, 1, 0);

    fp0 = __PS_NMSUB(fp0, fp5, fp4);

    // [ ix23 ] : Store fp6
    //ilwX[2][3] = fp6[0];
    __PSQ_STX(ilwX, 44, fp6, 1, 0);

    // fp13[ ( 11*22 - 21*12 ) * rdet ][ ( 20*12 - 10*22 ) * rdet ] : ix[0][0], ix[0][1]
    fp13 = __PS_MULS0(fp13, fp0);

    // fp12[ ( 21*02 - 01*22 ) * rdet ][ ( 00*22 - 20*02 ) * rdet ] : ix[1][0], ix[1][1]
    fp12 = __PS_MULS0(fp12, fp0);

    // [ ix00 ][ ix01 ] : Store fp13
    //ilwX[0][0] = fp13[0];
    //ilwX[0][1] = fp13[1];
    __PSQ_STX(ilwX, 0, fp13, 0, 0);

    // fp11[ ( 01*12 - 11*02 ) * rdet ][ ( 10*02 - 00*12 ) * rdet ] : ix[2][0], ix[2][1]
    fp11 = __PS_MULS0(fp11, fp0);

    // [ ix10 ][ ix11 ] : Store fp12
    //ilwX[1][0] = fp12[0];
    //ilwX[1][1] = fp12[1];
    __PSQ_STX(ilwX, 16, fp12, 0, 0);

    // fp10[ ( 10*21 - 20*11 ) * rdet ] : i[0][2]
    fp10 = __PS_MULS0(fp10, fp0);

    // [ ix20 ][ ix21 ] : Store fp11
    //ilwX[2][0] = fp11[0];
    //ilwX[2][1] = fp11[1];
    __PSQ_STX(ilwX, 32, fp11, 0, 0);

    // fp9 [ ( 20*01 - 00*21 ) * rdet ] : i[1][2]
    fp9 = __PS_MULS0(fp9, fp0);

    // [ ix02 ]         : Store fp10
    //ilwX[0][2] = fp10[0];
    __PSQ_STX(ilwX, 8, fp10, 1, 0);

    // fp8 [ ( 00*11 - 10*01 ) * rdet ] : i[2][2]
    fp8 = __PS_MULS0(fp8, fp0);

    // [ ix12 ]         : Store fp9
    //ilwX[1][2] = fp9[0];
    __PSQ_STX(ilwX, 24, fp9, 1, 0);

    // [ ix22 ]         : Store fp8
    //ilwX[2][2] = fp8[0];
    __PSQ_STX(ilwX, 40, fp8, 1, 0);

    return 1;
}
#endif

/*---------------------------------------------------------------------*


                             MODEL SECTION


*---------------------------------------------------------------------*/

/*---------------------------------------------------------------------*

Name:           MTXRotDeg

Description:    sets a rotation matrix about one of the X, Y or Z axes

Arguments:      m       matrix to be set

                axis    major axis about which to rotate.
                        axis is passed in as a character.
                        it must be one of 'X', 'x', 'Y', 'y', 'Z', 'z'

                deg     rotation angle in degrees.

                        note:  counter-clockwise rotation is positive.

Return:         none

*---------------------------------------------------------------------*/

/*---------------------------------------------------------------------*

Name:           MTXRotRad

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
void C_MTXRotRad ( Mtx m, char axis, f32 rad )
{

    f32 sinA, cosA;

    ASSERTMSG( (m != 0), MTX_ROTRAD_1 );

    // verification of "axis" will occur in MTXRotTrig

    sinA = sinf(rad);
    cosA = cosf(rad);

    C_MTXRotTrig( m, axis, sinA, cosA );
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXRotRad ( Mtx m, char axis, f32 rad )
{
    f32 sinA, cosA;

    sinA = sinf(rad);
    cosA = cosf(rad);

    PSMTXRotTrig( m, axis, sinA, cosA );
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXRotTrig

Description:    sets a rotation matrix about one of the X, Y or Z axes
                from specified trig ratios

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
void C_MTXRotTrig ( Mtx m, char axis, f32 sinA, f32 cosA )
{
    ASSERTMSG( (m != 0), MTX_ROTTRIG_1 );

    switch(axis)
    {

    case 'x':
    case 'X':
        m[0][0] =  1.0f;  m[0][1] =  0.0f;    m[0][2] =  0.0f;  m[0][3] = 0.0f;
        m[1][0] =  0.0f;  m[1][1] =  cosA;    m[1][2] = -sinA;  m[1][3] = 0.0f;
        m[2][0] =  0.0f;  m[2][1] =  sinA;    m[2][2] =  cosA;  m[2][3] = 0.0f;
        break;

    case 'y':
    case 'Y':
        m[0][0] =  cosA;  m[0][1] =  0.0f;    m[0][2] =  sinA;  m[0][3] = 0.0f;
        m[1][0] =  0.0f;  m[1][1] =  1.0f;    m[1][2] =  0.0f;  m[1][3] = 0.0f;
        m[2][0] = -sinA;  m[2][1] =  0.0f;    m[2][2] =  cosA;  m[2][3] = 0.0f;
        break;

    case 'z':
    case 'Z':
        m[0][0] =  cosA;  m[0][1] = -sinA;    m[0][2] =  0.0f;  m[0][3] = 0.0f;
        m[1][0] =  sinA;  m[1][1] =  cosA;    m[1][2] =  0.0f;  m[1][3] = 0.0f;
        m[2][0] =  0.0f;  m[2][1] =  0.0f;    m[2][2] =  1.0f;  m[2][3] = 0.0f;
        break;

    default:
        ASSERTMSG( 0, MTX_ROTTRIG_2 );
        break;

    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXRotTrig ( Mtx m, char axis, f32 sinA, f32 cosA )
{
    f32x2 nsinA;
    f32x2 fw0, fw1, fw2, fw3;
    f32x2 sinA10 = {sinA, 0.0f};
    f32x2 cosA10 = {cosA, 0.0f};

    //ps_neg      nsinA, sinA
    nsinA = __PS_NEG(sinA10);

    switch(axis)
    {
    case 'x':
    case 'X':
        //psq_st      fc1,  0(m), 1, 0
        __PSQ_ST(m, c11, 1, 0);

        //psq_st      fc0,  4(m), 0, 0
        __PSQ_STX(m, 4, c00, 0, 0);

        //ps_merge00  fw0, sinA, cosA
        fw0 = __PS_MERGE00(sinA10, cosA10);

        //psq_st      fc0, 12(m), 0, 0
        __PSQ_STX(m, 12, c00, 0, 0);

        //ps_merge00  fw1, cosA, nsinA
        fw1 = __PS_MERGE00(cosA10, nsinA);

        //psq_st      fc0, 28(m), 0, 0
        __PSQ_STX(m, 28, c00, 0, 0);

        //psq_st      fc0, 44(m), 1, 0
        __PSQ_STX(m, 44, c00, 1, 0);

        //psq_st      fw0, 36(m), 0, 0
        __PSQ_STX(m, 36, fw0, 0, 0);

        //psq_st      fw1, 20(m), 0, 0
        __PSQ_STX(m, 20, fw1, 0, 0);

        break;

    case 'y':
    case 'Y':
        //ps_merge00  fw0, cosA, fc0
        fw0 = __PS_MERGE00(cosA10, c00);

        //ps_merge00  fw1, fc0, fc1
        fw1 = __PS_MERGE00(c00, c11);

        //psq_st      fc0, 24(m), 0, 0
        __PSQ_STX(m, 24, c00, 0, 0);

        //psq_st      fw0,  0(m), 0, 0
        __PSQ_ST(m, fw0, 0, 0);

        //ps_merge00  fw2, nsinA, fc0
        fw2 = __PS_MERGE00(nsinA, c00);

        //ps_merge00  fw3, sinA, fc0
        fw3 = __PS_MERGE00(sinA10, c00);

        //psq_st      fw0, 40(m), 0, 0;
        __PSQ_STX(m, 40, fw0, 0, 0);

        //psq_st      fw1, 16(m), 0, 0;
        __PSQ_STX(m, 16, fw1, 0, 0);

        //psq_st      fw3,  8(m), 0, 0;
        __PSQ_STX(m, 8, fw3, 0, 0);

        //psq_st      fw2, 32(m), 0, 0;
        __PSQ_STX(m, 32, fw2, 0, 0);

        break;

    case 'z':
    case 'Z':

        //psq_st      fc0,  8(m), 0, 0
        __PSQ_STX(m, 8, c00, 0, 0);

        //ps_merge00  fw0, sinA, cosA
        fw0 = __PS_MERGE00(sinA10, cosA10);

        //ps_merge00  fw2, cosA, nsinA
        fw2 = __PS_MERGE00(cosA10, nsinA);

        //psq_st      fc0, 24(m), 0, 0
        __PSQ_STX(m, 24, c00, 0, 0);

        //psq_st      fc0, 32(m), 0, 0
        __PSQ_STX(m, 32, c00, 0, 0);

        //ps_merge00  fw1, fc1, fc0
        fw1 = __PS_MERGE00(c11, c00);

        //psq_st      fw0, 16(m), 0, 0
        __PSQ_STX(m, 16, fw0, 0, 0);

        //psq_st      fw2,  0(m), 0, 0
        __PSQ_ST(m, fw2, 0, 0);

        //psq_st      fw1, 40(m), 0, 0
        __PSQ_STX(m, 40, fw1, 0, 0);

        break;

    default:
        ASSERTMSG( 0, MTX_ROTTRIG_2 );
        break;
    }
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXRotAxisRad

Description:    sets a rotation matrix about an arbitrary axis


Arguments:      m       matrix to be set

                axis    ptr to a vector containing the x,y,z axis
                        components.
                        axis does not have to be a unit vector.

                deg     rotation angle in radians.

                        note:  counter-clockwise rotation is positive.

Return:         none

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXRotAxisRad( Mtx m, const Vec *axis, f32 rad )
{
    Vec vN;
    f32 s, c;             // sinTheta, cosTheta
    f32 t;                // ( 1 - cosTheta )
    f32 x, y, z;          // x, y, z components of normalized axis
    f32 xSq, ySq, zSq;    // x, y, z squared

    ASSERTMSG( (m    != 0), MTX_ROTAXIS_1  );
    ASSERTMSG( (axis != 0), MTX_ROTAXIS_2  );

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
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
static void _PSMTXRotAxisRadInternal(
    Mtx    m,
    const Vec *axis,
    f32    sT,
    f32    cT )
{
    f32x2    tT, sT2, cT2;
    f32x2    tmp0, tmp1, tmp2, tmp3, tmp4;
    f32x2    tmp5, tmp6, tmp7, tmp9, tmp8;

    // tmp0 = [x][y] : LOAD
    //psq_l       tmp0, 0(axis), 0, 0
    //tmp0[0] = axis->x;
    //tmp0[1] = axis->y;
    tmp0 = __PSQ_L(axis, 0, 0);

    // tmp1 = [z][z] : LOAD
    tmp1[0] = axis->z;
    tmp1[1] = axis->z;

    // tmp2 = [x*x][y*y]
    tmp2 = __PS_MUL(tmp0, tmp0);

    // tmp3 = [x*x+z*z][y*y+z*z]
    tmp3 = __PS_MADD(tmp1, tmp1, tmp2);

    // tmp4 = [S = x*x+y*y+z*z][z]
    tmp4 = __PS_SUM0(tmp3, tmp1, tmp2);

    // tT = 1.0F - cT
    tT[0] = tT[1] = 1.0f - cT;

    // tmp5 = [1.0/sqrt(S)] :estimation[E]
    tmp5[0] = tmp5[1] = __FRSQRTE(tmp4[0]);

    // Newton-Rapson refinement step
    // E' = E/2(3.0 - E*E*S)
    tmp2 = __PS_MUL(tmp5, tmp5);            // E*E
    tmp3 = __PS_MUL(tmp5, c0505);            // E/2
    tmp2 = __PS_NMSUB(tmp2, tmp4, c33);    // (3-E*E*S)
    tmp5 = __PS_MUL(tmp2, tmp3);            // (E/2)(3-E*E*S)

    // cT = [c][c]
    cT2[0] = cT2[1] = cT;

    // sT = [c][c]
    sT2[0] = sT2[1] = sT;

    // tmp0 = [nx = x/sqrt(S)][ny = y/sqrt(S)]
    tmp0 = __PS_MULS0(tmp0, tmp5);

    // tmp1 = [nz = z/sqrt(S)][nz = z/sqrt(S)]
    tmp1 = __PS_MULS0(tmp1, tmp5);

    // tmp4 = [t*nx][t*ny]
    tmp4 = __PS_MULS0(tmp0, tT);

    // tmp9 = [s*nx][s*ny]
    tmp9 = __PS_MULS0(tmp0, sT2);

    // tmp5 = [t*nz][t*nz]
    tmp5  = __PS_MULS0(tmp1, tT);

    // tmp3 = [t*nx*ny][t*ny*ny]
    tmp3  = __PS_MULS1(tmp4, tmp0);

    // tmp2 = [t*nx*nx][t*ny*nx]
    tmp2 = __PS_MULS0(tmp4, tmp0);

    // tmp4 = [t*nx*nz][t*ny*nz]
    tmp4 = __PS_MULS0(tmp4, tmp1);

    // tmp6 = [t*nx*nx-s*nz][t*ny*ny-s*nz]
    tmp6 = __PS_NMSUB(tmp1, sT2, tmp2);

    // tmp7 = [t*nx*ny+s*nz][t*ny*ny+s*nz]
    tmp7 = __PS_MADD(tmp1, sT2, tmp3);

    // tmp0 = [-s*nx][-s*ny]
    tmp0 = __PS_NEG(tmp9);

    // tmp8 = [t*nx*nz+s*ny][0] == [m02][m03]
    tmp8 = __PS_SUM0(tmp4, c00, tmp9);

    // tmp2 = [t*nx*nx+c][t*nx*ny-s*nz] == [m00][m01]
    tmp2  = __PS_SUM0(tmp2, tmp6, cT2);

    // tmp3 = [t*nx*ny+s*nz][t*ny*ny+c] == [m10][m11]
    tmp3 = __PS_SUM1(cT2, tmp7, tmp3);

    // tmp6 = [t*ny*nz-s*nx][0] == [m12][m13]
    tmp6 = __PS_SUM0(tmp0, c00 ,tmp4);

    // tmp8 [m02][m03] : STORE
    //psq_st      tmp8, 8(m), 0, 0
    //m[0][2] = tmp8[0];
    //m[0][3] = tmp8[1];
    __PSQ_STX(m, 8, tmp8, 0, 0);

    // tmp0 = [t*nx*nz-s*ny][t*ny*nz]
    tmp0 = __PS_SUM0(tmp4, tmp4, tmp0);

    // tmp2 [m00][m01] : STORE
    //psq_st      tmp2, 0(m), 0, 0
    //m[0][0] = tmp2[0];
    //m[0][1] = tmp2[1];
    __PSQ_STX(m, 0, tmp2, 0, 0);

    // tmp5 = [t*nz*nz][t*nz*nz]
    tmp5 = __PS_MULS0(tmp5, tmp1);

    // tmp3 [m10][m11] : STORE
    //psq_st      tmp3, 16(m), 0, 0
    //m[1][0] = tmp3[0];
    //m[1][1] = tmp3[1];
    __PSQ_STX(m, 16, tmp3, 0, 0);

    // tmp4 = [t*nx*nz-s*ny][t*ny*nz+s*nx] == [m20][m21]
    tmp4 = __PS_SUM1(tmp9, tmp0, tmp4);

    // tmp6 [m12][m13] : STORE
    //psq_st      tmp6, 24(m), 0, 0
    //m[1][2] = tmp6[0];
    //m[1][3] = tmp6[1];
    __PSQ_STX(m, 24, tmp6, 0, 0);

    // tmp5 = [t*nz*nz+c][0]   == [m22][m23]
    tmp5  = __PS_SUM0(tmp5, c00, cT2);

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

/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXRotAxisRad(
    Mtx             m,
    const Vec      *axis,
    f32             rad )
{
    f32     sinT, cosT;

    sinT = sinf(rad);
    cosT = cosf(rad);

    _PSMTXRotAxisRadInternal(m, axis, sinT, cosT);
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXTrans

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
void C_MTXTrans ( Mtx m, f32 xT, f32 yT, f32 zT )
{
    ASSERTMSG( (m != 0), MTX_TRANS_1 );

    m[0][0] = 1.0f;  m[0][1] = 0.0f;  m[0][2] = 0.0f;  m[0][3] =  xT;
    m[1][0] = 0.0f;  m[1][1] = 1.0f;  m[1][2] = 0.0f;  m[1][3] =  yT;
    m[2][0] = 0.0f;  m[2][1] = 0.0f;  m[2][2] = 1.0f;  m[2][3] =  zT;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXTrans( Mtx m, f32 xT, f32 yT, f32 zT )
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
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXTransApply

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
void C_MTXTransApply ( MTX_CONST Mtx src, Mtx dst, f32 xT, f32 yT, f32 zT )
{
    ASSERTMSG( (src != 0), MTX_TRANSAPPLY_1 );
    ASSERTMSG( (dst != 0), MTX_TRANSAPPLY_1 );

    if ( src != dst )
    {
        dst[0][0] = src[0][0];    dst[0][1] = src[0][1];    dst[0][2] = src[0][2];
        dst[1][0] = src[1][0];    dst[1][1] = src[1][1];    dst[1][2] = src[1][2];
        dst[2][0] = src[2][0];    dst[2][1] = src[2][1];    dst[2][2] = src[2][2];
    }

    dst[0][3] = src[0][3] + xT;
    dst[1][3] = src[1][3] + yT;
    dst[2][3] = src[2][3] + zT;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXTransApply( Mtx src, Mtx dst, f32 xT, f32 yT, f32 zT )
{
    f32x2 fp4, fp5, fp6, fp7, fp8, fp9;
    f32x2 xT10 = {xT, 0.0f};
    f32x2 yT10 = {yT, 0.0f};
    f32x2 zT10 = {zT, 0.0f};

    //psq_l       fp4, 0(src),        0, 0;
    fp4 = __PSQ_L(src, 0, 0);

    //frsp        xT, xT;                     // to make sure xT = single precision
    //psq_l       fp5, 8(src),        0, 0;
    fp5 = __PSQ_LX(src, 8, 0, 0);

    //frsp        yT, yT;                     // to make sure yT = single precision
    //psq_l       fp7, 24(src),       0, 0;
    fp7 = __PSQ_LX(src, 24, 0, 0);

    //frsp        zT, zT;                     // to make sure zT = single precision
    //psq_l       fp8, 40(src),       0, 0;
    fp8 = __PSQ_LX(src, 40, 0, 0);

    //psq_st      fp4, 0(dst),        0, 0;
    __PSQ_ST(dst, fp4, 0, 0);

    //ps_sum1     fp5, xT, fp5, fp5;
    fp5 = __PS_SUM1(xT10, fp5, fp5);

    //psq_l       fp6, 16(src),       0, 0;
    fp6 = __PSQ_LX(src, 16, 0, 0);

    //psq_st      fp5, 8(dst),        0, 0;
    __PSQ_STX(dst, 8, fp5, 0, 0);

    //ps_sum1     fp7, yT, fp7, fp7;
    fp7 = __PS_SUM1(yT10, fp7, fp7);

    //psq_l       fp9, 32(src),       0, 0;
    fp9 = __PSQ_LX(src, 32, 0, 0);

    //psq_st      fp6, 16(dst),       0, 0;
    __PSQ_STX(dst, 16, fp6, 0, 0);

    //ps_sum1     fp8, zT, fp8, fp8;
    fp8 = __PS_SUM1(zT10, fp8, fp8);

    //psq_st      fp7, 24(dst),       0, 0;
    __PSQ_STX(dst, 24, fp7, 0, 0);

    //psq_st      fp9, 32(dst),       0, 0;
    __PSQ_STX(dst, 32, fp9, 0, 0);

    //psq_st      fp8, 40(dst),       0, 0;
    __PSQ_STX(dst, 40, fp8, 0, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:            MTXScale

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
void C_MTXScale ( Mtx m, f32 xS, f32 yS, f32 zS )
{
    ASSERTMSG( (m != 0), MTX_SCALE_1 );


    m[0][0] = xS;    m[0][1] = 0.0f;  m[0][2] = 0.0f;  m[0][3] = 0.0f;
    m[1][0] = 0.0f;  m[1][1] = yS;    m[1][2] = 0.0f;  m[1][3] = 0.0f;
    m[2][0] = 0.0f;  m[2][1] = 0.0f;  m[2][2] = zS;    m[2][3] = 0.0f;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXScale( Mtx m, f32 xS, f32 yS, f32 zS )
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
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXScaleApply

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
void C_MTXScaleApply ( MTX_CONST Mtx src, Mtx dst, f32 xS, f32 yS, f32 zS )
{
    ASSERTMSG( (src != 0), MTX_SCALEAPPLY_1 );
    ASSERTMSG( (dst != 0), MTX_SCALEAPPLY_2 );

    dst[0][0] = src[0][0] * xS;     dst[0][1] = src[0][1] * xS;
    dst[0][2] = src[0][2] * xS;     dst[0][3] = src[0][3] * xS;

    dst[1][0] = src[1][0] * yS;     dst[1][1] = src[1][1] * yS;
    dst[1][2] = src[1][2] * yS;     dst[1][3] = src[1][3] * yS;

    dst[2][0] = src[2][0] * zS;     dst[2][1] = src[2][1] * zS;
    dst[2][2] = src[2][2] * zS;     dst[2][3] = src[2][3] * zS;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/

void PSMTXScaleApply ( MTX_CONST Mtx src, Mtx dst, f32 xS, f32 yS, f32 zS )
{
    //f32x2 fp0;
    //f32x2 fp1;
    f32x2 fp2;
    //f32x2 fp3;
    f32x2 fp4;
    f32x2 fp5;

    f32x2 fp6;
    f32x2 fp7;
    f32x2 fp8;
    //f32x2 fp9;
    //f32x2 fp10;
    //f32x2 fp11;

    f32x2 xS2 = {xS, xS};
    f32x2 yS2 = {yS, yS};
    f32x2 zS2 = {zS, zS};

    //psq_l       fp4, 0(src),        0, 0;
    fp4 = __PSQ_LX(src, 0, 0, 0);

    //psq_l       fp5, 8(src),        0, 0;
    fp5 = __PSQ_LX(src, 8, 0, 0);

    //ps_muls0    fp4, fp4, xS;
    fp4 = __PS_MUL(fp4, xS2);

    //psq_l       fp6, 16(src),       0, 0;
    fp6 = __PSQ_LX(src, 16, 0, 0);

    //ps_muls0    fp5, fp5, xS;
    fp5 = __PS_MUL(fp5, xS2);

    //psq_l       fp7, 24(src),       0, 0;
    fp7 = __PSQ_LX(src, 24, 0, 0);

    //ps_muls0    fp6, fp6, yS;
    fp6 = __PS_MUL(fp6, yS2);

    //psq_l       fp8, 32(src),       0, 0;
    fp8 = __PSQ_LX(src, 32, 0, 0);

    //psq_st      fp4, 0(dst),        0, 0;
    __PSQ_STX(dst, 0, fp4, 0, 0);

    //ps_muls0    fp7, fp7, yS;
    fp7 = __PS_MUL(fp7, yS2);

    //psq_l       fp2, 40(src),       0, 0;
    fp2 = __PSQ_LX(src, 40, 0, 0);

    //psq_st      fp5, 8(dst),        0, 0;
    __PSQ_STX(dst, 8, fp5, 0, 0);

    //ps_muls0    fp8, fp8, zS;
    fp8 = __PS_MUL(fp8, zS2);

    //psq_st      fp6, 16(dst),       0, 0;
    __PSQ_STX(dst, 16, fp6, 0, 0);

    //ps_muls0    fp2, fp2, zS;
    fp2 = __PS_MUL(fp2, zS2);

    //psq_st      fp7, 24(dst),       0, 0;
    __PSQ_STX(dst, 24, fp7, 0, 0);

    //psq_st      fp8, 32(dst),       0, 0;
    __PSQ_STX(dst, 32, fp8, 0, 0);

    //psq_st      fp2, 40(dst),       0, 0;
    __PSQ_STX(dst, 40, fp2, 0, 0);

}
#endif

/*---------------------------------------------------------------------*

Name:           MTXReflect

Description:    reflect a rotation matrix with respect to a plane.

Arguments:      m        matrix to be set

                p        point on the planar reflector.

                n       normal of the planar reflector.

Return:         none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXReflect ( Mtx m, const Vec *p, const Vec *n )
{
    f32 vxy, vxz, vyz, pdotn;

    vxy   = -2.0f * n->x * n->y;
    vxz   = -2.0f * n->x * n->z;
    vyz   = -2.0f * n->y * n->z;
    pdotn = 2.0f * C_VECDotProduct(p, n);

    m[0][0] = 1.0f - 2.0f * n->x * n->x;
    m[0][1] = vxy;
    m[0][2] = vxz;
    m[0][3] = pdotn * n->x;

    m[1][0] = vxy;
    m[1][1] = 1.0f - 2.0f * n->y * n->y;
    m[1][2] = vyz;
    m[1][3] = pdotn * n->y;

    m[2][0] = vxz;
    m[2][1] = vyz;
    m[2][2] = 1.0f - 2.0f * n->z * n->z;
    m[2][3] = pdotn * n->z;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*/
void PSMTXReflect ( Mtx m, const Vec *p, const Vec *n )
{
    f32x2    vn_xy, vn_z1, n2vn_xy, n2vn_z1, pdotn;
    f32x2    tmp0, tmp1, tmp2, tmp3;
    f32x2    tmp4, tmp5, tmp6, tmp7;

    // vn_z1 = [nz][1.0F] : LOAD
    //vn_z1[0] = n->z;
    //vn_z1[1] = 1.0F;
    vn_z1 = __PSQ_LX(n, 8, 1, 0);

    // vn_xy = [nx][ny]   : LOAD
    //vn_xy[0] = n->x;
    //vn_xy[1] = n->y;
    vn_xy = __PSQ_LX(n, 0, 0, 0);

    // tmp0 = [px][py]   : LOAD
    //tmp0[0] = p->x;
    //tmp0[1] = p->y;
    tmp0 = __PSQ_LX(p, 0, 0, 0);

    // n2vn_z1 = [-2nz][-2.0F]
    n2vn_z1 = __PS_NMADD(vn_z1, c11, vn_z1);

    // tmp1 = [pz][1.0F] : LOAD
    //psq_l       tmp1,  8(p), 1, 0
    //tmp1[0] = p->z;
    //tmp1[1] = 1.0F;
    tmp1 = __PSQ_LX(p, 8, 1, 0);

    // n2vn_xy = [-2nx][-2ny]
    n2vn_xy = __PS_NMADD(vn_xy, c11, vn_xy);

    // tmp4 = [-2nx*nz][-2ny*nz]   : [m20][m21]
    tmp4 = __PS_MULS0(vn_xy, n2vn_z1);

    // pdotn = [-2(px*nx)][-2(py*ny)]
    pdotn = __PS_MUL(n2vn_xy, tmp0);

    // tmp2 = [-2nx*nx][-2nx*ny]
    tmp2 = __PS_MULS0(vn_xy, n2vn_xy);

    // pdotn = [-2(px*nx+py*ny)][?]
    pdotn = __PS_SUM0(pdotn, pdotn, pdotn);

    // tmp3 = [-2nx*ny][-2ny*ny]
    tmp3 = __PS_MULS1(vn_xy, n2vn_xy);

    // tmp4 = [m20][m21] : STORE
    //m[2][0] = tmp4[0];
    //m[2][1] = tmp4[1];
    __PSQ_STX(m, 32, tmp4, 0, 0);

    // tmp2 = [1-2nx*nx][-2nx*ny]  : [m00][m01]
    tmp2  = __PS_SUM0(tmp2, tmp2, c11);

    // pdotn = [2(px*nx+py*ny+pz*nz)][?]
    pdotn  = __PS_NMADD(n2vn_z1, tmp1, pdotn);

    // tmp3 = [-2nx*ny][1-2ny*ny]  : [m10][m11]
    tmp3 = __PS_SUM1(c11, tmp3, tmp3);

    // tmp2 = [m00][m01] : STORE
    //m[0][0] = tmp2[0];
    //m[0][1] = tmp2[1];
    __PSQ_STX(m, 0, tmp2, 0, 0);

    // tmp5 = [pdotn*nx][pdotn*ny]
    tmp5 = __PS_MULS0(vn_xy, pdotn);

    // tmp6 = [-2nz][pdotn]
    tmp6 = __PS_MERGE00(n2vn_z1, pdotn);

    // tmp3 = [m10][m11] : STORE
    //m[1][0] = tmp3[0];
    //m[1][1] = tmp3[1];
    __PSQ_STX(m, 16, tmp3, 0, 0);

    // tmp7 = [-2nx*nz][pdotn*nx]  : [m02][m03]
    tmp7 = __PS_MERGE00(tmp4, tmp5);

    // tmp6 = [-2nz*nz][pdotn*nz]
    tmp6 = __PS_MULS0(tmp6, vn_z1);

    // tmp5 = [-2ny*nz][pdotn*ny]  : [m12][m13]
    tmp5 = __PS_MERGE11(tmp4, tmp5);

    // tmp7 = [m02][m03] : STORE
    //m[0][2] = tmp7[0];
    //m[0][3] = tmp7[1];
    __PSQ_STX(m, 8, tmp7, 0, 0);

    // tmp6 = [1-2nz*nz][pdotn*nz] : [m22][m23]
    tmp6 = __PS_SUM0(tmp6, tmp6, c11);

    // tmp5 = [m12][m13] : STORE
    //m[1][2] = tmp5[0];
    //m[1][3] = tmp5[1];
    __PSQ_STX(m, 24, tmp5, 0, 0);

    // tmp6 = [m22][m23] : STORE
    //m[2][2] = tmp6[0];
    //m[2][3] = tmp6[1];
    __PSQ_STX(m, 40, tmp6, 0, 0);
}
#endif


/*---------------------------------------------------------------------*

                             VIEW SECTION

*---------------------------------------------------------------------*/

/*---------------------------------------------------------------------*

Name:           MTXLookAt

Description:    compute a matrix to transform points to camera coordinates.

Arguments:      m        matrix to be set

                camPos   camera position.

                camUp    camera 'up' direction.

                target   camera aim point.

Return:         none

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXLookAt ( Mtx m, const Point3d *camPos, const Vec *camUp, const Point3d *target )
{
    Vec vLook,vRight,vUp;

    ASSERTMSG( (m != 0),      MTX_LOOKAT_1    );
    ASSERTMSG( (camPos != 0), MTX_LOOKAT_2    );
    ASSERTMSG( (camUp  != 0), MTX_LOOKAT_3    );
    ASSERTMSG( (target != 0), MTX_LOOKAT_4    );

    // compute unit target vector
    // use negative value to look down (-Z) axis
    vLook.x = camPos->x - target->x;
    vLook.y = camPos->y - target->y;
    vLook.z = camPos->z - target->z;
    VECNormalize( &vLook,&vLook );

    // vRight = camUp x vLook
    VECCrossProduct    ( camUp, &vLook, &vRight );
    VECNormalize( &vRight,&vRight );

    // vUp = vLook x vRight
    VECCrossProduct( &vLook, &vRight, &vUp );
    // Don't need to normalize vUp since it should already be unit length
    // VECNormalize( &vUp, &vUp );

    m[0][0] = vRight.x;
    m[0][1] = vRight.y;
    m[0][2] = vRight.z;
    m[0][3] = -( camPos->x * vRight.x + camPos->y * vRight.y + camPos->z * vRight.z );

    m[1][0] = vUp.x;
    m[1][1] = vUp.y;
    m[1][2] = vUp.z;
    m[1][3] = -( camPos->x * vUp.x + camPos->y * vUp.y + camPos->z * vUp.z );

    m[2][0] = vLook.x;
    m[2][1] = vLook.y;
    m[2][2] = vLook.z;
    m[2][3] = -( camPos->x * vLook.x + camPos->y * vLook.y + camPos->z * vLook.z );
}

/*---------------------------------------------------------------------*


                       TEXTURE PROJECTION SECTION


*---------------------------------------------------------------------*/

/*---------------------------------------------------------------------*

Name:           MTXLightFrustum

Description:    Compute a 3x4 projection matrix for texture projection

Arguments:      m        3x4 matrix to be set

                t        top coord. of view volume at the near clipping plane

                b        bottom coord of view volume at the near clipping plane

                lf       left coord. of view volume at near clipping plane

                r        right coord. of view volume at near clipping plane

                n        positive distance from camera to near clipping plane

                scaleS   scale in the S direction for projected coordinates
                         (usually 0.5)

                scaleT   scale in the T direction for projected coordinates
                         (usually 0.5)

                transS   translate in the S direction for projected coordinates
                         (usually 0.5)

                transT   translate in the T direction for projected coordinates
                         (usually 0.5)

Return:         none.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXLightFrustum  ( Mtx m, float t, float b, float lf, float r, float n,
                          float scaleS, float scaleT, float transS,
                          float transT )
{
    f32 tmp;

    ASSERTMSG( (m != 0),  MTX_LIGHT_FRUSTUM_1  );
    ASSERTMSG( (t != b),  MTX_LIGHT_FRUSTUM_2  );
    ASSERTMSG( (lf != r), MTX_LIGHT_FRUSTUM_3  );

    tmp     =  1.0f / (r - lf);
    m[0][0] =  ((2*n) * tmp) * scaleS;
    m[0][1] =  0.0f;
    m[0][2] =  (((r + lf) * tmp) * scaleS) - transS;
    m[0][3] =  0.0f;

    tmp     =  1.0f / (t - b);
    m[1][0] =  0.0f;
    m[1][1] =  ((2*n) * tmp) * scaleT;
    m[1][2] =  (((t + b) * tmp) * scaleT) - transT;
    m[1][3] =  0.0f;

    m[2][0] =  0.0f;
    m[2][1] =  0.0f;
    m[2][2] = -1.0f;
    m[2][3] =  0.0f;
}

/*---------------------------------------------------------------------*

Name:           MTXLightPerspective

Description:    compute a 3x4 perspective projection matrix from
                field of view and aspect ratio for texture projection.

Arguments:      m        3x4 matrix to be set

                fovy     total field of view in in degrees in the YZ plane

                aspect   ratio of view window width:height (X / Y)

                scaleS   scale in the S direction for projected coordinates
                         (usually 0.5)

                scaleT   scale in the T direction for projected coordinates
                         (usually 0.5)

                transS   translate in the S direction for projected coordinates
                         (usually 0.5)

                transT   translate in the T direction for projected coordinates
                         (usually 0.5)

Return:         none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXLightPerspective  ( Mtx m, f32 fovY, f32 aspect, float scaleS,
                              float scaleT, float transS, float transT )
{
    f32 angle;
    f32 cot;

    ASSERTMSG( (m != 0),                            MTX_LIGHT_PERSPECTIVE_1  );
    ASSERTMSG( ( (fovY > 0.0) && ( fovY < 180.0) ), MTX_LIGHT_PERSPECTIVE_2  );
    ASSERTMSG( (aspect != 0),                       MTX_LIGHT_PERSPECTIVE_3  );

    // find the cotangent of half the (YZ) field of view
    angle = fovY * 0.5f;
    angle = MTXDegToRad( angle );

    cot = 1.0f / tanf(angle);

    m[0][0] =    (cot / aspect) * scaleS;
    m[0][1] =    0.0f;
    m[0][2] =    -transS;
    m[0][3] =    0.0f;

    m[1][0] =    0.0f;
    m[1][1] =    cot * scaleT;
    m[1][2] =    -transT;
    m[1][3] =    0.0f;

    m[2][0] =    0.0f;
    m[2][1] =    0.0f;
    m[2][2] =   -1.0f;
    m[2][3] =    0.0f;
}

/*---------------------------------------------------------------------*

Name:           MTXLightOrtho

Description:    compute a 3x4 orthographic projection matrix.

Arguments:      m        matrix to be set

                t        top coord. of parallel view volume

                b        bottom coord of parallel view volume

                lf       left coord. of parallel view volume

                r        right coord. of parallel view volume

                scaleS   scale in the S direction for projected coordinates
                         (usually 0.5)

                scaleT   scale in the T direction for projected coordinates
                         (usually 0.5)

                transS   translate in the S direction for projected coordinates
                         (usually 0.5)

                transT   translate in the T direction for projected coordinates
                         (usually 0.5)

Return:         none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXLightOrtho ( Mtx m, f32 t, f32 b, f32 lf, f32 r, float scaleS,
                              float scaleT, float transS, float transT )
{
    f32 tmp;

    ASSERTMSG( (m != 0),  MTX_LIGHT_ORTHO_1     );
    ASSERTMSG( (t != b),  MTX_LIGHT_ORTHO_2     );
    ASSERTMSG( (lf != r), MTX_LIGHT_ORTHO_3     );

    tmp     =  1.0f / (r - lf);
    m[0][0] =  (2.0f * tmp * scaleS);
    m[0][1] =  0.0f;
    m[0][2] =  0.0f;
    m[0][3] =  ((-(r + lf) * tmp) * scaleS) + transS;

    tmp     =  1.0f / (t - b);
    m[1][0] =  0.0f;
    m[1][1] =  (2.0f * tmp) * scaleT;
    m[1][2] =  0.0f;
    m[1][3] =  ((-(t + b) * tmp)* scaleT) + transT;

    m[2][0] =  0.0f;
    m[2][1] =  0.0f;
    m[2][2] =  0.0f;
    m[2][3] =  1.0f;
}

/*---------------------------------------------------------------------*

Name:           MTXReorder

Description:    Creates a reordered (column-major) matrix from a
                row-major matrix, using paired single operations.
                Reordered matrices are required for the MTXRO*
                functions, which operate faster than their non-reordered
                counterparts.

Arguments:      src      source matrix.
                dest     destination matrix, note type is ROMtx.

Return:         none

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXReorder(MTX_CONST Mtx src, ROMtx dst)
{
    dst[0][0] = src[0][0];    dst[0][1] = src[1][0];    dst[0][2] = src[2][0];
    dst[1][0] = src[0][1];    dst[1][1] = src[1][1];    dst[1][2] = src[2][1];
    dst[2][0] = src[0][2];    dst[2][1] = src[1][2];    dst[2][2] = src[2][2];
    dst[3][0] = src[0][3];    dst[3][1] = src[1][3];    dst[3][2] = src[2][3];
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*/
void PSMTXReorder(MTX_CONST Mtx src, register ROMtx dest)
{
    f32x2 S00_S01, S02_S03, S10_S11, S12_S13, S20_S21, S22_S23;
    f32x2 D00_D10, D11_D21, D02_D12, D22_D03, D13_D23, D20_D01;

    //psq_l       S00_S01, 0(src),  0, 0
    S00_S01 = __PSQ_L(src, 0, 0);

    //psq_l       S10_S11, 16(src), 0, 0
    S10_S11 = __PSQ_LX(src, 16, 0, 0);

    //psq_l       S20_S21, 32(src), 0, 0
    S20_S21 = __PSQ_LX(src, 32, 0, 0);

    //psq_l       S02_S03, 8(src),  0, 0
    S02_S03 = __PSQ_LX(src, 8, 0, 0);

    //ps_merge00  D00_D10, S00_S01, S10_S11
    D00_D10 = __PS_MERGE00(S00_S01, S10_S11);

    //psq_l       S12_S13, 24(src), 0, 0
    S12_S13 = __PSQ_LX(src, 24, 0, 0);

    //ps_merge01  D20_D01, S20_S21, S00_S01
    D20_D01 = __PS_MERGE01(S20_S21, S00_S01);

    //psq_l       S22_S23, 40(src), 0, 0
    S22_S23 = __PSQ_LX(src, 40, 0, 0);

    //ps_merge11  D11_D21, S10_S11, S20_S21
    D11_D21 = __PS_MERGE11(S10_S11, S20_S21);

    //psq_st      D00_D10, 0(dest), 0, 0
    __PSQ_ST(dest, D00_D10, 0, 0);

    //ps_merge00  D02_D12, S02_S03, S12_S13
    D02_D12 = __PS_MERGE00(S02_S03, S12_S13);

    //psq_st      D20_D01, 8(dest), 0, 0
    __PSQ_STX(dest, 8, D20_D01, 0, 0);

    //ps_merge01  D22_D03, S22_S23, S02_S03
    D22_D03 = __PS_MERGE01(S22_S23, S02_S03);

    //psq_st      D11_D21, 16(dest),0, 0
    __PSQ_STX(dest, 16, D11_D21, 0, 0);

    //ps_merge11  D13_D23, S12_S13, S22_S23
    D13_D23 = __PS_MERGE11(S12_S13, S22_S23);

    //psq_st      D02_D12, 24(dest),0, 0
    __PSQ_STX(dest, 24, D02_D12, 0, 0);

    //psq_st      D22_D03, 32(dest),0,0
    __PSQ_STX(dest, 32, D22_D03, 0, 0);

    //psq_st      D13_D23, 40(dest),0,0
    __PSQ_STX(dest, 40, D13_D23, 0, 0);
}

/*===========================================================================*/


extern void _ASM_MTXRotAxisRadInternal(Mtx m, const Vec *axis, f32 sT, f32 cT);

void ASM_MTXRotAxisRad(Mtx        m,
                       const Vec *axis,
                       f32        rad ) {
    f32     sinT, cosT;

    sinT = sinf(rad);
    cosT = cosf(rad);

    _ASM_MTXRotAxisRadInternal(m, axis, sinT, cosT);
}

void ASM_MTXRotRad ( Mtx m, char axis, f32 rad )
{
    f32 sinA, cosA;

    sinA = sinf(rad);
    cosA = cosf(rad);

    ASM_MTXRotTrig( m, axis, sinA, cosA );
}

void ASM_QUATDivide( const Quaternion *p, const Quaternion *q, Quaternion *r)
{
    Quaternion qtmp;

    ASM_QUATIlwerse(q, &qtmp);
    ASM_QUATMultiply(&qtmp, p, r);
}
#endif
