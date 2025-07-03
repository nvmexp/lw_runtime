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

                             VECTOR SECTION

 *---------------------------------------------------------------------*/

//static const f32x2    c_zero  = {0.0F, 0.0F};
static const f32x2    c_half  = {0.5F, 0.5F};
static const f32x2    c_three = {3.0F, 3.0F};

/*---------------------------------------------------------------------*

Name:           VECAdd

Description:    add two vectors.


Arguments:      a    first vector.

                b    second vector.

                ab   resultant vector (a + b).
                     ok if ab == a or ab == b.


Return:         none.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_VECAdd ( const Vec *a, const Vec *b, Vec *ab )
{

    ASSERTMSG( ( a    != 0), VEC_ADD_1 );
    ASSERTMSG( ( b    != 0), VEC_ADD_2 );
    ASSERTMSG( ( ab != 0),   VEC_ADD_3 );


    ab->x = a->x + b->x;
    ab->y = a->y + b->y;
    ab->z = a->z + b->z;

}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSVECAdd ( const Vec *a, const Vec *b, Vec *ab )
{
    f32x2 V1_XY;
    f32x2 V2_XY;
    f32x2 V1_Z;
    f32x2 V2_Z;
    f32x2 D1_XY;
    f32x2 D1_Z;

    //load vectors XY
    V1_XY = __PSQ_L(a, 0, 0);
    V2_XY = __PSQ_L(b, 0, 0);

    //add vectors XY
    D1_XY = __PS_ADD(V1_XY, V2_XY);

    //store result XY
    __PSQ_ST(ab, D1_XY, 0, 0);

    //load vectors Z
    V1_Z = __PSQ_LX(a, 8, 1, 0);
    V2_Z = __PSQ_LX(b, 8, 1, 0);

    //add vectors Z
    D1_Z = __PS_ADD(V1_Z, V2_Z);

    //store result YZ
    __PSQ_STX(ab, 8, D1_Z, 1, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:           VECSubtract

Description:    subtract one vector from another.


Arguments:      a       first vector.

                b       second vector.

                a_b     resultant vector (a - b).
                        ok if a_b == a or a_b == b.


Return:         none.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_VECSubtract ( const Vec *a, const Vec *b, Vec *a_b )
{

    ASSERTMSG( ( a    != 0),    VEC_SUBTRACT_1     );
    ASSERTMSG( ( b    != 0),    VEC_SUBTRACT_2     );
    ASSERTMSG( ( a_b != 0),     VEC_SUBTRACT_3     );


    a_b->x = a->x - b->x;
    a_b->y = a->y - b->y;
    a_b->z = a->z - b->z;

}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSVECSubtract ( const Vec *a, const Vec *b, Vec *ab )
{
    f32x2 V1_XY;
    f32x2 V2_XY;
    f32x2 V1_Z;
    f32x2 V2_Z;
    f32x2 D1_XY;
    f32x2 D1_Z;

    //load vectors XY
    V1_XY = __PSQ_L(a, 0, 0);
    V2_XY = __PSQ_L(b, 0, 0);

    //sub vectors XY
    D1_XY = __PS_SUB(V1_XY, V2_XY);

    //store result XY
    __PSQ_ST(ab, D1_XY, 0, 0);

    //load vectors Z
    V1_Z = __PSQ_LX(a, 8, 1, 0);
    V2_Z = __PSQ_LX(b, 8, 1, 0);

    //sub vectors Z
    D1_Z = __PS_SUB(V1_Z, V2_Z);

    //store result Z
    __PSQ_STX(ab, 8, D1_Z, 1, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:           VECScale

Description:    multiply a vector by a scalar.


Arguments:      src     unscaled source vector.

                dst     scaled resultant vector ( src * scale).
                        ok if dst == src.

                scale   scaling factor.


Return:         none.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_VECScale ( const Vec *src, Vec *dst, f32 scale )
{

    ASSERTMSG( ( src  != 0),  VEC_SCALE_1  );
    ASSERTMSG( ( dst  != 0),  VEC_SCALE_2  );


    dst->x = src->x * scale;
    dst->y = src->y * scale;
    dst->z = src->z * scale;

}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSVECScale ( const Vec *src, Vec *dst, f32 scale )
{
    f32x2 V1_XY;
    f32x2 V1_Z;
    f32x2 D1_XY;
    f32x2 D1_Z;
    f32x2 SCALE = {scale, scale};

    //load vectors XY
    V1_XY = __PSQ_L(src, 0, 0);

    //load vectors Z
    V1_Z = __PSQ_LX(src, 8, 1, 0);

    //muls vectors XY
    D1_XY = __PS_MULS0(V1_XY, SCALE);

    //store result XY
    __PSQ_ST(dst, D1_XY, 0, 1);

    //muls vectors Z
    D1_Z = __PS_MULS0(V1_Z, SCALE);

    //store result YZ
    __PSQ_STX(dst, 8, D1_Z, 1, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:           VECNormalize

Description:    normalize a vector.


Arguments:      src     non-unit source vector.

                unit    resultant unit vector ( src / src magnitude ).
                        ok if unit == src


Return:         none.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_VECNormalize ( const Vec *src, Vec *unit )
{
    f32 mag;

    ASSERTMSG( (src != 0 ),     VEC_NORMALIZE_1  );
    ASSERTMSG( (unit != 0),     VEC_NORMALIZE_2  );

    mag = (src->x * src->x) + (src->y * src->y) + (src->z * src->z);

    ASSERTMSG( (mag != 0),      VEC_NORMALIZE_3  );

    mag = 1.0f / sqrtf(mag);

    unit->x = src->x * mag;
    unit->y = src->y * mag;
    unit->z = src->z * mag;

}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSVECNormalize ( const Vec *vec1, Vec *dst )
{
    f32x2 v1_xy, v1_z;
    f32x2 xx_zz, xx_yy;
    f32x2 sqsum;
    f32x2 rsqrt;
    f32x2 nwork0, nwork1;

    // X | Y
    //psq_l       v1_xy, 0(vec1), 0, 0;
    v1_xy = __PSQ_L(vec1, 0, 0);

    // X*X | Y*Y
    //ps_mul      xx_yy, v1_xy, v1_xy;
    xx_yy = __PS_MUL(v1_xy, v1_xy);

    // Z | 1
    //psq_l       v1_z, 8(vec1), 1, 0;
    v1_z = __PSQ_LX(vec1, 8, 1, 0);

    // X*X+Z*Z | Y*Y+1
    //ps_madd     xx_zz, v1_z, v1_z, xx_yy;
    xx_zz = __PS_MADD(v1_z, v1_z, xx_yy);

    // X*X+Z*Z+Y*Y | Z
    //ps_sum0     sqsum, xx_zz, v1_z, xx_yy;
    sqsum = __PS_SUM0(xx_zz, v1_z, xx_yy);

    // 1.0/sqrt : estimation[E]
    //frsqrte     rsqrt, sqsum;
    rsqrt = __PS_RSQRTE(sqsum);

    // Newton's refinement x 1
    // E' = (E/2)(3 - sqsum * E * E)
    //fmuls       nwork0, rsqrt, rsqrt;
    //fmuls       nwork1, rsqrt, c_half;
    //fnmsubs     nwork0, nwork0, sqsum, c_three;
    //fmuls       rsqrt, nwork0, nwork1;
    nwork0 = __PS_MUL(rsqrt, rsqrt);
    nwork1 = __PS_MUL(rsqrt, c_half);
    nwork0 = __PS_NMSUB(nwork0, sqsum, c_three);
    rsqrt = __PS_MUL(nwork0, nwork1);

    // X * mag | Y * mag
    //ps_muls0    v1_xy, v1_xy, rsqrt;
    v1_xy = __PS_MULS0(v1_xy, rsqrt);

    //psq_st      v1_xy, 0(dst), 0, 0;
    __PSQ_ST(dst, v1_xy, 0, 0);

    // Z * mag
    //ps_muls0    v1_z, v1_z, rsqrt;
    v1_z = __PS_MULS0(v1_z, rsqrt);

    //psq_st      v1_z, 8(dst), 1, 0;
    __PSQ_STX(dst, 8, v1_z, 1, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:           VECSquareMag

Description:    compute the square of the magnitude of a vector.


Arguments:      v    source vector.


Return:         square magnitude of the vector.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
f32 C_VECSquareMag ( const Vec *v )
{
    f32 sqmag;

    ASSERTMSG( (v != 0),  VEC_MAG_1 );

    sqmag = (v->x * v->x) + (v->y * v->y) + (v->z * v->z);

    return sqmag;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------*/

f32 PSVECSquareMag ( const Vec *v )
{
    f32x2    V1_XY, V1_ZZ, sqmag;

    // load X | Y
    V1_XY = __PSQ_L(v, 0, 0);

    // XX | YY
    V1_XY = __PS_MUL(V1_XY, V1_XY);

    // load Z | Z
    V1_ZZ[0] = v->z;
    V1_ZZ[1] = v->z;

    // XX + ZZ | YY + ZZ
    sqmag = __PS_MADD(V1_ZZ, V1_ZZ, V1_XY);
    sqmag = __PS_SUM0(sqmag, V1_XY, V1_XY);

    return (f32)sqmag[0];
}
#endif

/*---------------------------------------------------------------------*

Name:           VECMag

Description:    compute the magnitude of a vector.


Arguments:      v    source vector.


Return:         magnitude of the vector.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
f32 C_VECMag ( const Vec *v )
{
    return sqrtf( C_VECSquareMag(v) );
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*/
f32 PSVECMag ( const Vec *v )
{
    f32x2    vzz, vxy; //vyz, 
    f32x2    sqmag, rmag; //dyz, dxy, 
    f32x2    nwork0, nwork1;

    // Square mag callwlation
    //psq_l       vxy, 0(v), 0, 0
    vxy = __PSQ_L(v, 0, 0);

    //ps_mul      vxy, vxy, vxy
    vxy = __PS_MUL(vxy, vxy);
    vzz = __PSQ_LX(v, 8, 0, 0);

    //ps_madd     sqmag, vzz, vzz, vxy
    sqmag = __PS_MADD(vzz, vzz, vxy);

    // Square mag
    //ps_sum0     sqmag, sqmag, vxy, vxy
    sqmag = __PS_SUM0(sqmag, vxy, vxy);
    
    if (sqmag[0] != 0)
    {
        // 1.0/sqrt : estimation[E]
        //frsqrte     rmag, sqmag
        rmag = __PS_RSQRTE(sqmag);

        // Refinement x 1 : E' = (E/2)(3 - X*E*E)
        //fmul        nwork0, rsqmag, rsqmag
        nwork0 = __PS_MUL(rmag, rmag);

        //fmul        nwork1, rsqmag, c_half
        nwork1 = __PS_MUL(rmag, c_half);

        //fnmsub      nwork0, nwork0, mag, c_three
        nwork0 = __PS_NMSUB(nwork0, sqmag, c_three);

        //fmul        rsqmag, nwork0, nwork1
        rmag = __PS_MUL(nwork0, nwork1);

        // 1/sqrt(X) * X = sqrt(X)
        //fmuls       sqmag, sqmag, rmag
        sqmag = __PS_MUL(sqmag, rmag);
    }

    return (f32)sqmag[0];
}
#endif

/*---------------------------------------------------------------------*

Name:           VECDotProduct

Description:    compute the dot product of two vectors.


Arguments:      a    first vector.

                b    second vector.

                note:  input vectors do not have to be normalized.
                       input vectors are not normalized within the function.

                       if direct cosine computation of the angle
                       between a and b is desired, a and b should be
                       normalized prior to calling VECDotProduct.


Return:         dot product value.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
f32 C_VECDotProduct ( const Vec *a, const Vec *b )
{
    f32 dot;

    ASSERTMSG( (a != 0), VEC_DOTPRODUCT_1 );
    ASSERTMSG( (b != 0), VEC_DOTPRODUCT_2 );

    dot = (a->x * b->x) + (a->y * b->y) + (a->z * b->z);

    return dot;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
f32 PSVECDotProduct ( const Vec *vec1, const Vec *vec2 )
{
    f32x2    fp1, fp2, fp3, fp4, fp5;

    //psq_l    fp2, 4(vec1), 0, 0;
    fp2 = __PSQ_LX(vec1, 4, 0, 0);

    //psq_l    fp3, 4(vec2), 0, 0;
    fp3 = __PSQ_LX(vec2, 4, 0, 0);

    //ps_mul   fp2, fp2, fp3;
    fp2 = __PS_MUL(fp2, fp3);

    //psq_l    fp5, 0(vec1), 0, 0;
    fp5 = __PSQ_L(vec1, 0, 0);

    //psq_l    fp4, 0(vec2), 0, 0;
    fp4 = __PSQ_L(vec2, 0, 0);

    //ps_madd  fp3, fp5, fp4, fp2;
    fp3 = __PS_MADD(fp5, fp4, fp2);

    //ps_sum0  fp1, fp3, fp2, fp2;
    fp1 = __PS_SUM0(fp3, fp2, fp2);

    return (f32)fp1[0];
}
#endif

/*---------------------------------------------------------------------*

Name:           VECCrossProduct

Description:    compute the cross product of two vectors.


Arguments:      a       first vector.

                b       second vector.

                note:  input vectors do not have to be normalized.


                axb     resultant vector.
                        ok if axb == a or axb == b.


Return:         none.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_VECCrossProduct ( const Vec *a, const Vec *b, Vec *axb )
{
    Vec vTmp;


    ASSERTMSG( (a    != 0),   VEC_CROSSPRODUCT_1    );
    ASSERTMSG( (b    != 0),   VEC_CROSSPRODUCT_2    );
    ASSERTMSG( (axb != 0),    VEC_CROSSPRODUCT_3    );


    vTmp.x =  ( a->y * b->z ) - ( a->z * b->y );
    vTmp.y =  ( a->z * b->x ) - ( a->x * b->z );
    vTmp.z =  ( a->x * b->y ) - ( a->y * b->x );


    axb->x = vTmp.x;
    axb->y = vTmp.y;
    axb->z = vTmp.z;

}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSVECCrossProduct
(
    const Vec *vec1,
    const Vec *vec2,
    Vec *dst
)
{
    f32x2 fp0, fp1;
    f32x2 fp2 = {vec1->z, vec1->z};
    f32x2 fp3 = {vec2->z, vec2->z};
    f32x2 fp4, fp5, fp6, fp7, fp8, fp9, fp10;

    //x =   a.n[VY]*b.n[VZ] - a.n[VZ]*b.n[VY];
    //y =   a.n[VZ]*b.n[VX] - a.n[VX]*b.n[VZ];
    //z =   a.n[VX]*b.n[VY] - a.n[VY]*b.n[VX];

    // BX | BY
    fp1 = __PSQ_L(vec2, 0, 0);

    // AX | AY
    fp0 = __PSQ_L(vec1, 0, 0);

    // BY | BX
    fp6 = __PS_MERGE10(fp1, fp1);

    // BX*AZ | BY*AZ
    fp4 = __PS_MUL(fp1, fp2);

    // BX*AX | BY*AX
    fp7 = __PS_MULS0(fp1, fp0);

    // AX*BZ-BX*AZ | AY*BZ-BY*AZ
    fp5 = __PS_MSUB(fp0, fp3, fp4);

    // AX*BY-BX*AX | AY*BX-BY*AX
    fp8 = __PS_MSUB(fp0, fp6, fp7);

    // AY*BZ-AZ*BY | AY*BZ-AZ*BY
    fp9 = __PS_MERGE11(fp5, fp5);

    // AX*BZ-AZ*BX | AY*BX-AX*BY
    fp10 = __PS_MERGE01(fp5, fp8);

    //Store X
    __PSQ_ST(dst, fp9, 1, 0);

    // AZ*BX-AX*BZ | AX*BY-AY*BX
    fp10 = __PS_NEG(fp10);

    // store YZ
    __PSQ_STX(dst, 4, fp10, 0, 0);

}
#endif

/*---------------------------------------------------------------------*

Name:           VECHalfAngle

Description:    compute the vector halfway between two vectors.
                intended for use in computing spelwlar highlights


Arguments:      a     first vector.
                      this must point FROM the light source (tail)
                      TO the surface (head).

                b     second vector.
                      this must point FROM the viewer (tail)
                      TO the surface (head).

                note:     input vectors do not have to be normalized.


                half  resultant normalized 'half-angle' vector.
                      ok if half == a or half == b


Return:         none.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_VECHalfAngle ( const Vec *a, const Vec *b, Vec *half )
{
    Vec aTmp, bTmp, hTmp;


    ASSERTMSG( (a    != 0),    VEC_HALFANGLE_1  );
    ASSERTMSG( (b    != 0),    VEC_HALFANGLE_2  );
    ASSERTMSG( (half != 0),    VEC_HALFANGLE_3  );


    aTmp.x = -a->x;
    aTmp.y = -a->y;
    aTmp.z = -a->z;

    bTmp.x = -b->x;
    bTmp.y = -b->y;
    bTmp.z = -b->z;

    C_VECNormalize( &aTmp, &aTmp );
    C_VECNormalize( &bTmp, &bTmp );

    C_VECAdd( &aTmp, &bTmp, &hTmp );

    if ( C_VECDotProduct( &hTmp, &hTmp ) > 0.0F )
    {
        C_VECNormalize( &hTmp, half );
    }
    else    // The singular case returns zero vector
    {
        *half = hTmp;
    }

}

/*---------------------------------------------------------------------*

Name:           VECReflect

Description:    reflect a vector about a normal to a surface.


Arguments:      src        incident vector.

                normal     normal to surface.

                dst        normalized reflected vector.
                           ok if dst == src


Return:         none.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_VECReflect ( const Vec *src, const Vec *normal, Vec *dst )
{
    f32 cosA;
    Vec uI, uN;


    ASSERTMSG( (src != 0),     VEC_REFLECT_1  );
    ASSERTMSG( (normal != 0),  VEC_REFLECT_2  );
    ASSERTMSG( (dst != 0),     VEC_REFLECT_3  );


    // assume src is incident to a surface.
    // reverse direction of src so that src and normal
    // sit tail to tail.
    uI.x = -( src->x );
    uI.y = -( src->y );
    uI.z = -( src->z );


    // VECNormalize will catch any zero magnitude vectors
    C_VECNormalize( &uI,    &uI);
    C_VECNormalize( normal, &uN);

    // angle between the unit vectors
    cosA = C_VECDotProduct( &uI, &uN);


    // R = 2N(N.I) - I
    dst->x = (2.0f * uN.x * cosA) - uI.x;
    dst->y = (2.0f * uN.y * cosA) - uI.y;
    dst->z = (2.0f * uN.z * cosA) - uI.z;

    C_VECNormalize( dst, dst );

}

/*---------------------------------------------------------------------*

Name:           VECSquareDistance

Description:    Returns the square of the distance between vectors
                a and b.  Distance can be callwlated using the
                square root of the returned value.


Arguments:      a     first vector.

                b     second vector.


Return:         square distance of between vectors.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
f32 C_VECSquareDistance( const Vec *a, const Vec *b )
{
    Vec diff;

    diff.x = a->x - b->x;
    diff.y = a->y - b->y;
    diff.z = a->z - b->z;

    return (diff.x * diff.x) + (diff.y * diff.y) + (diff.z * diff.z);
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*/

f32 PSVECSquareDistance( const Vec *a, const Vec *b )
{
    f32x2    v0yz, v1yz, v0xy, v1xy;
    f32x2    dyz, dxy, sqdist;

    v0yz = __PSQ_LX(a, 4, 0, 0);
    v1yz = __PSQ_LX(b, 4, 0, 0);

    dyz = __PS_SUB(v0yz, v1yz); // [Y0-Y1][Z0-Z1]

    v0xy = __PSQ_L(a, 0, 0);
    v1xy = __PSQ_L(b, 0, 0);

    dyz = __PS_MUL(dyz, dyz);              // [dYdY][dZdZ]
    dxy = __PS_SUB(v0xy, v1xy);            // [X0-X1][Y0-Y1]

    sqdist = __PS_MADD(dxy, dxy, dyz);      // [dXdX+dYdY][dYdY+dZdZ]
    sqdist = __PS_SUM0(sqdist, dyz, dyz);   // [dXdX+dYdY+dZdZ][N/A]

    return (f32)sqdist[0];
}
#endif


/*---------------------------------------------------------------------*

Name:           VECDistance

Description:    Returns the distance between vectors a and b.


Arguments:      a     first vector.

                b     second vector.


Return:         distance between the two vectors.

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
f32 C_VECDistance( const Vec *a, const Vec *b )
{
    return sqrtf( C_VECSquareDistance( a, b ) );
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*/
f32 PSVECDistance( const Vec *a, const Vec *b )
{
    f32x2    v0yz, v1yz, v0xy, v1xy;
    f32x2    dyz, dxy, sqdist, rdist;
    f32x2    nwork0, nwork1;

    //psq_l       v0yz, 4(a), 0, 0           // [Y0][Z0]
    v0yz = __PSQ_LX(a, 4, 0, 0);

    //psq_l       v1yz, 4(b), 0, 0           // [Y1][Z1]
    v1yz = __PSQ_LX(b, 4, 0, 0);

    //ps_sub      dyz, v0yz, v1yz            // [Y0-Y1][Z0-Z1]
    dyz = __PS_SUB(v0yz, v1yz);

    //psq_l       v0xy, 0(a), 0, 0           // [X0][Y0]
    v0xy = __PSQ_L(a, 0, 0);

    //psq_l       v1xy, 0(b), 0, 0           // [X1][Y1]
    v1xy = __PSQ_L(b, 0, 0);

    //ps_mul      dyz, dyz, dyz              // [dYdY][dZdZ]
    dyz = __PS_MUL(dyz, dyz);

    //ps_sub      dxy, v0xy, v1xy            // [X0-X1][Y0-Y1]
    dxy = __PS_SUB(v0xy, v1xy);

    //ps_madd     sqdist, dxy, dxy, dyz      // [dXdX+dYdY][dYdY+dZdZ]
    sqdist = __PS_MADD(dxy, dxy, dyz);

    //ps_sum0     sqdist, sqdist, dyz, dyz   // [dXdX+dYdY+dZdZ][N/A]
    sqdist = __PS_SUM0(sqdist, dyz, dyz);
    
    if (sqdist[0] != 0)
    {
        // 1.0/sqrt : estimation[E]
        //frsqrte     rdist, sqdist
        rdist = __PS_RSQRTE(sqdist);

        // Refinement x 1 : E' = (E/2)(3 - X*E*E)
        //fmul        nwork0, rsqmag, rsqmag
        nwork0 = __PS_MUL(rdist, rdist);

        //fmul        nwork1, rsqmag, c_half
        nwork1 = __PS_MUL(rdist, c_half);

        //fnmsub      nwork0, nwork0, mag, c_three
        nwork0 = __PS_NMSUB(nwork0, sqdist, c_three);

        //fmul        rsqmag, nwork0, nwork1
        rdist = __PS_MUL(nwork0, nwork1);

        // 1/sqrt(X) * X = sqrt(X)
        //fmuls       sqdist, sqdist, rdist
        sqdist = __PS_MUL(sqdist, rdist);
    }

    return (f32)sqdist[0];
}
#endif


/*---------------------------------------------------------------------*

Name:           MTXMultVec

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
void C_MTXMultVec ( MTX_CONST Mtx m, const Vec *src, Vec *dst )
{
    Vec vTmp;

    ASSERTMSG( (m   != 0), MTX_MULTVEC_1 );
    ASSERTMSG( (src != 0), MTX_MULTVEC_2 );
    ASSERTMSG( (dst != 0), MTX_MULTVEC_3 );

    // a Vec has a 4th implicit 'w' coordinate of 1
    vTmp.x = m[0][0]*src->x + m[0][1]*src->y + m[0][2]*src->z + m[0][3];
    vTmp.y = m[1][0]*src->x + m[1][1]*src->y + m[1][2]*src->z + m[1][3];
    vTmp.z = m[2][0]*src->x + m[2][1]*src->y + m[2][2]*src->z + m[2][3];

    // copy back
    dst->x = vTmp.x;
    dst->y = vTmp.y;
    dst->z = vTmp.z;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that NO error checking is performed.
 *---------------------------------------------------------------------*/

void PSMTXMultVec ( MTX_CONST Mtx m, const Vec *src, Vec *dst )
{
    f32x2 fp0, fp1, fp2, fp3, fp4, fp5, fp6, fp8, fp9, fp10, fp11, fp12; // fp7, 
    const f32x2 zero = {0.0, 0.0};

    // load v[0], v[1]
    fp0 = __PSQ_L(src, 0, 0);

    // load m[0][0], m[0][1]
    fp2 = __PSQ_L(m, 0, 0);

    // load v[2], 1
    //fp1[0]=src->z;
    //fp1[1]=1.0F;
    fp1 = __PSQ_LX(src, 8, 1, 0);

    // m[0][0]*v[0], m[0][1]*v[1]
    fp4 = __PS_MUL(fp2, fp0);

    // load m[0][2], m[0][3]
    fp3 = __PSQ_LX(m, 8, 0, 0);

    // m[0][0]*v[0]+m[0][2]*v[2], m[0][1]*v[1]+m[0][3]
    fp5 = __PS_MADD(fp3, fp1, fp4);

    // load m[1][0], m[1][1]
    fp8 = __PSQ_LX(m, 16, 0, 0);
    fp6 = zero;

    // m[0][0]*v[0]+m[0][2]*v[2]+m[0][1]*v[1]+m[0][3], ???
    fp6 = __PS_SUM0(fp5, fp6, fp5);

    // load m[1][2], m[1][3]
    fp9 = __PSQ_LX(m, 24, 0, 0);

    // m[1][0]*v[0], m[1][1]*v[1]
    fp10 = __PS_MUL(fp8, fp0);

    // store dst[0]
    __PSQ_ST(dst, fp6, 1, 0);

    // m[1][0]*v[0]+m[1][2]*v[2], m[1][1]*v[1]+m[1][3]
    fp11 = __PS_MADD(fp9, fp1, fp10);

    // load m[2][0], m[2][1]
    fp2 = __PSQ_LX(m, 32, 0, 0);
    fp12 = zero;

    // m[1][0]*v[0]+m[1][2]*v[2]+m[2][1]*v[1]+m[1][3], ???
    fp12 = __PS_SUM0(fp11, fp12, fp11);

    // load m[2][2], m[2][3]
    fp3 = __PSQ_LX(m, 40, 0, 0);

    // m[0][0]*v[0], m[0][1]*v[1]
    fp4 = __PS_MUL(fp2, fp0);

    // store dst[1]
    __PSQ_STX(dst, 4, fp12, 1, 0);

    // m[0][0]*v[0]+m[0][2]*v[2], m[0][1]*v[1]+m[0][3]
    fp5 = __PS_MADD(fp3, fp1, fp4);
    fp6 = zero;

    // m[0][0]*v[0]+m[0][2]*v[2]+m[0][1]*v[1]+m[0][3], ???
    fp6 = __PS_SUM0(fp5, fp6, fp5);

    // store dst[2]
    __PSQ_STX(dst, 8, fp6, 1, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXMultVecArray

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
void C_MTXMultVecArray ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count )
{
    u32 i;
    Vec vTmp;

    ASSERTMSG( (m       != 0), MTX_MULTVECARRAY_1 );
    ASSERTMSG( (srcBase != 0), MTX_MULTVECARRAY_2 );
    ASSERTMSG( (dstBase != 0), MTX_MULTVECARRAY_3 );
    ASSERTMSG( (count > 1),    MTX_MULTVECARRAY_4 );

    for(i=0; i< count; i++)
    {
        // Vec has a 4th implicit 'w' coordinate of 1
        vTmp.x = m[0][0]*srcBase->x + m[0][1]*srcBase->y + m[0][2]*srcBase->z + m[0][3];
        vTmp.y = m[1][0]*srcBase->x + m[1][1]*srcBase->y + m[1][2]*srcBase->z + m[1][3];
        vTmp.z = m[2][0]*srcBase->x + m[2][1]*srcBase->y + m[2][2]*srcBase->z + m[2][3];

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
                Note that NO error checking is performed.

                The count should be greater than 1.
 *---------------------------------------------------------------------*/

void PSMTXMultVecArray ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count )
{
    u32 i;

    for ( i = 0 ; i < count ; i++ )
    {
        PSMTXMultVec(m, srcBase, dstBase);

        srcBase++;
        dstBase++;
    }
}
#endif

/*---------------------------------------------------------------------*

Name:         MTXMultVecSR

Description:  multiplies a vector by a matrix 3x3 (Scaling and Rotation)
              component.

              m x src = dst.

Arguments:    m       matrix.
              src     source vector for multiply.
              dst     resultant vector from multiply.

              note:   ok if src == dst.

Return:       none

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXMultVecSR ( MTX_CONST Mtx m, const Vec *src, Vec *dst )
{
    Vec vTmp;

    ASSERTMSG( (m   != 0), MTX_MULTVECSR_1 );
    ASSERTMSG( (src != 0), MTX_MULTVECSR_2 );
    ASSERTMSG( (dst != 0), MTX_MULTVECSR_3 );

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

void PSMTXMultVecSR ( MTX_CONST Mtx m, const Vec *src, Vec *dst )
{
    f32x2 fp0, fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12, fp13;

    // m[0][0], m[0][1]
    fp0 = __PSQ_L(m, 0, 0);

    // fp6 - x y
    fp6 = __PSQ_L(src, 0, 0);

    // m[1][0], m[1][1]
    fp2 = __PSQ_LX(m, 16, 0, 0);

    // fp8 = m00x m01y // next X
    fp8 = __PS_MUL(fp0, fp6);

    // m[2][0], m[2][1]
    fp4 = __PSQ_LX(m, 32, 0, 0);

    // fp10 = m10x m11y // next Y
    fp10 = __PS_MUL(fp2, fp6);

    // fp7 - z,1.0
    //fp7[0] = src->z;
    //fp7[1] = 1.0F;
    fp7 = __PSQ_LX(src, 8, 1, 0);

    // fp12 = m20x m21y // next Z
    fp12 = __PS_MUL(fp4, fp6);

    // m[1][2], m[1][3]
    fp3 = __PSQ_LX(m, 24, 0, 0);

    fp8 = __PS_SUM0(fp8, fp8, fp8);

    // m[2][2], m[2][3]
    fp5 = __PSQ_LX(m, 40, 0, 0);

    fp10 = __PS_SUM0(fp10, fp10, fp10);

    // m[0][2], m[0][3]
    fp1 = __PSQ_LX(m, 8, 0, 0);

    fp12 = __PS_SUM0(fp12, fp12, fp12);
    fp9 = __PS_MADD(fp1, fp7, fp8);

    // store X
    __PSQ_ST(dst, fp9, 1, 0);

    fp11 = __PS_MADD(fp3, fp7, fp10);

    // store Y
    __PSQ_STX(dst, 4, fp11, 1, 0);

    fp13 = __PS_MADD(fp5, fp7, fp12);

    //  store Z
    __PSQ_STX(dst, 8, fp13, 1, 0);
}
#endif

/*---------------------------------------------------------------------*

Name:           MTXMultVecArraySR

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
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXMultVecArraySR ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count )
{
    u32 i;
    Vec vTmp;

    ASSERTMSG( (m       != 0), MTX_MULTVECARRAYSR_1 );
    ASSERTMSG( (srcBase != 0), MTX_MULTVECARRAYSR_2 );
    ASSERTMSG( (dstBase != 0), MTX_MULTVECARRAYSR_3 );
    ASSERTMSG( (count > 1),    MTX_MULTVECARRAYSR_4 );

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
                Note that NO error checking is performed.
 *---------------------------------------------------------------------*/

void PSMTXMultVecArraySR ( MTX_CONST Mtx m, const Vec *srcBase, Vec *dstBase, u32 count )
{
    u32 i;

    for ( i = 0 ; i < count ; i++ )
    {
        PSMTXMultVecSR(m, srcBase, dstBase);

        srcBase++;
        dstBase++;
    }
}
#endif


/*---------------------------------------------------------------------*

Name:           MTXROMultVecArray

Description:    Multiplies an array of vectors by a reordered matrix,
                using paired single operations.
                OK if source = destination.
                NOTE: number of vertices transformed cannot be less than
                2.

                Note that NO error checking is performed.

Arguments:      m         reordered matrix.
                srcBase   start of source vector array.
                dstBase   start of resultant vector array.
                count     number of vectors in srcBase, dstBase arrays
                          COUNT MUST BE GREATER THAN 2.


Return:         none

*---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXROMultVecArray
(
    MTX_CONST ROMtx  m,      // r3
    const Vec   *srcBase,// r4
    Vec   *dstBase,// r5
    u32    count   // r6
)
{
    u32 i;
    Vec vTmp;

    ASSERTMSG( (m       != 0), MTX_MULTVECARRAY_1 );
    ASSERTMSG( (srcBase != 0), MTX_MULTVECARRAY_2 );
    ASSERTMSG( (dstBase != 0), MTX_MULTVECARRAY_3 );
    ASSERTMSG( (count > 1),    MTX_MULTVECARRAY_4 );

    for(i=0; i< count; i++)
    {
        // Vec has a 4th implicit 'w' coordinate of 1
        vTmp.x = m[0][0]*srcBase->x + m[1][0]*srcBase->y + m[2][0]*srcBase->z + m[3][0];
        vTmp.y = m[0][1]*srcBase->x + m[1][1]*srcBase->y + m[2][1]*srcBase->z + m[3][1];
        vTmp.z = m[0][2]*srcBase->x + m[1][2]*srcBase->y + m[2][2]*srcBase->z + m[3][2];

        // copy back
        dstBase->x = vTmp.x;
        dstBase->y = vTmp.y;
        dstBase->z = vTmp.z;

        srcBase++;
        dstBase++;
    }
}
/*===========================================================================*/
