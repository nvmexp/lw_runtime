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
//static const f32x2 c01 = {0.0F, 1.0F};
//static const f32x2 c10 = {1.0F, 0.0F};
static const f32x2 c11 = {1.0F, 1.0F};
static const f32x2 c22 = {2.0F, 2.0F};
static const f32x2 c33 = {3.0F, 3.0F};
static const f32x2 c0505 = {0.5F, 0.5F};
static const f32x2    epsilon = {QUAT_EPSILON, QUAT_EPSILON};

/*---------------------------------------------------------------------------*
  Name:         QUATAdd

  Description:  Returns the sum of two quaternions.

  Arguments:    p - first quaternion
                q - second quaternion
                r - resultant quaternion p+q

  Returns:      none
 *---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------------*/
void C_QUATAdd( const Quaternion *p, const Quaternion *q, Quaternion *r )
{
    ASSERTMSG( ( p != 0 ), QUAT_ADD_1 );
    ASSERTMSG( ( q != 0 ), QUAT_ADD_2 );
    ASSERTMSG( ( r != 0 ), QUAT_ADD_3 );

    r->x = p->x + q->x;
    r->y = p->y + q->y;
    r->z = p->z + q->z;
    r->w = p->w + q->w;
}

#if defined(CAFE)
/*---------------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------------*/
void PSQUATAdd( const Quaternion *p, const Quaternion *q, Quaternion *r )
{
    f32x2    pxy, qxy, rxy, pzw, qzw, rzw;

    //psq_l     pxy,  0(p), 0, 0
    //pxy[0] = p->x;
    //pxy[1] = p->y;
    pxy = __PSQ_LX(p, 0, 0, 0);

    //psq_l     qxy,  0(q), 0, 0
    //qxy[0] = q->x;
    //qxy[1] = q->y;
    qxy = __PSQ_LX(q, 0, 0, 0);

    //ps_add  rxy, pxy, qxy
    rxy = __PS_ADD(pxy, qxy);

    //psq_st    rxy,  0(r), 0, 0
    //r->x = rxy[0];
    //r->y = rxy[1];
    __PSQ_STX(r, 0, rxy, 0, 0);

    //psq_l     pzw,  8(p), 0, 0
    //pzw[0] = p->z;
    //pzw[1] = p->w;
    pzw = __PSQ_LX(p, 8, 0, 0);

    //psq_l     qzw,  8(q), 0, 0
    //qzw[0] = q->z;
    //qzw[1] = q->w;
    qzw = __PSQ_LX(q, 8, 0, 0);

    //ps_add  rzw, pzw, qzw
    rzw = __PS_ADD(pzw, qzw);

    //psq_st    rzw,  8(r), 0, 0
    //r->z = rzw[0];
    //r->w = rzw[1];
    __PSQ_STX(r, 8, rzw, 0, 0);
}
#endif

/*---------------------------------------------------------------------------*
  Name:         QUATSubtract

  Description:  Returns the difference of two quaternions p-q.

  Arguments:    p - left quaternion
                q - right quaternion
                r - resultant quaternion difference p-q

  Returns:      none
 *---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------------*/
void C_QUATSubtract( const Quaternion *p, const Quaternion *q, Quaternion *r )
{
    ASSERTMSG( ( p != 0 ), QUAT_SUBTRACT_1 );
    ASSERTMSG( ( q != 0 ), QUAT_SUBTRACT_2 );
    ASSERTMSG( ( r != 0 ), QUAT_SUBTRACT_3 );

    r->x = p->x - q->x;
    r->y = p->y - q->y;
    r->z = p->z - q->z;
    r->w = p->w - q->w;
}

#if defined(CAFE)
/*---------------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------------*/
void PSQUATSubtract( const Quaternion *p, const Quaternion *q, Quaternion *r )
{
    f32x2    pxy, qxy, rxy, pzw, qzw, rzw;

    //psq_l     pxy,  0(p), 0, 0
    //pxy[0] = p->x;
    //pxy[1] = p->y;
    pxy = __PSQ_LX(p, 0, 0, 0);

    //psq_l     qxy,  0(q), 0, 0
    //qxy[0] = q->x;
    //qxy[1] = q->y;
    qxy = __PSQ_LX(q, 0, 0, 0);

    //ps_sub  rxy, pxy, qxy
    rxy = __PS_SUB(pxy, qxy);

    //psq_st    rxy,  0(r), 0, 0
    //r->x = rxy[0];
    //r->y = rxy[1];
    __PSQ_STX(r, 0, rxy, 0, 0);

    //psq_l     pzw,  8(p), 0, 0
    //pzw[0] = p->z;
    //pzw[1] = p->w;
    pzw = __PSQ_LX(p, 8, 0, 0);

    //psq_l     qzw,  8(q), 0, 0
    //qzw[0] = q->z;
    //qzw[1] = q->w;
    qzw = __PSQ_LX(q, 8, 0, 0);

    //ps_sub  rzw, pzw, qzw
    rzw = __PS_SUB(pzw, qzw);

    //psq_st    rzw,  8(r), 0, 0
    //r->z = rzw[0];
    //r->w = rzw[1];
    __PSQ_STX(r, 8, rzw, 0, 0);

}
#endif

/*---------------------------------------------------------------------------*
  Name:         QUATMultiply

  Description:  Returns the product of two quaternions. The order of
                multiplication is important. (p*q != q*p)

  Arguments:    p - left quaternion
                q - right quaternion
                pq - resultant quaternion product p*q

  Returns:      none
 *---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------------*/
void C_QUATMultiply( const Quaternion *p, const Quaternion *q, Quaternion *pq )
{
    Quaternion *r;
    Quaternion  pqTmp;

    ASSERTMSG( ( p  != 0 ), QUAT_MULTIPLY_1 );
    ASSERTMSG( ( q  != 0 ), QUAT_MULTIPLY_2 );
    ASSERTMSG( ( pq != 0 ), QUAT_MULTIPLY_3 );

    if ( p == pq || q == pq )
    {
        r = &pqTmp;
    }
    else
    {
        r = pq;
    }

    r->w = p->w*q->w - p->x*q->x - p->y*q->y - p->z*q->z;
    r->x = p->w*q->x + p->x*q->w + p->y*q->z - p->z*q->y;
    r->y = p->w*q->y + p->y*q->w + p->z*q->x - p->x*q->z;
    r->z = p->w*q->z + p->z*q->w + p->x*q->y - p->y*q->x;

    if ( r == &pqTmp )
    {
        *pq = pqTmp;
    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------------*/

void PSQUATMultiply( const Quaternion *p, const Quaternion *q, Quaternion *pq )
{
    f32x2    pxy, pzw, qxy, qzw;
    f32x2    pnxy, pnzw, pnxny, pnznw;
    f32x2    rxy, rzw, sxy, szw;

    // [px][py] : Load
    //psq_l       pxy, 0(p), 0, 0
    //pxy[0] = p->x;
    //pxy[1] = p->y;
    pxy = __PSQ_LX(p, 0, 0, 0);

    // [pz][pw] : Load
    //psq_l       pzw, 8(p), 0, 0
    //pzw[0] = p->z;
    //pzw[1] = p->w;
    pzw = __PSQ_LX(p, 8, 0, 0);

    // [qx][qy] : Load
    //psq_l       qxy, 0(q), 0, 0
    //qxy[0] = q->x;
    //qxy[1] = q->y;
    qxy = __PSQ_LX(q, 0, 0, 0);

    // [-px][-py]
    //ps_neg      pnxny, pxy
    pnxny = __PS_NEG(pxy);

    // [qz][qw] : Load
    //psq_l       qzw, 8(q), 0, 0
    //qzw[0] = q->z;
    //qzw[1] = q->w;
    qzw = __PSQ_LX(q, 8, 0, 0);

    // [-pz][-pw]
    //ps_neg      pnznw, pzw
    pnznw = __PS_NEG(pzw);

    // [-px][py]
    //ps_merge01  pnxy, pnxny, pxy
    pnxy = __PS_MERGE01(pnxny, pxy);

    // [pz*qx][pw*qx]
    //ps_muls0    rxy, pzw, qxy
    rxy = __PS_MULS0(pzw, qxy);

    // [-px*qx][-py*qx]
    //ps_muls0    rzw, pnxny, qxy
    rzw = __PS_MULS0(pnxny, qxy);

    // [-pz][pw]
    //ps_merge01  pnzw, pnznw, pzw
    pnzw = __PS_MERGE01(pnznw, pzw);

    // [-px*qy][py*qy]
    //ps_muls1    szw, pnxy, qxy
    szw = __PS_MULS1(pnxy, qxy);

    // [pz*qx-px*qz][pw*qx+py*qz]
    //ps_madds0   rxy, pnxy, qzw, rxy
    rxy = __PS_MADDS0(pnxy, qzw, rxy);

    // [-pz*qy][pw*qy]
    //ps_muls1    sxy, pnzw, qxy
    sxy = __PS_MULS1(pnzw, qxy);

    // [-px*qx-pz*qz][-py*qx+pw*qz]
    //ps_madds0   rzw, pnzw, qzw, rzw
    rzw = __PS_MADDS0(pnzw, qzw, rzw);

    // [-px*qy-pz*qw][py*qy-pw*qw]
    //ps_madds1   szw, pnznw, qzw, szw
    szw = __PS_MADDS1(pnznw, qzw, szw);

    // [pw*qx+py*qz][pz*qx-px*qz]
    //ps_merge10  rxy, rxy, rxy
    rxy = __PS_MERGE10(rxy, rxy);

    // [-pz*qy+px*qw][pw*qy+py*qw]
    //ps_madds1   sxy, pxy, qzw, sxy
    sxy = __PS_MADDS1(pxy, qzw, sxy);

    // [-py*qx+pw*qz][-px*qx-pz*qz]
    //ps_merge10  rzw, rzw, rzw
    rzw = __PS_MERGE10(rzw, rzw);

    // [pw*qx+py*qz-pz*qy+px*qw][pz*qx-px*qz+pw*qy+py*qw] : [pqx][pqy]
    //ps_add      rxy, rxy, sxy
    rxy = __PS_ADD(rxy, sxy);

    // [pqx][pqy] : Store
    //psq_st      rxy, 0(pq), 0, 0
    //pq->x = rxy[0];
    //pq->y = rxy[1];
    __PSQ_STX(pq, 0, rxy, 0, 0);

    // [-py*qx+pw*qz+px*qy+pz*qw][-px*qx-pz*qz-py*qy+pw*qw] : [pqz][pqw]
    //ps_sub      rzw, rzw, szw
    rzw = __PS_SUB(rzw, szw);

    // [pqz][pqw] : Store
    //psq_st      rzw, 8(pq), 0, 0
    //pq->z = rzw[0];
    //pq->w = rzw[1];
    __PSQ_STX(pq, 8, rzw, 0, 0);
}
#endif

/*---------------------------------------------------------------------------*
  Name:         QUATScale

  Description:  Scales a quaternion.

  Arguments:    q     - quaternion
                r     - resultant scaled quaternion
                scale - float to scale by

  Returns:      none
 *---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------------*/
void C_QUATScale( const Quaternion *q, Quaternion *r, f32 scale )
{
    ASSERTMSG( ( q  != 0 ), QUAT_SCALE_1 );
    ASSERTMSG( ( r  != 0 ), QUAT_SCALE_2 );

    r->x = q->x * scale;
    r->y = q->y * scale;
    r->z = q->z * scale;
    r->w = q->w * scale;
}

#if defined(CAFE)
/*---------------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------------*/

void PSQUATScale( const Quaternion *q, Quaternion *r, f32 scale )
{
    f32x2    rxy, rzw;
    f32x2    scale2 = {scale, scale};

    //psq_l       rxy, 0(q), 0, 0
    //rxy[0] = q->x;
    //rxy[1] = q->y;
    rxy = __PSQ_LX(q, 0, 0, 0);

    //psq_l       rzw, 8(q), 0, 0
    //rzw[0] = q->z;
    //rzw[1] = q->w;
    rzw = __PSQ_LX(q, 8, 0, 0);

    //ps_muls0    rxy, rxy, scale
    rxy = __PS_MULS0(rxy, scale2);

    //psq_st      rxy, 0(r), 0, 0
    //r->x = rxy[0];
    //r->y = rxy[1];
    __PSQ_STX(r, 0, rxy, 0, 0);

    //ps_muls0    rzw, rzw, scale
    rzw = __PS_MULS0(rzw, scale2);

    //psq_st      rzw, 8(r), 0, 0
    //r->z = rzw[0];
    //r->w = rzw[1];
    __PSQ_STX(r, 8, rzw, 0, 0);
}
#endif

/*---------------------------------------------------------------------------*
  Name:         QUATDotProduct

  Description:  Returns the dot product of the two quaternions.

  Arguments:    p - first quaternion
                q - second quaternion

  Returns:      Dot product
 *---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------------*/
f32 C_QUATDotProduct( const Quaternion *p, const Quaternion *q )
{
    ASSERTMSG( ( p  != 0 ), QUAT_DOTPRODUCT_1 );
    ASSERTMSG( ( q  != 0 ), QUAT_DOTPRODUCT_2 );

    return (q->x*p->x + q->y*p->y + q->z*p->z + q->w*p->w);
}

#if defined(CAFE)
/*---------------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------------*/

f32 PSQUATDotProduct( const Quaternion *p, const Quaternion *q )
{
    f32x2    pxy, pzw, qxy, qzw, dp;

    //psq_l       pxy, 0(p), 0, 0
    //pxy[0] = p->x;
    //pxy[1] = p->y;
    pxy = __PSQ_LX(p, 0, 0, 0);

    //psq_l       qxy, 0(q), 0, 0
    //qxy[0] = q->x;
    //qxy[1] = q->y;
    qxy = __PSQ_LX(q, 0, 0, 0);

    //ps_mul      dp, pxy, qxy
    dp = __PS_MUL(pxy, qxy);

    //psq_l       pzw, 8(p), 0, 0
    //pzw[0] = p->z;
    //pzw[1] = p->w;
    pzw = __PSQ_LX(p, 8, 0, 0);

    //psq_l       qzw, 8(q), 0, 0
    //qzw[0] = q->z;
    //qzw[1] = q->w;
    qzw = __PSQ_LX(q, 8, 0, 0);

    //ps_madd     dp, pzw, qzw, dp
    dp = __PS_MADD(pzw, qzw, dp);

    //ps_sum0     dp, dp, dp, dp
    dp = __PS_SUM0(dp, dp, dp);

    return (f32)dp[0];
}
#endif

/*---------------------------------------------------------------------------*
  Name:         QUATNormalize

  Description:  Normalizes a quaternion

  Arguments:    src - the source quaternion
                unit - resulting unit quaternion

  Returns:      none
 *---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------------*/
void C_QUATNormalize( const Quaternion *src, Quaternion *unit )
{
    f32 mag;

    ASSERTMSG( ( src  != 0 ), QUAT_NORMALIZE_1 );
    ASSERTMSG( ( unit != 0 ), QUAT_NORMALIZE_2 );

    mag = (src->x * src->x) + (src->y * src->y) + (src->z * src->z) + (src->w * src->w);

    if ( mag >= QUAT_EPSILON )
    {
        mag = 1.0F / sqrtf(mag);

        unit->x = src->x * mag;
        unit->y = src->y * mag;
        unit->z = src->z * mag;
        unit->w = src->w * mag;
    }
    else
    {
        unit->x = unit->y = unit->z = unit->w = 0.0F;
    }
}

#if defined(CAFE)
/*---------------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------------*/

void PSQUATNormalize( const Quaternion *src, Quaternion *unit )
{
    f32x2    sxy, szw;
    f32x2    mag, rsqmag, diff;
    f32x2    nwork0, nwork1;

    //psq_l       sxy, 0(src), 0, 0
    //sxy[0] = src->x;
    //sxy[1] = src->y;
    sxy = __PSQ_LX(src, 0, 0, 0);

    // mag = [x*x][y*y]
    //ps_mul      mag, sxy, sxy
    mag = __PS_MUL(sxy, sxy);

    //psq_l       szw, 8(src), 0, 0
    //szw[0] = src->z;
    //szw[1] = src->w;
    szw = __PSQ_LX(src, 8, 0, 0);

    // c00 = [0.0F]
    //ps_sub      c00, epsilon, epsilon
    // mag = [x*x+z*z][y*y+w*w]
    //ps_madd     mag, szw, szw, mag
    mag = __PS_MADD(szw, szw, mag);

    // mag = [x*x+y*y+z*z+w*w][N/A]
    //ps_sum0     mag, mag, mag, mag
    mag = __PS_SUM0(mag, mag, mag);

    // rsqmag = 1.0F / sqrtf(mag) : estimation
    //frsqrte     rsqmag, mag
    rsqmag = __PS_RSQRTE(mag);

    // diff = mag - epsilon
    //ps_sub      diff, mag, epsilon
    diff = __PS_SUB(mag, epsilon);

    // Newton-Rapson refinement (x1) : E' = (E/2)(3 - X * E * E)
    //fmul        nwork0, rsqmag, rsqmag
    nwork0 = __PS_MUL(rsqmag, rsqmag);

    //fmul        nwork1, rsqmag, c0505
    nwork1 = __PS_MUL(rsqmag, c0505);

    //fnmsub      nwork0, nwork0, mag, c33
    nwork0 = __PS_NMSUB(nwork0, mag, c33);

    //fmul        rsqmag, nwork0, nwork1
    rsqmag = __PS_MUL(nwork0, nwork1);

    // rsqmag = ( mag >= epsilon ) ? rsqmag : 0
    //ps_sel      rsqmag, diff, rsqmag, c00
    rsqmag = __PS_SEL(diff, rsqmag, c00);

    // sxy = [x*rsqmag][y*rsqmag]
    //ps_muls0    sxy, sxy, rsqmag
    sxy = __PS_MULS0(sxy, rsqmag);

    // szw = [z*rsqmag][w*rsqmag]
    //ps_muls0    szw, szw, rsqmag
    szw = __PS_MULS0(szw, rsqmag);

    //psq_st      sxy, 0(unit), 0, 0
    //unit->x = sxy[0];
    //unit->y = sxy[1];
    __PSQ_STX(unit, 0, sxy, 0, 0);

    //psq_st      szw, 8(unit), 0, 0
    //unit->z = szw[0];
    //unit->w = szw[1];
    __PSQ_STX(unit, 8, szw, 0, 0);
}
#endif

/*---------------------------------------------------------------------------*
  Name:         QUATIlwerse

  Description:  Returns the ilwerse of the quaternion.

  Arguments:    src - the source quaternion
                ilw - resulting ilwerse quaternion

  Returns:      none
 *---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------------*/
void C_QUATIlwerse( const Quaternion *src, Quaternion *ilw )
{
    f32 mag, normilw;

    ASSERTMSG( ( src != 0 ), QUAT_ILWERSE_1 );
    ASSERTMSG( ( ilw != 0 ), QUAT_ILWERSE_2 );

    mag = ( src->x*src->x + src->y*src->y + src->z*src->z + src->w*src->w );

    if ( mag == 0.0f )
    {
        mag = 1.0f;
    }

    // [Ilwerse] = [Conjugate] / [Magnitude]
    normilw = 1.0f / mag;
    ilw->x = -src->x * normilw;
    ilw->y = -src->y * normilw;
    ilw->z = -src->z * normilw;
    ilw->w =  src->w * normilw;
}

#if defined(CAFE)
/*---------------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------------*/

void PSQUATIlwerse( const Quaternion *src, Quaternion *ilw )
{
    f32x2    sxy, szw, izz, iww;
    f32x2    mag, normilw, nnilw, nwork0;

    // load xy
    //psq_l       sxy, 0(src), 0, 0
    //sxy[0] = src->x;
    //sxy[1] = src->y;
    sxy = __PSQ_LX(src, 0, 0, 0);

    // mag = [x*x][y*y]
    //ps_mul      mag, sxy, sxy
    mag = __PS_MUL(sxy, sxy);

    // c00 = [0.0F]
    //ps_sub      c00, c11, c11

    // load zw
    //psq_l       szw, 8(src), 0, 0
    //szw[0] = src->z;
    //szw[1] = src->w;
    szw = __PSQ_LX(src, 8, 0, 0);

    // mag = [x*x+z*z][y*y+w*w]
    //ps_madd     mag, szw, szw, mag
    mag = __PS_MADD(szw, szw, mag);

    // c22 = [2.0F]
    //ps_add      c22, c11, c11

    // mag = [x*x+y*y+z*z+w*w][N/A]
    //ps_sum0     mag, mag, mag, mag
    mag = __PS_SUM0(mag, mag, mag);

    // zero check
    if ( mag[0] == 0.0f )
    {
        mag[0] = mag[1] = 1.0f;
    }

    // normilw = 1.0F / mag
    //fres        normilw, mag
    normilw = __PS_RES(mag);

    // Newton-Rapson refinment (x1) : E' = 2E-X*E*E
    //ps_nmsub    nwork0, mag, normilw, c22
    nwork0 = __PS_NMSUB(mag, normilw, c22);

    //ps_mul      normilw, normilw, nwork0
    normilw = __PS_MUL(normilw, nwork0);

    // nnilw = [ -normilw ]
    //ps_neg      nnilw, normilw
    nnilw = __PS_NEG(normilw);

    // iww = [ w*normilw ][ N/A ]
    //ps_muls1    iww, normilw, szw
    iww = __PS_MULS1(normilw, szw);

    // sxy = [ -x*normilw ][ -y*normilw ]
    //ps_muls0    sxy, sxy, nnilw
    sxy = __PS_MULS0(sxy, nnilw);

    // store w
    //psq_st      iww, 12(ilw), 1, 0
    //ilw->w = iww[0];
    __PSQ_STX(ilw, 12, iww, 1, 0);

    // izz = [ -z*normilw ][ N/A ]
    //ps_muls0    izz, szw, nnilw
    izz = __PS_MULS0(szw, nnilw);

    // store xy
    //psq_st      sxy, 0(ilw), 0, 0
    //ilw->x = sxy[0];
    //ilw->y = sxy[1];
    __PSQ_STX(ilw, 0, sxy, 0, 0);

    // store z
    //psq_st      izz, 8(ilw), 1, 0
    //ilw->z = izz[0];
    __PSQ_STX(ilw, 8, izz, 1, 0);
}
#endif

/*---------------------------------------------------------------------------*
  Name:         QUATDivide

  Description:  Returns the ratio of two quaternions.  Creates a result
                r = p/q such that q*r=p (order of multiplication is important).

  Arguments:    p - left quaternion
                q - right quaternion
                r - resultant ratio quaterion p/q

  Returns:      none
 *---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------------*/
void C_QUATDivide( const Quaternion *p, const Quaternion *q, Quaternion *r )
{
    Quaternion qtmp;

    ASSERTMSG( ( p != 0 ), QUAT_DIVIDE_1 );
    ASSERTMSG( ( q != 0 ), QUAT_DIVIDE_2 );
    ASSERTMSG( ( r != 0 ), QUAT_DIVIDE_3 );

    C_QUATIlwerse(q, &qtmp);
    C_QUATMultiply(&qtmp, p, r);
}

#if defined(CAFE)
/*---------------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------------*
            Note that this performs NO error checking.
 *---------------------------------------------------------------------------*/

void PSQUATDivide( const Quaternion *p, const Quaternion *q, Quaternion *r )
{
    Quaternion qtmp;

    PSQUATIlwerse(q, &qtmp);
    PSQUATMultiply(&qtmp, p, r);
}
#endif

/*---------------------------------------------------------------------------*
  Name:         QUATExp

  Description:  Exponentiate quaternion (where q.w == 0).

  Arguments:    q - pure quaternion
                r - resultant exponentiate quaternion (an unit quaternion)

  Returns:      none
 *---------------------------------------------------------------------------*/
void C_QUATExp( const Quaternion *q, Quaternion *r )
{
    f32 theta, scale;

    ASSERTMSG( ( q != 0 ), QUAT_EXP_1 );
    ASSERTMSG( ( r != 0 ), QUAT_EXP_2 );

    // pure quaternion check
    ASSERTMSG( ( q->w == 0.0F ), QUAT_EXP_3 );

    theta = sqrtf( q->x*q->x + q->y*q->y + q->z*q->z );
    scale = 1.0F;

    if ( theta > QUAT_EPSILON )
        scale = (f32)sinf(theta)/theta;

    r->x = scale * q->x;
    r->y = scale * q->y;
    r->z = scale * q->z;
    r->w = (f32)cosf(theta);
}


/*---------------------------------------------------------------------------*
  Name:         QUATLogN

  Description:  Returns the natural logarithm of a UNIT quaternion

  Arguments:    q - unit quaternion
                r - resultant logarithm quaternion (an pure quaternion)

  Returns:      none
 *---------------------------------------------------------------------------*/
void C_QUATLogN( const Quaternion *q, Quaternion *r )
{
    f32 theta,scale;

    ASSERTMSG( ( q != 0 ), QUAT_LOGN_1 );
    ASSERTMSG( ( r != 0 ), QUAT_LOGN_2 );

    scale = q->x*q->x + q->y*q->y + q->z*q->z;

    // unit quaternion check
#ifdef _DEBUG
    {
        f32 mag;
        mag = scale + q->z*q->z;
        if ( mag < 1.0F - QUAT_EPSILON || mag > 1.0F + QUAT_EPSILON )
        {
            ASSERTMSG(false, QUAT_LOGN_3 );
        }
    }
#endif

    scale = sqrtf(scale);
    theta = atan2f( scale, q->w );

    if ( scale > 0.0F )
        scale = theta/scale;

    r->x = scale*q->x;
    r->y = scale*q->y;
    r->z = scale*q->z;
    r->w = 0.0F;

}


/*---------------------------------------------------------------------------*
  Name:         QUATMakeClosest

  Description:  Modify q so it is on the same side of the hypersphere as qto

  Arguments:    q   - quaternion
                qto - quaternion to be close to
                r   - resultant modified quaternion

  Returns:      NONE
 *---------------------------------------------------------------------------*/
void C_QUATMakeClosest( const Quaternion *q, const Quaternion *qto, Quaternion *r )
{
    f32 dot;

    ASSERTMSG( ( q   != 0 ), QUAT_MAKECLOSEST_1 );
    ASSERTMSG( ( qto != 0 ), QUAT_MAKECLOSEST_2 );
    ASSERTMSG( ( r   != 0 ), QUAT_MAKECLOSEST_3 );

    dot =  q->x*qto->x + q->y*qto->y + q->z*qto->z + q->w*qto->w;

    if ( dot < 0.0f )
    {
        r->x = -q->x;
        r->y = -q->y;
        r->z = -q->z;
        r->w = -q->w;
    }
    else
    {
        *r = *q;
    }
}


/*---------------------------------------------------------------------------*
  Name:         QUATRotAxisRad

  Description:  Returns rotation quaternion about arbitrary axis.

  Arguments:    r    - resultant rotation quaternion
                axis - rotation axis
                rad  - rotation angle in radians

  Returns:      NONE
 *---------------------------------------------------------------------------*/
void C_QUATRotAxisRad( Quaternion *r, const Vec *axis, f32 rad )
{
    f32 half, sh, ch;
    Vec nAxis;

    ASSERTMSG( ( r    != 0 ), QUAT_ROTAXISRAD_1 );
    ASSERTMSG( ( axis != 0 ), QUAT_ROTAXISRAD_2 );

    VECNormalize(axis, &nAxis);

    half = rad * 0.50F;
    sh   = sinf(half);
    ch   = cosf(half);

    r->x = sh * nAxis.x;
    r->y = sh * nAxis.y;
    r->z = sh * nAxis.z;
    r->w = ch;
}


/*---------------------------------------------------------------------------*
  Name:         QUATMtx

  Description:  Colwerts a matrix to a unit quaternion.

  Arguments:    r   - result quaternion
                m   - the matrix

  Returns:      NONE
 *---------------------------------------------------------------------------*/
void C_QUATMtx( Quaternion *r, MTX_CONST Mtx m )
{
    f32 tr,s;
    s32 i,j,k;
    s32 nxt[3] = {1,2,0};
    f32 q[3];

    ASSERTMSG( ( r != 0 ), QUAT_MTX_1 );
    ASSERTMSG( ( m != 0 ), QUAT_MTX_2 );

    tr = m[0][0] + m[1][1] + m[2][2];
    if( tr > 0.0f )
    {
        s = (f32)sqrtf(tr + 1.0f);
        r->w = s * 0.5f;
        s = 0.5f / s;
        r->x = (m[2][1] - m[1][2]) * s;
        r->y = (m[0][2] - m[2][0]) * s;
        r->z = (m[1][0] - m[0][1]) * s;
    }
    else
    {
        i = 0;
        if (m[1][1] > m[0][0]) i = 1;
        if (m[2][2] > m[i][i]) i = 2;
        j = nxt[i];
        k = nxt[j];
        s = (f32)sqrtf( (m[i][i] - (m[j][j] + m[k][k])) + 1.0f );
        q[i] = s * 0.5f;

        if (s!=0.0f)
            s = 0.5f / s;

        r->w = (m[k][j] - m[j][k]) * s;
        q[j] = (m[i][j] + m[j][i]) * s;
        q[k] = (m[i][k] + m[k][i]) * s;

        r->x = q[0];
        r->y = q[1];
        r->z = q[2];
    }
}


/*---------------------------------------------------------------------------*
  Name:         QUATLerp

  Description:  Linear interpolation of two quaternions.

  Arguments:    p - first quaternion
                q - second quaternion
                r - resultant interpolated quaternion
                t - interpolation parameter

  Returns:      NONE
 *---------------------------------------------------------------------------*/
void C_QUATLerp( const Quaternion *p, const Quaternion *q, Quaternion *r, f32 t )
{
    ASSERTMSG( ( p != 0 ), QUAT_LERP_1 );
    ASSERTMSG( ( q != 0 ), QUAT_LERP_2 );
    ASSERTMSG( ( r != 0 ), QUAT_LERP_3 );

    r->x = t * ( q->x - p->x ) + p->x;
    r->y = t * ( q->y - p->y ) + p->y;
    r->z = t * ( q->z - p->z ) + p->z;
    r->w = t * ( q->w - p->w ) + p->w;
}


/*---------------------------------------------------------------------------*
  Name:         QUATSlerp

  Description:  Spherical linear interpolation of two quaternions.

  Arguments:    p - first quaternion
                q - second quaternion
                r - resultant interpolated quaternion
                t - interpolation parameter

  Returns:      NONE
 *---------------------------------------------------------------------------*/
void C_QUATSlerp( const Quaternion *p, const Quaternion *q, Quaternion *r, f32 t )
{
    f32 theta, sin_th, cos_th, tp, tq;

    ASSERTMSG( ( p != 0 ), QUAT_SLERP_1 );
    ASSERTMSG( ( q != 0 ), QUAT_SLERP_2 );
    ASSERTMSG( ( r != 0 ), QUAT_SLERP_3 );

    cos_th = p->x * q->x + p->y * q->y + p->z * q->z + p->w * q->w;
    tq     = 1.0F;

    if ( cos_th < 0.0F )
    {
        cos_th = -cos_th;
        tq     = -tq;
    }

    if ( cos_th <= 1.0F - QUAT_EPSILON )
    {
        theta  = acosf(cos_th);
        sin_th = sinf(theta);
        tp     = sinf((1.0F - t) * theta) / sin_th;
        tq    *= sinf( t * theta ) / sin_th;
    }
    else
    {
        // cos(theta) is close to 1.0F -> linear
        tp = 1.0F - t;
        tq = tq * t;
    }

    r->x = tp * p->x + tq * q->x;
    r->y = tp * p->y + tq * q->y;
    r->z = tp * p->z + tq * q->z;
    r->w = tp * p->w + tq * q->w;

}


/*---------------------------------------------------------------------------*
  Name:         QUATSquad

  Description:  Spherical cubic quadrangle interpolation of two quaternions
                with derrived inner-quadrangle quaternions.

                This will be used with the function QUATCompA().

  Arguments:    p - first quaternion
                a - derrived inner-quadrangle quaternion
                b - derrived inner-quadrangle quaternion
                q - second quaternion
                r - resultant interpolated quaternion
                t - interpolation parameter

  Returns:      NONE
 *---------------------------------------------------------------------------*/
void C_QUATSquad( const Quaternion *p, const Quaternion *a, const Quaternion *b,
                  const Quaternion *q, Quaternion *r, f32 t )
{
    Quaternion pq, ab;
    f32 t2;

    ASSERTMSG( ( p != 0 ), QUAT_SQUAD_1 );
    ASSERTMSG( ( a != 0 ), QUAT_SQUAD_2 );
    ASSERTMSG( ( b != 0 ), QUAT_SQUAD_3 );
    ASSERTMSG( ( q != 0 ), QUAT_SQUAD_4 );
    ASSERTMSG( ( r != 0 ), QUAT_SQUAD_5 );

    t2 = 2 * t * ( 1.0F - t );
    C_QUATSlerp(p, q, &pq, t);
    C_QUATSlerp(a, b, &ab, t);
    C_QUATSlerp(&pq, &ab, r, t2);
}


/*---------------------------------------------------------------------------*
  Name:         QUATCompA

  Description:  Compute a, the term used in Boehm-type interpolation
                a[n] = q[n]* qexp(-(1/4)*( ln(qilw(q[n])*q[n+1]) +
                                           ln( qilw(q[n])*q[n-1] )))

  Arguments:    qprev - previous quaternion
                q     - current quaternion
                qnext - next quaternion
                a     - result quaternion A

  Returns:      none
---------------------------------------------------------------------------*/
void C_QUATCompA( const Quaternion *qprev, const Quaternion *q, const Quaternion *qnext, Quaternion *a )
{
    Quaternion qm, qp, lqm, lqp, qpqm, exq;

    ASSERTMSG( ( qprev != 0 ), QUAT_COMPA_1 );
    ASSERTMSG( ( q     != 0 ), QUAT_COMPA_2 );
    ASSERTMSG( ( qnext != 0 ), QUAT_COMPA_3 );
    ASSERTMSG( ( a     != 0 ), QUAT_COMPA_4 );

    C_QUATDivide(qprev, q, &qm);
    C_QUATLogN(&qm, &lqm);
    C_QUATDivide(qnext, q, &qp);
    C_QUATLogN(&qp, &lqp);

    C_QUATAdd(&lqp, &lqm, &qpqm);
    C_QUATScale(&qpqm, &qpqm, -0.25F);

    C_QUATExp(&qpqm, &exq);
    C_QUATMultiply(q, &exq, a);
}

/*---------------------------------------------------------------------*

Name:            MTXQuat

Description:     sets a rotation matrix from a quaternion.


Arguments:       m        matrix to be set

                 q        ptr to quaternion.

Return:          none

 *---------------------------------------------------------------------*/
/*---------------------------------------------------------------------*
    C version
 *---------------------------------------------------------------------*/
void C_MTXQuat ( Mtx m, const Quaternion *q )
{
    f32 s,xs,ys,zs,wx,wy,wz,xx,xy,xz,yy,yz,zz;

    ASSERTMSG( (m != 0),                         MTX_QUAT_1     );
    ASSERTMSG( (q != 0),                         MTX_QUAT_2     );
    ASSERTMSG( ( q->x || q->y || q->z || q->w ), MTX_QUAT_3     );

    s = 2.0f / ( (q->x * q->x) + (q->y * q->y) + (q->z * q->z) + (q->w * q->w) );

    xs = q->x *  s;     ys = q->y *  s;  zs = q->z *  s;
    wx = q->w * xs;     wy = q->w * ys;  wz = q->w * zs;
    xx = q->x * xs;     xy = q->x * ys;  xz = q->x * zs;
    yy = q->y * ys;     yz = q->y * zs;  zz = q->z * zs;

    m[0][0] = 1.0f - (yy + zz);
    m[0][1] = xy   - wz;
    m[0][2] = xz   + wy;
    m[0][3] = 0.0f;

    m[1][0] = xy   + wz;
    m[1][1] = 1.0f - (xx + zz);
    m[1][2] = yz   - wx;
    m[1][3] = 0.0f;

    m[2][0] = xz   - wy;
    m[2][1] = yz   + wx;
    m[2][2] = 1.0f - (xx + yy);
    m[2][3] = 0.0f;
}

#if defined(CAFE)
/*---------------------------------------------------------------------*
    Paired-Single intrinsics version
 *---------------------------------------------------------------------*
                Note that this performs NO error checking.
 *---------------------------------------------------------------------*/
void PSMTXQuat ( Mtx m, const Quaternion *q )
{
    f32x2    scale;
    f32x2    V_XY;
    f32x2    V_ZW;
    f32x2    V_YX;

    f32x2    D0;
    f32x2    D1;
    f32x2    tmp2, tmp3, tmp4; //tmp1, 
    f32x2    tmp5, tmp6, tmp7, tmp8, tmp9;

    // V_XY = [qx][qy] : LOAD
    //V_XY[0] = q->x;
    //V_XY[1] = q->y;
    V_XY = __PSQ_LX(q, 0, 0, 0);

    // V_ZW = [qz][qw] : LOAD
    //V_ZW[0] = q->z;
    //V_ZW[1] = q->w;
    V_ZW = __PSQ_LX(q, 8, 0, 0);

    // tmp2 = [qx*qx][qy*qy]
    tmp2 = __PS_MUL(V_XY, V_XY);

    // V_YX = [qy][qx]
    V_YX = __PS_MERGE10(V_XY, V_XY);

    // tmp4 = [qx*qx+qz*qz][qy*qy+qw*qw]
    tmp4 = __PS_MADD(V_ZW, V_ZW, tmp2);

    // tmp3 = [qz*qz][qw*qw]
    tmp3 = __PS_MUL(V_ZW, V_ZW);

    // scale = [qx*qx+qy*qy+qz*qz+qw*qw]
    scale = __PS_SUM0(tmp4, tmp4, tmp4);
    scale[1] = scale[0];

    // tmp7 = [qy*qw][qx*qw]
    tmp7 = __PS_MULS1(V_YX, V_ZW);

    // Newton-Rapson refinment (1/X) : E' = 2E-X*E*E
    // tmp9 = [E = Est.(1/X)]
    tmp9 = __PS_RES(scale);

    // tmp4 = [qx*qx+qz*qz][qy*qy+qz*qz]
    tmp4 = __PS_SUM1(tmp3, tmp4, tmp2);

    // scale = [2-X*E]
    scale = __PS_NMSUB(scale, tmp9, c22);

    // tmp6 = [qz*qw][?]
    tmp6 = __PS_MULS1(V_ZW, V_ZW);

    // scale = [E(2-scale*E) = E']
    scale = __PS_MUL(tmp9, scale);

    // tmp2 = [qx*qx+qy*qy]
    tmp2 = __PS_SUM0(tmp2, tmp2, tmp2);

    // scale = [s = 2E' = 2.0F/(qx*qx+qy*qy+qz*qz+qw*qw)]
    scale = __PS_MUL(scale, c22);

    // tmp8 = [qx*qy+qz*qw][?]
    tmp8 = __PS_MADD(V_XY, V_YX, tmp6);

    // tmp6 = [qx*qy-qz*qw][?]
    tmp6 = __PS_MSUB(V_XY, V_YX, tmp6);

    // c00 [m03] : STORE
    //m[0][3] = c00[0];
    __PSQ_STX(m, 12, c00, 1, 0);

    // tmp2 = [1-s(qx*qx+qy*qy)]   : [m22]
    tmp2 = __PS_NMSUB(tmp2, scale, c11);

    // tmp4 = [1-s(qx*qx+qz*qz)][1-s(qy*qy+qz*qz)] : [m11][m00]
    tmp4 = __PS_NMSUB(tmp4, scale, c11);

    // c00 [m23] : STORE
    //m[2][3] = c00[0];
    __PSQ_STX(m, 44, c00, 1, 0);

    // tmp8 = [s(qx*qy+qz*qw)][?]  : [m10]
    tmp8 = __PS_MUL(tmp8, scale);

    // tmp6 = [s(qx*qy-qz*qw)][?]  : [m01]
    tmp6  = __PS_MUL(tmp6, scale);

    // tmp2 [m22] : STORE
    //m[2][2] = tmp2[0];
    __PSQ_STX(m, 40, tmp2, 1, 0);

    // tmp5 = [qx*qz+qy*qw][qy*qz+qx*qw]
    tmp5 = __PS_MADDS0(V_XY, V_ZW, tmp7);

    // D1 = [m10][m11]
    D1 = __PS_MERGE00(tmp8, tmp4);

    // tmp7 = [qx*qz-qy*qw][qy*qz-qx*qw]
    tmp7 = __PS_NMSUB(tmp7, c22, tmp5);

    // D1 [m10][m11] : STORE
    //m[1][0] = D1[0];
    //m[1][1] = D1[1];
    __PSQ_STX(m, 16, D1, 0, 0);

    // D0 = [m00][m01]
    D0 = __PS_MERGE10(tmp4, tmp6);

    // tmp5 = [s(qx*qz+qy*qw)][s(qy*qz+qx*qw)] : [m02][m21]
    tmp5 = __PS_MUL(tmp5, scale);

    // tmp7 = [s(qx*qz-qy*qw)][s(qy*qz-qx*qw)] : [m20][m12]
    tmp7 = __PS_MUL(tmp7, scale);

    // D0 [m00][m01] : STORE
    //m[0][0] = D0[0];
    //m[0][1] = D0[1];
    __PSQ_STX(m, 0, D0, 0, 0);

    // tmp5 [m02] : STORE
    //m[0][2] = tmp5[0];
    __PSQ_STX(m, 8, tmp5, 1, 0);

    // tmp3 = [m12][m13]
    D1 = __PS_MERGE10(tmp7, c00);

    // tmp9 = [m20][m21]
    D0 = __PS_MERGE01(tmp7, tmp5);

    // tmp3 [m12][m13] : STORE
    //m[1][2] = D1[0];
    //m[1][3] = D1[1];
    __PSQ_STX(m, 24, D1, 0, 0);

    // tmp9 [m20][m21] : STORE
    //m[2][0] = D0[0];
    //m[2][1] = D0[1];
    __PSQ_STX(m, 32, D0, 0, 0);
}
#endif

/*===========================================================================*/
