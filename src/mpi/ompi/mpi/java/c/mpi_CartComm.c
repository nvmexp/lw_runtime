/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/*
 * This file is almost a complete re-write for Open MPI compared to the
 * original mpiJava package. Its license and copyright are listed below.
 * See <path to ompi/mpi/java/README> for more information.
 */
/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/
/*
 * File         : mpi_CartComm.c
 * Headerfile   : mpi_CartComm.h
 * Author       : Sung-Hoon Ko, Xinying Li
 * Created      : Thu Apr  9 12:22:15 1998
 * Revision     : $Revision: 1.6 $
 * Updated      : $Date: 2003/01/16 16:39:34 $
 * Copyright: Northeast Parallel Architectures Center
 *            at Syralwse University 1998
 */
#include "ompi_config.h"

#include <stdlib.h>
#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_CartComm.h"
#include "mpiJava.h"

JNIEXPORT void JNICALL Java_mpi_CartComm_init(JNIElw *elw, jclass clazz)
{
    ompi_java.CartParmsInit = (*elw)->GetMethodID(elw,
            ompi_java.CartParmsClass, "<init>", "([I[Z[I)V");

    ompi_java.ShiftParmsInit = (*elw)->GetMethodID(elw,
            ompi_java.ShiftParmsClass, "<init>", "(II)V");
}

JNIEXPORT jobject JNICALL Java_mpi_CartComm_getTopo(
        JNIElw *elw, jobject jthis, jlong comm)
{
    int maxdims;
    int rc = MPI_Cartdim_get((MPI_Comm)comm, &maxdims);

    if(ompi_java_exceptionCheck(elw, rc))
        return NULL;

    jintArray     dims    = (*elw)->NewIntArray(elw, maxdims);
    jbooleanArray periods = (*elw)->NewBooleanArray(elw, maxdims);
    jintArray     coords  = (*elw)->NewIntArray(elw, maxdims);

    if(maxdims != 0)
    {
        jint *jDims, *jCoords;
        jboolean *jPeriods;
        int  *cDims, *cCoords, *cPeriods;

        ompi_java_getIntArray(elw, dims, &jDims, &cDims);
        ompi_java_getIntArray(elw, coords, &jCoords, &cCoords);
        ompi_java_getBooleanArray(elw, periods, &jPeriods, &cPeriods);

        rc = MPI_Cart_get((MPI_Comm)comm, maxdims, cDims, cPeriods, cCoords);
        ompi_java_exceptionCheck(elw, rc);

        ompi_java_releaseIntArray(elw, dims, jDims, cDims);
        ompi_java_releaseIntArray(elw, coords, jCoords, cCoords);
        ompi_java_releaseBooleanArray(elw, periods, jPeriods, cPeriods);
    }

    return (*elw)->NewObject(elw, ompi_java.CartParmsClass,
                             ompi_java.CartParmsInit, dims, periods, coords);
}

JNIEXPORT jobject JNICALL Java_mpi_CartComm_shift(
        JNIElw *elw, jobject jthis, jlong comm, jint direction, jint disp)
{
    int sr, dr;
    int rc = MPI_Cart_shift((MPI_Comm)comm, direction, disp, &sr, &dr);
    ompi_java_exceptionCheck(elw, rc);

    return (*elw)->NewObject(elw, ompi_java.ShiftParmsClass,
                             ompi_java.ShiftParmsInit, sr, dr);
}

JNIEXPORT jintArray JNICALL Java_mpi_CartComm_getCoords(
        JNIElw *elw, jobject jthis, jlong comm, jint rank)
{
    int maxdims;
    int rc = MPI_Cartdim_get((MPI_Comm)comm, &maxdims);

    if(ompi_java_exceptionCheck(elw, rc))
        return NULL;

    jintArray coords = (*elw)->NewIntArray(elw, maxdims);
    jint *jCoords;
    int  *cCoords;
    ompi_java_getIntArray(elw, coords, &jCoords, &cCoords);

    rc = MPI_Cart_coords((MPI_Comm)comm, rank, maxdims, cCoords);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_releaseIntArray(elw, coords, jCoords, cCoords);
    return coords;
}

JNIEXPORT jint JNICALL Java_mpi_CartComm_map(
        JNIElw *elw, jobject jthis, jlong comm,
        jintArray dims, jbooleanArray periods)
{
    int nDims = (*elw)->GetArrayLength(elw, dims);
    jint *jDims;
    jboolean *jPeriods;
    int *cDims, *cPeriods;
    ompi_java_getIntArray(elw, dims, &jDims, &cDims);
    ompi_java_getBooleanArray(elw, periods, &jPeriods, &cPeriods);

    int newrank;
    int rc = MPI_Cart_map((MPI_Comm)comm, nDims, cDims, cPeriods, &newrank);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_forgetIntArray(elw, dims, jDims, cDims);
    ompi_java_forgetBooleanArray(elw, periods, jPeriods, cPeriods);
    return newrank;
}

JNIEXPORT jint JNICALL Java_mpi_CartComm_getRank(
        JNIElw *elw, jobject jthis, jlong comm, jintArray coords)
{
    jint *jCoords;
    int  *cCoords;
    ompi_java_getIntArray(elw, coords, &jCoords, &cCoords);

    int rank;
    int rc = MPI_Cart_rank((MPI_Comm)comm, cCoords, &rank);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_forgetIntArray(elw, coords, jCoords, cCoords);
    return rank;
}

JNIEXPORT jlong JNICALL Java_mpi_CartComm_sub(
        JNIElw *elw, jobject jthis, jlong comm, jbooleanArray remainDims)
{
    jboolean *jRemainDims;
    int      *cRemainDims;
    ompi_java_getBooleanArray(elw, remainDims, &jRemainDims, &cRemainDims);

    MPI_Comm newcomm;
    int rc = MPI_Cart_sub((MPI_Comm)comm, cRemainDims, &newcomm);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_forgetBooleanArray(elw, remainDims, jRemainDims, cRemainDims);
    return (jlong)newcomm;
}

JNIEXPORT void JNICALL Java_mpi_CartComm_createDims_1jni(
        JNIElw *elw, jclass jthis, jint nNodes, jintArray dims)
{
    int   nDims = (*elw)->GetArrayLength(elw, dims);
    jint *jDims;
    int  *cDims;
    ompi_java_getIntArray(elw, dims, &jDims, &cDims);

    int rc = MPI_Dims_create(nNodes, nDims, cDims);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseIntArray(elw, dims, jDims, cDims);
}
