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
 * File         : mpi_Group.c
 * Headerfile   : mpi_Group.h
 * Author       : Xinying Li
 * Created      : Thu Apr  9 12:22:15 1998
 * Revision     : $Revision: 1.3 $
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
#include "mpi_Group.h"
#include "mpiJava.h"

JNIEXPORT void JNICALL Java_mpi_Group_init(JNIElw *elw, jclass clazz)
{
    ompi_java_setStaticLongField(elw, clazz,
            "nullHandle", (jlong)MPI_GROUP_NULL);

    ompi_java.GroupHandle = (*elw)->GetFieldID(elw, clazz, "handle", "J");
}

JNIEXPORT jlong JNICALL Java_mpi_Group_getEmpty(JNIElw *elw, jclass clazz)
{
    return (jlong)MPI_GROUP_EMPTY;
}

JNIEXPORT jint JNICALL Java_mpi_Group_getSize(
        JNIElw *elw, jobject jthis, jlong group)
{
    int size, rc;
    rc = MPI_Group_size((MPI_Group)group, &size);
    ompi_java_exceptionCheck(elw, rc);
    return size;
}

JNIEXPORT jint JNICALL Java_mpi_Group_getRank(
        JNIElw *elw, jobject jthis, jlong group)
{
    int rank, rc;
    rc = MPI_Group_rank((MPI_Group)group, &rank);
    ompi_java_exceptionCheck(elw, rc);
    return rank;
}

JNIEXPORT jlong JNICALL Java_mpi_Group_free(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Group group = (MPI_Group)handle;
    int rc = MPI_Group_free(&group);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)group;
}

JNIEXPORT jintArray JNICALL Java_mpi_Group_translateRanks(
        JNIElw *elw, jclass jthis, jlong group1,
        jintArray ranks1, jlong group2)
{
    jsize n = (*elw)->GetArrayLength(elw, ranks1);
    jintArray ranks2 = (*elw)->NewIntArray(elw,n);
    jint *jRanks1, *jRanks2;
    int  *cRanks1, *cRanks2;
    ompi_java_getIntArray(elw, ranks1, &jRanks1, &cRanks1);
    ompi_java_getIntArray(elw, ranks2, &jRanks2, &cRanks2);

    int rc = MPI_Group_translate_ranks((MPI_Group)group1, n, cRanks1,
                                       (MPI_Group)group2, cRanks2);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_forgetIntArray(elw, ranks1, jRanks1, cRanks1);
    ompi_java_releaseIntArray(elw, ranks2, jRanks2, cRanks2);
    return ranks2;
}

JNIEXPORT jint JNICALL Java_mpi_Group_compare(
        JNIElw *elw, jclass jthis, jlong group1, jlong group2)
{
    int result, rc;
    rc = MPI_Group_compare((MPI_Group)group1, (MPI_Group)group2, &result);
    ompi_java_exceptionCheck(elw, rc);
    return result;
}

JNIEXPORT jlong JNICALL Java_mpi_Group_union(
        JNIElw *elw, jclass jthis, jlong group1, jlong group2)
{
    MPI_Group newGroup;
    int rc = MPI_Group_union((MPI_Group)group1, (MPI_Group)group2, &newGroup);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)newGroup;
}

JNIEXPORT jlong JNICALL Java_mpi_Group_intersection(
        JNIElw *elw, jclass jthis, jlong group1, jlong group2)
{
    MPI_Group newGroup;

    int rc = MPI_Group_intersection(
             (MPI_Group)group1, (MPI_Group)group2, &newGroup);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)newGroup;
}

JNIEXPORT jlong JNICALL Java_mpi_Group_difference(
        JNIElw *elw, jclass jthis, jlong group1, jlong group2)
{
    MPI_Group newGroup;

    int rc = MPI_Group_difference(
             (MPI_Group)group1, (MPI_Group)group2, &newGroup);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)newGroup;
}

JNIEXPORT jlong JNICALL Java_mpi_Group_incl(
        JNIElw *elw, jobject jthis, jlong group, jintArray ranks)
{
    jsize n = (*elw)->GetArrayLength(elw, ranks);
    jint *jRanks;
    int  *cRanks;
    ompi_java_getIntArray(elw, ranks, &jRanks, &cRanks);

    MPI_Group newGroup;
    int rc = MPI_Group_incl((MPI_Group)group, n, cRanks, &newGroup);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_forgetIntArray(elw, ranks, jRanks, cRanks);
    return (jlong)newGroup;
}

JNIEXPORT jlong JNICALL Java_mpi_Group_excl(
        JNIElw *elw, jobject jthis, jlong group, jintArray ranks)
{
    jsize n = (*elw)->GetArrayLength(elw, ranks);
    jint *jRanks;
    int  *cRanks;
    ompi_java_getIntArray(elw, ranks, &jRanks, &cRanks);

    MPI_Group newGroup;
    int rc = MPI_Group_excl((MPI_Group)group, n, cRanks, &newGroup);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_forgetIntArray(elw, ranks, jRanks, cRanks);
    return (jlong)newGroup;
}

JNIEXPORT jlong JNICALL Java_mpi_Group_rangeIncl(
        JNIElw *elw, jobject jthis, jlong group, jobjectArray ranges)
{
    int i;
    MPI_Group newGroup;
    jsize n = (*elw)->GetArrayLength(elw, ranges);
    int (*cRanges)[3] = (int(*)[3])calloc(n, sizeof(int[3]));

    for(i = 0; i < n; i++)
    {
        jintArray ri = (*elw)->GetObjectArrayElement(elw, ranges, i);
        jint *jri = (*elw)->GetIntArrayElements(elw, ri, NULL);
        cRanges[i][0] = jri[0];
        cRanges[i][1] = jri[1];
        cRanges[i][2] = jri[2];
        (*elw)->ReleaseIntArrayElements(elw, ri, jri, JNI_ABORT);
        (*elw)->DeleteLocalRef(elw, ri);
    }

    int rc = MPI_Group_range_incl((MPI_Group)group, n, cRanges, &newGroup);
    ompi_java_exceptionCheck(elw, rc);
    free(cRanges);
    return (jlong)newGroup;
}

JNIEXPORT jlong JNICALL Java_mpi_Group_rangeExcl(
        JNIElw *elw, jobject jthis, jlong group, jobjectArray ranges)
{
    int i;
    MPI_Group newGroup;
    jsize n = (*elw)->GetArrayLength(elw, ranges);
    int (*cRanges)[3] = (int(*)[3])calloc(n, sizeof(int[3]));

    for(i = 0; i < n; i++)
    {
        jintArray ri = (*elw)->GetObjectArrayElement(elw, ranges, i);
        jint *jri = (*elw)->GetIntArrayElements(elw, ri, NULL);
        cRanges[i][0] = jri[0];
        cRanges[i][1] = jri[1];
        cRanges[i][2] = jri[2];
        (*elw)->ReleaseIntArrayElements(elw, ri, jri, JNI_ABORT);
        (*elw)->DeleteLocalRef(elw, ri);
    }

    int rc = MPI_Group_range_excl((MPI_Group)group, n, cRanges, &newGroup);
    ompi_java_exceptionCheck(elw, rc);
    free(cRanges);
    return (jlong)newGroup;
}
