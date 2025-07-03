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
 * Copyright (c) 2016      Los Alamos National Security, LLC. All rights
 *                         reserved.
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
 * File         : mpi_Request.c
 * Headerfile   : mpi_Request.h
 * Author       : Sung-Hoon Ko, Xinying Li, Bryan Carpenter
 * Created      : Thu Apr  9 12:22:15 1998
 * Revision     : $Revision: 1.11 $
 * Updated      : $Date: 2003/01/16 16:39:34 $
 * Copyright: Northeast Parallel Architectures Center
 *            at Syralwse University 1998
 */

#include "ompi_config.h"
#include <stdlib.h>
#include <assert.h>
#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_Request.h"
#include "mpiJava.h"

JNIEXPORT void JNICALL Java_mpi_Request_init(JNIElw *elw, jclass c)
{
    ompi_java.ReqHandle = (*elw)->GetFieldID(elw, c, "handle", "J");
}

static void setIndices(JNIElw *elw, jintArray indices, int *cIdx, int count)
{
    jint *jIdx;

    if(sizeof(int) == sizeof(jint))
    {
        jIdx = cIdx;
    }
    else
    {
        jIdx = (jint*)calloc(count, sizeof(jint));
        int i;

        for(i = 0; i < count; i++)
            jIdx[i] = cIdx[i];
    }

    (*elw)->SetIntArrayRegion(elw, indices, 0, count, jIdx);

    if(jIdx != cIdx)
        free(jIdx);
}

static jobjectArray newStatuses(JNIElw *elw, MPI_Status *statuses, int count)
{
    jobjectArray array = (*elw)->NewObjectArray(elw,
                         count, ompi_java.StatusClass, NULL);
    int i;
    for(i = 0; i < count; i++)
    {
        jobject st = ompi_java_status_new(elw, statuses + i);
        (*elw)->SetObjectArrayElement(elw, array, i, st);
        (*elw)->DeleteLocalRef(elw, st);
    }

    return array;
}

static jobjectArray newStatusesIndices(
        JNIElw *elw, MPI_Status *statuses, int *indices, int count)
{
    if(count < 0)
        return NULL;

    jobjectArray array = (*elw)->NewObjectArray(elw,
                         count, ompi_java.StatusClass, NULL);
    int i;
    for(i = 0; i < count; i++)
    {
        jobject st = ompi_java_status_newIndex(elw, statuses + i, indices[i]);
        (*elw)->SetObjectArrayElement(elw, array, i, st);
        (*elw)->DeleteLocalRef(elw, st);
    }

    return array;
}

JNIEXPORT jlong JNICALL Java_mpi_Request_getNull(JNIElw *elw, jclass clazz)
{
    return (jlong)MPI_REQUEST_NULL;
}

JNIEXPORT void JNICALL Java_mpi_Request_cancel(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Request req = (MPI_Request)handle;
    int rc = MPI_Cancel(&req);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_Request_free(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Request req = (MPI_Request)handle;
    int rc = MPI_Request_free(&req);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)req;
}

JNIEXPORT jlong JNICALL Java_mpi_Request_waitStatus(
        JNIElw *elw, jobject jthis, jlong handle, jlongArray stat)
{
    MPI_Request req = (MPI_Request)handle;
    MPI_Status status;
    int rc = MPI_Wait(&req, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        ompi_java_status_set(elw, stat, &status);
    
    return (jlong)req;
}

JNIEXPORT jlong JNICALL Java_mpi_Request_waitFor(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Request req = (MPI_Request)handle;
    int rc = MPI_Wait(&req, MPI_STATUS_IGNORE);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)req;
}

JNIEXPORT jobject JNICALL Java_mpi_Request_testStatus(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Request req = (MPI_Request)handle;
    int flag;
    MPI_Status status;
    int rc = MPI_Test(&req, &flag, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        (*elw)->SetLongField(elw, jthis, ompi_java.ReqHandle, (jlong)req);
    
    return flag ? ompi_java_status_new(elw, &status) : NULL;
}

JNIEXPORT jobject JNICALL Java_mpi_Request_getStatus(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Request req = (MPI_Request)handle;
    int flag;
    MPI_Status status;
    int rc = MPI_Request_get_status(req, &flag, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        (*elw)->SetLongField(elw, jthis, ompi_java.ReqHandle, (jlong)req);
    
    return flag ? ompi_java_status_new(elw, &status) : NULL;
}

JNIEXPORT jboolean JNICALL Java_mpi_Request_test(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Request req = (MPI_Request)handle;
    int flag;
    int rc = MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        (*elw)->SetLongField(elw, jthis, ompi_java.ReqHandle, (jlong)req);
    
    return flag ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_mpi_Request_waitAnyStatus(
        JNIElw *elw, jclass clazz, jlongArray requests, jobject stat)
{
    jboolean exception;
    int count = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    int index;
    MPI_Status status;
    int rc = MPI_Waitany(count, cReq, &index, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    
    if(!exception)
        ompi_java_status_setIndex(elw, stat, &status, index);
}

JNIEXPORT jint JNICALL Java_mpi_Request_waitAny(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    int count = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    int index;
    int rc = MPI_Waitany(count, cReq, &index, MPI_STATUS_IGNORE);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    return index;
}

JNIEXPORT jobject JNICALL Java_mpi_Request_testAnyStatus(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    int count = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    int index, flag;
    MPI_Status status;
    int rc = MPI_Testany(count, cReq, &index, &flag, &status);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    return flag ? ompi_java_status_newIndex(elw, &status, index) : NULL;
}

JNIEXPORT jint JNICALL Java_mpi_Request_testAny(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    int count = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    int index, flag;
    int rc = MPI_Testany(count, cReq, &index, &flag, MPI_STATUS_IGNORE);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    return index;
}

JNIEXPORT jobjectArray JNICALL Java_mpi_Request_waitAllStatus(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    int count = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    MPI_Status *statuses = (MPI_Status*)calloc(count, sizeof(MPI_Status));
    int rc = MPI_Waitall(count, cReq, statuses);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    jobjectArray jStatuses = newStatuses(elw, statuses, count);
    free(statuses);
    return jStatuses;
}

JNIEXPORT void JNICALL Java_mpi_Request_waitAll(
        JNIElw *elw, jclass jthis, jlongArray requests)
{
    int count = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    int rc = MPI_Waitall(count, cReq, MPI_STATUSES_IGNORE);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
}

JNIEXPORT jobjectArray JNICALL Java_mpi_Request_testAllStatus(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    int count = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    MPI_Status *statuses = (MPI_Status*)calloc(count, sizeof(MPI_Status));
    int flag;
    int rc = MPI_Testall(count, cReq, &flag, statuses);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    jobjectArray jStatuses = flag ? newStatuses(elw, statuses, count) : NULL;
    free(statuses);
    return jStatuses;
}

JNIEXPORT jboolean JNICALL Java_mpi_Request_testAll(
        JNIElw *elw, jclass jthis, jlongArray requests)
{
    int count = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    int flag;
    int rc = MPI_Testall(count, cReq, &flag, MPI_STATUSES_IGNORE);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    return flag ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobjectArray JNICALL Java_mpi_Request_waitSomeStatus(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    int incount = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    MPI_Status *statuses = (MPI_Status*)calloc(incount, sizeof(MPI_Status));
    int *indices = (int*)calloc(incount, sizeof(int));
    int outcount;
    int rc = MPI_Waitsome(incount, cReq, &outcount, indices, statuses);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    jobjectArray jStatuses = newStatusesIndices(elw, statuses, indices, outcount);
    free(statuses);
    free(indices);
    return jStatuses;
}

JNIEXPORT jintArray JNICALL Java_mpi_Request_waitSome(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    jboolean exception;
    int incount = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    int *indices = (int*)calloc(incount, sizeof(int));
    int outcount;
    int rc = MPI_Waitsome(incount, cReq, &outcount, indices, MPI_STATUSES_IGNORE);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    
    if(exception) {
        free(indices);
        return NULL;
    }

    jintArray jindices = NULL;

    if(outcount != MPI_UNDEFINED)
    {
        jindices = (*elw)->NewIntArray(elw, outcount);
        setIndices(elw, jindices, indices, outcount);
    }

    free(indices);
    return jindices;
}

JNIEXPORT jobjectArray JNICALL Java_mpi_Request_testSomeStatus(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    int incount = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    MPI_Status *statuses = (MPI_Status*)calloc(incount, sizeof(MPI_Status));
    int *indices = (int*)calloc(incount, sizeof(int));
    int outcount;
    int rc = MPI_Testsome(incount, cReq, &outcount, indices, statuses);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    jobjectArray jStatuses = newStatusesIndices(elw, statuses, indices, outcount);
    free(statuses);
    free(indices);
    return jStatuses;
}

JNIEXPORT jintArray JNICALL Java_mpi_Request_testSome(
        JNIElw *elw, jclass clazz, jlongArray requests)
{
    jboolean exception;
    int incount = (*elw)->GetArrayLength(elw, requests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, requests, &jReq, (void***)&cReq);
    int *indices = (int*)calloc(incount, sizeof(int));
    int outcount;
    int rc = MPI_Testsome(incount, cReq, &outcount, indices, MPI_STATUSES_IGNORE);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, requests, jReq, (void**)cReq);
    
    if(exception) {
        free(indices);
        return NULL;
    }
    
    jintArray jindices = NULL;

    if(outcount != MPI_UNDEFINED)
    {
        jindices = (*elw)->NewIntArray(elw, outcount);
        setIndices(elw, jindices, indices, outcount);
    }

    free(indices);
    return jindices;
}
