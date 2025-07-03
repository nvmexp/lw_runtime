/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
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
 * Copyright (c) 2015-2016 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015-2016 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Intel, Inc. All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
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
 * File         : mpi_MPI.c
 * Headerfile   : mpi_MPI.h
 * Author       : SungHoon Ko, Xinying Li (contributions from MAEDA Atusi)
 * Created      : Thu Apr  9 12:22:15 1998
 * Revision     : $Revision: 1.17 $
 * Updated      : $Date: 2003/01/17 01:50:37 $
 * Copyright: Northeast Parallel Architectures Center
 *            at Syralwse University 1998
 */
#include "ompi_config.h"

#include <stdio.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#endif
#include <poll.h>
#ifdef HAVE_LIBGEN_H
#include <libgen.h>
#endif

#include "opal/util/output.h"
#include "opal/datatype/opal_colwertor.h"
#include "opal/mca/base/mca_base_var.h"

#include "mpi.h"
#include "ompi/errhandler/errcode.h"
#include "ompi/errhandler/errcode-internal.h"
#include "ompi/datatype/ompi_datatype.h"
#include "mpi_MPI.h"
#include "mpiJava.h"

ompi_java_globals_t ompi_java = {0};
int ompi_mpi_java_eager = 65536;
opal_free_list_t ompi_java_buffers = {{{0}}};

static void bufferConstructor(ompi_java_buffer_t *item)
{
    item->buffer = malloc(ompi_mpi_java_eager);
}

static void bufferDestructor(ompi_java_buffer_t *item)
{
    free(item->buffer);
}

OBJ_CLASS_INSTANCE(ompi_java_buffer_t,
                   opal_free_list_item_t,
                   bufferConstructor,
                   bufferDestructor);

/*
 * Class:    mpi_MPI
 * Method:   loadGlobalLibraries
 *
 */
jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
    // Ensure that PSM signal hijacking is disabled *before* loading
    // the library (see comment in the function for more detail).
    opal_init_psm();

    return JNI_VERSION_1_6;
}

static void initFreeList(void)
{
    OBJ_CONSTRUCT(&ompi_java_buffers, opal_free_list_t);

    int r = opal_free_list_init(&ompi_java_buffers,
                                sizeof(ompi_java_buffer_t),
                                opal_cache_line_size,
                                OBJ_CLASS(ompi_java_buffer_t),
                                0, /* payload size */
                                0, /* payload align */
                                2, /* initial elements to alloc */
                                -1, /* max elements */
                                2, /* num elements per alloc */
                                NULL, /* mpool */
                                0, /* mpool reg flags */
                                NULL, /* unused0 */
                                NULL, /* item_init */
                                NULL /* inem_init context */);
    if(r != OPAL_SUCCESS)
    {
        fprintf(stderr, "Unable to initialize ompi_java_buffers.\n");
        exit(1);
    }
}

static jclass findClass(JNIElw *elw, const char *className)
{
    jclass c = (*elw)->FindClass(elw, className),
           r = (*elw)->NewGlobalRef(elw, c);

    (*elw)->DeleteLocalRef(elw, c);
    return r;
}

static void findClasses(JNIElw *elw)
{
    ompi_java.CartParmsClass  = findClass(elw, "mpi/CartParms");
    ompi_java.ShiftParmsClass = findClass(elw, "mpi/ShiftParms");
    ompi_java.GraphParmsClass = findClass(elw, "mpi/GraphParms");

    ompi_java.DistGraphNeighborsClass = findClass(
                                        elw, "mpi/DistGraphNeighbors");

    ompi_java.StatusClass    = findClass(elw, "mpi/Status");
    ompi_java.ExceptionClass = findClass(elw, "mpi/MPIException");

    ompi_java.ExceptionInit = (*elw)->GetMethodID(
                              elw, ompi_java.ExceptionClass,
                              "<init>", "(IILjava/lang/String;)V");

    ompi_java.IntegerClass = findClass(elw, "java/lang/Integer");
    ompi_java.LongClass    = findClass(elw, "java/lang/Long");

    ompi_java.IntegerValueOf = (*elw)->GetStaticMethodID(
            elw, ompi_java.IntegerClass, "valueOf", "(I)Ljava/lang/Integer;");
    ompi_java.LongValueOf = (*elw)->GetStaticMethodID(
            elw, ompi_java.LongClass, "valueOf", "(J)Ljava/lang/Long;");
}

static void deleteClasses(JNIElw *elw)
{
    (*elw)->DeleteGlobalRef(elw, ompi_java.CartParmsClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.ShiftParmsClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.VersionClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.CountClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.GraphParmsClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.DistGraphNeighborsClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.StatusClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.ExceptionClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.IntegerClass);
    (*elw)->DeleteGlobalRef(elw, ompi_java.LongClass);
}

JNIEXPORT jobject JNICALL Java_mpi_MPI_newInt2(JNIElw *elw, jclass clazz)
{
    struct { int a; int b; } s;
    int iOff = (int)((MPI_Aint)(&(s.b)) - (MPI_Aint)(&(s.a)));
    jclass c = (*elw)->FindClass(elw, "mpi/Int2");
    jmethodID m = (*elw)->GetMethodID(elw, c, "<init>", "(II)V");
    return (*elw)->NewObject(elw, c, m, iOff, sizeof(int));
}

JNIEXPORT jobject JNICALL Java_mpi_MPI_newShortInt(JNIElw *elw, jclass clazz)
{
    struct { short a; int b; } s;
    int iOff = (int)((MPI_Aint)(&(s.b)) - (MPI_Aint)(&(s.a)));
    jclass c = (*elw)->FindClass(elw, "mpi/ShortInt");
    jmethodID m = (*elw)->GetMethodID(elw, c, "<init>", "(III)V");
    return (*elw)->NewObject(elw, c, m, sizeof(short), iOff, sizeof(int));
}

JNIEXPORT jobject JNICALL Java_mpi_MPI_newLongInt(JNIElw *elw, jclass clazz)
{
    struct { long a; int b; } s;
    int iOff = (int)((MPI_Aint)(&(s.b)) - (MPI_Aint)(&(s.a)));
    jclass c = (*elw)->FindClass(elw, "mpi/LongInt");
    jmethodID m = (*elw)->GetMethodID(elw, c, "<init>", "(III)V");
    return (*elw)->NewObject(elw, c, m, sizeof(long), iOff, sizeof(int));
}

JNIEXPORT jobject JNICALL Java_mpi_MPI_newFloatInt(JNIElw *elw, jclass clazz)
{
    struct { float a; int b; } s;
    int iOff = (int)((MPI_Aint)(&(s.b)) - (MPI_Aint)(&(s.a)));
    jclass c = (*elw)->FindClass(elw, "mpi/FloatInt");
    jmethodID m = (*elw)->GetMethodID(elw, c, "<init>", "(II)V");
    return (*elw)->NewObject(elw, c, m, iOff, sizeof(int));
}

JNIEXPORT jobject JNICALL Java_mpi_MPI_newDoubleInt(JNIElw *elw, jclass clazz)
{
    struct { double a; int b; } s;
    int iOff = (int)((MPI_Aint)(&(s.b)) - (MPI_Aint)(&(s.a)));
    jclass c = (*elw)->FindClass(elw, "mpi/DoubleInt");
    jmethodID m = (*elw)->GetMethodID(elw, c, "<init>", "(II)V");
    return (*elw)->NewObject(elw, c, m, iOff, sizeof(int));
}

JNIEXPORT void JNICALL Java_mpi_MPI_initVersion(JNIElw *elw, jclass jthis)
{
    ompi_java.VersionClass = findClass(elw, "mpi/Version");
    ompi_java.VersionInit = (*elw)->GetMethodID(elw, ompi_java.VersionClass, "<init>", "(II)V");
}

JNIEXPORT jobjectArray JNICALL Java_mpi_MPI_Init_1jni(
        JNIElw *elw, jclass clazz, jobjectArray argv)
{
    jsize i;
    jclass string;
    jobject value;

    int len = (*elw)->GetArrayLength(elw, argv);
    char **sargs = (char**)calloc(len+1, sizeof(char*));

    for(i = 0; i < len; i++)
    {
        jstring jc = (jstring)(*elw)->GetObjectArrayElement(elw, argv, i);
        const char *s = (*elw)->GetStringUTFChars(elw, jc, NULL);
        sargs[i] = strdup(s);
        (*elw)->ReleaseStringUTFChars(elw, jc, s);
        (*elw)->DeleteLocalRef(elw, jc);
    }

    int rc = MPI_Init(&len, &sargs);
    
    if(ompi_java_exceptionCheck(elw, rc)) {
        for(i = 0; i < len; i++)
            free (sargs[i]);
        free(sargs);
        return NULL;
    }

    mca_base_var_register("ompi", "mpi", "java", "eager",
                          "Java buffers eager size",
                          MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                          OPAL_INFO_LVL_5,
                          MCA_BASE_VAR_SCOPE_READONLY,
                          &ompi_mpi_java_eager);

    string = (*elw)->FindClass(elw, "java/lang/String");
    value = (*elw)->NewObjectArray(elw, len, string, NULL);

    for(i = 0; i < len; i++)
    {
        jstring jc = (*elw)->NewStringUTF(elw, sargs[i]);
        (*elw)->SetObjectArrayElement(elw, value, i, jc);
        (*elw)->DeleteLocalRef(elw, jc);
        free (sargs[i]);
    }

    free (sargs);

    findClasses(elw);
    initFreeList();
    return value;
}

JNIEXPORT jint JNICALL Java_mpi_MPI_InitThread_1jni(
        JNIElw *elw, jclass clazz, jobjectArray argv, jint required)
{
    jsize i;
    int len = (*elw)->GetArrayLength(elw,argv);
    char **sargs = (char**)calloc(len+1, sizeof(char*));

    for(i = 0; i < len; i++)
    {
        jstring jc = (jstring)(*elw)->GetObjectArrayElement(elw, argv, i);
        const char *s = (*elw)->GetStringUTFChars(elw, jc, 0);
        sargs[i] = strdup(s);
        (*elw)->ReleaseStringUTFChars(elw, jc, s);
        (*elw)->DeleteLocalRef(elw, jc);
    }

    int provided;
    int rc = MPI_Init_thread(&len, &sargs, required, &provided);
    
    if(ompi_java_exceptionCheck(elw, rc)) {
        for(i = 0; i < len; i++)
            free (sargs[i]);
        free(sargs);
        return -1;
    }

    findClasses(elw);
    initFreeList();
    return provided;
}

JNIEXPORT jint JNICALL Java_mpi_MPI_queryThread_1jni(JNIElw *elw, jclass clazz)
{
    int provided;
    int rc = MPI_Query_thread(&provided);
    ompi_java_exceptionCheck(elw, rc);
    return provided;
}

JNIEXPORT jboolean JNICALL Java_mpi_MPI_isThreadMain_1jni(
                           JNIElw *elw, jclass clazz)
{
    int flag;
    int rc = MPI_Is_thread_main(&flag);
    ompi_java_exceptionCheck(elw, rc);
    return flag ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_mpi_MPI_Finalize_1jni(JNIElw *elw, jclass obj)
{
    OBJ_DESTRUCT(&ompi_java_buffers);
    int rc = MPI_Finalize();
    ompi_java_exceptionCheck(elw, rc);
    deleteClasses(elw);
}

JNIEXPORT jobject JNICALL Java_mpi_MPI_getVersionJNI(JNIElw *elw, jclass jthis)
{
	int version, subversion;
	int rc = MPI_Get_version(&version, &subversion);
	ompi_java_exceptionCheck(elw, rc);

	return (*elw)->NewObject(elw, ompi_java.VersionClass,
	                             ompi_java.VersionInit, version, subversion);
}

JNIEXPORT jstring JNICALL Java_mpi_MPI_getLibVersionJNI(JNIElw *elw, jclass jthis)
{
	int length;
	char version[MPI_MAX_LIBRARY_VERSION_STRING];
	int rc = MPI_Get_library_version(version, &length);
	ompi_java_exceptionCheck(elw, rc);

	return (*elw)->NewStringUTF(elw, version);
}

JNIEXPORT jint JNICALL Java_mpi_MPI_getProcessorName(
                       JNIElw *elw, jclass obj, jbyteArray buf)
{
    int len;
    jbyte* bufc = (jbyte*)((*elw)->GetByteArrayElements(elw, buf, NULL));
    int rc = MPI_Get_processor_name((char*)bufc, &len);
    ompi_java_exceptionCheck(elw, rc);
    (*elw)->ReleaseByteArrayElements(elw, buf, bufc, 0);
    return len;
}

JNIEXPORT jdouble JNICALL Java_mpi_MPI_wtime_1jni(JNIElw *elw, jclass jthis)
{
    return MPI_Wtime();
}

JNIEXPORT jdouble JNICALL Java_mpi_MPI_wtick_1jni(JNIElw *elw, jclass jthis)
{
    return MPI_Wtick();
}

JNIEXPORT jboolean JNICALL Java_mpi_MPI_isInitialized(JNIElw *elw, jclass jthis)
{
    int flag;
    int rc = MPI_Initialized(&flag);
    ompi_java_exceptionCheck(elw, rc);
    return flag ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_mpi_MPI_isFinalized(JNIElw *elw, jclass jthis)
{
    int flag;
    int rc = MPI_Finalized(&flag);
    ompi_java_exceptionCheck(elw, rc);
    return flag ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_mpi_MPI_attachBuffer_1jni(
                       JNIElw *elw, jclass jthis, jbyteArray buf)
{
    int size=(*elw)->GetArrayLength(elw,buf);
    jbyte* bufptr = (*elw)->GetByteArrayElements(elw, buf, NULL);
    int rc = MPI_Buffer_attach(bufptr,size);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_MPI_detachBuffer_1jni(
                       JNIElw *elw, jclass jthis, jbyteArray buf)
{
    int size;
    jbyte* bufptr;
    int rc = MPI_Buffer_detach(&bufptr, &size);
    ompi_java_exceptionCheck(elw, rc);

    if(buf != NULL)
        (*elw)->ReleaseByteArrayElements(elw,buf,bufptr,0);
}

void* ompi_java_getArrayCritical(void** bufBase, JNIElw *elw,
                                 jobject buf, int offset)
{
    *bufBase = (jbyte*)(*elw)->GetPrimitiveArrayCritical(elw, buf, NULL);
    return ((jbyte*)*bufBase) + offset;
}

void* ompi_java_getDirectBufferAddress(JNIElw *elw, jobject buf)
{
    /* Allow NULL buffers to send/recv 0 items as control messages. */
    return buf == NULL ? NULL : (*elw)->GetDirectBufferAddress(elw, buf);
}

static int getTypeExtent(JNIElw *elw, MPI_Datatype type)
{
    MPI_Aint lb, extent;
    int rc = MPI_Type_get_extent(type, &lb, &extent);
    ompi_java_exceptionCheck(elw, rc);
    int value = extent;
    assert(((MPI_Aint)value) == extent);
    return value;
}

static void getArrayRegion(JNIElw *elw, jobject buf, int baseType,
                           int offset, int length, void *ptr)
{
    switch(baseType)
    {
        case 0:
            break;
        case 1:
            (*elw)->GetByteArrayRegion(elw, buf, offset, length, ptr);
            break;
        case 2:
            (*elw)->GetCharArrayRegion(elw, buf, offset / 2, length / 2, ptr);
            break;
        case 3:
            (*elw)->GetShortArrayRegion(elw, buf, offset / 2, length / 2, ptr);
            break;
        case 4:
            (*elw)->GetBooleanArrayRegion(elw, buf, offset, length, ptr);
            break;
        case 5:
            (*elw)->GetIntArrayRegion(elw, buf, offset / 4, length / 4, ptr);
            break;
        case 6:
            (*elw)->GetLongArrayRegion(elw, buf, offset / 8, length / 8, ptr);
            break;
        case 7:
            (*elw)->GetFloatArrayRegion(elw, buf, offset / 4, length / 4, ptr);
            break;
        case 8:
            (*elw)->GetDoubleArrayRegion(elw, buf, offset / 8, length / 8, ptr);
            break;
        case 9:
            (*elw)->GetByteArrayRegion(elw, buf, offset, length, ptr);
            break;
        default:
            assert(0);
    }
}

static void setArrayRegion(JNIElw *elw, jobject buf, int baseType,
                           int offset, int length, void *ptr)
{
    switch(baseType)
    {
        case 0:
            break;
        case 1:
            (*elw)->SetByteArrayRegion(elw, buf, offset, length, ptr);
            break;
        case 2:
            (*elw)->SetCharArrayRegion(elw, buf, offset / 2, length / 2, ptr);
            break;
        case 3:
            (*elw)->SetShortArrayRegion(elw, buf, offset / 2, length / 2, ptr);
            break;
        case 4:
            (*elw)->SetBooleanArrayRegion(elw, buf, offset, length, ptr);
            break;
        case 5:
            (*elw)->SetIntArrayRegion(elw, buf, offset / 4, length / 4, ptr);
            break;
        case 6:
            (*elw)->SetLongArrayRegion(elw, buf, offset / 8, length / 8, ptr);
            break;
        case 7:
            (*elw)->SetFloatArrayRegion(elw, buf, offset / 4, length / 4, ptr);
            break;
        case 8:
            (*elw)->SetDoubleArrayRegion(elw, buf, offset / 8, length / 8, ptr);
            break;
        case 9:
            (*elw)->SetByteArrayRegion(elw, buf, offset, length, ptr);
            break;
        default:
            assert(0);
    }
}

static void* getBuffer(JNIElw *elw, ompi_java_buffer_t **item, int size)
{
    if(size > ompi_mpi_java_eager)
    {
        *item = NULL;
        return malloc(size);
    }
    else
    {
        opal_free_list_item_t *freeListItem;
        freeListItem = opal_free_list_get (&ompi_java_buffers);

        ompi_java_exceptionCheck(elw, NULL == freeListItem ? MPI_ERR_NO_MEM :
                                 MPI_SUCCESS);
        if (NULL == freeListItem) {
            return NULL;
        }

        *item = (ompi_java_buffer_t*)freeListItem;
        return (*item)->buffer;
    }
}

static void releaseBuffer(void *ptr, ompi_java_buffer_t *item)
{
    if(item == NULL)
    {
        free(ptr);
    }
    else
    {
        assert(item->buffer == ptr);
        opal_free_list_return (&ompi_java_buffers, (opal_free_list_item_t*)item);
    }
}

static int getCountv(int *counts, int *displs, int size)
{
    /* Maybe displs is not ordered. */
    int i, max = 0;

    for(i = 1; i < size; i++)
    {
        if(displs[max] < displs[i])
            max = i;
    }

    return displs[max] * counts[max];
}

static void* getReadPtr(ompi_java_buffer_t **item, JNIElw *elw, jobject buf,
                        int offset, int count, MPI_Datatype type, int baseType)
{
    int  length = count * getTypeExtent(elw, type);
    void *ptr   = getBuffer(elw, item, length);

    if(opal_datatype_is_contiguous_memory_layout(&type->super, count))
    {
        getArrayRegion(elw, buf, baseType, offset, length, ptr);
    }
    else
    {
        void *inBuf, *inBase;
        inBuf = ompi_java_getArrayCritical(&inBase, elw, buf, offset);

        int rc = opal_datatype_copy_content_same_ddt(
                 &type->super, count, ptr, inBuf);

        ompi_java_exceptionCheck(elw,
                rc==OPAL_SUCCESS ? OMPI_SUCCESS : OMPI_ERROR);

        (*elw)->ReleasePrimitiveArrayCritical(elw, buf, inBase, JNI_ABORT);
    }

    return ptr;
}

static void* getReadPtrRank(
        ompi_java_buffer_t **item, JNIElw *elw, jobject buf, int offset,
        int count, int size, int rank, MPI_Datatype type, int baseType)
{
    int  extent = getTypeExtent(elw, type),
         rLen   = extent * count,
         length = rLen * size,
         rDispl = rLen * rank,
         rOff   = offset + rDispl;
    void *ptr   = getBuffer(elw, item, length);
    void *rPtr  = (char*)ptr + rDispl;

    if(opal_datatype_is_contiguous_memory_layout(&type->super, count))
    {
        getArrayRegion(elw, buf, baseType, rOff, rLen, rPtr);
    }
    else
    {
        void *bufPtr, *bufBase;
        bufPtr = ompi_java_getArrayCritical(&bufBase, elw, buf, rOff);

        int rc = opal_datatype_copy_content_same_ddt(
                 &type->super, count, rPtr, bufPtr);

        ompi_java_exceptionCheck(elw,
                rc==OPAL_SUCCESS ? OMPI_SUCCESS : OMPI_ERROR);

        (*elw)->ReleasePrimitiveArrayCritical(elw, buf, bufBase, JNI_ABORT);
    }

    return ptr;
}

static void* getReadPtrvRank(
        ompi_java_buffer_t **item, JNIElw *elw, jobject buf,
        int offset, int *counts, int *displs, int size,
        int rank, MPI_Datatype type, int baseType)
{
    int  extent  = getTypeExtent(elw, type),
         length  = extent * getCountv(counts, displs, size);
    void *ptr    = getBuffer(elw, item, length);
    int  rootOff = offset + extent * displs[rank];

    if(opal_datatype_is_contiguous_memory_layout(&type->super, counts[rank]))
    {
        int  rootLength = extent * counts[rank];
        void *rootPtr   = (char*)ptr + extent * displs[rank];
        getArrayRegion(elw, buf, baseType, rootOff, rootLength, rootPtr);
    }
    else
    {
        void *inBuf, *inBase;
        inBuf = ompi_java_getArrayCritical(&inBase, elw, buf, rootOff);

        int rc = opal_datatype_copy_content_same_ddt(
                 &type->super, counts[rank], ptr, inBuf);

        ompi_java_exceptionCheck(elw,
                rc==OPAL_SUCCESS ? OMPI_SUCCESS : OMPI_ERROR);

        (*elw)->ReleasePrimitiveArrayCritical(elw, buf, inBase, JNI_ABORT);
    }

    return ptr;
}

static void* getReadPtrvAll(
        ompi_java_buffer_t **item, JNIElw *elw, jobject buf,
        int offset, int *counts, int *displs, int size,
        MPI_Datatype type, int baseType)
{
    int  i,
         extent  = getTypeExtent(elw, type),
         length  = extent * getCountv(counts, displs, size);
    void *ptr    = getBuffer(elw, item, length);

    if(opal_datatype_is_contiguous_memory_layout(&type->super, 2))
    {
        for(i = 0; i < size; i++)
        {
            int   iOff = offset + extent * displs[i],
                  iLen = extent * counts[i];
            void *iPtr = (char*)ptr + extent * displs[i];
            getArrayRegion(elw, buf, baseType, iOff, iLen, iPtr);
        }
    }
    else
    {
        void *bufPtr, *bufBase;
        bufPtr = ompi_java_getArrayCritical(&bufBase, elw, buf, offset);

        for(i = 0; i < size; i++)
        {
            int   iOff = extent * displs[i];
            char *iBuf = iOff + (char*)bufPtr,
                 *iPtr = iOff + (char*)ptr;

            int rc = opal_datatype_copy_content_same_ddt(
                     &type->super, counts[i], iPtr, iBuf);

            ompi_java_exceptionCheck(elw,
                    rc==OPAL_SUCCESS ? OMPI_SUCCESS : OMPI_ERROR);
        }

        (*elw)->ReleasePrimitiveArrayCritical(elw, buf, bufBase, JNI_ABORT);
    }

    return ptr;
}

static void* getWritePtr(ompi_java_buffer_t **item, JNIElw *elw,
                         int count, MPI_Datatype type)
{
    int extent = getTypeExtent(elw, type),
        length = count * extent;

    return getBuffer(elw, item, length);
}

static void* getWritePtrv(ompi_java_buffer_t **item, JNIElw *elw,
                          int *counts, int *displs, int size, MPI_Datatype type)
{
    int extent = getTypeExtent(elw, type),
        count  = getCountv(counts, displs, size),
        length = extent * count;

    return getBuffer(elw, item, length);
}

void ompi_java_getReadPtr(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw, jobject buf,
        jboolean db, int offset, int count, MPI_Datatype type, int baseType)
{
    if(buf == NULL || baseType == 0)
    {
        /* Allow NULL buffers to send/recv 0 items as control messages. */
        *ptr  = NULL;
        *item = NULL;
    }
    else if(db)
    {
        assert(offset == 0);
        *ptr  = (*elw)->GetDirectBufferAddress(elw, buf);
        *item = NULL;
    }
    else
    {
        *ptr = getReadPtr(item, elw, buf, offset, count, type, baseType);
    }
}

void ompi_java_getReadPtrRank(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw,
        jobject buf, jboolean db, int offset, int count, int size,
        int rank, MPI_Datatype type, int baseType)
{
    if(buf == NULL || baseType == 0)
    {
        /* Allow NULL buffers to send/recv 0 items as control messages. */
        *ptr  = NULL;
        *item = NULL;
    }
    else if(db)
    {
        assert(offset == 0);
        *ptr  = (*elw)->GetDirectBufferAddress(elw, buf);
        *item = NULL;
    }
    else
    {
        *ptr = getReadPtrRank(item, elw, buf, offset, count,
                              size, rank, type, baseType);
    }
}

void ompi_java_getReadPtrv(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw,
        jobject buf, jboolean db, int offset, int *counts, int *displs,
        int size, int rank, MPI_Datatype type, int baseType)
{
    if(buf == NULL)
    {
        /* Allow NULL buffers to send/recv 0 items as control messages. */
        *ptr  = NULL;
        *item = NULL;
    }
    else if(db)
    {
        assert(offset == 0);
        *ptr  = (*elw)->GetDirectBufferAddress(elw, buf);
        *item = NULL;
    }
    else if(rank == -1)
    {
        *ptr = getReadPtrvAll(item, elw, buf, offset, counts,
                              displs, size, type, baseType);
    }
    else
    {
        *ptr = getReadPtrvRank(item, elw, buf, offset, counts,
                               displs, size, rank, type, baseType);
    }
}

void ompi_java_releaseReadPtr(
        void *ptr, ompi_java_buffer_t *item, jobject buf, jboolean db)
{
    if(!db && buf && ptr)
        releaseBuffer(ptr, item);
}

void ompi_java_getWritePtr(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw,
        jobject buf, jboolean db, int count, MPI_Datatype type)
{
    if(buf == NULL)
    {
        /* Allow NULL buffers to send/recv 0 items as control messages. */
        *ptr  = NULL;
        *item = NULL;
    }
    else if(db)
    {
        *ptr  = (*elw)->GetDirectBufferAddress(elw, buf);
        *item = NULL;
    }
    else
    {
        *ptr = getWritePtr(item, elw, count, type);
    }
}

void ompi_java_getWritePtrv(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw, jobject buf,
        jboolean db, int *counts, int *displs, int size, MPI_Datatype type)
{
    if(buf == NULL)
    {
        /* Allow NULL buffers to send/recv 0 items as control messages. */
        *ptr  = NULL;
        *item = NULL;
    }
    else if(db)
    {
        *ptr  = (*elw)->GetDirectBufferAddress(elw, buf);
        *item = NULL;
    }
    else
    {
        *ptr = getWritePtrv(item, elw, counts, displs, size, type);
    }
}

void ompi_java_releaseWritePtr(
        void *ptr, ompi_java_buffer_t *item, JNIElw *elw, jobject buf,
        jboolean db, int offset, int count, MPI_Datatype type, int baseType)
{
    if(db || !buf || !ptr)
        return;

    if(opal_datatype_is_contiguous_memory_layout(&type->super, count))
    {
        int length = count * getTypeExtent(elw, type);
        setArrayRegion(elw, buf, baseType, offset, length, ptr);
    }
    else
    {
        void *inBuf, *inBase;
        inBuf = ompi_java_getArrayCritical(&inBase, elw, buf, offset);

        int rc = opal_datatype_copy_content_same_ddt(
                 &type->super, count, inBuf, ptr);

        ompi_java_exceptionCheck(elw,
                rc==OPAL_SUCCESS ? OMPI_SUCCESS : OMPI_ERROR);

        (*elw)->ReleasePrimitiveArrayCritical(elw, buf, inBase, 0);
    }

    releaseBuffer(ptr, item);
}

void ompi_java_releaseWritePtrv(
        void *ptr, ompi_java_buffer_t *item, JNIElw *elw,
        jobject buf, jboolean db, int offset, int *counts, int *displs,
        int size, MPI_Datatype type, int baseType)
{
    if(db || !buf || !ptr)
        return;

    int i;
    int extent = getTypeExtent(elw, type);

    if(opal_datatype_is_contiguous_memory_layout(&type->super, 2))
    {
        for(i = 0; i < size; i++)
        {
            int   iOff = offset + extent * displs[i],
                  iLen = extent * counts[i];
            void *iPtr = (char*)ptr + extent * displs[i];
            setArrayRegion(elw, buf, baseType, iOff, iLen, iPtr);
        }
    }
    else
    {
        void *bufPtr, *bufBase;
        bufPtr = ompi_java_getArrayCritical(&bufBase, elw, buf, offset);

        for(i = 0; i < size; i++)
        {
            int   iOff = extent * displs[i];
            char *iBuf = iOff + (char*)bufPtr,
                 *iPtr = iOff + (char*)ptr;

            int rc = opal_datatype_copy_content_same_ddt(
                     &type->super, counts[i], iBuf, iPtr);

            ompi_java_exceptionCheck(elw,
                    rc==OPAL_SUCCESS ? OMPI_SUCCESS : OMPI_ERROR);
        }

        (*elw)->ReleasePrimitiveArrayCritical(elw, buf, bufBase, 0);
    }

    releaseBuffer(ptr, item);
}

jobject ompi_java_Integer_valueOf(JNIElw *elw, jint i)
{
    return (*elw)->CallStaticObjectMethod(elw,
           ompi_java.IntegerClass, ompi_java.IntegerValueOf, i);
}

jobject ompi_java_Long_valueOf(JNIElw *elw, jlong i)
{
    return (*elw)->CallStaticObjectMethod(elw,
           ompi_java.LongClass, ompi_java.LongValueOf, i);
}

void ompi_java_getIntArray(JNIElw *elw, jintArray array,
                           jint **jptr, int **cptr)
{
    jint *jInts = (*elw)->GetIntArrayElements(elw, array, NULL);
    *jptr = jInts;

    if(sizeof(int) == sizeof(jint))
    {
        *cptr = (int*)jInts;
    }
    else
    {
        int i, length = (*elw)->GetArrayLength(elw, array);
        int *cInts = calloc(length, sizeof(int));

        for(i = 0; i < length; i++)
            cInts[i] = jInts[i];

        *cptr = cInts;
    }
}

void ompi_java_releaseIntArray(JNIElw *elw, jintArray array,
                               jint *jptr, int *cptr)
{
    if(jptr != cptr)
    {
        int i, length = (*elw)->GetArrayLength(elw, array);

        for(i = 0; i < length; i++)
            jptr[i] = cptr[i];

        free(cptr);
    }

    (*elw)->ReleaseIntArrayElements(elw, array, jptr, 0);
}

void ompi_java_forgetIntArray(JNIElw *elw, jintArray array,
                              jint *jptr, int *cptr)
{
    if(jptr != cptr)
        free(cptr);

    (*elw)->ReleaseIntArrayElements(elw, array, jptr, JNI_ABORT);
}

void ompi_java_getDatatypeArray(JNIElw *elw, jlongArray array,
                           jlong **jptr, MPI_Datatype **cptr)
{
    jlong *jLongs = (*elw)->GetLongArrayElements(elw, array, NULL);
    *jptr = jLongs;

    int i, length = (*elw)->GetArrayLength(elw, array);
    MPI_Datatype *cDatatypes = calloc(length, sizeof(MPI_Datatype));

    for(i = 0; i < length; i++){
        cDatatypes[i] = (MPI_Datatype)jLongs[i];
    }
    *cptr = cDatatypes;
}

void ompi_java_forgetDatatypeArray(JNIElw *elw, jlongArray array,
                              jlong *jptr, MPI_Datatype *cptr)
{
    if((long)jptr != (long)cptr)
        free(cptr);

    (*elw)->ReleaseLongArrayElements(elw, array, jptr, JNI_ABORT);
}

void ompi_java_getBooleanArray(JNIElw *elw, jbooleanArray array,
                               jboolean **jptr, int **cptr)
{
    int i, length = (*elw)->GetArrayLength(elw, array);
    jboolean *jb = (*elw)->GetBooleanArrayElements(elw, array, NULL);
    int *cb = (int*)calloc(length, sizeof(int));

    for(i = 0; i < length; i++)
        cb[i] = jb[i];

    *jptr = jb;
    *cptr = cb;
}

void ompi_java_releaseBooleanArray(JNIElw *elw, jbooleanArray array,
                                   jboolean *jptr, int *cptr)
{
    int i, length = (*elw)->GetArrayLength(elw, array);

    for(i = 0; i < length; i++)
        jptr[i] = cptr[i] ? JNI_TRUE : JNI_FALSE;

    free(cptr);
    (*elw)->ReleaseBooleanArrayElements(elw, array, jptr, 0);
}

void ompi_java_forgetBooleanArray(JNIElw *elw, jbooleanArray array,
                                  jboolean *jptr, int *cptr)
{
    free(cptr);
    (*elw)->ReleaseBooleanArrayElements(elw, array, jptr, JNI_ABORT);
}

void ompi_java_getPtrArray(JNIElw *elw, jlongArray array,
                           jlong **jptr, void ***cptr)
{
    jlong *jp = *jptr = (*elw)->GetLongArrayElements(elw, array, NULL);

    if(sizeof(jlong) == sizeof(void*))
    {
        *cptr = (void**)jp;
    }
    else
    {
        int i, length = (*elw)->GetArrayLength(elw, array);
        void **cp = *cptr = calloc(length, sizeof(void*));

        for(i = 0; i < length; i++)
            cp[i] = (void*)jp[i];
    }
}

void ompi_java_releasePtrArray(JNIElw *elw, jlongArray array,
                               jlong *jptr, void **cptr)
{
    if(jptr != (jlong*)cptr)
    {
        int i, length = (*elw)->GetArrayLength(elw, array);

        for(i = 0; i < length; i++)
            jptr[i] = (jlong)cptr[i];

        free(cptr);
    }

    (*elw)->ReleaseLongArrayElements(elw, array, jptr, 0);
}

/* This method checks whether an MPI or JNI exception has oclwrred.
 * If an exception oclwrs, the C code will continue running.  Once
 * code exelwtion returns to Java code, an exception is immediately
 * thrown.  Since an exception has oclwrred somewhere in the C code,
 * the object that is returned from C may not be valid.  This is not
 * an issue, however, as the assignment opperation will not be
 * exelwted.  The results of this method need not be checked if the
 * only following code cleans up memory and then returns to Java.
 * If existing objects are changed after a call to this method, the
 * results need to be checked and, if an error has oclwrred, the
 * code should instead cleanup any memory and return.
 */
jboolean ompi_java_exceptionCheck(JNIElw *elw, int rc)
{
    jboolean jni_exception;

    if (rc < 0) {
        /* handle ompi error code */
        rc = ompi_errcode_get_mpi_code (rc);
        /* ompi_mpi_errcode_get_class CAN NOT handle negative error codes.
         * all Open MPI MPI error codes should be > 0. */
        assert (rc >= 0);
    }
    jni_exception = (*elw)->ExceptionCheck(elw);

    if(MPI_SUCCESS == rc && JNI_FALSE == jni_exception)
    {
        return JNI_FALSE;
    }
    else if(MPI_SUCCESS != rc)
    {
        int     errClass = ompi_mpi_errcode_get_class(rc);
        char    *message = ompi_mpi_errnum_get_string(rc);
        jstring jmessage = (*elw)->NewStringUTF(elw, (const char*)message);

        jobject mpiex = (*elw)->NewObject(elw, ompi_java.ExceptionClass,
                                          ompi_java.ExceptionInit,
                                          rc, errClass, jmessage);
        (*elw)->Throw(elw, mpiex);
        (*elw)->DeleteLocalRef(elw, mpiex);
        (*elw)->DeleteLocalRef(elw, jmessage);
        return JNI_TRUE;
    }
    /* If we get here, a JNI error has oclwrred. */
    return JNI_TRUE;
}

void* ompi_java_attrSet(JNIElw *elw, jbyteArray jval)
{
    int length = (*elw)->GetArrayLength(elw, jval);
    void *cval = malloc(sizeof(int) + length);
    *((int*)cval) = length;

    (*elw)->GetByteArrayRegion(elw, jval,
            0, length, (jbyte*)cval + sizeof(int));

    return cval;
}

jbyteArray ompi_java_attrGet(JNIElw *elw, void *cval)
{
    int length = *((int*)cval);
    jbyteArray jval = (*elw)->NewByteArray(elw, length);

    (*elw)->SetByteArrayRegion(elw, jval,
            0, length, (jbyte*)cval + sizeof(int));

    return jval;
}

int ompi_java_attrCopy(void *attrValIn, void *attrValOut, int *flag)
{
    int length = *((int*)attrValIn) + sizeof(int);
    *((void**)attrValOut) = malloc(length);
    memcpy(*((void**)attrValOut), attrValIn, length);
    *flag = 1;
    return MPI_SUCCESS;
}

int ompi_java_attrDelete(void *attrVal)
{
    free(attrVal);
    return MPI_SUCCESS;
}
