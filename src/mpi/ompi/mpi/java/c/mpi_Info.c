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

#include "ompi_config.h"

#include <stdlib.h>
#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_Info.h"
#include "mpiJava.h"

JNIEXPORT jlong JNICALL Java_mpi_Info_create(JNIElw *elw, jobject jthis)
{
    MPI_Info info;
    int rc = MPI_Info_create(&info);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)info;
}

JNIEXPORT jlong JNICALL Java_mpi_Info_getElw(JNIElw *elw, jclass clazz)
{
    return (jlong)MPI_INFO_ELW;
}

JNIEXPORT jlong JNICALL Java_mpi_Info_getNull(JNIElw *elw, jclass clazz)
{
    return (jlong)MPI_INFO_NULL;
}

JNIEXPORT void JNICALL Java_mpi_Info_set(
        JNIElw *elw, jobject jthis, jlong handle, jstring jkey, jstring jvalue)
{
    const char *key   = (*elw)->GetStringUTFChars(elw, jkey,   NULL),
               *value = (*elw)->GetStringUTFChars(elw, jvalue, NULL);

    int rc = MPI_Info_set((MPI_Info)handle, (char*)key, (char*)value);
    ompi_java_exceptionCheck(elw, rc);

    (*elw)->ReleaseStringUTFChars(elw, jkey,   key);
    (*elw)->ReleaseStringUTFChars(elw, jvalue, value);
}

JNIEXPORT jstring JNICALL Java_mpi_Info_get(
        JNIElw *elw, jobject jthis, jlong handle, jstring jkey)
{
    MPI_Info info = (MPI_Info)handle;
    const char *key = (*elw)->GetStringUTFChars(elw, jkey, NULL);

    int rc, valueLen, flag;
    rc = MPI_Info_get_valuelen(info, (char*)key, &valueLen, &flag);

    if(ompi_java_exceptionCheck(elw, rc) || !flag)
    {
        (*elw)->ReleaseStringUTFChars(elw, jkey, key);
        return NULL;
    }

    char *value = (char*)calloc(valueLen + 1, sizeof(char));
    rc = MPI_Info_get((MPI_Info)info, (char*)key, valueLen, value, &flag);
    (*elw)->ReleaseStringUTFChars(elw, jkey, key);

    if(ompi_java_exceptionCheck(elw, rc) || !flag)
    {
        free(value);
        return NULL;
    }

    jstring jvalue = (*elw)->NewStringUTF(elw, value);
    free(value);
    return jvalue;
}

JNIEXPORT void JNICALL Java_mpi_Info_delete(
        JNIElw *elw, jobject jthis, jlong handle, jstring jkey)
{
    const char *key = (*elw)->GetStringUTFChars(elw, jkey, NULL);
    int rc = MPI_Info_delete((MPI_Info)handle, (char*)key);
    ompi_java_exceptionCheck(elw, rc);
    (*elw)->ReleaseStringUTFChars(elw, jkey, key);
}

JNIEXPORT jint JNICALL Java_mpi_Info_size(
        JNIElw *elw, jobject jthis, jlong handle)
{
    int rc, nkeys;
    rc = MPI_Info_get_nkeys((MPI_Info)handle, &nkeys);
    ompi_java_exceptionCheck(elw, rc);
    return (jint)nkeys;
}

JNIEXPORT jstring JNICALL Java_mpi_Info_getKey(
        JNIElw *elw, jobject jthis, jlong handle, jint i)
{
    char key[MPI_MAX_INFO_KEY + 1];
    int rc = MPI_Info_get_nthkey((MPI_Info)handle, i, key);

    return ompi_java_exceptionCheck(elw, rc)
           ? NULL : (*elw)->NewStringUTF(elw, key);
}

JNIEXPORT jlong JNICALL Java_mpi_Info_dup(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Info newInfo;
    int rc = MPI_Info_dup((MPI_Info)handle, &newInfo);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)newInfo;
}

JNIEXPORT jlong JNICALL Java_mpi_Info_free(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Info info = (MPI_Info)handle;
    int rc = MPI_Info_free(&info);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)info;
}

JNIEXPORT jboolean JNICALL Java_mpi_Info_isNull(
        JNIElw *elw, jobject jthis, jlong handle)
{
    return (MPI_Info)handle == MPI_INFO_NULL ? JNI_TRUE : JNI_FALSE;
}
