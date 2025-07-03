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
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef _MPIJAVA_H_
#define _MPIJAVA_H_

#include "mpi.h"
#include "opal/class/opal_free_list.h"

typedef struct {
    jfieldID  CommHandle;
    jfieldID  GroupHandle;
    jclass    CartParmsClass;
    jmethodID CartParmsInit;
    jclass    ShiftParmsClass;
    jmethodID ShiftParmsInit;
    jclass    VersionClass;
    jmethodID VersionInit;
    jclass    CountClass;
    jmethodID CountInit;
    jclass    GraphParmsClass;
    jmethodID GraphParmsInit;
    jclass    DistGraphNeighborsClass;
    jmethodID DistGraphNeighborsInit;
    jfieldID  DatatypeHandle;
    jfieldID  DatatypeBaseType;
    jfieldID  DatatypeBaseSize;
    jfieldID  MessageHandle;
    jfieldID  OpHandle;
    jfieldID  OpCommute;
    jmethodID OpCall;
    jfieldID  ReqHandle;
    jclass    StatusClass;
    jfieldID  StatusData;
    jclass    ExceptionClass;
    jmethodID ExceptionInit;
    jclass    IntegerClass;
    jmethodID IntegerValueOf;
    jclass    LongClass;
    jmethodID LongValueOf;
} ompi_java_globals_t;

extern ompi_java_globals_t ompi_java;

typedef struct ompi_java_buffer_t
{
    opal_free_list_item_t super;
    void *buffer;
} ompi_java_buffer_t;

OMPI_DECLSPEC OBJ_CLASS_DECLARATION(ompi_java_buffer_t);

void* ompi_java_getArrayCritical(void** bufBase, JNIElw *elw,
                                 jobject buf, int offset);

void* ompi_java_getDirectBufferAddress(JNIElw *elw, jobject buf);

/* Gets a buffer pointer for reading (copy from Java). */
void ompi_java_getReadPtr(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw, jobject buf,
        jboolean db, int offset, int count, MPI_Datatype type, int baseType);

/* Gets a buffer pointer for reading.
 * It only copies from java the rank data.
 * 'size' is the number of processes. */
void ompi_java_getReadPtrRank(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw,
        jobject buf, jboolean db, int offset, int count, int size,
        int rank, MPI_Datatype type, int baseType);

/* Gets a buffer pointer for reading, but it
 * 'size' is the number of processes.
 * if rank == -1 it copies all data from Java.
 * if rank != -1 it only copies from Java the rank data. */
void ompi_java_getReadPtrv(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw,
        jobject buf, jboolean db, int off, int *counts, int *displs,
        int size, int rank, MPI_Datatype type, int baseType);

/* Releases a buffer used for reading. */
void ompi_java_releaseReadPtr(
        void *ptr, ompi_java_buffer_t *item, jobject buf, jboolean db);

/* Gets a buffer pointer for writing. */
void ompi_java_getWritePtr(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw,
        jobject buf, jboolean db, int count, MPI_Datatype type);

/* Gets a buffer pointer for writing.
 * 'size' is the number of processes. */
void ompi_java_getWritePtrv(
        void **ptr, ompi_java_buffer_t **item, JNIElw *elw, jobject buf,
        jboolean db, int *counts, int *displs, int size, MPI_Datatype type);

/* Releases a buffer used for writing.
 * It copies data to Java. */
void ompi_java_releaseWritePtr(
        void *ptr, ompi_java_buffer_t *item, JNIElw *elw, jobject buf,
        jboolean db, int offset, int count, MPI_Datatype type, int baseType);

/* Releases a buffer used for writing.
 * It copies data to Java.
 * 'size' is the number of processes. */
void ompi_java_releaseWritePtrv(
        void *ptr, ompi_java_buffer_t *item, JNIElw *elw,
        jobject buf, jboolean db, int off, int *counts, int *displs,
        int size, MPI_Datatype type, int baseType);

void ompi_java_setStaticLongField(JNIElw *elw, jclass c,
                                  char *field, jlong value);

void ompi_java_setIntField(JNIElw *elw, jclass c, jobject obj,
                           char *field, jint value);

jobject ompi_java_Integer_valueOf(JNIElw *elw, jint i);
jobject ompi_java_Long_valueOf(JNIElw *elw, jlong i);

void ompi_java_getIntArray(
        JNIElw *elw, jintArray array, jint **jptr, int **cptr);
void ompi_java_releaseIntArray(
        JNIElw *elw, jintArray array, jint *jptr, int *cptr);
void ompi_java_forgetIntArray(
        JNIElw *elw, jintArray array, jint *jptr, int *cptr);

void ompi_java_getDatatypeArray(
        JNIElw *elw, jlongArray array, jlong **jptr, MPI_Datatype **cptr);
void ompi_java_forgetDatatypeArray(
        JNIElw *elw, jlongArray array, jlong *jptr, MPI_Datatype *cptr);

void ompi_java_getBooleanArray(
        JNIElw *elw, jbooleanArray array, jboolean **jptr, int **cptr);
void ompi_java_releaseBooleanArray(
        JNIElw *elw, jbooleanArray array, jboolean *jptr, int *cptr);
void ompi_java_forgetBooleanArray(
        JNIElw *elw, jbooleanArray array, jboolean *jptr, int *cptr);

void ompi_java_getPtrArray(
        JNIElw *elw, jlongArray array, jlong **jptr, void ***cptr);
void ompi_java_releasePtrArray(
        JNIElw *elw, jlongArray array, jlong *jptr, void **cptr);

jboolean ompi_java_exceptionCheck(JNIElw *elw, int rc);

void*      ompi_java_attrSet(JNIElw *elw, jbyteArray jval);
jbyteArray ompi_java_attrGet(JNIElw *elw, void *cval);
int        ompi_java_attrCopy(void *attrValIn, void *attrValOut, int *flag);
int        ompi_java_attrDelete(void *attrVal);

MPI_Op ompi_java_op_getHandle(
        JNIElw *elw, jobject jOp, jlong hOp, int baseType);

jobject ompi_java_status_new(JNIElw *elw, MPI_Status *status);
jobject ompi_java_status_newIndex(JNIElw *elw, MPI_Status *status, int index);

void ompi_java_status_set(
        JNIElw *elw, jlongArray jData, MPI_Status *status);
void ompi_java_status_setIndex(
        JNIElw *elw, jlongArray jData, MPI_Status *status, int index);

#endif /* _MPIJAVA_H_ */
