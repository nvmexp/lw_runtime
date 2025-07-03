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
 * Copyright (c) 2017      FUJITSU LIMITED.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include <stdlib.h>
#include <assert.h>
#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_File.h"
#include "mpiJava.h"

JNIEXPORT jlong JNICALL Java_mpi_File_open(
        JNIElw *elw, jobject jthis, jlong comm,
        jstring jfilename, jint amode, jlong info)
{
    const char* filename = (*elw)->GetStringUTFChars(elw, jfilename, NULL);
    MPI_File fh;

    int rc = MPI_File_open((MPI_Comm)comm, (char*)filename,
                           amode, (MPI_Info)info, &fh);

    ompi_java_exceptionCheck(elw, rc);
    (*elw)->ReleaseStringUTFChars(elw, jfilename, filename);
    return (jlong)fh;
}

JNIEXPORT jlong JNICALL Java_mpi_File_close(
        JNIElw *elw, jobject jthis, jlong fh)
{
    MPI_File file = (MPI_File)fh;
    int rc = MPI_File_close(&file);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)file;
}

JNIEXPORT void JNICALL Java_mpi_File_delete(
        JNIElw *elw, jclass clazz, jstring jfilename, jlong info)
{
    const char* filename = (*elw)->GetStringUTFChars(elw, jfilename, NULL);
    int rc = MPI_File_delete((char*)filename, (MPI_Info)info);
    ompi_java_exceptionCheck(elw, rc);
    (*elw)->ReleaseStringUTFChars(elw, jfilename, filename);
}

JNIEXPORT void JNICALL Java_mpi_File_setSize(
        JNIElw *elw, jobject jthis, jlong fh, jlong size)
{
    int rc = MPI_File_set_size((MPI_File)fh, (MPI_Offset)size);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_File_preallocate(
        JNIElw *elw, jobject jthis, jlong fh, jlong size)
{
    int rc = MPI_File_preallocate((MPI_File)fh, (MPI_Offset)size);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_File_getSize(
        JNIElw *elw, jobject jthis, jlong fh)
{
    MPI_Offset size;
    int rc = MPI_File_get_size((MPI_File)fh, &size);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)size;
}

JNIEXPORT jlong JNICALL Java_mpi_File_getGroup(
        JNIElw *elw, jobject jthis, jlong fh)
{
    MPI_Group group;
    int rc = MPI_File_get_group((MPI_File)fh, &group);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)group;
}

JNIEXPORT jint JNICALL Java_mpi_File_getAMode(
        JNIElw *elw, jobject jthis, jlong fh)
{
    int amode;
    int rc = MPI_File_get_amode((MPI_File)fh, &amode);
    ompi_java_exceptionCheck(elw, rc);
    return amode;
}

JNIEXPORT void JNICALL Java_mpi_File_setInfo(
        JNIElw *elw, jobject jthis, jlong fh, jlong info)
{
    int rc = MPI_File_set_info((MPI_File)fh, (MPI_Info)info);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_File_getInfo(
        JNIElw *elw, jobject jthis, jlong fh)
{
    MPI_Info info;
    int rc = MPI_File_get_info((MPI_File)fh, &info);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)info;
}

JNIEXPORT void JNICALL Java_mpi_File_setView(
        JNIElw *elw, jobject jthis, jlong fh, jlong disp,
        jlong etype, jlong filetype, jstring jdatarep, jlong info)
{
    const char* datarep = (*elw)->GetStringUTFChars(elw, jdatarep, NULL);

    int rc = MPI_File_set_view(
            (MPI_File)fh, (MPI_Offset)disp, (MPI_Datatype)etype,
            (MPI_Datatype)filetype, (char*)datarep, (MPI_Info)info);

    ompi_java_exceptionCheck(elw, rc);
    (*elw)->ReleaseStringUTFChars(elw, jdatarep, datarep);
}

JNIEXPORT void JNICALL Java_mpi_File_readAt(
        JNIElw *elw, jobject jthis, jlong fh, jlong fileOffset,
        jobject buf, jboolean db, jint off, jint count,
        jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getWritePtr(&ptr, &item, elw, buf, db, count, type);
    MPI_Status status;

    int rc = MPI_File_read_at((MPI_File)fh, (MPI_Offset)fileOffset,
                              ptr, count, type, &status);

    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseWritePtr(ptr, item, elw, buf, db, off, count, type, bType);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_readAtAll(
        JNIElw *elw, jobject jthis, jlong fh, jlong fileOffset,
        jobject buf, jboolean db, jint off, jint count,
        jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getWritePtr(&ptr, &item, elw, buf, db, count, type);
    MPI_Status status;

    int rc = MPI_File_read_at_all((MPI_File)fh, (MPI_Offset)fileOffset,
                                  ptr, count, type, &status);

    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseWritePtr(ptr, item, elw, buf, db, off, count, type, bType);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_writeAt(
        JNIElw *elw, jobject jthis, jlong fh, jlong fileOffset,
        jobject buf, jboolean db, jint off, jint count,
        jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getReadPtr(&ptr, &item, elw, buf, db, off, count, type, bType);
    MPI_Status status;

    int rc = MPI_File_write_at((MPI_File)fh, (MPI_Offset)fileOffset,
                               ptr, count, type, &status);

    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseReadPtr(ptr, item, buf, db);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_writeAtAll(
        JNIElw *elw, jobject jthis, jlong fh, jlong fileOffset,
        jobject buf, jboolean db, jint off, jint count,
        jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getReadPtr(&ptr, &item, elw, buf, db, off, count, type, bType);
    MPI_Status status;

    int rc = MPI_File_write_at_all((MPI_File)fh, (MPI_Offset)fileOffset,
                                   ptr, count, (MPI_Datatype)type, &status);

    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseReadPtr(ptr, item, buf, db);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT jlong JNICALL Java_mpi_File_iReadAt(
        JNIElw *elw, jobject jthis, jlong fh, jlong offset,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iread_at((MPI_File)fh, (MPI_Offset)offset,
                               ptr, count, (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_File_iReadAtAll(
        JNIElw *elw, jobject jthis, jlong fh, jlong offset,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iread_at_all((MPI_File)fh, (MPI_Offset)offset,
                                   ptr, count, (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_File_iWriteAt(
        JNIElw *elw, jobject jthis, jlong fh, jlong offset,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iwrite_at((MPI_File)fh, (MPI_Offset)offset,
                                ptr, count, (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_File_iWriteAtAll(
        JNIElw *elw, jobject jthis, jlong fh, jlong offset,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iwrite_at_all((MPI_File)fh, (MPI_Offset)offset,
                                    ptr, count, (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT void JNICALL Java_mpi_File_read(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jboolean db,
        jint off, jint count, jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getWritePtr(&ptr, &item, elw, buf, db, count, type);
    MPI_Status status;
    int rc = MPI_File_read((MPI_File)fh, ptr, count, type, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseWritePtr(ptr, item, elw, buf, db, off, count, type, bType);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_readAll(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jboolean db,
        jint off, jint count, jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getWritePtr(&ptr, &item, elw, buf, db, count, type);
    MPI_Status status;
    int rc = MPI_File_read_all((MPI_File)fh, ptr, count, type, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseWritePtr(ptr, item, elw, buf, db, off, count, type, bType);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_write(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jboolean db,
        jint off, jint count, jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getReadPtr(&ptr, &item, elw, buf, db, off, count, type, bType);
    MPI_Status status;
    int rc = MPI_File_write((MPI_File)fh, ptr, count, type, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseReadPtr(ptr, item, buf, db);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_writeAll(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jboolean db,
        jint off, jint count, jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getReadPtr(&ptr, &item, elw, buf, db, off, count, type, bType);
    MPI_Status status;
    int rc = MPI_File_write_all((MPI_File)fh, ptr, count, type, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseReadPtr(ptr, item, buf, db);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT jlong JNICALL Java_mpi_File_iRead(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iread((MPI_File)fh, ptr, count,
                            (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_File_iReadAll(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iread_all((MPI_File)fh, ptr, count,
                                (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_File_iWrite(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iwrite((MPI_File)fh, ptr, count,
                             (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_File_iWriteAll(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iwrite_all((MPI_File)fh, ptr, count,
                                 (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT void JNICALL Java_mpi_File_seek(
        JNIElw *elw, jobject jthis, jlong fh, jlong offset, jint whence)
{
    int rc = MPI_File_seek((MPI_File)fh, (MPI_Offset)offset, whence);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_File_getPosition(
        JNIElw *elw, jobject jthis, jlong fh)
{
    MPI_Offset offset;
    int rc = MPI_File_get_position((MPI_File)fh, &offset);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)offset;
}

JNIEXPORT jlong JNICALL Java_mpi_File_getByteOffset(
        JNIElw *elw, jobject jthis, jlong fh, jlong offset)
{
    MPI_Offset disp;
    int rc = MPI_File_get_byte_offset((MPI_File)fh, (MPI_Offset)offset, &disp);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)disp;
}

JNIEXPORT void JNICALL Java_mpi_File_readShared(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jboolean db,
        jint off, jint count, jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getWritePtr(&ptr, &item, elw, buf, db, count, type);
    MPI_Status status;
    int rc = MPI_File_read_shared((MPI_File)fh, ptr, count, type, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseWritePtr(ptr, item, elw, buf, db, off, count, type, bType);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_writeShared(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jboolean db,
        jint off, jint count, jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getReadPtr(&ptr, &item, elw, buf, db, off, count, type, bType);
    MPI_Status status;
    int rc = MPI_File_write_shared((MPI_File)fh, ptr, count, type, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseReadPtr(ptr, item, buf, db);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT jlong JNICALL Java_mpi_File_iReadShared(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iread_shared((MPI_File)fh, ptr, count,
                                   (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_File_iWriteShared(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    MPI_Request request;

    int rc = MPI_File_iwrite_shared((MPI_File)fh, ptr, count,
                                    (MPI_Datatype)type, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT void JNICALL Java_mpi_File_readOrdered(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jboolean db,
        jint off, jint count, jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getWritePtr(&ptr, &item, elw, buf, db, count, type);
    MPI_Status status;
    int rc = MPI_File_read_ordered((MPI_File)fh, ptr, count, type, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseWritePtr(ptr, item, elw, buf, db, off, count, type, bType);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_writeOrdered(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jboolean db,
        jint off, jint count, jlong jType, jint bType, jlongArray stat)
{
    jboolean exception;
    MPI_Datatype type = (MPI_Datatype)jType;
    void *ptr;
    ompi_java_buffer_t *item;
    ompi_java_getReadPtr(&ptr, &item, elw, buf, db, off, count, type, bType);
    MPI_Status status;
    int rc = MPI_File_write_ordered((MPI_File)fh, ptr, count, type, &status);
    exception = ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseReadPtr(ptr, item, buf, db);
    
    if(!exception)
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_seekShared(
        JNIElw *elw, jobject jthis, jlong fh, jlong offset, jint whence)
{
    int rc = MPI_File_seek_shared((MPI_File)fh, (MPI_Offset)offset, whence);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_File_getPositionShared(
        JNIElw *elw, jobject jthis, jlong fh)
{
    MPI_Offset offset;
    int rc = MPI_File_get_position_shared((MPI_File)fh, &offset);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)offset;
}

JNIEXPORT void JNICALL Java_mpi_File_readAtAllBegin(
        JNIElw *elw, jobject jthis, jlong fh, jlong offset,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);

    int rc = MPI_File_read_at_all_begin((MPI_File)fh, (MPI_Offset)offset,
                                        ptr, count, (MPI_Datatype)type);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_File_readAtAllEnd(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jlongArray stat)
{
    MPI_Status status;
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    int rc = MPI_File_read_at_all_end((MPI_File)fh, ptr, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_writeAtAllBegin(
        JNIElw *elw, jobject jthis, jlong fh, jlong fileOffset,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);

    int rc = MPI_File_write_at_all_begin((MPI_File)fh, (MPI_Offset)fileOffset,
                                         ptr, count, (MPI_Datatype)type);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_File_writeAtAllEnd(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jlongArray stat)
{
    MPI_Status status;
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    int rc = MPI_File_write_at_all_end((MPI_File)fh, ptr, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_readAllBegin(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);

    int rc = MPI_File_read_all_begin(
             (MPI_File)fh, ptr, count, (MPI_Datatype)type);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_File_readAllEnd(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jlongArray stat)
{
    MPI_Status status;
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    int rc = MPI_File_read_all_end((MPI_File)fh, ptr, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_writeAllBegin(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);

    int rc = MPI_File_write_all_begin(
             (MPI_File)fh, ptr, count, (MPI_Datatype)type);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_File_writeAllEnd(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jlongArray stat)
{
    MPI_Status status;
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    int rc = MPI_File_write_all_end((MPI_File)fh, ptr, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_readOrderedBegin(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);

    int rc = MPI_File_read_ordered_begin(
             (MPI_File)fh, ptr, count, (MPI_Datatype)type);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_File_readOrderedEnd(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jlongArray stat)
{
    MPI_Status status;
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    int rc = MPI_File_read_ordered_end((MPI_File)fh, ptr, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT void JNICALL Java_mpi_File_writeOrderedBegin(
        JNIElw *elw, jobject jthis, jlong fh,
        jobject buf, jint count, jlong type)
{
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);

    int rc = MPI_File_write_ordered_begin(
             (MPI_File)fh, ptr, count, (MPI_Datatype)type);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_File_writeOrderedEnd(
        JNIElw *elw, jobject jthis, jlong fh, jobject buf, jlongArray stat)
{
    MPI_Status status;
    void *ptr = (*elw)->GetDirectBufferAddress(elw, buf);
    int rc = MPI_File_write_ordered_end((MPI_File)fh, ptr, &status);
    
    if(!ompi_java_exceptionCheck(elw, rc))
        ompi_java_status_set(elw, stat, &status);
}

JNIEXPORT jint JNICALL Java_mpi_File_getTypeExtent(
        JNIElw *elw, jobject jthis, jlong fh, jlong type)
{
    MPI_Aint extent;

    int rc = MPI_File_get_type_extent(
             (MPI_File)fh, (MPI_Datatype)type, &extent);

    ompi_java_exceptionCheck(elw, rc);
    return (int)extent;
}

JNIEXPORT void JNICALL Java_mpi_File_setAtomicity(
        JNIElw *elw, jobject jthis, jlong fh, jboolean atomicity)
{
    int rc = MPI_File_set_atomicity((MPI_File)fh, atomicity);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jboolean JNICALL Java_mpi_File_getAtomicity(
        JNIElw *elw, jobject jthis, jlong fh)
{
    int atomicity;
    int rc = MPI_File_get_atomicity((MPI_File)fh, &atomicity);
    ompi_java_exceptionCheck(elw, rc);
    return atomicity ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_mpi_File_sync(
        JNIElw *elw, jobject jthis, jlong fh)
{
    int rc = MPI_File_sync((MPI_File)fh);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_File_setErrhandler(
        JNIElw *elw, jobject jthis, jlong fh, jlong errhandler)
{
    int rc = MPI_File_set_errhandler(
             (MPI_File)fh, (MPI_Errhandler)errhandler);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_File_getErrhandler(
        JNIElw *elw, jobject jthis, jlong fh)
{
    MPI_Errhandler errhandler;
    int rc = MPI_File_get_errhandler((MPI_File)fh, &errhandler);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)errhandler;
}

JNIEXPORT void JNICALL Java_mpi_File_callErrhandler(
        JNIElw *elw, jobject jthis, jlong fh, jint errorCode)
{
    int rc = MPI_File_call_errhandler((MPI_File)fh, errorCode);
    ompi_java_exceptionCheck(elw, rc);
}
