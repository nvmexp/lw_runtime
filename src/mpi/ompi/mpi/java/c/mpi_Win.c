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
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
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
#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_Win.h"
#include "mpiJava.h"

JNIEXPORT jlong JNICALL Java_mpi_Win_createWin(
        JNIElw *elw, jobject jthis, jobject jBase,
        jint size, jint dispUnit, jlong info, jlong comm)
{
    void *base = (*elw)->GetDirectBufferAddress(elw, jBase);
    MPI_Win win;

    int rc = MPI_Win_create(base, (MPI_Aint)size, dispUnit,
                            (MPI_Info)info, (MPI_Comm)comm, &win);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)win;
}

JNIEXPORT jlong JNICALL Java_mpi_Win_allocateWin(JNIElw *elw, jobject jthis,
                                                 jint size, jint dispUnit, jlong info, jlong comm, jobject jBase)
{
    void *basePtr = (*elw)->GetDirectBufferAddress(elw, jBase);
    MPI_Win win;

    int rc = MPI_Win_allocate((MPI_Aint)size, dispUnit,
                              (MPI_Info)info, (MPI_Comm)comm, basePtr, &win);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)win;
}

JNIEXPORT jlong JNICALL Java_mpi_Win_allocateSharedWin(JNIElw *elw, jobject jthis,
                                                       jint size, jint dispUnit, jlong info, jlong comm, jobject jBase)
{
    void *basePtr = (*elw)->GetDirectBufferAddress(elw, jBase);
    MPI_Win win;

    int rc = MPI_Win_allocate_shared((MPI_Aint)size, dispUnit,
                                     (MPI_Info)info, (MPI_Comm)comm, basePtr, &win);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)win;
}

JNIEXPORT jlong JNICALL Java_mpi_Win_createDynamicWin(
        JNIElw *elw, jobject jthis,
        jlong info, jlong comm)
{
    MPI_Win win;

    int rc = MPI_Win_create_dynamic(
                            (MPI_Info)info, (MPI_Comm)comm, &win);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)win;
}

JNIEXPORT void JNICALL Java_mpi_Win_attach(
        JNIElw *elw, jobject jthis, jlong win, jobject jBase,
        jint size)
{
    void *base = (*elw)->GetDirectBufferAddress(elw, jBase);

    int rc = MPI_Win_attach((MPI_Win)win, base, (MPI_Aint)size);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_detach(
        JNIElw *elw, jobject jthis, jlong win, jobject jBase)
{
    void *base = (*elw)->GetDirectBufferAddress(elw, jBase);

    int rc = MPI_Win_detach((MPI_Win)win, base);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_Win_getGroup(
        JNIElw *elw, jobject jthis, jlong win)
{
    MPI_Group group;
    int rc = MPI_Win_get_group((MPI_Win)win, &group);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)group;
}

JNIEXPORT void JNICALL Java_mpi_Win_put(
        JNIElw *elw, jobject jthis, jlong win, jobject origin,
        jint orgCount, jlong orgType, jint targetRank, jint targetDisp,
        jint targetCount, jlong targetType, jint baseType)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);

    int rc = MPI_Put(orgPtr, orgCount, (MPI_Datatype)orgType,
                     targetRank, (MPI_Aint)targetDisp, targetCount,
                     (MPI_Datatype)targetType, (MPI_Win)win);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_get(
        JNIElw *elw, jobject jthis, jlong win, jobject origin,
        jint orgCount, jlong orgType, jint targetRank, jint targetDisp,
        jint targetCount, jlong targetType, jint baseType)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);

    int rc = MPI_Get(orgPtr, orgCount, (MPI_Datatype)orgType,
                     targetRank, (MPI_Aint)targetDisp, targetCount,
                     (MPI_Datatype)targetType, (MPI_Win)win);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_aclwmulate(
        JNIElw *elw, jobject jthis, jlong win,
        jobject origin, jint orgCount, jlong orgType,
        jint targetRank, jint targetDisp, jint targetCount, jlong targetType,
        jobject jOp, jlong hOp, jint baseType)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);
    MPI_Op op = ompi_java_op_getHandle(elw, jOp, hOp, baseType);

    int rc = MPI_Aclwmulate(orgPtr, orgCount, (MPI_Datatype)orgType,
                            targetRank, (MPI_Aint)targetDisp, targetCount,
                            (MPI_Datatype)targetType, op, (MPI_Win)win);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_fence(
        JNIElw *elw, jobject jthis, jlong win, jint assertion)
{
    int rc = MPI_Win_fence(assertion, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_start(
        JNIElw *elw, jobject jthis, jlong win, jlong group, jint assertion)
{
    int rc = MPI_Win_start((MPI_Group)group, assertion, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_complete(
        JNIElw *elw, jobject jthis, jlong win)
{
    int rc = MPI_Win_complete((MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_post(
        JNIElw *elw, jobject jthis, jlong win, jlong group, jint assertion)
{
    int rc = MPI_Win_post((MPI_Group)group, assertion, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_waitFor(
        JNIElw *elw, jobject jthis, jlong win)
{
    int rc = MPI_Win_wait((MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jboolean JNICALL Java_mpi_Win_test(
        JNIElw *elw, jobject jthis, jlong win)
{
    int flag;
    int rc = MPI_Win_test((MPI_Win)win, &flag);
    ompi_java_exceptionCheck(elw, rc);
    return flag ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_mpi_Win_lock(
        JNIElw *elw, jobject jthis, jlong win,
        jint lockType, jint rank, jint assertion)
{
    int rc = MPI_Win_lock(lockType, rank, assertion, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_unlock(
        JNIElw *elw, jobject jthis, jlong win, jint rank)
{
    int rc = MPI_Win_unlock(rank, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_setErrhandler(
        JNIElw *elw, jobject jthis, jlong win, jlong errhandler)
{
    int rc = MPI_Win_set_errhandler(
             (MPI_Win)win, (MPI_Errhandler)errhandler);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_Win_getErrhandler(
        JNIElw *elw, jobject jthis, jlong win)
{
    MPI_Errhandler errhandler;
    int rc = MPI_Win_get_errhandler((MPI_Win)win, &errhandler);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)errhandler;
}

JNIEXPORT void JNICALL Java_mpi_Win_callErrhandler(
        JNIElw *elw, jobject jthis, jlong win, jint errorCode)
{
    int rc = MPI_Win_call_errhandler((MPI_Win)win, errorCode);
    ompi_java_exceptionCheck(elw, rc);
}

static int winCopyAttr(MPI_Win oldwin, int keyval, void *extraState,
                       void *attrValIn, void *attrValOut, int *flag)
{
    return ompi_java_attrCopy(attrValIn, attrValOut, flag);
}

static int winDeleteAttr(MPI_Win oldwin, int keyval,
                         void *attrVal, void *extraState)
{
    return ompi_java_attrDelete(attrVal);
}

JNIEXPORT jint JNICALL Java_mpi_Win_createKeyval_1jni(JNIElw *elw, jclass clazz)
{
    int rc, keyval;
    rc = MPI_Win_create_keyval(winCopyAttr, winDeleteAttr, &keyval, NULL);
    ompi_java_exceptionCheck(elw, rc);
    return keyval;
}

JNIEXPORT void JNICALL Java_mpi_Win_freeKeyval_1jni(
        JNIElw *elw, jclass clazz, jint keyval)
{
    int rc = MPI_Win_free_keyval((int*)(&keyval));
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_setAttr(
        JNIElw *elw, jobject jthis, jlong win, jint keyval, jbyteArray jval)
{
    void *cval = ompi_java_attrSet(elw, jval);
    int rc = MPI_Win_set_attr((MPI_Win)win, keyval, cval);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jobject JNICALL Java_mpi_Win_getAttr(
        JNIElw *elw, jobject jthis, jlong win, jint keyval)
{
    int flag;
    void *val;
    int rc = MPI_Win_get_attr((MPI_Win)win, keyval, &val, &flag);

    if(ompi_java_exceptionCheck(elw, rc) || !flag)
        return NULL;

    switch(keyval)
    {
        case MPI_WIN_SIZE:
            return ompi_java_Integer_valueOf(elw, (jint)(*((MPI_Aint*)val)));
        case MPI_WIN_DISP_UNIT:
            return ompi_java_Integer_valueOf(elw, (jint)(*((int*)val)));
        case MPI_WIN_BASE:
            return ompi_java_Long_valueOf(elw, (jlong)val);
        default:
            return ompi_java_attrGet(elw, val);
    }
}

JNIEXPORT void JNICALL Java_mpi_Win_deleteAttr(
        JNIElw *elw, jobject jthis, jlong win, jint keyval)
{
    int rc = MPI_Win_delete_attr((MPI_Win)win, keyval);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_Win_free(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Win win = (MPI_Win)handle;
    int rc = MPI_Win_free(&win);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)win;
}

JNIEXPORT jlong JNICALL Java_mpi_Win_getInfo(
        JNIElw *elw, jobject jthis, jlong handle)
{
    MPI_Win win = (MPI_Win)handle;
    MPI_Info info;
    int rc = MPI_Win_get_info((MPI_Win)win, &info);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)info;
}

JNIEXPORT void JNICALL Java_mpi_Win_setInfo(
        JNIElw *elw, jobject jthis, jlong handle, jlong i)
{
    MPI_Win win = (MPI_Win)handle;
    MPI_Info info = (MPI_Info)i;
    int rc = MPI_Win_set_info(win, info);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_Win_rPut(JNIElw *elw, jobject jthis,
    jlong win, jobject origin_addr, jint origin_count, jlong origin_type,
    jint target_rank, jint target_disp, jint target_count, jlong target_datatype,
    jint basetype)
{
    void *origPtr = ompi_java_getDirectBufferAddress(elw, origin_addr);
    MPI_Request request;

    int rc = MPI_Rput(origPtr, origin_count, (MPI_Datatype)origin_type,
                      target_rank, (MPI_Aint)target_disp, target_count, (MPI_Datatype)target_datatype,
                      (MPI_Win)win, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_Win_rGet(JNIElw *elw, jobject jthis, jlong win,
    jobject origin, jint orgCount, jlong orgType, jint targetRank, jint targetDisp,
    jint targetCount, jlong targetType, jint base)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);
    MPI_Request request;

    int rc = MPI_Rget(orgPtr, orgCount, (MPI_Datatype)orgType,
                      targetRank, (MPI_Aint)targetDisp, targetCount,
                      (MPI_Datatype)targetType, (MPI_Win)win, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jlong JNICALL Java_mpi_Win_rAclwmulate(JNIElw *elw, jobject jthis, jlong win,
    jobject origin, jint orgCount, jlong orgType, jint targetRank, jint targetDisp,
    jint targetCount, jlong targetType, jobject jOp, jlong hOp, jint baseType)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);
    MPI_Op op = ompi_java_op_getHandle(elw, jOp, hOp, baseType);
    MPI_Request request;

    int rc = MPI_Raclwmulate(orgPtr, orgCount, (MPI_Datatype)orgType,
                            targetRank, (MPI_Aint)targetDisp, targetCount,
                            (MPI_Datatype)targetType, op, (MPI_Win)win, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT void JNICALL Java_mpi_Win_getAclwmulate(JNIElw *elw, jobject jthis, jlong win,
                            jobject origin, jint orgCount, jlong orgType, jobject resultBuff, jint resultCount,
                                                  jlong resultType, jint targetRank, jint targetDisp, jint targetCount, jlong targetType,
                                                  jobject jOp, jlong hOp, jint baseType)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);
    void *resultPtr = (*elw)->GetDirectBufferAddress(elw, resultBuff);
    MPI_Op op = ompi_java_op_getHandle(elw, jOp, hOp, baseType);

    int rc = MPI_Get_aclwmulate(orgPtr, orgCount, (MPI_Datatype)orgType,
                                resultPtr, resultCount, (MPI_Datatype)resultType,
                                targetRank, (MPI_Aint)targetDisp, targetCount,
                                (MPI_Datatype)targetType, op, (MPI_Win)win);

    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT jlong JNICALL Java_mpi_Win_rGetAclwmulate(JNIElw *elw, jobject jthis, jlong win,
                                                    jobject origin, jint orgCount, jlong orgType, jobject resultBuff, jint resultCount,
                                                    jlong resultType, jint targetRank, jint targetDisp, jint targetCount, jlong targetType,
                                                    jobject jOp, jlong hOp, jint baseType)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);
    void *resultPtr = (*elw)->GetDirectBufferAddress(elw, resultBuff);
    MPI_Op op = ompi_java_op_getHandle(elw, jOp, hOp, baseType);
    MPI_Request request;

    int rc = MPI_Rget_aclwmulate(orgPtr, orgCount, (MPI_Datatype)orgType,
                                 resultPtr, resultCount, (MPI_Datatype)resultType,
                                 targetRank, (MPI_Aint)targetDisp, targetCount,
                                 (MPI_Datatype)targetType, op, (MPI_Win)win, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT void JNICALL Java_mpi_Win_lockAll(JNIElw *elw, jobject jthis, jlong win, jint assertion)
{
    int rc = MPI_Win_lock_all(assertion, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_unlockAll(JNIElw *elw, jobject jthis, jlong win)
{
    int rc = MPI_Win_unlock_all((MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_sync(JNIElw *elw, jobject jthis, jlong win)
{
    int rc = MPI_Win_sync((MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_flush(JNIElw *elw, jobject jthis, jlong win, jint targetRank)
{
    int rc = MPI_Win_flush(targetRank, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_flushAll(JNIElw *elw, jobject jthis, jlong win)
{
    int rc = MPI_Win_flush_all((MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_compareAndSwap (JNIElw *elw, jobject jthis, jlong win, jobject origin,
                                                    jobject compareAddr, jobject resultAddr, jlong dataType, jint targetRank, jint targetDisp)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);
    void *compPtr = (*elw)->GetDirectBufferAddress(elw, compareAddr);
    void *resultPtr = (*elw)->GetDirectBufferAddress(elw, resultAddr);

    int rc = MPI_Compare_and_swap(orgPtr, compPtr, resultPtr, (MPI_Datatype)dataType, 
	targetRank, targetDisp, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_fetchAndOp(JNIElw *elw, jobject jthis, jlong win, jobject origin,
                                               jobject resultAddr, jlong dataType, jint targetRank, jint targetDisp, jobject jOp, jlong hOp, jint baseType)
{
    void *orgPtr = (*elw)->GetDirectBufferAddress(elw, origin);
    void *resultPtr = (*elw)->GetDirectBufferAddress(elw, resultAddr);
    MPI_Op op = ompi_java_op_getHandle(elw, jOp, hOp, baseType);

    int rc = MPI_Fetch_and_op(orgPtr, resultPtr, (MPI_Datatype)dataType, targetRank, 
	targetDisp, op, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_flushLocal(JNIElw *elw, jobject jthis, jlong win, jint targetRank)
{
    int rc = MPI_Win_flush_local(targetRank, (MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_flushLocalAll(JNIElw *elw, jobject jthis, jlong win)
{
    int rc = MPI_Win_flush_local_all((MPI_Win)win);
    ompi_java_exceptionCheck(elw, rc);
}

JNIEXPORT void JNICALL Java_mpi_Win_setName(
        JNIElw *elw, jobject jthis, jlong handle, jstring jname)
{
    const char *name = (*elw)->GetStringUTFChars(elw, jname, NULL);
    int rc = MPI_Win_set_name((MPI_Win)handle, (char*)name);
    ompi_java_exceptionCheck(elw, rc);
    (*elw)->ReleaseStringUTFChars(elw, jname, name);
}

JNIEXPORT jstring JNICALL Java_mpi_Win_getName(
        JNIElw *elw, jobject jthis, jlong handle)
{
    char name[MPI_MAX_OBJECT_NAME];
    int len;
    int rc = MPI_Win_get_name((MPI_Win)handle, name, &len);

    if(ompi_java_exceptionCheck(elw, rc))
        return NULL;

    return (*elw)->NewStringUTF(elw, name);
}
