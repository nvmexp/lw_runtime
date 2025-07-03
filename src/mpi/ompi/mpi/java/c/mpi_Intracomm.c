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
 * File         : mpi_Intracomm.c
 * Headerfile   : mpi_Intracomm.h
 * Author       : Xinying Li, Bryan Carpenter
 * Created      : Thu Apr  9 12:22:15 1998
 * Revision     : $Revision: 1.10 $
 * Updated      : $Date: 2003/01/16 16:39:34 $
 * Copyright: Northeast Parallel Architectures Center
 *            at Syralwse University 1998
 */

#include "ompi_config.h"

#include <stdlib.h>
#include <string.h>
#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_Comm.h"
#include "mpi_Intracomm.h"
#include "mpiJava.h"

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_split(
        JNIElw *elw, jobject jthis, jlong comm, jint colour, jint key)
{
    MPI_Comm newcomm;
    int rc = MPI_Comm_split((MPI_Comm)comm, colour, key, &newcomm);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)newcomm;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_splitType(
        JNIElw *elw, jobject jthis, jlong comm, jint splitType, jint key, jlong info)
{
    MPI_Comm newcomm;
    int rc = MPI_Comm_split_type((MPI_Comm)comm, splitType, key, (MPI_Info)info, &newcomm);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)newcomm;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_create(
        JNIElw *elw, jobject jthis, jlong comm, jlong group)
{
    MPI_Comm newcomm;
    int rc = MPI_Comm_create((MPI_Comm)comm, (MPI_Group)group, &newcomm);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)newcomm;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_createGroup(
        JNIElw *elw, jobject jthis, jlong comm, jlong group, int tag)
{
    MPI_Comm newcomm;
    int rc = MPI_Comm_create_group((MPI_Comm)comm, (MPI_Group)group, tag, &newcomm);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)newcomm;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_createCart(
        JNIElw *elw, jobject jthis, jlong comm,
        jintArray dims, jbooleanArray periods, jboolean reorder)
{
    jint *jDims;
    int  *cDims;
    ompi_java_getIntArray(elw, dims, &jDims, &cDims);

    jboolean *jPeriods;
    int      *cPeriods;
    ompi_java_getBooleanArray(elw, periods, &jPeriods, &cPeriods);

    int ndims = (*elw)->GetArrayLength(elw, dims);
    MPI_Comm cart;

    int rc = MPI_Cart_create((MPI_Comm)comm, ndims, cDims,
                             cPeriods, reorder, &cart);

    ompi_java_exceptionCheck(elw, rc);
    ompi_java_forgetIntArray(elw, dims, jDims, cDims);
    ompi_java_forgetBooleanArray(elw, periods, jPeriods, cPeriods);
    return (jlong)cart;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_createGraph(
        JNIElw *elw, jobject jthis, jlong comm,
        jintArray index, jintArray edges, jboolean reorder)
{
    MPI_Comm graph;
    int nnodes = (*elw)->GetArrayLength(elw, index);

    jint *jIndex, *jEdges;
    int  *cIndex, *cEdges;
    ompi_java_getIntArray(elw, index, &jIndex, &cIndex);
    ompi_java_getIntArray(elw, edges, &jEdges, &cEdges);

    int rc = MPI_Graph_create((MPI_Comm)comm,
             nnodes, cIndex, cEdges, reorder, &graph);

    ompi_java_exceptionCheck(elw, rc);
    ompi_java_forgetIntArray(elw, index, jIndex, cIndex);
    ompi_java_forgetIntArray(elw, edges, jEdges, cEdges);
    return (jlong)graph;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_createDistGraph(
        JNIElw *elw, jobject jthis, jlong comm, jintArray sources,
        jintArray degrees, jintArray destins, jintArray weights,
        jlong info, jboolean reorder, jboolean weighted)
{
    MPI_Comm graph;
    int nnodes = (*elw)->GetArrayLength(elw, sources);

    jint *jSources, *jDegrees, *jDestins, *jWeights = NULL;
    int  *cSources, *cDegrees, *cDestins, *cWeights = MPI_UNWEIGHTED;
    ompi_java_getIntArray(elw, sources, &jSources, &cSources);
    ompi_java_getIntArray(elw, degrees, &jDegrees, &cDegrees);
    ompi_java_getIntArray(elw, destins, &jDestins, &cDestins);

    if(weighted)
        ompi_java_getIntArray(elw, weights, &jWeights, &cWeights);

    int rc = MPI_Dist_graph_create((MPI_Comm)comm,
             nnodes, cSources, cDegrees, cDestins, cWeights,
             (MPI_Info)info, reorder, &graph);

    ompi_java_exceptionCheck(elw, rc);
    ompi_java_forgetIntArray(elw, sources, jSources, cSources);
    ompi_java_forgetIntArray(elw, degrees, jDegrees, cDegrees);
    ompi_java_forgetIntArray(elw, destins, jDestins, cDestins);

    if(weighted)
        ompi_java_forgetIntArray(elw, weights, jWeights, cWeights);

    return (jlong)graph;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_createDistGraphAdjacent(
        JNIElw *elw, jobject jthis, jlong comm, jintArray sources,
        jintArray srcWeights, jintArray destins, jintArray desWeights,
        jlong info, jboolean reorder, jboolean weighted)
{
    MPI_Comm graph;

    int inDegree  = (*elw)->GetArrayLength(elw, sources),
        outDegree = (*elw)->GetArrayLength(elw, destins);

    jint *jSources, *jDestins, *jSrcWeights, *jDesWeights;
    int  *cSources, *cDestins, *cSrcWeights, *cDesWeights;
    ompi_java_getIntArray(elw, sources, &jSources, &cSources);
    ompi_java_getIntArray(elw, destins, &jDestins, &cDestins);

    if(weighted)
    {
        ompi_java_getIntArray(elw, srcWeights, &jSrcWeights, &cSrcWeights);
        ompi_java_getIntArray(elw, desWeights, &jDesWeights, &cDesWeights);
    }
    else
    {
        jSrcWeights = jDesWeights = NULL;
        cSrcWeights = cDesWeights = MPI_UNWEIGHTED;
    }

    int rc = MPI_Dist_graph_create_adjacent((MPI_Comm)comm,
             inDegree, cSources, cSrcWeights, outDegree, cDestins,
             cDesWeights, (MPI_Info)info, reorder, &graph);

    ompi_java_exceptionCheck(elw, rc);
    ompi_java_forgetIntArray(elw, sources, jSources, cSources);
    ompi_java_forgetIntArray(elw, destins, jDestins, cDestins);

    if(weighted)
    {
        ompi_java_forgetIntArray(elw, srcWeights, jSrcWeights, cSrcWeights);
        ompi_java_forgetIntArray(elw, desWeights, jDesWeights, cDesWeights);
    }

    return (jlong)graph;
}

JNIEXPORT void JNICALL Java_mpi_Intracomm_scan(
        JNIElw *elw, jobject jthis, jlong jComm,
        jobject sBuf, jboolean sdb, jint sOff,
        jobject rBuf, jboolean rdb, jint rOff, jint count,
        jlong jType, jint bType, jobject jOp, jlong hOp)
{
    MPI_Comm     comm = (MPI_Comm)jComm;
    MPI_Datatype type = (MPI_Datatype)jType;

    void *sPtr, *rPtr;
    ompi_java_buffer_t *sItem, *rItem;

    if(sBuf == NULL)
    {
        sPtr = MPI_IN_PLACE;
        ompi_java_getReadPtr(&rPtr,&rItem,elw,rBuf,rdb,rOff,count,type,bType);
    }
    else
    {
        ompi_java_getReadPtr(&sPtr,&sItem,elw,sBuf,sdb,sOff,count,type,bType);
        ompi_java_getWritePtr(&rPtr, &rItem, elw, rBuf, rdb, count, type);
    }

    MPI_Op op = ompi_java_op_getHandle(elw, jOp, hOp, bType);
    int rc = MPI_Scan(sPtr, rPtr, count, type, op, comm);
    ompi_java_exceptionCheck(elw, rc);

    if(sBuf != NULL)
        ompi_java_releaseReadPtr(sPtr, sItem, sBuf, sdb);

    ompi_java_releaseWritePtr(rPtr,rItem,elw,rBuf,rdb,rOff,count,type,bType);
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_iScan(
        JNIElw *elw, jobject jthis, jlong comm,
        jobject sendBuf, jobject recvBuf, jint count,
        jlong type, int baseType, jobject jOp, jlong hOp)
{
    void *sPtr, *rPtr;
    MPI_Request request;

    if(sendBuf == NULL)
        sPtr = MPI_IN_PLACE;
    else
        sPtr = (*elw)->GetDirectBufferAddress(elw, sendBuf);

    rPtr = (*elw)->GetDirectBufferAddress(elw, recvBuf);

    int rc = MPI_Iscan(sPtr, rPtr, count, (MPI_Datatype)type,
                       ompi_java_op_getHandle(elw, jOp, hOp, baseType),
                       (MPI_Comm)comm, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT void JNICALL Java_mpi_Intracomm_exScan(
        JNIElw *elw, jobject jthis, jlong jComm,
        jobject sBuf, jboolean sdb, jint sOff,
        jobject rBuf, jboolean rdb, jint rOff, jint count,
        jlong jType, int bType, jobject jOp, jlong hOp)
{
    MPI_Comm     comm = (MPI_Comm)jComm;
    MPI_Datatype type = (MPI_Datatype)jType;

    void *sPtr, *rPtr;
    ompi_java_buffer_t *sItem, *rItem;

    if(sBuf == NULL)
    {
        sPtr = MPI_IN_PLACE;
        ompi_java_getReadPtr(&rPtr,&rItem,elw,rBuf,rdb,rOff,count,type,bType);
    }
    else
    {
        ompi_java_getReadPtr(&sPtr,&sItem,elw,sBuf,sdb,sOff,count,type,bType);
        ompi_java_getWritePtr(&rPtr, &rItem, elw, rBuf, rdb, count, type);
    }

    MPI_Op op = ompi_java_op_getHandle(elw, jOp, hOp, bType);
    int rc = MPI_Exscan(sPtr, rPtr, count, type, op, comm);
    ompi_java_exceptionCheck(elw, rc);

    if(sBuf != NULL)
        ompi_java_releaseReadPtr(sPtr, sItem, sBuf, sdb);

    ompi_java_releaseWritePtr(rPtr,rItem,elw,rBuf,rdb,rOff,count,type,bType);
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_iExScan(
        JNIElw *elw, jobject jthis, jlong comm,
        jobject sendBuf, jobject recvBuf, jint count,
        jlong type, int bType, jobject jOp, jlong hOp)
{
    void *sPtr, *rPtr;

    if(sendBuf == NULL)
        sPtr = MPI_IN_PLACE;
    else
        sPtr = (*elw)->GetDirectBufferAddress(elw, sendBuf);

    rPtr = (*elw)->GetDirectBufferAddress(elw, recvBuf);
    MPI_Request request;

    int rc = MPI_Iexscan(sPtr, rPtr, count, (MPI_Datatype)type,
                         ompi_java_op_getHandle(elw, jOp, hOp, bType),
                         (MPI_Comm)comm, &request);

    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT jstring JNICALL Java_mpi_Intracomm_openPort(
                          JNIElw *elw, jclass clazz, jlong info)
{
    char port[MPI_MAX_PORT_NAME + 1];
    int rc = MPI_Open_port((MPI_Info)info, port);

    return ompi_java_exceptionCheck(elw, rc)
           ? NULL : (*elw)->NewStringUTF(elw, port);
}

JNIEXPORT void JNICALL Java_mpi_Intracomm_closePort_1jni(
                       JNIElw *elw, jclass clazz, jstring jport)
{
    const char *port = (*elw)->GetStringUTFChars(elw, jport, NULL);
    int rc = MPI_Close_port((char*)port);
    ompi_java_exceptionCheck(elw, rc);
    (*elw)->ReleaseStringUTFChars(elw, jport, port);
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_accept(
        JNIElw *elw, jobject jthis, jlong comm,
        jstring jport, jlong info, jint root)
{
    const char *port = jport == NULL ? NULL :
                       (*elw)->GetStringUTFChars(elw, jport, NULL);
    MPI_Comm newComm;

    int rc = MPI_Comm_accept((char*)port, (MPI_Info)info,
                             root, (MPI_Comm)comm, &newComm);

    ompi_java_exceptionCheck(elw, rc);

    if(jport != NULL)
        (*elw)->ReleaseStringUTFChars(elw, jport, port);

    return (jlong)newComm;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_connect(
        JNIElw *elw, jobject jthis, jlong comm,
        jstring jport, jlong info, jint root)
{
    const char *port = jport == NULL ? NULL :
                       (*elw)->GetStringUTFChars(elw, jport, NULL);
    MPI_Comm newComm;

    int rc = MPI_Comm_connect((char*)port, (MPI_Info)info,
                              root, (MPI_Comm)comm, &newComm);

    ompi_java_exceptionCheck(elw, rc);

    if(jport != NULL)
        (*elw)->ReleaseStringUTFChars(elw, jport, port);

    return (jlong)newComm;
}

JNIEXPORT void JNICALL Java_mpi_Intracomm_publishName(
        JNIElw *elw, jclass clazz, jstring jservice, jlong info, jstring jport)
{
    const char *service = (*elw)->GetStringUTFChars(elw, jservice, NULL),
               *port    = (*elw)->GetStringUTFChars(elw, jport,    NULL);

    int rc = MPI_Publish_name((char*)service, (MPI_Info)info, (char*)port);
    ompi_java_exceptionCheck(elw, rc);

    (*elw)->ReleaseStringUTFChars(elw, jservice, service);
    (*elw)->ReleaseStringUTFChars(elw, jport,    port);
}

JNIEXPORT void JNICALL Java_mpi_Intracomm_unpublishName(
        JNIElw *elw, jclass clazz, jstring jservice, jlong info, jstring jport)
{
    const char *service = (*elw)->GetStringUTFChars(elw, jservice, NULL),
               *port    = (*elw)->GetStringUTFChars(elw, jport,    NULL);

    int rc = MPI_Unpublish_name((char*)service, (MPI_Info)info, (char*)port);
    ompi_java_exceptionCheck(elw, rc);

    (*elw)->ReleaseStringUTFChars(elw, jservice, service);
    (*elw)->ReleaseStringUTFChars(elw, jport,    port);
}

JNIEXPORT jstring JNICALL Java_mpi_Intracomm_lookupName(
        JNIElw *elw, jclass clazz, jstring jservice, jlong info)
{
    char port[MPI_MAX_PORT_NAME + 1];
    const char *service = (*elw)->GetStringUTFChars(elw, jservice, NULL);

    int rc = MPI_Lookup_name((char*)service, (MPI_Info)info, port);
    (*elw)->ReleaseStringUTFChars(elw, jservice, service);

    return ompi_java_exceptionCheck(elw, rc)
           ? NULL : (*elw)->NewStringUTF(elw, port);
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_spawn(
        JNIElw *elw, jobject jthis, jlong comm, jstring jCommand,
        jobjectArray jArgv, jint maxprocs, jlong info, jint root,
        jintArray errCodes)
{
    int i, rc;
    MPI_Comm intercomm;
    const char* command = (*elw)->GetStringUTFChars(elw, jCommand, NULL);

    jint *jErrCodes;
    int  *cErrCodes = MPI_ERRCODES_IGNORE;

    if(errCodes != NULL)
        ompi_java_getIntArray(elw, errCodes, &jErrCodes, &cErrCodes);

    char **argv = MPI_ARGV_NULL;

    if(jArgv != NULL)
    {
        jsize argvLength = (*elw)->GetArrayLength(elw, jArgv);
        argv = (char**)calloc(argvLength + 1, sizeof(char*));

        for(i = 0; i < argvLength; i++)
        {
            jstring a = (*elw)->GetObjectArrayElement(elw, jArgv, i);
            argv[i] = strdup((*elw)->GetStringUTFChars(elw, a, NULL));
            (*elw)->DeleteLocalRef(elw, a);
        }

        argv[argvLength] = NULL;
    }

    rc = MPI_Comm_spawn((char*)command, argv, maxprocs, (MPI_Info)info,
                        root, (MPI_Comm)comm, &intercomm, cErrCodes);

    ompi_java_exceptionCheck(elw, rc);

    if(jArgv != NULL)
    {
        jsize argvLength = (*elw)->GetArrayLength(elw, jArgv);

        for(i = 0; i < argvLength; i++)
        {
            jstring a = (*elw)->GetObjectArrayElement(elw, jArgv, i);
            (*elw)->ReleaseStringUTFChars(elw, a, argv[i]);
            (*elw)->DeleteLocalRef(elw, a);
        }

        free(argv);
    }

    if(errCodes != NULL)
        ompi_java_releaseIntArray(elw, errCodes, jErrCodes, cErrCodes);

    (*elw)->ReleaseStringUTFChars(elw, jCommand, command);
    return (jlong)intercomm;
}

JNIEXPORT jlong JNICALL Java_mpi_Intracomm_spawnMultiple(
        JNIElw *elw, jobject jthis, jlong comm, jobjectArray jCommands,
        jobjectArray jArgv, jintArray maxProcs, jlongArray info,
        jint root, jintArray errCodes)
{
    int i, rc;
    MPI_Comm intercomm;
    jlong *jInfo = (*elw)->GetLongArrayElements(elw, info, NULL);

    jint *jMaxProcs, *jErrCodes;
    int  *cMaxProcs, *cErrCodes = MPI_ERRCODES_IGNORE;
    ompi_java_getIntArray(elw, maxProcs, &jMaxProcs, &cMaxProcs);

    if(errCodes != NULL)
        ompi_java_getIntArray(elw, errCodes, &jErrCodes, &cErrCodes);

    int commandsLength = (*elw)->GetArrayLength(elw, jCommands),
        infoLength     = (*elw)->GetArrayLength(elw, info);

    char **commands = calloc(commandsLength, sizeof(char*)),
         ***argv    = MPI_ARGVS_NULL;
    MPI_Info *cInfo = calloc(infoLength, sizeof(MPI_Info));

    for(i = 0; i < infoLength; i++)
        cInfo[i] = (MPI_Info)jInfo[i];

    for(i = 0; i < commandsLength; i++)
    {
        jstring a = (*elw)->GetObjectArrayElement(elw, jCommands, i);
        commands[i] = (char*)(*elw)->GetStringUTFChars(elw, a, NULL);
        (*elw)->DeleteLocalRef(elw, a);
    }

    if(jArgv != NULL)
    {
        int argvLength = (*elw)->GetArrayLength(elw, jArgv);
        argv = calloc(argvLength, sizeof(char**));

        for(i = 0; i < argvLength; i++)
        {
            jobjectArray arr = (*elw)->GetObjectArrayElement(elw, jArgv, i);
            int j, length = (*elw)->GetArrayLength(elw, arr);
            argv[i] = calloc(length + 1, sizeof(char*));

            for(j = 0; j < length; j++)
            {
                jstring a = (*elw)->GetObjectArrayElement(elw, arr, j);
                argv[i][j] = (char*)(*elw)->GetStringUTFChars(elw, a, NULL);
                (*elw)->DeleteLocalRef(elw, a);
            }

            argv[i][length] = NULL;
            (*elw)->DeleteLocalRef(elw, arr);
        }
    }

    rc = MPI_Comm_spawn_multiple(
            commandsLength, commands, argv, cMaxProcs, cInfo,
            root, (MPI_Comm)comm, &intercomm, cErrCodes);

    ompi_java_exceptionCheck(elw, rc);

    if(jArgv != NULL)
    {
        int argvLength = (*elw)->GetArrayLength(elw, jArgv);

        for(i = 0; i < argvLength; i++)
        {
            jobjectArray arr = (*elw)->GetObjectArrayElement(elw, jArgv, i);
            int j, length = (*elw)->GetArrayLength(elw, arr);

            for(j = 0; j < length; j++)
            {
                jstring a = (*elw)->GetObjectArrayElement(elw, arr, j);
                (*elw)->ReleaseStringUTFChars(elw, a, argv[i][j]);
                (*elw)->DeleteLocalRef(elw, a);
            }

            (*elw)->DeleteLocalRef(elw, arr);
            free(argv[i]);
        }

        free(argv);
    }

    for(i = 0; i < commandsLength; i++)
    {
        jstring a = (*elw)->GetObjectArrayElement(elw, jCommands, i);
        (*elw)->ReleaseStringUTFChars(elw, a, commands[i]);
        (*elw)->DeleteLocalRef(elw, a);
    }

    if(errCodes != NULL)
        ompi_java_releaseIntArray(elw, errCodes, jErrCodes, cErrCodes);

    free(cInfo);
    free(commands);
    (*elw)->ReleaseLongArrayElements(elw, info, jInfo, JNI_ABORT);
    ompi_java_forgetIntArray(elw, maxProcs, jMaxProcs, cMaxProcs);
    return (jlong)intercomm;
}
