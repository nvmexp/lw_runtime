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
 * File         : mpi_GraphComm.c
 * Headerfile   : mpi_GraphComm.h
 * Author       : Xinying Li
 * Created      : Thu Apr  9 12:22:15 1998
 * Revision     : $Revision: 1.2 $
 * Updated      : $Date: 2003/01/16 16:39:34 $
 * Copyright: Northeast Parallel Architectures Center
 *            at Syralwse University 1998
 */
#include "ompi_config.h"

#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_GraphComm.h"
#include "mpiJava.h"

JNIEXPORT void JNICALL Java_mpi_GraphComm_init(JNIElw *elw, jclass clazz)
{
    ompi_java.GraphParmsInit = (*elw)->GetMethodID(elw,
            ompi_java.GraphParmsClass, "<init>", "([I[I)V");
    ompi_java.DistGraphNeighborsInit = (*elw)->GetMethodID(elw,
            ompi_java.DistGraphNeighborsClass, "<init>", "([I[I[I[IZ)V");
}

JNIEXPORT jobject JNICALL Java_mpi_GraphComm_getDims(
        JNIElw *elw, jobject jthis, jlong comm)
{
    int maxInd, maxEdg;
    int rc = MPI_Graphdims_get((MPI_Comm)comm, &maxInd, &maxEdg);

    if(ompi_java_exceptionCheck(elw, rc))
        return NULL;

    jintArray index = (*elw)->NewIntArray(elw, maxInd),
              edges = (*elw)->NewIntArray(elw, maxEdg);

    jint *jIndex, *jEdges;
    int  *cIndex, *cEdges;
    ompi_java_getIntArray(elw, index, &jIndex, &cIndex);
    ompi_java_getIntArray(elw, edges, &jEdges, &cEdges);

    rc = MPI_Graph_get((MPI_Comm)comm, maxInd, maxEdg, cIndex, cEdges);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_releaseIntArray(elw, index, jIndex, cIndex);
    ompi_java_releaseIntArray(elw, edges, jEdges, cEdges);

    return (*elw)->NewObject(elw, ompi_java.GraphParmsClass,
                             ompi_java.GraphParmsInit, index, edges);
}

JNIEXPORT jintArray JNICALL Java_mpi_GraphComm_getNeighbors(
        JNIElw *elw, jobject jthis, jlong comm, jint rank)
{
    int maxNs;
    int rc = MPI_Graph_neighbors_count((MPI_Comm)comm, rank, &maxNs);

    if(ompi_java_exceptionCheck(elw, rc))
        return NULL;

    jintArray neighbors = (*elw)->NewIntArray(elw, maxNs);
    jint *jNeighbors;
    int  *cNeighbors;
    ompi_java_getIntArray(elw, neighbors, &jNeighbors, &cNeighbors);

    rc = MPI_Graph_neighbors((MPI_Comm)comm, rank, maxNs, cNeighbors);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_releaseIntArray(elw, neighbors, jNeighbors, cNeighbors);
    return neighbors;
}

JNIEXPORT jobject JNICALL Java_mpi_GraphComm_getDistGraphNeighbors(
        JNIElw *elw, jobject jthis, jlong comm)
{
    int inDegree, outDegree, weighted;

    int rc = MPI_Dist_graph_neighbors_count(
             (MPI_Comm)comm, &inDegree, &outDegree, &weighted);

    if(ompi_java_exceptionCheck(elw, rc))
        return NULL;

    jintArray sources      = (*elw)->NewIntArray(elw, inDegree),
              srcWeights   = (*elw)->NewIntArray(elw, inDegree),
              destinations = (*elw)->NewIntArray(elw, outDegree),
              destWeights  = (*elw)->NewIntArray(elw, outDegree);

    jint *jSources, *jSrcWeights, *jDestinations, *jDestWeights;
    int  *cSources, *cSrcWeights, *cDestinations, *cDestWeights;

    ompi_java_getIntArray(elw, sources,      &jSources,      &cSources);
    ompi_java_getIntArray(elw, srcWeights,   &jSrcWeights,   &cSrcWeights);
    ompi_java_getIntArray(elw, destinations, &jDestinations, &cDestinations);
    ompi_java_getIntArray(elw, destWeights,  &jDestWeights,  &cDestWeights);

    rc = MPI_Dist_graph_neighbors((MPI_Comm)comm,
            inDegree, cSources, cSrcWeights,
            outDegree, cDestinations, cDestWeights);

    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releaseIntArray(elw, sources,      jSources,      cSources);
    ompi_java_releaseIntArray(elw, srcWeights,   jSrcWeights,   cSrcWeights);
    ompi_java_releaseIntArray(elw, destinations, jDestinations, cDestinations);
    ompi_java_releaseIntArray(elw, destWeights,  jDestWeights,  cDestWeights);

    return (*elw)->NewObject(elw,
           ompi_java.DistGraphNeighborsClass, ompi_java.DistGraphNeighborsInit,
           sources, srcWeights, destinations, destWeights,
           weighted ? JNI_TRUE : JNI_FALSE);
}

JNIEXPORT jint JNICALL Java_mpi_GraphComm_map(
        JNIElw *elw, jobject jthis, jlong comm,
        jintArray index, jintArray edges)
{
    int nNodes = (*elw)->GetArrayLength(elw, index);
    jint *jIndex, *jEdges;
    int  *cIndex, *cEdges;
    ompi_java_getIntArray(elw, index, &jIndex, &cIndex);
    ompi_java_getIntArray(elw, edges, &jEdges, &cEdges);

    int newrank;
    int rc = MPI_Graph_map((MPI_Comm)comm, nNodes, cIndex, cEdges, &newrank);
    ompi_java_exceptionCheck(elw, rc);

    ompi_java_releaseIntArray(elw, index, jIndex, cIndex);
    ompi_java_releaseIntArray(elw, edges, jEdges, cEdges);
    return newrank;
}
