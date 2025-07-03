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
#include <assert.h>
#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_Prequest.h"
#include "mpiJava.h"

JNIEXPORT jlong JNICALL Java_mpi_Prequest_start(
        JNIElw *elw, jobject jthis, jlong jRequest)
{
    MPI_Request request = (MPI_Request)jRequest;
    int rc = MPI_Start(&request);
    ompi_java_exceptionCheck(elw, rc);
    return (jlong)request;
}

JNIEXPORT void JNICALL Java_mpi_Prequest_startAll(
        JNIElw *elw, jclass clazz, jlongArray prequests)
{
    int count = (*elw)->GetArrayLength(elw, prequests);
    jlong* jReq;
    MPI_Request *cReq;
    ompi_java_getPtrArray(elw, prequests, &jReq, (void***)&cReq);
    int rc = MPI_Startall(count, cReq);
    ompi_java_exceptionCheck(elw, rc);
    ompi_java_releasePtrArray(elw, prequests, jReq, (void**)cReq);
}
