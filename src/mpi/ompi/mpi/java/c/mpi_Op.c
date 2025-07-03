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
 * Copyright (c) 2018      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
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
 * File         : mpi_Op.c
 * Headerfile   : mpi_Op.h
 * Author       : Xinying Li, Bryan Carpenter
 * Created      : Thu Apr  9 12:22:15 1998
 * Revision     : $Revision: 1.7 $
 * Updated      : $Date: 2003/01/16 16:39:34 $
 * Copyright: Northeast Parallel Architectures Center
 *            at Syralwse University 1998
 */
#include "ompi_config.h"

#ifdef HAVE_TARGETCONDITIONALS_H
#include <TargetConditionals.h>
#endif

#include "mpi.h"
#include "mpi_Op.h"
#include "mpiJava.h"
#include "ompi/op/op.h"

JNIEXPORT void JNICALL Java_mpi_Op_init(JNIElw *elw, jclass clazz)
{
    ompi_java.OpHandle  = (*elw)->GetFieldID(elw, clazz, "handle", "J");
    ompi_java.OpCommute = (*elw)->GetFieldID(elw, clazz, "commute", "Z");

    ompi_java.OpCall = (*elw)->GetMethodID(elw, clazz, "call",
                       "(Ljava/lang/Object;Ljava/lang/Object;I)V");
}

JNIEXPORT void JNICALL Java_mpi_Op_getOp(JNIElw *elw, jobject jthis, jint type)
{
    static MPI_Op Ops[] = {
        MPI_OP_NULL, MPI_MAX, MPI_MIN, MPI_SUM,
        MPI_PROD, MPI_LAND, MPI_BAND, MPI_LOR, MPI_BOR, MPI_LXOR,
        MPI_BXOR, MPI_MINLOC, MPI_MAXLOC, MPI_REPLACE, MPI_NO_OP
    };
    (*elw)->SetLongField(elw,jthis, ompi_java.OpHandle, (jlong)Ops[type]);
}

static jobject setBooleanArray(JNIElw *elw, void *vec, int len)
{
    jobject obj = (*elw)->NewBooleanArray(elw, len);

    if(obj != NULL)
        (*elw)->SetBooleanArrayRegion(elw, obj, 0, len, vec);

    return obj;
}

static void getBooleanArray(JNIElw *elw, jobject obj, void *vec, int len)
{
    (*elw)->GetBooleanArrayRegion(elw, obj, 0, len, vec);
}

static void opIntercept(void *ilwec, void *inoutvec, int *count,
                        MPI_Datatype *datatype, int baseType,
                        void *jnielw, void *object)
{
    JNIElw  *elw  = jnielw;
    jobject jthis = object;
    jobject jin, jio;

    MPI_Aint lb, extent;
    int rc = MPI_Type_get_extent(*datatype, &lb, &extent);

    if(ompi_java_exceptionCheck(elw, rc))
        return;

    int len = (*count) * extent;

    if(baseType == 4)
    {
        jin = setBooleanArray(elw, ilwec, len);
        jio = setBooleanArray(elw, inoutvec, len);
    }
    else
    {
        jin = (*elw)->NewDirectByteBuffer(elw, ilwec, len);
        jio = (*elw)->NewDirectByteBuffer(elw, inoutvec, len);
    }

    if((*elw)->ExceptionCheck(elw))
        return;

    (*elw)->CallVoidMethod(elw, jthis, ompi_java.OpCall, jin, jio, *count);

    if(baseType == 4)
        getBooleanArray(elw, jio, inoutvec, len);

    (*elw)->DeleteLocalRef(elw, jin);
    (*elw)->DeleteLocalRef(elw, jio);
}

MPI_Op ompi_java_op_getHandle(JNIElw *elw, jobject jOp, jlong hOp, int baseType)
{
    MPI_Op op = (MPI_Op)hOp;

    if(op == NULL)
    {
        /* It is an uninitialized user Op. */
        int commute = (*elw)->GetBooleanField(
                      elw, jOp, ompi_java.OpCommute);

        int rc = MPI_Op_create((MPI_User_function*)opIntercept, commute, &op);

        if(ompi_java_exceptionCheck(elw, rc))
            return NULL;

        (*elw)->SetLongField(elw, jOp, ompi_java.OpHandle, (jlong)op);
        ompi_op_set_java_callback(op, elw, jOp, baseType);
    }

    return op;
}

JNIEXPORT void JNICALL Java_mpi_Op_free(JNIElw *elw, jobject jthis)
{
    MPI_Op op = (MPI_Op)((*elw)->GetLongField(elw, jthis, ompi_java.OpHandle));

    if(op != NULL && op != MPI_OP_NULL)
    {
        int rc = MPI_Op_free(&op);
        ompi_java_exceptionCheck(elw, rc);
        ((*elw)->SetLongField(elw,jthis,ompi_java.OpHandle,(long)MPI_OP_NULL));
    }
}

JNIEXPORT jboolean JNICALL Java_mpi_Op_isNull(JNIElw *elw, jobject jthis)
{
    MPI_Op op = (MPI_Op)((*elw)->GetLongField(elw, jthis, ompi_java.OpHandle));
    return op == NULL || op == MPI_OP_NULL ? JNI_TRUE : JNI_FALSE;
}
