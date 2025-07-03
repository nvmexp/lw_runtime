/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpMemory.h"
#include "org_openucx_jucx_ucp_UcpRemoteKey.h"


JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpMemory_unmapMemoryNative(JNIElw *elw, jclass cls,
                                                      jlong context_ptr, jlong mem_ptr)
{
    ucs_status_t status = ucp_mem_unmap((ucp_context_h)context_ptr, (ucp_mem_h)mem_ptr);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(elw, status);
    }
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpMemory_getRkeyBufferNative(JNIElw *elw, jclass cls,
                                                        jlong context_ptr, jlong mem_ptr)
{
    void *rkey_buffer;
    size_t rkey_size;

    ucs_status_t status = ucp_rkey_pack((ucp_context_h)context_ptr, (ucp_mem_h)mem_ptr,
                                        &rkey_buffer, &rkey_size);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(elw, status);
    }
    return elw->NewDirectByteBuffer(rkey_buffer, rkey_size);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpMemory_releaseRkeyBufferNative(JNIElw *elw, jclass cls, jobject rkey_buf)
{
    ucp_rkey_buffer_release(elw->GetDirectBufferAddress(rkey_buf));
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpRemoteKey_rkeyDestroy(JNIElw *elw, jclass cls, jlong rkey_ptr)
{
    ucp_rkey_destroy((ucp_rkey_h) rkey_ptr);
}
