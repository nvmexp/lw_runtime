/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpEndpoint.h"

#include <string.h>    /* memset */

#include <ucp/core/ucp_ep.inl> /* ucp_ep_peer_name */


static void error_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    JNIElw* elw = get_jni_elw();
    JNU_ThrowExceptionByStatus(elw, status);
    ucs_error("JUCX: endpoint error handler: %s", ucs_status_string(status));
}

JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_createEndpointNative(JNIElw *elw, jclass cls,
                                                           jobject ucp_ep_params,
                                                           jlong worker_ptr)
{
    ucp_ep_params_t ep_params;
    jfieldID field;
    ucp_worker_h ucp_worker = (ucp_worker_h)worker_ptr;
    ucp_ep_h endpoint;

    // Get field mask
    jclass ucp_ep_params_class = elw->GetObjectClass(ucp_ep_params);
    field = elw->GetFieldID(ucp_ep_params_class, "fieldMask", "J");
    ep_params.field_mask = elw->GetLongField(ucp_ep_params, field);

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_REMOTE_ADDRESS) {
        field = elw->GetFieldID(ucp_ep_params_class, "ucpAddress", "Ljava/nio/ByteBuffer;");
        jobject buf = elw->GetObjectField(ucp_ep_params, field);
        ep_params.address = static_cast<const ucp_address_t *>(elw->GetDirectBufferAddress(buf));
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE) {
        field = elw->GetFieldID(ucp_ep_params_class, "errorHandlingMode", "I");
        ep_params.err_mode =  static_cast<ucp_err_handling_mode_t>(elw->GetIntField(ucp_ep_params, field));
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_USER_DATA) {
        field = elw->GetFieldID(ucp_ep_params_class, "userData", "Ljava/nio/ByteBuffer;");
        jobject user_data = elw->GetObjectField(ucp_ep_params, field);
        ep_params.user_data = elw->GetDirectBufferAddress(user_data);
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_FLAGS) {
        field = elw->GetFieldID(ucp_ep_params_class, "flags", "J");
        ep_params.flags = elw->GetLongField(ucp_ep_params, field);
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR) {
        struct sockaddr_storage worker_addr;
        socklen_t addrlen;
        memset(&worker_addr, 0, sizeof(struct sockaddr_storage));

        field = elw->GetFieldID(ucp_ep_params_class,
                                "socketAddress", "Ljava/net/InetSocketAddress;");
        jobject sock_addr = elw->GetObjectField(ucp_ep_params, field);

        if (j2cInetSockAddr(elw, sock_addr, worker_addr, addrlen)) {
            ep_params.sockaddr.addr = (const struct sockaddr*)&worker_addr;
            ep_params.sockaddr.addrlen = addrlen;
        }
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_CONN_REQUEST) {
        field = elw->GetFieldID(ucp_ep_params_class, "connectionRequest", "J");
        ep_params.conn_request = reinterpret_cast<ucp_conn_request_h>(elw->GetLongField(ucp_ep_params, field));
    }

    ep_params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.err_handler.cb = error_handler;

    ucs_status_t status = ucp_ep_create(ucp_worker, &ep_params, &endpoint);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(elw, status);
    }

    return (native_ptr)endpoint;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_destroyEndpointNative(JNIElw *elw, jclass cls,
                                                            jlong ep_ptr)
{
    ucp_ep_destroy((ucp_ep_h)ep_ptr);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_closeNonBlockingNative(JNIElw *elw, jclass cls,
                                                             jlong ep_ptr, jint mode)
{
    ucs_status_ptr_t request = ucp_ep_close_nb((ucp_ep_h)ep_ptr, mode);

    return process_request(request, NULL);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_unpackRemoteKey(JNIElw *elw, jclass cls,
                                                      jlong ep_ptr, jlong addr)
{
    ucp_rkey_h rkey;

    ucs_status_t status = ucp_ep_rkey_unpack((ucp_ep_h)ep_ptr, (void *)addr, &rkey);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(elw, status);
    }

    jobject result = new_rkey_instance(elw, rkey);

    /* Coverity thinks that rkey is a leaked object here,
     * but it's stored in a UcpRemoteKey object */
    /* coverity[leaked_storage] */
    return result;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_putNonBlockingNative(JNIElw *elw, jclass cls,
                                                           jlong ep_ptr, jlong laddr,
                                                           jlong size, jlong raddr,
                                                           jlong rkey_ptr, jobject callback)
{
    ucs_status_ptr_t request = ucp_put_nb((ucp_ep_h)ep_ptr, (void *)laddr, size, raddr,
                                          (ucp_rkey_h)rkey_ptr, jucx_request_callback);

    ucs_trace_req("JUCX: put_nb request %p to %s, of size: %zu, raddr: %zu",
                  request, ucp_ep_peer_name((ucp_ep_h)ep_ptr), size, raddr);
    return process_request(request, callback);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_putNonBlockingImplicitNative(JNIElw *elw, jclass cls,
                                                                   jlong ep_ptr, jlong laddr,
                                                                   jlong size, jlong raddr,
                                                                   jlong rkey_ptr)
{
    ucs_status_t status = ucp_put_nbi((ucp_ep_h)ep_ptr, (void *)laddr, size, raddr,
                                      (ucp_rkey_h)rkey_ptr);

    if (UCS_STATUS_IS_ERR(status)) {
        JNU_ThrowExceptionByStatus(elw, status);
    }
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_getNonBlockingNative(JNIElw *elw, jclass cls,
                                                           jlong ep_ptr, jlong raddr,
                                                           jlong rkey_ptr, jlong laddr,
                                                           jlong size, jobject callback)
{
    ucs_status_ptr_t request = ucp_get_nb((ucp_ep_h)ep_ptr, (void *)laddr, size,
                                          raddr, (ucp_rkey_h)rkey_ptr, jucx_request_callback);

    ucs_trace_req("JUCX: get_nb request %p to %s, raddr: %zu, size: %zu, result address: %zu",
                  request, ucp_ep_peer_name((ucp_ep_h)ep_ptr), raddr, size, laddr);
    return process_request(request, callback);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_getNonBlockingImplicitNative(JNIElw *elw, jclass cls,
                                                                   jlong ep_ptr, jlong raddr,
                                                                   jlong rkey_ptr, jlong laddr,
                                                                   jlong size)
{
    ucs_status_t status = ucp_get_nbi((ucp_ep_h)ep_ptr, (void *)laddr, size, raddr,
                                      (ucp_rkey_h)rkey_ptr);

    if (UCS_STATUS_IS_ERR(status)) {
        JNU_ThrowExceptionByStatus(elw, status);
    }
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendTaggedNonBlockingNative(JNIElw *elw, jclass cls,
                                                                  jlong ep_ptr, jlong addr,
                                                                  jlong size, jlong tag,
                                                                  jobject callback)
{
    ucs_status_ptr_t request = ucp_tag_send_nb((ucp_ep_h)ep_ptr, (void *)addr, size,
                                               ucp_dt_make_contig(1), tag, jucx_request_callback);

    ucs_trace_req("JUCX: send_tag_nb request %p to %s, size: %zu, tag: %ld",
                  request, ucp_ep_peer_name((ucp_ep_h)ep_ptr), size, tag);
    return process_request(request, callback);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendStreamNonBlockingNative(JNIElw *elw, jclass cls,
                                                                  jlong ep_ptr, jlong addr,
                                                                  jlong size, jobject callback)
{
    ucs_status_ptr_t request = ucp_stream_send_nb((ucp_ep_h)ep_ptr, (void *)addr, size,
                                                  ucp_dt_make_contig(1), jucx_request_callback, 0);

    ucs_trace_req("JUCX: send_stream_nb request %p to %s, size: %zu",
                  request, ucp_ep_peer_name((ucp_ep_h)ep_ptr), size);
    return process_request(request, callback);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_recvStreamNonBlockingNative(JNIElw *elw, jclass cls,
                                                                  jlong ep_ptr, jlong addr,
                                                                  jlong size, jlong flags,
                                                                  jobject callback)
{
    size_t rlength;
    ucs_status_ptr_t request = ucp_stream_recv_nb((ucp_ep_h)ep_ptr, (void *)addr, size,
                                                  ucp_dt_make_contig(1), stream_recv_callback,
                                                  &rlength, flags);

    ucs_trace_req("JUCX: recv_stream_nb request %p to %s, size: %zu",
                  request, ucp_ep_peer_name((ucp_ep_h)ep_ptr), size);

    if (request == NULL) {
        // If request completed immidiately.
        return process_completed_stream_recv(rlength, callback);
    }

    return process_request(request, callback);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_flushNonBlockingNative(JNIElw *elw, jclass cls,
                                                             jlong ep_ptr,
                                                             jobject callback)
{
    ucs_status_ptr_t request = ucp_ep_flush_nb((ucp_ep_h)ep_ptr, 0, jucx_request_callback);

    return process_request(request, callback);
}
