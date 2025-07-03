/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
extern "C" {
  #include <ucs/arch/cpu.h>
  #include <ucs/debug/assert.h>
  #include <ucs/debug/debug.h>
}

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <pthread.h>   /* pthread_yield */


static JavaVM *jvm_global;
static jclass jucx_request_cls;
static jfieldID native_id_field;
static jfieldID recv_size_field;
static jmethodID on_success;
static jmethodID jucx_request_constructor;
static jclass ucp_rkey_cls;
static jmethodID ucp_rkey_cls_constructor;
static jclass ucp_tag_msg_cls;
static jmethodID ucp_tag_msg_cls_constructor;

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void* reserved) {
    ucs_debug_disable_signals();
    jvm_global = jvm;
    JNIElw* elw;
    if (jvm->GetElw(reinterpret_cast<void**>(&elw), JNI_VERSION_1_1) != JNI_OK) {
       return JNI_ERR;
    }

    jclass jucx_request_cls_local = elw->FindClass("org/openucx/jucx/ucp/UcpRequest");
    jucx_request_cls = (jclass) elw->NewGlobalRef(jucx_request_cls_local);
    jclass jucx_callback_cls = elw->FindClass("org/openucx/jucx/UcxCallback");
    native_id_field = elw->GetFieldID(jucx_request_cls, "nativeId", "Ljava/lang/Long;");
    recv_size_field = elw->GetFieldID(jucx_request_cls, "recvSize", "J");
    on_success = elw->GetMethodID(jucx_callback_cls, "onSuccess",
                                  "(Lorg/openucx/jucx/ucp/UcpRequest;)V");
    jucx_request_constructor = elw->GetMethodID(jucx_request_cls, "<init>", "(J)V");

    jclass ucp_rkey_cls_local = elw->FindClass("org/openucx/jucx/ucp/UcpRemoteKey");
    ucp_rkey_cls = (jclass) elw->NewGlobalRef(ucp_rkey_cls_local);
    ucp_rkey_cls_constructor = elw->GetMethodID(ucp_rkey_cls, "<init>", "(J)V");
    jclass ucp_tag_msg_cls_local = elw->FindClass("org/openucx/jucx/ucp/UcpTagMessage");
    ucp_tag_msg_cls = (jclass) elw->NewGlobalRef(ucp_tag_msg_cls_local);
    ucp_tag_msg_cls_constructor = elw->GetMethodID(ucp_tag_msg_cls, "<init>", "(JJJ)V");
    return JNI_VERSION_1_1;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *jvm, void *reserved) {
    JNIElw* elw;
    if (jvm->GetElw(reinterpret_cast<void**>(&elw), JNI_VERSION_1_1) != JNI_OK) {
        return;
    }

    if (jucx_request_cls != NULL) {
        elw->DeleteGlobalRef(jucx_request_cls);
    }
}

bool j2cInetSockAddr(JNIElw *elw, jobject sock_addr, sockaddr_storage& ss,  socklen_t& sa_len)
{
    jfieldID field;
    memset(&ss, 0, sizeof(ss));
    sa_len = 0;

    if (sock_addr == NULL) {
        JNU_ThrowException(elw, "j2cInetSockAddr: InetSocketAddr is null");
        return false;
    }

    jclass inetsockaddr_cls = elw->GetObjectClass(sock_addr);

    // Get sockAddr->port
    jmethodID getPort = elw->GetMethodID(inetsockaddr_cls, "getPort", "()I");
    jint port = elw->CallIntMethod(sock_addr, getPort);

    // Get sockAddr->getAddress (InetAddress)
    jmethodID getAddress = elw->GetMethodID(inetsockaddr_cls, "getAddress",
                                            "()Ljava/net/InetAddress;");
    jobject inet_address = elw->CallObjectMethod(sock_addr, getAddress);

    if (inet_address == NULL) {
        JNU_ThrowException(elw, "j2cInetSockAddr: InetSocketAddr.getAddress is null");
        return false;
    }

    jclass inetaddr_cls = elw->GetObjectClass(inet_address);

    // Get address family. In Java IPv4 has addressFamily = 1, IPv6 = 2.
    field = elw->GetFieldID(inetaddr_cls, "holder",
                            "Ljava/net/InetAddress$InetAddressHolder;");
    jobject inet_addr_holder = elw->GetObjectField(inet_address, field);
    jclass inet_addr_holder_cls = elw->GetObjectClass(inet_addr_holder);
    field = elw->GetFieldID(inet_addr_holder_cls, "family", "I");
    jint family = elw->GetIntField(inet_addr_holder, field);

    field = elw->GetStaticFieldID(inetaddr_cls, "IPv4", "I");
    const int JAVA_IPV4_FAMILY = elw->GetStaticIntField(inetaddr_cls, field);
    field = elw->GetStaticFieldID(inetaddr_cls, "IPv6", "I");
    const int JAVA_IPV6_FAMILY = elw->GetStaticIntField(inetaddr_cls, field);

    // Get the byte array that stores the IP address bytes in the InetAddress.
    jmethodID get_addr_bytes = elw->GetMethodID(inetaddr_cls, "getAddress", "()[B");
    jobject ip_byte_array = elw->CallObjectMethod(inet_address, get_addr_bytes);

    if (ip_byte_array == NULL) {
        JNU_ThrowException(elw, "j2cInetSockAddr: InetAddr.getAddress.getAddress is null");
        return false;
    }

    jbyteArray addressBytes = static_cast<jbyteArray>(ip_byte_array);

    if (family == JAVA_IPV4_FAMILY) {
        // Deal with Inet4Address instances.
        // We should represent this Inet4Address as an IPv4 sockaddr_in.
        ss.ss_family = AF_INET;
        sockaddr_in &sin = reinterpret_cast<sockaddr_in &>(ss);
        sin.sin_port = htons(port);
        jbyte *dst = reinterpret_cast<jbyte *>(&sin.sin_addr.s_addr);
        elw->GetByteArrayRegion(addressBytes, 0, 4, dst);
        sa_len = sizeof(sockaddr_in);
        return true;
    } else if (family == JAVA_IPV6_FAMILY) {
        jclass inet6_addr_cls = elw->FindClass("java/net/Inet6Address");
        ss.ss_family = AF_INET6;
        sockaddr_in6& sin6 = reinterpret_cast<sockaddr_in6&>(ss);
        sin6.sin6_port = htons(port);
        // IPv6 address. Copy the bytes...
        jbyte *dst = reinterpret_cast<jbyte *>(&sin6.sin6_addr.s6_addr);
        elw->GetByteArrayRegion(addressBytes, 0, 16, dst);
        // ...and set the scope id...
        jmethodID getScopeId = elw->GetMethodID(inet6_addr_cls, "getScopeId", "()I");
        sin6.sin6_scope_id = elw->CallIntMethod(inet_address, getScopeId);
        sa_len = sizeof(sockaddr_in6);
        return true;
    }
    JNU_ThrowException(elw, "Unknown InetAddress family");
    return false;
}

static inline void jucx_context_reset(struct jucx_context* ctx)
{
    ctx->callback = NULL;
    ctx->jucx_request = NULL;
    ctx->status = UCS_INPROGRESS;
    ctx->length = 0;
}

void jucx_request_init(void *request)
{
     struct jucx_context *ctx = (struct jucx_context *)request;
     jucx_context_reset(ctx);
     ucs_relwrsive_spinlock_init(&ctx->lock, 0);
}

JNIElw* get_jni_elw()
{
    void *elw;
    jint rs = jvm_global->AttachLwrrentThread(&elw, NULL);
    ucs_assert_always(rs == JNI_OK);
    return (JNIElw*)elw;
}

static inline void set_jucx_request_completed(JNIElw *elw, jobject jucx_request,
                                              struct jucx_context *ctx)
{
    elw->SetObjectField(jucx_request, native_id_field, NULL);
    if ((ctx != NULL) && (ctx->length > 0)) {
        elw->SetLongField(jucx_request, recv_size_field, ctx->length);
    }
}

static inline void call_on_success(jobject callback, jobject request)
{
    JNIElw *elw = get_jni_elw();
    elw->CallVoidMethod(callback, on_success, request);
}

static inline void call_on_error(jobject callback, ucs_status_t status)
{
    if (status == UCS_ERR_CANCELED) {
        ucs_debug("JUCX: Request canceled");
    } else {
        ucs_error("JUCX: request error: %s", ucs_status_string(status));
    }

    JNIElw *elw = get_jni_elw();
    jclass callback_cls = elw->GetObjectClass(callback);
    jmethodID on_error = elw->GetMethodID(callback_cls, "onError", "(ILjava/lang/String;)V");
    jstring error_msg = elw->NewStringUTF(ucs_status_string(status));
    elw->CallVoidMethod(callback, on_error, status, error_msg);
}

static inline void jucx_call_callback(jobject callback, jobject jucx_request,
                                      ucs_status_t status)
{
    if (status == UCS_OK) {
        UCS_PROFILE_CALL_VOID(call_on_success, callback, jucx_request);
    } else {
        call_on_error(callback, status);
    }
}

UCS_PROFILE_FUNC_VOID(jucx_request_callback, (request, status), void *request, ucs_status_t status)
{
    struct jucx_context *ctx = (struct jucx_context *)request;
    ucs_relwrsive_spin_lock(&ctx->lock);
    if (ctx->jucx_request == NULL) {
        // here because 1 of 2 reasons:
        // 1. progress is in another thread and got here earlier then process_request happened.
        // 2. this callback is inside ucp_tag_recv_nb function.
        ctx->status = status;
        ucs_relwrsive_spin_unlock(&ctx->lock);
        return;
    }

    JNIElw *elw = get_jni_elw();
    set_jucx_request_completed(elw, ctx->jucx_request, ctx);

    if (ctx->callback != NULL) {
        jucx_call_callback(ctx->callback, ctx->jucx_request, status);
        elw->DeleteGlobalRef(ctx->callback);
    }

    elw->DeleteGlobalRef(ctx->jucx_request);
    jucx_context_reset(ctx);
    ucp_request_free(request);
    ucs_relwrsive_spin_unlock(&ctx->lock);
}

void recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info)
{
    struct jucx_context *ctx = (struct jucx_context *)request;
    ctx->length = info->length;
    jucx_request_callback(request, status);
}

void stream_recv_callback(void *request, ucs_status_t status, size_t length)
{
    struct jucx_context *ctx = (struct jucx_context *)request;
    ctx->length = length;
    jucx_request_callback(request, status);
}

UCS_PROFILE_FUNC(jobject, process_request, (request, callback), void *request, jobject callback)
{
    JNIElw *elw = get_jni_elw();
    jobject jucx_request = elw->NewObject(jucx_request_cls, jucx_request_constructor,
                                          (native_ptr)request);

    if (UCS_PTR_IS_PTR(request)) {
        struct jucx_context *ctx = (struct jucx_context *)request;
        ucs_relwrsive_spin_lock(&ctx->lock);
        if (ctx->status == UCS_INPROGRESS) {
            // request not completed yet, install user callback
            if (callback != NULL) {
                ctx->callback = elw->NewGlobalRef(callback);
            }
            ctx->jucx_request = elw->NewGlobalRef(jucx_request);
        } else {
            // request was completed whether by progress in other thread or inside
            // ucp_tag_recv_nb function call.
            set_jucx_request_completed(elw, jucx_request, ctx);
            if (callback != NULL) {
                jucx_call_callback(callback, jucx_request, ctx->status);
            }
            jucx_context_reset(ctx);
            ucp_request_free(request);
        }
        ucs_relwrsive_spin_unlock(&ctx->lock);
    } else {
        set_jucx_request_completed(elw, jucx_request, NULL);
        if (UCS_PTR_IS_ERR(request)) {
            JNU_ThrowExceptionByStatus(elw, UCS_PTR_STATUS(request));
            if (callback != NULL) {
                call_on_error(callback, UCS_PTR_STATUS(request));
            }
        } else if (callback != NULL) {
            call_on_success(callback, jucx_request);
        }
    }
    return jucx_request;
}

jobject process_completed_stream_recv(size_t length, jobject callback)
{
    JNIElw *elw = get_jni_elw();
    jobject jucx_request = elw->NewObject(jucx_request_cls, jucx_request_constructor, NULL);
    elw->SetObjectField(jucx_request, native_id_field, NULL);
    elw->SetLongField(jucx_request, recv_size_field, length);
    if (callback != NULL) {
        jucx_call_callback(callback, jucx_request, UCS_OK);
    }
    return jucx_request;
}

void jucx_connection_handler(ucp_conn_request_h conn_request, void *arg)
{
    jobject jucx_conn_handler = reinterpret_cast<jobject>(arg);

    JNIElw *elw = get_jni_elw();

    // Construct connection request class instance
    jclass conn_request_cls = elw->FindClass("org/openucx/jucx/ucp/UcpConnectionRequest");
    jmethodID conn_request_constructor = elw->GetMethodID(conn_request_cls, "<init>", "(J)V");
    jobject jucx_conn_request = elw->NewObject(conn_request_cls, conn_request_constructor,
                                               (native_ptr)conn_request);

    // Call onConnectionRequest method
    jclass jucx_conn_hndl_cls = elw->FindClass("org/openucx/jucx/ucp/UcpListenerConnectionHandler");
    jmethodID on_conn_request = elw->GetMethodID(jucx_conn_hndl_cls, "onConnectionRequest",
                                       "(Lorg/openucx/jucx/ucp/UcpConnectionRequest;)V");
    elw->CallVoidMethod(jucx_conn_handler, on_conn_request, jucx_conn_request);
    elw->DeleteGlobalRef(jucx_conn_handler);
}


jobject new_rkey_instance(JNIElw *elw, ucp_rkey_h rkey)
{
    return elw->NewObject(ucp_rkey_cls, ucp_rkey_cls_constructor, (native_ptr)rkey);
}

jobject new_tag_msg_instance(JNIElw *elw, ucp_tag_message_h msg_tag,
                             ucp_tag_recv_info_t *info_tag)
{
    return elw->NewObject(ucp_tag_msg_cls, ucp_tag_msg_cls_constructor,
                         (native_ptr)msg_tag, info_tag->length, info_tag->sender_tag);
}
