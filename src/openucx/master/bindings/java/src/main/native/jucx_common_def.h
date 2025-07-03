/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef HELPER_H_
#define HELPER_H_

#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>
#include <ucs/profile/profile.h>
#include <ucs/type/spinlock.h>

#include <jni.h>


typedef uintptr_t native_ptr;

#define JUCX_DEFINE_LONG_CONSTANT(_name) do { \
    jfieldID field = elw->GetStaticFieldID(cls, #_name, "J"); \
    if (field != NULL) { \
        elw->SetStaticLongField(cls, field, _name); \
    } \
} while(0)

#define JUCX_DEFINE_INT_CONSTANT(_name) do { \
    jfieldID field = elw->GetStaticFieldID(cls, #_name, "I"); \
    if (field != NULL) { \
        elw->SetStaticIntField(cls, field, _name); \
    } \
} while(0)

/**
 * Throw a Java exception by name. Similar to SignalError.
 */
#define JNU_ThrowException(_elw, _msg) do { \
    jclass _cls = _elw->FindClass("org/openucx/jucx/UcxException"); \
    ucs_error("JUCX: %s", _msg); \
    if (_cls != 0) { /* Otherwise an exception has already been thrown */ \
        _elw->ThrowNew(_cls, _msg); \
    } \
} while(0)

#define JNU_ThrowExceptionByStatus(_elw, _status) do { \
    JNU_ThrowException(_elw, ucs_status_string(_status)); \
} while(0)

/**
 * @brief Utility to colwert Java InetSocketAddress class (corresponds to the Network Layer 4
 * and consists of an IP address and a port number) to corresponding sockaddr_storage struct.
 * Supports IPv4 and IPv6.
 */
bool j2cInetSockAddr(JNIElw *elw, jobject sock_addr, sockaddr_storage& ss, socklen_t& sa_len);

struct jucx_context {
    jobject callback;
    volatile jobject jucx_request;
    ucs_status_t status;
    ucs_relwrsive_spinlock_t lock;
    size_t length;
};

void jucx_request_init(void *request);

/**
 * @brief Get the jni elw object. To be able to call java methods from ucx async callbacks.
 */
JNIElw* get_jni_elw();

/**
 * @brief Send callback used to ilwoke java callback class on completion of ucp operations.
 */
void jucx_request_callback(void *request, ucs_status_t status);

/**
 * @brief Recv callback used to ilwoke java callback class on completion of ucp tag_recv_nb operation.
 */
void recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);

/**
 * @brief Recv callback used to ilwoke java callback class on completion of ucp stream_recv_nb operation.
 */
void stream_recv_callback(void *request, ucs_status_t status, size_t length);

/**
 * @brief Utility to process request logic: if request is pointer - set callback to request context.
 * If request is status - call callback directly.
 * Returns jucx_request object, that could be monitored on completion.
 */
jobject process_request(void *request, jobject callback);

/**
 * @brief Call java callback on completed stream recv operation, that didn't ilwoke callback.
 */
jobject process_completed_stream_recv(size_t length, jobject callback);

void jucx_connection_handler(ucp_conn_request_h conn_request, void *arg);

/**
 * @brief Creates new jucx rkey class.
 */
jobject new_rkey_instance(JNIElw *elw, ucp_rkey_h rkey);

/**
 * @brief Creates new jucx tag_msg class.
 */
jobject new_tag_msg_instance(JNIElw *elw, ucp_tag_message_h msg_tag,
                             ucp_tag_recv_info_t *info_tag);

#endif
