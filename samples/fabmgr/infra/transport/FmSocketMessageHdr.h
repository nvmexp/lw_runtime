/* 
 * File:   FmSocketMessageHdr.h
 */

#ifndef FM_SOCKET_MESSAGE_HDR_H
#define	FM_SOCKET_MESSAGE_HDR_H

/* Align to byte boundaries */
#pragma pack(1)

#define FM_PROTO_MAGIC 0xabbcbcab /* Used to fill up the msgID for fm message header */

/** 
 * ID of a request from the client to the host engine. This is unique for a given connection
 * but not unique across connections
 */
typedef unsigned int fm_request_id_t;

/**
 * FM Message Header. Note that this is byte-swapped to network order (big endian)
 * before transport. FmMessage::UpdateMsgHdr() does this automatically
 */
typedef struct
{
    int msgId;                   /* Identifier to represent FM protocol (FM_PROTO_MAGIC) */
    fm_request_id_t requestId; /* Represent Message by a request ID */
    int length;                  /* Length of encoded message */
    int msgType;                 /* Type of encoded message. One of FM_MSG_? */
    int status;                  /* Status. One of FM_PROTO_ST_? */
} fm_message_header_t;

#pragma pack() /* Undo the 1-byte alignment */

/**
 * The following defines are used to recognize type of FM messages
 */
#define FM_MSG_PROTO_REQUEST      0x0100 /* A Google protobuf-based request */
#define FM_MSG_PROTO_RESPONSE     0x0200 /* A Google protobuf-based response to a request */
#define FM_MSG_MODULE_COMMAND     0x0300 /* A module command message */
#define FM_MSG_POLICY_NOTIFY      0x0400 /* Async notification of a policy violation */
#define FM_MSG_REQUEST_NOTIFY     0x0500 /* Notify an async request that it will receive no further updates */

/* FM_MSG_REQUEST_NOTIFY - Notify an async request that it will receive 
 *                           no further updates 
 **/
typedef struct
{
    unsigned int requestId; /* This is redundant with the header requestId, but we need
                               message contents */
} fm_msg_request_notify_t;

/**
 * The following defines are used to represent the status of the message
 */
#define FM_PROTO_ST_SUCCESS           0 /* Successful */


#endif	/* FM_SOCKET_MESSAGE_HDR_H */
