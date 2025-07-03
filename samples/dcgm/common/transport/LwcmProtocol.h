/* 
 * File:   LwcmProtocol.h
 */

#ifndef LWCM_PROTOCOL_H
#define	LWCM_PROTOCOL_H

#include "dcgm_structs.h"
#include <g_lwconfig.h>


/* Align to byte boundaries */
#pragma pack(1)

#define DCGM_PROTO_MAGIC 0xabbcbcab /* Used to fill up the msgID for lwcm message header */

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
#define DCGM_PROTO_MAX_MESSAGE_SIZE (64*1024*1024) /* Maximum size of a single DCGM socket message (64 MB) */
#else
#define DCGM_PROTO_MAX_MESSAGE_SIZE (4*1024*1024) /* Maximum size of a single DCGM socket message (4 MB) */
#endif
/** 
 * ID of a request from the client to the host engine. This is unique for a given connection
 * but not unique across connections
 */
typedef unsigned int dcgm_request_id_t;

/**
 * DCGM Message Header. Note that this is byte-swapped to network order (big endian)
 * before transport. LwcmMessage::UpdateMsgHdr() does this automatically
 */
typedef struct
{
    int msgId;                   /* Identifier to represent DCGM protocol (DCGM_PROTO_MAGIC) */
    dcgm_request_id_t requestId; /* Represent Message by a request ID */
    int length;                  /* Length of encoded message */
    int msgType;                 /* Type of encoded message. One of DCGM_MSG_? */
    int status;                  /* Status. One of DCGM_PROTO_ST_? */
} dcgm_message_header_t;

#pragma pack() /* Undo the 1-byte alignment */

/**
 * The following defines are used to recognize type of DCGM messages
 */
#define DCGM_MSG_PROTO_REQUEST      0x0100 /* A Google protobuf-based request */
#define DCGM_MSG_PROTO_RESPONSE     0x0200 /* A Google protobuf-based response to a request */
#define DCGM_MSG_MODULE_COMMAND     0x0300 /* A module command message */
#define DCGM_MSG_POLICY_NOTIFY      0x0400 /* Async notification of a policy violation */
#define DCGM_MSG_REQUEST_NOTIFY     0x0500 /* Notify an async request that it will receive no further updates */

/* DCGM_MSG_POLICY_NOTIFY - Signal a client that a policy has been violated */
typedef struct 
{
    int begin;                    /* Whether this is the first response (1) or the second (0).
                                     This will determine if beginCB or finishCB is called. */
    dcgmPolicyCallbackResponse_t response; /* Policy response to pass to client callbacks */
} dcgm_msg_policy_notify_t;

/* DCGM_MSG_REQUEST_NOTIFY - Notify an async request that it will receive 
 *                           no further updates 
 **/
typedef struct
{
    unsigned int requestId; /* This is redundant with the header requestId, but we need
                               message contents */
} dcgm_msg_request_notify_t;

/**
 * The following defines are used to represent the status of the message
 */
#define DCGM_PROTO_ST_SUCCESS           0 /* Successful */


class LwcmMessage
{
public:
    LwcmMessage();
    LwcmMessage(dcgm_message_header_t *header); /* Set header from a network packet (big endian) */
    ~LwcmMessage();
    
    /**
     * This method updates the message header to be sent over socket. mMessageHdr will be
     * the big-endian version of the passed in parameters after this call. Use SwapHeader
     * to get these values back to little endian again
     */
    void UpdateMsgHdr(int msgType, dcgm_request_id_t requestId, int status, int length);

    /**
     * Swap the message header between big endian and little endian. The header is in
     * big endian format on the socket, so if you want to look at the fields, you need to
     * swap it back
     */
    void SwapHeader(void);
    
    /**
     * This method sets the message content passed by the caller
     */
    void UpdateMsgContent(char *buf, unsigned int length);
    
    /**
     * This method creates the content buffer for the DCGM message. 
     */
    void CreateDataBuf(int length);
    
    /**
     * This method returns reference to Lwcm Message Header
     */
    dcgm_message_header_t* GetMessageHdr();
    
    /**
     * This method returns reference to Lwcm Message content
     */
    void * GetContent();
    
    /**
     * This message is used to get the length of the message
     */
    unsigned int GetLength();
    
    /**
     * This message is used to set request id corresponding to the message
     * @return 
     */
    void SetRequestId(dcgm_request_id_t requestId);
    
    /**
     * This message is used to get request id corresponding to the message 
     */
    unsigned int GetRequestId();
    
private:
    dcgm_message_header_t mMessageHdr;  /* Sender populates the message to be sent */
    char *mBuf;                         /* Contains reference of encoded message to be sent/recvd */
    bool mSelfManagedBuffer;            /* Set to true when buf is self allocated */
    unsigned int mLength;               /* Length of message */
    dcgm_request_id_t mRequestId;       /* Maintains Request ID for the message */
};

#endif	/* DCGM_PROTOCOL_H */
