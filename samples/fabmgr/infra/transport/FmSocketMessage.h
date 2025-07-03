/* 
 * File:   FmProtocol.h
 */

#ifndef FM_PROTOCOL_H
#define	FM_PROTOCOL_H

#include "FMErrorCodesInternal.h"
#include "FmSocketMessageHdr.h"

class FmSocketMessage
{
public:
    FmSocketMessage();
    FmSocketMessage(fm_message_header_t *header); /* Set header from a network packet (big endian) */
    ~FmSocketMessage();
    
    /**
     * This method updates the message header to be sent over socket. mMessageHdr will be
     * the big-endian version of the passed in parameters after this call. Use SwapHeader
     * to get these values back to little endian again
     */
    void UpdateMsgHdr(int msgType, fm_request_id_t requestId, int status, int length);

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
     * This method creates the content buffer for the FM message. 
     */
    void CreateDataBuf(int length);
    
    /**
     * This method returns reference to Fm Message Header
     */
    fm_message_header_t* GetMessageHdr();
    
    /**
     * This method returns reference to Fm Message content
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
    void SetRequestId(fm_request_id_t requestId);
    
    /**
     * This message is used to get request id corresponding to the message 
     */
    unsigned int GetRequestId();
    
private:
    fm_message_header_t mMessageHdr;  /* Sender populates the message to be sent */
    char *mBuf;                         /* Contains reference of encoded message to be sent/recvd */
    bool mSelfManagedBuffer;            /* Set to true when buf is self allocated */
    unsigned int mLength;               /* Length of message */
    fm_request_id_t mRequestId;       /* Maintains Request ID for the message */
};

#endif	/* FM_PROTOCOL_H */
