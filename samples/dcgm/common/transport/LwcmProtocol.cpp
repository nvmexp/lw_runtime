#include "LwcmProtocol.h"
#include <arpa/inet.h>
#include <memory.h>

LwcmMessage::LwcmMessage()
{
    mSelfManagedBuffer = false;
}

LwcmMessage::LwcmMessage(dcgm_message_header_t *header)
{
    mSelfManagedBuffer = false;
    memcpy(&mMessageHdr, header, sizeof(mMessageHdr));
}

LwcmMessage::~LwcmMessage() 
{
    if (mSelfManagedBuffer) {
        delete [] mBuf;
    }
}

void LwcmMessage::UpdateMsgHdr(int msgType, dcgm_request_id_t requestId, int status, int length)
{
    mMessageHdr.msgId = DCGM_PROTO_MAGIC;
    mMessageHdr.requestId = requestId;
    mMessageHdr.msgType = msgType;
    mMessageHdr.status = status;
    mMessageHdr.length = length;
    SwapHeader();
    return;
}

void LwcmMessage::SwapHeader(void)
{
    mMessageHdr.msgId = htonl(mMessageHdr.msgId);
    mMessageHdr.requestId = htonl(mMessageHdr.requestId);
    mMessageHdr.msgType = htonl(mMessageHdr.msgType);
    mMessageHdr.status = htonl(mMessageHdr.status);
    mMessageHdr.length = htonl(mMessageHdr.length);
}

void LwcmMessage::UpdateMsgContent(char* buf, unsigned int length)
{
    mBuf = buf;
    mLength = length;
}


void LwcmMessage::CreateDataBuf(int length)
{
    mBuf = new char [length];
    mLength = length;
    mSelfManagedBuffer = true;
}

dcgm_message_header_t* LwcmMessage::GetMessageHdr()
{
    return &mMessageHdr;
}

void *LwcmMessage::GetContent()
{
    return mBuf;
}

unsigned int LwcmMessage::GetLength()
{
    return mLength;
}

dcgm_request_id_t LwcmMessage::GetRequestId()
{
    return mRequestId;
}

void LwcmMessage::SetRequestId(dcgm_request_id_t requestId)
{
    mRequestId = requestId;
}
