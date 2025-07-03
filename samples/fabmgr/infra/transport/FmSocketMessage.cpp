#include "FmSocketMessage.h"
#ifdef __linux__
#include <arpa/inet.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif
#include <memory.h>

FmSocketMessage::FmSocketMessage()
{
    mSelfManagedBuffer = false;
    mBuf = NULL;
    mLength = 0;
    mRequestId = 0;
    memset(&mMessageHdr, 0, sizeof(mMessageHdr));
}

FmSocketMessage::FmSocketMessage(fm_message_header_t *header)
{
    mSelfManagedBuffer = false;
    memcpy(&mMessageHdr, header, sizeof(mMessageHdr));
    mBuf = NULL;
    mLength = 0;
    mRequestId = 0;
}

FmSocketMessage::~FmSocketMessage() 
{
    if (mSelfManagedBuffer) {
        delete [] mBuf;
    }
}

void FmSocketMessage::UpdateMsgHdr(int msgType, fm_request_id_t requestId, int status, int length)
{
    mMessageHdr.msgId = FM_PROTO_MAGIC;
    mMessageHdr.requestId = requestId;
    mMessageHdr.msgType = msgType;
    mMessageHdr.status = status;
    mMessageHdr.length = length;
    SwapHeader();
    return;
}

void FmSocketMessage::SwapHeader(void)
{
    mMessageHdr.msgId = htonl(mMessageHdr.msgId);
    mMessageHdr.requestId = htonl(mMessageHdr.requestId);
    mMessageHdr.msgType = htonl(mMessageHdr.msgType);
    mMessageHdr.status = htonl(mMessageHdr.status);
    mMessageHdr.length = htonl(mMessageHdr.length);
}

void FmSocketMessage::UpdateMsgContent(char* buf, unsigned int length)
{
    mBuf = buf;
    mLength = length;
}


void FmSocketMessage::CreateDataBuf(int length)
{
    mBuf = NULL;
    mBuf = new char [length];
    mLength = length;
    mSelfManagedBuffer = true;
}

fm_message_header_t* FmSocketMessage::GetMessageHdr()
{
    return &mMessageHdr;
}

void *FmSocketMessage::GetContent()
{
    return mBuf;
}

unsigned int FmSocketMessage::GetLength()
{
    return mLength;
}

fm_request_id_t FmSocketMessage::GetRequestId()
{
    return mRequestId;
}

void FmSocketMessage::SetRequestId(fm_request_id_t requestId)
{
    mRequestId = requestId;
}
