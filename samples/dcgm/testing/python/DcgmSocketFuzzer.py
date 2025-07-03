
import socket
import struct
import random
import time

'''
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
'''

DCGM_PROTO_MAGIC = 0xabbcbcab

#DCGM Message types
DCGM_MSG_PROTO_REQUEST = 0x0100 # A Google protobuf-based request
DCGM_MSG_PROTO_RESPONSE = 0x0200 # A Google protobuf-based response to a request
DCGM_MSG_MODULE_COMMAND = 0x0300 # A module command message 
DCGM_MSG_POLICY_NOTIFY = 0x0400 # Async notification of a policy violation
DCGM_MSG_REQUEST_NOTIFY = 0x0500 # Notify an async request that it will receive no further updates



def getDcgmMessageBytes(msgId=DCGM_PROTO_MAGIC, requestId=None, length=20, msgType=DCGM_MSG_PROTO_REQUEST, status=0, fillMessageToSize=True):
    if requestId is None:
        requestId = random.randint(0, 2000000000)
    
    #packing as big-endian ('>') since DCGM's header is network byte order
    
    messageBytes = struct.pack(">IIiii", msgId, requestId, length-20, msgType, status)

    if length > 20 and fillMessageToSize :
        for i in range(20, length, 1):
            messageBytes += struct.pack("B", random.randint(0, 255))

    return messageBytes


def getSocketObj():
    socketObj = socket.create_connection(('127.0.0.1', 5555))
    return socketObj

def testCompleteGarbage():
    print("Sending random garbage and expecting a hang-up")
    for i in range(0, 50):
        socketObj = getSocketObj()
        numBytesSent = 0
        while True:
            data = struct.pack("B", random.randint(0, 255))
            try:
                socketObj.sendall(data)
            except socket.error as serr:
                print("Got expected socket error after %d bytes sent" % numBytesSent)
                break
            numBytesSent += 1

def testMessageContentsGarbage():
    print("Sending message body as random garbage expecting a hang-up")
    for i in range(0, 500):
        socketObj = getSocketObj()
        numBytesSent = 0
        numPacketsSent = 0
        while True:
            messageLength = random.randint(21, 1000)
            data = getDcgmMessageBytes(length=messageLength)
            try:
                socketObj.sendall(data)
            except socket.error as serr:
                print("Got expected socket error after %d packets (%d bytes) sent" % (numPacketsSent, numBytesSent))
                break
            numPacketsSent += 1
            numBytesSent += messageLength

            if numPacketsSent % 1000 == 0:
                print("Sent %d packets. %u bytes so far." % (numPacketsSent, numBytesSent))

def testGiantAndNegativeMessages():
    print("Sending huge and negative message lengths")
    
    duration = 20.0
    startTime = time.time()

    i = 0
    while time.time() - startTime < duration:
        i += 1
        socketObj = getSocketObj()
        numBytesSent = 0
        numPacketsSent = 0
        while True:
            if i & 1:
                #Humungous
                messageLength = random.randint(1000000000, 2000000000)
            else:
                #Negative size
                messageLength = random.randint(-1000000000, -1)
            data = getDcgmMessageBytes(length=messageLength, fillMessageToSize=False)
            try:
                socketObj.sendall(data)
            except socket.error as serr:
                print("Got expected socket error after %d packets (%d bytes) sent" % (numPacketsSent, numBytesSent))
                break
            numPacketsSent += 1

#testGiantAndNegativeMessages()
testMessageContentsGarbage()
#testCompleteGarbage()

