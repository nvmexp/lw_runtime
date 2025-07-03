/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "eglstream.h"

#if defined(EXTENSION_LIST)

EXTENSION_LIST(EXTLST_EXTERN)

/*
 * Print Status of EGLSTREAM
 */
void PrintEGLStreamState(EGLint streamState)
{
    #define STRING_VAL(x) {""#x"", x}
    struct {
        char *name;
        EGLint val;
    } EGLState[9] = {
        STRING_VAL(EGL_STREAM_STATE_CREATED_KHR),
        STRING_VAL(EGL_STREAM_STATE_CONNECTING_KHR),
        STRING_VAL(EGL_STREAM_STATE_EMPTY_KHR),
        STRING_VAL(EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR),
        STRING_VAL(EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR),
        STRING_VAL(EGL_STREAM_STATE_DISCONNECTED_KHR),
        STRING_VAL(EGL_BAD_STREAM_KHR),
        STRING_VAL(EGL_BAD_STATE_KHR),
        { NULL, 0 }
    };
    int i = 0;

    while(EGLState[i].name) {
        if(streamState == EGLState[i].val) {
            LOG_INFO("%s\n", EGLState[i].name);
            return;
        }
        i++;
    }
    LOG_ERR("Invalid state: %d.\n", streamState);
}

/*
 * Initialize the EGL Stream object for single process
 */
bool EGLStreamInitSingleProcess()
{
    const EGLint fifoLength = gTestArgs.fifoLength;
    // No EGL metadata
    static const EGLint streamAttrMailboxMode[] = { EGL_STREAM_TYPE_LW, EGL_STREAM_LOCAL_LW,
                                                    EGL_STREAM_PROTOCOL_LW, EGL_STREAM_LOCAL_LW,
                                                    EGL_STREAM_ENDPOINT_LW, EGL_STREAM_LOCAL_LW,
                                                    EGL_NONE };
    static const EGLint streamAttrFIFOMode[] = { EGL_STREAM_TYPE_LW, EGL_STREAM_LOCAL_LW,
                                                 EGL_STREAM_PROTOCOL_LW, EGL_STREAM_LOCAL_LW,
                                                 EGL_STREAM_ENDPOINT_LW, EGL_STREAM_LOCAL_LW,
                                                 EGL_STREAM_FIFO_LENGTH_KHR, fifoLength,
                                                 EGL_NONE };
    int iReturn;

    // Create the EGLStream
    gEglState.eglStream = eglCreateStreamKHR(gEglState.display,
                                             (fifoLength > 0) ? streamAttrFIFOMode : streamAttrMailboxMode);
    if (gEglState.eglStream == EGL_NO_STREAM_KHR) {
        LOG_ERR("eglCreateStreamKHR Failed.\n");
        goto fail;
    }

    // Set stream attributes

    // latency set between acquire and display
    iReturn = eglStreamAttribKHR(gEglState.display, gEglState.eglStream,
                                 EGL_CONSUMER_LATENCY_USEC_KHR, gEglState.latency);
    if (!iReturn) {
        LOG_ERR("eglStreamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed.\n");
        goto fail;
    }

    // timeout for consumer acquire
    iReturn = eglStreamAttribKHR(gEglState.display, gEglState.eglStream,
                                 EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, gEglState.acquireTimeout);
    if (!iReturn) {
        LOG_ERR("eglStreamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed.\n");
        goto fail;
    }

    return true;

fail:
    EGLStreamFini();
    return false;
}

/*
 * Setup stream parameters for cross-process or cross-partition.
 */
static bool createEGLStreamCrossP(bool crossPartition, bool producer)
{
    EGLint attr[MAX_EGL_STREAM_ATTR * 2 + 1];
    int attrIdx = 0;
    EGLint streamState = EGL_STREAM_STATE_EMPTY_KHR;
    EGLBoolean ret = EGL_FALSE;
    const EGLint fifoLength = gTestArgs.fifoLength;

    LOG_INFO("Creating EGLStream: [%s, %s]\n",
              crossPartition ? "Cross-Partition" : "Cross-Process",
              producer ? "Producer" : "Consumer");

    if (fifoLength > 0) {
        attr[attrIdx++] = EGL_STREAM_FIFO_LENGTH_KHR;
        attr[attrIdx++] = fifoLength;
    }

    // Create stream with the right attributes.
    {
        attr[attrIdx++] = EGL_CONSUMER_LATENCY_USEC_KHR;
        attr[attrIdx++] = gEglState.latency;

        attr[attrIdx++] = EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR;
        attr[attrIdx++] = gEglState.acquireTimeout;

        attr[attrIdx++] = EGL_STREAM_TYPE_LW;
        attr[attrIdx++] = crossPartition ? EGL_STREAM_CROSS_PARTITION_LW :
                              EGL_STREAM_CROSS_PROCESS_LW;

        attr[attrIdx++] = EGL_STREAM_ENDPOINT_LW;
        attr[attrIdx++] = producer ? EGL_STREAM_PRODUCER_LW : EGL_STREAM_CONSUMER_LW;

        attr[attrIdx++] = EGL_STREAM_PROTOCOL_LW;
        attr[attrIdx++] = EGL_STREAM_PROTOCOL_SOCKET_LW;

        attr[attrIdx++] = EGL_SOCKET_HANDLE_LW;
        attr[attrIdx++] = producer ? gEglState.iClient->getSockID() : gEglState.iServer->getClientSockID();

        attr[attrIdx++] = EGL_SOCKET_TYPE_LW;
        attr[attrIdx++] = crossPartition ? EGL_SOCKET_TYPE_INET_LW :
                              EGL_SOCKET_TYPE_UNIX_LW;
        attr[attrIdx++] = EGL_NONE;
    }

    gEglState.eglStream = eglCreateStreamKHR(gEglState.display, attr);

    do {
        pthread_yield();
        ret = eglQueryStreamKHR(gEglState.display, gEglState.eglStream, EGL_STREAM_STATE_KHR, &streamState);
        if (!ret) {
            LOG_ERR("Could not query EGL stream state");
            return false;
        }
    } while ((streamState == EGL_STREAM_STATE_INITIALIZING_LW) ||
             (producer && streamState != EGL_STREAM_STATE_CONNECTING_KHR));

    LOG_INFO("eglStream %p streamState 0x%x\n", gEglState.eglStream, streamState);

    if ((streamState != EGL_STREAM_STATE_CREATED_KHR) &&
        (!producer || (streamState != EGL_STREAM_STATE_CONNECTING_KHR))) {
        LOG_ERR("EGL stream is not in valid starting state");
        return false;
    }

    return true;
}

static bool EGLStreamInitProducerCrossP(bool crossPartition)
{
    int tries = 20;

    if (!gEglState.iClient->createSocket()) {
        return false;
    }

    // Try again in case consumer did not bind the socket in time.
    while (!gEglState.iClient->connect()) {
        sleep(1);
        if (--tries == 0 || gSignalStop) {
            LOG_ERR("Failed to connect, exit. errno=%d\n", errno);
            return false;
        }
        LOG_INFO("Try to connect again ...\n");
    }

    if (!gEglState.iClient->send("Hi, This is a sample producer")) {
        return false;
    }

    return createEGLStreamCrossP(crossPartition, true);
}

static bool EGLStreamInitConsumerCrossP(bool crossPartition)
{
    int tries = 20;

    if (!gEglState.iServer->createSocket()) {
        return false;
    }

    while (!gEglState.iServer->bind()) {
        sleep(1);
        if (--tries == 0 || gSignalStop) {
            LOG_ERR("Failed to bind, exit. errno=%d\n", errno);
            return false;
        }
        LOG_INFO("Try to bind again ...\n");
    }

    if (!gEglState.iServer->listen()) {
        return false;
    }

    SocketID clientsockfd = gEglState.iServer->accept();
    if (clientsockfd < 0) {
        return false;
    }
    LOG_INFO("%d is the client sockfd!\n", clientsockfd);
    (void)clientsockfd;

    if (!gEglState.iServer->receive()) {
        return false;
    }

    return createEGLStreamCrossP(crossPartition, false);
}

/*
 * Initialize the EGL Stream object for cross-process or
 * cross-partition
 */
bool EGLStreamInitCrossP()
{
    switch (gTestArgs.processMode)
    {
        case CROSS_PROCESS:
        {
            switch (gTestArgs.procType) {
            case PRODUCER:
                gEglState.iClient = new UnixClient;
                if (!EGLStreamInitProducerCrossP(false)) {
                    goto fail;
                }
                break;
            case CONSUMER:
                gEglState.iServer = new UnixServer;
                if (!EGLStreamInitConsumerCrossP(false)) {
                    goto fail;
                }
                break;
            default:
                assert(0);
                break;
            }
            break;
        }
        case CROSS_PARTITION:
        {
            switch (gTestArgs.procType) {
            case PRODUCER:
                gEglState.iClient = new IPClient(gTestArgs.ipAddr);
                if (!EGLStreamInitProducerCrossP(true)) {
                    goto fail;
                }
                break;
            case CONSUMER:
                gEglState.iServer = new IPServer;
                if (!EGLStreamInitConsumerCrossP(true)) {
                    goto fail;
                }
                break;
            default:
                assert(0);
                break;
            }
            break;
        }
        default:
            assert(0);
            break;
    }
    return true;

fail:
    EGLStreamFini();
    return false;
}

/*
 * Destroy EGL Stream Object
 */
void EGLStreamFini()
{
    if ((gEglState.display != EGL_NO_DISPLAY) && (gEglState.eglStream != EGL_NO_STREAM_KHR)) {
        eglDestroyStreamKHR(gEglState.display, gEglState.eglStream);
        gEglState.eglStream = EGL_NO_STREAM_KHR;
    }
}

/*
 * Query the status of EGL Stream
 */
int EGLStreamQuery(
    EGLDisplay display,
    EGLStreamKHR eglStream,
    EGLenum attribute,
    void *value,
    bool printState)
{
    switch (attribute) {
    case EGL_PRODUCER_FRAME_KHR:
        if (!eglQueryStreamu64KHR(display, eglStream,
                                  attribute, (EGLuint64KHR*)value)) {
            LOG_ERR("eglQueryStreamu64KHR failed for EGL_PRODUCER_FRAME_KHR.\n");
            return 0;
        }
        break;
    case EGL_CONSUMER_FRAME_KHR:
        if (!eglQueryStreamu64KHR(display, eglStream,
                                  attribute, (EGLuint64KHR*)value)) {
            LOG_ERR("eglQueryStreamu64KHR failed for EGL_CONSUMER_FRAME_KHR.\n");
            return 0;
        }
        break;
    case EGL_CONSUMER_LATENCY_USEC_KHR:
        if (!eglQueryStreamKHR(display, eglStream, attribute, (EGLint*)value)) {
            LOG_ERR("eglQueryStreamKHR failed for EGL_CONSUMER_LATENCY_USEC_KHR.\n");
            return 0;
        }
        break;
    case EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR:
        if (!eglQueryStreamKHR(display, eglStream, attribute, (EGLint*)value)) {
            LOG_ERR("eglQueryStreamKHR failed for EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR.\n");
            return 0;
        }
        break;
    case EGL_STREAM_FIFO_LENGTH_KHR:
        if (!eglQueryStreamKHR(display, eglStream, attribute, (EGLint*)value)) {
            LOG_ERR("eglQueryStreamKHR failed for EGL_STREAM_FIFO_LENGTH_KHR.\n");
            return 0;
        }
        break;
    default:
        if(!eglQueryStreamKHR(display, eglStream, attribute, (EGLint*)value)) {
            LOG_ERR("eglQueryStreamKHR failed.\n");
            return 0;
        }
        if (printState) {
            PrintEGLStreamState(*(EGLint*)value);
        }
        break;
    }

    return 1;
}

#endif
