#ifndef LWDAEGLTYPEDEFS_H
#define LWDAEGLTYPEDEFS_H

#include <lwdaEGL.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in lwdaEGL.h
 */
#define PFN_lwGraphicsEGLRegisterImage  PFN_lwGraphicsEGLRegisterImage_v7000
#define PFN_lwEGLStreamConsumerConnect  PFN_lwEGLStreamConsumerConnect_v7000
#define PFN_lwEGLStreamConsumerConnectWithFlags  PFN_lwEGLStreamConsumerConnectWithFlags_v8000
#define PFN_lwEGLStreamConsumerDisconnect  PFN_lwEGLStreamConsumerDisconnect_v7000
#define PFN_lwEGLStreamConsumerAcquireFrame  PFN_lwEGLStreamConsumerAcquireFrame_v7000
#define PFN_lwEGLStreamConsumerReleaseFrame  PFN_lwEGLStreamConsumerReleaseFrame_v7000
#define PFN_lwEGLStreamProducerConnect  PFN_lwEGLStreamProducerConnect_v7000
#define PFN_lwEGLStreamProducerDisconnect  PFN_lwEGLStreamProducerDisconnect_v7000
#define PFN_lwEGLStreamProducerPresentFrame  PFN_lwEGLStreamProducerPresentFrame_v7000
#define PFN_lwEGLStreamProducerReturnFrame  PFN_lwEGLStreamProducerReturnFrame_v7000
#define PFN_lwGraphicsResourceGetMappedEglFrame  PFN_lwGraphicsResourceGetMappedEglFrame_v7000
#define PFN_lwEventCreateFromEGLSync  PFN_lwEventCreateFromEGLSync_v9000


/**
 * Type definitions for functions defined in lwdaEGL.h
 */
typedef LWresult (LWDAAPI *PFN_lwGraphicsEGLRegisterImage_v7000)(LWgraphicsResource LWDAAPI *pLwdaResource, EGLImageKHR image, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamConsumerConnect_v7000)(LWeglStreamConnection LWDAAPI *conn, EGLStreamKHR stream);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamConsumerConnectWithFlags_v8000)(LWeglStreamConnection LWDAAPI *conn, EGLStreamKHR stream, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamConsumerDisconnect_v7000)(LWeglStreamConnection LWDAAPI *conn);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamConsumerAcquireFrame_v7000)(LWeglStreamConnection LWDAAPI *conn, LWgraphicsResource LWDAAPI *pLwdaResource, LWstream LWDAAPI *pStream, unsigned int timeout);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamConsumerReleaseFrame_v7000)(LWeglStreamConnection LWDAAPI *conn, LWgraphicsResource pLwdaResource, LWstream LWDAAPI *pStream);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamProducerConnect_v7000)(LWeglStreamConnection LWDAAPI *conn, EGLStreamKHR stream, EGLint width, EGLint height);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamProducerDisconnect_v7000)(LWeglStreamConnection LWDAAPI *conn);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamProducerPresentFrame_v7000)(LWeglStreamConnection LWDAAPI *conn, LWeglFrame_v1 eglframe, LWstream LWDAAPI *pStream);
typedef LWresult (LWDAAPI *PFN_lwEGLStreamProducerReturnFrame_v7000)(LWeglStreamConnection LWDAAPI *conn, LWeglFrame_v1 LWDAAPI *eglframe, LWstream LWDAAPI *pStream);
typedef LWresult (LWDAAPI *PFN_lwGraphicsResourceGetMappedEglFrame_v7000)(LWeglFrame_v1 LWDAAPI *eglFrame, LWgraphicsResource resource, unsigned int index, unsigned int mipLevel);
typedef LWresult (LWDAAPI *PFN_lwEventCreateFromEGLSync_v9000)(LWevent LWDAAPI *phEvent, EGLSyncKHR eglSync, unsigned int flags);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
