/* 
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/**
 * @file lwdspstr.h
 *
 * @brief
 * Interface definitions for the LWPU Display Streaming API.
 */

/**
 * @mainpage lwdspstr
 *
 *	LWPU Display Capture Streaming API
 *
 * This file defines an application programming interface (API) for
 * capturing a stream of display frames.  Frames may be captured fully
 * composited, as separate base, overlay, and cursor planes, or as
 * compositied base and overlay planes plus changes to the cursor
 * position and content.  The capture mode may be changed at any time,
 * to take effect with the next frame.
 *
 * The API supports using buffers supplied by either the caller or the
 * underlying library, with a common method for releasing them.  The
 * use of multiple buffers may be selected, to allow for pipelining of
 * traffic, as when the destination is remote.  One may specify either
 * single use of the buffers (meaning a buffer will be used just once
 * until returned to the pool by the user of the API) or reuse
 * (meaning a given buffer will be overwritten at each frame time).
 *
 * A buffer may be supplied as either a user virtual address or an
 * LWPU resource manager (RM) client handle and memory object handle
 * with offset.
 *
 * Multiple heads are supported, each of which may have a different
 * display descriptor.  Heads may be real, in which case the real
 * display descriptor (EDID) from the monitor is used, or virtual, in
 * which case the caller must supply an EDID for each virtual head.
 * This API may be used with real heads for remote viewing of the real
 * display (as with VNC), or with virtual heads in a virtual desktop
 * server environment.
 *
 * The API is message-oriented, to maximize the opportunities for
 * asynchrony and for time-based processing, to minimize context
 * switching overhead.  The caller registers with the library by
 * providing a callback routine reference to the initialization
 * routine, for the library to use in sending messages, and may then
 * call the message sending routine to deliver messages to the
 * library.
 *
 * The library is thread-safe, but holds no locks across callbacks
 * to the user, whether for message delivery or buffer release, so
 * the user is responsible for conlwrrency control outside the
 * library.  Also, the library may make a callback from one thread
 * while the user is calling into the library from the same or 
 * different thread, so the user's locking discipline should avoid
 * deadlocks (as by not holding an exclusive lock across a call to
 * the library, if it could be acquired in a callback).
 *
 * A typical application sequence for using the library would
 * be as follows:
 *
 * - Obtain a reference to graphics device or display path
 *   with a graphics API such as OpenGL or DirectX
 * - Call LwDspStrInit() to open a stream with respect to the
 *   graphics device or display path
 * - Optionally, send a PipeDepth message to the library to set
 *   the depth of frame pipelining
 * - Optionally, allocate frame buffers via the graphics API
 *   and supply them to the library via ProvideBuffer messages
 * - Supply the library with an EDID the display device
 *   using an EDID message.
 * - As the library calls back to deliver Frame messages,
 *   process and display each frame.  Release each buffer
 *   after processing.
 * - When disconnecting, call LwDspStrTerminate()
 *   to close the stream.
 *
 * The definitions are grouped as follows:
 * - @ref CommonTypes
 * - @ref FunctionMessageHeaders
 */

#ifndef _LWDSPSTR_H_
/*!< @cond lwdspstr */
#define _LWDSPSTR_H_

#ifndef _WIN32
#include <stdint.h>
  /* "inline" is a keyword */
#else /* _WIN32 */
#ifndef _STDINT_H
typedef unsigned char       uint8_t;
typedef          char        int8_t;
typedef unsigned short     uint16_t;
typedef          short      int16_t;
typedef unsigned int       uint32_t;
typedef          int        int32_t;
typedef unsigned long long uint64_t;
typedef          long long  int64_t;
#endif /* _STDINT_H */
# define inline /* nothing */
#endif /* _WIN32 */
#ifndef NULL
#define NULL ((void *) 0)
#endif
/*!< @endcond */

/**********************************************************************/
/**
* @defgroup CommonTypes     Common type definitions.
*/
/**********************************************************************/
/*@{*/

/**
 * Message type
 */

enum LwDspMessageTypeE {
    LwDspStrMessageTypeNop = 0, /*!< no content (both directions) */
    LwDspStrMessageTypeProvideBuffer = 1, 
    /*!< supply buffer for future frame (user to library) */
    LwDspStrMessageTypeRevokeBuffer = 2,
    /*!< revoke use of previously supplied buffer (both directions) */
    LwDspStrMessageTypeFrame = 3,
    /*!< deliver a frame (library to user) */
    LwDspStrMessageTypeEDID = 4,
    /*!< deliver an EDID (user to library) */
    LwDspStrMessageTypeFrameFormat = 5,
    /*!< set frame format (library to user) */
    LwDspStrMessageTypeFrameMode = 6,
    /*!< set frame delivery mode (user to library) */
    LwDspStrMessageTypeHdcpDownstream = 7,
    /*!< deliver HDCP downstream message (library to user) */
    LwDspStrMessageTypeHdcpUpstream = 8,
    /*!< deliver HDCP upstream message (user to library) */
    LwDspStrMessageTypeControlMonitor = 9,
    /*!< control monitor (both directions) */
    LwDspStrMessageTypeRequestEDID = 10,
    /*!< request EDID delivery (library to user) */
    LwDspStrMessageTypeUpdateLwrsor = 11,
    /*!< update cursor position (library to user) */
    LwDspStrMessageTypeDifferentialFrame = 12,
    /*!< deliver a differential frame (library to user) */
    LwDspStrMessageTypePipeDepth = 13,
    /*!< set or report pipeline depth (both directions) */
    LwDspStrMessageTypeRectangleList = 14,
    /*!< set rectangle list */
};

typedef uint16_t LwDspStrMessageType;	
/*!< Message type */
	
/**
 * Message mode
 */

enum LwDspStrMessageModeE {
    LwDspStrMessageModeRequest = 0, /*!< normal message */
    LwDspStrMesasgeModeResponce = 1, 
    /*!< normal response (when defined for a message type) */
    LwDspStrMessageModeReject = 2
    /*!< rejection response for a unknown message */
};

typedef uint16_t LwDspStrMessageMode;
/*!< message mode */

/**
 * The message descriptor is typically separate from the buffer itself,
 * and contains a reference (either a pointer or a handle) for the
 * actual buffer.  It includes a callback function reference, to 
 * be called when the reference count is reduced to zero.
 *
 * The message content is addressed by a list of one or more
 * memory descriptors.  The type-specific message header oclwpies
 * the first bytes of the memory described, followed by the content,
 * if any.
 */

typedef struct LwDspStrMessageS *pLwDspStrMessage;

/**
 * List header reference.
 */

typedef struct LwDspStrListHeaderS *pLwDspStrListHeader;

/**
 * List header.
 *
 * This structure is used at the start of other structures which the
 * environment keeps in lists.  Plugins should not access it.
 */

typedef struct LwDspStrListHeaderS {
    pLwDspStrListHeader next; /*!< next item in list */
    pLwDspStrListHeader prev; /*!< previous item in list */
} LwDspStrListHeader;

/**
 * Graphics object class
 */

enum LwDspStrGraphicsObjectClassE {
    LwDspStrGraphicsObjectClassNull = 0,
    /*!< null object */
    LwDspStrGraphicsObjectClassGL = 1,
    /*!< OpenGL handle */
    LwDspStrGraphicsObjectClassDirectX = 2,
    /*!< DirectX handle */
    LwDspStrGraphicsObjectClassInternal = 3,
    /*!< LWPU handle */
    LwDspStrGraphicsObjectClassLogical = 4
    /*!< logical handle */
};

typedef uint16_t LwDspStrGraphicsObjectClass;
/*!< graphics object class container */

/**
 * Graphics object reference.
 *
 * In some contexts, reference is made to a graphics
 * object, using this type.
 * 
 * The reference may be considered to be null if class
 * is LwDspStrGraphicsObjectClassNull.
 *
 * For the Logical class, the device index is the ordinal
 * of the set of devices in the system, counting from 0.
 * The subdevice index is the ordinal of the subdevice within
 * the device.  For a system with just one GPU, both indexes
 * are 0.  The Logical class may only be used in a context
 * where a display device is expected, not in a buffer 
 * context.
 *
 * For the Internal class, the values are the internal
 * LWPU handles.  This class is not usable except by other
 * LWPU libraries, since the handles are not obtainable
 * by user programs.
 */

typedef struct {
    LwDspStrGraphicsObjectClass objectClass;
    /*!< class of object */
    uint16_t pad;
    /*!< reserved */
    union {
        struct {
            uint32_t bufferobj; /*!< GL buffer object handle */
        } OpenGL;
        /*!< handle values for the OpenGL class */
        struct {
            void *resource; /*!< DirectX resource pointer */
        } DirectX;
        /*!< handle values for the DirectX class */
        struct {
            uint32_t hClient; /*!< client handle */
            uint32_t hDevice; /*!< device or subdevice object handle */
            uint32_t hObject; /*!< object handle */
        } internal;
        /*!< handle values for the Internal class */
        struct {
            uint32_t deviceIndex; /*!< index of device in system */
            uint32_t subdeviceIndex; /*!< index of subdevice within device */
        } logical;
        /*!< device and subdevice indexes for the Logical class */
    } handle;
    /*!< handle values for the classes */
} LwDspStrGraphicsObject;

#if __STDC_VERSION__ >= 199901L
static const LwDspStrGraphicsObject LW_DSPSTR_NULL_GRAPHICS_OBJECT = 
    {
        .objectClass = LwDspStrGraphicsObjectClassNull
    };
/*!< NULL value for a graphics object */
#else
/*!< @cond lwdspstr */
static LwDspStrGraphicsObject LW_DSPSTR_NULL_GRAPHICS_OBJECT = { 0 };
/*!< @endcond */
#endif

/**
 * 
 * The data elements of a buffer are defined by a variable length
 * array of this type.  An element may be either a pointer to virtual
 * space in the application or an an offset in an LWPU memory or
 * context DMA object, of the specified length.  The former should be
 * used except for actual frame data, which may be either, but the
 * library will work correctly if an LWPU object is used.
 *
 * If an object is specified, then pData should be NULL or the address
 * of a mapping of the object into virtual space.  The library may
 * change pData from NULL in a message sent to it, but will restore it
 * to NULL before releasing its last reference.
 */

typedef struct {
    LwDspStrGraphicsObject object;
    /*!< graphics object or LW_DSPSTR_NULL_GRAPHICS_OBJECT */
    uint64_t offset;  /*!< offset in object or 0             */
    uint64_t pData; 
    /*!< reference to data array (address in virtual space) */
    uint64_t length; /*!< length of buffer element in bytes */
} LwDspStrBufferElement;

typedef LwDspStrBufferElement *pLwDspStrBufferElement;
/*!< reference to a buffer element */

/**
 * Release the last reference to a message object.
 * This callback routine is called when the reference
 * count is decremented to zero in LwDspStrReleaseBuffer().
 *
 * @param[in] pBuf         Buffer reference
 * @param[in] pPrivate     Private pointer supplied on LwDspStrInit() call
 * @param[in] pConnection  Library-private pointer returned by LwDspStrInit() call
 */

typedef void (*pLwDspStrMessageFree)(pLwDspStrMessage pBuf,
                                     void *pPrivate,
                                     void *pConnection);

/**
 * Message descriptor.
 *
 * LwDspStrMessage serves as an abstract buffer, comprising a buffer
 * header and a logical array of bytes.  The array of bytes is the
 * concatenation of the data blocks defined by the members of the
 * element array (having count members).  The element array is thus
 * like a scatter/gather list for a hardware DMA engine.  
 * 
 * The most common type of element is a pointer to a data block and a
 * length.  Alternate elements may be graphics surfaces.  
 *
 * An actual message used in the API is contained in the logical array
 * of bytes.  A message has a header (of one of the LwDspStrHeader*
 * types defined later) optionally followed by data.  For a frame
 * message, the first element of the buffer would likely point to a
 * data block the size of the header, and the second block would point
 * to the mapped address of a graphics surface.  This avoids the need
 * to copy the large block of frame content.  If the content element
 * is a preallocated graphics surface, the latter may be passed
 * directly to LWCA or a graphics API.
 *
 * Use of the listHead is reserved to the current owner of the message
 * (the application after the library has sent the message to the
 * application, and the library after the application has send the
 * message to the library).
 *
 * While an allocator may choose to place the LwDspStrMessage
 * structure, the element list, and space for the message header in a
 * single allocation of memory, users of messages should always walk
 * the element list, directly or via the utility routines
 * LwDspStrBufferPullup() and LwDspStrBufferApply(), to locate tyhe
 * message header, and should not assume that any given part of the
 * message is contained within a single buffer element of the message.
 *
 * The sequence value is not inspected or changed by the library for 
 * messages sent by the user.  The library will set sequence to a
 * monotonically increasing number in successive messages it sends 
 * to the user.   The sequence value may be used in log entries
 * to correlate information about a given logical message, but has
 * no operational function.
 */

typedef struct LwDspStrMessageS {
    LwDspStrListHeader listHead;
    /*!< Header for list of buffers (reserved for 
     *   current owner of buffer (library when delivered
     *   by user to library, user when deliverd by 
     *   library to user) */
    LwDspStrMessageType messageType;
    /*!< message function code */
    LwDspStrMessageMode mode; /*!< request/response/rejection */
    uint32_t signature;         /*!< set to LW_DSPSTR_MC_SIGNATURE */
    uint32_t version;           /*!< set to LW_DSPSTR_MC_VERSION */
    uint32_t headerLength;    
    /*!< length of function-specific header (in the data area) */
    uint32_t sequence;          /*!< sequence number */
    pLwDspStrMessageFree pFree;
    /*!< pointer to message free callback */
    uint32_t references;
    /*!< count of references to message */
    uint32_t count;
    /*!< Number of elements in the array of data elements */
    pLwDspStrBufferElement element;
    /*!< Reference to the array of data elements */
} LwDspStrMessage;
/*!< message descriptor */

#define LW_DSPSTR_MC_SIGNATURE ((((uint32_t) ('N')) << 0) | \
                                (((uint32_t) ('v')) << 8) | \
                                (((uint32_t) ('D')) << 16) | \
                                (((uint32_t) ('S')) << 24))
/*!< Message signature (to deduce endianness) */
#define LW_DSPSTR_MC_VERSION ((uint32_t) 0x00010000)
/*!< Message version 1.0.0 */

/**
 * Add a reference to a message object.
 *
 * @param[in] pBuf     Message reference
 * @param[in] pConnection  Library-private pointer returned by LwDspStrInit() call
 */

extern void
LwDspStrMessageHold(pLwDspStrMessage pBuf,
                    void *pConnection);

/**
 * Release a reference to a message object.
 * Call the message free callback if the reference
 * count is reduced to zero.
 *
 * @param[in] pBuf     Message reference
 * @param[in] pConnection  Library-private pointer returned by LwDspStrInit() call
 */

extern void
LwDspStrMessageRelease(pLwDspStrMessage pBuf,
                       void *pConnection);

/**
 * Count total length of a message.
 *
 * LwDspStrMessageLength() is a utility routine, which may
 * be used to callwlate the tolal length of a message buffer.
 *
 * @param[in] pBuf         Buffer reference
 * @returns Total buffer length
 */

static inline uint64_t
LwDspStrMessageLength(pLwDspStrMessage pBuf)
{
    LwDspStrBufferElement *last;
    LwDspStrBufferElement *current;
    uint64_t messageLength;

    if (pBuf == NULL)
        return(0);
    for (current = pBuf->element,
             messageLength = 0,
             last = (current +
                     pBuf->count);
         current < last;
         current++) {
        messageLength += current->length;
    }
    return(messageLength);
}

/**
 * Allocate a message buffer.
 *
 * The resulting object includes the LwDspStrMessage, the
 * LwDspStrBufferElement array, and the specified amount data storage.
 * The first item in the element array is set to point to the total
 * data area allocated, if the element array has at least one element.
 * dataSize must be zero if the element array count is zero.  The
 * pFree pointer is set to LwDspStrMessageFree, but may be changed by
 * the caller.  The element array and the data area, if allocated,
 * will be aligned on an uint64_t boundary.
 *
 * This is a utility function, for the colwenience of the library
 * implementation and the library user.   Since the free function
 * is specified in the LwDspStrMessage, any allocator or combination
 * of allocators may be used.
 *
 * @param[out] pBuf         Reference to variable to receive pointer to message buffer.
 *                          Set to NULL on an error.
 * @param[in] elementCount Count of elements required (zero or more)
 * @param[in] dataSize     Size of data area required (may be zero)
 * @param[in] pConnection  Library-private pointer returned by LwDspStrInit() call
 * @returns Error code:
 * -            LWOS_STATUS_SUCCESS Successful allocation
 * -            LWOS_STATUS_ERROR_ILWALID_ARGUMENT Zero element_count when
 *                         dataSize is non-zero
 * -            LWOS_STATUS_ERROR_OPERATING_SYSTEM Not enough memory
 */

extern uint32_t
LwDspStrMessageAlloc(pLwDspStrMessage *pBuf,
                     uint32_t elementCount,
                     uint32_t dataSize,
                     void *pConnection);

/**
 * Free a message buffer allocated via LwDspStrMessageAlloc().
 *
 * Frees the buffer.  It does nothing if pBuf is NULL.  This should
 * not be called if the references field is non-zero, unless the
 * caller can account for all the references.  That is, this routine
 * should normally be called from LwDspStrMessageRelease().
 *
 * @param[in] pBuf         Message reference
 * @param[in] pPrivate     Private pointer supplied on LwDspStrInit() call
 * @param[in] pConnection  Library-private pointer returned by LwDspStrInit() call
 */

extern void
LwDspStrMessageFree(pLwDspStrMessage pBuf,
                    void *pPrivate,
                    void *pConnection);

/**
 * Send a message buffer to the library.  The caller must have a hold
 * on the buffer across the call, and should not release it (as in a
 * separate thread) until the call returns.
 *
 * @param[in] pBuf         Reference to message buffer
 * @param[in] pPrivate     Private pointer supplied on LwDspStrInit() call
 * @param[in] pConnection  Library-private pointer returned by LwDspStrInit() call
 */

extern void
LwDspStrMessageSend(pLwDspStrMessage pBuf,
                    void *pPrivate,
                    void *pConnection);
 
/**
 * Pull up message data into a buffer as needed.
 *
 * If the data in the mesage at the specified offset
 * is contiguous, return a pointer to it.  Otherwise, 
 * copy the data to the temporary buffer and return
 * a pointer to that buffer.  The temporary buffer must
 * be large enough for the specified data length and always
 * be supplied.
 * 
 * LwDspStrBufferPullup() and LwDspStrBufferApply() are colwenience
 * utility routines for accessing message buffers.  
 * LwDspStrBufferPullup() allows a caller to get a pointer to
 * a range of bytes (typically a structure) from the message in a
 * contiguous location. Usually, this will just be a pointer into one
 * of the message buffers, but, if the range is split across one or
 * more elements of the buffer, the bytes will be copied into the
 * supplied temporary buffer and a pointer to that area
 * returned. LwDspStrBufferApply() is used to loop over an array of
 * ranges of bytes (typically an array of structures).
 *
 * @param[in] pBuf       Reference to the message
 * @param[in] dataOffset Offset to data 
 * @param[in] dataLength Length of data
 * @param[in] pTemp      Reference to temporary buffer
 * @param[out] ppData    Reference to variable to receive
 *                       data pointer.
 * @returns Error code:
 * -   LWOS_STATUS_SUCCESS Data located
 * -   LWOS_STATUS_ERROR_ILWALID_ARGUMENT 
 *                       Invalid offset or length out of
 *                       range
 */

extern uint32_t
LwDspStrBufferPullup(pLwDspStrMessage pBuf,
                     uint32_t dataOffset,
                     uint32_t dataLength,
                     void *pTemp,
                     void **ppData);

/**
 * Type of function to a apply to a buffer.
 *
 * Used with LwDspStrBufferApply().
 *
 * @param[in] pOpaque Pointer supplied by caller of
 *                    LwDspStrBufferApply()
 * @param[in] pData   Data to be processed
 * @returns uint32_t:
 * -            1     Terminate loop early
 * -            0     Continue processing
 */

typedef uint32_t (*pLwDspStrBufferApplyFunction)(void *pOpaque,
                                              void *pData);


/**
 * Apply function to message data in a buffer.
 *
 * For each unit of data, apply the specified function.  If the unit
 * data in the buffer is contiguous, pass a pointer to it.  Otherwise,
 * copy the data to the temporary buffer and pass a pointer to that
 * buffer.  The temporary buffer must be large enough for the
 * specified unitLength, and must always be supplied.  The dataLength
 * must be a multiple of the unitLength.
 *
 * @param[in] pBuf       Reference to the buffer
 * @param[in] dataOffset Offset to data 
 * @param[in] dataLength Length of data
 * @param[in] unitLength Length of each unit of data
 * @param[in] pTemp      Reference to temporary buffer
 * @param[in] pFunction  Reference to function to apply
 * @param[in] pOpaque    Opaque pointer to pass to function
 * @returns Error code:
 * -   LWOS_STATUS_SUCCESS Data located
 * -   LWOS_STATUS_ERROR_ILWALID_ARGUMENT 
 *                       NULL pointer or 
 *                       offset or length out of
 *                       range or dataLength not
 *                       a multiple of unitLength
 */

extern uint32_t
LwDspStrBufferApply(pLwDspStrMessage pBuf,
                    uint32_t dataOffset,
                    uint32_t dataLength,
                    uint32_t unitLength,
                    void *pTemp,
                    pLwDspStrBufferApplyFunction pFunction,
                    void *pOpaque);

/**
 * Send a message buffer to the user  The caller must have a hold
 * on the buffer across the call, and should not release it (as in a
 * separate thread) until the call returns.
 *
 * @param[in] pBuf         Reference to message buffer
 * @param[in] pPrivate     Private pointer supplied on LwDspStrInit() call
 * @param[in] pConnection  Library-private pointer returned by LwDspStrInit() call
 */

typedef void (*pLwDspStrMessageSend)(pLwDspStrMessage pBuf,
                                     void *pPrivate,
                                     void *pConnection);
 
/**
 * Initialize library connection.
 *
 * This routine must be called before other use of the library
 * for a given device.  If the object is a GPU (not a specific display),
 * then in messages which use a displayNumber should specify the number
 * the head to which the message applies.  If the object is a specific 
 * display device,the displayNumber must be 0.
 *
 * When working with a logical display split over two heads of GPU,
 * the user could have a separate connetion for each, or could specify the
 * GPU and get frames for both (in separate messages) via the same connection.
 * For SLI configurations, where a logical GPU may have multiple real GPUs
 * as slaves, and where the display is composed of multiple hardware display
 * surfaces, the library treats the master GPU as the owner of the defined
 * logical heads, so the displays refer to the logical constructs, not the
 * separate heads on the individual slave GPUs.
 *
 * @param[in] object        Identification of the display entity (GPU).
 * @param[in] pMessageSend  Callback function for sending messages
 * @param[in] pPrivate      Private pointer to pass through to callback
 * @param[out] ppConnection Receives library-private connection pointer,
 *                          to be supplied on subsequent calls
 * @returns Error code:
 * -            LWOS_STATUS_SUCCESS Successful initialization
 * -            LWOS_STATUS_ERROR_ILWALID_ARGUMENT 
 *                          Invalid client, device, or display handles,
 *                          or NULL pMessageSend
 * -            LWOS_STATUS_ERROR_OPERATING_SYSTEM Not enough memory
 * -            LWOS_STATUS_ERROR_IN_USE Library connection to specified
 *                           device is already in use
 */

extern uint32_t
LwDspStrInit(LwDspStrGraphicsObject object,
             pLwDspStrMessageSend pMessageSend,
             void *pPrivate,
             void **ppConnection);

/**
 * Terminate library connection.
 *
 * This routine must be called to close a library connection to a device.
 *
 * @param[in] pConnection   Library-private connection pointer
 * @returns Error code:
 * -            LWOS_STATUS_SUCCESS Successful initialization
 * -            LWOS_STATUS_ERROR_ILWALID_ARGUMENT 
 *                          Invalid pConnection
 */

extern uint32_t
LwDspStrTerminate(void *pConnection);

/*@}*/

/**********************************************************************/
/**
 * @defgroup FunctionMessageHeaders Message Headers
 */
/**********************************************************************/
/*@{*/

/**
 * Pixel format type.
 */

enum LwDspStrPixelFormatE {
    LW_DSPSTR_PIXEL_FORMAT_MIN = 0,           /*!< minimum value in range */

    LW_DSPSTR_PIXEL_FORMAT_8 = 0,             /*!< 256 colors via palette in 8 bits in 1 byte  */
    LW_DSPSTR_PIXEL_FORMAT_15 = 1,            /*!< X1R5G5B5 in 2 bytes */
    LW_DSPSTR_PIXEL_FORMAT_16 = 2,            /*!< R5G6B5 2 bytes */
    LW_DSPSTR_PIXEL_FORMAT_32 = 3,            /*!< A8R868B8 in 4 bytes */
    LW_DSPSTR_PIXEL_FORMAT_32_BGR = 4,        /*!< A8B8G8R8 in 4 bytes */

    LW_DSPSTR_PIXEL_FORMAT_MAX = 4            /*!< maximum value in range */
};

typedef uint32_t LwDspStrPixelFormat;
/*!< pixel format type */

/**
 * Display plane types
 */

typedef enum {
    LwDspStrPlaneBase = (1u << 0),  /*!< base plane         */
    LwDspStrPlaneOverlay = (1u << 1), /*!< overlay plane    */
    LwDspStrPlaneLwrsor = (1u << 2) /*!< cursor plane       */
} LwDspStrPlane;

typedef uint32_t LwDspStrPlaneSet; /*!< set of plane types     */

/**
 * Display frame configuration record.
 *
 * When this record is supplied by the library, the pitch value will
 * be correctly set.   When this record is supplied by the user of
 * the library, pitch may be zero, and will be set appropriately by the
 * library when the configuration is used.
 */

typedef struct {
    uint32_t displayNumber;    /*!< number of head          */
    uint32_t height;        /*!< height in pixels   */
    uint32_t width;         /*!< width in pixels    */
    uint32_t pitch;         /*!< length of a line in bytes */
    LwDspStrPixelFormat pixelType; /*!< pixel format   */
    LwDspStrPlaneSet planeSet; 
    /*!< set of planes to which this configuration
     *   applies                                    */
} LwDspStrFrameConfiguration;

#define LW_DSPSTR_DISPLAY_ALL ((uint32_t) (~0u))
/*!< reserved value for display number to indicate all displays */

/**
 * Power levels
 */

typedef enum {
    LwDspStrPowerLevelOn = 0,
    LwDspStrPowerLevelStandby = 1,
    LwDspStrPowerLevelSuspend = 2,
    LwDspStrPowerLevelOff = 3
} LwDspStrPowerLevel;

/**
 * Block (rectangle) descriptor.
 */

typedef struct {
    uint32_t x;      /*!< X position of first block */
    uint32_t y;      /*!< Y position of first block */
    uint32_t height; /*!< height of block in pixels */
    uint32_t width;  /*!< width of block in pixels */
    uint32_t pitch;  /*!< pitch (row length of block) in bytes */
} LwDspStrBlockDescriptor;

/**
 * Basic message header (both directions)
 *
 * This basic header is used for several simple message types,
 * such as LwDspStrMessageTypeEDID.  The header is followed by
 * contentLnegth bytes of EDID data.
 *
 * The EDID (extended device identification record) describes the
 * capabilities of a display device.   The EDID format is defined
 * in a collection of VESA standards.  For an application displaying
 * on a real device, the application may obtain the EDID of the real
 * device via the software interface to the display.  For an application
 * displaying in a window, the application may synthesize an EDID. 
 */

typedef struct {
    uint32_t displayNumber;    /*!< number of head          */
    uint32_t contentLength;    /*!< length of EDID data          */
} LwDspStrHeaderBasic; 

/**
 * LwDspStrMessageTypeNop message header (both directions)
 *
 * A response mode message is returned, with the 
 * version set to the interface version supported by
 * the recipient.
 */

typedef struct {
} LwDspStrHeaderNop;

/**
 * LwDspStrMessageTypeProvideBuffer message header (user to library)
 *
 * This message supplies a buffer for holding the content in later
 * frame messages.  The library holds the buffer until the last frame
 * message pointing at the buffer is freed, and until it has been used
 * in reuseCount messages or has been the subject of a
 * LwDspStrMessageTypeRevokeBuffer message.  If reuseCount is
 * LW_DSPSTR_REUSE_UNLIMITED, the library uses the buffer until the
 * use of the buffer is revoked, and hence does not decrement the
 * reuseCount on use.
 * 
 * For the purpose of use as a frame buffer, entire data area
 * following the header, less the first contentOffset bytes, is the
 * buffer.
 *
 * A buffer will not be reused until after the message referencing
 * it has been released, except when bufferAvailable is not null and
 * bufferAvailable is posted true.
 *
 * The buffer is immediately released without being used if 
 * reuseCount is 0.
 *
 * If the bufferTag matches a bufferTag for a buffer previously
 * provided and not yet reused, a "revoke buffer" response is sent
 * for the previous buffer.
 *
 * The bufferTag need be unique only with respect to displayNumber.
 * It may not be LW_DSPSTR_BUFFER_TAG_NULL.
 *
 * bufferAvailable and bufferFilled, if not null, are set to semaphore
 * objects known to a graphics API.  They are intended to tested and
 * set by hardware graphics commands, not CPU software, and may be
 * used to avoid software interaction with the display pipeline.  When
 * bufferAvailable is not null, the library will clear it before
 * delivering the buffer via a messaage or via setting bufferFilled.
 * Similarly, the application using the library (or hardware graphics
 * commands acting on its behalf) should clear bufferFilled before
 * returning the buffer to the library, whether via a message or via
 * setting bufferAvailable.
 * 
 * If the library has at least one buffer provided via a provide
 * buffer message and not revoked via a revoke buffer message, the
 * library will only use provided buffers for frames.  If the number
 * of such buffers is less than the pipeline depth (as set by the pipe
 * depth message), the library will limit itself ot an effective
 * pipeline depth equal to the number of buffers.  If not buffers have
 * been provided, the library will supply buffers for frames, up to
 * the pipeline depth.
 *
 * Use of the provide buffer message is most appropriate if the
 * application intends to pass the buffers to LWCA or other GPU
 * processing (such as texture mapping).
 */

typedef struct {
    uint32_t displayNumber;    /*!< number of head          */
    LwDspStrPlaneSet planeSet; /*!< set of planes for which this
                                *   buffer is to be used        */
    LwDspStrGraphicsObject bufferAvailable;
    /*!< semaphore to be tested to determine when a reusable buffer
     *   is available, or LW_DSPSTR_NULL_GRAPHICS_OBJECT if none. */
    uint32_t bufferAvailableIndex;
    /*!< index of semaphore in bufferAvailable object */
    LwDspStrGraphicsObject bufferFilled;
    /*!< semaphore to be set to indicate when a reusable buffer
     *   is available, or LW_DSPSTR_NULL_GRAPHICS_OBJECT if none. */
    uint32_t bufferFilledIndex;
    /*!< index of semaphore in bufferFilled object */
    uint32_t reuseCount;       /*!< count of times buffer may be reused */
    uint32_t contentOffset;    /*!< offset of content from end of header */
    uint32_t bufferTag;        /*!< unique handle for buffer    */
    uint32_t contentLength;    /*!< length of pixel data buffer */
} LwDspStrHeaderProvideBuffer;

#define LW_DSPSTR_BUFFER_TAG_NULL ((uint32_t) (0u))
/*!< reserved value for bufferTag to indicate no tag in a frame message */

#define LW_DSPSTR_REUSE_UNLIMITED (~ ((uint32_t) 0u))
/*!< reserved value for reuseCount to indicate unlimited reuse */

/**
 * LwDspStrMessageTypeRevokeBuffer message header (both directions)
 *
 * A revoke buffer message from the user to the library specifies
 * the bufferTag of the provide buffer message to be revoked.
 *
 * A revoke buffer message from the library to the user reuses
 * the provide buffer message which supplied the buffer, with the
 * message type changed and with the reuseCount reduced by the count
 * of times the buffer has been used.  A revoke buffer message from
 * library is always sent in response to a revoke buffer message from
 * the user.  If the library has no buffer matching bufferTag,
 * it sends the revoke buffer message back to the user with the
 * reuseCount set to 0.
 *
 * reuseCount is ignored on a message from user to library.
 *
 */

typedef struct {
    uint32_t displayNumber;    /*!< number of head          */
    uint32_t reuseCount;       /*!< count of times buffer may be reused */
    uint32_t bufferTag;        /*!< unique handle for buffer    */
} LwDspStrHeaderRevokeBuffer;

/**
 * LwDspStrMessageTypeFrame message header (library to user).
 *
 * The planeSet in frameConfiguration will be set to LwDspStrPlaneBase
 * or LwDspStrPlaneOverlay if compositing is not enabled, to
 * (LwDspStrPlaneBase | LwDspStrPlaneOverlay) if compositing is 
 * enabled and host cursor is not enabled, and to (LwDspStrPlaneBase |
 * LwDspStrPlaneOverlay | LwDspStrPlaneLwrsor) otherwise.
 * 
 * Header is followed by pixels in row-major order, arranged according
 * to pixel format, starting at contentOffset bytes beyond the end
 * of the header.
 *
 * xPosition and yPosition give the position of the frame relative to
 * the screen.   These are both zero for a base or composited frame, 
 * and are only non-zero for an overlay frame if the overlay frame is
 * smaller than the base frame.
 *
 * If the message is using data space from a buffer provided by user
 * of the library via the "provide buffer" message, the bufferTag is
 * If the buffer is one provided by user of the library via the
 * "provide buffer" message, the bufferTag is set to the bufferTag of
 * that buffer, and the reuseCount is set to the remaining reuseCount
 * of that buffer, after it has been decremented for the use of the
 * buffer in the frame message.  If reuseCount for the provided buffer
 * is LW_DSPSTR_REUSE_UNLIMITED, then reuseCount is not decremented
 * and hence is the same value if this message.
 * 
 * If the buffer is not one provided by the user of the library,
 * bufferTag is LW_DSPSTR_BUFFER_TAG_NULL and reuseCount is 0.
 */

typedef struct {
    LwDspStrFrameConfiguration frameConfiguration; 
    /*!< frame shape and pixel type  */
    uint32_t reuseCount;       /*!< count of times buffer may be reused */
    uint32_t xPosition;        /*!< X pixel position of upper left corner */
    uint32_t yPosition;        /*!< Y pixel position of upper left corner */
    uint32_t contentOffset;    /*!< offset of content from end of header */
    uint32_t bufferTag;        /*!< unique handle for buffer    */
    uint32_t contentLength;        /*!< length of pixel data */
} LwDspStrHeaderFrame;

/**
 * LwDspStrMessageTypeDifferentialFrame message header (library to user).
 *
 * This message is the same as for LwDspStrMessageTypeFrame, except
 * that the pixel data is in differential format.  Differential 
 * format is as follows:
 *
 * - zero or more block runs:
 * --     LwDspStrBlockDescriptor block description
 * --     uint32_t         number of conselwtive blocks
 * --     blocks, with pixels in each block in row-major order
 *
 * The block description includes the X and Y position, height,
 * width and pitch (row length in bytes) for the first block.
 * The blocks in a run all have the same Y position, height, width,
 * and pitch, with the X(i) = (X(0) + (i * width)) for block i.
 *
 * contentLength includes just the above data (all of the block runs).
 */

typedef LwDspStrHeaderFrame LwDspStrHeaderDifferentialFrame;

/**
 * LwDspStrMessageTypeEDID message header (user to library).
 *
 * Header is followed by the EDID, in contentLength bytes.
 */

typedef LwDspStrHeaderBasic LwDspStrHeaderEDID;

/**
 * LwDspStrMessageTypeFrameFormat message header (library to user)
 *
 * This message informs the user of a change in the frame configuration
 */

typedef struct {
    LwDspStrFrameConfiguration frameConfiguration; 
    /*!< frame shape and pixel format */
} LwDspStrHeaderFrameFormat;

/**
 * Compositing mode flags
 */

enum LwDspStrFrameModeE {
    LwDspStrFrameModeCompositeOverlay = (1u << 0),
    /*!< composite overlay into base frames, if overlay is enabled */
    LwDspStrFrameModeCompositeLwrsor = (1u << 1),
    /*!< composite cursor into base frames */
    LwDspStrframeModeUpdateLwrsor = (1u << 2),
    /*!< send cursor update messages (without images unless
     *   LwDspStrFrameModeLwrsorImageUpdate is also set) */
    LwDspStrFrameModeLwrsorImageUpdate = (1u << 3),
    /*!< send images with cursor update messages */
    LwDspStrFrameModeSingleFrameMessage = (1u << 4),
    /*!< send just one frame message per reusable buffer */
    LwDspStrFrameModeBlockDifferential = (1u << 5),
    /*!< deliver frames in block differential format */
    LwDspStrFrameModeVirtualOnly = (1u << 6)
    /*!< if display is mirrored with a real display, 
     *   disable the real display                   */
};

typedef uint32_t LwDspStrFrameMode;
/*!< set of frame mode flags */

/**
 * LwDspStrMessageTypeFrameMode message header (both directions)
 *
 * This message specifies, for a given head, whether the user wants
 * base, overlay, and cursor to be composited or delivered separately,
 * as well as whether multiple frame messages should be sent for
 * reusable buffers.  The library acknowledges the message by sending
 * the message back, possibly modified to reflect what is possible.
 *
 * If LwDspStrFrameModeSingleFrameMessage is set, then, for any given
 * reusable buffer, a frame message will be sent only on the first time
 * it is filled.  This mode will in general only be useful if the
 * bufferAvailable and bufferFilled semaphore references were specified
 * in the "provide buffer" message for the buffer.  This mode applies
 * only to base and overlay frames, and not to cursor messages (which
 * are sent only when the cursor is not being composited.
 *
 * If LwDspStrFrameModeBlockDifferential is set, then
 * LwDspStrMessageTypeFrameDifferential messages are sent instead of
 * LwDspStrMessageTypeFrame messages, and the block size is determined
 * by differentialHeight and differentialWidth.  If the specified 
 * size is not possible on a user to library message, the response
 * message from the library to the user will specify the largest block
 * size not wider or taller than the requested size.  If there is no
 * such size, or if differential mode is not available for the given
 * display, the LwDspStrFrameModeBlockDifferential bit will be cleared.
 *
 * The default value for mode is (LwDspStrFrameModeCompositeOverlay |
 * LwDspStrFrameModeCompositeLwrsor) for a real head and
 * (LwDspStrFrameModeCompositeOverlay |
 * LwDspStrFrameModeCompositeLwrsor | LwDspStrFrameModeVirtualOnly) a
 * purely virtual head.  In the latter case,
 * LwDspStrFrameModeVirtualOnly will always be set in messages from
 * the library to the user.
 * 
 * The user may assume that the new mode is in effect for all frames
 * delivered after the response message.  When frames are being
 * delivered in single frame message mode, the caller must tolerate
 * the change being asynchronous to the response message delivery, or
 * tempoarily disable single frame message mode.
 */

typedef struct {
    uint32_t displayNumber;    /*!< number of head          */
    LwDspStrFrameMode mode; /*!< compositing mode flags */
    uint32_t differentialHeight; /*!< block height for differential mode */
    uint32_t differentialWidth; /*!< block width for differential mode */
} LwDspStrHeaderFrameMode;

/**
 *    LwDspStrMessageTypeHdcpDownstream message header (library to user) 
 *
 * This message delivers an HDCP message to the specified display, of
 * contentLength bytes, following the header.
 *
 * HDCP (high-bandwidth digital content protection) is standard managed
 * by Digital Content Protection LLC.  
 */

typedef LwDspStrHeaderBasic LwDspStrHeaderHdcpDownstream;

/**
 *    LwDspStrMessageTypeHdcpUpstream message header (user to library)
 *
 * This message delivers an HDCP message from the specified display,
 * of contentLength bytes, following the header.
 */

typedef LwDspStrHeaderBasic LwDspStrHeaderHdcpUpstream;

/**
 *    LwDspStrMessageTypeControlMonitor message header (both directions)
 */

typedef struct {
    uint32_t displayNumber;    /*!< number of head          */
    LwDspStrPowerLevel powerLevel;   /*!< monitor power level */
} LwDspStrHeaderControlMonitor;

/**
 *    LwDspStrMessageRequestEDID message header (library to user)
 *
 * This message requests that the user supply the EDID from the
 * specified display.  contentLength should be set to 0.
 *
 * The application must in general supply an EDID, since in general
 * there will be no real local display, as when the display is remote.
 */

typedef LwDspStrHeaderBasic LwDspStrHeaderRequestEDID;

/**
 * LwDspStrMessageTypeUpdateLwrsor message header (library to user).
 *
 * The planeSet in the frameConfiguration will be set to
 * LwDspStrPlaneLwrsor.
 *
 * If contentLength is non-zero, header is followed by pixels in
 * row-major order, arranged according to pixel format, starting at
 * contentOffset bytes beyond the end of the header.  The cursor
 * image will be included in this way at any time it may have 
 * changed, but the user of the library should not assume it has
 * changed because it has been included.
 *
 * xPosition and yPosition give the position of the cursor hot spot
 * relative to the screen. 
 *
 * xHotSpot and yHotSpot give the position of the lwrsort hot spot
 * relative to the cursor image.
 *
 * If the message is using data space from a buffer provided by user
 * of the library via the "provide buffer" message, the bufferTag is
 * set to the bufferTag of that buffer, and the reuseCount is set to
 * the remaining reuseCount of that buffer, after it has been
 * decremented for the use of the buffer in the frame message.  If
 * reuseCount for the provided buffer is LW_DSPSTR_REUSE_UNLIMITED,
 * then reuseCount is not decremented and hence is the same value if
 * this message.
 * 
 * If the buffer is not one provided by the user of the library,
 * bufferTag is LW_DSPSTR_BUFFER_TAG_NULL and reuseCount is 0.
 */

typedef struct {
    LwDspStrFrameConfiguration frameConfiguration; 
    /*!< frame shape and pixel type  */
    uint32_t reuseCount;       /*!< count of times buffer may still be reused */
    uint32_t xPosition;        /*!< X pixel position of cursor hot spot */
    uint32_t yPosition;        /*!< Y pixel position of cursor hot spot */
    uint32_t xHotSpot;         /*!< X pixel offset of hot spot in cursor image */
    uint32_t yHotSpot;         /*!< Y pixel offset of hot spot in cursor image */
    uint32_t contentOffset;    /*!< offset of content from end of header */
    uint32_t bufferTag;        /*!< unique handle for buffer    */
    uint32_t contentLength;    /*!< length of pixel data */
} LwDspStrHeaderUpdateLwrsor;

/**
 * LwDspStrMessageTypePipeDepth message header (both directions).
 *
 * Set (user to library) or report (library to user) the pipeline
 * depth for frame buffers for the specified head.  The library will
 * not have more than pipeDepth buffers outstanding with the user,
 * even if the user has provided more buffers via provide buffer
 * messages, except temporarily when the user reduces the limit when
 * the user owns more than pipeDepth frames.  Setting the pipeDepth to
 * zero suspends frame delivery.
 *
 * On receiving such a message, the library updates its limit and
 * sends the message back as confirmation.
 *
 * The default value of pipeDepth is 1.   After initialization, the library
 * will begin delivering frames, and continue until termination, except
 * when at least pipeDepth messages are outstanding with the application.
 */

typedef struct {
    uint32_t displayNumber;    /*!< number of head          */
    uint32_t pipeDepth;        /*!< maximum frame pipeline depth */
    uint32_t requestTag;       /*!< tag which may be used by application  
                                *   to match responses to requests */
} LwDspStrHeaderPipeDepth;

/**
 * LwDspStrMessageTypeRectangleList message header (user to library).
 *
 * Set the rectangle list for the stream.   If rectangleCount is zero,
 * the stream is set to retrieve the entire display (the default).
 * If the rectangleCount is non-zero, only data within the specified
 * rectangle list is delivered in each frame.   If diffeential mode,
 * only changed blocks within the specified rectangles are delivered.
 * When not in differential mode, the frames are delivered in differential
 * format (defined above), but all pixels, rather than just changed pixels,
 * are delivered.
 *
 * The rectangle list is an array of rectangleCount LwDspStrBlockDescriptor
 * records in the message following the header.
 */

typedef struct {
    uint32_t displayNumber;    /*!< number of head          */
    uint32_t rectangleCount;   /*!< count of rectangles in list */
} LwDspStrheaderRectangleList;

/*@}*/

#endif /* _LWDSPSTR_H_ */

/*
  ;; Local Variables: **
  ;; mode:c **
  ;; c-basic-offset:4 **
  ;; tab-width:4 **
  ;; indent-tabs-mode:nil **
  ;; End: **
*/
