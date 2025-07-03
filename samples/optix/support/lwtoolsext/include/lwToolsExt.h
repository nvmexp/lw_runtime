/*
* Copyright 2009-2012  LWPU Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to LWPU ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to LWPU and is being provided under the terms and conditions
* of a form of LWPU software license agreement.
*
* LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

/** \mainpage
 * \section Introduction
 * The LWPU Tools Extension library is a set of functions that a
 * developer can use to provide additional information to tools.
 * The additional information is used by the tool to improve
 * analysis and visualization of data.
 *
 * The library introduces close to zero overhead if no tool is
 * attached to the application.  The overhead when a tool is
 * attached is specific to the tool.
 */

#ifndef LWTOOLSEXT_H_
#define LWTOOLSEXT_H_

#if defined(_MSC_VER) /* Microsoft Visual C++ Compiler */
    #ifdef LWTX_EXPORTS
        #define LWTX_DECLSPEC
    #else
        #define LWTX_DECLSPEC __declspec(dllimport)
    #endif /* LWTX_EXPORTS */
    #define LWTX_API __stdcall
#else /* GCC and most other compilers */
    #define LWTX_DECLSPEC
    #define LWTX_API
#endif /* Platform */

/**
 * The lwToolsExt library depends on stdint.h.  If the build tool chain in use
 * does not include stdint.h then define LWTX_STDINT_TYPES_ALREADY_DEFINED
 * and define the following types:
 * <ul>
 *   <li>uint8_t
 *   <li>int8_t
 *   <li>uint16_t
 *   <li>int16_t
 *   <li>uint32_t
 *   <li>int32_t
 *   <li>uint64_t
 *   <li>int64_t
 *   <li>uintptr_t
 *   <li>intptr_t
 * </ul>
 #define LWTX_STDINT_TYPES_ALREADY_DEFINED if you are using your own header file.
 */
#ifndef LWTX_STDINT_TYPES_ALREADY_DEFINED
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * Tools Extension API version
 */
#define LWTX_VERSION 1

/**
 * Size of the lwtxEventAttributes_t structure.
 */
#define LWTX_EVENT_ATTRIB_STRUCT_SIZE ( (uint16_t)( sizeof(lwtxEventAttributes_v1) ) )

#define LWTX_NO_PUSH_POP_TRACKING ((int)-2)

typedef uint64_t lwtxRangeId_t;

/** \page EVENT_ATTRIBUTES Event Attributes
 *
 * \ref MARKER_AND_RANGES can be annotated with various attributes to provide
 * additional information for an event or to guide the tool's visualization of
 * the data. Each of the attributes is optional and if left unused the
 * attributes fall back to a default value.
 *
 * To specify any attribute other than the text message, the \ref
 * EVENT_ATTRIBUTE_STRUCTURE "Event Attribute Structure" must be used.
 */

/** ---------------------------------------------------------------------------
 * Color Types
 * ------------------------------------------------------------------------- */
typedef enum lwtxColorType_t
{
    LWTX_COLOR_UNKNOWN  = 0,                 /**< Color attribute is unused. */
    LWTX_COLOR_ARGB     = 1                  /**< An ARGB color is provided. */
} lwtxColorType_t;

/** ---------------------------------------------------------------------------
 * Payload Types
 * ------------------------------------------------------------------------- */
typedef enum lwtxPayloadType_t
{
    LWTX_PAYLOAD_UNKNOWN                = 0,   /**< Color payload is unused. */
    LWTX_PAYLOAD_TYPE_UNSIGNED_INT64    = 1,   /**< A unsigned integer value is used as payload. */
    LWTX_PAYLOAD_TYPE_INT64             = 2,   /**< A signed integer value is used as payload. */
    LWTX_PAYLOAD_TYPE_DOUBLE            = 3    /**< A floating point value is used as payload. */
} lwtxPayloadType_t;

/** ---------------------------------------------------------------------------
 * Message Types
 * ------------------------------------------------------------------------- */
typedef enum lwtxMessageType_t
{
    LWTX_MESSAGE_UNKNOWN        = 0,         /**< Message payload is unused. */
    LWTX_MESSAGE_TYPE_ASCII     = 1,         /**< A character sequence is used as payload. */
    LWTX_MESSAGE_TYPE_UNICODE   = 2          /**< A wide character sequence is used as payload. */
} lwtxMessageType_t;

/** \brief Event Attribute Structure.
 * \anchor EVENT_ATTRIBUTE_STRUCTURE
 *
 * This structure is used to describe the attributes of an event. The layout of
 * the structure is defined by a specific version of the tools extension
 * library and can change between different versions of the Tools Extension
 * library.
 *
 * \par Initializing the Attributes
 *
 * The caller should always perform the following three tasks when using
 * attributes:
 * <ul>
 *    <li>Zero the structure
 *    <li>Set the version field
 *    <li>Set the size field
 * </ul>
 *
 * Zeroing the structure sets all the event attributes types and values
 * to the default value.
 *
 * The version and size field are used by the Tools Extension
 * implementation to handle multiple versions of the attributes structure.
 *
 * It is recommended that the caller use one of the following to methods
 * to initialize the event attributes structure:
 *
 * \par Method 1: Initializing lwtxEventAttributes for future compatibility
 * \code
 * lwtxEventAttributes_t eventAttrib = {0};
 * eventAttrib.version = LWTX_VERSION;
 * eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
 * \endcode
 *
 * \par Method 2: Initializing lwtxEventAttributes for a specific version
 * \code
 * lwtxEventAttributes_t eventAttrib = {0};
 * eventAttrib.version = 1;
 * eventAttrib.size = (uint16_t)(sizeof(lwtxEventAttributes_v1));
 * \endcode
 *
 * If the caller uses Method 1 it is critical that the entire binary
 * layout of the structure be configured to 0 so that all fields
 * are initialized to the default value.
 *
 * The caller should either use both LWTX_VERSION and
 * LWTX_EVENT_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
 * and a versioned type (Method 2).  Using a mix of the two methods
 * will likely cause either source level incompatibility or binary
 * incompatibility in the future.
 *
 * \par Settings Attribute Types and Values
 *
 *
 * \par Example:
 * \code
 * // Initialize
 * lwtxEventAttributes_t eventAttrib = {0};
 * eventAttrib.version = LWTX_VERSION;
 * eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
 *
 * // Configure the Attributes
 * eventAttrib.colorType = LWTX_COLOR_ARGB;
 * eventAttrib.color = 0xFF880000;
 * eventAttrib.messageType = LWTX_MESSAGE_TYPE_ASCII;
 * eventAttrib.message.ascii = "Example";
 * \endcode
 *
 * In the example the caller does not have to set the value of
 * \ref ::lwtxEventAttributes_v1::category or
 * \ref ::lwtxEventAttributes_v1::payload as these fields were set to
 * the default value by {0}.
 * \sa
 * ::lwtxMarkEx
 * ::lwtxRangeStartEx
 * ::lwtxRangePushEx
 */
typedef struct lwtxEventAttributes_v1
{
    /**
    * \brief Version flag of the structure.
    *
    * Needs to be set to LWTX_VERSION to indicate the version of LWTX APIs
    * supported in this header file. This can optionally be overridden to
    * another version of the tools extension library.
    */
    uint16_t version;

    /**
    * \brief Size of the structure.
    *
    * Needs to be set to the size in bytes of the event attribute
    * structure used to specify the event.
    */
    uint16_t size;

    /**
     * \brief ID of the category the event is assigned to.
     *
     * A category is a user-controlled ID that can be used to group
     * events.  The tool may use category IDs to improve filtering or
     * enable grouping of events in the same category. The functions
     * \ref ::lwtxNameCategoryA or \ref ::lwtxNameCategoryW can be used
     * to name a category.
     *
     * Default Value is 0
     */
    uint32_t category;

    /** \brief Color type specified in this attribute structure.
     *
     * Defines the color format of the attribute structure's \ref COLOR_FIELD
     * "color" field.
     *
     * Default Value is LWTX_COLOR_UNKNOWN
     */
    int32_t colorType;              /* lwtxColorType_t */

    /** \brief Color assigned to this event. \anchor COLOR_FIELD
     *
     * The color that the tool should use to visualize the event.
     */
    uint32_t color;

    /**
     * \brief Payload type specified in this attribute structure.
     *
     * Defines the payload format of the attribute structure's \ref PAYLOAD_FIELD
     * "payload" field.
     *
     * Default Value is LWTX_PAYLOAD_UNKNOWN
     */
    int32_t payloadType;            /* lwtxPayloadType_t */

    int32_t reserved0;

    /**
     * \brief Payload assigned to this event. \anchor PAYLOAD_FIELD
     *
     * A numerical value that can be used to annotate an event. The tool could
     * use the payload data to reconstruct graphs and diagrams.
     */
    union payload_t
    {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
    } payload;

    /** \brief Message type specified in this attribute structure.
     *
     * Defines the message format of the attribute structure's \ref MESSAGE_FIELD
     * "message" field.
     *
     * Default Value is LWTX_MESSAGE_UNKNOWN
     */
    int32_t messageType;            /* lwtxMessageType_t */

    /** \brief Message assigned to this attribute structure. \anchor MESSAGE_FIELD
     *
     * The text message that is attached to an event.
     */
    union message_t
    {
        const char* ascii;
        const wchar_t* unicode;
    } message;

} lwtxEventAttributes_v1;

typedef struct lwtxEventAttributes_v1 lwtxEventAttributes_t;

/* ========================================================================= */
/** \defgroup MARKER_AND_RANGES Marker and Ranges
 *
 * Markers and ranges are used to describe events at a specific time (markers)
 * or over a time span (ranges) during the exelwtion of the application
 * respectively. The additional information is presented alongside all other
 * captured data and facilitates understanding of the collected information.
 */

/* ========================================================================= */
/** \name Markers
 */
/** \name Markers
 */
/** \addtogroup MARKER_AND_RANGES
 * \section MARKER Marker
 *
 * A marker describes a single point in time.  A marker event has no side effect
 * on other events.
 *
 * @{
 */

/* ------------------------------------------------------------------------- */
/** \brief Marks an instantaneous event in the application.
 *
 * A marker can contain a text message or specify additional information
 * using the event attributes structure.  These attributes include a text
 * message, color, category, and a payload. Each of the attributes is optional
 * and can only be sent out using the \ref lwtxMarkEx function.
 * If \ref lwtxMarkA or \ref lwtxMarkW are used to specify the the marker
 * or if an attribute is unspecified then a default value will be used.
 *
 * \param eventAttrib - The event attribute structure defining the marker's
 * attribute types and attribute values.
 *
 * \par Example:
 * \code
 * // zero the structure
 * lwtxEventAttributes_t eventAttrib = {0};
 * // set the version and the size information
 * eventAttrib.version = LWTX_VERSION;
 * eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
 * // configure the attributes.  0 is the default for all attributes.
 * eventAttrib.colorType = LWTX_COLOR_ARGB;
 * eventAttrib.color = 0xFF880000;
 * eventAttrib.messageType = LWTX_MESSAGE_TYPE_ASCII;
 * eventAttrib.message.ascii = "Example lwtxMarkEx";
 * lwtxMarkEx(&eventAttrib);
 * \endcode
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxMarkEx(const lwtxEventAttributes_t* eventAttrib);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Marks an instantaneous event in the application.
 *
 * A marker created using \ref lwtxMarkA or \ref lwtxMarkW contains only a
 * text message.
 *
 * \param message     - The message associated to this marker event.
 *
 * \par Example:
 * \code
 * lwtxMarkA("Example lwtxMarkA");
 * lwtxMarkW(L"Example lwtxMarkW");
 * \endcode
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxMarkA(const char* message);
LWTX_DECLSPEC void LWTX_API lwtxMarkW(const wchar_t* message);
/** @} */

/** @} */ /* END MARKER_AND_RANGES */

/* ========================================================================= */
/** \name Start/Stop Ranges
 */
/** \addtogroup MARKER_AND_RANGES
 * \section INDEPENDENT_RANGES Start/Stop Ranges
 *
 * Start/Stop ranges denote a time span that can expose arbitrary conlwrrency -
 * opposed to Push/Pop ranges that only support nesting. In addition the start
 * of a range can happen on a different thread than the end. For the
 * correlation of a start/end pair an unique correlation ID is used that is
 * returned from the start API call and needs to be passed into the end API
 * call.
 *
 * @{
 */

/* ------------------------------------------------------------------------- */
/** \brief Marks the start of a range.
 *
 * \param eventAttrib - The event attribute structure defining the range's
 * attribute types and attribute values.
 *
 * \return The unique ID used to correlate a pair of Start and End events.
 *
 * \remarks Ranges defined by Start/End can overlap.
 *
 * \par Example:
 * \code
 * lwtxEventAttributes_t eventAttrib = {0};
 * eventAttrib.version = LWTX_VERSION;
 * eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
 * eventAttrib.category = 3;
 * eventAttrib.colorType = LWTX_COLOR_ARGB;
 * eventAttrib.color = 0xFF0088FF;
 * eventAttrib.messageType = LWTX_MESSAGE_TYPE_ASCII;
 * eventAttrib.message.ascii = "Example RangeStartEnd";
 * lwtxRangeId_t rangeId = lwtxRangeStartEx(&eventAttrib);
 * // ...
 * lwtxRangeEnd(rangeId);
 * \endcode
 *
 * \sa
 * ::lwtxRangeEnd
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartEx(const lwtxEventAttributes_t* eventAttrib);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Marks the start of a range.
 *
 * \param message     - The event message associated to this range event.
 *
 * \return The unique ID used to correlate a pair of Start and End events.
 *
 * \remarks Ranges defined by Start/End can overlap.
 *
 * \par Example:
 * \code
 * lwtxRangeId_t r1 = lwtxRangeStartA("Range 1");
 * lwtxRangeId_t r2 = lwtxRangeStartW(L"Range 2");
 * lwtxRangeEnd(r1);
 * lwtxRangeEnd(r2);
 * \endcode
 * \sa
 * ::lwtxRangeEnd
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartA(const char* message);
LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartW(const wchar_t* message);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Marks the end of a range.
 *
 * \param id - The correlation ID returned from a lwtxRangeStart call.
 *
 * \sa
 * ::lwtxRangeStartEx
 * ::lwtxRangeStartA
 * ::lwtxRangeStartW
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxRangeEnd(lwtxRangeId_t id);
/** @} */

/** @} */


/* ========================================================================= */
/** \name Push/Pop Ranges
 */
/** \addtogroup MARKER_AND_RANGES
 * \section PUSH_POP_RANGES Push/Pop Ranges
 *
 * Push/Pop ranges denote nested time ranges. Nesting is maintained per thread
 * and does not require any additional correlation mechanism. The duration of a
 * push/pop range is defined by the corresponding pair of Push/Pop API calls.
 *
 * @{
 */

/* ------------------------------------------------------------------------- */
/** \brief Marks the start of a nested range
 *
 * \param eventAttrib - The event attribute structure defining the range's
 * attribute types and attribute values.
 *
 * \return The 0 based level of range being started.  If an error oclwrs a
 * negative value is returned.
 *
 * \par Example:
 * \code
 * lwtxEventAttributes_t eventAttrib = {0};
 * eventAttrib.version = LWTX_VERSION;
 * eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
 * eventAttrib.colorType = LWTX_COLOR_ARGB;
 * eventAttrib.color = 0xFFFF0000;
 * eventAttrib.messageType = LWTX_MESSAGE_TYPE_ASCII;
 * eventAttrib.message.ascii = "Level 0";
 * lwtxRangePushEx(&eventAttrib);
 *
 * // Re-use eventAttrib
 * eventAttrib.messageType = LWTX_MESSAGE_TYPE_UNICODE;
 * eventAttrib.message.unicode = L"Level 1";
 * lwtxRangePushEx(&eventAttrib);
 *
 * lwtxRangePop();
 * lwtxRangePop();
 * \endcode
 *
 * \sa
 * ::lwtxRangePop
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC int LWTX_API lwtxRangePushEx(const lwtxEventAttributes_t* eventAttrib);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Marks the start of a nested range
 *
 * \param message     - The event message associated to this range event.
 *
 * \return The 0 based level of range being started.  If an error oclwrs a
 * negative value is returned.
 *
 * \par Example:
 * \code
 * lwtxRangePushA("Level 0");
 * lwtxRangePushW(L"Level 1");
 * lwtxRangePop();
 * lwtxRangePop();
 * \endcode
 *
 * \sa
 * ::lwtxRangePop
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC int LWTX_API lwtxRangePushA(const char* message);
LWTX_DECLSPEC int LWTX_API lwtxRangePushW(const wchar_t* message);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Marks the end of a nested range
 *
 * \return The level of the range being ended. If an error oclwrs a negative
 * value is returned on the current thread.
 *
 * \sa
 * ::lwtxRangePushEx
 * ::lwtxRangePushA
 * ::lwtxRangePushW
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC int LWTX_API lwtxRangePop(void);
/** @} */

/** @} */

/* ========================================================================= */
/** \defgroup RESOURCE_NAMING Resource Naming
 *
 * This section covers calls that allow to annotate objects with user-provided
 * names in order to allow for a better analysis of complex trace data. All of
 * the functions take the handle or the ID of the object to name and the name.
 * The functions can be called multiple times during the exelwtion of an
 * application, however, in that case it is implementation dependent which
 * name will be reported by the tool.
 *
 * \section RESOURCE_NAMING_LWTX LWTX Resource Naming
 * The LWPU Tools Extension library allows to attribute events with additional
 * information such as category IDs. These category IDs can be annotated with
 * user-provided names using the respective resource naming functions.
 *
 * \section RESOURCE_NAMING_OS OS Resource Naming
 * In order to enable a tool to report system threads not just by their thread
 * identifier, the LWPU Tools Extension library allows to provide user-given
 * names to these OS resources.
 * @{
 */

/* ------------------------------------------------------------------------- */
/** \name Functions for LWTX Resource Naming
 */
/** @{
 * \brief Annotate an LWTX category.
 *
 * Categories are used to group sets of events. Each category is identified
 * through a unique ID and that ID is passed into any of the marker/range
 * events to assign that event to a specific category. The lwtxNameCategory
 * function calls allow the user to assign a name to a category ID.
 *
 * \param category - The category ID to name.
 * \param name     - The name of the category.
 *
 * \remarks The category names are tracked per process.
 *
 * \par Example:
 * \code
 * lwtxNameCategory(1, "Memory Allocation");
 * lwtxNameCategory(2, "Memory Transfer");
 * lwtxNameCategory(3, "Memory Object Lifetime");
 * \endcode
 *
 * \version \LWTX_VERSION_1
 */
LWTX_DECLSPEC void LWTX_API lwtxNameCategoryA(uint32_t category, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameCategoryW(uint32_t category, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \name Functions for OS Resource Naming
 */
/** @{
 * \brief Annotate an OS thread.
 *
 * Allows the user to name an active thread of the current process. If an
 * invalid thread ID is provided or a thread ID from a different process is
 * used the behavior of the tool is implementation dependent.
 *
 * \param threadId - The ID of the thread to name.
 * \param name     - The name of the thread.
 *
 * \par Example:
 * \code
 * lwtxNameOsThread(GetLwrrentThreadId(), "MAIN_THREAD");
 * \endcode
 *
 * \version \LWTX_VERSION_1
 */
LWTX_DECLSPEC void LWTX_API lwtxNameOsThreadA(uint32_t threadId, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameOsThreadW(uint32_t threadId, const wchar_t* name);
/** @} */

/** @} */ /* END RESOURCE_NAMING */

/* ========================================================================= */

#ifdef UNICODE
    #define lwtxMark            lwtxMarkW
    #define lwtxRangeStart      lwtxRangeStartW
    #define lwtxRangePush       lwtxRangePushW
    #define lwtxNameCategory    lwtxNameCategoryW
    #define lwtxNameOsThread    lwtxNameOsThreadW
#else
    #define lwtxMark            lwtxMarkA
    #define lwtxRangeStart      lwtxRangeStartA
    #define lwtxRangePush       lwtxRangePushA
    #define lwtxNameCategory    lwtxNameCategoryA
    #define lwtxNameOsThread    lwtxNameOsThreadA
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LWTOOLSEXT_H_ */
