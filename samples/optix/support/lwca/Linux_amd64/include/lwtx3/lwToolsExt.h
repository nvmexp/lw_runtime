/*
* Copyright 2009-2016  LWPU Corporation.  All rights reserved.
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

/** \file lwToolsExt.h
 */

/* ========================================================================= */
/** \mainpage
 * \tableofcontents
 * \section INTRODUCTION Introduction
 *
 * The LWPU Tools Extension library is a set of functions that a
 * developer can use to provide additional information to tools.
 * The additional information is used by the tool to improve
 * analysis and visualization of data.
 *
 * The library introduces close to zero overhead if no tool is
 * attached to the application.  The overhead when a tool is
 * attached is specific to the tool.
 *
 * \section INITIALIZATION_SECTION Initialization
 *
 * Typically the tool's library that plugs into LWTX is indirectly 
 * loaded via elwiromental properties that are platform specific. 
 * For some platform or special cases, the user may be required 
 * to instead explicity initialize instead though.   This can also
 * be helpful to control when the API loads a tool's library instead
 * of what would typically be the first function call to emit info.
 * For these rare case, see \ref INITIALIZATION for additional information.
 *
 * \section MARKERS_AND_RANGES Markers and Ranges
 *
 * Markers and ranges are used to describe events at a specific time (markers)
 * or over a time span (ranges) during the exelwtion of the application
 * respectively. 
 *
 * \subsection MARKERS Markers
 * 
 * Markers denote specific moments in time.
 * 
 * 
 * See \ref DOMAINS and \ref EVENT_ATTRIBUTES for additional information on
 * how to specify the domain.
 * 
 * \subsection THREAD_RANGES Thread Ranges
 *
 * Thread ranges denote nested time ranges. Nesting is maintained per thread
 * per domain and does not require any additional correlation mechanism. The
 * duration of a thread range is defined by the corresponding pair of
 * lwtxRangePush* to lwtxRangePop API calls.
 *
 * See \ref DOMAINS and \ref EVENT_ATTRIBUTES for additional information on
 * how to specify the domain.
 *
 * \subsection PROCESS_RANGES Process Ranges
 *
 * Process ranges denote a time span that can expose arbitrary conlwrrency, as 
 * opposed to thread ranges that only support nesting. In addition the range
 * start event can happen on a different thread than the end marker. For the 
 * correlation of a start/end pair an unique correlation ID is used that is
 * returned from the start API call and needs to be passed into the end API
 * call.
 *
 * \subsection EVENT_ATTRIBUTES Event Attributes
 *
 * \ref MARKERS_AND_RANGES can be annotated with various attributes to provide
 * additional information for an event or to guide the tool's visualization of
 * the data. Each of the attributes is optional and if left unused the
 * attributes fall back to a default value. The attributes include:
 * - color
 * - category
 *
 * To specify any attribute other than the text message, the \ref
 * EVENT_ATTRIBUTE_STRUCTURE "Event Attribute Structure" must be used.
 *
 * \section DOMAINS Domains
 *
 * Domains enable developers to scope annotations. By default all events and
 * annotations are in the default domain. Additional domains can be registered.
 * This allows developers to scope markers, ranges, and resources names to
 * avoid conflicts.
 *
 * The function ::lwtxDomainCreateA or ::lwtxDomainCreateW is used to create
 * a named domain.
 * 
 * Each domain maintains its own
 * - categories
 * - thread range stacks
 * - registered strings
 *
 * The function ::lwtxDomainDestroy marks the end of the domain. Destroying 
 * a domain unregisters and destroys all objects associated with it such as 
 * registered strings, resource objects, named categories, and started ranges. 
 *
 * \section RESOURCE_NAMING Resource Naming
 *
 * This section covers calls that allow to annotate objects with user-provided
 * names in order to allow for a better analysis of complex trace data. All of
 * the functions take the handle or the ID of the object to name and the name.
 * The functions can be called multiple times during the exelwtion of an
 * application, however, in that case it is implementation dependent which
 * name will be reported by the tool.
 * 
 * \subsection CATEGORY_NAMING Category Naming
 *
 * Some function in this library support associating an integer category 
 * to enable filtering and sorting.  The category naming functions allow 
 * the application to associate a user friendly name with the integer 
 * category.  Support for domains have been added in LWTX_VERSION_2 to 
 * avoid collisions when domains are developed independantly. 
 *
 * \subsection RESOURCE_OBJECTS Resource Objects
 *
 * Resource objects are a generic mechanism for attaching data to an application 
 * resource.  The identifier field makes the association to a pointer or handle, 
 * while the type field helps provide deeper understanding of the identifier as 
 * well as enabling differentiation in cases where handles generated by different
 * APIs may collide.  The resource object may also have an associated message to
 * associate with the application resource, enabling further annotation of this 
 * object and how it is used.
 * 
 * The resource object was introduced in LWTX_VERSION_2 to supersede existing naming
 * functions and allow the application resource identified by those functions to be
 * associated to a domain.  The other naming functions are still supported for backward
 * compatibility but will be associated only to the default domain.
 *
 * \subsection RESOURCE_NAMING_OS Resource Naming
 * 
 * Some operating system resources creation APIs do not support providing a user friendly 
 * name, such as some OS thread creation APIs.  This API support resource naming though 
 * both through resource objects and functions following the pattern 
 * lwtxName[RESOURCE_TYPE][A|W](identifier, name).  Resource objects introduced in LWTX_VERSION 2 
 * supersede the other functions with a a more general method of assigning names to OS resources,
 * along with associating them to domains too.  The older lwtxName* functions are only associated 
 * with the default domain.
 * \section EXTENSIONS Optional Extensions
 * Optional extensions will either appear within the existing sections the extend or appear 
 * in the "Related Pages" when they introduce new concepts.
 */

 /**
 * Tools Extension API version
 */
#if defined(LWTX_VERSION) && LWTX_VERSION < 3
#error "Trying to #include LWTX version 3 in a source file where an older LWTX version has already been included.  If you are not directly using LWTX (the LWPU Tools Extension library), you are getting this error because libraries you are using have included different versions of LWTX.  Suggested solutions are: (1) reorder #includes so the newest LWTX version is included first, (2) avoid using the conflicting libraries in the same .c/.cpp file, or (3) update the library using the older LWTX version to use the newer version instead."
#endif

/* Header guard */
#if !defined(LWTX_VERSION)
#define LWTX_VERSION 3

#if defined(_MSC_VER)
#define LWTX_API __stdcall
#define LWTX_INLINE_STATIC __inline static
#else /*defined(__GNUC__)*/
#define LWTX_API
#define LWTX_INLINE_STATIC inline static
#endif /* Platform */

#if defined(LWTX_NO_IMPL)
/* When omitting implementation, avoid declaring functions inline */
/* without definitions, since this causes compiler warnings. */
#define LWTX_DECLSPEC
#elif defined(LWTX_EXPORT_API)
/* Allow overriding definition of LWTX_DECLSPEC when exporting API. */
/* Default is empty, meaning non-inline with external linkage. */
#if !defined(LWTX_DECLSPEC)
#define LWTX_DECLSPEC
#endif
#else
/* Normal LWTX usage defines the LWTX API inline with static */
/* (internal) linkage. */
#define LWTX_DECLSPEC LWTX_INLINE_STATIC
#endif

#include "lwtxDetail/lwtxLinkOnce.h"

#define LWTX_VERSIONED_IDENTIFIER_L3(NAME, VERSION) NAME##_v##VERSION
#define LWTX_VERSIONED_IDENTIFIER_L2(NAME, VERSION) LWTX_VERSIONED_IDENTIFIER_L3(NAME, VERSION)
#define LWTX_VERSIONED_IDENTIFIER(NAME) LWTX_VERSIONED_IDENTIFIER_L2(NAME, LWTX_VERSION)

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
 * #define LWTX_STDINT_TYPES_ALREADY_DEFINED if you are using your own header file.
 */
#ifndef LWTX_STDINT_TYPES_ALREADY_DEFINED
#include <stdint.h>
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** 
* Result Codes
*/

#define LWTX_SUCCESS 0
#define LWTX_FAIL 1
#define LWTX_ERR_INIT_LOAD_PROPERTY 2
#define LWTX_ERR_INIT_ACCESS_LIBRARY 3
#define LWTX_ERR_INIT_LOAD_LIBRARY 4
#define LWTX_ERR_INIT_MISSING_LIBRARY_ENTRY_POINT 5
#define LWTX_ERR_INIT_FAILED_LIBRARY_ENTRY_POINT 6
#define LWTX_ERR_NO_INJECTION_LIBRARY_AVAILABLE 7

/**
 * Size of the lwtxEventAttributes_t structure.
 */
#define LWTX_EVENT_ATTRIB_STRUCT_SIZE ( (uint16_t)( sizeof(lwtxEventAttributes_t) ) )

#define LWTX_NO_PUSH_POP_TRACKING ((int)-2)

typedef uint64_t lwtxRangeId_t;

/* Forward declaration of opaque domain registration structure */
struct lwtxDomainRegistration_st;
typedef struct lwtxDomainRegistration_st lwtxDomainRegistration;

/* \brief Domain Handle Structure.
* \anchor DOMAIN_HANDLE_STRUCTURE
*
* This structure is opaque to the user and is used as a handle to reference
* a domain.  This type is returned from tools when using the LWTX API to
* create a domain.
*
*/
typedef lwtxDomainRegistration* lwtxDomainHandle_t;

/* Forward declaration of opaque string registration structure */
struct lwtxStringRegistration_st;
typedef struct lwtxStringRegistration_st lwtxStringRegistration;

/* \brief Registered String Handle Structure.
* \anchor REGISTERED_STRING_HANDLE_STRUCTURE
*
* This structure is opaque to the user and is used as a handle to reference
* a registered string.  This type is returned from tools when using the LWTX
* API to create a registered string.
*
*/
typedef lwtxStringRegistration* lwtxStringHandle_t;

/* ========================================================================= */
/** \defgroup GENERAL General
 * @{
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
 * Message Types
 * ------------------------------------------------------------------------- */
typedef enum lwtxMessageType_t
{
    LWTX_MESSAGE_UNKNOWN          = 0,    /**< Message payload is unused. */
    LWTX_MESSAGE_TYPE_ASCII       = 1,    /**< A character sequence is used as payload. */
    LWTX_MESSAGE_TYPE_UNICODE     = 2,     /**< A wide character sequence is used as payload. */
    /* LWTX_VERSION_2 */
    LWTX_MESSAGE_TYPE_REGISTERED  = 3,    /**< A unique string handle that was registered
                                                with \ref lwtxDomainRegisterStringA() or 
                                                \ref lwtxDomainRegisterStringW(). */
} lwtxMessageType_t;

typedef union lwtxMessageValue_t
{
    const char* ascii;
    const wchar_t* unicode;
    /* LWTX_VERSION_2 */
    lwtxStringHandle_t registered;
} lwtxMessageValue_t;


/** @} */ /*END defgroup*/
/* ------------------------------------------------------------------------- */
/** \brief Force initialization (optional)
*
* Force LWTX library to initialize.  The first call to any LWTX API function
* will automatically initialize the entire API.  This can make the first call
* much slower than subsequent calls.  In applications where the first call to
* LWTX may be in a performance-critical section, calling lwtxInitialize before
* any performance-critical sections will ensure LWTX initialization oclwrs at
* an acceptable time.  Since lwtxInitialize takes no parameters and has no
* expected behavior besides initialization, it is colwenient to add a call to
* lwtxInitialize in LWTX-instrumented applications that need to force earlier
* initialization without changing any other code.  For example, if an app's
* first LWTX call is lwtxDomainCreate, and it is diffilwlt to move that call
* earlier because the domain handle must be stored in an object only created
* at that point, adding a call to lwtxInitialize at the top of main() will
* ensure the later call to lwtxDomainCreate is as fast as possible.
*
* \version \LWTX_VERSION_3
*
* \param reserved - must be zero or NULL.
*
* @{ */
LWTX_DECLSPEC void LWTX_API lwtxInitialize(const void* reserved);
/** @} */


/** @} */ /*END defgroup*/

/* ========================================================================= */
/** \defgroup EVENT_ATTRIBUTES Event Attributes
* @{
*/

/** ---------------------------------------------------------------------------
* Payload Types
* ------------------------------------------------------------------------- */
typedef enum lwtxPayloadType_t
{
    LWTX_PAYLOAD_UNKNOWN = 0,   /**< Color payload is unused. */
    LWTX_PAYLOAD_TYPE_UNSIGNED_INT64 = 1,   /**< A 64 bit unsigned integer value is used as payload. */
    LWTX_PAYLOAD_TYPE_INT64 = 2,   /**< A 64 bit signed integer value is used as payload. */
    LWTX_PAYLOAD_TYPE_DOUBLE = 3,   /**< A 64 bit floating point value is used as payload. */
    /* LWTX_VERSION_2 */
    LWTX_PAYLOAD_TYPE_UNSIGNED_INT32 = 4,   /**< A 32 bit floating point value is used as payload. */
    LWTX_PAYLOAD_TYPE_INT32 = 5,   /**< A 32 bit floating point value is used as payload. */
    LWTX_PAYLOAD_TYPE_FLOAT = 6    /**< A 32 bit floating point value is used as payload. */
} lwtxPayloadType_t;

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
 * \ref ::lwtxEventAttributes_v2::category or
 * \ref ::lwtxEventAttributes_v2::payload as these fields were set to
 * the default value by {0}.
 * \sa
 * ::lwtxDomainMarkEx
 * ::lwtxDomainRangeStartEx
 * ::lwtxDomainRangePushEx
 */
typedef struct lwtxEventAttributes_v2
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
        /* LWTX_VERSION_2 */
        uint32_t uiValue;
        int32_t iValue;
        float fValue;
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
    lwtxMessageValue_t message;

} lwtxEventAttributes_v2;

typedef struct lwtxEventAttributes_v2 lwtxEventAttributes_t;

/** @} */ /*END defgroup*/
/* ========================================================================= */
/** \defgroup MARKERS_AND_RANGES Markers and Ranges
 *
 * See \ref MARKERS_AND_RANGES for more details
 *
 * @{
 */

/** \name Marker */

/* ------------------------------------------------------------------------- */
/** \brief Marks an instantaneous event in the application.
*
* A marker can contain a text message or specify additional information
* using the event attributes structure.  These attributes include a text
* message, color, category, and a payload. Each of the attributes is optional
* and can only be sent out using the \ref lwtxDomainMarkEx function.
*
* lwtxDomainMarkEx(NULL, event) is equivalent to calling
* lwtxMarkEx(event).
*
* \param domain    - The domain of scoping the category.
* \param eventAttrib - The event attribute structure defining the marker's
* attribute types and attribute values.
*
* \sa
* ::lwtxMarkEx
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC void LWTX_API lwtxDomainMarkEx(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Marks an instantaneous event in the application.
 *
 * A marker can contain a text message or specify additional information
 * using the event attributes structure.  These attributes include a text
 * message, color, category, and a payload. Each of the attributes is optional
 * and can only be sent out using the \ref lwtxMarkEx function.
 * If \ref lwtxMarkA or \ref lwtxMarkW are used to specify the marker
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
 * \sa
 * ::lwtxDomainMarkEx
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
 * \sa
 * ::lwtxDomainMarkEx
 * ::lwtxMarkEx
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxMarkA(const char* message);
LWTX_DECLSPEC void LWTX_API lwtxMarkW(const wchar_t* message);
/** @} */


/** \name Process Ranges */

/* ------------------------------------------------------------------------- */
/** \brief Starts a process range in a domain.
*
* \param domain    - The domain of scoping the category.
* \param eventAttrib - The event attribute structure defining the range's
* attribute types and attribute values.
*
* \return The unique ID used to correlate a pair of Start and End events.
*
* \remarks Ranges defined by Start/End can overlap.
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("my domain");
* lwtxEventAttributes_t eventAttrib = {0};
* eventAttrib.version = LWTX_VERSION;
* eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
* eventAttrib.messageType = LWTX_MESSAGE_TYPE_ASCII;
* eventAttrib.message.ascii = "my range";
* lwtxRangeId_t rangeId = lwtxDomainRangeStartEx(&eventAttrib);
* // ...
* lwtxDomainRangeEnd(rangeId);
* \endcode
*
* \sa
* ::lwtxDomainRangeEnd
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxDomainRangeStartEx(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Starts a process range.
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
 * eventAttrib.message.ascii = "Example Range";
 * lwtxRangeId_t rangeId = lwtxRangeStartEx(&eventAttrib);
 * // ...
 * lwtxRangeEnd(rangeId);
 * \endcode
 *
 * \sa
 * ::lwtxRangeEnd
 * ::lwtxDomainRangeStartEx
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartEx(const lwtxEventAttributes_t* eventAttrib);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Starts a process range.
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
 *
 * \sa
 * ::lwtxRangeEnd
 * ::lwtxRangeStartEx
 * ::lwtxDomainRangeStartEx
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartA(const char* message);
LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartW(const wchar_t* message);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Ends a process range.
*
* \param domain - The domain 
* \param id - The correlation ID returned from a lwtxRangeStart call.
*
* \remarks This function is offered completeness but is an alias for ::lwtxRangeEnd. 
* It does not need a domain param since that is associated iwth the range ID at ::lwtxDomainRangeStartEx
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("my domain");
* lwtxEventAttributes_t eventAttrib = {0};
* eventAttrib.version = LWTX_VERSION;
* eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
* eventAttrib.messageType = LWTX_MESSAGE_TYPE_ASCII;
* eventAttrib.message.ascii = "my range";
* lwtxRangeId_t rangeId = lwtxDomainRangeStartEx(&eventAttrib);
* // ...
* lwtxDomainRangeEnd(rangeId);
* \endcode
*
* \sa
* ::lwtxDomainRangeStartEx
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC void LWTX_API lwtxDomainRangeEnd(lwtxDomainHandle_t domain, lwtxRangeId_t id);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Ends a process range.
 *
 * \param id - The correlation ID returned from an lwtxRangeStart call.
 *
 * \sa
 * ::lwtxDomainRangeStartEx
 * ::lwtxRangeStartEx
 * ::lwtxRangeStartA
 * ::lwtxRangeStartW
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxRangeEnd(lwtxRangeId_t id);
/** @} */

/** \name Thread Ranges */

/* ------------------------------------------------------------------------- */
/** \brief Starts a nested thread range.
*
* \param domain    - The domain of scoping.
* \param eventAttrib - The event attribute structure defining the range's
* attribute types and attribute values.
*
* \return The 0 based level of range being started. This value is scoped to the domain.
* If an error oclwrs, a negative value is returned.
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("example domain");
* lwtxEventAttributes_t eventAttrib = {0};
* eventAttrib.version = LWTX_VERSION;
* eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
* eventAttrib.colorType = LWTX_COLOR_ARGB;
* eventAttrib.color = 0xFFFF0000;
* eventAttrib.messageType = LWTX_MESSAGE_TYPE_ASCII;
* eventAttrib.message.ascii = "Level 0";
* lwtxDomainRangePushEx(domain, &eventAttrib);
*
* // Re-use eventAttrib
* eventAttrib.messageType = LWTX_MESSAGE_TYPE_UNICODE;
* eventAttrib.message.unicode = L"Level 1";
* lwtxDomainRangePushEx(domain, &eventAttrib);
*
* lwtxDomainRangePop(domain); //level 1
* lwtxDomainRangePop(domain); //level 0
* \endcode
*
* \sa
* ::lwtxDomainRangePop
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC int LWTX_API lwtxDomainRangePushEx(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Starts a nested thread range.
 *
 * \param eventAttrib - The event attribute structure defining the range's
 * attribute types and attribute values.
 *
 * \return The 0 based level of range being started. This level is per domain.
 * If an error oclwrs a negative value is returned.
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
 * ::lwtxDomainRangePushEx
 * ::lwtxRangePop
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC int LWTX_API lwtxRangePushEx(const lwtxEventAttributes_t* eventAttrib);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Starts a nested thread range.
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
 * ::lwtxDomainRangePushEx
 * ::lwtxRangePop
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC int LWTX_API lwtxRangePushA(const char* message);
LWTX_DECLSPEC int LWTX_API lwtxRangePushW(const wchar_t* message);
/** @} */


/* ------------------------------------------------------------------------- */
/** \brief Ends a nested thread range.
*
* \return The level of the range being ended. If an error oclwrs a negative
* value is returned on the current thread.
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreate("example library");
* lwtxDomainRangePushA(domain, "Level 0");
* lwtxDomainRangePushW(domain, L"Level 1");
* lwtxDomainRangePop(domain);
* lwtxDomainRangePop(domain);
* \endcode
*
* \sa
* ::lwtxRangePushEx
* ::lwtxRangePushA
* ::lwtxRangePushW
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC int LWTX_API lwtxDomainRangePop(lwtxDomainHandle_t domain);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Ends a nested thread range.
 *
 * \return The level of the range being ended. If an error oclwrs a negative
 * value is returned on the current thread.
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
 * ::lwtxRangePushEx
 * ::lwtxRangePushA
 * ::lwtxRangePushW
 *
 * \version \LWTX_VERSION_0
 * @{ */
LWTX_DECLSPEC int LWTX_API lwtxRangePop(void);
/** @} */


/** @} */ /*END defgroup*/
/* ========================================================================= */
/** \defgroup RESOURCE_NAMING Resource Naming
 *
 * See \ref RESOURCE_NAMING for more details
 *
 * @{
 */


/*  ------------------------------------------------------------------------- */
/** \name Functions for Generic Resource Naming*/
/*  ------------------------------------------------------------------------- */

/*  ------------------------------------------------------------------------- */
/** \cond SHOW_HIDDEN
* \brief Resource typing helpers.  
*
* Classes are used to make it easy to create a series of resource types 
* per API without collisions 
*/
#define LWTX_RESOURCE_MAKE_TYPE(CLASS, INDEX) ((((uint32_t)(LWTX_RESOURCE_CLASS_ ## CLASS))<<16)|((uint32_t)(INDEX)))
#define LWTX_RESOURCE_CLASS_GENERIC 1
/** \endcond */

/* ------------------------------------------------------------------------- */
/** \brief Generic resource type for when a resource class is not available.
*
* \sa
* ::lwtxDomainResourceCreate
*
* \version \LWTX_VERSION_2
*/
typedef enum lwtxResourceGenericType_t
{
    LWTX_RESOURCE_TYPE_UNKNOWN = 0,
    LWTX_RESOURCE_TYPE_GENERIC_POINTER = LWTX_RESOURCE_MAKE_TYPE(GENERIC, 1), /**< Generic pointer assumed to have no collisions with other pointers. */
    LWTX_RESOURCE_TYPE_GENERIC_HANDLE = LWTX_RESOURCE_MAKE_TYPE(GENERIC, 2), /**< Generic handle assumed to have no collisions with other handles. */
    LWTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE = LWTX_RESOURCE_MAKE_TYPE(GENERIC, 3), /**< OS native thread identifier. */
    LWTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX = LWTX_RESOURCE_MAKE_TYPE(GENERIC, 4) /**< POSIX pthread identifier. */
} lwtxResourceGenericType_t;



/** \brief Resource Attribute Structure.
* \anchor RESOURCE_ATTRIBUTE_STRUCTURE
*
* This structure is used to describe the attributes of a resource. The layout of
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
* Zeroing the structure sets all the resource attributes types and values
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
* lwtxResourceAttributes_t attribs = {0};
* attribs.version = LWTX_VERSION;
* attribs.size = LWTX_RESOURCE_ATTRIB_STRUCT_SIZE;
* \endcode
*
* \par Method 2: Initializing lwtxEventAttributes for a specific version
* \code
* lwtxResourceAttributes_v0 attribs = {0};
* attribs.version = 2;
* attribs.size = (uint16_t)(sizeof(lwtxResourceAttributes_v0));
* \endcode
*
* If the caller uses Method 1 it is critical that the entire binary
* layout of the structure be configured to 0 so that all fields
* are initialized to the default value.
*
* The caller should either use both LWTX_VERSION and
* LWTX_RESOURCE_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
* and a versioned type (Method 2).  Using a mix of the two methods
* will likely cause either source level incompatibility or binary
* incompatibility in the future.
*
* \par Settings Attribute Types and Values
*
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("example domain");
*
* // Initialize
* lwtxResourceAttributes_t attribs = {0};
* attribs.version = LWTX_VERSION;
* attribs.size = LWTX_RESOURCE_ATTRIB_STRUCT_SIZE;
*
* // Configure the Attributes
* attribs.identifierType = LWTX_RESOURCE_TYPE_GENERIC_POINTER;
* attribs.identifier.pValue = (const void*)pMutex;
* attribs.messageType = LWTX_MESSAGE_TYPE_ASCII;
* attribs.message.ascii = "Single thread access to database.";
*
* lwtxResourceHandle_t handle = lwtxDomainResourceCreate(domain, attribs);
* \endcode
*
* \sa
* ::lwtxDomainResourceCreate
*/
typedef struct lwtxResourceAttributes_v0
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
    * Needs to be set to the size in bytes of this attribute
    * structure.
    */
    uint16_t size;

    /**
    * \brief Identifier type specifies how to interpret the identifier field
    *
    * Defines the identifier format of the attribute structure's \ref RESOURCE_IDENTIFIER_FIELD
    * "identifier" field.
    *
    * Default Value is LWTX_RESOURCE_TYPE_UNKNOWN
    */
    int32_t identifierType;            /* values from enums following the pattern lwtxResource[name]Type_t */

    /**
    * \brief Identifier for the resource. 
    * \anchor RESOURCE_IDENTIFIER_FIELD
    *
    * An identifier may be a pointer or a handle to an OS or middleware API object.
    * The resource type will assist in avoiding collisions where handles values may collide.
    */
    union identifier_t
    {
        const void* pValue;
        uint64_t ullValue;
    } identifier;

    /** \brief Message type specified in this attribute structure.
    *
    * Defines the message format of the attribute structure's \ref RESOURCE_MESSAGE_FIELD
    * "message" field.
    *
    * Default Value is LWTX_MESSAGE_UNKNOWN
    */
    int32_t messageType;            /* lwtxMessageType_t */

    /** \brief Message assigned to this attribute structure. \anchor RESOURCE_MESSAGE_FIELD
    *
    * The text message that is attached to a resource.
    */
    lwtxMessageValue_t message;

} lwtxResourceAttributes_v0;

typedef struct lwtxResourceAttributes_v0 lwtxResourceAttributes_t;

/* \cond SHOW_HIDDEN 
* \version \LWTX_VERSION_2
*/
#define LWTX_RESOURCE_ATTRIB_STRUCT_SIZE ( (uint16_t)( sizeof(lwtxResourceAttributes_v0) ) )
typedef struct lwtxResourceHandle* lwtxResourceHandle_t;
/** \endcond */



/* ------------------------------------------------------------------------- */
/** \brief Create a resource object to track and associate data with OS and middleware objects
*
* Allows users to associate an API handle or pointer with a user-provided name.
* 
*
* \param domain - Domain to own the resource object
* \param attribs - Attributes to be associated with the resource
*
* \return A handle that represents the newly created resource object.
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("example domain");
* lwtxResourceAttributes_t attribs = {0};
* attribs.version = LWTX_VERSION;
* attribs.size = LWTX_RESOURCE_ATTRIB_STRUCT_SIZE;
* attribs.identifierType = LWTX_RESOURCE_TYPE_GENERIC_POINTER;
* attribs.identifier.pValue = (const void*)pMutex;
* attribs.messageType = LWTX_MESSAGE_TYPE_ASCII;
* attribs.message.ascii = "Single thread access to database.";
* lwtxResourceHandle_t handle = lwtxDomainResourceCreate(domain, attribs);
* \endcode
*
* \sa
* ::lwtxResourceAttributes_t
* ::lwtxDomainResourceDestroy
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC lwtxResourceHandle_t LWTX_API lwtxDomainResourceCreate(lwtxDomainHandle_t domain, lwtxResourceAttributes_t* attribs);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Destroy a resource object to track and associate data with OS and middleware objects
*
* Allows users to associate an API handle or pointer with a user-provided name.
*
* \param resource - Handle to the resource in which to operate.
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("example domain");
* lwtxResourceAttributes_t attribs = {0};
* attribs.version = LWTX_VERSION;
* attribs.size = LWTX_RESOURCE_ATTRIB_STRUCT_SIZE;
* attribs.identifierType = LWTX_RESOURCE_TYPE_GENERIC_POINTER;
* attribs.identifier.pValue = (const void*)pMutex;
* attribs.messageType = LWTX_MESSAGE_TYPE_ASCII;
* attribs.message.ascii = "Single thread access to database.";
* lwtxResourceHandle_t handle = lwtxDomainResourceCreate(domain, attribs);
* lwtxDomainResourceDestroy(handle);
* \endcode
*
* \sa
* ::lwtxDomainResourceCreate
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC void LWTX_API lwtxDomainResourceDestroy(lwtxResourceHandle_t resource);
/** @} */


/** \name Functions for LWTX Category Naming*/

/* ------------------------------------------------------------------------- */
/**
* \brief Annotate an LWTX category used within a domain.
*
* Categories are used to group sets of events. Each category is identified
* through a unique ID and that ID is passed into any of the marker/range
* events to assign that event to a specific category. The lwtxDomainNameCategory
* function calls allow the user to assign a name to a category ID that is
* specific to the domain.
*
* lwtxDomainNameCategory(NULL, category, name) is equivalent to calling
* lwtxNameCategory(category, name).
*
* \param domain    - The domain of scoping the category.
* \param category  - The category ID to name.
* \param name      - The name of the category.
*
* \remarks The category names are tracked per domain.
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("example");
* lwtxDomainNameCategoryA(domain, 1, "Memory Allocation");
* lwtxDomainNameCategoryW(domain, 2, L"Memory Transfer");
* \endcode
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC void LWTX_API lwtxDomainNameCategoryA(lwtxDomainHandle_t domain, uint32_t category, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxDomainNameCategoryW(lwtxDomainHandle_t domain, uint32_t category, const wchar_t* name);
/** @} */

/** \brief Annotate an LWTX category.
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
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameCategoryA(uint32_t category, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameCategoryW(uint32_t category, const wchar_t* name);
/** @} */

/** \name Functions for OS Threads Naming*/

/* ------------------------------------------------------------------------- */
/** \brief Annotate an OS thread.
 *
 * Allows the user to name an active thread of the current process. If an
 * invalid thread ID is provided or a thread ID from a different process is
 * used the behavior of the tool is implementation dependent.
 *
 * Tools expect thread ID to be a number that uniquely identifies the thread
 * at the time of the call. Note that a thread's ID can be reused after
 * it is destroyed. Tools may choose how to handle aliasing of thread IDs.
 *
 * POSIX pthread_t type returned by pthread_self() may not comply with these
 * expectations. Please use OS-specific thread ID instead of pthread_t.
 *
 * The thread name is associated to the default domain.  To support domains 
 * use resource objects via ::lwtxDomainResourceCreate.
 *
 * \param threadId - The ID of the thread to name.
 * \param name     - The name of the thread.
 *
 * \par Examples:
 * MS Windows:
 * \code
 * #include <windows.h>
 * lwtxNameOsThread(GetLwrrentThreadId(), "Current thread");
 * lwtxNameOsThread(GetThreadId(SomeThreadHandle), "Other thread");
 * \endcode
 *
 * Android:
 * \code
 * #include <unistd.h>
 * lwtxNameOsThreadA(gettid(), "Current thread");
 * lwtxNameOsThreadA(getpid(), "Main thread");
 * \endcode
 *
 * Linux:
 * \code
 * #include <sys/syscall.h>
 * lwtxNameOsThreadA(syscall(SYS_gettid), "Current thread");
 * \endcode
 * \code
 * #include <unistd.h>
 * lwtxNameOsThreadA(getpid(), "Main thread");
 * \endcode
 *
 * OS X:
 * \code
 * #include <sys/syscall.h>
 * lwtxNameOsThreadA(syscall(SYS_thread_selfid), "Current thread");
 * \endcode
 * \code
 * #include <pthread.h>
 * __uint64_t id;
 * pthread_threadid_np(pthread_self(), &id);
 * lwtxNameOsThreadA(id, "Current thread");
 * pthread_threadid_np(somePThreadId, &id);
 * lwtxNameOsThreadA(id, "Other thread");
 * \endcode
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameOsThreadA(uint32_t threadId, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameOsThreadW(uint32_t threadId, const wchar_t* name);
/** @} */


/** @} */ /*END defgroup*/
/* ========================================================================= */
/** \defgroup STRING_REGISTRATION String Registration
*
* Registered strings are intended to increase performance by lowering instrumentation
* overhead.  String may be registered once and the handle may be passed in place of
* a string where an the APIs may allow.
*
* See \ref STRING_REGISTRATION for more details
*
* @{
*/

/* ------------------------------------------------------------------------- */
/** \brief Register a string.

* Registers an immutable string with LWTX. Once registered the pointer used
* to register the domain name can be used in lwtxEventAttributes_t
* \ref MESSAGE_FIELD. This allows LWTX implementation to skip copying the
* contents of the message on each event invocation.
*
* String registration is an optimization. It is recommended to use string
* registration if the string will be passed to an event many times.
*
* String are not unregistered, except that by unregistering the entire domain
*
* \param domain  - Domain handle. If NULL then the global domain is used.
* \param string    - A unique pointer to a sequence of characters.
*
* \return A handle representing the registered string.
*
* \par Example:
* \code
* lwtxDomainCreateA("com.lwpu.lwtx.example");
* lwtxStringHandle_t message = lwtxDomainRegisterStringA(domain, "registered string");
* lwtxEventAttributes_t eventAttrib = {0};
* eventAttrib.version = LWTX_VERSION;
* eventAttrib.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
* eventAttrib.messageType = LWTX_MESSAGE_TYPE_REGISTERED;
* eventAttrib.message.registered = message;
* \endcode
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC lwtxStringHandle_t LWTX_API lwtxDomainRegisterStringA(lwtxDomainHandle_t domain, const char* string);
LWTX_DECLSPEC lwtxStringHandle_t LWTX_API lwtxDomainRegisterStringW(lwtxDomainHandle_t domain, const wchar_t* string);
/** @} */

/** @} */ /*END defgroup*/
/* ========================================================================= */
/** \defgroup DOMAINS Domains
*
* Domains are used to group events to a developer defined scope. Middleware
* vendors may also scope their own events to avoid collisions with the
* the application developer's events, so that the application developer may
* inspect both parts and easily differentiate or filter them.  By default
* all events are scoped to a global domain where NULL is provided or when
* using APIs provided b versions of LWTX below v2
*
* Domains are intended to be typically long lived objects with the intention
* of logically separating events of large modules from each other such as
* middleware libraries from each other and the main application.
*
* See \ref DOMAINS for more details
*
* @{
*/

/* ------------------------------------------------------------------------- */
/** \brief Register a LWTX domain.
*
* Domains are used to scope annotations. All LWTX_VERSION_0 and LWTX_VERSION_1
* annotations are scoped to the global domain. The function lwtxDomainCreate
* creates a new named domain.
*
* Each domain maintains its own lwtxRangePush and lwtxRangePop stack.
*
* \param name - A unique string representing the domain.
*
* \return A handle representing the domain.
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("com.lwpu.lwtx.example");
*
* lwtxMarkA("lwtxMarkA to global domain");
*
* lwtxEventAttributes_t eventAttrib1 = {0};
* eventAttrib1.version = LWTX_VERSION;
* eventAttrib1.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
* eventAttrib1.message.ascii = "lwtxDomainMarkEx to global domain";
* lwtxDomainMarkEx(NULL, &eventAttrib1);
*
* lwtxEventAttributes_t eventAttrib2 = {0};
* eventAttrib2.version = LWTX_VERSION;
* eventAttrib2.size = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
* eventAttrib2.message.ascii = "lwtxDomainMarkEx to com.lwpu.lwtx.example";
* lwtxDomainMarkEx(domain, &eventAttrib2);
* lwtxDomainDestroy(domain);
* \endcode
*
* \sa
* ::lwtxDomainDestroy
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC lwtxDomainHandle_t LWTX_API lwtxDomainCreateA(const char* name);
LWTX_DECLSPEC lwtxDomainHandle_t LWTX_API lwtxDomainCreateW(const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Unregister a LWTX domain.
*
* Unregisters the domain handle and frees all domain specific resources.
*
* \param domain    - the domain handle
*
* \par Example:
* \code
* lwtxDomainHandle_t domain = lwtxDomainCreateA("com.lwpu.lwtx.example");
* lwtxDomainDestroy(domain);
* \endcode
*
* \sa
* ::lwtxDomainCreateA
* ::lwtxDomainCreateW
*
* \version \LWTX_VERSION_2
* @{ */
LWTX_DECLSPEC void LWTX_API lwtxDomainDestroy(lwtxDomainHandle_t domain);
/** @} */


/** @} */ /*END defgroup*/
/* ========================================================================= */
/** \cond SHOW_HIDDEN */

#ifdef UNICODE
    #define lwtxMark            lwtxMarkW
    #define lwtxRangeStart      lwtxRangeStartW
    #define lwtxRangePush       lwtxRangePushW
    #define lwtxNameCategory    lwtxNameCategoryW
    #define lwtxNameOsThread    lwtxNameOsThreadW
    /* LWTX_VERSION_2 */
    #define lwtxDomainCreate         lwtxDomainCreateW
    #define lwtxDomainRegisterString lwtxDomainRegisterStringW
    #define lwtxDomainNameCategory   lwtxDomainNameCategoryW
#else
    #define lwtxMark            lwtxMarkA
    #define lwtxRangeStart      lwtxRangeStartA
    #define lwtxRangePush       lwtxRangePushA
    #define lwtxNameCategory    lwtxNameCategoryA
    #define lwtxNameOsThread    lwtxNameOsThreadA
    /* LWTX_VERSION_2 */
    #define lwtxDomainCreate         lwtxDomainCreateA
    #define lwtxDomainRegisterString lwtxDomainRegisterStringA
    #define lwtxDomainNameCategory   lwtxDomainNameCategoryA
#endif

/** \endcond */

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#define LWTX_IMPL_GUARD /* Ensure other headers cannot included directly */

#include "lwtxDetail/lwtxTypes.h"

#ifndef LWTX_NO_IMPL
#include "lwtxDetail/lwtxImpl.h"
#endif /*LWTX_NO_IMPL*/

#undef LWTX_IMPL_GUARD

#endif /* !defined(LWTX_VERSION) */
