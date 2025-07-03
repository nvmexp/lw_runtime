/*
 * Lightweight protocol buffers.
 * 
 * Copyright 2009 Simon Kallweit
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This file contains the definitions used by the code generated
 * by the protobuf compiler.
 */

#ifndef __PRBRT_H__
#define __PRBRT_H__


#define LW_STATUS           LwU32

#ifndef LW_TRUE
typedef LwU8 LwBool;
#define LW_TRUE           ((LwBool)(0 == 0))
#define LW_FALSE          ((LwBool)(0 != 0))
#endif

#define LWD_EXPORT __cdecl

#define PRB_FIELD_DEFAULTS              1
#define PRB_ADD_OPTIONAL_DEFAULT_FIELDS 0
#define PRB_PRINT_MESSAGE_LENGTH        0
#define PRB_LWSTOM_PRINT_ROUTINES       0

// Set these to 1 to help with debugging.
#define PRB_FIELD_NAMES                 0
#define PRB_MESSAGE_NAMES               0
#define PRB_METHOD_NAMES                0
#define PRB_SERVICE_NAMES               0


#define PRB_STATUS LwU32
#define PRB_OK 0
#define PRB_ERR_BUFFER_TOO_SMALL 1
#define PRB_ERR_INSUFFICIENT_RESOURCES 2
#define PRB_ERR_ILWALID_REQUEST 3
#define PRB_ERR_ILWALID_MESSAGE 4 // generic decoding error

// Maximum depth of message embedding
#ifndef PRB_MAX_DEPTH
#define PRB_MAX_DEPTH 8
#endif

// Maximum number of required fields in a message
#ifndef PRB_MAX_REQUIRED_FIELDS
#define PRB_MAX_REQUIRED_FIELDS 16
#endif

// Provide enum names as strings 
#ifndef PRB_ENUM_NAMES
#define PRB_ENUM_NAMES 0
#endif

#if PRB_ENUM_NAMES
#define PRB_MAYBE_ENUM_NAME(n) n,
#else
#define PRB_MAYBE_ENUM_NAME(n)
#endif

// Provide field names as strings 
#ifndef PRB_FIELD_NAMES
#define PRB_FIELD_NAMES 0
#endif

#if PRB_FIELD_NAMES
#define PRB_MAYBE_FIELD_NAME(n) n,
#else
#define PRB_MAYBE_FIELD_NAME(n)
#endif

// Provide field default values 
#ifndef PRB_FIELD_DEFAULTS
#define PRB_FIELD_DEFAULTS 0
#endif

#if PRB_FIELD_DEFAULTS
#define PRB_MAYBE_FIELD_DEFAULT_DEF(n) n
#define PRB_MAYBE_FIELD_DEFAULT(n) n,
#else
#define PRB_MAYBE_FIELD_DEFAULT_DEF(n)
#define PRB_MAYBE_FIELD_DEFAULT(n)
#endif

// Provide message names as strings 
#ifndef PRB_MESSAGE_NAMES
#define PRB_MESSAGE_NAMES 0
#endif

#if PRB_MESSAGE_NAMES
#define PRB_MAYBE_MESSAGE_NAME(n) n,
#else
#define PRB_MAYBE_MESSAGE_NAME(n)
#endif

// Provide method names as strings 
#ifndef PRB_METHOD_NAMES
#define PRB_METHOD_NAMES 0
#endif

#if PRB_METHOD_NAMES
#define PRB_MAYBE_METHOD_NAME(n) n,
#else
#define PRB_MAYBE_METHOD_NAME(n)
#endif

// Provide service names as strings 
#ifndef PRB_SERVICE_NAMES
#define PRB_SERVICE_NAMES 0
#endif

#if PRB_SERVICE_NAMES
#define PRB_MAYBE_SERVICE_NAME(n) n,
#else
#define PRB_MAYBE_SERVICE_NAME(n)
#endif

// Field labels 
#define PRB_REQUIRED       0
#define PRB_OPTIONAL       1
#define PRB_REPEATED       2

// Field value types 
#define PRB_DOUBLE         0
#define PRB_FLOAT          1
#define PRB_INT32          2
#define PRB_INT64          3
#define PRB_UINT32         4
#define PRB_UINT64         5
#define PRB_SINT32         6
#define PRB_SINT64         7
#define PRB_FIXED32        8
#define PRB_FIXED64        9
#define PRB_SFIXED32       10
#define PRB_SFIXED64       11
#define PRB_BOOL           12
#define PRB_ENUM           13
#define PRB_STRING         14
#define PRB_BYTES          15
#define PRB_MESSAGE        16

// Field flags 
#define PRB_HAS_DEFAULT    (1 << 0)
#define PRB_IS_PACKED      (1 << 1)
#define PRB_IS_DEPRECATED  (1 << 2)

typedef struct
{
    unsigned int label : 2;
    unsigned int typ : 6;
    unsigned int flags : 8;
} PRB_FIELD_OPTS;

// Protocol buffer wire types 
typedef enum 
{
    WT_VARINT = 0,
    WT_64BIT  = 1,
    WT_STRING = 2,
    WT_32BIT  = 5
} WIRE_TYPE;

// Protocol buffer wire values 
typedef union
{
    LwU64 varint;
    LwU64 int64;
    struct {
        LwU64 len;
        const void *data;
    } string;
    LwU32 int32;
} WIRE_VALUE;

typedef struct
{
    char *str;
    LwU32 len;
} PRB_VALUE_STRING;

typedef struct
{
    LwU8 *data;
    LwU32 len;
} PRB_VALUE_BYTES;

typedef struct
{
    void *data;
    LwU32 len;
} PRB_VALUE_MESSAGE;

typedef union
{
    LwF64 double_;
    LwF32 float_;
    LwS32 int32;
    LwS64 int64;
    LwU32 uint32;
    LwU64 uint64;
    LwBool bool_;
    PRB_VALUE_STRING string;
    PRB_VALUE_BYTES bytes;
    PRB_VALUE_MESSAGE message;
    int enum_;
    int null;
} PRB_VALUE;

typedef struct
{
    int value;
#if PRB_ENUM_NAMES
    const char *name;
#endif
} PRB_ENUM_MAPPING;

typedef struct
{
    const PRB_ENUM_MAPPING *mappings;
    LwU32 count;
#if PRB_ENUM_NAMES
    const char *name;
#endif
} PRB_ENUM_DESC;

struct PRB_MSG_DESC;

//* Protocol buffer field descriptor 
typedef struct PRB_FIELD_DESC
{
    LwU32 number;
    PRB_FIELD_OPTS opts;
    const struct PRB_MSG_DESC *msg_desc;
    const PRB_ENUM_DESC *enum_desc;
#if PRB_FIELD_NAMES
    const char *name;
#endif
#if PRB_FIELD_DEFAULTS
    const PRB_VALUE *def;
#endif
} PRB_FIELD_DESC;

//* Protocol buffer message descriptor 
typedef struct PRB_MSG_DESC
{
    LwU32 num_fields;
    const PRB_FIELD_DESC *fields;
#if PRB_MESSAGE_NAMES
    const char *name;
#endif
} PRB_MSG_DESC;

// Forward declaration 
struct PRB_SERVICE_DESC;

// Protocol buffer method descriptor 
struct PRB_METHOD_DESC
{
    const struct PRB_SERVICE_DESC *service;
    const PRB_MSG_DESC *req_desc;
    const PRB_MSG_DESC *res_desc;
#if PRB_METHOD_NAMES
    const char *name;
#endif
};

// Protocol buffer service descriptor 
typedef struct PRB_SERVICE_DESC
{
    LwU32 num_methods;
    const struct PRB_METHOD_DESC *methods;
#if PRB_SERVICE_NAMES
    const char *name;
#endif
} PRB_SERVICE_DESC;

#endif
