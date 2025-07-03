 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: prbdec.cpp                                                        *|
|*                                                                            *|
|*          Implements a lightweight protobuf message decoder.                *|
|*                                                                            *|
 \****************************************************************************/
#include "../common.h"

#define UNUSED(x)  (void)(x)

//
// includes
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//
#include "lwdcommon.h"
#include "prbdec.h"

//
// printing/debug definitions
//
#if defined(_WIN32) || defined(_WIN64)
#define FORMAT64 "I64" // for VC < 8.0
#else
#define FORMAT64 "ll"
#endif

static LwBool bDebugPrint = LW_FALSE;
static LwU32  debugPrintIndent = 0;

#define DEBUG_PRINTF if (bDebugPrint) lwdPrintf
#define DEBUG_PRINTF_INDENT() if (bDebugPrint) _printIndent(debugPrintIndent)

#if PRB_MESSAGE_NAMES && PRB_FIELD_NAMES
#define LWDPRINTF_NAMES(msg, namesFmt, ...) lwdPrintf(msg namesFmt "\n", __VA_ARGS__)
#else
#define LWDPRINTF_NAMES(msg, namesFmt, ...) lwdPrintf(msg "\n")
#endif

static const char *PRB_TYPE_NAMES[] =
{
    "double", // PRB_DOUBLE
    "float", // PRB_FLOAT
    "int32", // PRB_INT32
    "int64", // PRB_INT64
    "uint32", // PRB_UINT32
    "uint64", // PRB_UINT64
    "sint32", // PRB_SINT32
    "sint64", // PRB_SINT64
    "fixed32", // PRB_FIXED32
    "fixed64", // PRB_FIXED64
    "sfixed32", // PRB_SFIXED32
    "sfixed64", // PRB_SFIXED64
    "bool", // PRB_BOOL
    "enum", // PRB_ENUM
    "string", // PRB_STRING
    "bytes", // PRB_BYTES
    "message", // PRB_MESSAGE
};

//
// static function declarations
//
static LwU32 _prbBufLeft(const PRB_DECODE_BUF *buf);
static LwU64 _prbDecodeVarint(PRB_DECODE_BUF *buf);
static PRB_STATUS _prbDecodeWireValue(LwU32 wireType, PRB_DECODE_BUF *pBuf, WIRE_VALUE *pValue);
static PRB_STATUS _prbDecodePackedField(PRB_FIELD *pField, const void *data, LwU32 length);
static void _prbCreateField(PRB_FIELD *pField, const PRB_FIELD_DESC *pFieldDesc);
static void _prbFieldDestroyEmbedded(PRB_FIELD *pField);
static void _prbDestroyField(PRB_FIELD *pField);
static PRB_STATUS _prbFieldAddValue(PRB_FIELD *pField, WIRE_VALUE *pValue);
static PRB_STATUS _prbFieldAddDefault(PRB_FIELD *pField);
static void _printIndent(LwU32 indentLevel);

#if PRB_MESSAGE_NAMES && PRB_FIELD_NAMES
static void _prbPrintSimpleValue(LwU32 type, const PRB_VALUE *pValue);
static void _prbPrintEnum(const PRB_ENUM_DESC *pEnum, int value);
static void _prbPrintBytes(const LwU8* bytes, LwU32 count, LwU32 indentLevel);
static const char *_prbGetFieldTypeName(const PRB_FIELD *pField);
#endif

#if PRB_FIELD_NAMES
static void _prbPrintFieldOutline(const PRB_FIELD *pField, LwU32 indentLevel);
#endif

/*!
 * @brief Create an empty protobuf message.
 *
 * @param [out] pMsg Holds the empty message.
 * @param [in] pMsgDesc Describes the type of message being created.
 *
 * @returns PRB_OK
 * @returns PRB_ERR_INSUFFICIENT_RESOURCES if out of memory.
 */
PRB_STATUS LWD_EXPORT
prbCreateMsg(PRB_MSG *pMsg, const PRB_MSG_DESC *pMsgDesc)
{
    LwU32 i = 0;

    pMsg->desc = pMsgDesc;
    pMsg->fields = (PRB_FIELD*)malloc(sizeof(PRB_FIELD) * pMsgDesc->num_fields);
    if (pMsg->fields == NULL)
    {
        LWDPRINTF_NAMES("Failed to allocate memory for message", " %s",
                        pMsgDesc->name);
        return PRB_ERR_INSUFFICIENT_RESOURCES;
    }
    pMsg->mergedMsgLen = 0; // added to on each decode

    for (i = 0; i < pMsgDesc->num_fields; ++i)
    {
        _prbCreateField(&pMsg->fields[i], &pMsgDesc->fields[i]);
    }

    return PRB_OK;
}

/*!
 * @brief Destroys a protobuf message relwrsively, freeing memory resources.
 *
 * @param [inout] pMsg Message to be relwrsively destroyed and zeroed out.
 */
void LWD_EXPORT
prbDestroyMsg(PRB_MSG *pMsg)
{
    LwU32 i = 0;

    for (i = 0; i < pMsg->desc->num_fields; ++i)
    {
        _prbDestroyField(&pMsg->fields[i]);
    }

    free(pMsg->fields);

    pMsg->desc = NULL;
    pMsg->fields = NULL;
    pMsg->mergedMsgLen = 0;
}

/*!
 * @brief Decodes a protobuf buffer into a message.
 *
 * Note that you can call prbDecodeMsg on a single pMsg multiple times using
 * different input buffers. Duplicate message fields are merged according
 * to the protobuf standard.
 *
 * See http://code.google.com/apis/protocolbuffers/docs/encoding.html
 * for details.
 *
 * @param [inout] pMsg Message created by prbCreateMsg with the same message type
 *                     that <data> was encoded with.
 * @param [in] data Pointer to the protobuf encoded buffer.
 * @param [in] length Length of the buffer in bytes.
 *
 * @returns PRB_OK
 * @returns PRB_ERR_ILWALID_MESSAGE if there was an error decoding the buffer.
 * @returns PRB_ERR_INSUFFICIENT_RESOURCES if out of memory.
 */
PRB_STATUS LWD_EXPORT
prbDecodeMsg(PRB_MSG *pMsg, const void *data, LwU32 length)
{
    PRB_DECODE_BUF buffer =
    {
        (const LwU8 *)data,
        (const LwU8 *)data,
        (const LwU8 *)data + length
    };
    LwU32 wireType = 0;
    LwU32 tagNumber = 0;
    WIRE_VALUE value = { 0 };
    LwU32 i = 0;
    PRB_STATUS status;

    pMsg->mergedMsgLen += length;

    while (_prbBufLeft(&buffer) > 0)
    {
        // decode key
        value.varint = _prbDecodeVarint(&buffer);
        wireType = (LwU32)(value.varint & 7);
        tagNumber = (LwU32)(value.varint >> 3);
        DEBUG_PRINTF_INDENT();
        DEBUG_PRINTF("Tag = %d ", tagNumber);

        // decode wire value
        status = _prbDecodeWireValue(wireType, &buffer, &value);
        if (status != PRB_OK)
        {
            return status;
        }

        // add value to appropriate field if found
        for (i = 0; i < pMsg->desc->num_fields; ++i)
        {
            if (pMsg->fields[i].desc->number == tagNumber)
            {
                if (pMsg->fields[i].desc->opts.flags & PRB_IS_PACKED)
                {
                    DEBUG_PRINTF("Packed\n");
                    status = _prbDecodePackedField(&pMsg->fields[i],
                                                   value.string.data,
                                                   (LwU32)value.string.len);
                }
                else
                {
                    DEBUG_PRINTF("Not packed ");
                    status = _prbFieldAddValue(&pMsg->fields[i], &value);
                }
                if (status != PRB_OK)
                {
                    return status;
                }
                break;
            }
        }

#if PRB_MESSAGE_NAMES && PRB_FIELD_NAMES
        if (i == pMsg->desc->num_fields)
        {
            // Tag was not found.
            lwdPrintf(
                "WARNING: %s contains unknown tag number %d.  "
                "Update the dump program.\n",
                pMsg->desc->name,
                tagNumber);
        }
#endif
    }

    // validate fields
    for (i = 0; i < pMsg->desc->num_fields; ++i)
    {
        switch (pMsg->fields[i].desc->opts.label)
        {
            case PRB_REQUIRED:
                if (pMsg->fields[i].count == 0)
                {
                    _prbFieldAddDefault(&pMsg->fields[i]);

                    LWDPRINTF_NAMES("Missing required field", " %s.%s",
                        pMsg->desc->name, pMsg->fields[i].desc->name);

                    // We don't want to lose the entire protobuf over a missing
                    // required field, so soldier on.
                }
                break;
            case PRB_OPTIONAL:
#if PRB_ADD_OPTIONAL_DEFAULT_FIELDS
                if ((pMsg->fields[i].count == 0) &&
                    (pMsg->fields[i].desc->opts.typ != PRB_MESSAGE))
                {
                    _prbFieldAddDefault(&pMsg->fields[i]);
                }
#endif // PRB_ADD_OPTIONAL_DEFAULT_FIELDS
                break;
        }
    }

    return PRB_OK;
}

/*!
 * @brief Get a field from a decoded message.
 *
 * @param [in] pMsg The message that contains the field.
 * @param [in] pFieldDesc The descriptor specifying which field to get.
 *
 * @returns The requested field.
 * @returns NULL if not found.
 */
const PRB_FIELD * LWD_EXPORT
prbGetField(const PRB_MSG *pMsg, const PRB_FIELD_DESC *pFieldDesc)
{
    LwU32 i = 0;

    for (i = 0; i < pMsg->desc->num_fields; ++i)
    {
        if (pMsg->fields[i].desc == pFieldDesc)
        {
            return &pMsg->fields[i];
        }
    }

    LWDPRINTF_NAMES("Invalid message field", " %s.%s",
        pMsg->desc->name, pFieldDesc->name);

    return NULL;
}

#if PRB_FIELD_NAMES
/*!
 * @brief Get a field from a decoded message using the field's name.
 *
 * @param [in] pMsg The message that contains the field.
 * @param [in] fieldName The name of the field to get.
 *
 * @returns The requested field.
 * @returns NULL if not found.
 */
const PRB_FIELD * LWD_EXPORT
prbGetFieldByName(const PRB_MSG *pMsg, const char *fieldName)
{
    LwU32 i = 0;

    for (i = 0; i < pMsg->desc->num_fields; ++i)
    {
        if (strcmp(pMsg->fields[i].desc->name, fieldName) == 0)
        {
            return &pMsg->fields[i];
        }
    }

    LWDPRINTF_NAMES("Invalid message field", " %s.%s",
        pMsg->desc->name, fieldName);

    return NULL;
}
#endif

#if PRB_MESSAGE_NAMES && PRB_FIELD_NAMES
/*!
 * @brief Prints a message relwrsively to stdout in a text format.
 *
 * @param [in] pMsg The decoded message to print out.
 * @param [in] indentLevel The number of spaces to indent divided by 4. This is
 *                         used in the relwrsion to indent nested messages.
 *                         This should usually be set to 0 for the initial call.
 */
void LWD_EXPORT
prbPrintMsg(const PRB_MSG *pMsg, LwU32 indentLevel)
{
    LwU32 i = 0;

#if PRB_LWSTOM_PRINT_ROUTINES
    if (lwstomPrintMsg(pMsg, indentLevel + 1))
        return;
#endif

    _printIndent(indentLevel);
    lwdPrintf("{\n");

    for (i = 0; i < pMsg->desc->num_fields; ++i)
    {
        prbPrintField(&pMsg->fields[i], NULL, indentLevel + 1);
    }

    _printIndent(indentLevel);
    lwdPrintf("}\n");
}

/*!
 * @brief Prints the structure of a message relwrsively to stdout in text format.
 *
 * The outline structure that is printed shows which fields in the message are
 * populated and the length of repeated fields.
 *
 * @param [in] pMsg The decoded message to print out the structure of.
 * @param [in] indentLevel The number of spaces to indent divided by 4. This is
 *                         used in the relwrsion to indent nested messages.
 *                         This should usually be set to 0 for the initial call.
 */
void LWD_EXPORT
prbPrintMsgOutline(const PRB_MSG *pMsg, LwU32 indentLevel)
{
    LwU32 i = 0;

    for (i = 0; i < pMsg->desc->num_fields; ++i)
    {
        _prbPrintFieldOutline(&pMsg->fields[i], indentLevel + 1);
    }
}

/*!
 * @brief Prints a field relwrsively to stdout in text format.
 *
 * @param [in] pField The decoded field to print.
 * @param [in] pIndex This can be used to specifify a single element of a
                      repeated field. If NULL, then all elements of the field
                      are printed.
 * @param [in] indentLevel The number of spaces to indent divided by 4. This is
 *                         used in the relwrsion to indent nested fields.
 *                         This should usually be set to 0 for the initial call.
 */
void LWD_EXPORT
prbPrintField(const PRB_FIELD *pField, const LwU32 *pIndex, LwU32 indentLevel)
{
    LwU32 i;
    const char *typeName = _prbGetFieldTypeName(pField);

#if PRB_LWSTOM_PRINT_ROUTINES
    if (lwstomPrintField(pField, pIndex, indentLevel))
        return;
#endif

    if ((pField->count == 1) || (pIndex != NULL))
    {
        _printIndent(indentLevel);
        lwdPrintf("%s %s", typeName, pField->desc->name);
        if (pIndex == NULL)
        {
            i = 0;
        }
        else
        {
            i = *pIndex;
            lwdPrintf("[%u]", i);
        }
        lwdPrintf(" = ");

        if (i >= pField->count)
        {
            lwdPrintf("Invalid index!\n");
            return;
        }

#if PRB_PRINT_MESSAGE_LENGTH
        if (pField->desc->opts.typ == PRB_MESSAGE)
        {
            lwdPrintf(" // Len: %d", ((const PRB_MSG *)pField->values[i].message.data)->mergedMsgLen);
        }
#endif

        switch (pField->desc->opts.typ)
        {
            case PRB_ENUM:
                _prbPrintEnum(pField->desc->enum_desc, pField->values[i].enum_);
                lwdPrintf("\n");
                break;
            case PRB_BYTES:
                lwdPrintf("\n");
                _prbPrintBytes(pField->values[i].bytes.data,
                               pField->values[i].bytes.len, indentLevel);
                break;
            case PRB_MESSAGE:
                lwdPrintf("\n");
                prbPrintMsg((const PRB_MSG *)pField->values[i].message.data,
                            indentLevel);
                break;
            default:
                _prbPrintSimpleValue(pField->desc->opts.typ, &pField->values[i]);
                lwdPrintf("\n");
                break;
        }
    }
    else if (pField->count > 1)
    {
        _printIndent(indentLevel);
        lwdPrintf("%s %s[%u] =\n", typeName, pField->desc->name, pField->count);
        _printIndent(indentLevel);
        lwdPrintf("{\n");
        ++indentLevel;

        switch (pField->desc->opts.typ)
        {
            case PRB_ENUM:
                for (i = 0; i < pField->count; ++i)
                {
                    _printIndent(indentLevel);
                    lwdPrintf("[%u] = ", i);
                    _prbPrintEnum(pField->desc->enum_desc, pField->values[i].enum_);
                    lwdPrintf("\n");
                }
                break;
            case PRB_BYTES:
                for (i = 0; i < pField->count; ++i)
                {
                    _printIndent(indentLevel);
                    lwdPrintf("[%u] =\n", i);
                    _prbPrintBytes(pField->values[i].bytes.data,
                                  pField->values[i].bytes.len, indentLevel);
                }
                break;
            case PRB_MESSAGE:
                for (i = 0; i < pField->count; ++i)
                {
                    _printIndent(indentLevel);
#if PRB_PRINT_MESSAGE_LENGTH
                    lwdPrintf("[%u] = // Len: %d\n", i, ((const PRB_MSG *)pField->values[i].message.data)->mergedMsgLen);
#else
                    lwdPrintf("[%u] =\n", i);
#endif
                    prbPrintMsg((const PRB_MSG *)pField->values[i].message.data,
                                indentLevel);
                }
                break;
            default:
                for (i = 0; i < pField->count; ++i)
                {
                    _printIndent(indentLevel);
                    lwdPrintf("[%u] = ", i);
                    _prbPrintSimpleValue(pField->desc->opts.typ, &pField->values[i]);
                    lwdPrintf("\n");
                }
                break;
        }

        --indentLevel;
        _printIndent(indentLevel);
        lwdPrintf("}\n");
    }
}
#endif

/*!
 * @brief Translate an enum value into an enum name.
 *
 * @param [in] pEnumDesc The enum descriptor that maps values to names.
 * @param [out] value The decoded enum value.
 *
 * @returns The corresponding enum name.
 * @returns NULL if the enum value was not found.
 */
const char* LWD_EXPORT
prbGetEnumValueName(const PRB_ENUM_DESC *pEnumDesc, int value)
{
    UNUSED(pEnumDesc);
    UNUSED(value);

#if PRB_ENUM_NAMES
    LwU32 i;

    for (i = 0; i < pEnumDesc->count; ++i)
    {
        if (pEnumDesc->mappings[i].value == value)
        {
            return pEnumDesc->mappings[i].name;
        }
    }

    lwdPrintf("Invalid %s enum value %d\n", pEnumDesc->name, value);
#endif
    return NULL;
}

/*!
 * @brief Enable or disable debug printing during calls to prbDecodeMsg.
 *
 * @param [in] bDbgPrint True enables debug prints. False disables debug prints.
 */
void LWD_EXPORT
prbSetDebugPrintFlag(LwBool bDbgPrint)
{
    bDebugPrint = bDbgPrint;
}

#if PRB_MESSAGE_NAMES
/*!
 * @brief Find a message descriptor by name in an array of
 *        message descriptors provided by the client.
 *
 * @param [in] msgDescs Array of pointers to message descriptors,
 *                      terminated by a NULL pointer.
 * @param [in] name Name of the desired message descriptor.
 *
 * @returns A pointer to the desired message descriptor or NULL if not found.
 */
const PRB_MSG_DESC * LWD_EXPORT
prbGetMsgDescByName(const PRB_MSG_DESC **msgDescs, const char *name)
{
    const PRB_MSG_DESC **pPDesc;

    for (pPDesc = msgDescs; *pPDesc; ++pPDesc)
    {
        if (strcmp((*pPDesc)->name, name) == 0)
        {
            return *pPDesc;
        }
    }

    return NULL;
}
#endif

/*!
 * @brief Return a message contained within a message based on the message descriptor.
 *
 * This function is useful when looking for a specific message within
 * a message.  Here is an example of looking for the RC_GENERICDATA message
 * inside the DCL_ERRORBLOCK.
 *   prbStatus = prbCreateMsg(&PrbMsg, DCL_ERRORBLOCK);
 *   prbStatus = prbDecodeMsg(&PrbMsg, protocolBuffer, size);
 *   RcMsg = prbGetMsg(&PrbMsg, RC_GENERICDATA);
 * One can then examine the specific data within that message or just
 * print out that message out instead of the entire DCL_ERRORBLOCK message.
 *
 * @param [in] pMsg The decoded message to search in
 * @param [in] pMsgDesc Describes the type of message being searched for
 *
 * @returns A pointer to the desired message or NULL if not found
 *
 */
const PRB_MSG * LWD_EXPORT
prbGetMsg(const PRB_MSG *pMsg, const PRB_MSG_DESC *pMsgDesc)
{
    LwU32 i = 0;
    const PRB_MSG *pRetMsg;

    /* If message descriptor matches, search is over.  Return message. */
    if (pMsg->desc == pMsgDesc)
    {
        return pMsg;
    }

    /* Otherwise, walk through each field, and continue the search for
     * the message.  If message is found, then return back up. */
    for (i = 0; i < pMsg->desc->num_fields; ++i)
    {
        pRetMsg = prbGetMsgFromField(&pMsg->fields[i], pMsgDesc);
        if (NULL != pRetMsg)
        {
            return pRetMsg;
        }
    }

    return NULL;
}

/*!
 * @brief Finds a message in a field based on message descriptor.  Used
 *        in conjunction with prbGetMsg to find specific messages.
 *
 * @param [in] pField The decoded field to search in
 * @param [in] pMsgDesc Describes the type of message being searched for
 *
 * @returns A pointer to the desired message or NULL if not found
 */
const PRB_MSG * LWD_EXPORT
prbGetMsgFromField(const PRB_FIELD *pField, const PRB_MSG_DESC *pMsgDesc)
{
    LwU32 i = 0;
    const PRB_MSG *pRetMsg;

    /* Make sure there is at least a field */
    if (pField->count > 0)
    {
        /* Only care to look at the field if it has messages in it */
        if (PRB_MESSAGE == pField->desc->opts.typ)
        {
            for (i = 0; i < pField->count; ++i)
            {
                pRetMsg = prbGetMsg((const PRB_MSG *)pField->values[i].message.data, pMsgDesc);
                if (NULL != pRetMsg)
                {
                    return pRetMsg;
                }
            }
        }
    }

    return NULL;
}

/*!
 * @brief Query how many unread bytes are left in a protobuf buffer.
 *
 * @param [in] buf The protobuf buffer.
 *
 * @returns The number of unread bytes left in the buffer.
 */
static LwU32
_prbBufLeft(const PRB_DECODE_BUF *buf)
{
    if (buf->pos > buf->end)
    {
        lwdPrintf(
            "WARNING: _prbBufLeft overrun.  pos is %d, end is %d.\n",
            buf->pos,
            buf->end);

        return 0;
    }

    return (LwU32)(buf->end - buf->pos);
}

/*!
 * @brief Decode a variable integer in base-128 format.
 *
 * See http://code.google.com/apis/protocolbuffers/docs/encoding.html for details.
 *
 * @param [in] buf The probobuf buffer.
 *
 * @returns The decoded value as a 64-bit unsigned integer.
 */
static LwU64
_prbDecodeVarint(PRB_DECODE_BUF *buf)
{
    LwU64 value = 0;
    LwU32 shift = 0;
    while ((_prbBufLeft(buf) > 0) && (*buf->pos & 0x80))
    {
        value |= (LwU64)(*buf->pos & 0x7F) << shift;
        ++buf->pos;
        shift += 7;
    }
    if (_prbBufLeft(buf) > 0)
    {
        value |= (LwU64)*buf->pos << shift;
        ++buf->pos;
    }
    return value;
}

/*!
 * @brief Decode a typed wire value in a raw protobuf buffer.
 *
 * See http://code.google.com/apis/protocolbuffers/docs/encoding.html#structure for details.
 *
 * @param [in] wireType The wire type to be decoded.
 * @param [in] pBuf The protobuf buffer.
 * @param [out] pValue The structure to store the decoded wire value.
 *
 * @returns PRB_OK
 * @returns PRB_ERR_ILWALID_MESSAGE if wireType is invalid.
 */
static PRB_STATUS
_prbDecodeWireValue(LwU32 wireType, PRB_DECODE_BUF *pBuf, WIRE_VALUE *pValue)
{
    LwU32 i;

    switch (wireType)
    {
        case WT_VARINT:
            pValue->varint = _prbDecodeVarint(pBuf);
            DEBUG_PRINTF("WT_VARINT: 0x%"FORMAT64"X %"FORMAT64"d ", pValue->varint, pValue->varint);
            break;
        case WT_64BIT:
            pValue->int64 = 0;
            for (i = 0; i < 64; i += 8)
            {
                pValue->int64 |= (LwU64)*(pBuf->pos++) << i;
            }
            DEBUG_PRINTF("WT_64BIT: 0x%"FORMAT64"X %"FORMAT64"d ", pValue->int64, pValue->int64);
            break;
        case WT_STRING:
            pValue->string.len = _prbDecodeVarint(pBuf);
            pValue->string.data = pBuf->pos;
            pBuf->pos += pValue->string.len;
            DEBUG_PRINTF("WT_STRING len = 0x%"FORMAT64"X %"FORMAT64"u ", pValue->string.len, pValue->string.len);
            break;
        case WT_32BIT:
            pValue->int32 = 0;
            for (i = 0; i < 32; i += 8)
            {
                pValue->int32 |= (LwU32)*(pBuf->pos++) << i;
            }
            DEBUG_PRINTF("WT_32BIT: 0x%X %d ", pValue->int32, pValue->int32);
            break;
        default:
            lwdPrintf("Unrecognized wire type %u\n", wireType);
            return PRB_ERR_ILWALID_MESSAGE;
    }

    return PRB_OK;
}

/*!
 * @brief Decode a packed repeated field.
 *
 * See http://code.google.com/apis/protocolbuffers/docs/encoding.html#optional for details.
 *
 * @param [inout] pField Field created by _prbCreateField with the same field type
 *                       that <data> was encoded with.
 * @param [in] data Pointer to the protobuf encoded buffer.
 * @param [in] length Length of the buffer in bytes.
 *
 * @returns PRB_OK
 * @returns PRB_ERR_ILWALID_MESSAGE if there was an error decoding the buffer.
 * @returns PRB_ERR_INSUFFICIENT_RESOURCES if out of memory.
 */
static PRB_STATUS
_prbDecodePackedField(PRB_FIELD *pField, const void *data, LwU32 length)
{
    PRB_DECODE_BUF buffer =
    {
        (const LwU8 *)data,
        (const LwU8 *)data,
        (const LwU8 *)data + length
    };
    LwU32 wireType = WT_VARINT;
    WIRE_VALUE value = { 0 };
    PRB_STATUS status;

    ++debugPrintIndent;

    // determine the wire type
    switch (pField->desc->opts.typ)
    {
        case PRB_DOUBLE:
        case PRB_FIXED64:
        case PRB_SFIXED64:
            wireType = WT_64BIT;
            break;
        case PRB_FLOAT:
        case PRB_FIXED32:
        case PRB_SFIXED32:
            wireType = WT_32BIT;
            break;
        case PRB_INT32:
        case PRB_UINT32:
        case PRB_SINT32:
        case PRB_INT64:
        case PRB_UINT64:
        case PRB_SINT64:
        case PRB_BOOL:
        case PRB_ENUM:
            wireType = WT_VARINT;
            break;
        default:
            lwdPrintf("Invalid packed field wire type %u\n", wireType);
            return PRB_ERR_ILWALID_MESSAGE;
    }

    while (_prbBufLeft(&buffer) > 0)
    {
        // decode wire value
        DEBUG_PRINTF_INDENT();
        status = _prbDecodeWireValue(wireType, &buffer, &value);
        if (status != PRB_OK)
        {
            return status;
        }

        // add value to the field
        status = _prbFieldAddValue(pField, &value);
        if (status != PRB_OK)
        {
            return status;
        }
    }

    --debugPrintIndent;

    return PRB_OK;
}

/*!
 * @brief Create a protobuf field.
 *
 * @param [out] pField The newly created field.
 * @param [in] pFieldDesc The descriptor specifying which field to create.
 */
static void
_prbCreateField(PRB_FIELD *pField, const PRB_FIELD_DESC *pFieldDesc)
{
    pField->desc = pFieldDesc;
    pField->values = NULL;
    pField->count = 0;
}

/*!
 * @brief Destroy any nested values within a field, freeing resources.
 *
 * @param [inout] pField The field for which to destroy nested values.
 */
static void
_prbFieldDestroyEmbedded(PRB_FIELD *pField)
{
    LwU32 i = 0;

    switch (pField->desc->opts.typ)
    {
        case PRB_MESSAGE:
            for (i = 0; i < pField->count; ++i)
            {
                prbDestroyMsg((PRB_MSG *)pField->values[i].message.data);
            }
            // no break - free memory
        case PRB_STRING:
        case PRB_BYTES:
            for (i = 0; i < pField->count; ++i)
            {
                free(pField->values[i].bytes.data);
                pField->values[i].bytes.data = NULL;
                pField->values[i].bytes.len = 0;
            }
            break;
    }
}

/*!
 * @brief Destroy a field relwrsively, freeing resources.
 *
 * @param [inout] pField The field to destroy.
 */
static void
_prbDestroyField(PRB_FIELD *pField)
{
    _prbFieldDestroyEmbedded(pField);
    free(pField->values);

    pField->desc = NULL;
    pField->values = NULL;
    pField->count = 0;
}

/*!
 * @brief Add a raw wire value to a field.
 *
 * If the field is optional or required any previous value is replaced or in the
 * case of message fields merged.
 * If the field is repeated, then the field automatically grows to fit the new value.
 *
 * @param [inout] pField The field to add a value to.
 * @param [in] pValue The raw wire value to add.
 *
 * @returns PRB_OK
 * @returns PRB_ERR_ILWALID_MESSAGE if there was an error decoding the wire value.
 * @returns PRB_ERR_INSUFFICIENT_RESOURCES if out of memory.
 */
static PRB_STATUS
_prbFieldAddValue(PRB_FIELD *pField, WIRE_VALUE *pValue)
{
    LwU32 slot = 0;
    PRB_STATUS status;

    // allocate or reuse field memory
    switch (pField->desc->opts.label)
    {
        case PRB_REQUIRED:
        case PRB_OPTIONAL:
            if (pField->count == 1)
            {
                if (pField->desc->opts.typ != PRB_MESSAGE)
                {
                    _prbFieldDestroyEmbedded(pField);
                }
            }
            else
            {
                pField->values = (PRB_VALUE *)malloc(sizeof(PRB_VALUE));
                pField->count = 1;
                if (pField->desc->opts.typ == PRB_MESSAGE)
                {
                    pField->values[slot].message.data = NULL;
                }
            }
            break;
        case PRB_REPEATED:
            slot = pField->count;
            pField->values = (PRB_VALUE *)realloc(pField->values,
                                (++pField->count) * sizeof(PRB_VALUE));
            if (pField->desc->opts.typ == PRB_MESSAGE)
            {
                pField->values[slot].message.data = NULL;
            }
            break;
        default:
            lwdPrintf("Unrecognized field qualifier %u\n", pField->desc->opts.label);
            return PRB_ERR_ILWALID_MESSAGE;
    }
    if (pField->values == NULL)
    {
        LWDPRINTF_NAMES("Failed to allocate memory for field", " %s.%s[%u]",
            pField->desc->msg_desc->name, pField->desc->name, slot);
        return PRB_ERR_INSUFFICIENT_RESOURCES;
    }

    // add new value
    switch (pField->desc->opts.typ)
    {
        case PRB_DOUBLE:
            DEBUG_PRINTF("PRB_DOUBLE: %f\n", *(LwF64 *)&pValue->int64);
            pField->values[slot].double_ = *(LwF64 *)&pValue->int64;
            break;
        case PRB_FLOAT:
            DEBUG_PRINTF("PRB_FLOAT: %f\n", *(LwF32 *)&pValue->int32);
            pField->values[slot].float_ = *(LwF32 *)&pValue->int32;
            break;
        case PRB_INT32:
            DEBUG_PRINTF("PRB_INT32: %d\n", *(LwS32 *)&pValue->varint);
            pField->values[slot].int32 = *(LwS32 *)&pValue->varint;
            break;
        case PRB_INT64:
            DEBUG_PRINTF("PRB_INT64: %"FORMAT64"d\n", *(LwS64 *)&pValue->varint);
            pField->values[slot].int64 = *(LwS64 *)&pValue->varint;
            break;
        case PRB_UINT32:
            DEBUG_PRINTF("PRB_UINT32: %u\n", *(LwU32 *)&pValue->varint);
            pField->values[slot].uint32 = *(LwU32 *)&pValue->varint;
            break;
        case PRB_UINT64:
            DEBUG_PRINTF("PRB_UINT64: %"FORMAT64"u\n", *(LwU64 *)&pValue->varint);
            pField->values[slot].uint64 = *(LwU64 *)&pValue->varint;
            break;
        case PRB_SINT32:
            //
            // Zig-zag decoding.  We have to cast to signed before the shift,
            // so sign is propagated.
            //
            DEBUG_PRINTF("PRB_SINT32: %d\n",
                ((LwS32)pValue->varint ^ (((LwS32)pValue->varint << 31) >> 31)) >> 1);
            pField->values[slot].int32 =
                ((LwS32)pValue->varint ^ (((LwS32)pValue->varint << 31) >> 31)) >> 1;
            break;
        case PRB_SINT64:
            //
            // Zig-zag decoding.  We have to cast to signed before the shift,
            // so sign is propagated.
            //
            DEBUG_PRINTF("PRB_SINT64: %"FORMAT64"d\n",
                ((LwS64)pValue->varint ^ (((LwS64)pValue->varint << 63) >> 63)) >> 1);
            pField->values[slot].int64 =
                ((LwS64)pValue->varint ^ (((LwS64)pValue->varint << 63) >> 63)) >> 1;
            break;
        case PRB_FIXED32:
            DEBUG_PRINTF("PRB_FIXED32: %u\n", *(LwU32 *)&pValue->int32);
            pField->values[slot].uint32 = *(LwU32 *)&pValue->int32;
            break;
        case PRB_FIXED64:
            DEBUG_PRINTF("PRB_FIXED64: %"FORMAT64"u\n", *(LwU64 *)&pValue->int64);
            pField->values[slot].uint64 = *(LwU64 *)&pValue->int64;
            break;
        case PRB_SFIXED32:
            DEBUG_PRINTF("PRB_SFIXED32: %d\n", *(LwS32 *)&pValue->int32);
            pField->values[slot].int32 = *(LwS32 *)&pValue->int32;
            break;
        case PRB_SFIXED64:
            DEBUG_PRINTF("PRB_SFIXED64: %"FORMAT64"d\n", *(LwS64 *)&pValue->int64);
            pField->values[slot].int64 = *(LwS64 *)&pValue->int64;
            break;
        case PRB_BOOL:
            DEBUG_PRINTF("PRB_BOOL: %u\n", *(LwU32 *)&pValue->varint);
            pField->values[slot].bool_ = *(LwBool *)&pValue->varint;
            break;
        case PRB_ENUM:
            DEBUG_PRINTF("PRB_ENUM: %d\n", *(int *)&pValue->varint);
            pField->values[slot].enum_ = *(int *)&pValue->varint;
            break;
        case PRB_STRING:
            DEBUG_PRINTF("PRB_STRING\n");
            pField->values[slot].string.str =
                (char *) malloc((size_t)pValue->string.len + 1);
            if (pField->values[slot].string.str == NULL)
            {
                LWDPRINTF_NAMES("Failed to allocate memory for string",
                    " %s.%s[%u] of length %"FORMAT64"u",
                    pField->desc->msg_desc->name, pField->desc->name, slot,
                    pValue->string.len);
                return PRB_ERR_INSUFFICIENT_RESOURCES;
            }
            memcpy(pField->values[slot].string.str,
                   pValue->string.data,
                   (size_t)pValue->string.len);
            pField->values[slot].string.str[pValue->string.len] = '\0';

            pField->values[slot].string.len = (LwU32)pValue->string.len;
            break;
        case PRB_BYTES:
            DEBUG_PRINTF("PRB_BYTES\n");
            pField->values[slot].bytes.data =
                (LwU8 *) malloc((size_t)pValue->string.len);
            if (pValue->string.len > 0 && pField->values[slot].bytes.data == NULL)
            {
                LWDPRINTF_NAMES("Failed to allocate memory for bytes",
                    " %s.%s[%u] of length %"FORMAT64"u",
                    pField->desc->msg_desc->name, pField->desc->name, slot,
                    pValue->string.len);
                return PRB_ERR_INSUFFICIENT_RESOURCES;
            }
            memcpy(pField->values[slot].bytes.data,
                   pValue->string.data,
                   (size_t)pValue->string.len);

            pField->values[slot].bytes.len = (LwU32)pValue->string.len;
            break;
        case PRB_MESSAGE:
            DEBUG_PRINTF("PRB_MESSAGE\n");

            if (pField->values[slot].message.data == NULL)
            {
                pField->values[slot].message.data = malloc(sizeof(PRB_MSG));
                pField->values[slot].message.len = sizeof(PRB_MSG);

                if (pField->values[slot].message.data == NULL)
                {
                    LWDPRINTF_NAMES("Failed to allocate message field", " %s.%s[%u]",
                        pField->desc->msg_desc->name, pField->desc->name, slot);
                    return PRB_ERR_INSUFFICIENT_RESOURCES;
                }

                status = prbCreateMsg((PRB_MSG *)pField->values[slot].message.data,
                    pField->desc->msg_desc);

                if (status != PRB_OK)
                {
                    LWDPRINTF_NAMES("Failed to create message field", " %s.%s[%u]",
                        pField->desc->msg_desc->name, pField->desc->name, slot);
                    return status;
                }
            }

            ++debugPrintIndent;
            status = prbDecodeMsg((PRB_MSG *)pField->values[slot].message.data,
                                  pValue->string.data, (LwU32)pValue->string.len);
            --debugPrintIndent;
            return status;
        default:
            DEBUG_PRINTF("Unrecognized field type %u\n", pField->desc->opts.typ);
            return PRB_ERR_ILWALID_MESSAGE;
    }

    return PRB_OK;
}

/*!
 * @brief Set a field to its default value.
 *
 * @param [inout] pField The field to set to default.
 *
 * @returns PRB_OK
 * @returns PRB_ERR_ILWALID_MESSAGE if the field was a message field.
 * @returns PRB_ERR_INSUFFICIENT_RESOURCES if out of memory.
 */
static PRB_STATUS
_prbFieldAddDefault(PRB_FIELD *pField)
{
    const PRB_VALUE *def = pField->desc->def;

    pField->values = (PRB_VALUE *)malloc(sizeof(PRB_VALUE));
    pField->count = 1;

    if (pField->values == NULL)
    {
        LWDPRINTF_NAMES("Failed to allocate memory for default field", " %s.%s",
               pField->desc->msg_desc->name, pField->desc->name);
        return PRB_ERR_INSUFFICIENT_RESOURCES;
    }

    if (pField->desc->opts.flags & PRB_HAS_DEFAULT)
    {
        if (def == NULL)
        {
            lwdPrintf("Warning! Field with default flag has no default value.\n");
        }
    }
    else
    {
        if (def != NULL)
        {
            lwdPrintf("Warning! Field without default flag has default value.\n");
        }
        def = NULL;
    }

    if (def != NULL)
    {
        switch (pField->desc->opts.typ)
        {
            case PRB_DOUBLE:
                pField->values->double_ = def->double_;
                break;
            case PRB_FLOAT:
                pField->values->float_ = def->float_;
                break;
            case PRB_INT32:
            case PRB_UINT32:
            case PRB_SINT32:
            case PRB_FIXED32:
            case PRB_SFIXED32:
                pField->values->uint32 = def->uint32;
                break;
            case PRB_INT64:
            case PRB_UINT64:
            case PRB_SINT64:
            case PRB_FIXED64:
            case PRB_SFIXED64:
                pField->values->uint64 = def->uint64;
                break;
            case PRB_BOOL:
                pField->values->bool_ = def->bool_;
                break;
            case PRB_ENUM:
                pField->values->enum_ = def->enum_;
                break;
            case PRB_STRING:
                pField->values->string.str = (char *)malloc(def->string.len + 1);
                if (pField->values->string.str == NULL)
                {
                    LWDPRINTF_NAMES("Failed to allocate memory for default string",
                        " %s.%s of length %u", pField->desc->msg_desc->name,
                        pField->desc->name, def->string.len);
                    return PRB_ERR_INSUFFICIENT_RESOURCES;
                }
                strncpy(pField->values->string.str, def->string.str, def->string.len + 1);
                pField->values->string.len = def->string.len;
                break;
            case PRB_BYTES:
                pField->values->bytes.data = (LwU8 *)malloc(def->bytes.len);
                if (def->bytes.len > 0 && pField->values->bytes.data == NULL)
                {
                    LWDPRINTF_NAMES("Failed to allocate memory for default bytes",
                        " %s.%s of length %u\n", pField->desc->msg_desc->name,
                        pField->desc->name, def->bytes.len);
                    return PRB_ERR_INSUFFICIENT_RESOURCES;
                }
                strncpy((char *)pField->values->bytes.data, (char *)def->bytes.data, def->bytes.len);
                pField->values->bytes.len = def->bytes.len;
                break;
            default:
                lwdPrintf("Invalid default field type %u\n", pField->desc->opts.typ);
                return PRB_ERR_ILWALID_MESSAGE;
        }
    }
    else
    {
        switch (pField->desc->opts.typ)
        {
            case PRB_DOUBLE:
                pField->values->double_ = 0;
                break;
            case PRB_FLOAT:
                pField->values->float_ = 0;
                break;
            case PRB_INT32:
            case PRB_UINT32:
            case PRB_SINT32:
            case PRB_FIXED32:
            case PRB_SFIXED32:
                pField->values->uint32 = 0;
                break;
            case PRB_INT64:
            case PRB_UINT64:
            case PRB_SINT64:
            case PRB_FIXED64:
            case PRB_SFIXED64:
                pField->values->uint64 = 0;
                break;
            case PRB_BOOL:
                pField->values->bool_ = 0;
                break;
            case PRB_ENUM:
                pField->values->enum_ = pField->desc->enum_desc->mappings[0].value;
                break;
            case PRB_STRING:
                pField->values->string.str = (char *)malloc(1);
                if (pField->values->string.str == NULL)
                {
                    LWDPRINTF_NAMES("Failed to allocate memory for default string",
                        " %s.%s", pField->desc->msg_desc->name, pField->desc->name);
                    return PRB_ERR_INSUFFICIENT_RESOURCES;
                }
                pField->values->string.str[0] = '\0';
                pField->values->string.len = 0;
                break;
            case PRB_BYTES:
                pField->values->bytes.data = NULL;
                pField->values->bytes.len = 0;
                break;
            default:
                lwdPrintf("Invalid default field type %u\n", pField->desc->opts.typ);
                return PRB_ERR_ILWALID_MESSAGE;
        }
    }

    return PRB_OK;
}

/*!
 * @brief Print spaces to stdout to indent formatted output.
 *
 * @param [in] indentLevel The number of spaces to print divided by 4.
 */
static void
_printIndent(LwU32 indentLevel)
{
    LwU32 i = 0;
    for(i = 0; i < indentLevel; ++i)
    {
        lwdPrintf("    ");
    }
}

#if PRB_MESSAGE_NAMES && PRB_FIELD_NAMES

/*!
 * @brief Print a simple (number or string) protobuf value.
 *
 * @param [in] type The protobuf value type.
 * @param [in] pValue The value to print.
 */
static void
_prbPrintSimpleValue(LwU32 type, const PRB_VALUE *pValue)
{
    switch (type)
    {
        case PRB_DOUBLE:
            lwdPrintf("%e", pValue->double_);
            break;
        case PRB_FLOAT:
            lwdPrintf("%e", pValue->float_);
            break;
        case PRB_INT32:
        case PRB_UINT32:
        case PRB_SINT32:
        case PRB_FIXED32:
        case PRB_SFIXED32:
            lwdPrintf("0x%08X", pValue->uint32);
            break;
        case PRB_INT64:
        case PRB_UINT64:
        case PRB_SINT64:
        case PRB_FIXED64:
        case PRB_SFIXED64:
            lwdPrintf("0x%016"FORMAT64"X", pValue->uint64);
            break;
        case PRB_BOOL:
            lwdPrintf("%d", (int)pValue->bool_);
            break;
        case PRB_STRING:
            lwdPrintf("\"%s\"", pValue->string.str);
            break;
    }
}

/*!
 * @brief Print an enum protobuf value in string form if possible.
 *
 * @param [in] pEnum Descriptor for the enum.
 * @param [in] pValue The enum value to print.
 */
static void
_prbPrintEnum(const PRB_ENUM_DESC *pEnum, int value)
{
    const char *name = prbGetEnumValueName(pEnum, value);
    lwdPrintf("%s (%d)", name ? name : "???", value);
}

/*!
 * @brief Print a byte string protobuf value.
 *
 * @param [in] bytes The bytes to print.
 * @param [in] count The number of bytes.
 * @param [in] indentLevel The number of spaces to indent divided by 4.
 */
static void
_prbPrintBytes(const LwU8* bytes, LwU32 count, LwU32 indentLevel)
{
    LwU32 i = 0;

    _printIndent(indentLevel);
    lwdPrintf("{\n");
    ++indentLevel;

    while (i < count)
    {
        _printIndent(indentLevel);
        lwdPrintf("0x%08X:", i);
        while (i < count)
        {
            lwdPrintf(" 0x%02X", (LwU32)bytes[i]);
            if ((++i % 16) == 0)
            {
                break;
            }
        }
        lwdPrintf("\n");
    }

    --indentLevel;
    _printIndent(indentLevel);
    lwdPrintf("}\n");
}

/*!
 * @brief Get the type name of a field.
 *
 * @param [in] pField The field to get the type name of.
 *
 * @returns The type name of the field.
 */
static const char *
_prbGetFieldTypeName(const PRB_FIELD *pField)
{
    switch (pField->desc->opts.typ)
    {
#if PRB_ENUM_NAMES
        case PRB_ENUM:
            return pField->desc->enum_desc->name;
#endif
        case PRB_MESSAGE:
#if PRB_MESSAGE_NAMES
            return pField->desc->msg_desc->name;
#endif
        default:
            return PRB_TYPE_NAMES[pField->desc->opts.typ];
    }
}

/*!
 * @brief Print an outline of a field relwrsively to stdout in text format.
 *
 * @param [in] pField The field to print the outline of.
 * @param [in] indentLevel The number of spaces to print divided by 4.
 */
static void
_prbPrintFieldOutline(const PRB_FIELD *pField, LwU32 indentLevel)
{
    LwU32 i;
    const char *typeName = _prbGetFieldTypeName(pField);

    if (pField->count == 1)
    {
        _printIndent(indentLevel);
        lwdPrintf("%s %s\n", typeName, pField->desc->name);

        if (pField->desc->opts.typ == PRB_MESSAGE)
        {
            prbPrintMsgOutline((const PRB_MSG *)pField->values[0].message.data,
                               indentLevel);
        }
    }
    else if (pField->count > 1)
    {
        if (pField->desc->opts.typ == PRB_MESSAGE)
        {
            for (i = 0; i < pField->count; ++i)
            {
                _printIndent(indentLevel);
                lwdPrintf("%s %s[%u]\n", typeName, pField->desc->name, i);
                prbPrintMsgOutline((const PRB_MSG *)pField->values[i].message.data,
                                   indentLevel);
            }
        }
        else
        {
            _printIndent(indentLevel);
            lwdPrintf("%s %s[%u]\n", typeName, pField->desc->name, pField->count);

        }
    }
}
#endif
