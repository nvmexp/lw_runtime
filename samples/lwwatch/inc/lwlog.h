/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2013 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef _LWLOG_H_
#define _LWLOG_H_


#ifdef __cplusplus
// Define all functions as cdecl, so they can be used by non-windbg LwWatch
extern "C" {
#endif

#include "lwmisc.h"

/*********************** LWLOG DECODER LIBRARY *****************************\
*                                                                           *
* Module: LWWATCH/LWLOG.H                                                   *
*                                                                           *
\***************************************************************************/

#if defined(WIN32) && !defined(USERMODE)
#define LWLOG_CMD_ENABLED 1
#else
#define LWLOG_CMD_ENABLED 0
#endif

#define LWLOG_HELP                                                                                                                    \
    "lw: !lw.lwlog commands:\n"                                                                                                       \
    "lw:    -load    <database>                  - Loads the decoder database from the <database> file.\n"                            \
    "lw:    -unload                              - Unloads all LwLog data and the decoder database.\n"                                \
    "lw:    -list                                - Lists the basic info of all LwLog buffers.\n"                                      \
    "lw:    -list    print                       - Lists only the print LwLog buffers.\n"                                             \
    "lw:    -list    <bufferNum>                 - Show info only about the <bufferNum> buffer.\n"                                    \
    "lw:    -dump    all -f <outfile>            - Dumps all LwLog data and all buffers. Save the dump to <outfile>.\n"               \
    "lw:    -dump    -b <bufferNum> -f <outfile> - Dumps <bufferNum> LwLog buffer to <outfile>.\n"                                    \
    "lw:    -decode  <numEntries>                - Decode the last <numEntries> of a buffer. \n"                                      \
    "lw:                                           If <numEntries> is 0 or 'all' or omitted, entire buffer will be decoded.\n"        \
    "lw:             -b <bufferNum>              - Decode the <bufferNum> LwLog buffer. This parameter cannot be omitted.\n"          \
    "lw:             -f <outfile>                - Save the decoded string to this file. If omitted, it will be printed on screen.\n" \
    "lw:             -html                       - Decode the buffer as HTML instead of plaintext.\n"                                 \
    "lw:    -d                                   - Alias for -decode.\n"

#define LWLOG_FILTER_HELP           \
   "<rules>           ::= <rule> | <rule> \",\" <rules>\n"                                                                                                                                                                      \
   "<rule>            ::= <domain> <flags> | <route_domain> \">\" <buffer_num>\n"                                                                                                                                               \
   "<domain>          ::= \"\" | <filename> <opt_line_range>\n"                                                                                                                                                                 \
   "<opt_line_range>  ::= \"\" | <line_start> | <line_start> <line_end>\n"                                                                                                                                                      \
   "<line_start>      ::= \":\" <int>\n"                                                                                                                                                                                        \
   "<line_end>        ::= \"-\" <int>\n"                                                                                                                                                                                        \
   "<flags>           ::= <flag> | <flag> \" \" <flags>\n"                                                                                                                                                                      \
   "<flag>            ::= <action> <target>\n"                                                                                                                                                                                  \
   "<action>          ::= \"\" | \"+\" | \"-\" | \"++\" | \"--\"\n"                                                                                                                                                             \
   "<target>          ::= <level> | <module> \n"                                                                                                                                                                                \
   "<level>           ::= \"ALL\" | \"REGTRACE\" | \"INFO\" | \"NOTICE\" | \"WARNINGS\" | \"ERRORS\" | \"HW_ERROR\" | \"FATAL\"\n"                                                                                              \
   "<module>          ::= \"mod\" <int> | <rm_module>\n"                                                                                                                                                                        \
   "<rm_module>       ::= \"GLOBAL\" | \"ARCH\" | \"OS\" | \"DISP\" | \"FIFO\" | \"GR\" | \"HEAP\" | \"VIDEO\" | \"MP\" | \"POWER\" | \"PERFCTL\" | \"CAP\" | \"FB\" | \"THERMAL\" | \"PMU\" | \"SPB\" | \"PMGR\" | \"SEC2\"\n" \
   "<route_domain>    ::= <filename> | <level> | <module> | \"ALL\"\n"                                                                                                                                                          \
   "<buffer_num>      ::= <int>\n"


#define LWLOG_DATABASE_PATH     "//lwsym/lwlog/LWRM.lwlog"

//
// Initialize the LwLog addresses.
// Also loads the database from LwSym.
//
void lwlogInit();
//
// Reload a given buffer from the target machine.
// If the given buffer has been freed on the target machine, this function
// will free the local copy as well.
//
void lwlogReloadBuffer(LwU32 bufferIndex);
//
// Reload a given buffer header from the target machine.
// If the given buffer has been freed on the target machine, this function
// will free the local copy as well.
//
void lwlogReloadBufferHeader(LwU32 bufferIndex);
//
// Clean up everything that was allocated.
//
void lwlogFreeAll();
//
// Load the LwLog decoder database from the given file
// If dbFile is NULL, database will be loaded from LwSym
//
void lwlogLoadDatabase(const char *dbfile);
//
// Print the basic info about a given buffer.
// Prints the index, tag, type, size, position and flags.
// prefix is an optional string to prefix the given buffer info.
//
void lwlogPrintBufferInfo(LwU32 bufferIndex, const char *prefix);
//
// Lists all buffers with their basic info
//
void lwlogPrintAllBuffersInfo();
//
// Lists all PRINT buffers with their basic info
//
void lwlogPrintAllPrintBuffersInfo();
//
// Decodes the given PRINT buffer, storing the result in pDest.
// printBufferIndex - Index in the print logger (0-7), not main buffer index.
// bHtml - If true, the buffer will be decoded as HTML instead of plaintext
// nLast - Only the last nLast entries will be decoded. (nLast == 0) means all.
//
void lwlogDecodePrintBuffer(LwU32 printBufferIndex, char *pDest, LwBool bHtml, LwU32 nLast);
//
// Dumps the specified buffer to a given file.
// The file must already be open for writing.
// This function doesn't close or flush the file.
//
void lwlogDumpBuffer(LwU32 bufferIndex, FILE *file);
//
// Dumps the loggers, as well as all buffers to a given file.
// The file must already be open for writing.
// This function doesn't close the file. It does flush it.
//
void lwlogDumpAll(FILE *file);
//
// Dump the header required for decoding a buffer
//
void lwlogDumpHeader(LwU32 bufferIndex, FILE *f);

//
//  Filtering code
//

//
// Apply the given filter string
//
void lwlogApplyFilter(const char *filter);
//
// Show all active filters
//
void lwlogListFilters();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _LWLOG_H_
