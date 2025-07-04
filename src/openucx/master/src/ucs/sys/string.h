/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_STRING_H_
#define UCS_STRING_H_

#include "compiler_def.h"
#include <ucs/type/status.h>
#include <ucs/sys/math.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>

BEGIN_C_DECLS

/** @file string.h */

/* value which specifies "infinity" for a numeric variable */
#define UCS_NUMERIC_INF_STR "inf"

/* value which specifies "auto" for a variable */
#define UCS_VALUE_AUTO_STR "auto"

/* the numeric value of "infinity" */
#define UCS_MEMUNITS_INF    ((size_t)-1)
#define UCS_ULUNITS_INF     ((unsigned long)-1)

/* value which specifies "auto" for a numeric variable */
#define UCS_MEMUNITS_AUTO   ((size_t)-2)
#define UCS_ULUNITS_AUTO    ((unsigned long)-2)
#define UCS_HEXUNITS_AUTO   ((uint16_t)-2)

#define UCS_BANDWIDTH_AUTO  (-1.0)

/**
 * Expand a partial path to full path.
 *
 * @param path       Path to expand.
 * @param fullpath   Filled with full path.
 * @param max        Room in "fullpath"
 */
void ucs_expand_path(const char *path, char *fullpath, size_t max);


/**
 * Fill a filename template. The following values in the string are replaced:
 *  %p - replaced by process id
 *  %h - replaced by host name
 *
 * @param tmpl   File name template (possibly containing formatting sequences)
 * @param buf    Filled with resulting file name
 * @param max    Maximal size of destination buffer.
 */
void ucs_fill_filename_template(const char *tmpl, char *buf, size_t max);


/**
 * Format a string to a buffer of given size, and fill the rest of the buffer
 * with '\0'. Also, guarantee that the last char in the buffer is '\0'.
 *
 * @param buf  Buffer to format the string to.
 * @param size Buffer size.
 * @param fmt  Format string.
 */
void ucs_snprintf_zero(char *buf, size_t size, const char *fmt, ...)
    UCS_F_PRINTF(3, 4);


/**
 * Same as strncpy(), but guarantee that the last char in the buffer is '\0'.
 */
void ucs_strncpy_zero(char *dest, const char *src, size_t max);


/**
 * Return a number filled with the first characters of the string.
 */
uint64_t ucs_string_to_id(const char *str);


/**
 * Colwert a memory units value to a string which is abbreviated if possible.
 * For example:
 *  1024 -> 1kb
 *
 * @param value  Value to colwert.
 * @param buf    Buffer to place the string.
 * @param max    Maximal length of the buffer.
 *
 * @return Pointer to 'buf', which holds the resulting string.
 */
char *ucs_memunits_to_str(size_t value, char *buf, size_t max);


/**
 * Colwert a string holding memory units to a numeric value.
 *
 *  @param buf   String to colwert
 *  @param dest  Numeric value of the string
 *
 *  @return UCS_OK if successful, or error code otherwise.
 */
ucs_status_t ucs_str_to_memunits(const char *buf, void *dest);


/**
 *  Return the numeric value of the memunits prefix.
 *  For example:
 *  'M' -> 1048576
 */
size_t ucs_string_quantity_prefix_value(char prefix);


/**
 * Format a string to a buffer of given size, and guarantee that the last char
 * in the buffer is '\0'.
 *
 * @param buf  Buffer to format the string to.
 * @param size Buffer size.
 * @param fmt  Format string.
 */
void ucs_snprintf_safe(char *buf, size_t size, const char *fmt, ...)
    UCS_F_PRINTF(3, 4);


/**
 * Copy string limited by len bytes. Destination string is always ended by '\0'
 *
 * @param dst Destination buffer
 * @param src Source string
 * @param len Maximum string length to copy
 *
 * @return address of destination buffer
 */
char* ucs_strncpy_safe(char *dst, const char *src, size_t len);


/**
 * Remove whitespace characters in the beginning and end of the string, as
 * detected by isspace(3). Returns a pointer to the new string (which may be a
 * substring of 'str'). The original string 'str' may be modified in-place.
 *
 * @param str  String to remove whitespaces from.
 * @return Pointer to the new string, with leading/trailing whitespaces removed.
 */
char *ucs_strtrim(char *str);


/**
 * Get pointer to file name in path, same as basename but do not
 * modify source string.
 *
 * @param path Path to parse.
 * 
 * @return file name
 */
static UCS_F_ALWAYS_INLINE const char* ucs_basename(const char *path)
{
    const char *name = strrchr(path, '/');

    return (name == NULL) ? path : name + 1;
}


/**
 * Dump binary array into string in hex format. Destination string is
 * always ended by '\0'.
 *
 * @param data     Source array to dump.
 * @param length   Length of source array in bytes.
 * @param buf      Destination string.
 * @param max      Max length of destination string including terminating
 *                 '\0' byte.
 * @param per_line Number of bytes in source array to print per line
 *                 or SIZE_MAX for single line.
 * 
 * @return address of destination buffer
 */
const char *ucs_str_dump_hex(const void* data, size_t length, char *buf,
                             size_t max, size_t per_line);


/**
 * Colwert the given flags to a string that represents them.
 *
 * @param  str            String to hold the flags string values.
 * @param  max            Size of the string.
 * @param  flags          Flags to be colwerted.
 * @param  str_table      Colwersion table - from flag value to a string.
 *
 * @return String that holds the representation of the given flags.
 */
const char* ucs_flags_str(char *str, size_t max,
                          uint64_t flags, const char **str_table);

END_C_DECLS

#endif
