/*
 * Copyright (C) 2004 LWPU Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __MSG_H__
#define __MSG_H__

#include <stdarg.h>
#include <stdio.h>


/*
 * Define a printf format attribute macro.  This definition is based on the one
 * from Xfuncproto.h, available in the 'xproto' package at
 * http://xorg.freedesktop.org/releases/individual/proto/
 */

#if defined(__GNUC__) && ((__GNUC__ * 100 + __GNUC_MINOR__) >= 203)
# define LW_ATTRIBUTE_PRINTF(x,y) __attribute__((__format__(__printf__,x,y)))
#else /* not gcc >= 2.3 */
# define LW_ATTRIBUTE_PRINTF(x,y)
#endif


/*
 * LW_VSNPRINTF(): macro that assigns buf using vsnprintf().  This is
 * correct for differing semantics of the vsnprintf() return value:
 *
 * -1 when the buffer is not long enough (glibc < 2.1)
 *
 *   or
 *
 * the length the string would have been if the buffer had been large
 * enough (glibc >= 2.1)
 *
 * This macro allocates memory for buf; the caller should free it when
 * done.
 */

#define LW_FMT_BUF_LEN 256

#define LW_VSNPRINTF(buf, fmt)                                  \
do {                                                            \
    if (!fmt) {                                                 \
        (buf) = NULL;                                           \
    } else {                                                    \
        va_list ap;                                             \
        int len, lwrrent_len = LW_FMT_BUF_LEN;                  \
                                                                \
        while (1) {                                             \
            (buf) = lwalloc(lwrrent_len);                       \
                                                                \
            va_start(ap, fmt);                                  \
            len = vsnprintf((buf), lwrrent_len, (fmt), ap);     \
            va_end(ap);                                         \
                                                                \
            if ((len > -1) && (len < lwrrent_len)) {            \
                break;                                          \
            } else if (len > -1) {                              \
                lwrrent_len = len + 1;                          \
            } else {                                            \
                lwrrent_len += LW_FMT_BUF_LEN;                  \
            }                                                   \
                                                                \
            lwfree(buf);                                        \
        }                                                       \
    }                                                           \
} while (0)


/*
 * verbosity, controls output of errors, warnings and other
 * information.
 */

typedef enum {
    LW_VERBOSITY_NONE = 0,                    /* no errors, warnings or info */
    LW_VERBOSITY_ERROR,                       /* errors only */
    LW_VERBOSITY_DEPRECATED,                  /* errors and deprecation messages */
    LW_VERBOSITY_WARNING,                     /* errors and all warnings */
    LW_VERBOSITY_ALL,                         /* errors, all warnings and other info */
    LW_VERBOSITY_DEFAULT = LW_VERBOSITY_ALL
} LwVerbosity;

LwVerbosity lw_get_verbosity(void);
void        lw_set_verbosity(LwVerbosity level);


/*
 * Formatted I/O functions
 */

void reset_lwrrent_terminal_width(unsigned short new_val);

void lw_error_msg(const char *fmt, ...)                LW_ATTRIBUTE_PRINTF(1, 2);
void lw_deprecated_msg(const char *fmt, ...)           LW_ATTRIBUTE_PRINTF(1, 2);
void lw_warning_msg(const char *fmt, ...)              LW_ATTRIBUTE_PRINTF(1, 2);
void lw_info_msg(const char *prefix,
                 const char *fmt, ...)                 LW_ATTRIBUTE_PRINTF(2, 3);
void lw_info_msg_to_file(FILE *stream,
                         const char *prefix,
                         const char *fmt, ...)         LW_ATTRIBUTE_PRINTF(3, 4);
void lw_msg(const char *prefix, const char *fmt, ...)  LW_ATTRIBUTE_PRINTF(2, 3);
void lw_msg_preserve_whitespace(const char *prefix,
                                const char *fmt, ...)  LW_ATTRIBUTE_PRINTF(2, 3);


/*
 * TextRows structure and helper functions
 */

typedef struct {
    char **t; /* the text rows */
    int n;    /* number of rows */
    int m;    /* maximum row length */
} TextRows;

TextRows *lw_format_text_rows(const char *prefix, const char *str, int width,
                              int word_boundary);
void lw_text_rows_append(TextRows *t, const char *msg);
void lw_concat_text_rows(TextRows *t0, TextRows *t1);
void lw_free_text_rows(TextRows *t);


#endif /* __MSG_H__ */
