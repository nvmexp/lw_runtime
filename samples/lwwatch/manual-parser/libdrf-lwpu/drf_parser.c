/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/* For isblank() */
#define _ISOC99_SOURCE

#include "ctype.h"

#include <drf_types.h>
#include "drf.h"
#include "drf_macro.h"
#include "drf_parser.h"

typedef enum {
    PARSER_STATE_START = 0,
    PARSER_STATE_IN_WHITESPACE,
    PARSER_STATE_IN_NUMBER,
    PARSER_STATE_IN_DECIMAL_NUMBER,
    PARSER_STATE_IN_HEX_NUMBER,
    PARSER_STATE_FINISH
} drf_parser_state;

void drf_parse_replacement(const char *repl, drf_macro_type *macro_type,
        uint32_t *a, uint32_t *b)
{
    drf_parser_state parser_state = PARSER_STATE_START;
    uint32_t x = 0, y;
    char c;

    *macro_type = MACRO_TYPE_CONSTANT;
    *a = *b = 0;

    while (1) {
        c = *repl++;
        if (c == '\0') {
            switch (parser_state) {
                case PARSER_STATE_START:
                    *macro_type = MACRO_TYPE_ZERO_LENGTH;
                default:
                    parser_state = PARSER_STATE_FINISH;
            }
        }
        switch (parser_state) {
            case PARSER_STATE_START:
                if (isblank(c)) {
                    parser_state = PARSER_STATE_IN_WHITESPACE;
                    break;
                }
            case PARSER_STATE_IN_WHITESPACE:
                if (isdigit(c)) {
                    x = (c - '0');
                    parser_state = PARSER_STATE_IN_NUMBER;
                } else if (!isblank(c)) {
                    *macro_type = MACRO_TYPE_UNKNOWN_OTHER;
                    return;
                }
                break;
            case PARSER_STATE_IN_NUMBER:
                if ((c == 'x') || (c == 'X')) {
                    parser_state = PARSER_STATE_IN_HEX_NUMBER;
                    break;
                } else
                    parser_state = PARSER_STATE_IN_DECIMAL_NUMBER;
            case PARSER_STATE_IN_DECIMAL_NUMBER:
                if (isdigit(c)) {
                    x = (x * 10 + (c - '0'));
                    break;
                } else if (c != ':') {
                    *macro_type = MACRO_TYPE_UNKNOWN_OTHER;
                    return;
                }
            case PARSER_STATE_IN_HEX_NUMBER:
                if (isxdigit(c)) {
                    y =  ((c >= 'a') ? (c - 'a' + 10) :
                            ((c >= 'A') ? (c - 'A' + 10) : (c - '0')));
                    x = ((x << 4) + y);
                    break;
                } else if ((c == ':') &&
                        (*macro_type == MACRO_TYPE_CONSTANT)) {
                    *macro_type = MACRO_TYPE_RANGE;
                    *a = x;
                    parser_state = PARSER_STATE_START;
                } else {
                    *macro_type = MACRO_TYPE_UNKNOWN_OTHER;
                    return;
                }
                break;
            case PARSER_STATE_FINISH:
                *b = x;
                return;
        }
    }
}
