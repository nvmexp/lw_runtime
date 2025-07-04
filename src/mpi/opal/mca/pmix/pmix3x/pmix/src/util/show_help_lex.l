%option nounput
%option noinput

%{ /* -*- C -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2006 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2016      Intel, Inc. All rights reserved
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <src/include/pmix_config.h>

#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "src/util/show_help_lex.h"

BEGIN_C_DECLS

/*
 * public functions
 */
extern int pmix_show_help_finish_parsing(void);

/*
 * local functions
 */
static int pmix_show_help_yywrap(void);

END_C_DECLS

/*
 * global variables
 */
int pmix_show_help_yynewlines = 1;
bool pmix_show_help_parse_done = false;

%}

WHITE       [\f\t\v ]
CHAR        [A-Za-z0-9_\-\.]

%x CHOMP

%%

#.*\n               ; /* comment line */

^\[.+\]/[^\]\n]*\n { BEGIN(CHOMP); return PMIX_SHOW_HELP_PARSE_TOPIC; }

<CHOMP>.*\n { BEGIN(INITIAL); }

.*/\n { BEGIN(CHOMP); return PMIX_SHOW_HELP_PARSE_MESSAGE; }

%%

/* Old flex (2.5.4a? and older) does not define a destroy function */
#if !defined(YY_FLEX_SUBMINOR_VERSION)
#define YY_FLEX_SUBMINOR_VERSION 0
#endif

#if (YY_FLEX_MAJOR_VERSION < 2) || (YY_FLEX_MAJOR_VERSION == 2 && (YY_FLEX_MINOR_VERSION < 5 || (YY_FLEX_MINOR_VERSION == 5 && YY_FLEX_SUBMINOR_VERSION < 5)))
int pmix_show_help_yylex_destroy(void)
{
    if (NULL != YY_LWRRENT_BUFFER) {
        yy_delete_buffer(YY_LWRRENT_BUFFER);
#if defined(YY_LWRRENT_BUFFER_LVALUE)
        YY_LWRRENT_BUFFER_LVALUE = NULL;
#else
        YY_LWRRENT_BUFFER = NULL;
#endif  /* YY_LWRRENT_BUFFER_LVALUE */
    }
    return YY_NULL;
}
#endif

static int pmix_show_help_yywrap(void)
{
    pmix_show_help_parse_done = true;
    return 1;
}


/*
 * Ensure that we have a valid yybuffer to use.  Specifically, if this
 * scanner is ilwoked a second time, finish_parsing() (above) will
 * have been exelwted, and the current buffer will have been freed.
 * Flex doesn't recognize this fact because as far as it's concerned,
 * its internal state was already initialized, so it thinks it should
 * have a valid buffer.  Hence, here we ensure to give it a valid
 * buffer.
 */
int pmix_show_help_init_buffer(FILE *file)
{
    YY_BUFFER_STATE buf = yy_create_buffer(file, YY_BUF_SIZE);
    yy_switch_to_buffer(buf);

    return 0;
}
