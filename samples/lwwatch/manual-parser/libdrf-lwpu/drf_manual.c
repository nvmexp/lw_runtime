/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#include <sys/types.h>
#include <regex.h>
#if LWOS_IS_UNIX
#include <libgen.h> // for basename(3)
#endif
#include <mcpp_lib.h>
#include "uthash.h"

#define HASH_ADD_STRPTR(head, str, el) \
    HASH_ADD_KEYPTR(hh, head, str, strlen(str), el)

#define elementsof(x) \
    (sizeof(x) / sizeof((x)[0]))

#if !defined(min)
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#include <drf_types.h>
#include "drf.h"
#include "drf_state.h"
#include "drf_mcpp.h"
#include "drf_macro.h"
#include "drf_parser.h"
#include "drf_register.h"
#include "drf_field.h"
#include "drf_define.h"

static __drf_state_t *state;

#define DRF_MANUAL_SET_GLOBAL_STATE(__state) (state = (__state))
#define DRF_MANUAL_RESET_GLOBAL_STATE() (state = NULL)

#define WORK_BUFFER_SIZE 1024
static char work_buffer[WORK_BUFFER_SIZE];

#define MCPP_BUFFER_SIZE_INCREMENT (16 * WORK_BUFFER_SIZE)
static uint32_t mcpp_buffer_size;

static char *mcpp_buffer;
static uint32_t mcpp_buffer_index;

enum {
    PENDING_MACROS = 0,
    QUEUED_MACROS,
    INTRACTABLE_MACROS,
    EXPANDED_MACROS,
    REGISTERS,
    REGISTER_FIELDS,
    REGISTER_FIELD_VALUES,
    NUM_HASH_TABLES
};

static drf_macro_t *hash_tables[NUM_HASH_TABLES];

#ifdef DEBUG
/* By default, print 5 warnings before supressing. */
#define DEFAULT_MAX_WARNINGS 5
#else
#define DEFAULT_MAX_WARNINGS 0
#endif

static int should_suppress_warning(void)
{
    /* Read 'num_warnings' from an environment variable and cache its
     * value so we don't have to repeatedly parse it. */
    static int num_warnings = -1;
    static int max_warnings = DEFAULT_MAX_WARNINGS;

    if (max_warnings < 0) {
        return 0;
    }

    if (num_warnings == -1) {
        char *opt = getelw("LW_DRF_MANUAL_PARSER_MAX_WARNINGS");
        if (opt != NULL) {
            max_warnings = atoi(opt);
        }
        num_warnings = max_warnings;
        if (num_warnings > 0) {
            num_warnings++;
        }
    }

    if (num_warnings == 0) {
        return 1;
    }
    num_warnings--;
    if (num_warnings == 0) {
        fprintf(stderr, "libdrf-lwpu: reached limit of %d warnings; "
                "suppressing further messages.\nThe environment variable "
                "LW_DRF_MANUAL_PARSER_MAX_WARNINGS may be used to adjust "
                "the limit (negative means no limit).\n",
                max_warnings);
                return 1;
    }
    return 0;
}

static void emit_warning(const char *fmt, ...)
{
    va_list args;

    if (should_suppress_warning()) {
        return;
    }

    fprintf(stderr, "libdrf-lwpu: warning: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/*
 * Read the "tags" comment at the end of the manual definition, and pull out
 * the 'entity type' character.
 *
 * See:
 *  //hw/lwgpu/manuals/src/key_legend.ref
 * or
 *  https://wiki.lwpu.com/gpuhwdept/index.php/Manuals_tools#Manual_Entities
 * for a details about the tags formatting and meaning of each type.
 */
static char get_manuals_entity(const char *comment,
                               const char *fname)
{
    /* Comment should be of the form below (E is the entity): */
       /* ----E */
    /* 0123456789a */

    /* Verify the structure of the tag comment is as expected */
    if (comment[0] != '/' ||
        comment[1] != '*' ||
        comment[2] != ' ' ||
        comment[8] != ' ' ||
        comment[9] != '*' ||
        comment[10] != '/' ||
        comment[11] != '\0') {
        emit_warning("Unexpected comment '%s'.\n", comment);
        return '\0';
    }
    return comment[7];
}

static void macro_defined_callback(const char *name, short nargs,
        const char *parmnames, const char *repl,
        const char *fname, long mline)
{
    drf_macro_t **hash_table;
    drf_macro_t *macro;
    drf_macro_type macro_type;
    drf_device_t *device;
    uint32_t a, b;

    device = state->lwrrent_device;
    if (strstr(name, device->name) != name)
        return;
    drf_parse_replacement(repl, &macro_type, &a, &b);
    switch (macro_type) {
        case MACRO_TYPE_CONSTANT:
            if (nargs > 0) {
                drf_set_errno(EILWAL);
                return;
            }
            break;
        case MACRO_TYPE_ZERO_LENGTH:
            return;
        default:
            break;
    }

    /* If this is a "layout" definition, it's not a register.  Skip it. */
    if (get_manuals_entity(mcpp_get_last_comment(), fname) == 'L') {
        return;
    }

    macro = calloc(1, (sizeof(*macro) + strlen(name)));
    if (!macro) {
        if (state->debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        return;
    }
    macro->macro_type = macro_type;
    macro->a = a;
    macro->b = b;
    macro->n_args = (uint8_t)nargs;
    strcpy((char *)&macro->name[0], name);
    switch (macro_type) {
        case MACRO_TYPE_CONSTANT:
        case MACRO_TYPE_RANGE:
            hash_table = &hash_tables[EXPANDED_MACROS];
            break;
        default:
            hash_table = &hash_tables[PENDING_MACROS];
            break;
    }
    HASH_ADD_STRPTR(*hash_table, macro->name, macro);
}

static void macro_undefined_callback(const char *name, const char *fname,
        long mline)
{
    drf_macro_t *macro;

    HASH_FIND_STR(hash_tables[EXPANDED_MACROS], name, macro);
    if (macro) {
        HASH_DEL(hash_tables[EXPANDED_MACROS], macro);
        free(macro);
    }
    HASH_FIND_STR(hash_tables[PENDING_MACROS], name, macro);
    if (macro) {
        HASH_DEL(hash_tables[PENDING_MACROS], macro);
        free(macro);
    }
}

static CALLBACKS macro_callbacks = {
    macro_defined_callback, 
    macro_undefined_callback, 
    NULL, 
    NULL
};

static void pragma_eval_callback(const char *expr,
        int valid, unsigned long long val, const char *fname, long mline)
{
    drf_macro_type macro_type;
    drf_macro_t *macro;
    const char *name;

    macro = NULL;
    if (valid) {
        switch (expr[1]) {
            case '?':
                macro_type = MACRO_TYPE_RANGE;
                name = (expr + 2);
                break;
            default:
                macro_type = MACRO_TYPE_CONSTANT;
                name = expr;
                break;
        }
        HASH_FIND_STR(hash_tables[QUEUED_MACROS], name, macro);
    }
    if (macro) {
        switch (expr[0]) {
            case '1':
                macro->a = (uint32_t)val;
                break;
            default:
                macro->macro_type = macro_type;
                macro->b = (uint32_t)val;
                if (macro->macro_type != MACRO_TYPE_CONSTANT)
                    return;
                break;
        }
        HASH_DEL(hash_tables[QUEUED_MACROS], macro);
        HASH_ADD_STRPTR(hash_tables[EXPANDED_MACROS], macro->name, macro);
    }
}

static CALLBACKS pragma_callbacks = {
    NULL, 
    NULL, 
    NULL, 
    pragma_eval_callback
};

static void check_mcpp_buffer_space(void)
{
    uint32_t new_size = 0;

    if ((mcpp_buffer_size == 0) ||
            (mcpp_buffer_index >= (mcpp_buffer_size - WORK_BUFFER_SIZE))) {
        new_size = (mcpp_buffer_size + MCPP_BUFFER_SIZE_INCREMENT);
        mcpp_buffer = realloc(mcpp_buffer, new_size);
        mcpp_buffer_size = new_size;
    }
}

static void queue_mcpp_suppression(const char *arg, int set)
{
    const char *pragma = {
        "#pragma MCPP %ssuppress %s\n"
    };
    int written;

    check_mcpp_buffer_space();
    written = sprintf((mcpp_buffer + mcpp_buffer_index),
            pragma, (set ? "" : "end_"), arg);
    mcpp_buffer_index += written;
}

static void queue_mcpp_evaluations(const char *expr)
{
    drf_macro_t *macro;
    uint32_t i;
    const char *pragmas[] = {
        "#pragma MCPP eval %s\n", "#pragma MCPP eval 0?%s\n",
        "#pragma MCPP eval 1?%s\n"
    };
    int written;

    check_mcpp_buffer_space();
    macro = calloc(1, (sizeof(*macro) + strlen(expr)));
    if (!macro) {
        if (state->debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        return;
    }
    strcpy((char *)&macro->name[0], expr);
    macro->flags = 0;
    for (i = 0; i < elementsof(pragmas); i++) {
        written = sprintf((mcpp_buffer + mcpp_buffer_index),
                pragmas[i], expr);
        mcpp_buffer_index += written;
    }
    HASH_ADD_STRPTR(hash_tables[QUEUED_MACROS], macro->name, macro);
}

static void evaluate_n_args_macro(char *expr, char *s,
        uint32_t *args, uint32_t n_args)
{
    uint32_t i;
    int written;

    for (i = 0; i < args[0]; i++) {
        written = sprintf(s, "%u", i);
        if (n_args > 1) {
            written += sprintf((s + written), ",");
            evaluate_n_args_macro(expr, (s + written), &args[1],
                    (n_args - 1));
        } else {
            sprintf((s + written), ")");
            queue_mcpp_evaluations(expr);
        }
    }
}

static int evaluate_macro(drf_macro_t *macro)
{
    drf_macro_t *tmp;
    uint32_t i, j, args[DRF_MACRO_MAX_N_ARGS];
    char *name, *expr;
    const char *formats[] = {
        "%s__SIZE_%u", "%s___SIZE_%u"
    };

    if (macro->n_args == 0) {
        queue_mcpp_evaluations(macro->name);
        return 0;
    }
    name = work_buffer;
    for (i = 0; i < macro->n_args; i++) {
        for (j = 0; j < elementsof(formats); j++) {
            sprintf(name, formats[j], macro->name, (i + 1));
            HASH_FIND_STR(hash_tables[EXPANDED_MACROS], name, tmp);
            if (tmp)
                break;
        }
        if (!tmp)
            return -ENXIO;
        else if ((tmp->n_args > 0) ||
                (tmp->macro_type != MACRO_TYPE_CONSTANT)) {
            return -EILWAL;
        }
        HASH_DEL(hash_tables[EXPANDED_MACROS], tmp);
        args[i] = min(tmp->b, DRF_MACRO_MAX_N_ARGS_DIMENSION_SIZE);
        free(tmp);
    }
    expr = work_buffer;
    sprintf(expr, "%s(", macro->name);
    evaluate_n_args_macro(expr, (expr + strlen(expr)), args,
            macro->n_args);
    return 0;
}

static void process_pending_macros(void)
{
    drf_macro_t *macro;
    int ret;

    queue_mcpp_suppression("errors", 1);
    macro = hash_tables[PENDING_MACROS];
    while (macro) {
        HASH_DEL(hash_tables[PENDING_MACROS], macro);
        ret = evaluate_macro(macro);
        if (ret < 0) {
            HASH_ADD_STRPTR(hash_tables[INTRACTABLE_MACROS],
                    macro->name, macro);
        } else
            free(macro);
        macro = hash_tables[PENDING_MACROS];
    }
    queue_mcpp_suppression("errors", 0);
}

static void process_intractable_macro(drf_macro_t *macro)
{
    drf_macro_t *tmp, *iterator;

    HASH_ITER(hh, hash_tables[EXPANDED_MACROS], tmp, iterator) {
        if (strstr(tmp->name, macro->name) != tmp->name)
            continue;
        HASH_DEL(hash_tables[EXPANDED_MACROS], tmp);
        free(tmp);
    }
}

static void process_intractable_macros(void)
{
    drf_macro_t *macro;

    macro = hash_tables[INTRACTABLE_MACROS];
    while (macro) {
        HASH_DEL(hash_tables[INTRACTABLE_MACROS], macro);
        process_intractable_macro(macro);
        free(macro);
        macro = hash_tables[INTRACTABLE_MACROS];
    }
}

static int compare_registers_by_address(const void* _a, const void* _b)
{
    const drf_register_t* register_a = *(const drf_register_t**)_a;
    const drf_register_t* register_b = *(const drf_register_t**)_b;

    if (register_a->address == register_b->address)
        return 0;

    if (register_a->address < register_b->address)
        return -1;

    return 1;
}

static int compare_defines_by_value(const void* _a, const void* _b)
{
    const drf_define_t* define_a = *(const drf_define_t**)_a;
    const drf_define_t* define_b = *(const drf_define_t**)_b;

    if (define_a->value == define_b->value)
        return 0;

    if (define_a->value < define_b->value)
        return -1;

    return 1;
}

static int compare_fields_by_lsb(const void* _a, const void* _b)
{
    const drf_field_t* field_a = *(const drf_field_t**)_a;
    const drf_field_t* field_b = *(const drf_field_t**)_b;

    if (field_a->lsb == field_b->lsb)
        return 0;

    if (field_a->lsb < field_b->lsb)
        return -1;

    return 1;
}

static int sort_by_name(drf_macro_t *a, drf_macro_t *b)
{
    return strcmp(a->name, b->name);
}

static void sort_macros(void)
{
    HASH_SORT(hash_tables[EXPANDED_MACROS], sort_by_name);
}

static void process_expanded_macro(drf_device_t *device, drf_macro_t *macro)
{
    drf_macro_t **hash_table;
    drf_macro_t *tmp_1, *tmp_2, *tmp_3;
    uint32_t i, length;
    char *str;

    switch (macro->macro_type) {
        case MACRO_TYPE_CONSTANT:
            hash_table = &hash_tables[REGISTER_FIELDS];
            break;
        case MACRO_TYPE_RANGE:
            hash_table = &hash_tables[REGISTERS];
            break;
        default:
            return;
    }
    length = strlen(device->name);
    tmp_1 = NULL;
    for (i = strlen(macro->name); i > length; i--) {
        if (macro->name[i] != '_')
            continue;
        HASH_FIND(hh, *hash_table, macro->name, i, tmp_1);
        if (tmp_1)
            break;
    }
    switch (macro->macro_type) {
        case MACRO_TYPE_CONSTANT:
            HASH_DEL(hash_tables[EXPANDED_MACROS], macro);
            tmp_3 = NULL;
            while (tmp_1) {
                if (strstr(macro->name, tmp_1->name)) {
                    if (!tmp_3 || (strlen(tmp_3->name) < strlen(tmp_1->name)))
                        tmp_3 = tmp_1;
                }
                tmp_1 = tmp_1->next;
            }
            if (tmp_3) {
                HASH_FIND_STR(hash_tables[REGISTER_FIELD_VALUES],
                        tmp_3->name, tmp_2);
                if (tmp_2) {
                    macro->next = tmp_2->next;
                    tmp_2->next = macro;
                } else {
                    HASH_ADD_STRPTR(hash_tables[REGISTER_FIELD_VALUES],
                            tmp_3->name, macro);
                }
                break;
            }
            str = strchr(macro->name, '(');
            if (str) {
                i = (str - macro->name);
                HASH_FIND(hh, hash_tables[REGISTERS], macro->name,
                        i, tmp_2);
                if (!tmp_2) {
                    tmp_2 = malloc(sizeof(*tmp_2) + i);
                    if (!tmp_2) {
                        if (state->debug) {
                            fprintf(stderr, "malloc() failed (%s)!\n",
                                    strerror(errno));
                        }
                        drf_set_errno(errno);
                    } else {
                        memcpy(tmp_2, macro, sizeof(*tmp_2) + i);
                        tmp_2->name[i] = '\0';
                        tmp_2->flags |= DRF_MACRO_FLAGS_ALIAS;
                        HASH_ADD_KEYPTR(hh, hash_tables[REGISTERS],
                                tmp_2->name, i, tmp_2);
                    }
                }
            }
            HASH_ADD_STRPTR(hash_tables[REGISTERS], macro->name,
                    macro);
            break;
        case MACRO_TYPE_RANGE:
            if (!tmp_1)
                break;
            HASH_DEL(hash_tables[EXPANDED_MACROS], macro);
            HASH_FIND_STR(hash_tables[REGISTER_FIELDS], tmp_1->name,
                    tmp_2);
            if (tmp_2) {
                macro->next = tmp_2->next;
                tmp_2->next = macro;
            } else {
                HASH_ADD_STRPTR(hash_tables[REGISTER_FIELDS],
                        tmp_1->name, macro);
                str = strchr(macro->name, '(');
                if (str) {
                    i = (str - macro->name);
                    tmp_3 = malloc(sizeof(*tmp_3) + i);
                    if (!tmp_3) {
                        if (state->debug) {
                            fprintf(stderr, "malloc() failed (%s)!\n",
                                    strerror(errno));
                        }
                        drf_set_errno(errno);
                    } else {
                        memcpy(tmp_3, macro, sizeof(*tmp_3) + i);
                        tmp_3->name[i] = '\0';
                        tmp_3->flags |= DRF_MACRO_FLAGS_ALIAS;
                        tmp_3->next = macro->next;
                        macro->next = tmp_3;
                    }
                }
            }
            break;
        default:
            break;
    }
}

static void process_expanded_macros(drf_device_t *device)
{
    drf_macro_t *macro, *iterator;

    HASH_ITER(hh, hash_tables[EXPANDED_MACROS], macro, iterator)
        process_expanded_macro(device, macro);
}

static drf_define_t *drf_manual_alloc_define(drf_macro_t *macro)
{
    drf_define_t *define;

    define = calloc(1, (sizeof(*define) + strlen(macro->name)));
    if (!define) {
        if (state->debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        return NULL;
    }
    strcpy((char *)&define->name[0], macro->name);
    define->value = macro->b;
    return define;
}

static int drf_manual_strcmp(const char *s1, const char *s2, uint32_t *i)
{
    unsigned char c1, c2;
    int ret = 0;

    *i = 0;
    do {
        c1 = *s1++;
        c2 = *s2++;
        if (c1 != c2) {
            ret = ((c1 < c2) ? -1 : 1);
            break;
        }
        (*i)++;
    } while (c1 != '\0');
    return ret;
}

static drf_field_t *drf_manual_alloc_field(drf_macro_t *macro)
{
    char *str;
    drf_macro_t *tmp, *iterator;
    drf_field_t *field;
    drf_define_t *define, **defines;
    uint32_t i, j = 0, k;
    int ret;

    field = calloc(1, (sizeof(*field) + strlen(macro->name)));
    if (!field) {
        if (state->debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        return NULL;
    }
    str = strchr(macro->name, '(');
    i = (str ? (str - macro->name) : strlen(macro->name));
    HASH_FIND(hh, hash_tables[REGISTER_FIELD_VALUES], macro->name, i, tmp);
    iterator = tmp;
    while (iterator) {
        if (!(iterator->flags & DRF_MACRO_FLAGS_ALIAS))
            field->n_defines++;
        iterator = iterator->next;
    }
    defines = calloc(1, (sizeof(*defines) * field->n_defines));
    if (!defines) {
        if (state->debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        return NULL;
    }
    field->defines = defines;
    iterator = tmp;
    while (iterator) {
        if (!(iterator->flags & DRF_MACRO_FLAGS_ALIAS)) {
            define = drf_manual_alloc_define(iterator);
            if (!define)
                return NULL;
            defines[j++] = define;
        }
        iterator = iterator->next;
    }
    qsort(defines, field->n_defines, sizeof(drf_define_t*), compare_defines_by_value);
    for (i = 0; i < field->n_defines; i++) {
        for (j = (i + 1); j < field->n_defines; j++) {
            if (defines[i]->value != defines[j]->value)
                break;
            ret = drf_manual_strcmp(defines[i]->name, defines[j]->name, &k);
            if (ret != 0) {
                if (ret > 0) {
                    str = &defines[j]->name[k];
                    if ((*str == '(') || (strcmp(str, "INIT") == 0))
                        continue;
                } else {
                    str = &defines[i]->name[k];
                    if ((*str != '(') && (strcmp(str, "INIT") != 0))
                        continue;
                }
                define = defines[i];
                defines[i] = defines[j];
                defines[j] = define;
            }
        }
    }
    strcpy((char *)&field->name[0], macro->name);
    field->msb = macro->a;
    field->lsb = macro->b;
    return field;
}

static drf_register_t *drf_manual_alloc_register(drf_macro_t *macro)
{
    char *str;
    drf_macro_t *tmp, *iterator;
    drf_register_t *_register;
    drf_field_t *field, **fields;
    uint32_t i, j = 0;

    _register = calloc(1, (sizeof(*_register) + strlen(macro->name)));
    if (!_register) {
        if (state->debug) {
            fprintf(stderr, "calloc() failed (%s)!\n",
                    strerror(errno));
        }
        drf_set_errno(errno);
        return NULL;
    }
    str = strchr(macro->name, '(');
    i = (str ? (str - macro->name) : strlen(macro->name));
    HASH_FIND(hh, hash_tables[REGISTER_FIELDS], macro->name, i, tmp);
    iterator = tmp;
    while (iterator) {
        if (!(iterator->flags & DRF_MACRO_FLAGS_ALIAS))
            _register->n_fields++;
        iterator = iterator->next;
    }
    fields = calloc(1, (sizeof(*fields) * _register->n_fields));
    if (!fields) {
        if (state->debug)
            fprintf(stderr, "calloc() failed (%s)!\n", strerror(errno));
        drf_set_errno(errno);
        return NULL;
    }
    _register->fields = fields;
    iterator = tmp;
    while (iterator) {
        if (!(iterator->flags & DRF_MACRO_FLAGS_ALIAS)) {
            field = drf_manual_alloc_field(iterator);
            if (!field)
                return NULL;
            fields[j++] = field;
        }
        iterator = iterator->next;
    }
    qsort(fields, _register->n_fields, sizeof(drf_field_t*), compare_fields_by_lsb);
    strcpy((char *)&_register->name[0], macro->name);
    _register->address = macro->b;
    return _register;
}

static int drf_manual_discover_registers(drf_device_t *device)
{
    char *str, *addendum = NULL;
    drf_macro_t *macro, *iterator, *tmp_0, *tmp_1;
    drf_register_t *_register, **registers;
    uint32_t i, j = 0, n_registers;

    state->lwrrent_device = device;
    if (!strcmp(device->fname, "<buffer>")) {
        drf_mcpp_run_prologue();
        for (i = 0; state->mem_buffers[i].data; i++) {
            drf_mcpp_parse_mem_buffer(state->mem_buffers[i].data,
                    state->mem_buffers[i].data_size,
                    &macro_callbacks);
        }
    } else {
        addendum = malloc(strlen(device->fname) + 10);
        if (!addendum) {
            if (state->debug)
                fprintf(stderr, "malloc() failed (%s)!\n", strerror(errno));
            drf_set_errno(errno);
            return -1;
        }
        strcpy(addendum, device->fname);
        drf_mcpp_run_prologue();
        drf_mcpp_parse_header_file(device->fname, 0, &macro_callbacks);
        str = strcasestr(addendum, ".h");
        if (str) {
            strcpy(str, "_addendum.h");
            str = basename(addendum);
            if (state->addendums) {
                for (i = 0; state->addendums[i]; i++) {
                    if (strcasestr(state->addendums[i], str)) {
                        drf_mcpp_parse_header_file(state->addendums[i], 0,
                                &macro_callbacks);
                    }
                }
            }
        }
    }
    if (drf_errno)
        goto failed;
    process_pending_macros();
    if (drf_errno)
        goto failed;
    if (HASH_COUNT(hash_tables[QUEUED_MACROS])) {
        drf_mcpp_parse_mem_buffer(mcpp_buffer, (mcpp_buffer_index + 1),
                &pragma_callbacks);
    }
    if (drf_errno)
        goto failed;
    process_intractable_macros();
    if (drf_errno)
        goto failed;
    sort_macros();
    process_expanded_macros(device);
    if (drf_errno)
        goto failed;
    HASH_ITER(hh, hash_tables[REGISTERS], macro, iterator) {
        if (macro->flags & DRF_MACRO_FLAGS_ALIAS)
            continue;
        if (!DRF_DEVICE_MATCH_INITIAL(device, macro->b))
            continue;
        device->n_registers++;
    }
    registers = calloc(1, (sizeof(*registers) * device->n_registers));
    if (!registers) {
        if (state->debug) {
            fprintf(stderr, "calloc() failed (%s)!\n",
                    strerror(errno));
        }
        drf_set_errno(errno);
        goto failed;
    }
    device->registers = registers;
    HASH_ITER(hh, hash_tables[REGISTERS], macro, iterator) {
        if (macro->flags & DRF_MACRO_FLAGS_ALIAS)
            continue;
        if (!DRF_DEVICE_MATCH_INITIAL(device, macro->b))
            continue;
        if (device->initial_base != device->base) {
            macro->b -= device->initial_base;
            macro->b += device->base;
        }
        _register = drf_manual_alloc_register(macro);
        if (!_register)
            goto failed;
        registers[j++] = _register;
    }
    n_registers = j;
    qsort(registers, n_registers, sizeof(drf_register_t*), compare_registers_by_address);

failed:
    drf_mcpp_run_epilogue();
    free(addendum);
    for (i = PENDING_MACROS; i <= REGISTER_FIELD_VALUES; i++) {
        HASH_ITER(hh, hash_tables[i], macro, iterator) {
            tmp_0 = macro->next;
            while (tmp_0) {
                tmp_1 = tmp_0->next;
                free(tmp_0);
                tmp_0 = tmp_1;
            }
            HASH_DEL(hash_tables[i], macro);
            free(macro);
        }
    }
    return (drf_errno ? -1 : 0);
}

int drf_manual_lookup_by_regular_expression(drf_state_t *__state,
        const char *regex, drf_register_t ***regs,
        uint32_t *n_regs)
{
    drf_device_t *device;
    drf_register_t *_register, **registers = NULL;
    uint32_t i, j, n_registers = 0;
    regex_t preg;
    int ret;

    DRF_MANUAL_SET_GLOBAL_STATE(__state);
    do {
        __drf_state_t *state = __state;
        drf_errno = 0;
        ret = regcomp(&preg, regex,
                (REG_EXTENDED | REG_NOSUB | REG_ICASE));
        if (ret) {
            if (state->debug)
                fprintf(stderr, "regcomp() failed (%s)!\n", strerror(errno));
            switch (ret) {
                case REG_ESPACE:
                    drf_set_errno(ENOMEM);
                    break;
                default:
                    drf_set_errno(EILWAL);
                    break;
            }
            goto failed;
        }
        for (i = 0; i < state->n_devices; i++) {
            device = state->devices[i];
            if (strncmp(device->name, regex, strlen(device->name)) != 0)
            {
                continue;
            }
            if (!device->registers) {
                ret = drf_manual_discover_registers(device);
                if (ret < 0)
                    goto failed;
            }
            for (j = 0; j < device->n_registers; j++) {
                _register = device->registers[j];
                if (regexec(&preg, _register->name, 0, NULL, 0))
                    continue;
                n_registers++;
                registers = realloc(registers,
                        (sizeof(*registers) * n_registers));
                if (!registers) {
                    if (state->debug) {
                        fprintf(stderr, "calloc() failed (%s)!\n",
                                strerror(errno));
                    }
                    drf_set_errno(errno);
                    goto failed;
                }
                registers[n_registers-1] = device->registers[j];
            }
        }
        *regs = registers;
        *n_regs = n_registers;
    } while (0);
    if (!n_registers)
        drf_set_errno(ENOENT);
failed:
    DRF_MANUAL_RESET_GLOBAL_STATE();
    if (drf_errno)
        free(registers);
    return (drf_errno ? -1 : 0);
}

int drf_manual_lookup_by_name(drf_state_t *__state, const char *name,
        drf_register_t ***regs, uint32_t *n_regs)
{
    drf_device_t *device;
    char *str;
    drf_register_t *_register, **registers = NULL;
    uint32_t i, j, n, n_registers = 0;
    int ret;

    DRF_MANUAL_SET_GLOBAL_STATE(__state);
    do {
        __drf_state_t *state = __state;
        drf_errno = 0;
        for (i = 0; i < state->n_devices; i++) {
            device = state->devices[i];
            if (strcasestr(name, device->name) != name)
                continue;
            if (!device->registers) {
                ret = drf_manual_discover_registers(device);
                if (ret < 0)
                    goto failed;
            }
            for (j = 0; j < device->n_registers; j++) {
                _register = device->registers[j];
                if (strcasecmp(_register->name, name)) {
                    str = strchr(_register->name, '(');
                    n = (str ? (str - _register->name) : strlen(_register->name));
                    if (strlen(name) != n)
                        continue;
                    if (strncasecmp(_register->name, name, n))
                        continue;
                }
                n_registers++;
                registers = realloc(registers,
                        (sizeof(*registers) * n_registers));
                if (!registers) {
                    if (state->debug) {
                        fprintf(stderr, "calloc() failed (%s)!\n",
                                strerror(errno));
                    }
                    drf_set_errno(errno);
                    goto failed;
                }
                registers[n_registers-1] = device->registers[j];
            }
        }
        *regs = registers;
        *n_regs = n_registers;
    } while (0);
    if (!n_registers)
        drf_set_errno(ENOENT);
failed:
    DRF_MANUAL_RESET_GLOBAL_STATE();
    if (drf_errno)
        free(registers);
    return (drf_errno ? -1 : 0);
}

int drf_manual_lookup_by_address(drf_state_t *__state, uint32_t address,
        drf_register_t ***regs, uint32_t *n_regs)
{
    drf_device_t *device;
    drf_register_t *_register, **registers = NULL;
    uint32_t i, j, n_registers = 0;
    int ret;

    DRF_MANUAL_SET_GLOBAL_STATE(__state);
    do {
        __drf_state_t *state = __state;
        drf_errno = 0;
        for (i = 0; i < state->n_devices; i++) {
            device = state->devices[i];
            if (!DRF_DEVICE_MATCH(state->devices[i], address))
                continue;
            if (!device->registers) {
                ret = drf_manual_discover_registers(device);
                if (ret < 0)
                    goto failed;
            }
            for (j = 0; j < device->n_registers; j++) {
                _register = device->registers[j];
                if (_register->address != address)
                    continue;
                n_registers++;
                registers = realloc(registers,
                        (sizeof(*registers) * n_registers));
                if (!registers) {
                    if (state->debug) {
                        fprintf(stderr, "calloc() failed (%s)!\n",
                                strerror(errno));
                    }
                    drf_set_errno(errno);
                    goto failed;
                }
                registers[n_registers-1] = device->registers[j];
            }
        }
        *regs = registers;
        *n_regs = n_registers;
    } while (0);
    if (!n_registers)
        drf_set_errno(ENOENT);
failed:
    DRF_MANUAL_RESET_GLOBAL_STATE();
    return (drf_errno ? -1 : 0);
}

int drf_manual_lookup_by_address_range(drf_state_t *__state,
        uint32_t start_address, uint32_t end_address, drf_register_t ***regs,
        uint32_t *n_regs)
{
    drf_device_t *device = NULL;
    uint32_t n_registers = 0;
    drf_register_t **registers = NULL;
    uint32_t address, i = 0, j;
    int ret;

    DRF_MANUAL_SET_GLOBAL_STATE(__state);
    do {
        __drf_state_t *state = __state;
        drf_errno = 0;
        address = start_address;
        while (address < end_address) {
            if (!device || !DRF_DEVICE_MATCH(device, address)) {
                for (i = 0; i < state->n_devices; i++) {
                    if (DRF_DEVICE_MATCH(state->devices[i], address))
                        break;
                }
            }
            if (i < state->n_devices) {
                device = state->devices[i];
                if (!device->registers) {
                    ret = drf_manual_discover_registers(device);
                    if (ret < 0)
                        goto failed;
                }
                for (j = 0; j < device->n_registers; j++) {
                    if (device->registers[j]->address == address)
                        break;
                }
                if (j < device->n_registers) {
                    n_registers++;
                    registers = realloc(registers,
                            (sizeof(*registers) * n_registers));
                    if (!registers) {
                        if (state->debug) {
                            fprintf(stderr, "calloc() failed (%s)!\n",
                                    strerror(errno));
                        }
                        drf_set_errno(errno);
                        goto failed;
                    }
                    registers[n_registers-1] = device->registers[j];
                }
            }
            address += 4;
        }
        *regs = registers;
        *n_regs = n_registers;
    } while (0);
    if (!n_registers)
        drf_set_errno(ENOENT);
failed:
    DRF_MANUAL_RESET_GLOBAL_STATE();
    return (drf_errno ? -1 : 0);
}
