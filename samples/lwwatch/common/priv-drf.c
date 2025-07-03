/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#ifndef DRF_WINDOWS
#include <dirent.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#endif
#include <errno.h>

#include <drf.h>
#include <drf_index.h>
#include <drf_manual.h>
#include <drf_device.h>
#include <drf_register.h>
#include <drf_field.h>
#include <drf_define.h>

#include "os.h"
#include "chip.h"
#include "hal.h"
#include "exts.h"
#include "priv.h"

#define MAX_STR_LEN 32

#ifdef DRF_WINDOWS
#define snprintf _snprintf
typedef unsigned __int32 uint32_t;
#endif

#define DRF_FIELD_VALUE(value, msb, lsb)                        \
    (((value) >> (lsb)) & ((1ULL << ((msb) - (lsb) + 1)) - 1))

#define DRF_FIELD_MASK(msb, lsb)                \
    ((1ULL << ((msb) - (lsb) + 1)) - 1)

#define DRF_LINE_BUFFER_SIZE 1024

typedef struct {
    drf_state_t *state;
    uint64_t lwBar0;
    char paths[MAX_PATHS][MAX_STR_LEN];
    int n_paths;
    char class[MAX_STR_LEN];
} manual_t;

static manual_t manuals[MAX_GPUS];

// Given a register and field ID, return pointer to the field name with a special
// case for registers that contain() fields.
static char* get_field_name( drf_register_t *_register, uint32_t fieldId )
{
    char* str;
    uint32_t n;
    str = strchr( _register->name, '(' );
    n = (uint32_t) (str ? (str - _register->name) : strlen(_register->name) );
    return( &(_register->fields[fieldId]->name[n+1]) );
}

// Given a field and value, return pointer to a symbolic name, or NULL if none found
static char* get_define_name( drf_field_t *_field, uint32_t field_value )
{
    uint32_t define_num, n;
    char* define_name = NULL;
    char* str = NULL;
    drf_define_t *this_define;
    for( define_num = 0 ; define_num < _field->n_defines ; define_num++ ) {
        this_define = _field->defines[define_num];
        if( field_value == this_define->value ) {
            str = strchr( _field->name, '(' );
            n = (uint32_t) (str ? (str - _field->name) : strlen(_field->name) );
            define_name = &(this_define->name[n+1]);
            break;
        }
    }
    return( define_name );
}

// Print a register's fields, possibly skipping registers and fields that are zero.
static void decode_register(drf_register_t *_register, uint32_t value, BOOL skip_zeroes)
{
    uint32_t i, field_value;
    drf_field_t *field;
    char *field_name, *define_name;
    char field_info[DRF_LINE_BUFFER_SIZE];

    // Print nothing if skipping zeroes
    if( (PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES == skip_zeroes) && (! value) ) {
        return;
    }

    // BADFxxxx means the value is likely bogus due to clocks being
    // off, the register being unconnected, or the unit with this
    // register was floorswept.  Print a warning but continue with
    // parsing as it's remotely possible the value is legitimate.

    dprintf ("%-25s @(0x%08x) = 0x%08x", _register->name, _register->address, value);
    if( (value & 0xFFFF0000) == 0xBADF0000 ) {
        dprintf(" ** WARNING: likely bogus register value 0xBADFxxxx" );
    }
    dprintf("\n");

    // Printing another line with the field value is redundant if
    // there's only 1 field
    if( 1  == _register->n_fields ) {
        return;
    }

    for (i = 0 ; i < _register->n_fields ; i++) {
        // Get field information including possibly a symbolic name matching the value.
        field = _register->fields[i];
        field_value = DRF_FIELD_VALUE(value, field->msb, field->lsb);

        // If we're skipping zeroes, move on to the next field
        if ((PRIV_DUMP_REGISTER_FLAGS_SKIP_ZEROES == skip_zeroes) && (! field_value)) {
            continue;
        }

        field_name = get_field_name( _register, i );
        define_name = get_define_name( field, field_value );

        snprintf(field_info, sizeof(field_info), "%s (%u:%u)", field_name,
                 field->msb, field->lsb);
        dprintf("                %-33s = <%s> [0x%04x]\n", field_info,
                    define_name ? define_name : "", field_value);
    }
}

static int write_register_field(drf_register_t *_register, char *field_name,
                                char *define_name, uint32_t data)
{
    uint32_t i, j, n, value;
    drf_field_t *field;
    drf_define_t *define;
    char *str;

    str = strchr(_register->name, '(');
    n = (uint32_t) (str ? (str - _register->name) : strlen(_register->name));

    for (i = 0; i < _register->n_fields; i++) {
        field = _register->fields[i];
        if (!strcasecmp(&field->name[n+1], field_name))
            break;
    }
    if (i == _register->n_fields)
        return -ENOENT;
    if (define_name) {
        str = strchr(field->name, '(');
        n = (uint32_t) (str ? (str - field->name) : strlen(field->name));
        for (j = 0; j < field->n_defines; j++) {
            define = field->defines[j];
            if (!strcasecmp(&define->name[n+1], define_name))
                break;
        }
        if (j == field->n_defines)
            return -ENOENT;
        data = define->value;
    }

    if (data & ~DRF_FIELD_MASK(field->msb, field->lsb))
        return -EILWAL;

    value = GPU_REG_RD32(_register->address);

    value &= ~(DRF_FIELD_MASK(field->msb, field->lsb) << field->lsb);
    value |= (data << field->lsb);

    GPU_REG_WR32(_register->address, value);

    return 0;
}

static struct {
    int status;
    const char *error_message;
} manpath_errors[] = {
    { ENOENT,  "The path '%s' does not exist.\n"                },
    { EACCES,  "You lack permission to access the path '%s'.\n" },
    { ENOTDIR, "The path '%s' is not a directory.\n"            },
    { ELOOP,   "The path '%s' contains a symlink loop.\n"       }
};

#define MANPATH_ERRORS                                          \
    (sizeof(manpath_errors) / sizeof(manpath_errors[0]))

static const char *manpath_help = "Please set your LWW_MANUAL_SDK "
    "environment variable to point to your //sw/resman/manuals "
    "directory.";

static void print_manpath_error(const char *manpath, int status)
{
    const char *message = "%s";
    uint32_t i;

    for (i = 0; i < MANPATH_ERRORS; i++) {
        if (manpath_errors[i].status == status) {
            message = manpath_errors[i].error_message;
            break;
        }
    }

    dprintf(message, manpath);
    dprintf("%s\n", manpath_help);
}

static manual_t *get_chip_manual(void)
{
    manual_t *manual = NULL;
    drf_state_t *state;
    char *manpath, *path, *paths[MAX_PATHS];
    struct stat stat_buf;
    char *file, **files = NULL;
    uint32_t n_files = 0;
    DIR *dp;
    struct dirent *ep;
    drf_device_t **devices;
    uint32_t n_devices;
    unsigned int j;
    int ret, i;
    char *str;

    manpath = getelw("LWW_MANUAL_SDK");
    if (!manpath) {
        dprintf("%s\n", manpath_help);
        return NULL;
    }

    ret = stat(manpath, &stat_buf);
    if (ret < 0) {
        switch (errno) {
        case ENOENT:
        case EACCES:
        case ENOTDIR:
        case ELOOP:
            print_manpath_error(manpath, errno);
            return NULL;
        default:
            dprintf("stat() failed (%d).\n", errno);
            return NULL;
        case 0:
            break;
        }
    }

    for (i = 0; i < MAX_GPUS; i++) {
        manual = &manuals[i];
        if (lwBar0 == manual->lwBar0)
            break;
    }

    if (i == MAX_GPUS) {
        for (i = 0; i < MAX_GPUS; i++) {
            manual = &manuals[i];
            if (manuals[i].lwBar0 == 0)
                break;
        }

        if (i == MAX_GPUS) {
            dprintf("MAX_GPUS exceeded; please update %s.\n",
                    __FUNCTION__);
            return NULL;
        }

        for (i = 0; i < MAX_PATHS; i++) {
            paths[i] = manual->paths[i];
        }

        if (!GetManualsDir(paths, manual->class,
                           &manual->n_paths)) {
            dprintf("Unsupported GPU. Please update GetManualsDir().\n");
            return NULL;
        }

        for (i = (manual->n_paths - 1); i >= 0; i--) {
            path = malloc(strlen(manpath) + strlen(paths[i]) + 2);
            if (!path) {
                dprintf("malloc() failed (%d)!\n",
                        errno);
                goto failed;
            }
            sprintf(path, "%s/%s", manpath, paths[i]);

            dp = opendir(path);
            if (!dp) {
                dprintf("opendir(%s) failed (%d).\n", path, errno);
                goto failed;
            }

            while ((ep = readdir(dp))) {
                if (strncasecmp(ep->d_name, "dev_", 4)) {
                    if (strncasecmp(ep->d_name, "disp_dsi_sc", 11))
                        continue;
                }
                if (!strcmp(ep->d_name, "."))
                    continue;
                if (!strcmp(ep->d_name, ".."))
                    continue;
                str = strstr(ep->d_name, ".h");
                if (!str)
                    str = strstr(ep->d_name, ".H");
                if (!str || strcasecmp(str, ".h")) {
                    /* If this isn't a header file, skip it. */
                    continue;
                }

                file = malloc(strlen(path) + strlen(ep->d_name) + 3);
                if (!file) {
                    dprintf("malloc() failed (%d)!\n",
                            errno);
                    goto failed;
                }
                n_files++;
                files = realloc(files, (sizeof(char *) * (n_files + 1)));
                if (!files) {
                    dprintf("realloc() failed (%d)!\n",
                            errno);
                    goto failed;
                }

                sprintf(file, "%s/%s", path, ep->d_name);
                files[n_files-1] = file;
            }

            closedir(dp);
            free(path);

            if (n_files)
                files[n_files] = NULL;
        }

        if (!n_files) {
            dprintf("No manuals found in the path '%s'. %s\n",
                    manpath, manpath_help);
            goto failed;
        }

        ret = drf_state_alloc((void *)files, &state);
        if (ret < 0) {
            switch (drf_errno) {
            case ENOENT:
                dprintf("No manuals found in the path '%s'. %s\n",
                        manpath, manpath_help);
                goto failed;
            default:
                dprintf("Failed to allocate state (%d)\n", drf_errno);
                goto failed;
            }
        }

        ret = drf_index_get_devices(state, &devices, &n_devices);
        if (ret < 0) {
            drf_state_free(manual->state);
            dprintf("Failed to index files (%d)\n", drf_errno);
            goto failed;
        }

        manual->lwBar0 = lwBar0;
        manual->state = state;
    }

    return manual;

failed:
    if (files) {
        for (j = 0; j < n_files; j++)
            free(files[j]);
        free(files);
    }

    return NULL;
}

static void print_priv_dump_usage(void)
{
    dprintf("usage: pd <address | name | expression>\n");
}

// Expected values for skip_zeroes are defined in priv.h
void priv_dump_register( const char *params, LwU32 skip_zeroes )
{
    manual_t *manual;
    char *args, *endptr, *nptr = (char *)params;
    uint32_t address;
    drf_state_t *state;
    char *name = NULL, *regex = NULL;
    drf_register_t *_register, **registers;
    uint32_t n_registers;
    unsigned int j;
    int ret, c, i;

    if (*nptr == '\0') {
        print_priv_dump_usage();
        return;
    }

    args = malloc(strlen(params) + 1);
    if (!args) {
        dprintf("malloc() failed (%d)!\n", errno);
        return;
    }
    strcpy(args, params);

    // Remove trailing spaces
    for (i = (int) (strlen(args) - 1); i >= 0; i--) {
        if (args[i] != ' ')
            break;
        args[i] = '\0';
    }

    // See if this is a hardcoded address or symbolic name, and if the latter, is it a regex
    address = strtoul(nptr, &endptr, 0);
    if (*endptr != '\0') {
        name = (char *)args;

        for (j = 0; j < strlen(name); j++) {
            c = name[j];
            if (!isalnum(c) && (c != '_') && (c != '(') && (c != ')')) {
                regex = name;
                break;
            }
        }
    }

    if (!(manual = get_chip_manual()))
        goto failed;
    state = manual->state;

    if (regex) {
        ret = drf_manual_lookup_by_regular_expression(state, regex,
                                                      &registers, &n_registers);
    } else if (name) {
        ret = drf_manual_lookup_by_name(state, name, &registers,
                                        &n_registers);
    } else {
        ret = drf_manual_lookup_by_address(state, address, &registers,
                                           &n_registers);
    }

    if (ret < 0) {
        switch (drf_errno) {
        case ENOENT:
            dprintf("No such register(s) \"%s\".\n", params);
            goto failed;
        default:
            dprintf("Failed to look up register(s) \"%s\" (%d).\n",
                    params, drf_errno);
            goto failed;
        }
    }
    for (j = 0; j < n_registers; j++) {
        _register = registers[j];
        decode_register(_register, GPU_REG_RD32(_register->address), skip_zeroes );
    }

failed:
    free(args);
}

// This is the 'original' priv_dump called by the 'pd' command
// and from other C code.
void priv_dump( const char *params )
{
    priv_dump_register( params, PRIV_DUMP_REGISTER_FLAGS_DEFAULT );
}

static void print_priv_emit_usage(void)
{
    dprintf("usage: pe <name>.<field> <data>\n");
}

void priv_emit(const char *params)
{
    manual_t *manual;
    char *args, *str, *endptr, *nptr;
    drf_state_t *state;
    char *name, *field_name;
    uint32_t data;
    char *define_name = NULL;
    drf_register_t *_register, **registers;
    uint32_t n_registers;
    unsigned int j;
    int ret, c, i;

    if (*params == '\0') {
        print_priv_emit_usage();
        return;
    }

    str = strchr(params, '.');
    if (!str) {
        print_priv_emit_usage();
        return;
    }

    str = strchr(params, ' ');
    if (!str) {
        print_priv_emit_usage();
        return;
    }

    args = malloc(strlen(params) + 1);
    if (!args) {
        dprintf("malloc() failed (%d)!\n", errno);
        return;
    }
    strcpy(args, params);

    for (i = (int) (strlen(args) - 1); i >= 0; i--) {
        if (args[i] != ' ')
            break;
        args[i] = '\0';
    }

    name = args;
    str = strchr(name, '.');
    *str = '\0';

    for (j = 0; j < strlen(name); j++) {
        c = name[j];
        if (!isalnum(c) && (c != '_') && (c != '(') && (c != ')')) {
            print_priv_emit_usage();
            goto failed;
        }
    }

    field_name = (str + 1);
    str = strchr(field_name, ' ');
    *str = '\0';

    for (j = 0; j < strlen(field_name); j++) {
        c = field_name[j];
        if (!isalnum(c) && (c != '_') && (c != '(') && (c != ')')) {
            print_priv_emit_usage();
            goto failed;
        }
    }

    nptr = (str + 1);
    data = strtoul(nptr, &endptr, 0);
    if (*endptr != '\0')
        define_name = nptr;

    if (!(manual = get_chip_manual()))
        goto failed;
    state = manual->state;

    ret = drf_manual_lookup_by_name(state, name, &registers,
                                    &n_registers);

    if (ret < 0) {
        switch (drf_errno) {
        case ENOENT:
            dprintf("No such register(s).\n");
            goto failed;
        default:
            dprintf("Failed to look up register(s) (%d).\n",
                    drf_errno);
            goto failed;
        }
    }

    for (j = 0; j < n_registers; j++) {
        _register = registers[j];
        ret = write_register_field(_register, field_name, define_name, data);
        switch (ret) {
        case -ENOENT:
            dprintf("No such field/define.\n");
            goto failed;
        case -EILWAL:
            dprintf("Value 0x%x is not valid.\n", data);
            goto failed;
        }
    }

failed:
    free(args);
}

BOOL parseManualReg(LwU32 address, LwU32 data, BOOL isListAll)
{
    manual_t *manual = NULL;
    drf_state_t *state;
    drf_register_t *_register, **registers;
    uint32_t i, n_registers;
    int ret;

    if (!(manual = get_chip_manual()))
        return FALSE;
    state = manual->state;

    ret = drf_manual_lookup_by_address(state, address, &registers,
                                       &n_registers);

    if (ret < 0) {
        switch (drf_errno) {
        case ENOENT:
            dprintf("No such register(s).\n");
            return FALSE;
        default:
            dprintf("Failed to look up register(s) (%d).\n",
                    drf_errno);
            return FALSE;
        }
    }

    for (i = 0; i < n_registers; i++) {
        _register = registers[i];
        decode_register(_register, data, FALSE );
    }

    return TRUE;
}

// Get a register name from the manuals.
//
// Returns TRUE  - Register name found
//         FALSE - Register name not found.
//
BOOL getManualRegName(LwU32 address, char *szRegisterName, LwU32 nNameLength)
{
    manual_t *manual = NULL;
    drf_register_t **registers;
    uint32_t n_registers;
    int ret;

    if (nNameLength < 4)
        return FALSE;

    manual = get_chip_manual();

    if (manual == NULL)
        return FALSE;

    ret = drf_manual_lookup_by_address(manual->state, address, &registers,
                                       &n_registers);
    if (ret < 0)
        return FALSE;

    if (n_registers < 1)
        return FALSE;

    strncpy(szRegisterName, registers[0]->name, nNameLength);
    szRegisterName[nNameLength - 1] = 0;  // Append trailing 0.

    return TRUE;
}
