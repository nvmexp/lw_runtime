/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include <drf.h>
#include <drf_index.h>
#include <drf_manual.h>
#include <drf_device.h>
#include <drf_register.h>
#include <drf_field.h>
#include <drf_define.h>

#include "drf_util.h"

static const char *help_text = \
    ""
    "drf-lookup - DRF manual parser utility\n"
    "\n"
    "Usage: drf-lookup [options] [name|offset|expr] [value]\n"
    "\n"
    "Options:\n"
    "  -d, --decode-register\n"
    "      Instruct drf-lookup to decode the given value as if it had\n"
    "      been read from a specific register. This register must\n"
    "      be identified via its 'name' or 'offset'.\n"
    "  -h, --help\n"
    "      Print usage information.\n"
    "  -i, --dump-index\n"
    "      Print the table of devices defined by the manual header files\n"
    "      found in the path (or paths) specified via the '-p' option.\n"
    "  -l, --list-files\n"
    "      List the manual header files found in the path (or paths)\n"
    "      specified via the '-p' option.\n"
    "  -r, --match-regex\n"
    "      When specified, this option prompts drf-lookup to print\n"
    "      the addresses, fields and possible field values of any registers\n"
    "      with names that match the regular expression 'expr'.\n"
    "  -p, --path='path'\n"
    "      Specify a directory in which to search for DRF-style header\n"
    "      files. This argument must be given at least once, and can\n"
    "      be specified multiple times.\n"
    "  -t, --terse\n"
    "      Make drf-lookup omit field and possible field value information\n"
    "      when listing registers.\n"
    "\n"
    "Examples:\n"
    "  drf-lookup -p /path/to/g84 -l\n"
    "      List the devices defined by the manual header files found in\n"
    "      the directory '/path/to/g84'.\n"
    "\n"
    "  drf-lookup -p /path/to/gk107 -d LW_PMC_BOOT_0 0x0e43a0a2\n"
    "      Decode the value '0x0e43a0a2' as if it had been read from the\n"
    "      register 'LW_PMC_BOOT_0'.\n"
    "\n"
    "  drf-lookup -p /path/to/g92 -r 'LW_PBUS_DEBUG_[0-9]'\n"
    "      Look up and print information about any registers with names\n"
    "      that match the regular expression 'LW_PBUS_DEBUG_[0-9]'.\n"
    "";

static struct option options[] = {
    { "decode-register", no_argument,       NULL, 'd' },
    { "dump-index",      no_argument,       NULL, 'i' },
    { "help",            no_argument,       NULL, 'h' },
    { "list-files",      no_argument,       NULL, 'l' },
    { "path",            required_argument, NULL, 'p' },
    { "match-regex",     no_argument,       NULL, 'r' },
    { "terse",           no_argument,       NULL, 't' },
    { NULL,              0,                 NULL,  0  },
};

#define DRF_FIELD_VALUE(value, msb, lsb) \
    (((value) >> (lsb)) & ((1ULL << ((msb) - (lsb) + 1)) - 1))

static int terse_mode = 0;

static void drf_list_registers(drf_register_t **registers, uint32_t n_registers)
{
    uint32_t i, j, k;
    drf_register_t *_register;
    drf_field_t *field;
    drf_define_t *define;
    char range[6];

    for (i = 0; i < n_registers; i++) {
        _register = registers[i];
        printf("%-70s 0x%08x\n", _register->name, _register->address);
        if (terse_mode)
            continue;
        for (j = 0; j < _register->n_fields; j++) {
            field = _register->fields[j];
            sprintf(range, "%u:%u", field->msb, field->lsb);
            printf("%-70s %10s\n", field->name, range);
            for (k = 0; k < field->n_defines; k++) {
                define = field->defines[k];
                printf("%-70s 0x%08x\n", define->name, define->value);
            }
        }
    }
}

static void drf_decode_registers(drf_register_t **registers,
        uint32_t n_registers, uint32_t value)
{
    uint32_t i, j, k, n, field_value;
    drf_register_t *_register;
    drf_field_t *field;
    drf_define_t *define;
    char *str, *field_name, *define_name;
    char field_info[128];

    for (k = 0; k < n_registers; k++) {
        _register = registers[k];
        printf("%s\n", _register->name);
        printf("@(0x%08x = 0x%08x)\n", _register->address, value);
        for (i = 0; i < _register->n_fields; i++) {
            field = _register->fields[i];
            field_value = DRF_FIELD_VALUE(value, field->msb, field->lsb);
            str = strchr(_register->name, '(');
            n = (str ? (str - _register->name) : strlen(_register->name));
            field_name = &field->name[n+1];
            define_name = "";
            for (j = 0; j < field->n_defines; j++) {
                define = field->defines[j];
                if (field_value == define->value) {
                    define_name = &define->name[strlen(field->name)+1];
                    break;
                }
            }
            snprintf(field_info, sizeof(field_info), "%s (%u:%u)",
                    field_name, field->msb, field->lsb);
            printf("\t\t%-33s = <%s> [0x%04x]\n", field_info, define_name,
                    field_value);
        }
    }
}

int main(int argc, char **argv)
{
    drf_state_t *state;
    char *optstr;
    int c, ret;
    struct stat stat_buf;
    const char **paths = NULL, **files = NULL;
    uint32_t n_paths = 0, n_files = 0;
    char *name = NULL, *file;
    char *regex = NULL;
    uint32_t start_address, end_address;
    uint32_t value, i;
    int decode_register = 0, dump_index = 0;
    int list_files = 0, match_regex = 0;
    struct dirent *ep;
    DIR *dp;
    drf_device_t **devices;
    uint32_t n_devices;
    drf_register_t **registers = NULL;
    uint32_t n_registers = 0;
    char *str;

    optstr = "p:dhilrt";
    while (1) {
        c = getopt_long(argc, argv, optstr, options, NULL);
        if (c < 0)
            break;

        switch (c) {
            case 'd':
                decode_register = 1;
                break;
            case 'h':
                printf("%s", help_text);
                exit(0);
                break;
            case 'i':
                dump_index = 1;
                break;
            case 'l':
                list_files = 1;
                break;
            case 'p':
                ret = stat(optarg, &stat_buf);
                switch (errno) {
                    case ENOENT:
                    case EACCES:
                    case ELOOP:
                    case ENOTDIR:
                        fprintf(stderr, "Bad directory.\n");
                        exit(1);
                    default:
                        fprintf(stderr, "stat() failed (%s).\n",
                                strerror(errno));
                        exit(1);
                    case 0:
                        break;
                }
                if (!S_ISDIR(stat_buf.st_mode)) {
                    fprintf(stderr, "Bad directory.\n");
                    exit(1);
                }
                paths = realloc(paths, (++n_paths * sizeof(char *)));
                if (!paths) {
                    fprintf(stderr, "realloc() failed (%s).\n",
                            strerror(errno));
                    exit(1);
                }
                paths[n_paths-1] = optarg;
                break;
            case 'r':
                match_regex = 1;
                break;
            case 't':
                terse_mode = 1;
                break;
            default:
                fprintf(stderr, "Bad command line option.\n");
                exit(1);
        }
    }

    if (n_paths == 0) {
        fprintf(stderr, "Bad command line.\n");
        exit(1);
    }

    argc -= optind;

    if (dump_index || list_files) {
        if (argc != 0) {
            fprintf(stderr, "Bad command line.\n");
            exit(1);
        }
    } else if (decode_register) {
        if (argc != 2) {
            fprintf(stderr, "Bad command line.\n");
            exit(1);
        }

        ret = drf_parse_integer(argv[optind], &start_address);
        if (ret < 0) {
            name = argv[optind];
            if (strstr(name, "LW") != name) {
                fprintf(stderr, "Bad address.\n");
                exit(1);
            }
        }

        ret = drf_parse_integer(argv[optind+1], &value);
        if (ret < 0) {
            printf("Bad argument.\n");
            exit(1);
        }
    } else {
        if (argc != 1) {
            fprintf(stderr, "Bad command line.\n");
            exit(1);
        }

        if (match_regex)
            regex = argv[optind];
        else {
            ret = drf_parse_range(argv[optind], &start_address,
                    &end_address);
            if (ret < 0) {
                ret = drf_parse_integer(argv[optind], &start_address);
                if (ret < 0) {
                    name = argv[optind];
                    if (strstr(name, "LW") != name) {
                        fprintf(stderr, "Bad argument.\n");
                        exit(1);
                    }
                }
                end_address = start_address;
            }
        }
    }

    for (i = 0; i < n_paths; i++) {
        dp = opendir(paths[i]);
        if (!dp) {
            fprintf(stderr, "opendir() failed (%s).\n", strerror(errno));
            exit(1);
        }

        while ((ep = readdir(dp))) {
            if (strncasecmp(ep->d_name, "dev_", 4)) {
                if (strncasecmp(ep->d_name, "disp_dsi_sc", 11))
                    continue;
            }
            memset(&stat_buf, 0, sizeof(stat_buf));
            file = malloc(strlen(paths[i]) + strlen(ep->d_name) + 2);
            if (!file) {
                fprintf(stderr, "malloc() failed (%s).\n",
                        strerror(errno));
                exit(1);
            }
            sprintf(file, "%s/%s", paths[i], ep->d_name);
            ret = stat(file, &stat_buf);
            if (ret < 0) {
                fprintf(stderr, "stat() failed (%s).\n", strerror(errno));
                exit(1);
            }
            if (!S_ISREG(stat_buf.st_mode)) {
                free(file);
                continue;
            }
            str = strcasestr(ep->d_name, ".h");
            if (!str || strcasecmp(str, ".h")) {
                /* If this isn't a header file, skip it. */
                continue;
            }
            n_files++;
            files = realloc(files, (sizeof(char *) * (n_files + 1)));
            if (!files) {
                fprintf(stderr, "realloc() failed (%s).\n",
                        strerror(errno));
                exit(1);
            }
            files[n_files-1] = file;
        }

        if (n_files)
            files[n_files] = NULL;

        closedir(dp);
    }

    if (list_files) {
        for (i = 0; i < n_files; i++)
            printf("%s\n", files[i]);
        exit(0);
    }

    if (!n_files) {
        fprintf(stderr, "No manuals found in path(s).\n");
        exit(1);
    }

    ret = drf_state_alloc(files, &state);
    if (ret < 0) {
        switch (drf_errno) {
            case ENOENT:
                fprintf(stderr, "No manuals found in path(s).\n");
                break;
            default:
                fprintf(stderr, "Failed to allocate DRF state (%s).\n",
                    strerror(drf_errno));
                break;
        }
        exit(1);
    }

    ret = drf_index_get_devices(state, &devices, &n_devices);
    if (ret < 0) {
        fprintf(stderr, "Failed to index files (%s).\n",
                strerror(drf_errno));
        exit(1);
    }

    if (dump_index) {
        for (i = 0; i < n_devices; i++) {
            printf("0x%08x:0x%08x %s\n", devices[i]->extent,
                    devices[i]->base, devices[i]->name);
        }
    } else if (decode_register) {
        if (name) {
            ret = drf_manual_lookup_by_name(state, name, &registers,
                    &n_registers);
        } else {
            ret = drf_manual_lookup_by_address(state, start_address,
                    &registers, &n_registers);
        }

        if (ret < 0) {
            switch (drf_errno) {
                case ENOENT:
                    fprintf(stderr, "No such register(s).\n");
                    exit(1);
                default:
                    fprintf(stderr, "Failed to look up register(s) (%s).\n",
                            strerror(drf_errno));
                    exit(1);
                case 0:
                    break;
            }
        }

        drf_decode_registers(registers, n_registers, value);
    } else {
        if (match_regex) {
            ret = drf_manual_lookup_by_regular_expression(state, regex,
                    &registers, &n_registers);
        } else if (name) {
            ret = drf_manual_lookup_by_name(state, name, &registers,
                    &n_registers);
        } else if (start_address == end_address) {
            ret = drf_manual_lookup_by_address(state, start_address,
                    &registers, &n_registers);
        } else {
            ret = drf_manual_lookup_by_address_range(state, start_address,
                    end_address, &registers, &n_registers);
        }

        if (ret < 0) {
            switch (drf_errno) {
                case ENOENT:
                    fprintf(stderr, "No such register(s).\n");
                    exit(1);
                default:
                    fprintf(stderr, "Failed to look up register(s) (%s).\n",
                            strerror(drf_errno));
                    exit(1);
            }
        }

        drf_list_registers(registers, n_registers);
    }

    drf_state_free(state);

    return 0;
}
