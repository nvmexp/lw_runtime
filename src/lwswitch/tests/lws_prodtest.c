/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * lws_flash - based on lwswitch ioctl test.
 *
 *
 * Usage: lws_prodtest [--switch N] [--path manual_path] [--chip chip_name]
 *
 * With no args each test is run in order on /dev/lwswitch0.
 *  - Other switch devices can be specified like '--switch 2' to correspond to /dev/lwswitch2.
 *
 * Examples:
 *  lws_prodtest
 */

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>
#include <string.h>
#include <sys/ioctl.h>

#include "lwtypes.h"
#include "lwlink.h"
#include "ctrl/ctrl2080/ctrl2080lwlink.h"
#include "ctrl_dev_lwswitch.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"

#define IOCTL(fd, cmd, params, rc) \
    rc = ioctl(fd, cmd, params)

#define CHECKV(rc, str, ...) \
    if (!(rc)) \
    { \
        fflush(stdout); \
        perror("    *** failed!"); \
        errors++; \
    }

#define CHECK(rc)   CHECKV(rc, "%s", "")

static const struct option ioctl_opts[] =
{
    { "switch",  required_argument, NULL, 's' },
    { "path",    required_argument, NULL, 'p' },
    { "chip",    required_argument, NULL, 'c' },
    { 0, 0, 0, 0 }
};

typedef struct
{
    LwU32   block_id;
    char*   file_name;
} FILE_LIST_TYPE;

FILE_LIST_TYPE  file_list[] =
{
    {REGISTER_RW_ENGINE_AFS,            "dev_afs_ip.h"},
    {REGISTER_RW_ENGINE_NPORT,          "dev_nport_ip.h"},
    {REGISTER_RW_ENGINE_NPORT,          "dev_egress_ip.h"},
    {REGISTER_RW_ENGINE_NPORT,          "dev_ftstate_ip.h"},
    {REGISTER_RW_ENGINE_NPORT,          "dev_ingress_ip.h"},
    {REGISTER_RW_ENGINE_NPORT,          "dev_route_ip.h"},
    {REGISTER_RW_ENGINE_MINION,         "dev_minion_ip.h"},
    {REGISTER_RW_ENGINE_NPG,            "dev_npg_ip.h"},
    {REGISTER_RW_ENGINE_NPG_PERFMON,    "dev_npgperf_ip.h"},
    {REGISTER_RW_ENGINE_XP3G,           "dev_lw_xp.h"},
    {REGISTER_RW_ENGINE_XVE,            "dev_lw_xve.h"},
    {REGISTER_RW_ENGINE_DLPL,           "dev_lwl_ip.h"},
    {REGISTER_RW_ENGINE_SIOCTRL,        "dev_lwlctrl_ip.h"},
    {REGISTER_RW_ENGINE_LWLIPT,         "dev_lwlipt_ip.h"},
    {REGISTER_RW_ENGINE_TX_PERFMON,     "dev_lwlperf_ip.h"},    // REGISTER_RW_ENGINE_RX_PERFMON
    {REGISTER_RW_ENGINE_SAW,            "dev_lwlsaw_ip.h"},
    {REGISTER_RW_ENGINE_LWLTLC,         "dev_lwltlc_ip.h"},
    {REGISTER_RW_ENGINE_RAW,            "dev_lws.h"},
    {REGISTER_RW_ENGINE_RAW,            "dev_pri_ringmaster.h"},
    {REGISTER_RW_ENGINE_RAW,            "dev_pri_ringstation_prt.h"},
    {REGISTER_RW_ENGINE_RAW,            "dev_pri_ringstation_sys.h"},
//    {REGISTER_RW_ENGINE_SWX,            "dev_swx_ip.h"}
};

const LwU32 file_list_count = sizeof(file_list)/sizeof(file_list[0]);

#define HEADER_PATH "../../../common/inc/hwref/lwswitch"
#define CHIP_PATH   "svnp01"

#define DEFINE_STRING       "#define "
#define PROD_STRING         "__PROD"
#define DEVICE_STRING       "/* RW--D */"
#define REGISTER_STRING     "/* RW-4R */"
#define FIELD_STRING        ":"
//#define FIELD_ALT1_STRING   "/* RWIVF */"
//#define FIELD_ALT2_STRING   "/* RWEVF */"
#define VALUE_STRING        "/* RW--V */"

main(int argc, char **argv)
{
    char file[64];
    int fd, rc;
    int errors = 0;
    int opt, long_idx;
    int minor = 0;

    char chip_name[256] = CHIP_PATH;
    char header_path[256] = HEADER_PATH;

    FILE *register_fd;
    char reg_file[256];
    int idx_file;
    char register_line[256];
    char field_line[256];
    char line[256];
    int line_num;
    int register_line_num;
    int field_line_num;

    char register_name[256];
    char field_name[256];
    int register_offset;
    int field_hi, field_lo;
    int prod_value;

    char *temp;
    char *temp_end;
    int copy_size;

    int prod_count = 0;
    int error_parse_count = 0;
    int error_prod_count = 0;

    int eng_instance;
    LWSWITCH_REGISTER_READ register_r;
    int val;

    while (1)
    {
        opt = getopt_long(argc, argv, "s:p:c:", ioctl_opts, &long_idx);
        if (opt == -1)
            break;
        switch (opt)
        {
            case 's':
                minor  = atoi(optarg);
                break;
            case 'p':
                strcpy(header_path, optarg);
                break;
            case 'c':
                strcpy(chip_name, optarg);
                break;
            default:
                printf("usage: lws_prodtest [--switch n] [--path manual_path] [--chip chip_name]\n");
                exit(1);
        }
    }

    sprintf(file, "/dev/lwswitch%d", minor);

    fd = open(file, O_RDONLY);
    if (fd < 0)
    {
        printf("open of %s failed!\n", file);
        exit(1);
    }

    printf("Chip: %s\n", chip_name);
    printf("Path: %s\n", header_path);

    for (idx_file = 0; idx_file < file_list_count; idx_file++)
    {
        reg_file[0] = 0;
        strcat(reg_file, header_path);
        strcat(reg_file, "/");
        strcat(reg_file, chip_name);
        strcat(reg_file, "/");
        strcat(reg_file, file_list[idx_file].file_name);

        if( (register_fd = fopen(reg_file, "r")) == NULL)
        {
            printf("Could not open the reg file %s.  Skipping\n", reg_file);
            continue;
        }

        printf("Processing %s.\n", reg_file);

        line_num = 0;
        register_line[0] = 0;
        field_line[0] = 0;
        register_line_num = 0;
        field_line_num = 0;

        while(fgets(line, sizeof(line), register_fd))
        {
            line_num++;
            line[strlen(line)-1] = 0;           // get rid of newline

            if (strstr(line, DEFINE_STRING) != line)
            {
                // Skip any line that doesn't start with '#define '
                continue;
            }

            if (strstr(line, DEVICE_STRING) != 0)
            {
                // Filter out device range entries so they aren't confused with fields
                continue;
            }

            if (strstr(line, PROD_STRING) != 0)
            {
                prod_count++;

                if ((field_line_num == 0) || (register_line_num == 0))
                {
                    printf("Error: Can not find register or field definition\n");
                    printf("[%5d][PROD]:%s\n", line_num, line);
                    error_parse_count++;
                    continue;
                }

                temp = strchr(&register_line[strlen(DEFINE_STRING)], ' ');
                if (temp)
                {
                    // First, find the register name
                    copy_size = (int) (temp - &register_line[strlen(DEFINE_STRING)]);
                    strncpy(register_name, &register_line[strlen(DEFINE_STRING)], copy_size);
                    register_name[copy_size] = 0;

                    // Extract the register offset
                    temp = strstr(register_line, register_name);
                    temp += strlen(register_name);
                    register_offset = strtol(temp, NULL, 16);

                    // printf("[%5d][NAME]:%s = 0x%08x\n", register_line_num, register_name, register_offset);

                    // Verify register name is in field name
                    temp = strstr(field_line, register_name);
                    if (temp)
                    {
                        // Extract field name
                        temp += strlen(register_name);
                        temp_end = strchr(temp, ' ');
                        copy_size = (int) (temp_end - temp);
                        strncpy(field_name, temp, copy_size);
                        field_name[copy_size] = 0;
                        // printf("[%5d][FLD ]:%s\n", field_line_num, field_name);

                        // Verify field name is in prod name
                        temp = strstr(line, field_name);
                        if (temp)
                        {
                            // Now strip out the field extents
                            temp = strchr(field_line, ':');
                            if (temp)
                            {
                                field_hi = atoi(temp-2);
                                field_lo = atoi(temp+1);
                                if ((field_hi > 31) ||
                                    (field_lo <  0) ||
                                    (field_lo > field_hi))
                                {
                                    printf("Error parsing field extents %d:%d in line [%5d][FLD ]:%s\n",
                                        field_hi, field_lo,
                                        field_line_num, field_line);
                                    error_parse_count++;
                                }
                                else
                                {
                                    // printf("[%5d][FLD ]:%s %d:%d\n", field_line_num, field_name, field_hi, field_lo);

                                    temp = strstr(line, PROD_STRING);
                                    temp += strlen(PROD_STRING);
                                    prod_value = strtol(temp, NULL, 16);

                                    // printf("[%5d][PROD]:%s=0x%08x\n", line_num, "__PROD", prod_value);

                                    for (eng_instance = 0; eng_instance < 18; eng_instance++)
                                    {
                                        register_r.engine = file_list[idx_file].block_id;
                                        register_r.instance = eng_instance;
                                        register_r.offset = register_offset;
                                        register_r.val = 0;

                                        IOCTL(fd, IOCTL_LWSWITCH_REGISTER_READ, &register_r, rc);
                                        if (rc != 0)
                                        {
                                            // printf("ERROR: [0x%02x(%2d)]: %s%s [0x%06x(%d:%d)] read error\n",
                                            //       file_list[idx_file].block_id,
                                            //       eng_instance,
                                            //       register_name, field_name,
                                            //       register_offset,
                                            //       field_hi, field_lo);
                                            break;
                                        }

                                        val = (register_r.val >> field_lo) & ((0x1 << (field_hi - field_lo + 1))-1);

                                        if (val != prod_value)
                                        {
                                            printf("ERROR: [0x%02x(%2d)]: %s%s [0x%06x(%d:%d)] expected 0x%08x but read 0x%08x\n",
                                                file_list[idx_file].block_id,
                                                eng_instance,
                                                register_name, field_name,
                                                register_offset,
                                                field_hi, field_lo,
                                                prod_value, val);
                                            error_prod_count++;
                                        }
                                        else
                                        {
                                            // printf("OK: [0x%02x(%2d)]: %s%s [0x%06x(%d:%d)] expected 0x%08x and read 0x%08x\n",
                                            //     file_list[idx_file].block_id,
                                            //     eng_instance,
                                            //     register_name, field_name,
                                            //     register_offset,
                                            //     field_hi, field_lo,
                                            //     prod_value, val);
                                        }

                                        if (file_list[idx_file].block_id == REGISTER_RW_ENGINE_RAW)
                                        {
                                            break;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                printf("Error parsing field extents in line [%5d][FLD ]:%s\n",
                                    field_line_num, field_line);
                                error_parse_count++;
                            }
                        }
                        else
                        {
                            printf("Error parsing line [%5d][PROD]:%s\n", line_num, line);
                            error_parse_count++;
                        }
                    }
                    else
                    {
                        printf("Error parsing line [%5d][FLD ]:%s\n", field_line_num, field_line);
                        error_parse_count++;
                    }
                }
                else
                {
                    printf("Error parsing line [%5d][NAME]:%s\n", register_line_num, register_name);
                    error_parse_count++;
                }
            }

            if (strstr(line, REGISTER_STRING) != 0)
            {
                strcpy(register_line, line);
                register_line_num = line_num;
            }

            if ((strstr(line, FIELD_STRING) != 0) /*||
                (strstr(line, FIELD_ALT1_STRING) != 0) ||
                (strstr(line, FIELD_ALT2_STRING) != 0)*/)
            {
                // printf("[%5d][FLD ]:%s\n", line_num, line);
                strcpy(field_line, line);
                field_line_num = line_num;
            }

        }
    }

    printf("Processed %d __PROD values\n", prod_count);
    printf("%d parse errors\n", error_parse_count);
    printf("%d __PROD mismatches\n", error_prod_count);
}
