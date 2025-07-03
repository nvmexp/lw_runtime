/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#define _GNU_SOURCE
#include <sys/types.h>
#include <stdio.h>
#include <getopt.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>

#include <readline/readline.h>
#include <readline/history.h>

#include <jtag.h>
#include <jtag_chain.h>
#include <jtag_tap.h>
#include <jtag_device.h>
#include <jtag_tmc.h>
#include <jtag_data.h>
#include <jtag2host_intfc_api.h>
#include <jtag_private.h>

#include "os.h"
#include "exts.h"

#undef GPU_REG_RD32
#undef GPU_REG_WR32
#undef GPU_REG_RD08
#undef GPU_REG_WR08

static char init_args[19];
static int debug;

static jtag_chain_t *chain;
static jtag_device_t *devices[JTAG_CHAIN_MAX_DEVICES];
static jtag_device_t *device;
static jtag_tmc_t *tmc;

#define GPU_REG_RD32(offset) JTAG_RD32(offset)
#define GPU_REG_WR32(offset,value) JTAG_WR32(offset, value)
#define GPU_REG_RD08(offset) JTAG_RD08(offset)
#define GPU_REG_WR08(offset,value) JTAG_WR08(offset, value)

#define IS_IN_BAR0(address) \
    ((address >= lwBar0) && (address < (lwBar0 + (16 << 20))))

static void JTAG_WR32(uint32_t address, uint32_t value)
{
    int ret;
    ret = jtag2host_intfc_write(chain, device, tmc, 0, 0, address, value, 0xf);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "jtag2host_intfc_write() failed (%d).\n",
                    jtag_errno);
        }
    }
}

static uint32_t JTAG_RD32(uint32_t address)
{
    int ret;
    uint32_t value;
    ret = jtag2host_intfc_read(chain, device, tmc, 0, 0, address, &value);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "jtag2host_intfc_read() failed (%d).\n",
                    jtag_errno);
        }
        value = 0xffffffff;
    }
    return value;
}

static void JTAG_WR08(uint32_t address, uint32_t value)
{
    int ret;
    value = ((uint32_t)value << ((address & 0x3) << 3));
    ret = jtag2host_intfc_write(chain, device, tmc, 0, 0, (address & ~3),
            value, (1 << (address & 3)));
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "jtag2host_intfc_write() failed (%d).\n",
                    jtag_errno);
        }
    }
}

static uint8_t JTAG_RD08(uint32_t address)
{
    int ret;
    uint32_t value;
    ret = jtag2host_intfc_read(chain, device, tmc, 0, 0, (address & ~0x3),
            &value);
    value = ((value >> ((address & 0x3) << 3)) & 0xff);
    if (ret < 0) {
        if (debug) {
            fprintf(stderr, "jtag2host_intfc_read() failed (%d).\n",
                    jtag_errno);
        }
        value = 0xff;
    }
    return value;
}

static struct option options[] = {
#if defined(JTAG_RPC)
    { "host",      required_argument, NULL, 'h' },
#elif defined(JTAG_URJTAG)
    { "cable",     required_argument, NULL, 'c' },
#endif
    { "data-path", required_argument, NULL, 'r' },
    { "frequency", required_argument, NULL, 'f' },
    { NULL,        0,                 NULL,  0  }
};

static int loop_callback(char *name, char *args)
{
    int ret = ENOTSUP;
    signal_t trst;

    if (strcasecmp(name, "tap") == 0) {
        if (!args || (strlen(args) == 0))
            fprintf(stderr, "%s: expects an argument.\n", name);
        else {
            if (strcasecmp(args, "suspend") == 0) {
                SIGNAL(trst) = SIGNAL_LO;
                jtag_tap_set_trst(chain->tap_state, trst);
            } else if (strcasecmp(args, "resume") == 0) {
                SIGNAL(trst) = SIGNAL_HIGH;
                jtag_tap_set_trst(chain->tap_state, trst);
                sprintf(init_args, PhysAddr_FMT, lwBar0);
                init(init_args);
            } else
                fprintf(stderr, "%s: not a valid argument.\n", args);
        }
        ret = 0;
    }

    return ret;
}

static int parse_integer(const char *str, uint32_t *value)
{
    char *endptr, *nptr = (char *)str;

    if (*nptr != '\0') {
        *value = strtoul(nptr, &endptr, 0);
        if (*endptr == '\0')
            return 0;
    }

    return -EILWAL;
}

int main(int argc, char **argv)
{
#if defined(JTAG_RPC)
    char *default_host = "127.0.0.1";
#elif defined(JTAG_URJTAG)
    char *default_cable = "e845";
#endif
    char *data_path = NULL;
    uint32_t tck_frequency = 0;
    void *arg = NULL;
    jtag_generic_device_t *generic_device;
    char *optstr, *endptr;
    int i, c, ret;

    if (getelw("JTAG_DEBUG"))
        debug = 1;

    optstr = "";
#if defined(JTAG_RPC)
    optstr = "f:h:r:";
#elif defined(JTAG_URJTAG)
    optstr = "c:f:r:";
#endif
    while (1) {
        c = getopt_long(argc, argv, optstr, options, NULL);
        if (c < 0)
            break;

        switch (c) {
            case 'f':
                ret = parse_integer(optarg, &tck_frequency);
                if ((ret < 0) ||
                        (tck_frequency == 0)) {
                    fprintf(stderr, "Bad TCK frequency.\n");
                    exit(1);
                }
                break;
            case 'c':
            case 'h':
                arg = optarg;
                break;
            case 'r':
                data_path = optarg;
                break;
            default:
                fprintf(stderr, "Bad command line option.\n");
                exit(1);
        }
    }

    argc -= optind;

    if (!data_path)
        data_path = getelw("JTAG_DATA_PATH");

#if defined(JTAG_RPC)
    if (!arg)
        arg = getelw("JTAG_HOST");
    if (!arg)
        arg = default_host;
#elif defined(JTAG_URJTAG)
    if (!arg)
        arg = default_cable;
#endif

    if (argc > 1) {
        fprintf(stderr, "Bad command line.\n");
        exit(1);
    } else if (argc == 1) {
        lwBar0 = strtoul(argv[optind], &endptr, 0);
        if (*endptr != '\0') {
            fprintf(stderr, "Bad BAR0 address.\n");
            exit(1);
        }
    }

    chain = calloc(sizeof(jtag_chain_t), 1);
    if (!chain) {
        fprintf(stderr, "Failed to allocate chain (%d).\n",
                errno);
        exit(1);
    }

    ret = jtag_tap_init(arg, &chain->tap_state);
    if (ret < 0) {
        fprintf(stderr, "Failed to initialize TAP (%d).\n",
                jtag_errno);
        exit(1);
    }

    ret = jtag_chain_interrogate(chain);
    if (ret < 0) {
        fprintf(stderr, "Failed to interrogate chain (%d).\n",
                jtag_errno);
        exit(1);
    }
    if (chain->length == 0) {
        fprintf(stderr, "No devices in chain.\n");
        exit(1);
    }

    for (i = 0; i < chain->length; i++) {
        generic_device = &chain->devices[i];
        ret = jtag_device_alloc(generic_device, &devices[i]);
        if (ret < 0) {
            fprintf(stderr, "Failed to allocate device.\n");
            exit(1);
        }
    }

    for (i = 0; i < chain->length; i++) {
        device = devices[i];
        ret = jtag_data_configure_device(data_path, device);
        if (ret == ENOTSUP) {
            fprintf(stderr, "Unknown device.\n");
            exit(1);
        } else if (ret < 0) {
            fprintf(stderr, "Failed to configure device.\n");
            exit(1);
        }
        jtag_device_sync_registers(device);
    }

    if (tck_frequency == 0) {
        tck_frequency = devices[0]->tck_frequency;
        for (i = 1; i < chain->length; i++) {
            device = devices[i];
            tck_frequency = min(device->tck_frequency, tck_frequency);
        }
    }

    ret = jtag_tap_set_frequency(chain->tap_state, tck_frequency);
    if (ret < 0) {
        fprintf(stderr, "Failed to set TCK frequency (%s).\n",
                strerror(jtag_errno));
        exit(1);
    }

    ret = jtag_private_reconfigure_chain(chain, devices);
    if (ret < 0) {
        fprintf(stderr, "Failed to reconfigure chain.\n");
        exit(1);
    }

    sprintf(init_args, PhysAddr_FMT, lwBar0);
    init(init_args);

    main_loop(loop_callback);

    jtag_tap_shutdown(chain->tap_state);
    osDestroyHal();
    free(chain);

    return 0;
}

void initLwWatch(void)
{
    int i, j, ret;
    uint32_t tmc_index;

    for (tmc = NULL, i = 0; i < chain->length; i++) {
        device = devices[i];

        tmc_index = (device->chain_length - 1);
        tmc = &device->tmcs[tmc_index];

        ret = jtag2host_intfc_enable_access(chain, device, tmc);
        if (ret < 0) {
            fprintf(stderr,
                    "Failed to unlock JTAG2HOST_INTFC instruction.\n");
            return;
        }
        if (lwBar0 == 0) {
            lwBar0 = JTAG_RD32(0x88010);
            break;
        } else if (lwBar0 == JTAG_RD32(0x88010))
            break;

        tmc = NULL;
    }

    if ((lwBar0 == 0) || (tmc == NULL)) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }

    ret = jtag2host_intfc_enable_access(chain, device, tmc);
    if (ret < 0) {
        fprintf(stderr, "Failed to unlock JTAG.\n");
        return;
    }
    if ((JTAG_RD32(0x88000) & 0xffff) != 0x10de) {
        fprintf(stderr, "BAR0 address invalid; call 'init <BAR0>' to begin.\n");
        tmc = NULL;
        return;
    }

    osInit();
}

LW_STATUS readPhysicalMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *read)
{
    uint64_t i;

    if ((buffer == NULL) || (read == NULL))
        return LW_ERR_GENERIC;

    if (!IS_IN_BAR0(address)) {
        fprintf(stderr, "STUB: readPhysicalMem()\n");
        return LW_ERR_GENERIC;
    }

    *read = 0;
    for (i = 0; i < bytes; i++) {
        *((uint8_t *)buffer + i) = JTAG_RD08((address - lwBar0) + i);
        (*read)++;
    }

    return LW_OK;
}

LW_STATUS writePhysicalMem(LwU64 address, void *buffer, LwU64 bytes, LwU64 *written)
{
    uint64_t i;

    if ((buffer == NULL) || (written == NULL))
        return LW_ERR_GENERIC;

    if (!IS_IN_BAR0(address)) {
        fprintf(stderr, "STUB: writePhysicalMem()\n");
        return LW_ERR_GENERIC;
    }

    *written = 0;
    for (i = 0; i < bytes; i++) {
        JTAG_WR08(((address - lwBar0) + i), *((uint8_t *)buffer + i));
        (*written)++;
    }

    return LW_OK;
}

LwU32 osRegRd32(PhysAddr offset)
{
    if ((lwBar0 == 0) || (tmc == NULL)) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_REG_RD32(offset);
}

void osRegWr32(PhysAddr offset, LwU32 data)
{
    if ((lwBar0 == 0) || (tmc == NULL)) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    GPU_REG_WR32(offset, data);
}

LwU8 osRegRd08(PhysAddr offset)
{
    if ((lwBar0 == 0) || (tmc == NULL)) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_REG_RD08(offset);
}

void osRegWr08(PhysAddr offset, LwU8 data)
{
    if ((lwBar0 == 0) || (tmc == NULL)) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    GPU_REG_WR08(offset, data);
}

LwU32 RD_PHYS32(PhysAddr address)
{
    if (IS_IN_BAR0(address))
        return JTAG_RD32(address - lwBar0);
    fprintf(stderr, "STUB: RD_PHYS32()\n");
    return 0;
}

void WR_PHYS32(PhysAddr address, LwU32 data)
{
    if (IS_IN_BAR0(address)) {
        JTAG_WR32((address - lwBar0), data);
        return;
    }
    fprintf(stderr, "STUB: WR_PHYS32()\n");
}

LwU32 FB_RD32(LwU32 offset)
{
    if (lwBar1 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    fprintf(stderr, "STUB: FB_RD32()\n");
    return 0;
}

LwU32 FB_RD32_64(LwU64 offset)
{
    if ((lwBar1 & ~0xffffffffULL) != 0) {
        fprintf(stderr, "FB_RD32_64(): offset 0x%llx too large.\n", offset);
        return 0;
    }
    return FB_RD32((LwU32)offset);
}

void FB_WR32(LwU32 offset, LwU32 data)
{
    if (lwBar1 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    fprintf(stderr, "STUB: FB_WR32()\n");
}

LwU8 SYSMEM_RD08(LwU64 address)
{
    if (IS_IN_BAR0(address))
        return JTAG_RD08(address - lwBar0);
    fprintf(stderr, "STUB: SYSMEM_RD08()\n");
    return 0;
}

LwU32 SYSMEM_RD32(LwU64 address)
{
    if (IS_IN_BAR0(address))
        return JTAG_RD32(address - lwBar0);
    fprintf(stderr, "STUB: SYSMEM_RD32()\n");
    return 0;
}

void SYSMEM_WR32(LwU64 address, LwU32 data)
{
    if (IS_IN_BAR0(address)) {
        JTAG_WR32((address - lwBar0), data);
        return;
    }
    fprintf(stderr, "STUB: SYSMEM_WR32()\n");
}

LW_STATUS osPciRead08(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU8 *value, LwU32 offset)
{
    if (!value)
        return LW_ERR_GENERIC;
    fprintf(stderr, "STUB: osPciRead08()\n");
    return LW_ERR_GENERIC;
}

LW_STATUS osPciRead16(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU16 *value, LwU32 offset)
{
    if (!value)
        return LW_ERR_GENERIC;
    fprintf(stderr, "STUB: osPciRead16()\n");
    return LW_ERR_GENERIC;
}

LW_STATUS osPciRead32(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU32 *value, LwU32 offset)
{
    if (!value)
        return LW_ERR_GENERIC;
    fprintf(stderr, "STUB: osPciRead32()\n");
    return LW_ERR_GENERIC;
}

LW_STATUS osPciWrite32(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU32 Data, LwU32 Offset)
{
    fprintf(stderr, "STUB: osPciWrite32()\n");
    return LW_ERR_GENERIC;
}

LW_STATUS osPciFindDevices(LwU16 DeviceId, LwU16 VendorId, osPciCallback callback)
{
    fprintf(stderr, "STUB: osPciFindDevices()\n");
    return LW_ERR_GENERIC;
}

LW_STATUS osPciFindDevicesByClass(LwU32 classCode, osPciCallback callback)
{
    fprintf(stderr, "STUB: osPciFindDevicesByClass()\n");
    return LW_ERR_GENERIC;
}

LW_STATUS osPciGetBarInfo
(
    LwU16 DomainNumber,
    LwU8 BusNumber,
    LwU8 Device,
    LwU8 Function,
    LwU8 BarIndex,
    LwU64 *BaseAddr,
    LwU64 *BarSize)
{
    return LW_ERR_GENERIC;
}

LW_STATUS osMapDeviceMemory
(
    LwU64 BaseAddr,
    LwU64 Size,
    MemProtFlags prot,
    void **ppBar
)
{
    return LW_ERR_GENERIC;
}

LW_STATUS osUnMapDeviceMemory(void *pBar, LwU64 BarSize)
{
    return LW_ERR_GENERIC;
}

LwU32 GPU_REG_RD32_DIRECT(PhysAddr reg)
{
    return JTAG_RD32(reg);
}

void GPU_REG_WR32_DIRECT(PhysAddr reg, LwU32 value)
{
    JTAG_WR32(lwBar0 + (reg), value);
}
