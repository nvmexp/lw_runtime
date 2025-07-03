/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include "mobile.h"
#include "disp.h"
#include <chip.h>

#undef NDEBUG
#include <assert.h>

#define LWW_MOBILE_USAGE "usage: %s [BAR0] [BAR0_size] [BAR1] [BAR1_size] [chipId(0x4000 or 0x2100)] [ip4_addr] [port] [os]\n"

static int sockFd = -1;
static char hostname[256];
static char portStr[256];
extern char osname[256];

static char ip4Addr[INET_ADDRSTRLEN];
lww_main lwwatch;

static int initRegops()
{
    struct sockaddr_in *saddr;
    int ret;
    struct addrinfo hints = {0}, *addr;

    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    ret = getaddrinfo(hostname, portStr, &hints, &addr);
    if (ret < 0) {
        fprintf(stderr, "failed to resolve hostname %s, error %d (%s)\n", hostname, errno, strerror(errno));
        return -1;
    }

    saddr = (struct sockaddr_in*)addr->ai_addr;

    sockFd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockFd < 0) {
        fprintf(stderr, "failed to create socket, error %d (%s)\n", errno, strerror(errno));
        return -1;
    }

    printf("Created socket\n");

    if (inet_ntop(addr->ai_family, &saddr->sin_addr, ip4Addr, sizeof ip4Addr) == NULL) {
        fprintf(stderr, "failed to colwert address to binary, error %d (%s)\n", errno, strerror(errno));
        return -1;
    }

    printf("Connecting to %s (%s), port %d\n", hostname, ip4Addr, atoi(portStr));

    ret = connect(sockFd, (struct sockaddr*)saddr, sizeof (struct sockaddr));
    if (ret < 0) {
        fprintf(stderr, "failed to connect to server, error %d (%s)\n", errno, strerror(errno));
        return -1;
    }

    freeaddrinfo(addr);

    printf("Connected\n");

    return 0;
}

static void deInitRegops()
{
    close(sockFd);
}

int mobile_init_args(int argc, char** argv) {
    if (argc != 9) {
        fprintf(stderr, LWW_MOBILE_USAGE, argv[0]);
        return 1;
    } else {
        lwBar0 = strtoul(argv[1], NULL, 0);
        lwBar1 = strtoul(argv[3], NULL, 0);

        lwwatch.GPU_BARs[0].base_address = lwBar0;
        lwwatch.GPU_BARs[0].size = strtoul(argv[2], NULL, 0);

        lwwatch.GPU_BARs[1].base_address = lwBar1;
        lwwatch.GPU_BARs[1].size = strtoul(argv[4], NULL, 0);

        isTegraHack = strtoul(argv[5], NULL, 0);

        strncpy(hostname, argv[6], sizeof hostname);
        strncpy(portStr, argv[7], sizeof portStr);
        strncpy(osname, argv[8], sizeof osname);

        return 0;
    }
}

int mobile_regOps(LwU64 offset, uint32_t *value, LwBool isRead, int devid, int alignment)
{
    int ret = 0, bytes;
    RegOp op = {0};
    char *buf;

    op.devId = devid;
    op.offset = offset;
    op.value = isRead ? 0: *value;
    op.isRead = isRead ? 1: 0;
    op.alignment = alignment;

    /* Write the request */
    for (buf = (char*)&op, bytes = 0; bytes < sizeof op; bytes += ret) {
        ret = write(sockFd, buf + bytes, sizeof op - bytes);
        if (ret == 0) {
            fprintf(stderr, "received EOF when writing request\n");
            goto close;
            break;
        }
        else if (ret < 0) {
            if (errno == EINTR)
                continue;
            else {
                fprintf(stderr, "failed to write packet (%s)\n",
                        strerror(errno));
                return ret;
            }
        }
    }

    /* Read the response */
    for (buf = (char*)&op, bytes = 0; bytes < sizeof op; bytes += ret) {
        ret = read(sockFd, buf + bytes, sizeof op - bytes);
        if (ret == 0) {
            fprintf(stderr, "received EOF when reading response\n");
            goto close;
            break;
        }
        else if (ret < 0) {
            if (errno == EINTR)
                continue;
            else {
                fprintf(stderr, "failed to read packet (%s)\n",
                        strerror(errno));
                return ret;
            }
        }
    }

close:
    *value = op.value;
    if (op.status != 0)
        fprintf(stderr, "regops status returned error. offset: %" PRIx64 ", devid: %d\n", (uint64_t)offset, devid);

    return 0;
}

/* Always add at least this many bytes when extending the buffer.  */
#define MIN_CHUNK 64

/* Read up to (and including) a delimiter DELIM1 from STREAM into *LINEPTR
   + OFFSET (and NUL-terminate it).  If DELIM2 is non-zero, then read up
   and including the first oclwrrence of DELIM1 or DELIM2.  *LINEPTR is
   a pointer returned from malloc (or NULL), pointing to *N characters of
   space.  It is realloc'd as necessary.  Return the number of characters
   read (not including the NUL terminator), or -1 on error or EOF.  */

static int
mobile_getstr( char **lineptr, size_t* n, FILE* stream, int delim1, int delim2,
               size_t offset)
{
    int numCharactersRead = 0;
    while( !feof(stream) ) {
        char c;
        c = (char) getc( stream );
        if (c == EOF)
            return -1;

        while(numCharactersRead + offset >= (*n)) {
            // resize
            (*n) += MIN_CHUNK;
            (*lineptr) = realloc( (*lineptr), (*n) );
            if ( !(*lineptr) ) return -1;
        }

        (*lineptr)[ (numCharactersRead++) + offset] = c;

        if (c == delim1 || (delim2 && (c == delim2))) {
            // found
            (*lineptr)[ (numCharactersRead) + offset] = '\0';
            return numCharactersRead;
        }
    }
    return -1;
}

ssize_t mobile_getline(char** lineptr, size_t* n, FILE* stream) {
    return mobile_getstr(lineptr, n, stream, '\n', 0, 0);
}

void initLwWatch() {
    int prot = PROT_READ | PROT_WRITE, flags = MAP_SHARED, ret;
    uint16_t command;
    uint32_t class, low;
    int i, j, k;
    LwBool is_virtual = LW_FALSE;

    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }

    ret = initRegops();
    if (ret < 0) {
        exit(1);

    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    is_virtual = isVirtual();
    if (is_virtual)
    {
        // Set LwwatchMode as we need some PF registers before this can be set in exts cmds
        setLwwatchMode(LW_TRUE);
    }

    if (osInit() != LW_OK) {
        fprintf(stderr, "WARNING: Operating System specific initializations failed\n");
    }

    // if we're running in a VGPU (i.e. VGX guest) need to set LwWatch mode bits
    if (is_virtual)
    {
        setLwwatchMode(LW_FALSE);
    }
}

int mem_ops(LwU64 address, uint32_t *pValue, LwBool isRead, int devid, int alignment)
{
    if (devid == DEV_LWGPU) {
        /*
         * Access to GPU display aperture needs to be translated to SOC aperture.
         * Ignore translation if disp HAL is not configured yet.
         */
        LwU64 translatedAddress = address;
        if (pDisp[indexGpu].dispTranslateGpuRegAddrToSoc) {
            translatedAddress = pDisp[indexGpu].dispTranslateGpuRegAddrToSoc(address);
        }

        if (translatedAddress == address) {
            address = address - lwBar0;
        } else {
            address = translatedAddress;
            devid = DEV_NON_LWGPU;
        }
    }

    return mobile_regOps(address, pValue, isRead, devid, alignment);
}

void MMIO_WR32(uint64_t address, uint32_t value)
{
    const int alignment = sizeof(int);
    int devId;

    if ((address % alignment) != 0) {
        fprintf(stderr, "MMIO_WR32(): 0x%" PRIx64 " is unaligned!\n", address);
        return;
    }
    devId = IS_IN_MEM_REGION(lwwatch.GPU_BARs[0], address) ? DEV_LWGPU : DEV_NON_LWGPU;

    mem_ops(address, &value, LW_FALSE, devId, alignment);
}

uint32_t MMIO_RD32(uint64_t address)
{
    const int alignment = sizeof(int);

    int  devId;
    uint32_t value;

    if ((address % alignment) != 0) {
        fprintf(stderr, "MMIO_RD32(): 0x%" PRIx64 " is unaligned!\n", address);
        return 0;
    }

    devId = IS_IN_MEM_REGION(lwwatch.GPU_BARs[0], address) ? DEV_LWGPU : DEV_NON_LWGPU;

    mem_ops(address, &value, LW_TRUE, devId, alignment);

    return value;
}

void MMIO_WR08(uint64_t address, uint8_t value)
{
    int devId;

    devId = IS_IN_MEM_REGION(lwwatch.GPU_BARs[0], address) ? DEV_LWGPU : DEV_NON_LWGPU;

    uint32_t value32 = value;
    mem_ops(address, &value32, LW_FALSE, devId, sizeof(uint8_t));
}

uint8_t MMIO_RD08(uint64_t address)
{
    int  devId;
    uint32_t value;

    devId = IS_IN_MEM_REGION(lwwatch.GPU_BARs[0], address) ? DEV_LWGPU : DEV_NON_LWGPU;

    mem_ops(address, &value, LW_TRUE, devId, sizeof(uint8_t));

    return (uint8_t)value;
}

LW_STATUS osPciRead08(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU8 *value, LwU32 offset)
{
    /* An invalid operation - shouldn't be called */
    assert(0);
    return LW_OK;
}

LW_STATUS osPciRead16(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU16 *value, LwU32 offset)
{
    /* An invalid operation - shouldn't be called */
    assert(0);
    return LW_OK;
}

LW_STATUS osPciRead32(LwU16 domain, LwU8 bus, LwU8 device, LwU8 function, LwU32 *value, LwU32 offset)
{
    /* An invalid operation - shouldn't be called */
    assert(0);
    return LW_OK;
}

int main(int argc, char** argv) {
    int chk;

    memset(&(lwwatch.RAM_regions), 0, sizeof(lwwatch.RAM_regions));
    memset(&(lwwatch.current), 0, sizeof(lwwatch.current));
    memset(&(lwwatch.GPU_BARs), 0, sizeof(lwwatch.GPU_BARs));

    if (find_RAM_regions( &mobile_getline ) != 0) {
        fprintf(stderr, "Unable to locate system RAM regions!\n");
        return 1;
    }

    if ((chk = mobile_init_args(argc, argv)))
        return chk;

    sprintf(lwwatch.arguments, PhysAddr_FMT, lwBar0);

    init(lwwatch.arguments);

    main_loop(NULL);

    return 0;
}

/* Below functions are implemented for mobile just for
 * compatibility towards lwwatch tool as these are
 * maningful for dGPU.
 */
void mobile_do_nothing()
{
    /* An invalid operation - shaouln't be called */
    assert(0);
}

static LW_STATUS readWritePhysicalMem(
    LwU64 address,
    void *buffer,
    LwU64 bytes,
    LwU64 *bytes_accessed,
    LwBool is_read)
{
    uint64_t i;
    uint32_t value;

    const uint32_t alignment =
            (((address % 4) == 0) && ((bytes % 4) == 0)) ?
            sizeof(uint32_t) : sizeof(uint8_t);

    const int devid =
            (IS_IN_MEM_REGION(lwwatch.GPU_BARs[0], address)) ?
            DEV_LWGPU : DEV_NON_LWGPU;

    *bytes_accessed = 0;
    const LwU64 num_read = bytes / alignment;
    for (i = 0; i < num_read; i++) {
        const LwU64 offset = i * alignment;
        const int stat = mem_ops(
                address + offset,
                (uint32_t *)((unsigned char*)buffer + offset),
                is_read,
                devid,
                alignment);
        if (stat != 0) {
            return LW_ERR_GENERIC;
        }
        (*bytes_accessed)+=alignment;
    }

    return LW_OK;
}

LW_STATUS readPhysicalMem(
    LwU64 address,
    void *buffer,
    LwU64 bytes,
    LwU64 *read)
{
    return readWritePhysicalMem(address, buffer, bytes, read, LW_TRUE);
}

LW_STATUS writePhysicalMem(
    LwU64 address,
    void *buffer,
    LwU64 bytes,
    LwU64 *written)
{
    return readWritePhysicalMem(address, buffer, bytes, written, LW_FALSE);
}

int find_RAM_regions( ssize_t (*lw_getline)(char**, size_t*, FILE*) )
{
    // Skip this as we don't need to query the lwwatch host memory.
    return 0;
}

LwU32 GPU_REG_RD32_DIRECT(PhysAddr offset)
{
    return MMIO_RD32(lwBar0 + (offset));
}

void GPU_REG_WR32_DIRECT(PhysAddr offset, LwU32 value)
{
    MMIO_WR32((lwBar0 + (offset)), (value));
}


LwU32 osRegRd32(PhysAddr offset)
{
    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_REG_RD32(offset);
}

void osRegWr32(PhysAddr offset, LwU32 data)
{
    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    GPU_REG_WR32(offset, data);
}

LwU8 osRegRd08(PhysAddr offset)
{
    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_REG_RD08(offset);
}

void osRegWr08(PhysAddr offset, LwU8 data)
{
    if (lwBar0 == 0) {
        fprintf(stderr, "BAR0 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    GPU_REG_WR08(offset, data);
}

LwU32 RD_PHYS32(PhysAddr address)
{
    return MMIO_RD32(address);
}

void WR_PHYS32(PhysAddr address, LwU32 data)
{
    MMIO_WR32(address, data);
}

LwU32 FB_RD32(LwU32 offset)
{
    if (lwBar1 == 0) {
        fprintf(stderr, "BAR1 address unknown; call 'init <BAR0>' to begin.\n");
        return 0;
    }
    return GPU_FB_RD32(offset);
}

LwU32 FB_RD32_64(LwU64 offset)
{
    if ((lwBar1 & ~0xffffffffULL) != 0) {
        fprintf(stderr, "FB_RD32_64(): offset 0x%llx too large!\n", offset);
        return 0;
    }
    return FB_RD32((LwU32)offset);
}

void FB_WR32(LwU32 offset, LwU32 data)
{
    if (lwBar1 == 0) {
        fprintf(stderr, "BAR1 address unknown; call 'init <BAR0>' to begin.\n");
        return;
    }
    GPU_FB_WR32(offset, data);
}

LwU8 SYSMEM_RD08(LwU64 address)
{
    if ((address & ~0xffffffffULL) != 0) {
        fprintf(stderr, "SYSMEM_RD08(): address 0x%llx too large!\n", address);
        return 0;
    }
    return MMIO_RD08(address);
}


LwU32 SYSMEM_RD32(LwU64 address)
{
    if ((address & ~0xffffffffULL) != 0) {
        fprintf(stderr, "SYSMEM_RD32(): address 0x%llx too large!\n", address);
        return 0;
    }
    return MMIO_RD32(address);
}

void SYSMEM_WR32(LwU64 address, LwU32 data)
{
    if ((address & ~0xffffffffULL) != 0) {
        fprintf(stderr, "SYSMEM_WR32(): address 0x%llx too large!\n", address);
        return;
    }
    MMIO_WR32(address, data);
}

//-----------------------------------------------------
// osPciFindDevices
//
//-----------------------------------------------------
LW_STATUS osPciFindDevices(LwU16 DeviceId, LwU16 VendorId, osPciCallback callback)
{
    return LW_ERR_GENERIC;
}

//-----------------------------------------------------
// osPciWrite32
//
//-----------------------------------------------------
LwU32 osPciWrite32(LwU16 DomainNumber, LwU8 BusNumber, LwU8 Device, LwU8 Function, LwU32 Data, LwU32 Offset)
{
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
