//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// osWin.h
// Windows OS dependent routines...
//*****************************************************

#ifndef _OSWIN_H_
#define _OSWIN_H_

//
// includes
//
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "lwwatch.h"
#include <lwtypes.h>
#include "lwutil.h"

//
// defines
//

#define MaxLWDevices 128
#define LwU64_FMT   "0x%016I64x"
#define LwU40_FMT   "0x%010I64x"

//
// globals
//
extern ULONG OSMajorVersion;
extern ULONG OSMinorVersion;

#define OS_IS_VISTA_OR_HIGHER(osmv)         (osmv >= 5000)
#define OS_IS_WINXP_OR_HIGHER(osmv)         (osmv >= 2600)
#define OS_IS_WIN2K_OR_HIGHER(osmv)         (osmv >= 2195)

extern U032    CPUFreq;

//
// Copied from NTDDK.H
//
typedef struct _PCI_SLOT_NUMBER {
    union {
        struct {
            ULONG  DeviceNumber:5;
            ULONG  FunctionNumber:3;
            ULONG  Reserved:24;
        } bits;
        ULONG  AsULONG;
    } u;
} PCI_SLOT_NUMBER, *PPCI_SLOT_NUMBER;

//
// Copied from NTDDK.H
//
typedef enum _BUS_DATA_TYPE {
    ConfigurationSpaceUndefined = -1,
    Cmos,
    EisaConfiguration,
    Pos,
    CbusConfiguration,
    PCIConfiguration,
    VMEConfiguration,
    NuBusConfiguration,
    PCMCIAConfiguration,
    MPIConfiguration,
    MPSAConfiguration,
    PNPISAConfiguration,
    SgiInternalConfiguration,
    MaximumBusDataType
} BUS_DATA_TYPE, *PBUS_DATA_TYPE;

__inline VOID
GetBusData(
           ULONG             BusDataType,
           ULONG             BusNumber,
           ULONG             Device,
           ULONG             Function,
           PVOID             Buffer,
           ULONG             Offset,
           ULONG             Length
           )
{
    PBUSDATA pbd;
    PCI_SLOT_NUMBER slot;
    slot.u.AsULONG = 0x0;    // make sure all fields are initialized
    slot.u.bits.DeviceNumber   = Device;
    slot.u.bits.FunctionNumber = Function;

    pbd = (PBUSDATA)LocalAlloc(LPTR, sizeof(*pbd) );
    if (pbd) {
        ZeroMemory( Buffer, Length );
        ZeroMemory( pbd, sizeof(*pbd) );
        pbd->BusDataType = BusDataType;
        pbd->BusNumber = BusNumber;
        pbd->SlotNumber = slot.u.AsULONG;
        pbd->Buffer = Buffer;
        pbd->Offset = Offset;
        pbd->Length = Length;
        Ioctl( IG_GET_BUS_DATA, (PVOID)pbd, sizeof(*pbd) );
        LocalFree( pbd );
    }
}

//
// I/O structures
//
typedef struct
{
  HANDLE h;
  int state;
  char *dirname;
} DIR;

struct dirent
{
  long d_ino;               /* inode number */
  long d_off;               /* offset to this dirent */
  unsigned short d_reclen;  /* length of this d_name */
  char d_name[MAX_PATH];    /* file name (null-terminated) */
};

enum { DIR_INIT, DIR_READING, DIR_CLOSED };

//
// Windows specific routines
//
VOID    initLwWatch(void);
BOOL    FindLWDevice(void);
U032    readPhysicalMem(ULONG64 address, PVOID buf, ULONG size, U032 *sizer);
U032    writePhysicalMem(ULONG64 address, PVOID buf, ULONG size, PULONG sizew);
U032    readVirtMem(ULONG64 address, PVOID buf, ULONG size, PULONG sizer);
U032    writeVirtMem(ULONG64 address, PVOID buf, ULONG size, PULONG sizew);
LwU64   virtToPhys(LwU64 virtAddr, LwU32 pid);
LwU64   physToVirt(LwU64 physAddr, LwU32 flags);
VOID    ScanLWTopology(U032 PcieConfigSpaceBase);

//
// I/O routines
//
DIR*    opendir(const char *dirname);
struct  dirent* readdir(DIR *d);
int     closedir(DIR *d);


int strcasecmp(const char *s0, const char *s1);
int strncasecmp(const char *s0, const char *s1, int n);

#endif // _OSWIN_H_
