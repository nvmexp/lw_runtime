/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _METHODPARSE_H_
#define _METHODPARSE_H_

#include "os.h"
#include "class/cl406e.h"

//
// defines
//
#define MAX_STR_LEN             256

//
// Index register count; see _parseIReg().
// Lwrrently set to 3 since (i,j,k,l) registers are non-existent.
//
#define MAX_IREG_COUNT          3

// This is the location of the class headers on the server
#define CLASS_PATH_SERVER       "\\\\lws\\depot\\class\\cl%04x.h"

// Additonal path from elwironmental variable LWW_CLASS_SDK
#define CLASS_PATH_LOCAL        "cl%04x.h"

// Additonal path from elwironmental variable LWW_MANUAL_SDK if ref file
#define MANUAL_H_PATH_LOCAL      "%s.h"

//
// All chipInfo with numbers less than 60 will generate a path with the 'lw' prefix
// Otherwise a path with be generated with the 'g' prefix
// Exceptions can be listed in the MANUAL_PATH_TABLE
// 
#define LW_TO_G_CHIPINFO         0x60

#define MANUAL_LIST \
    {               \
        "dev_bsp",\
        "dev_bus",\
        "dev_cipher",\
        "dev_dac",\
        "dev_disp",\
        "dev_dvd",\
        "dev_ext_devices",\
        "dev_fb",\
        "dev_fb_heavy",\
        "dev_fifo",\
        "dev_graphics",\
        "dev_graphics_nobundle",\
        "dev_host",\
        "dev_host_diag",\
        "dev_master",\
        "dev_media",\
        "dev_misc",\
        "dev_mpeg",\
        "dev_multichip_bridge",\
        "dev_pbs",\
        "dev_pm",\
        "dev_pmgr",\
        "dev_ram",\
        "dev_timer",\
        "dev_tremapper",\
        "dev_trim",\
        "dev_tvo",\
        "dev_vga",\
        "dev_video",\
        "dev_vp2",\
        "dev_pwr_pri",\
        "dev_vic_pri",\
        "dev_mspdec_pri",\
        "dev_msvld_pri",\
        "dev_msppp_pri",\
        "dev_msenc_pri",\
        "dev_lwenc_pri_sw",\
        "dev_lwdec_pri",\
        "dev_lwjpg_pri",\
    }

// The max number of spaces to add in order to align everything nicely.
#define ALIGNMENT_SPACES        40

//
// These are the methods that should never be parsed.
// NOTE: can also be without %03lX i.e. LW4176_NOTIFICATION_ will also work
// 
#define BLOCK_LIST_CLASS        {"LW%03lX_NOTIFICATION_",\
                                 "LW%03lX_NOTIFIERS_"}

//
// These are the methods that should never be parsed.
// NOTE: can also be without %03lX i.e. LW4176_NOTIFICATION_ will also work
// 
#define BLOCK_LIST_MANUAL       {"LW_CONFIG",\
                                 "LW_EXPROM",\
                                 "LW_UDMA"}

#define MAX_LIST_ELEMENTS       100

#define HOST_METHODS            {{LW406E_SET_OBJECT,                "LWXXXX_SET_OBJECT"},\
                                 {LW406E_SET_REFERENCE,             "LWXXXX_SET_REFERENCE"},\
                                 {LW406E_SET_CONTEXT_DMA_SEMAPHORE, "LWXXXX_SET_CONTEXT_DMA_SEMAPHORE"},\
                                 {LW406E_SEMAPHORE_OFFSET,          "LWXXXX_SEMAPHORE_OFFSET"},\
                                 {LW406E_SEMAPHORE_ACQUIRE,         "LWXXXX_SEMAPHORE_ACQUIRE"},\
                                 {LW406E_SEMAPHORE_RELEASE,         "LWXXXX_SEMAPHORE_RELEASE"},\
                                 {LW406E_QUADRO_VERIFY,             "LWXXXX_QUADRO_VERIFY"},\
                                 {LW406E_SPLIT_POINT,               "LWXXXX_SPLIT_POINT"}}


typedef struct {
    LwU32   high;
    LwU32   low;
} BoundListElement_t;

typedef struct {
    long                startFilePos;
    BoundListElement_t  boundList[MAX_LIST_ELEMENTS];
    LwU32               boundListElements;
} ManualTableElement_t;

typedef struct {
    LwU32               methodNum;
    char*               methodName;
} HostMethods_t;

typedef struct
{
    LwU32 width;
    LwU32 length;
    LwU32 indexValue;
    char c;
} ireg_t;

BOOL parseClassHeader(LwU32 classnum, LwU32 method, LwU32 data);
BOOL isValidClassHeader(LwU32 classNum);
BOOL parseManualReg(LwU32 addr, LwU32 data, BOOL isListAll);

#endif // _METHODPARSE_H_

