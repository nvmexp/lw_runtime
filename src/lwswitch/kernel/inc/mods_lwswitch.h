/*******************************************************************************
    Copyright (c) 2016-2020 LWpu Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#ifndef _MODS_LWSWITCH_H_
#define _MODS_LWSWITCH_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    LwU32   domain;
    LwU32   bus;
    LwU32   device;
    LwU32   function;
} lwswitch_mods_pci_info;

typedef struct
{
    LwU64 version;
} lwswitch_mods_bios;

/*
 * Initializes lwswitch library, preparing the driver to register
 * discovered devices into the core library.
 */
LwlStatus lwswitch_mods_lib_load(lwswitch_mods_pci_info *pAllowedDevices, LwU32 numAllowedDevices, LwU32 printLevel);

/*
 * Shuts down the lwswitch library, deregistering its devices from
 * the core and freeing core operating system accounting info.
 */
LwlStatus lwswitch_mods_lib_unload(void);

/*
 * Initialize all switch devices.
 *
 * @returns                 LWL_SUCCESS
 */
LwlStatus lwswitch_mods_initialize_all_devices(void);

LwlStatus lwswitch_mods_ctrl(UINT32 instance, LwU32 ctrlid, void *pParams, LwU32 paramSize);

LwlStatus lwswitch_mods_get_device_info(LwU32 instance, LwU32 *linkMask, struct lwlink_pci_info *pciInfo);

LwlStatus lwswitch_mods_get_bios_rom_version(LwU32 instance, lwswitch_mods_bios *retBiosInfo);

#ifdef __cplusplus
}
#endif

#endif //_MODS_LWSWITCH_H_
