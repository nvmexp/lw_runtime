/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "error_lwswitch.h"
#include "pmgr_lwswitch.h"

void
lwswitch_i2c_init
(
    lwswitch_device *device
)
{
    PLWSWITCH_OBJI2C pI2c = lwswitch_os_malloc(sizeof(struct LWSWITCH_OBJI2C));
    lwswitch_os_memset(pI2c, 0, sizeof(struct LWSWITCH_OBJI2C));
    device->pI2c = pI2c;
}

void
lwswitch_i2c_destroy
(
    lwswitch_device *device
)
{
    if (device->pI2c == NULL)
        return;

    lwswitch_os_free(device->pI2c);
    device->pI2c = NULL;
}

/*! @brief Set up a port to use a PMGR implementation.
 *
 *  @param[in]  device          LwSwitch device
 *  @param[in]  port            The port identifier for the bus.
 */
void
_lwswitch_i2c_set_port_pmgr
(
    lwswitch_device *device,
    LwU32   port
)
{
    LwU32 i;
    LwU32 device_allow_list_size;
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *device_allow_list;
    PLWSWITCH_OBJI2C pI2c = device->pI2c;

    LWSWITCH_ASSERT(port < LWSWITCH_MAX_I2C_PORTS);

    pI2c->PortInfo[port] = FLD_SET_DRF(_I2C, _PORTINFO, _DEFINED, _PRESENT, pI2c->PortInfo[port]);
    pI2c->Ports[port].defaultSpeedMode = LWSWITCH_I2C_SPEED_MODE_100KHZ;

    device_allow_list = pI2c->i2c_allow_list;
    device_allow_list_size = pI2c->i2c_allow_list_size;

    for (i = 0; i < device_allow_list_size; i++)
    {
        if (port == device_allow_list[i].i2cPortLogical)
        {
            pI2c->PortInfo[port] = FLD_SET_DRF(_I2C, _PORTINFO,
                                               _ACCESS_ALLOWED, _TRUE,
                                               pI2c->PortInfo[port]);
            break;
        }
    }
}

