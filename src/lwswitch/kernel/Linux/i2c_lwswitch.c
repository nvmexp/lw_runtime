/*******************************************************************************
    Copyright (c) 2021 LWPU Corporation

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

#include "linux_lwswitch.h"
#include <linux/i2c.h>

#if defined(CONFIG_I2C) || defined(CONFIG_I2C_MODULE)

#define LWSWITCH_I2C_GET_PARENT(adapter) \
            (LWSWITCH_DEV *)pci_get_drvdata(to_pci_dev((adapter)->dev.parent));

#define LWSWITCH_I2C_GET_ALGO_DATA(adapter) \
            (lwswitch_i2c_algo_data *)(adapter)->algo_data;

typedef struct
{
    LwU32 port;
} lwswitch_i2c_algo_data;

static int
lwswitch_i2c_algo_master_xfer
(
    struct i2c_adapter *adapter,
    struct i2c_msg msgs[],
    int num
)
{
    int rc;
    int i;
    LwU32 port;
    LwlStatus status = LWL_SUCCESS;
    lwswitch_i2c_algo_data *i2c_algo_data;
    LWSWITCH_DEV *lwswitch_dev;
    const unsigned int supported_i2c_flags = I2C_M_RD
#if defined (I2C_M_DMA_SAFE)
    | I2C_M_DMA_SAFE
#endif
    ;

    lwswitch_dev = LWSWITCH_I2C_GET_PARENT(adapter);
    if (lwswitch_dev == NULL)
    {
        return -ENODEV;
    }

    rc = mutex_lock_interruptible(&lwswitch_dev->device_mutex);
    if (rc)
    {
        return rc;
    }

    if (lwswitch_dev->unusable)
    {
        printk(KERN_INFO "%s: a stale fd detected\n", lwswitch_dev->name);
        status = LWL_ERR_ILWALID_STATE;
        goto lwswitch_i2c_algo_master_xfer_exit;
    }

    i2c_algo_data = LWSWITCH_I2C_GET_ALGO_DATA(adapter);
    if (i2c_algo_data == NULL)
    {
        status = LWL_ERR_ILWALID_STATE;
        goto lwswitch_i2c_algo_master_xfer_exit;
    }

    port = i2c_algo_data->port;

    for (i = 0; (i < num) && (status == LWL_SUCCESS); i++)
    {
        if (msgs[i].flags & ~supported_i2c_flags)
        {
            status = LWL_ERR_NOT_SUPPORTED;
        }
        else
        {
            status = lwswitch_lib_i2c_transfer(lwswitch_dev->lib_device, port,
                                               (msgs[i].flags & I2C_M_RD) ?
                                                   LWSWITCH_I2C_CMD_READ : LWSWITCH_I2C_CMD_WRITE,
                                               (LwU8)(msgs[i].addr & 0x7f), 0,
                                               (LwU32)(msgs[i].len & 0xffffUL),
                                               (LwU8 *)msgs[i].buf);
        }
    }

lwswitch_i2c_algo_master_xfer_exit:
    mutex_unlock(&lwswitch_dev->device_mutex);

    rc = lwswitch_map_status(status);
    return (rc == 0) ? num : rc;
}

static int
lwswitch_i2c_algo_smbus_xfer
(
    struct i2c_adapter *adapter,
    u16 addr,
    unsigned short flags,
    char read_write,
    u8 command,
    int protocol,
    union i2c_smbus_data *data
)
{
    int rc = -EIO;
    LwU32 port;
    LwU8 cmd;
    LwU32 len;
    LwU8 type;
    LwU8 *xfer_data;
    LwlStatus status = LWL_SUCCESS;
    lwswitch_i2c_algo_data *i2c_algo_data;
    LWSWITCH_DEV *lwswitch_dev;

    lwswitch_dev = LWSWITCH_I2C_GET_PARENT(adapter);
    if (lwswitch_dev == NULL)
    {
        return -ENODEV;
    }

    rc = mutex_lock_interruptible(&lwswitch_dev->device_mutex);
    if (rc)
    {
        return rc;
    }

    if (lwswitch_dev->unusable)
    {
        printk(KERN_INFO "%s: a stale fd detected\n", lwswitch_dev->name);
        status = LWL_ERR_ILWALID_STATE;
        goto lwswitch_i2c_algo_smbus_xfer_exit;
    }

    i2c_algo_data = LWSWITCH_I2C_GET_ALGO_DATA(adapter);
    if (i2c_algo_data == NULL)
    {
        status = LWL_ERR_ILWALID_STATE;
        goto lwswitch_i2c_algo_smbus_xfer_exit;
    }

    port = i2c_algo_data->port;

    switch (protocol)
    {
        case I2C_SMBUS_QUICK:
        {
            cmd = 0;
            len = 0;
            type = (read_write == I2C_SMBUS_READ) ?
                       LWSWITCH_I2C_CMD_SMBUS_QUICK_READ :
                       LWSWITCH_I2C_CMD_SMBUS_QUICK_WRITE;
            xfer_data = NULL;
            break;
        }
        case I2C_SMBUS_BYTE:
        {
            cmd = 0;
            len = 1;

            if (read_write == I2C_SMBUS_READ)
            {
                type = LWSWITCH_I2C_CMD_READ;
                xfer_data = (LwU8 *)&data->byte;
            }
            else
            {
                type = LWSWITCH_I2C_CMD_WRITE;
                xfer_data = &command;
            }
            break;
        }
        case I2C_SMBUS_BYTE_DATA:
        {
            cmd = (LwU8)command;
            len = 1;
            type = (read_write == I2C_SMBUS_READ) ?
                       LWSWITCH_I2C_CMD_SMBUS_READ :
                       LWSWITCH_I2C_CMD_SMBUS_WRITE;
            cmd = (LwU8)command;
            xfer_data = (LwU8 *)&data->byte;
            break;
        }
        case I2C_SMBUS_WORD_DATA:
        {
            cmd = (LwU8)command;
            len = 2;
            type = (read_write == I2C_SMBUS_READ) ?
                       LWSWITCH_I2C_CMD_SMBUS_READ :
                       LWSWITCH_I2C_CMD_SMBUS_WRITE;
            xfer_data = (LwU8 *)&data->word;
            break;
        }
        default:
        {
            status = LWL_BAD_ARGS;
            goto lwswitch_i2c_algo_smbus_xfer_exit;
        }
    }

    status = lwswitch_lib_i2c_transfer(lwswitch_dev->lib_device, port,
                                       type, (LwU8)(addr & 0x7f),
                                       cmd, len, (LwU8 *)xfer_data);

lwswitch_i2c_algo_smbus_xfer_exit:
    mutex_unlock(&lwswitch_dev->device_mutex);

    return lwswitch_map_status(status);
}

static u32 lwswitch_i2c_algo_functionality(struct i2c_adapter *adapter)
{
    return (I2C_FUNC_I2C             |
            I2C_FUNC_SMBUS_QUICK     |
            I2C_FUNC_SMBUS_BYTE      |
            I2C_FUNC_SMBUS_BYTE_DATA |
            I2C_FUNC_SMBUS_WORD_DATA);
}

static struct i2c_algorithm lwswitch_i2c_algo = {
    .master_xfer      = lwswitch_i2c_algo_master_xfer,
    .smbus_xfer       = lwswitch_i2c_algo_smbus_xfer,
    .functionality    = lwswitch_i2c_algo_functionality,
};

struct i2c_adapter lwswitch_i2c_adapter_prototype = {
    .owner             = THIS_MODULE,
    .algo              = &lwswitch_i2c_algo,
    .algo_data         = NULL,
};

struct i2c_adapter *
lwswitch_i2c_add_adapter
(
    LWSWITCH_DEV *lwswitch_dev,
    LwU32 port
)
{
    struct i2c_adapter *adapter = NULL;
    int rc = 0;
    struct pci_dev *pci_dev;
    lwswitch_i2c_algo_data *i2c_algo_data = NULL;

    if (lwswitch_dev == NULL)
    {
        printk(KERN_ERR "lwswitch_dev is NULL!\n");
        return NULL;
    }

    adapter = lwswitch_os_malloc(sizeof(struct i2c_adapter));
    if (adapter == NULL)
    {
        return NULL;
    }

    lwswitch_os_memcpy(adapter,
                       &lwswitch_i2c_adapter_prototype,
                       sizeof(struct i2c_adapter));

    i2c_algo_data = lwswitch_os_malloc(sizeof(lwswitch_i2c_algo_data));
    if (i2c_algo_data == NULL)
    {
        goto cleanup;
    }

    i2c_algo_data->port = port;
    pci_dev = lwswitch_dev->pci_dev;
    adapter->dev.parent = &pci_dev->dev;
    adapter->algo_data = (void *)i2c_algo_data;

    rc = lwswitch_os_snprintf(adapter->name,
                              sizeof(adapter->name),
                              "LWPU LWSwitch i2c adapter %u at %x:%02x.%u",
                              port,
                              LW_PCI_BUS_NUMBER(pci_dev),
                              LW_PCI_SLOT_NUMBER(pci_dev),
                              PCI_FUNC(pci_dev->devfn));
    if ((rc < 0) && (rc >= sizeof(adapter->name)))
    {
        goto cleanup;
    }

    rc = i2c_add_adapter(adapter);
    if (rc < 0)
    {
        goto cleanup;
    }

    return adapter;

cleanup:
    lwswitch_os_free(i2c_algo_data);
    lwswitch_os_free(adapter);

    return NULL;
}

void
lwswitch_i2c_del_adapter
(
    struct i2c_adapter *adapter
)
{
    if (adapter != NULL)
    {
        lwswitch_os_free(adapter->algo_data);
        i2c_del_adapter(adapter);
        lwswitch_os_free(adapter);
    }
}

#else // (defined(CONFIG_I2C) || defined(CONFIG_I2C_MODULE))

struct i2c_adapter *
lwswitch_i2c_add_adapter
(
    LWSWITCH_DEV *lwswitch_dev,
    LwU32 port
)
{
    return NULL;
}

void
lwswitch_i2c_del_adapter
(
    struct i2c_adapter *adapter
)
{
}

#endif // (defined(CONFIG_I2C) || defined(CONFIG_I2C_MODULE))
