/*******************************************************************************
    Copyright (c) 2013-2021 LWPU Corporation

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

#include "lwswitch.h"
#include "lwmisc.h"
#include "UtilOS.h"

#include "lr10/dev_ingress_ip.h"
#include "lr10/dev_nport_ip.h"

#ifdef __linux__

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
extern "C" {
    #include <linux/i2c-dev.h>
    #include <linux/i2c.h>
}

#define LW_ARRAY_ELEMENTS(x)       ((sizeof(x)/sizeof((x)[0])))

#define LWSWITCH_VALID_I2C_DEVICE_7BIT_ADDRESS_START 0x08
#define LWSWITCH_VALID_I2C_DEVICE_7BIT_ADDRESS_STOP  0x77
#define LWSWITCH_DEV_I2C_FMT             "/dev/i2c-%d"
#define LWSWITCH_VALID_I2C_FUNCS         (I2C_FUNC_I2C             | \
                                          I2C_FUNC_SMBUS_QUICK     | \
                                          I2C_FUNC_SMBUS_BYTE      | \
                                          I2C_FUNC_SMBUS_BYTE_DATA | \
                                          I2C_FUNC_SMBUS_WORD_DATA)

LwU32 i2c_allow_list_addresses_lr10[] = {0x50, 0x51, 0x52, 0x53,
                                         0x54, 0x55, 0x56, 0x57};

LwU32 i2c_allow_list_ports_lr10[] = {1, 2};

const LwU32 i2c_address_allow_list_size_lr10 =
    LW_ARRAY_ELEMENTS(i2c_allow_list_addresses_lr10);

const LwU32 i2c_port_allow_list_size_lr10 =
    LW_ARRAY_ELEMENTS(i2c_allow_list_ports_lr10);

using namespace lwswitch;

TEST_F(LWSwitchDeviceTest, I2cDeviceFunctionalityTest)
{
    int fd;
    LwU32 i;
    LwU32 numI2cAdapters;
    LwU32 adapters[LWSWITCH_CTRL_NUM_I2C_PORTS];
    char i2c_file[100];
    int ret;
    unsigned long funcs = 0;

    if (isArchSv10())
    {
        printf("[ SKIPPED  ] Test not supported on pre-LR10.\n");
        return;
    }

    numI2cAdapters = getNumI2cAdapters();
    getI2cAdapters(adapters, sizeof(adapters));

    for (i = 0; i < numI2cAdapters; i++)
    {
        if (snprintf(i2c_file, sizeof(i2c_file),
                     LWSWITCH_DEV_I2C_FMT, adapters[i]) < 0)
        {
            FAIL() << "Error getting I2C file.";
        }

        fd = open(i2c_file, O_RDWR);
        ASSERT_NE(fd, -1) << "Error opening " << i2c_file;

        ret = ioctl(fd, I2C_FUNCS, &funcs);
        if (ret < 0)
        {
            FAIL() << "Could not do IOCTL(I2C_FUNCS).";
        }

        ASSERT_EQ(funcs, LWSWITCH_VALID_I2C_FUNCS);

        close(fd);
    }
}

static __s32 i2c_smbus_quick_read(int fd)
{
    struct i2c_smbus_ioctl_data args;

    args.read_write = 0;
    args.command = 0;
    args.size = I2C_SMBUS_QUICK;
    args.data = NULL;

    return ioctl(fd, I2C_SMBUS, &args);
}

static bool isProcessInUninterruptibleSleep(int pid)
{
    char cmd[32] = { 0 };
    char result[16] = { 0 };

    snprintf(cmd, sizeof(cmd), "ps -o stat --pid %d", pid);

    FILE *fp = popen(cmd, "r");
    if (fp != NULL)
    {
        while (fgets(result, sizeof(result), fp) != NULL);
        pclose(fp);
        return (result[0] == 'D');
    }

    return false;
}

class LWSwitchDeviceI2cProcessExitTest : public LWSwitchDeviceTest
{
private:
    void getLWSwitchValidI2cDevice()
    {
        LwU32 i;
        int ret;
        int fd;
        LwU32 numI2cAdapters;
        LwU32 adapters[LWSWITCH_CTRL_NUM_I2C_PORTS];

        numI2cAdapters = getNumI2cAdapters();
        getI2cAdapters(adapters, sizeof(adapters));

        bHasDevice = LW_FALSE;

        for (i = 0; i < numI2cAdapters; i++)
        {
            if (snprintf(i2c_file, sizeof(i2c_file),
                         LWSWITCH_DEV_I2C_FMT, adapters[i]) < 0)
            {
                throw runtime_error("Error getting I2C File\n");
            }

            fd = open(i2c_file, O_RDWR);
            if (fd < 0)
            {
                throw runtime_error("Error opening I2C File\n");
            }

            for (devAddr = LWSWITCH_VALID_I2C_DEVICE_7BIT_ADDRESS_START;
                 devAddr <= LWSWITCH_VALID_I2C_DEVICE_7BIT_ADDRESS_STOP;
                 devAddr++)
            {
                ret = ioctl(fd, I2C_SLAVE, devAddr);
                if (ret < 0)
                {
                    continue;
                }

                ret = i2c_smbus_quick_read(fd);
                if (ret < 0)
                {
                    continue;
                }

                bHasDevice = LW_TRUE;

                break;
            }

            close(fd);

            if (bHasDevice)
            {
                break;
            }
        }
    }

public:
    void SetUp()
    {
        pid_t pid;
        int sockets[2] = {-1, -1};

        LWSwitchDeviceTestBase::SetUp();

        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0)
        {
            throw runtime_error("Failed to open socket pair");
        }

        getLWSwitchValidI2cDevice();

        pid = fork();

        if (pid != 0)
        {
            bIsParent = LW_TRUE;
            childPid = pid;

            close(sockets[1]);
            socketFd = sockets[0];
        }
        else
        {
            bIsParent = LW_FALSE;

            close(sockets[0]);
            socketFd = sockets[1];
        }
    }

    void TearDown()
    {
        LWSwitchDeviceTestBase::TearDown();
    }

    LwBool bHasDevice;
    int childPid;
    LwBool bIsParent;
    int socketFd;
    char i2c_file[100];
    int devAddr;
};

TEST_F(LWSwitchDeviceI2cProcessExitTest, I2cDeviceUnbindRebindTest)
{
    if (isArchSv10())
    {
        printf("[ SKIPPED  ] Test not supported on pre-LR10.\n");
        return;
    }

    if (bIsParent)
    {
        int wstatus;
        char dummy = 0;

        if (!bHasDevice)
        {
            kill(childPid, SIGKILL);
            ASSERT_EQ(waitpid(childPid, NULL, 0), childPid);
            printf("[ SKIPPED  ] No I2C devices found.\n");
            return;
        }

        // Signal to start test 1.
        ASSERT_TRUE(write(socketFd, &dummy, sizeof(dummy)) == sizeof(dummy));

        // Wait to confirm child has I2C file opened.
        ASSERT_TRUE(read(socketFd, &dummy, sizeof(dummy)) == sizeof(dummy));
        unbindRebindDevice();

        // Signal to start test 2.
        ASSERT_TRUE(write(socketFd, &dummy, sizeof(dummy)) == sizeof(dummy));

        unbindRebindDevice();

        ASSERT_TRUE(write(socketFd, &dummy, sizeof(dummy)) == sizeof(dummy));

        ASSERT_EQ(waitpid(childPid, &wstatus, 0), childPid);
        ASSERT_TRUE(WIFEXITED(wstatus));
        ASSERT_EQ(WEXITSTATUS(wstatus), 0);
    }
    else
    {
        LwU32 i;
        int fd;
        int ret;
        char dummy = 0;

        ASSERT_FALSE(isProcessInUninterruptibleSleep(getppid()));

        // 1. Wait for signal to test unbind/rebind on an open I2C file
        EXPECT_TRUE(read(socketFd, &dummy, sizeof(dummy)) == sizeof(dummy));

        fd = open(i2c_file, O_RDWR);
        if (fd < 0)
        {
            throw runtime_error("Error opening I2C File\n");
        }

        // Signal that I2C file is opened.
        ASSERT_TRUE(write(socketFd, &dummy, sizeof(dummy)) == sizeof(dummy));

        //
        // Hold I2C file open.
        // Wait until Parent is in uninterruptible sleep.
        //
        while(!isProcessInUninterruptibleSleep(getppid()));

        close(fd);

        // 2. Wait for signal to test multiple I2C transactions during unbind/rebind
        EXPECT_TRUE(read(socketFd, &dummy, sizeof(dummy)) == sizeof(dummy));

        for (i = 0; i < 100; i++)
        {
            fd = open(i2c_file, O_RDWR);
            if (fd < 0)
            {
                break;
            }

            ret = ioctl(fd, I2C_SLAVE, devAddr);
            if (ret < 0)
            {
                throw runtime_error("Error setting slave address!\n");
            }

            // Issue I2C transaction. Return doesn't matter
            i2c_smbus_quick_read(fd);
            close(fd);
        }

        // On failed open, wait for rebind to finish
        EXPECT_TRUE(read(socketFd, &dummy, sizeof(dummy)) == sizeof(dummy));

        // Check I2C transaction one last time after unbind/rebind
        fd = open(i2c_file, O_RDWR);
        ASSERT_GE(fd, 0);
        ASSERT_GE(ioctl(fd, I2C_SLAVE, devAddr), 0);
        ASSERT_EQ(i2c_smbus_quick_read(fd), 0);

        EXPECT_FALSE(isProcessInUninterruptibleSleep(getppid()));

        exit(HasFailure() ? -1 : 0);
    }
}

TEST_F(LWSwitchDeviceTest, I2cDevicePortsInAllowListTest)
{
    LwU32 i, j;
    LwU32 numI2cAdapters;
    LwU32 adapterPortNum[LWSWITCH_CTRL_NUM_I2C_PORTS];
    LwBool bInAllowList;

    if (isArchSv10())
    {
        printf("[ SKIPPED  ] Test not supported on pre-LR10.\n");
        return;
    }

    numI2cAdapters = getNumI2cAdapters();
    getI2cPortNums(adapterPortNum, sizeof(adapterPortNum));

    for (i = 0; i < numI2cAdapters; i++)
    {
        bInAllowList = LW_FALSE;

        for (j = 0; j < i2c_port_allow_list_size_lr10; j++)
        {
            if (adapterPortNum[i] == i2c_allow_list_ports_lr10[j])
            {
                bInAllowList = LW_TRUE;
            }
        }

        ASSERT_TRUE(bInAllowList == LW_TRUE);
    }
}

TEST_F(LWSwitchDeviceTest, I2cDeviceInAllowListTest)
{
    int fd;
    LwU32 i, j;
    LwU32 numI2cAdapters;
    LwU32 adapters[LWSWITCH_CTRL_NUM_I2C_PORTS];
    char i2c_file[100];
    int ret;
    int devAddr;

    if (isArchSv10())
    {
        printf("[ SKIPPED  ] Test not supported on pre-LR10.\n");
        return;
    }

    numI2cAdapters = getNumI2cAdapters();
    getI2cAdapters(adapters, sizeof(adapters));

    for (i = 0; i < numI2cAdapters; i++)
    {
        if (snprintf(i2c_file, sizeof(i2c_file),
                     LWSWITCH_DEV_I2C_FMT, adapters[i]) < 0)
        {
            FAIL() << "Error getting I2C file.";
        }

        for (j = 0; j < i2c_address_allow_list_size_lr10; j++)
        {
            devAddr = i2c_allow_list_addresses_lr10[j];

            fd = open(i2c_file, O_RDWR);
            ASSERT_NE(fd, -1) << "Error opening " << i2c_file;

            ret = ioctl(fd, I2C_SLAVE, devAddr);
            if (ret < 0)
            {
                close(fd);
                FAIL() << "Could not do IOCTL(I2C_SLAVE).";
            }

            ret = i2c_smbus_quick_read(fd);
            close(fd);

            //
            // The devices may or may not be on the bus, so it
            // is possible that the smbus read will fail.
            //
            ASSERT_LE(ret, 0);
        }
    }
}

TEST_F(LWSwitchDeviceTest, I2cDeviceNotInAllowListTest)
{
    int fd;
    LwU32 i, j;
    LwU32 numI2cAdapters;
    LwU32 adapters[LWSWITCH_CTRL_NUM_I2C_PORTS];
    char i2c_file[100];
    int ret;
    int devAddr;
    LwBool bInAllowList;

    if (isArchSv10())
    {
        printf("[ SKIPPED  ] Test not supported on pre-LR10.\n");
        return;
    }

    numI2cAdapters = getNumI2cAdapters();
    getI2cAdapters(adapters, sizeof(adapters));

    for (i = 0; i < numI2cAdapters; i++)
    {
        if (snprintf(i2c_file, sizeof(i2c_file),
                     LWSWITCH_DEV_I2C_FMT, adapters[i]) < 0)
        {
            FAIL() << "Error getting I2C file.";
        }

        for (devAddr = LWSWITCH_VALID_I2C_DEVICE_7BIT_ADDRESS_START;
             devAddr <= LWSWITCH_VALID_I2C_DEVICE_7BIT_ADDRESS_STOP;
             devAddr++)
        {
            bInAllowList = LW_FALSE;

            for (j = 0; j < i2c_address_allow_list_size_lr10; j++)
            {
                if (devAddr == i2c_allow_list_addresses_lr10[j])
                {
                    bInAllowList = LW_TRUE;
                    break;
                }
            }

            // Skip devices in the allowlist
            if (bInAllowList)
            {
                continue;
            }

            fd = open(i2c_file, O_RDWR);
            ASSERT_NE(fd, -1) << "Error opening " << i2c_file;

            ret = ioctl(fd, I2C_SLAVE, devAddr);
            if (ret < 0)
            {
                close(fd);
                FAIL() << "Could not do IOCTL(I2C_SLAVE).";
            }

            ret = i2c_smbus_quick_read(fd);
            close(fd);

            ASSERT_LT(ret, 0);
        }
    }
}

#endif // __linux__

