LWSWITCH TESTS

The tests here are designed to test the Linux driver interface from user mode. They
have a dependency on both gtest and the Linux interface.  Given the FSF style
development we did not look at traditional mods tests.

The goal is lwrrently to write API sanity tests on the switch alone. This framework
does not interact with RM or LWCA, and cannot configure the system and pass traffic.
It is somewhere between a unit test and a system test.  But it does provide some
driver coverage before we run more complicated systems with Fabric Manager or mods.

Originally tests were written as direct C code with a few helper macros. After that
we moved to the gtest framework. The goal is to move all tests into the gtest
framework and remove all the basic C tests.

The main advantage of using gtest are:
   - Takes some burden off the test writing - the ASSERT helper is really nice.
   - Well tested and accepted framework, including inside lwpu
   - Consistent output format allows tools/infra (DVS,etc.) to parse test results.

The tests are run on a system not running fabric manager or mods. It is also expected
that the tests are run on a system with un-initialized lwlinks. If the test fails
due to initialized links, please reset the devices using "lwpu-smi -r".
NOTE:
HW claims that using SBR on LWLink connected devices with initialized links can
lead to back-drive and physical damage. Hence using lwpu-smi as the reset
mechanism is preferred given that it will reset all the connected devices in the
system.


The driver should be unloaded and reloaded to re-initialize the switch after running any tests.

REQUIRED TESTS

New ioctls added to the driver require both a functional test and a BadInput test.

BUILDING

The C tests build with a simple "make" on with enough of the source tree to get various
headers will build the tests.

Building the gtest is more diffilwlt and require unix-build and lwmake. See the Linux build
wiki pages for more details on unix build.

   unix-build --tools $BUILD_TOOLS_DIR --extra $_BUILDTOOLS_DIR lwmake debug amd64

GTEST NOTES

We have a few different Test fixtures to help set up different types of tests. The goal
is to put a lot of the common set-up code in the fixture to make writing the test simple.

By default the test runs on lwswitch0. The --index argument can be used to select
a specific device to test. The run_tests_all_devices.sh shell script can be used
to run the test on all the devices in the system. Lwrrently the tests are per-switch
tests and adding more complicated multi-switch device tests might need more dislwssion.

We also instantiate a custom listener, which allows us to print extra information. This
is used to print which device failed (lwswitch 0, etc).

ADDING A TEST

Most tests focus on the ioctl interface and will use the LwSwitchDeviceTest fixture. The
fixture will open/close the device automatically and provides some common helper routines.

Tests are declared with the TEST_F macro.

    TEST_F(LWSwitchDeviceTest, IoctlI2CIndexedBadInput)
    {
         // test body with "fd" open and ready to use.
    }

The macros automatically add the test to the test list.

ADDING FILES

Adding files for new test buckets is easy. Add the file and update makefile.lwmake to include it.

FURTHER DOCUMENTATION

Gtest itself has good documentation online.
