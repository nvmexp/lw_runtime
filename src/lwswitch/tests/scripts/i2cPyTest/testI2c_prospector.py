#!/usr/bin/elw python3

#
# need to import smbus2 first
# pip3 install smbus2
#
from smbus2 import SMBus, i2c_msg
from functools import wraps
from multiprocessing.dummy import Pool as ThreadPool
import os
import re
import sys
import time
import unittest

BDF_PATTERN = re.compile(r'([0-9a-fA-F]+:[0-9a-dA-F]+.[0-9a-fA-F]+)')
ADAPTER_PATTERN = re.compile(r'adapter ([0-9])')
I2C_BUS_PATTERN = re.compile(r'i2c-([0-9]+)')

NUM_WRITE_READS = 16
SCRATCH_REG_ADDR = 136
EXPECTED_NUM_ADAPTERS = 2
EXPECTED_NUM_OSFP_PER_ADAPTER = 4
EXPECTED_TOTAL_OSFP = EXPECTED_NUM_ADAPTERS * \
                      EXPECTED_NUM_OSFP_PER_ADAPTER

i2c_dev_dir = "/sys/bus/i2c/devices/"
i2c_dir_dev_name_fmt = "/name"
lwswitch_file_delimit = "LWPU LWSwitch i2c adapter"
i2c_dev_addr_range = [0x03, 0x77]
osfp_dev_addr = [0x50, 0x51, 0x52, 0x53]

#
# The allowed device addresses were extracted from:
# https://confluence.lwpu.com/display/DH/LWSwitch+FW+for+LWLink3+Optical+Support#LWSwitchFWforLWLink3OpticalSupport-Wolf.1
#
prospector_wolf_port_allow_list = [1, 2]
prospector_wolf_device_allow_list = [0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57]

#
# Global dictionary that stores LwSwitch i2c dev info
#
lwswitch_dev_info = {}

# A second-order decorator
def decdec(inner_dec):
    def ddmain(outer_dec):
        def decwrapper(f):
            wrapped = inner_dec(outer_dec(f))
            def fwrapper(*args, **kwargs):
               return wrapped(*args, **kwargs)
            return fwrapper
        return decwrapper
    return ddmain

# Wrapper for iterating though osfp devices
def for_all_osfp_devices(fn):
    @wraps(fn)
    def wrapper(*args, **kwds):
        for bdf, lwswitch in lwswitch_dev_info.items():
            for osfp in lwswitch.osfpDevs:
                kwds["osfp"] = osfp
                fn(*args, **kwds)
    return wrapper

# Wrapper for osfp bank/page switches
@decdec(for_all_osfp_devices)
def for_all_osfp_bank_page_switches(fn):
    def wrapper(*args, **kwds):
        osfp = kwds["osfp"]

        old_bank, old_page = osfp.i2cReadBlk(126, 2)

        # Set bank/page to 0, 0x9f
        osfp.cmisSetBankAndPage(0, 0x9f)

        fn(*args, **kwds)

        # Reset bank/page
        osfp.cmisSetBankAndPage(old_bank, old_page)

    return wrapper

# Wrapper for iterating through LwSwitch devices
def for_all_lwswitches(fn):
    @wraps(fn)
    def wrapper(*args, **kwds):
        for buf, lwswitch in lwswitch_dev_info.items():
            kwds["lwswitch"] = lwswitch
            fn(*args, **kwds)
    return wrapper

class LwSwitch:
    def __init__(self, bdf, i2c_info):
        self.bdf = bdf
        self.i2cInfo = i2c_info
        self.osfpDevs = []
        self.pingedI2cAddr = []
        self.adapterCount = 0

        osfp_index = 0
        for adapter_num in i2c_info: 

            self.adapterCount += 1
            i2c_bus = i2c_info[adapter_num]
            for dev_addr in range(i2c_dev_addr_range[0], i2c_dev_addr_range[1] + 1):
                i2c_dev = self.I2cDev(bdf, i2c_bus, dev_addr, adapter_num, osfp_index)

                if (i2c_dev.i2cWriteQuick() == True):
                    # Add all pinged addresses to check allowlist functionality
                    if (dev_addr not in self.pingedI2cAddr):
                        self.pingedI2cAddr.append(dev_addr)

                    #
                    # Add only common device addresses to test I2C transactions.
                    # There are some addresses that exist on Prospector, with
                    # non-osfp modules attached, that are valid osfp devices
                    # on Wolf.
                    #
                    if (dev_addr in osfp_dev_addr):
                        self.osfpDevs.append(i2c_dev)
                        osfp_index += 1

    def displayProperties(self):
        print(f"LwSwitch device {self.bdf} properties:")
        print(f"    Number of OSFPS: {len(self.osfpDevs)}")
        for osfp in self.osfpDevs:
            vendorName, partName, serialNum = osfp.cmisGetVendorInfo()
            print(f"    OFSP {osfp.index} bus /dev/i2c-{osfp.bus} adapter: {osfp.adapter} address: 0x{osfp.addr:x}")
            print(f"        Vendor: {vendorName}, Part number: {partName}, Serial Number: {serialNum}")

    class I2cDev:
        def __init__(self, bdf, bus, addr, adapter, index):
            self.bdf = bdf
            self.bus = bus
            self.addr = addr
            self.adapter = adapter
            self.index = index

        def error_msg(self, test):
            print(f"Error: LwSwitch {self.bdf} OSFP {self.index} {test} verification failed!")

        def i2cWriteQuick(self):
            """
            Perform quick transaction

            Returns:
                Boolean if command succeeds
            """
            try:
                with SMBus(self.bus) as bus:
                    bus.write_quick(self.addr)

                return True
            except IOError:
                return False
            except:
                assert (0), \
                    f"Error: LwSwitch {self.bdf} OSFP {self.index} Quick Write unknown exception!"

        def i2cRead(self, regAddr):
            """
            Perform 1-byte read

            Attributes:
                regAddr -- The address offset to read.

            Returns:
                The value read from the specified offset.
            """
            with SMBus(self.bus) as bus:
                val = bus.read_byte_data(self.addr, regAddr)

            return val

        def i2cReadWord(self, regAddr):
            """
            Perform 2-byte read

            Attributes:
                regAddr -- The address offset to read.

            Returns:
                The value read from the specified offset.
            """
            with SMBus(self.bus) as bus:
                val = bus.read_word_data(self.addr, regAddr)

            return val

        def i2cReadBlk(self, regAddr, length):
            """
            Perform multi-byte read

            Attributes:
                regAddr -- The address offset to read.
                length  -- How many bytes to read starting from the specified offset. Up to 128.

            Returns:
                List of bytes read from the specified offset.
            """

            #
            # Combine a series of i2c read and write operations in a single transaction
            # (with repeated start bits but no stop bits in between). This method takes
            # i2c_msg instances as input, which must be created first with i2c_msg.read()
            # or i2c_msg.write()
            #

            # Write 1 byte to set offset to read from
            writeOffsetMsg = i2c_msg.write(self.addr, [regAddr])

            # Read out the bytes from the offset
            readMsg = i2c_msg.read(self.addr, length)

            with SMBus(self.bus) as bus:
                bus.i2c_rdwr(writeOffsetMsg, readMsg)

            return list(readMsg)

        def i2cWrite(self, regAddr, writeVal, bVerifyWrite):
            """
            Perform 1-byte write

            Attributes:
                regAddr  -- The address offset to write.
                writeVal -- The byte value to write to offset.

            Returns:
                None
            """
            with SMBus(self.bus) as bus:
                bus.write_byte_data(self.addr, regAddr, writeVal)

            time.sleep(0.005)

            if (bVerifyWrite):
                readBack = self.i2cRead(regAddr)

                assert (readBack == writeVal), \
                    f"Error: LwSwitch {self.bdf} OSFP {self.index} Byte write verification failed! (wrote: {hex(writeVal)}, read: {hex(readBack)})"

        def i2cWriteWord(self, regAddr, writeVal, bVerifyWrite):
            """
            Perform 2-byte write

            Attributes:
                regAddr  -- The address offset to write.
                writeVal -- The byte value to write to offset.

            Returns:
                None

            """
            with SMBus(self.bus) as bus:
                bus.write_word_data(self.addr, regAddr, writeVal)

            time.sleep(0.005)

            if (bVerifyWrite):
                readBack = self.i2cReadWord(regAddr)

                assert (readBack == writeVal), \
                    f"Error: LwSwitch {self.bdf} OSFP {self.index} Word write verification failed! (wrote: {hex(writeVal)}, read: {hex(readBack)})"

        def i2cWriteBlk(self, regAddr, writeArray, bVerifyWrite):
            """
            Perform block write

            Attributes:
                regAddr      -- The address offset to read.
                writeArray   -- List of bytes to write. Up to 128.
                bVerifyWrite -- Bool to perform a read-back to verify read data matches what is written

            Returns:
                None
            """

            #
            # Combine a series of i2c read and write operations in a single transaction
            # (with repeated start bits but no stop bits in between). This method takes
            # i2c_msg instances as input, which must be created first with i2c_msg.read()
            # or i2c_msg.write()
            #

            data = writeArray.copy()
            # Block write first writes regAddr, then data bytes all in one xfr
            data.insert(0, regAddr)

            writeMsg = i2c_msg.write(self.addr, data)
            with SMBus(self.bus) as bus:
                bus.i2c_rdwr(writeMsg)

            time.sleep(0.005)

            if (bVerifyWrite):
                readBack = self.i2cReadBlk(regAddr, len(writeArray))

                assert (readBack == writeArray), \
                    f"Error: LwSwitch {self.bdf} OSFP {self.index} Block write verification failed! (wrote: {writeArray}, read: {readBack})"

        def cmisSetBankAndPage(self, bank, page):

            #
            # Write in the selected bank and page (10h) into Page 0h, byte 126
            # CMIS4 states, "For a bank change, the host shall write the Bank Select
            # and Page Select registers in the same TWI transaction."
            #
            data = [bank, page]
            self.i2cWriteBlk(126, data, True)

        def cmisGetVendorInfo(self):
            self.cmisSetBankAndPage(0, 0)

            readArr = self.i2cReadBlk(129, 16)
            vendorName = ''
            
            for byte in readArr:
                vendorName += chr(byte)

            readArr = self.i2cReadBlk(148, 16)
            partNumber = ''

            for byte in readArr:
                partNumber += chr(byte)

            readArr = self.i2cReadBlk(166, 16)
            serialNumber = ''

            for byte in readArr:
                serialNumber += chr(byte)

            vendorName = vendorName.strip()
            partNumber = partNumber.strip()
            serialNumber = serialNumber.strip()

            return vendorName, partNumber, serialNumber

class testI2c(unittest.TestCase):

    lwrrentResult = None

    @classmethod
    def setResult(cls, amount, errors, failures, skipped):
        cls.amount, cls.errors, cls.failures, cls.skipped = \
            amount, errors, failures, skipped

    def tearDown(self):
        amount   = self.lwrrentResult.testsRun
        errors   = self.lwrrentResult.errors
        failures = self.lwrrentResult.failures
        skipped  = self.lwrrentResult.skipped
        self.setResult(amount, errors, failures, skipped)

    @classmethod
    def tearDownClass(cls):
        numTests = cls.amount
        numErrors = len(cls.errors)
        numFailures = len(cls.failures)
        numSuccess = numTests - numErrors - numFailures
        pctSuccess = (numSuccess / numTests) * 100
        print(f"\nI2C Test score: {pctSuccess}")

    def run(self, result=None):
        self.lwrrentResult = result
        unittest.TestCase.run(self, result)

    @for_all_lwswitches
    def test_port_allow_list(self, lwswitch):
        for adapter_num in lwswitch.i2cInfo:
            assert (adapter_num in prospector_wolf_port_allow_list), \
                f"I2C port {adapter_num} not in allow list!"

    @for_all_lwswitches
    def test_device_allow_list(self, lwswitch):
        #
        # Go through all pinged I2C devices and check
        # if they are in the allow list.
        #
        for addr in lwswitch.pingedI2cAddr:
            assert (addr in prospector_wolf_device_allow_list), \
                f"I2C device addr {hex(addr)} not in allow list!"

    @for_all_lwswitches
    def test_ilwalid_device_addr(self, lwswitch):
        for adapter_num in lwswitch.i2cInfo:
            assert (adapter_num in prospector_wolf_port_allow_list), \
                f"I2C port {adapter_num} not in allow list!"

            #
            # Ping I2C devices not in the allow list and
            # confirm the transaction fails.
            #
            i2c_bus = lwswitch.i2cInfo[adapter_num]
            for dev_addr in range(i2c_dev_addr_range[0], i2c_dev_addr_range[1] + 1):
                if (dev_addr not in prospector_wolf_device_allow_list):
                    i2cDev = lwswitch.I2cDev(None, i2c_bus, dev_addr, adapter_num, None)

                    assert(i2cDev.i2cWriteQuick() == False), \
                        f"Was able to issue I2C transaction to addr {hex(dev_addr)} not in the allow list!"

    @for_all_osfp_devices
    def test_i2c_quick(self, osfp):
        assert (osfp.i2cWriteQuick() == True), \
            f"LwSwitch {osfp.bdf} OSFP {osfp.index} Quick write failed!"

    @for_all_osfp_devices
    def test_i2c_read(self, osfp):
        val = osfp.i2cRead(0x0)
        assert (val == 0x19), \
            f"LwSwitch {osfp.bdf} OSFP {osfp.index} byte read failed! (expected: 0x19, read: {hex(val)})"

    @for_all_osfp_bank_page_switches
    def test_i2c_write_read_byte(self, osfp):
        # Set to 0 first
        for i in range(NUM_WRITE_READS):
            osfp.i2cWrite(SCRATCH_REG_ADDR, 0, True)

        # Verify random byte write/reads
        for i in range(NUM_WRITE_READS):
            randByte = os.urandom(1)
            osfp.i2cWrite(SCRATCH_REG_ADDR, int.from_bytes(randByte, "big"), True)

        # Verify larger than one byte writes are not reflected in reads
        word = int.from_bytes(os.urandom(2), "big") | 0x1000
        osfp.i2cWrite(SCRATCH_REG_ADDR, word, False)

        readBack = osfp.i2cRead(SCRATCH_REG_ADDR)

        assert (readBack != word), \
            f"LwSwitch {osfp.bdf} OSFP {osfp.index} byte read failed! (read: {hex(readBack)} should not equal write: {hex(word)})"
        assert (readBack == (word & 0xff)), \
            f"LwSwitch {osfp.bdf} OSFP {osfp.index} byte read failed! (read: {hex(readBack)} should be lower byte of write: {hex(word)})"

    @for_all_osfp_bank_page_switches
    def test_i2c_write_read_word(self, osfp):
        # Set to 0 first
        for i in range(NUM_WRITE_READS):
            osfp.i2cWriteWord(SCRATCH_REG_ADDR, 0, True)

        # Verify random word writes/reads
        for i in range(NUM_WRITE_READS):
            randWord = os.urandom(2)
            osfp.i2cWriteWord(SCRATCH_REG_ADDR, int.from_bytes(randWord, "big"), True)

        # Verify larger than one word writes are not reflected in reads
        doubleWord = int.from_bytes(os.urandom(4), "big") | 0x10000000
        osfp.i2cWriteWord(SCRATCH_REG_ADDR, doubleWord, False)

        readBack = osfp.i2cReadWord(SCRATCH_REG_ADDR)

        assert (readBack != doubleWord), \
            f"LwSwitch {osfp.bdf} OSFP {osfp.index} word read failed! (read: {hex(readBack)} should not equal write: {hex(doubleWord)})"
        assert (readBack == (doubleWord & 0xffff)), \
            f"LwSwitch {osfp.bdf} OSFP {osfp.index} word read failed! (read: {hex(readBack)} should be lower word of write: {hex(doubleWord)})"

    @for_all_osfp_bank_page_switches
    def test_i2c_write_read_blk(self, osfp):
        data = [0] * 16
        osfp.i2cWriteBlk(SCRATCH_REG_ADDR, data, True)

        for i in range(len(data)):
            randByte = os.urandom(1)
            data[i] = int.from_bytes(randByte, "big")

        osfp.i2cWriteBlk(SCRATCH_REG_ADDR, data, True)

    @for_all_osfp_bank_page_switches
    def test_i2c_atomic_transactions(self, osfp):
        def exelwte_transaction(osfp):
            # Set to 0 first
            for i in range(NUM_WRITE_READS):
                osfp.i2cWrite(SCRATCH_REG_ADDR, 0, True)

            # Verify random byte write/reads
            for i in range(NUM_WRITE_READS):
                randByte = os.urandom(1)
                osfp.i2cWrite(SCRATCH_REG_ADDR, int.from_bytes(randByte, "big"), True)

        pool = ThreadPool(5)
        results = pool.map(exelwte_transaction, [osfp])

        pool.close()
        pool.join()

def getLwSwitchI2cInfo():
    #
    # Dictionary that stores device i2c information
    # in the format {Device BDF : {adapter_num: i2c bus, ...}, ...}
    #
    i2c_info_dict = {}

    for (dev_dir, dev_dir_names, dev_files) in os.walk(i2c_dev_dir):
        for i2c_file in dev_dir_names:
            i2c_name_file = i2c_dev_dir + i2c_file + i2c_dir_dev_name_fmt
            name_file     = open(i2c_name_file, "r")
            lines         = name_file.readlines()

            i2c_bus_regex = re.search(I2C_BUS_PATTERN, i2c_file)
            if (i2c_bus_regex is None):
                sys.exit("Could not extract I2C Bus number!")

            i2c_bus = i2c_bus_regex.group(1)

            for line in lines:
                line = line.strip()

                # Skip non-LWSwitch entries
                if (lwswitch_file_delimit not in line):
                    continue

                bdf_regex = re.search(BDF_PATTERN, line)
                if (bdf_regex is None):
                    sys.exit("Could not extract Device BDF!")

                adapter_regex = re.search(ADAPTER_PATTERN, line)
                if (adapter_regex is None):
                    sys.exit("Could not extract I2C Adapter number!")

                bdf = bdf_regex.group()
                adapter_num = adapter_regex.group(1)

                # Store Device I2C information
                if (bdf not in i2c_info_dict):
                    i2c_dict = {int(adapter_num) : int(i2c_bus)}
                    i2c_info_dict[bdf] = i2c_dict
                else:
                    tmp_dict = i2c_info_dict[bdf]
                    tmp_dict[int(adapter_num)] = int(i2c_bus)
                    i2c_info_dict[bdf] = dict(sorted(tmp_dict.items(), key=lambda item: item[1]))

    # Return sorted dictionary, where the key to sort is the first available adapter_num
    return dict(sorted(i2c_info_dict.items(), key = lambda item: item[1][next(iter(item[1]))]))

def getLwSwitchInfo():
    for bdf, lwswitch in lwswitch_dev_info.items():
        lwswitch.displayProperties()

def main():
    lwswitch_i2c_info = getLwSwitchI2cInfo()

    if (not lwswitch_i2c_info):
        print("Did not find any devices. Skipping.")
        return 0

    for bdf in lwswitch_i2c_info:
       lwswitch = LwSwitch(bdf, lwswitch_i2c_info[bdf])

       if (lwswitch.adapterCount == 0):
           sys.exit("No LwSwitch I2C devices found! Please check that system has LwSwitches or "
                    "try enabling the regkey: LwSwitchRegDwords='I2cPortControlMask=0x7'")

       numAdapters = lwswitch.adapterCount
       numOsfps    = len(lwswitch.osfpDevs)


       if ((numAdapters < EXPECTED_NUM_ADAPTERS) or (numOsfps < EXPECTED_TOTAL_OSFP)):
           adapters = [adapter for adapter in lwswitch.i2cInfo]
           osfp     = [(osfp.adapter, hex(osfp.addr)) for osfp in lwswitch.osfpDevs]
           print(f"Info: Expected {EXPECTED_NUM_ADAPTERS} adapters for LwSwitch {lwswitch.bdf} and "
                 f"{EXPECTED_TOTAL_OSFP} osfps, but only found {numAdapters} adapters: "
                 f"{adapters} and {numOsfps} osfps: {osfp}")

       lwswitch_dev_info[bdf] = lwswitch

    getLwSwitchInfo()

    print("\nStarting tests:")
    unittest.main()
    return 0

if __name__ == "__main__":
    main()

