/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _ROM_LWSWITCH_H_
#define _ROM_LWSWITCH_H_

#include "pmgr_lwswitch.h"
#include "io_lwswitch.h"

//
// When parsing BIOS tables these wrappers help protect against reading and using
// fields that may not be present in the ROM image by checking the offset against
// the structure size.
//
#define LW_OFFSETOF_MEMBER(_basePtr, _member)                                  \
    ((LwUPtr)(((LwU8 *)(&((_basePtr)->_member))) - ((LwU8 *)(_basePtr))))

#define LWSWITCH_ELEMENT_PRESENT(_ptr, _element, _size)          \
    (LW_OFFSETOF_MEMBER((_ptr), _element) + sizeof((_ptr)->_element) <= (_size))

#define LWSWITCH_ELEMENT_READ(_ptr, _element, _size, _default)   \
    (LWSWITCH_ELEMENT_PRESENT(_ptr, _element, _size) ?           \
        ((_ptr)->_element) : (_default))

#define LWSWITCH_ELEMENT_VALIDATE(_ptr, _element, _size, _default, _expected)   \
    do                                                                          \
    {                                                                           \
        LwU32 data = LWSWITCH_ELEMENT_READ(_ptr, _element, _size, _default);    \
        if (data != (_expected))                                                \
        {                                                                       \
            LWSWITCH_PRINT(device, SETUP,                                       \
                "Element '%s->%s'=0x%x but expected 0x%x\n",                    \
                #_ptr, #_element, data, (LwU32) (_expected));                   \
        }                                                                       \
    } while(0)

#define LWSWITCH_ELEMENT_CHECK(_ptr, _element, _size, _default)                 \
    LWSWITCH_ELEMENT_VALIDATE(_ptr, _element, _size, _default, _default)

#if (defined(_WIN32) || defined(_WIN64))
#define LWSWITCH_STRUCT_PACKED_ALIGNED(typeName, bytes)                        \
    __pragma(pack(push, bytes)) typedef struct typeName 

#define LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX               __pragma(pack(pop))
#else
#define LWSWITCH_STRUCT_PACKED_ALIGNED(typeName, bytes)                        \
    typedef struct __attribute__((packed, aligned(bytes))) 

#define LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX
#endif // (defined(_WIN32) || defined(_WIN64))

//
// AT24CM02 EEPROM
// http://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-8828-SEEPROM-AT24CM02-Datasheet.pdf
#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
// 2Mb EEPROM used on all SV10 platforms except E3600 A01
#endif
//

#define AT24CM02_INDEX_SIZE     18          // Addressing bits
#define AT24CM02_BLOCK_SIZE     256         // R/W block size (bytes)

//
// AT24C02C EEPROM
// http://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-8700-SEEPROM-AT24C01C-02C-Datasheet.pdf
#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
// 2kb EEPROM used on SV10 E3600 A01 platform
#endif
//

#define AT24C02C_INDEX_SIZE     8           // Addressing bits
#define AT24C02C_BLOCK_SIZE     8           // R/W block size (bytes)

//
// AT24C02D EEPROM
// http://ww1.microchip.com/downloads/en/devicedoc/atmel-8871f-seeprom-at24c01d-02d-datasheet.pdf
// 2kb EEPROM used on LR10 P4790 B00 platform
//

#define AT24C02D_INDEX_SIZE     8           // Addressing bits
#define AT24C02D_BLOCK_SIZE     8           // R/W block size (bytes)

typedef struct
{
    LwU32 i2c_port;
    LwU32 i2c_address;
    LwU32 device_type;
    LwU32 index_size;
    LwU32 block_size;
    LwU32 block_count;
    LwU32 eeprom_size;
} LWSWITCH_EEPROM_TYPE;

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_EEPROM_HEADER, 1)
{
    char    signature[4];
    LwU16   version;
    LwU16   header_size;
    LwU16   pci_vendor_id;
    LwU16   pci_device_id;
    LwU16   pci_system_vendor_id;
    LwU16   pci_system_device_id;
    LwU16   firmware_size;
    LwU8    reserved[13];
    LwU8    checksum;
} LWSWITCH_EEPROM_HEADER;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_BIT_HEADER, 1)
{
    LwU16   id;
    char    signature[4];
    LwU16   bcd_version;
    LwU8    header_size;
    LwU8    token_size;
    LwU8    token_entries;
    LwU8    checksum;
} LWSWITCH_BIT_HEADER;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_BIT_TOKEN_CLOCK_PTRS                0x43
#define LWSWITCH_BIT_TOKEN_LWINIT_PTRS               0x49
#define LWSWITCH_BIT_TOKEN_NOP                       0x4E
#define LWSWITCH_BIT_TOKEN_PERF_PTRS                 0x50
#define LWSWITCH_BIT_TOKEN_BRIDGE_FW_DATA            0x52
#define LWSWITCH_BIT_TOKEN_INTERNAL_USE_ONLY_PTRS    0x69
#define LWSWITCH_BIT_TOKEN_DCB_PTRS                  0x6E

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_BIT_TOKEN, 1)
{
    LwU8    id;
    LwU8    data_version;
    LwU16   data_size;
    LwU16   data_offset;
} LWSWITCH_BIT_TOKEN;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

// 0x43: BIT_TOKEN_CLOCK_PTRS
LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_BIT_CLOCK_PTRS, 1)
{
    LwU32 pll_info_table;
    LwU32 vbe_mode_pclk;
    LwU32 clocks_table;
    LwU32 clocks_programming;
    LwU32 nafll;
    LwU32 adc_table;
    LwU32 freq_control;
} LWSWITCH_BIT_CLOCK_PTRS;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_CLOCK_PTRS_PLL_INFO_VERSION    0x50

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_PLL_INFO_HEADER, 1)
{
    LwU8  version;
    LwU8  header_size;
    LwU8  entry_size;
    LwU8  entry_count;
} LWSWITCH_PLL_INFO_HEADER;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_PLL_INFO_ENTRY, 1)
{
    LwU8  pll_id;
    LwU16 ref_min_mhz;
    LwU16 ref_max_mhz;
    LwU16 vco_min_mhz;
    LwU16 vco_max_mhz;
    LwU16 update_min_mhz;
    LwU16 update_max_mhz;
    LwU8  m_min;
    LwU8  m_max;
    LwU8  n_min;
    LwU8  n_max;
    LwU8  pl_min;
    LwU8  pl_max;
} LWSWITCH_PLL_INFO_ENTRY;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_PLL_ID_SYSPLL      0x07

// 0x49: BIT_TOKEN_LWINIT_PTRS
LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_BIT_LWINIT_PTRS, 1)
{
    LwU16 init_script;
    LwU16 macro_index;
    LwU16 macro_table;
    LwU16 condition;
    LwU16 io_condition;
    LwU16 io_flag_condition;
    LwU16 init_function;
    LwU16 private_boot;
    LwU16 data_arrays;
    LwU16 pcie_settings;
    LwU16 devinit;
    LwU16 devinit_size;
    LwU16 boot_script;
    LwU16 boot_script_size;
    LwU16 lwlink_config;
    LwU16 boot_script_nonGC6;
    LwU16 boot_script_nonGC6_size;
} LWSWITCH_BIT_LWINIT_PTRS;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_LWLINK_CONFIG, 1)
{
    LwU8    version;
    LwU8    size;
    LwU16   reserved;
    LwU64   link_disable_mask;      // 1 = disable
    LwU64   link_speed_mask;        // 1 = safe mode
    LwU64   link_refclk_mask;       // 0 = 100MHz, 1 = 133MHz
    LwU8    flags;
    LwU64   ac_coupled_mask;        // 0 = DC, 1 = AC
} LWSWITCH_LWLINK_CONFIG;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

// 0x52: BIT_TOKEN_BRIDGE_FW_DATA
LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_BIT_BRIDGE_FW_DATA, 1)
{
    LwU32 firmware_version;
    LwU8  oem_version;
    LwU16 firmware_size;
    char  BIOS_MOD_date[8];
    LwU32 firmware_flags;
    LwU16 eng_product_name;
    LwU8  eng_product_name_size;
    LwU16 lwswitch_instance_id;
} LWSWITCH_BIT_BRIDGE_FW_DATA;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_BIT_BRIDGE_FW_DATA_FLAGS_BUILD               0:0
#define LWSWITCH_BIT_BRIDGE_FW_DATA_FLAGS_BUILD_REL           0
#define LWSWITCH_BIT_BRIDGE_FW_DATA_FLAGS_BUILD_ENG           1
#define LWSWITCH_BIT_BRIDGE_FW_DATA_FLAGS_I2C                 1:1
#define LWSWITCH_BIT_BRIDGE_FW_DATA_FLAGS_I2C_MASTER          0
#define LWSWITCH_BIT_BRIDGE_FW_DATA_FLAGS_I2C_NOT_MASTER      1

// 0x69: BIT_TOKEN_INTERNAL_USE_ONLY_PTRS
LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_BIT_INTERNAL_USE_ONLY_PTRS, 1)
{
    LwU32 bios_version;
    LwU8  oem_version;
    LwU16 features;
    LwU32 perforce;
    LwU16 board_id;
    LwU16 dac_data;
    char build_date[8];
    LwU16 device_id;
    LwU32 int_flags_1;
    LwU32 int_flags_2;
    LwU16 preservation;
    LwU8  lwflash_id;
    LwU8  hierarchy_id;
    LwU8  int_flags_3;
    LwU16 alt_device_id;
    LwU32 kda_buffer;
    LwU32 int_flags_4;
    LwU8  chip_sku[3];
    LwU8  chip_sku_modifier;
    LwU32 lw_project;
    LwU32 lw_project_sku;
    LwU8  cdp[5];
    LwU8  lw_project_sku_modifier;
    LwU8  business_cycle;
    LwU8  int_flags_5;
    LwU8  cert_flag;
    LwU8  int_flags_6;
    LwU16 alt_board_id;
    LwU8  build_guid[16];
    LwU32 devid_override_list;
    LwU16 min_netlist;
    LwU8  min_rm_revlock;
    LwU8  lwrr_vbios_revlock;
} LWSWITCH_BIT_INTERNAL_USE_ONLY_PTRS;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

// 0x6E: BIT_TOKEN_DCB_PTRS
LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_BIT_DCB_PTRS, 1)
{
    LwU16 dcb_header_ptr;
} LWSWITCH_BIT_DCB_PTRS;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_DCB_HEADER_VERSION_41  0x41
#define LWSWITCH_DCB_HEADER_SIGNATURE   0x4edcbdcb

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_DCB_HEADER, 1)
{
    LwU8  version;
    LwU8  header_size;
    LwU8  entry_count;
    LwU8  entry_size;
    LwU16 ccb_block_ptr;
    LwU32 dcb_signature;
    LwU16 gpio_table;
    LwU16 input_devices;
    LwU16 personal_cinema;
    LwU16 spread_spectrum;
    LwU16 i2c_devices;
    LwU16 connectors;
    LwU8  flags;
    LwU16 hdtv;
    LwU16 switched_outputs;
    LwU32 display_patch;
    LwU32 connector_patch;
} LWSWITCH_DCB_HEADER;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_GPIO_TABLE_VERSION_42  0x42

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_GPIO_TABLE, 1)
{
    LwU8    version;
    LwU8    header_size;
    LwU8    entry_count;
    LwU8    entry_size;
    LwU16   ext_gpio_master;
} LWSWITCH_GPIO_TABLE;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_GPIO_ENTRY, 1)
{
    LwU8    pin;
    LwU8    function;
    LwU8    output;
    LwU8    input;
    LwU8    misc;
} LWSWITCH_GPIO_ENTRY;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_GPIO_ENTRY_PIN_NUM                   5:0
#define LWSWITCH_GPIO_ENTRY_PIN_IO_TYPE               6:6
#define LWSWITCH_GPIO_ENTRY_PIN_INIT_STATE            7:7

#define LWSWITCH_GPIO_ENTRY_FUNCTION                  7:0
#define LWSWITCH_GPIO_ENTRY_FUNCTION_THERMAL_EVENT    17
#define LWSWITCH_GPIO_ENTRY_FUNCTION_OVERTEMP         35
#define LWSWITCH_GPIO_ENTRY_FUNCTION_THERMAL_ALERT    52
#define LWSWITCH_GPIO_ENTRY_FUNCTION_THERMAL_CRITICAL 53
#define LWSWITCH_GPIO_ENTRY_FUNCTION_POWER_ALERT      76
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID0    209
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID1    210
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID2    211
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID3    212
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID4    213
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID5    214
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID6    215
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID7    216
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID8    217
#define LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID9    218
#define LWSWITCH_GPIO_ENTRY_FUNCTION_SKIP_ENTRY      255

#define LWSWITCH_GPIO_ENTRY_OUTPUT                    7:0

#define LWSWITCH_GPIO_ENTRY_INPUT_HW_SELECT           4:0
#define LWSWITCH_GPIO_ENTRY_INPUT_HW_SELECT_NONE            0
#define LWSWITCH_GPIO_ENTRY_INPUT_HW_SELECT_THERMAL_ALERT   22
#define LWSWITCH_GPIO_ENTRY_INPUT_HW_SELECT_POWER_ALERT     23
#define LWSWITCH_GPIO_ENTRY_INPUT_GSYNC               5:5
#define LWSWITCH_GPIO_ENTRY_INPUT_OPEN_DRAIN          6:6
#define LWSWITCH_GPIO_ENTRY_INPUT_PWM                 7:7
//#define LWSWITCH_GPIO_ENTRY_INPUT_3V3                ?:?

#define LWSWITCH_GPIO_ENTRY_MISC_LOCK                 3:0
#define LWSWITCH_GPIO_ENTRY_MISC_IO                   7:4
#define LWSWITCH_GPIO_ENTRY_MISC_IO_UNUSED              0x0
#define LWSWITCH_GPIO_ENTRY_MISC_IO_ILW_OUT             0x1
#define LWSWITCH_GPIO_ENTRY_MISC_IO_ILW_OUT_TRISTATE    0x3
#define LWSWITCH_GPIO_ENTRY_MISC_IO_OUT                 0x4
#define LWSWITCH_GPIO_ENTRY_MISC_IO_IN_STEREO_TRISTATE  0x6
#define LWSWITCH_GPIO_ENTRY_MISC_IO_ILW_OUT_TRISTATE_LO 0x9
#define LWSWITCH_GPIO_ENTRY_MISC_IO_ILW_IN              0xB
#define LWSWITCH_GPIO_ENTRY_MISC_IO_OUT_TRISTATE        0xC
#define LWSWITCH_GPIO_ENTRY_MISC_IO_IN                  0xE

#define LWSWITCH_I2C_VERSION            0x40

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_I2C_TABLE, 1)
{
    LwU8    version;
    LwU8    header_size;
    LwU8    entry_count;
    LwU8    entry_size;
    LwU8    flags;
} LWSWITCH_I2C_TABLE;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_I2C_ENTRY, 1)
{
    LwU32   device;
} LWSWITCH_I2C_ENTRY;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_I2C_ENTRY_TYPE         7:0
#define LWSWITCH_I2C_ENTRY_ADDRESS      15:8
#define LWSWITCH_I2C_ENTRY_RESERVED1    19:16
#define LWSWITCH_I2C_ENTRY_PORT_1       20:20
#define LWSWITCH_I2C_ENTRY_WR_ACCESS    23:21
#define LWSWITCH_I2C_ENTRY_RD_ACCESS    26:24
#define LWSWITCH_I2C_ENTRY_PORT_2       27:27
#define LWSWITCH_I2C_ENTRY_RESERVED2    31:28

#define LWSWITCH_CCB_VERSION            0x41

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_CCB_TABLE, 1)
{
    LwU8    version;
    LwU8    header_size;
    LwU8    entry_count;
    LwU8    entry_size;
    LwU8    comm_port[4];
} LWSWITCH_CCB_TABLE;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

LWSWITCH_STRUCT_PACKED_ALIGNED(_LWSWITCH_CCB_ENTRY, 1)
{
    LwU32   device;
} LWSWITCH_CCB_ENTRY;
LWSWITCH_STRUCT_PACKED_ALIGNED_SUFFIX

#define LWSWITCH_CCB_DEVICE_I2C_PORT    4:0
#define LWSWITCH_CCB_DEVICE_DPAUX       9:5
#define LWSWITCH_CCB_DEVICE_VOLTAGE     10:10
#define LWSWITCH_CCB_DEVICE_RESERVED    27:11
#define LWSWITCH_CCB_DEVICE_I2C_SPEED   31:28

#define LWSWITCH_CCB_DEVICE_I2C_SPEED_DEFAULT   0x0
#define LWSWITCH_CCB_DEVICE_I2C_SPEED_100KHZ    0x1
#define LWSWITCH_CCB_DEVICE_I2C_SPEED_200KHZ    0x2
#define LWSWITCH_CCB_DEVICE_I2C_SPEED_400KHZ    0x3
#define LWSWITCH_CCB_DEVICE_I2C_SPEED_800KHZ    0x4
#define LWSWITCH_CCB_DEVICE_I2C_SPEED_1600KHZ   0x5
#define LWSWITCH_CCB_DEVICE_I2C_SPEED_3400KHZ   0x6
#define LWSWITCH_CCB_DEVICE_I2C_SPEED_60KHZ     0x7
#define LWSWITCH_CCB_DEVICE_I2C_SPEED_300KHZ    0x8

//
// Firmware data
//

#define LWSWITCH_PRODUCT_NAME_MAX_LEN       64

typedef struct
{
    LwBool valid;
    LwU32 ref_min_mhz;
    LwU32 ref_max_mhz;
    LwU32 vco_min_mhz;
    LwU32 vco_max_mhz;
    LwU32 update_min_mhz;
    LwU32 update_max_mhz;
    LwU32 m_min;
    LwU32 m_max;
    LwU32 n_min;
    LwU32 n_max;
    LwU32 pl_min;
    LwU32 pl_max;
} LWSWITCH_PLL_LIMITS;

typedef struct
{
    LwBool valid;
    LwU32  i2c_speed;
    LwBool i2c_33v;
} LWSWITCH_I2C_PORT;

#define LWSWITCH_MAX_I2C_DEVICES    16

typedef struct
{
    LwU32   pin;
    LwU32   function;
    LwU32   hw_select;
    LwU32   misc;
} LWSWITCH_GPIO_INFO;

#define LWSWITCH_MAX_GPIO_PINS      25

typedef struct
{
    LwU32   firmware_size;

    // ROM Header
    LwU16   pci_vendor_id;
    LwU16   pci_device_id;
    LwU16   pci_system_vendor_id;
    LwU16   pci_system_device_id;

    // Firmware data
    struct
    {
        LwBool bridge_fw_found;
        LwU32 firmware_version;
        LwU8  oem_version;
        char  BIOS_MOD_date[8];
        LwBool fw_release_build;
        char  product_name[LWSWITCH_PRODUCT_NAME_MAX_LEN+1];
        LwU16 instance_id;
    } bridge;

    // Clocks
    struct
    {
        LwBool clocks_found;
        LWSWITCH_PLL_LIMITS sys_pll;
    } clocks;

    // Internal
    struct
    {
        LwBool internal_found;
        LwU32 int_firmware_version;
        LwU8  int_oem_version;
        LwU32 int_p4_cl;
        LwU16 int_board_id;
        char  int_build_date[8];
        char  int_chip_sku[3];
        char  int_chip_sku_modifier;
        char  int_project[4];
        char  int_project_sku[4];
        char  int_project_sku_modifier;
    } internal;

    // LWLink init
    struct 
    {
        LwBool link_config_found;
        LwU64 link_enable_mask;             // 1 = enabled
        LwU64 link_ac_coupled_mask;         // 0 = DC, 1 = AC
    } lwlink;

    // DCB
    struct
    {
        LwBool              dcb_found;
        LWSWITCH_I2C_PORT   i2c[LWSWITCH_MAX_I2C_PORTS];
        LwU32               i2c_device_count;
        LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE i2c_device[LWSWITCH_MAX_I2C_DEVICES];
        LwU32               gpio_pin_count;
        LWSWITCH_GPIO_INFO  gpio_pin[LWSWITCH_MAX_GPIO_PINS];
    } dcb;

} LWSWITCH_FIRMWARE;

#define LWSWITCH_FIRMWARE_BRIDGE_INSTANCE_ID_UNKNOWN    0xFFFF
#define LWSWITCH_FIRMWARE_BRIDGE_INSTANCE_ID_NORMAL     0xFFFE

void
lwswitch_read_rom_tables
(
    lwswitch_device *device,
    LWSWITCH_FIRMWARE *firmware
);


#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 

#if !defined(BIOSTYPES_H_FILE)
#define bios_U008  LwU32
#define bios_U016  LwU32
#define bios_U032  LwU32
#define bios_S008  LwS32
#define bios_S016  LwS32
#define bios_S032  LwS32
#endif // !defined(BIOSTYPES_H_FILE)

/**************************************************************************************************************
*   Description:
*       Definitions of BIOS BIT structures as defined starting in Core 5
*
**************************************************************************************************************/
#if !defined(_BIT_H_)
#define BIT_HEADER_ID                     0xB8FF
#define BIT_HEADER_SIGNATURE              0x00544942  // "BIT\0"
#define BIT_HEADER_SIZE_OFFSET            8
#define BIT_HEADER_LATEST_KNOWN_VERSION   0x100
#endif // !defined(_BIT_H_)

#define PCI_ROM_HEADER_SIZE               0x18
#define PCI_DATA_STRUCT_SIZE              0x1c
#define PCI_ROM_HEADER_PCI_DATA_SIZE      (PCI_ROM_HEADER_SIZE + PCI_DATA_STRUCT_SIZE) // ROM Header + PCI Dat Structure size
#define PCI_EXP_ROM_SIGNATURE             0xaa55
#define PCI_DATA_STRUCT_SIGNATURE         0x52494350 // "PCIR" in dword format

#define LWLINK_CONFIG_DATA_HEADER_VER_20    0x2
#define LWLINK_CONFIG_DATA_HEADER_20_SIZE   8
#define LWLINK_CONFIG_DATA_HEADER_20_FMT    "6b1w"

typedef struct _PCI_DATA_STRUCT
{
    bios_U032       sig;                    //  00h: Signature, the string "PCIR" or LWPU's alternate "NPDS"
    bios_U016       vendorID;               //  04h: Vendor Identification
    bios_U016       deviceID;               //  06h: Device Identification
    bios_U016       deviceListPtr;          //  08h: Device List Pointer
    bios_U016       pciDataStructLen;       //  0Ah: PCI Data Structure Length
    bios_U008       pciDataStructRev;       //  0Ch: PCI Data Structure Revision
    bios_U008       classCode[3];           //  0Dh: Class Code
    bios_U016       imageLen;               //  10h: Image Length (units of 512 bytes)
    bios_U016       vendorRomRev;           //  12h: Revision Level of the Vendor's ROM
    bios_U008       codeType;               //  14h: holds NBSI_OBJ_CODE_TYPE (0x70) and others
    bios_U008       lastImage;              //  15h: Last Image Indicator: bit7=1 is lastImage
    bios_U016       maxRunTimeImageLen;     //  16h: Maximum Run-time Image Length (units of 512 bytes)
    bios_U016       configUtilityCodePtr;   //  18h: Pointer to Configurations Utility Code Header
    bios_U016       CMDTFCLPEntryPointPtr;  //  1Ah: Pointer to DMTF CLP Entry Point
} PCI_DATA_STRUCT, *PPCI_DATA_STRUCT;
#define PCI_DATA_STRUCT_FMT "1d4w4b2w2b3w"

// BIT_TOKEN_LWINIT_PTRS       0x49 // 'I' Initialization Table Pointers
struct BIT_DATA_LWINIT_PTRS_V1
{
   bios_U016 InitScriptTablePtr;      // Init script table pointer
   bios_U016 MacroIndexTablePtr;      // Macro index table pointer
   bios_U016 MacroTablePtr;           // Macro table pointer
   bios_U016 ConditionTablePtr;       // Condition table pointer
   bios_U016 IoConditionTablePtr;     // IO Condition table pointer
   bios_U016 IoFlagConditionTablePtr; // IO Flag Condition table pointer
   bios_U016 InitFunctionTablePtr;    // Init Function table pointer
   bios_U016 VBIOSPrivateTablePtr;    // VBIOS private table pointer
   bios_U016 DataArraysTablePtr;      // Data arrays table pointer
   bios_U016 PCIESettingsScriptPtr;   // PCI-E settings script pointer
   bios_U016 DevinitTablesPtr;        // Pointer to tables required by Devinit opcodes
   bios_U016 DevinitTablesSize;       // Size of tables required by Devinit opcodes
   bios_U016 BootScriptsPtr;          // Pointer to Devinit Boot Scripts
   bios_U016 BootScriptsSize;         // Size of Devinit Boot Scripts
   bios_U016 LwlinkConfigDataPtr;     // Pointer to LWLink Config Data
};
#define BIT_DATA_LWINIT_PTRS_V1_30_FMT "15w"
typedef struct BIT_DATA_LWINIT_PTRS_V1 BIT_DATA_LWINIT_PTRS_V1;

#define BIT_TOKEN_BIOSDATA          0x42 // 'B' BIOS Data
#define BIT_TOKEN_LWINIT_PTRS       0x49 // 'I'
#define BIT_TOKEN_INTERNAL_USE      0x69 // 'i' Internal Use Only Data

struct BIT_HEADER_V1_00
{
    bios_U016 Id;            // BMP=0x7FFF/BIT=0xB8FF
    bios_U032 Signature;     // 0x00544942 - BIT Data Structure Signature
    bios_U016 BCD_Version;   // BIT Version - 0x0100 for 1.00
    bios_U008 HeaderSize;    // This version is 12 bytes long
    bios_U008 TokenSize;     // This version has 6 byte long Tokens
    bios_U008 TokenEntries;  // Number of Entries
    bios_U008 HeaderChksum;  // 0 Checksum of the header
};
#define BIT_HEADER_V1_00_FMT "1w1d1w4b"
typedef struct BIT_HEADER_V1_00 BIT_HEADER_V1_00;

struct BIT_TOKEN_V1_00
{
    bios_U008 TokenId;
    bios_U008 DataVersion;
    bios_U016 DataSize;
    bios_U016 DataPtr;
};
#define BIT_TOKEN_V1_00_FMT "2b2w"
typedef struct BIT_TOKEN_V1_00 BIT_TOKEN_V1_00;


// BIT_TOKEN_BIOSDATA          0x42 // 'B' BIOS Data
struct BIT_DATA_BIOSDATA_V1
{
    bios_U032 Version;                // BIOS Binary Version Ex. 5.40.00.01.12 = 0x05400001
    bios_U008 OemVersion;             // OEM Version Number  Ex. 5.40.00.01.12 = 0x12
                                      // OEM can override the two fields above
    bios_U008 Checksum;               // Filled by MakeVGA
    bios_U016 Int15CallbacksPost;     //
    bios_U016 Int15CallbacksSystem;   //
    bios_U016 BoardId;                //
    bios_U016 FrameCount;             // Frame count for signon message delay
    bios_U008 BiosmodDate[8];         // '00/00/04' Date BIOSMod was last run
};
#define BIT_DATA_BIOSDATA_V1_FMT    "1d2b4w8b"
typedef struct BIT_DATA_BIOSDATA_V1 BIT_DATA_BIOSDATA_V1;

struct BIT_DATA_BIOSDATA_V2
{
    bios_U032 Version;                // BIOS Binary Version Ex. 5.40.00.01.12 = 0x05400001
    bios_U008 OemVersion;             // OEM Version Number  Ex. 5.40.00.01.12 = 0x12
    // OEM can override the two fields above
    bios_U008 Checksum;               // Filled by MakeVGA
    bios_U016 Int15CallbacksPost;     //
    bios_U016 Int15CallbacksSystem;   //
    bios_U016 FrameCount;             // Frame count for signon message delay
    bios_U032 Reserved1;
    bios_U032 Reserved2;
    bios_U008 MaxHeadsAtPost;
    bios_U008 MemorySizeReport;
    bios_U008 HorizontalScaleFactor;
    bios_U008 VerticalScaleFactor;
    bios_U016 DataTablePtr;
    bios_U016 RomPackPtr;
    bios_U016 AppliedRomPacksPtr;
    bios_U008 AppliedRomPackMax;
    bios_U008 AppliedRomPackCount;
    bios_U008 ModuleMapExternal;
    bios_U032 CompressionInfoPtr;
};
#define BIT_DATA_BIOSDATA_V2_FMT "1d2b3w2d4b3w3b1d"
typedef struct BIT_DATA_BIOSDATA_V2 BIT_DATA_BIOSDATA_V2;

#ifndef PCI_VENDOR_ID_LWIDIA
#define PCI_VENDOR_ID_LWIDIA            0x10DE
#endif


//#define BIT_TOKEN_INTERNAL_USE      0x69 // 'i' Internal Use Only Data
struct BIT_DATA_INTERNAL_USE_V1
{
    bios_U032 Version;                // BIOS Binary Version Ex. 5.40.00.01.12 = 0x05400001
    bios_U008 OemVersion;             // OEM Version Number  Ex. 5.40.00.01.12 = 0x12
                                      // OEM cannot override the two fields above
    bios_U016 Features;               // BIOS Compiled Features
    bios_U032 P4MagicNumber;          // Perforce Checkin Number of release.
};
#define BIT_DATA_INTERNAL_USE_V1_FMT "1d1b1w1d"
typedef struct BIT_DATA_INTERNAL_USE_V1 BIT_DATA_INTERNAL_USE_V1;

//
// This definition is incomplete as not all fields are required
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
// See https://wiki.lwpu.com/engwiki/index.php/VBIOS/VBIOS_Various_Topics/VBIOS_BIT#BIT_INTERNAL_USE_ONLY_PTRS_.28Version_2.29
#endif
//
struct BIT_DATA_INTERNAL_USE_V2
{
    bios_U032 Version;                // BIOS Binary Version Ex. 5.40.00.01.12 = 0x05400001
    bios_U008 OemVersion;             // OEM Version Number  Ex. 5.40.00.01.12 = 0x12
                                      // OEM cannot override the two fields above
    bios_U016 Features;               // BIOS Compiled Features
    bios_U032 P4MagicNumber;          // Perforce Checkin Number of release.
    bios_U016 BoardID;                // Id of the board the BIOS is built on  
};
#define BIT_DATA_INTERNAL_USE_V2_FMT "1d1b1w1d1w"
typedef struct BIT_DATA_INTERNAL_USE_V2 BIT_DATA_INTERNAL_USE_V2;

typedef struct _lwlink_Config_Data_Header_20
{
    bios_U008 Version;           // LWLink Config Data Structure version
    bios_U008 HeaderSize;        // Size of header
    bios_U008 BaseEntrySize;
    bios_U008 BaseEntryCount;
    bios_U008 LinkEntrySize;
    bios_U008 LinkEntryCount;
    bios_U016 Reserved;          // Reserved
} LWLINK_CONFIG_DATA_HEADER_20, *PLWLINK_CONFIG_DATA_HEADER_20;

#define LW_LWLINK_VBIOS_PARAM0_LINK                             0:0
#define LW_LWLINK_VBIOS_PARAM0_LINK_ENABLE                      0x0
#define LW_LWLINK_VBIOS_PARAM0_LINK_DISABLE                     0x1
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW_LWLINK_VBIOS_PARAM0_ACTIVE_REPEATER                  1:1
#define LW_LWLINK_VBIOS_PARAM0_ACTIVE_REPEATER_NOT_PRESENT      0x0
#define LW_LWLINK_VBIOS_PARAM0_ACTIVE_REPEATER_PRESENT          0x1
#else
#define LW_LWLINK_VBIOS_PARAM0_RESERVED1                        1:1
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW_LWLINK_VBIOS_PARAM0_ACDC_MODE                        2:2
#define LW_LWLINK_VBIOS_PARAM0_ACDC_MODE_DC                     0x0
#define LW_LWLINK_VBIOS_PARAM0_ACDC_MODE_AC                     0x1
#define LW_LWLINK_VBIOS_PARAM0_RECEIVER_DETECT                  3:3
#define LW_LWLINK_VBIOS_PARAM0_RECEIVER_DETECT_DISABLE          0x0
#define LW_LWLINK_VBIOS_PARAM0_RECEIVER_DETECT_ENABLE           0x1
#define LW_LWLINK_VBIOS_PARAM0_RESTORE_PHY_TRAINING             4:4
#define LW_LWLINK_VBIOS_PARAM0_RESTORE_PHY_TRAINING_DISABLE     0x0
#define LW_LWLINK_VBIOS_PARAM0_RESTORE_PHY_TRAINING_ENABLE      0x1
#define LW_LWLINK_VBIOS_PARAM0_SLM                              5:5
#define LW_LWLINK_VBIOS_PARAM0_SLM_DISABLE                      0x0
#define LW_LWLINK_VBIOS_PARAM0_SLM_ENABLE                       0x1
#define LW_LWLINK_VBIOS_PARAM0_L2                               6:6
#define LW_LWLINK_VBIOS_PARAM0_L2_DISABLE                       0x0
#define LW_LWLINK_VBIOS_PARAM0_L2_ENABLE                        0x1
#define LW_LWLINK_VBIOS_PARAM0_RESERVED2                        7:7

#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE                        7:0
#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE_50_00000               0x00
#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE_16_00000               0x01
#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE_20_00000               0x02
#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE_25_00000               0x03
#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE_25_78125               0x04
#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE_32_00000               0x05
#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE_40_00000               0x06
#define LW_LWLINK_VBIOS_PARAM1_LINE_RATE_53_12500               0x07

#define LW_LWLINK_VBIOS_PARAM2_LINE_CODE_MODE                   7:0
#define LW_LWLINK_VBIOS_PARAM2_LINE_CODE_MODE_NRZ               0x00
#define LW_LWLINK_VBIOS_PARAM2_LINE_CODE_MODE_NRZ_128B130       0x01
#define LW_LWLINK_VBIOS_PARAM2_LINE_CODE_MODE_NRZ_PAM4          0x03

#define LW_LWLINK_VBIOS_PARAM3_REFERENCE_CLOCK_MODE                     1:0
#define LW_LWLINK_VBIOS_PARAM3_REFERENCE_CLOCK_MODE_COMMON              0x0
#define LW_LWLINK_VBIOS_PARAM3_REFERENCE_CLOCK_MODE_RSVD                0x1
#define LW_LWLINK_VBIOS_PARAM3_REFERENCE_CLOCK_MODE_NON_COMMON_NO_SS    0x2
#define LW_LWLINK_VBIOS_PARAM3_REFERENCE_CLOCK_MODE_NON_COMMON_SS       0x3

#define LW_LWLINK_VBIOS_PARAM3_RESERVED1                        3:2
#define LW_LWLINK_VBIOS_PARAM3_CLOCK_MODE_BLOCK_CODE            5:4
#define LW_LWLINK_VBIOS_PARAM3_CLOCK_MODE_BLOCK_CODE_OFF        0x0
#define LW_LWLINK_VBIOS_PARAM3_CLOCK_MODE_BLOCK_CODE_ECC96      0x1
#define LW_LWLINK_VBIOS_PARAM3_CLOCK_MODE_BLOCK_CODE_ECC88      0x2
#define LW_LWLINK_VBIOS_PARAM3_RESERVED2                        7:6

#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM                               7:0
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_RSVD                          0x00
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_A0_SINGLE_PRESENT             0x01
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_A1_PRESENT_ARRAY              0x02
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_A2_FINE_GRAINED_EXHAUSTIVE    0x04
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_A3_RSVD                       0x08
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_A4_FOM_CENTRIOD               0x10
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_A5_RSVD                       0x20
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_A6_RSVD                       0x40
#define LW_LWLINK_VBIOS_PARAM4_TXTRAIN_OPTIMIZATION_ALGORITHM_A7_RSVD                       0x80

#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_ADJUSTMENT_ALGORITHM                                 4:0
#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_ADJUSTMENT_ALGORITHM_B0_NO_ADJUSTMENT                0x1
#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_ADJUSTMENT_ALGORITHM_B1_FIXED_ADJUSTMENT             0x2
#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_ADJUSTMENT_ALGORITHM_B2_RSVD                         0x4
#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_ADJUSTMENT_ALGORITHM_B3_RSVD                         0x8

#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_FOM_FORMAT                           7:5
#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_FOM_FORMAT_FOM_A                     0x1
#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_FOM_FORMAT_FOM_B                     0x2
#define LW_LWLINK_VBIOS_PARAM5_TXTRAIN_FOM_FORMAT_FOM_C                     0x4

#define LW_LWLINK_VBIOS_PARAM6_TXTRAIN_MINIMUM_TRAIN_TIME_MANTISSA           3:0
#define LW_LWLINK_VBIOS_PARAM6_TXTRAIN_MINIMUM_TRAIN_TIME_EXPONENT           7:4

#define LWLINK_CONFIG_DATA_BASEENTRY_FMT "1b"
#define LWLINK_CONFIG_DATA_LINKENTRY_FMT "7b"
// Version 2.0 Link Entry and Base Entry
typedef struct _lwlink_config_data_baseentry_20
{
     LwU8  positionId;
} LWLINK_CONFIG_DATA_BASEENTRY;

typedef struct _lwlink_config_data_linkentry_20
{
    // VBIOS configuration Data
     LwU8  lwLinkparam0;
     LwU8  lwLinkparam1;
     LwU8  lwLinkparam2;
     LwU8  lwLinkparam3;
     LwU8  lwLinkparam4;
     LwU8  lwLinkparam5;
     LwU8  lwLinkparam6;
} LWLINK_CONFIG_DATA_LINKENTRY;


// Union of different VBIOS configuration table formats
typedef union __lwlink_Config_Data_Header
{
    LWLINK_CONFIG_DATA_HEADER_20 ver_20;
} LWLINK_CONFIG_DATA_HEADER, *PLWLINK_CONFIG_DATA_HEADER;

typedef struct _lwlink_vbios_config_data_baseentry_20
{
     bios_U008  positionId;
} LWLINK_VBIOS_CONFIG_DATA_BASEENTRY;

typedef struct _lwlink_vbios_config_data_linkentry_20
{
    // VBIOS configuration Data
     bios_U008  lwLinkparam0;
     bios_U008  lwLinkparam1;
     bios_U008  lwLinkparam2;
     bios_U008  lwLinkparam3;
     bios_U008  lwLinkparam4;
     bios_U008  lwLinkparam5;
     bios_U008  lwLinkparam6;
} LWLINK_VBIOS_CONFIG_DATA_LINKENTRY, *PLWLINK_VBIOS_CONFIG_DATA_LINKENTRY;

//
// LWSwitch driver structures
//

#define LWSWITCH_NUM_BIOS_LWLINK_CONFIG_BASE_ENTRY    12

typedef struct
{
    LWLINK_CONFIG_DATA_BASEENTRY link_vbios_base_entry[LWSWITCH_NUM_BIOS_LWLINK_CONFIG_BASE_ENTRY];
    LWLINK_CONFIG_DATA_LINKENTRY link_vbios_entry[LWSWITCH_NUM_BIOS_LWLINK_CONFIG_BASE_ENTRY][LWSWITCH_MAX_LINK_COUNT];
    LwU32                        identified_Link_entries[LWSWITCH_NUM_BIOS_LWLINK_CONFIG_BASE_ENTRY];
    LwU32                        link_base_entry_assigned;
    LwU64                        vbios_disabled_link_mask;

    LwU32                        bit_address;
    LwU32                        pci_image_address;
    LwU32                        lwlink_config_table_address;
} LWSWITCH_BIOS_LWLINK_CONFIG;

#define LWSWITCH_DCB_PTR_OFFSET 0x36

typedef struct _lwswitch_vbios_dcb_header_41
{
    bios_U008 version;
    bios_U008 header_size;
    bios_U008 entry_count;
    bios_U008 entry_size;
    bios_U016 ccb_block_ptr;
    bios_U032 dcb_signature;
    bios_U016 gpio_table;
    bios_U016 input_devices;
    bios_U016 personal_cinema;
    bios_U016 spread_spectrum;
    bios_U016 i2c_devices;
    bios_U016 connectors;
    bios_U008 flags;
    bios_U016 hdtv;
    bios_U016 switched_outputs;
    bios_U032 display_patch;
    bios_U032 connector_patch;
} LWSWITCH_VBIOS_DCB_HEADER;
#define LWSWITCH_VBIOS_DCB_HEADER_FMT "4b1w1d6w1b2w2d"

typedef struct _lwswitch_vbios_ccb_table_41
{
    bios_U008    version;
    bios_U008    header_size;
    bios_U008    entry_count;
    bios_U008    entry_size;
    bios_U008    comm_port[4];
} LWSWITCH_VBIOS_CCB_TABLE;
#define LWSWITCH_VBIOS_CCB_TABLE_FMT "8b"

typedef struct _lwswitch_vbios_i2c_table_40
{
    bios_U008    version;
    bios_U008    header_size;
    bios_U008    entry_count;
    bios_U008    entry_size;
    bios_U008    flags;
} LWSWITCH_VBIOS_I2C_TABLE;
#define LWSWITCH_I2C_TABLE_FMT "5b"

typedef struct _lwswitch_vbios_i2c_entry
{
    bios_U032   device;
} LWSWITCH_VBIOS_I2C_ENTRY;
#define LWSWITCH_I2C_ENTRY_FMT "1d"

#endif //_ROM_LWSWITCH_H_

