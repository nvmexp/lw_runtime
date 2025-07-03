#include "dcgm_fields.h"
#include "dcgm_fields_internal.h"
#include "hashtable.h"
#include "MurmurHash3.h"
#include <string.h>
#include <malloc.h>
#include "logging.h"
#include "lwml.h"

/* Not using an object on purpose in case this code needs to be ported to C */


/**
 * Width and unit enums for field values.
 */
enum UNIT
{
    DCGM_FIELD_UNIT_TEMP_C,      /* Temperature in Celcius */
    DCGM_FIELD_UNIT_POW_W,       /* Watts */
    DCGM_FIELD_UNIT_BW_KBPS,     /* KB/sec */
    DCGM_FIELD_UNIT_BW_MBPS,     /* MB/sec */
    DCGM_FIELD_UNIT_MILLIJOULES, /* mJ (millijoules) */
};

enum WIDTH 
{
    DCGM_FIELD_WIDTH_40,
    DCGM_FIELD_WIDTH_20,
    DCGM_FIELD_WIDTH_7,
    DCGM_FIELD_WIDTH_5,
    DCGM_FIELD_WIDTH_16,
    DCGM_FIELD_WIDTH_10
};

/* Has this module been initialized? */
static int dcgmFieldsInitialized = 0;
hashtable_t dcgmFieldsKeyToIdMap = {0};

/*****************************************************************************/
/* Static field information. Call DcgmFieldsPopulateOneFieldWithFormatting() to add entries to this table */
static dcgm_field_meta_t *dcgmFieldMeta[DCGM_FI_MAX_FIELDS] = {0};

/*****************************************************************************/

/**
 * The function returns int value of enum for width
 */
static int getWidthForEnum(enum WIDTH enumVal)
{
    switch(enumVal)
    {
        case DCGM_FIELD_WIDTH_40:
            return 40;
            break;
        
        case DCGM_FIELD_WIDTH_20:
            return 20;
            break;
        
        case DCGM_FIELD_WIDTH_7:
            return 7;
            break;
        
        case DCGM_FIELD_WIDTH_5:
            return 5;
            break;
        
        case DCGM_FIELD_WIDTH_16:
            return 16;
            break;
        
        case DCGM_FIELD_WIDTH_10:
            return 10;
            break;
        
        default:
            return 10;
            break; 
    }
}

/**
 * The function returns string value of enum for Units.
 */
static char* getTextForEnum(enum UNIT enumVal)
{
    switch (enumVal)
    {
        case DCGM_FIELD_UNIT_TEMP_C:
            return " C";
            break;

        case DCGM_FIELD_UNIT_POW_W:
            return " W";
            break;

        case DCGM_FIELD_UNIT_BW_KBPS:
            return "KB/s";
            break;

        case DCGM_FIELD_UNIT_BW_MBPS:
            return "MB/s";
            break;

        case DCGM_FIELD_UNIT_MILLIJOULES:
            return " mJ";
            break;

        default:
            return "";
            break;
    }
}

/*****************************************************************************/
static int DcgmFieldsPopulateOneFieldWithFormatting(unsigned short fieldId, char fieldType,
                                      unsigned char  size, char *tag,int scope, int lwmlFieldId, 
                                      char *shortName, char *unit, short width)
{
    dcgm_field_meta_t *fieldMeta = 0;

    if(!fieldId || fieldId >= DCGM_FI_MAX_FIELDS)
        return -1;

    fieldMeta = (dcgm_field_meta_t *)malloc(sizeof(*fieldMeta));
    if(!fieldMeta)
        return -3; /* Out of memory */

    memset(fieldMeta, 0, sizeof(*fieldMeta));

    fieldMeta->fieldId = fieldId;
    fieldMeta->fieldType = fieldType;
    fieldMeta->size = size;
    strncpy(fieldMeta->tag, tag, sizeof(fieldMeta->tag)-1);
    fieldMeta->scope = scope;
    fieldMeta->lwmlFieldId = lwmlFieldId;
    
    fieldMeta->valueFormat = (dcgm_field_output_format_t*)malloc(sizeof(*fieldMeta->valueFormat));
    memset(fieldMeta->valueFormat, 0, sizeof(*fieldMeta->valueFormat));
    
    strncpy(fieldMeta->valueFormat->shortName,shortName, sizeof(fieldMeta->valueFormat->shortName));
    strncpy(fieldMeta->valueFormat->unit, unit, sizeof(fieldMeta->valueFormat->unit));
    fieldMeta->valueFormat->width = width;

    dcgmFieldMeta[fieldMeta->fieldId] = fieldMeta;
    return 0;
}

/* Do static initialization of the global field list */
static int DcgmFieldsPopulateFieldTableWithFormatting(void)
{
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DRIVER_VERSION, DCGM_FT_STRING, 0, "driver_version", DCGM_FS_GLOBAL, 0, "DRVER", "#", getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_LWML_VERSION, DCGM_FT_STRING, 0, "lwml_version", DCGM_FS_GLOBAL, 0, "LWVER", "#", getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROCESS_NAME, DCGM_FT_STRING, 0, "process_name", DCGM_FS_GLOBAL, 0, "PRNAM", "", getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_COUNT, DCGM_FT_INT64, 8, "device_count", DCGM_FS_GLOBAL, 0, "DVCNT","",getWidthForEnum(DCGM_FIELD_WIDTH_5)); 
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_NAME, DCGM_FT_STRING, 0, "name", DCGM_FS_DEVICE, 0, "DVNAM","",getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BRAND, DCGM_FT_STRING, 0, "brand", DCGM_FS_DEVICE, 0, "DVBRN","",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWML_INDEX, DCGM_FT_INT64, 8, "lwml_index", DCGM_FS_DEVICE, 0, "LWIDX","",getWidthForEnum(DCGM_FIELD_WIDTH_5)); //
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SERIAL, DCGM_FT_STRING, 0, "serial_number", DCGM_FS_DEVICE, 0, "SRNUM","",getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_AFFINITY_0, DCGM_FT_INT64, 8, "cpu_affinity_0", DCGM_FS_DEVICE, 0, "CAFF0","",getWidthForEnum(DCGM_FIELD_WIDTH_10)); 
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_AFFINITY_1, DCGM_FT_INT64, 8, "cpu_affinity_1", DCGM_FS_DEVICE, 0, "CAFF1","",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_AFFINITY_2, DCGM_FT_INT64, 8, "cpu_affinity_2", DCGM_FS_DEVICE, 0, "CAFF2","",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CPU_AFFINITY_3, DCGM_FT_INT64, 8, "cpu_affinity_3", DCGM_FS_DEVICE, 0, "CAFF3","",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_UUID, DCGM_FT_STRING, 0, "uuid", DCGM_FS_DEVICE, 0, "UUID#", "",getWidthForEnum(DCGM_FIELD_WIDTH_40));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MINOR_NUMBER, DCGM_FT_INT64, 8, "minor_number", DCGM_FS_DEVICE, 0, "MNNUM", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_OEM_INFOROM_VER, DCGM_FT_STRING, 0, "oem_inforom_version", DCGM_FS_DEVICE, 0, "OEMVR","#",getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_INFOROM_VER, DCGM_FT_STRING, 0, "ecc_inforom_version", DCGM_FS_DEVICE, 0, "EIVER", "#", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_INFOROM_VER, DCGM_FT_STRING, 0, "power_inforom_version", DCGM_FS_DEVICE, 0, "PIVER", "#", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_INFOROM_IMAGE_VER, DCGM_FT_STRING, 0, "inforom_image_version", DCGM_FS_DEVICE, 0, "IIVER", "#", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_INFOROM_CONFIG_CHECK, DCGM_FT_INT64, 8, "inforom_config_checksum", DCGM_FS_DEVICE, 0, "CCSUM", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCI_BUSID, DCGM_FT_STRING, 0, "pci_busid", DCGM_FS_DEVICE, 0, "PCBID", "#", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCI_COMBINED_ID, DCGM_FT_INT64, 8, "pci_combined_id", DCGM_FS_DEVICE, 0, "PCCID", "#", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCI_SUBSYS_ID, DCGM_FT_INT64, 8, "pci_subsys_id", DCGM_FS_DEVICE, 0, "PCSID", "#", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_TX_THROUGHPUT, DCGM_FT_INT64, 8, "pcie_tx_throughput", DCGM_FS_DEVICE, 0, "TXTPT", getTextForEnum(DCGM_FIELD_UNIT_BW_KBPS), getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_RX_THROUGHPUT, DCGM_FT_INT64, 8, "pcie_rx_throughput", DCGM_FS_DEVICE, 0, "RXTPT", getTextForEnum(DCGM_FIELD_UNIT_BW_KBPS), getWidthForEnum(DCGM_FIELD_WIDTH_7));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_REPLAY_COUNTER, DCGM_FT_INT64, 8, "pcie_replay_counter", DCGM_FS_DEVICE, 0, "RPCTR","#", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SM_CLOCK, DCGM_FT_INT64, 8, "sm_clock", DCGM_FS_DEVICE, 0, "SMCLK","",getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_CLOCK, DCGM_FT_INT64, 8, "memory_clock", DCGM_FS_DEVICE, 0, "MMCLK", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VIDEO_CLOCK, DCGM_FT_INT64, 8, "video_clock", DCGM_FS_DEVICE, 0, "VICLK", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_APP_SM_CLOCK, DCGM_FT_INT64, 8, "sm_app_clock", DCGM_FS_DEVICE, 0, "SACLK", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_APP_MEM_CLOCK, DCGM_FT_INT64, 8, "mem_app_clock", DCGM_FS_DEVICE, 0, "MACLK", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, DCGM_FT_INT64, 0,
                                             "lwrrent_clock_throttle_reasons", DCGM_FS_DEVICE, 0, "DVCCTR", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MAX_SM_CLOCK, DCGM_FT_INT64, 8, "sm_max_clock", DCGM_FS_DEVICE, 0, "SMMAX","",getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MAX_MEM_CLOCK, DCGM_FT_INT64, 8, "memory_max_clock", DCGM_FS_DEVICE, 0, "MMMAX", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MAX_VIDEO_CLOCK, DCGM_FT_INT64, 8, "video_max_clock", DCGM_FS_DEVICE, 0, "VIMAX", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_AUTOBOOST, DCGM_FT_INT64, 8, "autoboost", DCGM_FS_DEVICE, 0, "ATBST", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_TEMP, DCGM_FT_INT64, 8, "gpu_temp", DCGM_FS_DEVICE, 0, "TMPTR", 
                                            (char*)getTextForEnum(DCGM_FIELD_UNIT_TEMP_C), getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SLOWDOWN_TEMP, DCGM_FT_INT64, 8, "slowdown_temp", DCGM_FS_DEVICE, 0, "SDTMP", getTextForEnum(DCGM_FIELD_UNIT_TEMP_C), getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SHUTDOWN_TEMP, DCGM_FT_INT64, 8, "shutdown_temp", DCGM_FS_DEVICE, 0, "SHTMP", getTextForEnum(DCGM_FIELD_UNIT_TEMP_C), getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_MGMT_LIMIT, DCGM_FT_DOUBLE, 8, "power_management_limit", DCGM_FS_DEVICE, 0, "PMLMT", getTextForEnum(DCGM_FIELD_UNIT_POW_W), getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN, DCGM_FT_DOUBLE, 8, "power_management_limit_min", DCGM_FS_DEVICE, 0, "PMMIN", getTextForEnum(DCGM_FIELD_UNIT_POW_W), getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX, DCGM_FT_DOUBLE, 8, "power_management_limit_max", DCGM_FS_DEVICE, 0, "PMMAX", getTextForEnum(DCGM_FIELD_UNIT_POW_W), getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF, DCGM_FT_DOUBLE, 8, "power_management_limit_default", DCGM_FS_DEVICE, 0, "PMDEF", getTextForEnum(DCGM_FIELD_UNIT_POW_W), getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_USAGE, DCGM_FT_DOUBLE, 8, "power_usage", DCGM_FS_DEVICE, 0, "POWER", getTextForEnum(DCGM_FIELD_UNIT_POW_W), getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, DCGM_FT_INT64, 8, "total_energy_consumption", DCGM_FS_DEVICE, LWML_FI_DEV_TOTAL_ENERGY_CONSUMPTION,"TOTEC", getTextForEnum(DCGM_FIELD_UNIT_MILLIJOULES),getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ENFORCED_POWER_LIMIT, DCGM_FT_DOUBLE, 8, "enforced_power_limit", DCGM_FS_DEVICE, 0, "EPLMT", getTextForEnum(DCGM_FIELD_UNIT_POW_W), getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PSTATE, DCGM_FT_INT64, 8, "pstate", DCGM_FS_DEVICE, 0, "PSTAT", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FAN_SPEED, DCGM_FT_INT64, 8, "fan_speed", DCGM_FS_DEVICE, 0, "FANSP", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_COMPUTE_MODE, DCGM_FT_INT64, 8, "compute_mode", DCGM_FS_DEVICE, 0, "CMMOD", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_LWRRENT, DCGM_FT_INT64, 8, "ecc", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_LWRRENT, "ECLWR", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_PENDING, DCGM_FT_INT64, 8, "ecc_pending", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_PENDING, "ECPEN", "",getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, DCGM_FT_INT64, 8, "ecc_sbe_volatile_total", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_VOL_TOTAL, "ESVTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, DCGM_FT_INT64, 8, "ecc_dbe_volatile_total", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_VOL_TOTAL,"EDVTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_TOTAL, DCGM_FT_INT64, 8, "ecc_sbe_aggregate_total", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_AGG_TOTAL, "ESATL", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_TOTAL, DCGM_FT_INT64, 8, "ecc_dbe_aggregate_total", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_AGG_TOTAL, "EDATL","",getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_L1, DCGM_FT_INT64, 8, "ecc_sbe_volatile_l1", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_VOL_L1, "ESVL1","",getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_L1, DCGM_FT_INT64, 8, "ecc_dbe_volatile_l1", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_VOL_L1, "EDVL1","",getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_L2, DCGM_FT_INT64, 8, "ecc_sbe_volatile_l2", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_VOL_L2, "ESVL2", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_L2, DCGM_FT_INT64, 8, "ecc_dbe_volatile_l2", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_VOL_L2, "EDVL2", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_DEV, DCGM_FT_INT64, 8, "ecc_sbe_volatile_device", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_VOL_DEV, "ESVDV", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_DEV, DCGM_FT_INT64, 8, "ecc_dbe_volatile_device", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_VOL_DEV, "EDVDV", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_REG, DCGM_FT_INT64, 8, "ecc_sbe_volatile_register", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_VOL_REG, "ESVRG", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_REG, DCGM_FT_INT64, 8, "ecc_dbe_volatile_register", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_VOL_REG, "EDVRG","", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_VOL_TEX, DCGM_FT_INT64, 8, "ecc_sbe_volatile_texture", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_VOL_TEX, "ESVTX", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_VOL_TEX, DCGM_FT_INT64, 8, "ecc_dbe_volatile_texture", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_VOL_TEX, "EDVTX", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_L1, DCGM_FT_INT64, 8, "ecc_sbe_aggregate_l1", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_AGG_L1, "ESAL1", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_L1, DCGM_FT_INT64, 8, "ecc_dbe_aggregate_l1", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_AGG_L1, "EDAL1", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_L2, DCGM_FT_INT64, 8, "ecc_sbe_aggregate_l2", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_AGG_L2, "ESAL2", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_L2, DCGM_FT_INT64, 8, "ecc_dbe_aggregate_l2", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_AGG_L2, "EDAL2", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_DEV, DCGM_FT_INT64, 8, "ecc_sbe_aggregate_device", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_AGG_DEV, "ESADV", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_DEV, DCGM_FT_INT64, 8, "ecc_dbe_aggregate_device", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_AGG_DEV, "EDADV", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_REG, DCGM_FT_INT64, 8, "ecc_sbe_aggregate_register", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_AGG_REG, "ESARG", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_REG, DCGM_FT_INT64, 8, "ecc_dbe_aggregate_register", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_AGG_REG, "EDARG", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_SBE_AGG_TEX, DCGM_FT_INT64, 8, "ecc_sbe_aggregate_texture", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_SBE_AGG_TEX, "ESATX", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ECC_DBE_AGG_TEX, DCGM_FT_INT64, 8, "ecc_dbe_aggregate_texture", DCGM_FS_DEVICE,
                               LWML_FI_DEV_ECC_DBE_AGG_TEX, "EDATX", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_UTIL, DCGM_FT_INT64, 8, "gpu_utilization", DCGM_FS_DEVICE, 0, "GPUTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_COPY_UTIL, DCGM_FT_INT64, 8, "mem_copy_utilization", DCGM_FS_DEVICE, 0, "MLWTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ENC_UTIL, DCGM_FT_INT64, 8, "enc_utilization", DCGM_FS_DEVICE, 0, "ELWTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_DEC_UTIL, DCGM_FT_INT64, 8, "dec_utilization", DCGM_FS_DEVICE, 0, "DLWTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VBIOS_VERSION, DCGM_FT_STRING, 0, "vbios_version", DCGM_FS_DEVICE, 0, "VBVER", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BAR1_TOTAL, DCGM_FT_INT64, 8, "bar1_total", DCGM_FS_DEVICE, 0, "B1TTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BAR1_USED, DCGM_FT_INT64, 8, "bar1_used", DCGM_FS_DEVICE, 0, "B1USE", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BAR1_FREE, DCGM_FT_INT64, 8, "bar1_free", DCGM_FS_DEVICE, 0, "B1FRE", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FB_TOTAL, DCGM_FT_INT64, 8, "fb_total", DCGM_FS_DEVICE, 0, "FBTTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FB_FREE, DCGM_FT_INT64, 8, "fb_free", DCGM_FS_DEVICE, 0, "FBFRE", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FB_USED, DCGM_FT_INT64, 8, "fb_used", DCGM_FS_DEVICE, 0, "FBUSD", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VIRTUAL_MODE, DCGM_FT_INT64, 8, "virtualization_mode", DCGM_FS_DEVICE, 0, "VMODE", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_INSTANCE_IDS, DCGM_FT_BINARY, 0, "active_vgpu_instance_ids", DCGM_FS_DEVICE, 0, "VGIID", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_UTILIZATIONS, DCGM_FT_BINARY, 0, "vgpu_instance_utilizations", DCGM_FS_DEVICE, 0, "VIUTL", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION, DCGM_FT_BINARY, 0, "vgpu_instance_per_process_utilization", DCGM_FS_DEVICE, 0, "VIPPU" , "",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_VM_ID, DCGM_FT_STRING, 0, "vgpu_instance_vm_id", DCGM_FS_DEVICE, 0, "VVMID", "",getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_VM_NAME, DCGM_FT_STRING, 0, "vgpu_instance_vm_name", DCGM_FS_DEVICE, 0, "VMNAM", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_TYPE, DCGM_FT_INT64, 8, "vgpu_instance_type", DCGM_FS_DEVICE, 0, "VITYP", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_UUID, DCGM_FT_STRING, 0, "vgpu_instance_uuid", DCGM_FS_DEVICE, 0, "VUUID", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_DRIVER_VERSION, DCGM_FT_STRING, 0, "vgpu_instance_driver_version", DCGM_FS_DEVICE, 0, "VDVER", "",getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_MEMORY_USAGE, DCGM_FT_INT64, 8, "vgpu_instance_memory_usage", DCGM_FS_DEVICE, 0, "VMUSG","",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_LICENSE_STATUS, DCGM_FT_INT64, 8, "vgpu_instance_license_status", DCGM_FS_DEVICE, 0, "VLCST", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT, DCGM_FT_INT64, 8, "vgpu_instance_frame_rate_limit", DCGM_FS_DEVICE, 0, "VFLIM","", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_PCI_ID, DCGM_FT_STRING, 0, "vgpu_instance_pci_id", DCGM_FS_DEVICE, 0, "VPCIID","", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_ENC_STATS, DCGM_FT_BINARY, 0, "vgpu_instance_enc_stats", DCGM_FS_DEVICE, 0, "VSTAT", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO, DCGM_FT_BINARY, 0, "vgpu_instance_enc_sessions_info", DCGM_FS_DEVICE, 0, "VSINF","", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_FBC_STATS, DCGM_FT_BINARY, 0, "vgpu_instance_fbc_stats", DCGM_FS_DEVICE, 0, "VFSTAT","", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO, DCGM_FT_BINARY, 0, "vgpu_instance_fbc_sessions_info", DCGM_FS_DEVICE, 0, "VFINF","", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SUPPORTED_TYPE_INFO, DCGM_FT_BINARY, 0, "supported_type_info", DCGM_FS_DEVICE, 0, "SPINF", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS, DCGM_FT_BINARY, 0, "creatable_vgpu_type_ids", DCGM_FS_DEVICE, 0, "CGPID","",getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ENC_STATS, DCGM_FT_BINARY, 0, "enc_stats", DCGM_FS_DEVICE, 0, "ENSTA","",getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FBC_STATS, DCGM_FT_BINARY, 0, "fbc_stats", DCGM_FS_DEVICE, 0, "FBCSTA","",getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_FBC_SESSIONS_INFO, DCGM_FT_BINARY, 0, "fbc_sessions_info", DCGM_FS_DEVICE, 0, "FBCINF","",getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_ACCOUNTING_DATA, DCGM_FT_BINARY, 0, "accounting_data", DCGM_FS_DEVICE, 0, "ACCDT","",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_RETIRED_SBE, DCGM_FT_INT64, 8, "retired_pages_sbe", DCGM_FS_DEVICE,
                               0, "RPSBE", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_RETIRED_DBE, DCGM_FT_INT64, 8, "retired_pages_dbe", DCGM_FS_DEVICE,
                               LWML_FI_DEV_RETIRED_DBE, "RPDBE", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GRAPHICS_PIDS, DCGM_FT_BINARY, 0, "graphics_pids", DCGM_FS_DEVICE, 0, "GPIDS", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_COMPUTE_PIDS, DCGM_FT_BINARY, 0, "compute_pids", DCGM_FS_DEVICE, 0, "CMPID", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SUPPORTED_CLOCKS, DCGM_FT_BINARY, 0, "supported_clocks", DCGM_FS_DEVICE, 0, "SPCLK", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_SYNC_BOOST, DCGM_FT_BINARY, 0, "sync_boost", DCGM_FS_GLOBAL, 0, "SYBST", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_RETIRED_PENDING, DCGM_FT_INT64, 8, "retired_pages_pending", DCGM_FS_DEVICE,
                               LWML_FI_DEV_RETIRED_PENDING, "RPPEN","",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_INFOROM_CONFIG_VALID, DCGM_FT_INT64, 8, "inforom_config_valid", DCGM_FS_DEVICE, 0, "ICVLD","", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_XID_ERRORS, DCGM_FT_INT64, 8, "xid_errors", DCGM_FS_DEVICE, 0, "XIDER","", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_MAX_LINK_GEN, DCGM_FT_INT64, 8, "pcie_max_link_gen", DCGM_FS_DEVICE, 0, "PCIMG","", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH, DCGM_FT_INT64, 8, "pcie_max_link_width", DCGM_FS_DEVICE, 0, "PCIMW","", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_LINK_GEN, DCGM_FT_INT64, 8, "pcie_link_gen", DCGM_FS_DEVICE, 0, "PCILG","", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_PCIE_LINK_WIDTH, DCGM_FT_INT64, 8, "pcie_link_width", DCGM_FS_DEVICE, 0, "PCILW","", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_POWER_VIOLATION, DCGM_FT_INT64, 8, "power_violation", DCGM_FS_DEVICE, LWML_FI_DEV_PERF_POLICY_POWER,"PVIOL","",getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_THERMAL_VIOLATION, DCGM_FT_INT64, 8, "thermal_violation", DCGM_FS_DEVICE, LWML_FI_DEV_PERF_POLICY_THERMAL, "TVIOL", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_GPU_TOPOLOGY_PCI, DCGM_FT_BINARY, 0, "system_topology_pci", DCGM_FS_GLOBAL, 0, "STVCI", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_GPU_TOPOLOGY_LWLINK, DCGM_FT_BINARY, 0, "system_topology_lwlink", DCGM_FS_GLOBAL, 0, "STLWL", "", getWidthForEnum(DCGM_FIELD_WIDTH_16));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_GPU_TOPOLOGY_AFFINITY, DCGM_FT_BINARY, 0, "system_affinity", DCGM_FS_GLOBAL, 0, "SYSAF", "", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_SYNC_BOOST_VIOLATION, DCGM_FT_INT64, 8, "sync_boost_violation", DCGM_FS_DEVICE, LWML_FI_DEV_PERF_POLICY_SYNC_BOOST,"SBVIO","", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_BOARD_LIMIT_VIOLATION, DCGM_FT_INT64, 8, "board_limit_violation", DCGM_FS_DEVICE, LWML_FI_DEV_PERF_POLICY_BOARD_LIMIT, "BLVIO", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LOW_UTIL_VIOLATION, DCGM_FT_INT64, 8, "low_util_violation", DCGM_FS_DEVICE, LWML_FI_DEV_PERF_POLICY_LOW_UTILIZATION, "LUVIO","", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_RELIABILITY_VIOLATION, DCGM_FT_INT64, 8, "reliability_violation", DCGM_FS_DEVICE, LWML_FI_DEV_PERF_POLICY_RELIABILITY, "RVIOL", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION, DCGM_FT_INT64, 8, "app_clock_violation", DCGM_FS_DEVICE, LWML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS, "TAPCV", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION, DCGM_FT_INT64, 8, "base_clock_violation", DCGM_FS_DEVICE, LWML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS, "TAPBC", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES, DCGM_FT_DOUBLE, 8, "mem_util_samples", DCGM_FS_DEVICE, 0, "MUSAM", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_UTIL_SAMPLES, DCGM_FT_DOUBLE, 8, "gpu_util_samples", DCGM_FS_DEVICE, 0, "GUSAM", "", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L0, DCGM_FT_INT64, 8, "lwlink_flit_crc_error_count_l0", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L0,"NFEL0", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L1, DCGM_FT_INT64, 8, "lwlink_flit_crc_error_count_l1", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L1, "NFEL1", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L2, DCGM_FT_INT64, 8, "lwlink_flit_crc_error_count_l2", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L2, "NFEL2", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L3, DCGM_FT_INT64, 8, "lwlink_flit_crc_error_count_l3", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L3, "NFEL3", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L4, DCGM_FT_INT64, 8, "lwlink_flit_crc_error_count_l4", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L4, "NFEL4", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L5, DCGM_FT_INT64, 8, "lwlink_flit_crc_error_count_l5", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L5, "NFEL5", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL, DCGM_FT_INT64, 8, "lwlink_flit_crc_error_count_total", DCGM_FS_DEVICE, LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL, "NFELT", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L0, DCGM_FT_INT64, 8, "lwlink_data_crc_error_count_l0", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L0, "NDEL0", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L1, DCGM_FT_INT64, 8, "lwlink_data_crc_error_count_l1", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L1, "NDEL1", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L2, DCGM_FT_INT64, 8, "lwlink_data_crc_error_count_l2", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L2, "NDEL2", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L3, DCGM_FT_INT64, 8, "lwlink_data_crc_error_count_l3", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L3, "NDEL3", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L4, DCGM_FT_INT64, 8, "lwlink_data_crc_error_count_l4", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L4, "NDEL4", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L5, DCGM_FT_INT64, 8, "lwlink_data_crc_error_count_l5", DCGM_FS_DEVICE,
                               LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L5, "NDEL5", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL, DCGM_FT_INT64, 8, "lwlink_data_crc_error_count_total", DCGM_FS_DEVICE,LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL, "NDELT", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0, DCGM_FT_INT64, 8, "lwlink_replay_error_count_l0", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0, "NREL0", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1, DCGM_FT_INT64, 8, "lwlink_replay_error_count_l1", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1, "NREL1", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2, DCGM_FT_INT64, 8, "lwlink_replay_error_count_l2", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2, "NREL2", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3, DCGM_FT_INT64, 8, "lwlink_replay_error_count_l3", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3, "NREL3", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4, DCGM_FT_INT64, 8, "lwlink_replay_error_count_l4", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4, "NREL4", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5, DCGM_FT_INT64, 8, "lwlink_replay_error_count_l5", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5, "NREL5", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL, DCGM_FT_INT64, 8, "lwlink_replay_error_count_total", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL, "NRELT", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0, DCGM_FT_INT64, 8, "lwlink_recovery_error_count_l0", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0, "NRCL0", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1, DCGM_FT_INT64, 8, "lwlink_recovery_error_count_l1", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1, "NRCL1", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2, DCGM_FT_INT64, 8, "lwlink_recovery_error_count_l2", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2, "NRCL2", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3, DCGM_FT_INT64, 8, "lwlink_recovery_error_count_l3", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3, "NRCL3", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4, DCGM_FT_INT64, 8, "lwlink_recovery_error_count_l4", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4, "NRCL4", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5, DCGM_FT_INT64, 8, "lwlink_recovery_error_count_l5", DCGM_FS_DEVICE, 
                               LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5, "NRCL5", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL, DCGM_FT_INT64, 8, "lwlink_recovery_error_count_total", DCGM_FS_DEVICE, LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL, "NRCLT", "", getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L0,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l0",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L0,
                                             "NBWL0",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L1,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l1",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L1,
                                             "NBWL1",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L2,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l2",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L2,
                                             "NBWL2",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L3,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l3",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L3,
                                             "NBWL3",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L4,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l4",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L4,
                                             "NBWL4",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L5,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l5",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L5,
                                             "NBWL5",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_total",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_TOTAL,
                                             "NBWLT",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_flit_crc_error_count_l6",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L6,
                                             "NFEL6",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_flit_crc_error_count_l7",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L7,
                                             "NFEL7",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_flit_crc_error_count_l8",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L8,
                                             "NFEL8",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_flit_crc_error_count_l9",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L9,
                                             "NFEL9",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_flit_crc_error_count_l10",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L10,
                                             "NFEL10",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_flit_crc_error_count_l11",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L11,
                                             "NFEL11",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_data_crc_error_count_l6",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L6,
                                             "NDEL6",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_data_crc_error_count_l7",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L7,
                                             "NDEL7",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_data_crc_error_count_l8",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L8,
                                             "NDEL8",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_data_crc_error_count_l9",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L9,
                                             "NDEL9",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_data_crc_error_count_l10",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L10,
                                             "NDEL10",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_data_crc_error_count_l11",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L11,
                                             "NDEL11",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_replay_error_count_l6",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L6,
                                             "NREL6",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_replay_error_count_l7",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L7,
                                             "NREL7",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_replay_error_count_l8",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L8,
                                             "NREL8",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_replay_error_count_l9",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L9,
                                             "NREL9",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_replay_error_count_l10",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L10,
                                             "NREL10",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_replay_error_count_l11",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L11,
                                             "NREL11",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_recovery_error_count_l6",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L6,
                                             "NRCL6",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_recovery_error_count_l7",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L7,
                                             "NRCL7",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_recovery_error_count_l8",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L8,
                                             "NRCL8",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_recovery_error_count_l9",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L9,
                                             "NRCL9",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_recovery_error_count_l10",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L10,
                                             "NRCL10",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_recovery_error_count_l11",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L11,
                                             "NRCL11",
                                             "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L6,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l6",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L6,
                                             "NBWL6",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L7,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l7",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L7,
                                             "NBWL7",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L8,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l8",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L8,
                                             "NBWL8",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L9,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l9",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L9,
                                             "NBWL9",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L10,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l10",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L10,
                                             "NBWL10",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWLINK_BANDWIDTH_L11,
                                             DCGM_FT_INT64,
                                             8,
                                             "lwlink_bandwidth_l11",
                                             DCGM_FS_DEVICE,
                                             LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L11,
                                             "NBWL11",
                                             getTextForEnum(DCGM_FIELD_UNIT_BW_MBPS),
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));

    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_MEMORY_TEMP, DCGM_FT_INT64, 8, "memory_temp", DCGM_FS_DEVICE,
                               LWML_FI_DEV_MEMORY_TEMP, "MMTMP","C", getWidthForEnum(DCGM_FIELD_WIDTH_10));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_GPU_LWLINK_ERRORS, DCGM_FT_INT64, 8, "gpu_lwlink_errors",
                               DCGM_FE_GPU, 0, "GLWERR","", getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P00, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p00",
                               DCGM_FS_ENTITY, 0, "SLL00", "", getWidthForEnum(DCGM_FIELD_WIDTH_20) );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P00, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p00", 
                               DCGM_FS_ENTITY, 0, "SLM00", "", getWidthForEnum(DCGM_FIELD_WIDTH_20) );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P00, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p00",
                               DCGM_FS_ENTITY, 0, "SHL00", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P00, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p00",
                               DCGM_FS_ENTITY, 0, "SLX00", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P01, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p01",
                               DCGM_FS_ENTITY, 0, "SLL01", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P01, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p01", 
                               DCGM_FS_ENTITY, 0, "SLM01", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P01, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p01",
                               DCGM_FS_ENTITY, 0, "SLH01", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P01, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p01",
                               DCGM_FS_ENTITY, 0, "SLX01", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P02, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p02",
                               DCGM_FS_ENTITY, 0, "SLL02", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P02, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p02",
                               DCGM_FS_ENTITY, 0, "SLM02", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P02, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p02",
                               DCGM_FS_ENTITY, 0, "SLH02", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P02, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p02",
                               DCGM_FS_ENTITY, 0, "SLX02", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P03, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p03", 
                               DCGM_FS_ENTITY, 0, "SLL03", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P03, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p03",
                               DCGM_FS_ENTITY, 0, "SLM03", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P03, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p03",
                               DCGM_FS_ENTITY, 0, "SLH03", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P03, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p03",
                               DCGM_FS_ENTITY, 0, "SLX03", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P04, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p04",
                               DCGM_FS_ENTITY, 0, "SLL04", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P04, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p04",
                               DCGM_FS_ENTITY, 0, "SLM04", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P04, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p04",
                               DCGM_FS_ENTITY, 0, "SLH04", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P04, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p04",
                               DCGM_FS_ENTITY, 0, "SLX04", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P05, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p05",
                               DCGM_FS_ENTITY, 0, "SLL05", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P05, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p05", 
                               DCGM_FS_ENTITY, 0, "SLM05", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P05, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p05",
                               DCGM_FS_ENTITY, 0, "SLH05", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P05, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p05",
                               DCGM_FS_ENTITY, 0, "SLX05", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P06, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p06",
                               DCGM_FS_ENTITY, 0, "SLL06", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P06, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p06",
                               DCGM_FS_ENTITY, 0, "SLM06", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P06, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p06",
                               DCGM_FS_ENTITY, 0, "SLH06", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P06, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p06",
                               DCGM_FS_ENTITY, 0, "SLX06", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P07, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p07",
                               DCGM_FS_ENTITY, 0, "SLL07", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P07, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p07", 
                               DCGM_FS_ENTITY, 0, "SLM07", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P07, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p07",
                               DCGM_FS_ENTITY, 0, "SLH07", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P07, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p07",
                               DCGM_FS_ENTITY, 0, "SLX07", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P08, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p08",
                               DCGM_FS_ENTITY, 0, "SLL08", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P08, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p08", 
                               DCGM_FS_ENTITY, 0, "SLM08", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P08, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p08",
                               DCGM_FS_ENTITY, 0, "SLH08", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P08, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p08",
                               DCGM_FS_ENTITY, 0, "SLX08", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P09, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p09",
                               DCGM_FS_ENTITY, 0, "SLL09", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P09, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p09", 
                               DCGM_FS_ENTITY, 0, "SLM09", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P09, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p09",
                               DCGM_FS_ENTITY, 0, "SLH09", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P09, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p09",
                               DCGM_FS_ENTITY, 0, "SLX09", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P10, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p10",
                               DCGM_FS_ENTITY, 0, "SLL10", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P10, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p10", 
                               DCGM_FS_ENTITY, 0, "SLM10", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P10, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p10",
                               DCGM_FS_ENTITY, 0, "SLH10", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P10, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p10",
                               DCGM_FS_ENTITY, 0, "SLX10", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P11, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p11",
                               DCGM_FS_ENTITY, 0, "SLL11", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P11, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p11", 
                               DCGM_FS_ENTITY, 0, "SLM11", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P11, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p11",
                               DCGM_FS_ENTITY, 0, "SLH11", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P11, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p11",
                               DCGM_FS_ENTITY, 0, "SLX11", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P12, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p12",
                               DCGM_FS_ENTITY, 0, "SLL12", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P12, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p12", 
                               DCGM_FS_ENTITY, 0, "SLM12", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P12, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p12",
                               DCGM_FS_ENTITY, 0, "SLH12", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P12, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p12",
                               DCGM_FS_ENTITY, 0, "SLX12", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P13, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p13",
                               DCGM_FS_ENTITY, 0, "SLL13", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P13, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p13", 
                               DCGM_FS_ENTITY, 0, "SLM13", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P13, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p13",
                               DCGM_FS_ENTITY, 0, "SLH13", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P13, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p13",
                               DCGM_FS_ENTITY, 0, "SLX13", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P14, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p14",
                               DCGM_FS_ENTITY, 0, "SLL14", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P14, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p14", 
                               DCGM_FS_ENTITY, 0, "SLM14", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P14, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p14",
                               DCGM_FS_ENTITY, 0, "SLH14", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P14, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p14",
                               DCGM_FS_ENTITY, 0, "SLX14", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P15, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p15",
                               DCGM_FS_ENTITY, 0, "SLL15", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P15, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p15",
                               DCGM_FS_ENTITY, 0, "SLM15", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P15, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p15",
                               DCGM_FS_ENTITY, 0, "SLH15", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P15, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p15",
                               DCGM_FS_ENTITY, 0, "SLX15", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P16, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p16",
                               DCGM_FS_ENTITY, 0, "SLL16", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P16, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p16", 
                               DCGM_FS_ENTITY, 0, "SLM16", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P16, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p16",
                               DCGM_FS_ENTITY, 0, "SLH16", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P16, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p16",
                               DCGM_FS_ENTITY, 0, "SLX16", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P17, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_low_p17",
                               DCGM_FS_ENTITY, 0, "SLL17", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P17, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_med_p17",
                               DCGM_FS_ENTITY, 0, "SLM17", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P17, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_high_p17",
                               DCGM_FS_ENTITY, 0, "SLH17", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P17, DCGM_FT_INT64, 8, "lwswitch_latency_histogram_max_p17",
                               DCGM_FS_ENTITY, 0, "SLX17", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P00, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p00", 
                               DCGM_FS_ENTITY, 0, "ST000", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P00, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p00",
                               DCGM_FS_ENTITY, 0, "SR000", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P01, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p01",
                               DCGM_FS_ENTITY, 0, "ST001", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P01, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p01",
                               DCGM_FS_ENTITY, 0, "SR001", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P02, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p02", 
                               DCGM_FS_ENTITY, 0, "ST002", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P02, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p02",
                               DCGM_FS_ENTITY, 0, "SR002", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P03, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p03",
                               DCGM_FS_ENTITY, 0, "ST003", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P03, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p03", 
                               DCGM_FS_ENTITY, 0, "SR003", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P04, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p04",
                               DCGM_FS_ENTITY, 0, "ST004", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P04, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p04",
                               DCGM_FS_ENTITY, 0, "SR004", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P05, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p05", 
                               DCGM_FS_ENTITY, 0, "ST005", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P05, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p05",
                               DCGM_FS_ENTITY, 0, "SR005", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P06, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p06",
                               DCGM_FS_ENTITY, 0, "ST006", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P06, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p06",
                               DCGM_FS_ENTITY, 0, "SR006", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P07, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p07",
                               DCGM_FS_ENTITY, 0, "ST007", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P07, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p07", 
                               DCGM_FS_ENTITY, 0, "SR007", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P08, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p08",
                               DCGM_FS_ENTITY, 0, "ST008", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P08, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p08",
                               DCGM_FS_ENTITY, 0, "SR008", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P09, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p09",
                               DCGM_FS_ENTITY, 0, "ST009", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P09, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p09",
                               DCGM_FS_ENTITY, 0, "SR009", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P10, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p10",
                               DCGM_FS_ENTITY, 0, "ST010", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P10, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p10",
                               DCGM_FS_ENTITY, 0, "SR010", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P11, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p11",
                               DCGM_FS_ENTITY, 0, "ST011", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P11, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p11",
                               DCGM_FS_ENTITY, 0, "SR011", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P12, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p12",
                               DCGM_FS_ENTITY, 0, "ST012", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P12, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p12",
                               DCGM_FS_ENTITY, 0, "SR012", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P13, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p13",
                               DCGM_FS_ENTITY, 0, "ST013", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P13, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p13",
                               DCGM_FS_ENTITY, 0, "SR013", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P14, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p14",
                               DCGM_FS_ENTITY, 0, "ST014", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P14, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p14",
                               DCGM_FS_ENTITY, 0, "SR014", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P15, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p15",
                               DCGM_FS_ENTITY, 0, "ST015", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P15, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p15",
                               DCGM_FS_ENTITY, 0, "SR015", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P16, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p16",
                               DCGM_FS_ENTITY, 0, "ST016", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P16, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p16",
                               DCGM_FS_ENTITY, 0, "SR016", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P17, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_0_p17",
                               DCGM_FS_ENTITY, 0, "ST017", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P17, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_0_p17",
                               DCGM_FS_ENTITY, 0, "SR017", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P00, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p00",
                               DCGM_FS_ENTITY, 0, "ST100", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P00, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p00",
                               DCGM_FS_ENTITY, 0, "SR100", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P01, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p01",
                               DCGM_FS_ENTITY, 0, "ST101", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P01, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p01",
                               DCGM_FS_ENTITY, 0, "SR101", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P02, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p02",
                               DCGM_FS_ENTITY, 0, "ST102", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P02, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p02",
                               DCGM_FS_ENTITY, 0, "SR102", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P03, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p03",
                               DCGM_FS_ENTITY, 0, "ST103", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P03, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p03",
                               DCGM_FS_ENTITY, 0, "SR103", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P04, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p04",
                               DCGM_FS_ENTITY, 0, "ST104", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P04, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p04",
                               DCGM_FS_ENTITY, 0, "SR104", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P05, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p05",
                               DCGM_FS_ENTITY, 0, "ST105", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P05, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p05",
                               DCGM_FS_ENTITY, 0, "SR105", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P06, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p06",
                               DCGM_FS_ENTITY, 0, "ST106", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P06, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p06",
                               DCGM_FS_ENTITY, 0, "SR106", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P07, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p07",
                               DCGM_FS_ENTITY, 0, "ST107", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P07, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p07",
                               DCGM_FS_ENTITY, 0, "SR107", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P08, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p08",
                               DCGM_FS_ENTITY, 0, "ST108", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P08, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p08",
                               DCGM_FS_ENTITY, 0, "SR108", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P09, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p09",
                               DCGM_FS_ENTITY, 0, "ST109", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P09, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p09",
                               DCGM_FS_ENTITY, 0, "SR109", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P10, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p10",
                               DCGM_FS_ENTITY, 0, "ST110", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P10, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p10",
                               DCGM_FS_ENTITY, 0, "SR110", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P11, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p11",
                               DCGM_FS_ENTITY, 0, "ST111", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P11, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p11",
                               DCGM_FS_ENTITY, 0, "SR111", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P12, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p12",
                               DCGM_FS_ENTITY, 0, "ST112", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P12, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p12",
                               DCGM_FS_ENTITY, 0, "SR112", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P13, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p13",
                               DCGM_FS_ENTITY, 0, "ST113", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P13, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p13",
                               DCGM_FS_ENTITY, 0, "SR113", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P14, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p14",
                               DCGM_FS_ENTITY, 0, "ST114", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P14, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p14",
                               DCGM_FS_ENTITY, 0, "SR114", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P15, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p15",
                               DCGM_FS_ENTITY, 0, "ST115", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P15, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p15",
                               DCGM_FS_ENTITY, 0, "SR115", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P16, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p16",
                               DCGM_FS_ENTITY, 0, "ST116", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P16, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p16",
                               DCGM_FS_ENTITY, 0, "SR116", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P17, DCGM_FT_INT64, 8, "lwswitch_bandwidth_tx_1_p17",
                               DCGM_FS_ENTITY, 0, "ST117", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P17, DCGM_FT_INT64, 8, "lwswitch_bandwidth_rx_1_p17",
                               DCGM_FS_ENTITY, 0, "SR117", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS, DCGM_FT_INT64, 8, "lwswitch_fatal_error", 
                               DCGM_FS_ENTITY, 0, "SEN00", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS, DCGM_FT_INT64, 8, "lwswitch_non_fatal_error", 
                               DCGM_FS_ENTITY, 0, "SEN01", "", getWidthForEnum(DCGM_FIELD_WIDTH_20)  );
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_DEV_LWDA_COMPUTE_CAPABILITY, DCGM_FT_INT64, 0,
                                             "lwda_compute_capability", DCGM_FS_DEVICE, 0, "DVCCC", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_GR_ENGINE_ACTIVE, DCGM_FT_DOUBLE, 0,
                                             "gr_engine_active", DCGM_FS_DEVICE, 0, "GRACT", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_SM_ACTIVE, DCGM_FT_DOUBLE, 0,
                                             "sm_active", DCGM_FS_DEVICE, 0, "SMACT", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_SM_OCLWPANCY, DCGM_FT_DOUBLE, 0,
                                             "sm_oclwpancy", DCGM_FS_DEVICE, 0, "SMOCC", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, DCGM_FT_DOUBLE, 0,
                                             "tensor_active", DCGM_FS_DEVICE, 0, "TENSO", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_DRAM_ACTIVE, DCGM_FT_DOUBLE, 0,
                                             "dram_active", DCGM_FS_DEVICE, 0, "DRAMA", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_FP64_ACTIVE, DCGM_FT_DOUBLE, 0,
                                             "fp64_active", DCGM_FS_DEVICE, 0, "FP64A", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_FP32_ACTIVE, DCGM_FT_DOUBLE, 0,
                                             "fp32_active", DCGM_FS_DEVICE, 0, "FP32A", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PIPE_FP16_ACTIVE, DCGM_FT_DOUBLE, 0,
                                             "fp16_active", DCGM_FS_DEVICE, 0, "FP16A", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_5));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PCIE_TX_BYTES, DCGM_FT_INT64, 0,
                                             "pcie_tx_bytes", DCGM_FS_DEVICE, 0, "PCITX", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_PCIE_RX_BYTES, DCGM_FT_INT64, 0,
                                             "pcie_rx_bytes", DCGM_FS_DEVICE, 0, "PCIRX", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_LWLINK_TX_BYTES, DCGM_FT_INT64, 0,
                                             "lwlink_tx_bytes", DCGM_FS_DEVICE, 0, "LWLTX", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    DcgmFieldsPopulateOneFieldWithFormatting(DCGM_FI_PROF_LWLINK_RX_BYTES, DCGM_FT_INT64, 0,
                                             "lwlink_rx_bytes", DCGM_FS_DEVICE, 0, "LWLRX", "",
                                             getWidthForEnum(DCGM_FIELD_WIDTH_20));
    return 0;
}

/*****************************************************************************/
/*
 * Hash function for measurementcollection keys (char *)
 *
 */
static unsigned int DcgmFieldsKeyHashCB(const void *key)
{
    char *val = (char *)key;
    int len = (int)strlen(val);
    unsigned int retVal;

    MurmurHash3_x86_32(val, len, 0, &retVal);
    return retVal;
}

/*****************************************************************************/
/*
 * Key compare function
 *
 * Hashtable wants a non-zero return for equal keys, so ilwert strcmp()'s return
 */
static int DcgmFieldsKeyCmpCB(const void *key1, const void *key2)
{
    int st = strcmp((const char *)key1, (const char *)key2); /* Case sensitive, just like the hash function */

    return (!st); /* Ilwerted on purpose */
}

/*****************************************************************************/
/*
 * Key free function
 *
 */
static void DcgmFieldsKeyFreeCB(void *key)
{
    /* Since this is a malloc'd string, free it */
    free(key);
}

/*****************************************************************************/
/*
 * Value free function
 *
 */
static void DcgmFieldsValueFreeCB(void *value)
{
    dcgm_field_meta_t * fieldMeta = (dcgm_field_meta_t *)value;
    if(fieldMeta->valueFormat)
    {
        free(fieldMeta->valueFormat);
    }
}

/*****************************************************************************/
static int DcgmFieldsPopulateHashTable(void)
{
    int st, i;
    dcgm_field_meta_p fieldMeta = 0;
    dcgm_field_meta_p found = 0;

    st = hashtable_init(&dcgmFieldsKeyToIdMap, DcgmFieldsKeyHashCB,
                        DcgmFieldsKeyCmpCB, DcgmFieldsKeyFreeCB,
                        DcgmFieldsValueFreeCB);
    if(st)
        return -1;

    for(i = 0; i < DCGM_FI_MAX_FIELDS; i++)
    {
        fieldMeta = dcgmFieldMeta[i];
        if(!fieldMeta)
            continue;

        /* Make sure the field tag doesn't exist already */
        found = (dcgm_field_meta_p)hashtable_get(&dcgmFieldsKeyToIdMap, fieldMeta->tag);
        if(found)
        {
            PRINT_ERROR("%s %u %u", "Found duplicate tag %s with id %u while inserting id %u",
                        fieldMeta->tag, found->fieldId, fieldMeta->fieldId);
            hashtable_close(&dcgmFieldsKeyToIdMap);
            return -1;
        }

        /* Doesn't exist. Insert ours, using tag as the key */
        st = hashtable_set(&dcgmFieldsKeyToIdMap, strdup(fieldMeta->tag), fieldMeta);
        if(st)
        {
            PRINT_ERROR("%d %s", "Error %d while inserting tag %s", st, fieldMeta->tag);
            hashtable_close(&dcgmFieldsKeyToIdMap);
            return -1;
        }
    }

    return 0;
}

/*****************************************************************************/
int DcgmFieldsInit(void)
{
    int i;
    int st;

    if(dcgmFieldsInitialized)
        return 0;

    st = DcgmFieldsPopulateFieldTableWithFormatting();
    if(st)
        return -1;

    /* Create the hash table of tags to IDs */
    st = DcgmFieldsPopulateHashTable();
    if(st)
        return -1;

    dcgmFieldsInitialized = 1;
    return 0;
}

/*****************************************************************************/
int DcgmFieldsTerm(void)
{
    int i;

    if(!dcgmFieldsInitialized)
        return 0; /* Nothing to do */

    hashtable_close(&dcgmFieldsKeyToIdMap);

    /* Zero the structure just to be sure */
    memset(&dcgmFieldsKeyToIdMap, 0, sizeof(dcgmFieldsKeyToIdMap));

    for(i = 0; i < DCGM_FI_MAX_FIELDS; i++)
    {
        if(!dcgmFieldMeta[i])
            continue;

        free(dcgmFieldMeta[i]);
        dcgmFieldMeta[i] = 0;
    }

    dcgmFieldsInitialized = 0;
    return 0;
}

/*****************************************************************************/
dcgm_field_meta_p DcgmFieldGetById(unsigned short fieldId)
{
    dcgm_field_meta_p retVal = 0;

    if(!dcgmFieldsInitialized)
        return 0;
    if(fieldId >= DCGM_FI_MAX_FIELDS)
        return 0;

    return dcgmFieldMeta[fieldId];
}

/*****************************************************************************/
dcgm_field_meta_p DcgmFieldGetByTag(char *tag)
{
    dcgm_field_meta_p retVal = 0;

    if(!dcgmFieldsInitialized || !tag)
        return 0;

    retVal = (dcgm_field_meta_p)hashtable_get(&dcgmFieldsKeyToIdMap, tag);
    return retVal;
}

/*****************************************************************************/
char *DcgmFieldsGetEntityGroupString(dcgm_field_entity_group_t entityGroupId)
{
    switch(entityGroupId)
    {
        default:
        case DCGM_FE_NONE:
            return "None";
        case DCGM_FE_GPU:
            return "GPU";
        case DCGM_FE_VGPU:
            return "vGPU";
        case DCGM_FE_SWITCH:
            return "Switch";
    }
}

/*****************************************************************************/
