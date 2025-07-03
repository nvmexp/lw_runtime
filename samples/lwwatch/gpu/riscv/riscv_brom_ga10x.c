/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <string.h>
#include <lwtypes.h>
#include <lwmisc.h>
#include <ampere/ga102/lw_pmu_riscv_address_map.h>
#include <ampere/ga102/dev_riscv_pri.h>
#include <ampere/ga102/dev_falcon_v4.h>
#include <ampere/ga102/dev_se_pri.h>

#include "riscv_prv.h"
#include "riscv_porting.h"

#define MK_D(X, V) {V, #X}

typedef struct {
    LwU32 val;
    const char *desc;
} reg_decodes;

#define MK_RES(X) {LW_PRISCV_RISCV_BR_RETCODE_RESULT_##X, #X}
static reg_decodes br_result[] = {
    MK_RES(INIT),
    MK_RES(RUNNING),
    MK_RES(FAIL),
    MK_RES(PASS),
    {0, NULL}
};

#define MK_PH(X) {LW_PRISCV_RISCV_BR_RETCODE_PHASE_##X, #X}
static reg_decodes br_phase[] = {
    MK_PH(ENTRY                    ),
    MK_PH(INIT_DEVICE              ),
    MK_PH(LOAD_PUBLIC_KEY          ),
    MK_PH(LOAD_PKC_BOOT_PARAM      ),
    MK_PH(SHA_MANIFEST             ),
    MK_PH(VERIFY_MANIFEST_SIGNATURE),
    MK_PH(DECRYPT_MANIFEST         ),
    MK_PH(SANITIZE_MANIFEST        ),
    MK_PH(LOAD_FMC                 ),
    MK_PH(VERIFY_FMC_DIGEST        ),
    MK_PH(DECRYPT_FMC              ),
    MK_PH(DECRYPT_FUSEKEY          ),
    MK_PH(REVOKE_RESOURCE          ),
    MK_PH(CONFIGURE_FMC_ELW        ),
    {0, NULL}
};

#define MK_SY(X) {LW_PRISCV_RISCV_BR_RETCODE_SYNDROME_##X, #X}
static reg_decodes br_syndromes[] = {
    MK_SY(INIT                               ),
    MK_SY(DMA_FB_ADDRESS_ERROR               ),
    MK_SY(DMA_NACK_ERROR                     ),
    MK_SY(SHA_ACQUIRE_MUTEX_ERROR            ),
    MK_SY(SHA_EXELWTION_ERROR                ),
    MK_SY(DIO_READ_ERROR                     ),
    MK_SY(DIO_WRITE_ERROR                    ),
    MK_SY(SE_PDI_ILWALID_ERROR               ),
    MK_SY(SE_PKEYHASH_ILWALID_ERROR          ),
    MK_SY(SW_PKEY_DIGEST_ERROR               ),
    MK_SY(SE_PKA_RETURN_CODE_ERROR           ),
    MK_SY(SCP_LOAD_SECRET_ERROR              ),
    MK_SY(SCP_TRAPPED_DMA_NOT_ALIGNED_ERROR  ),
    MK_SY(MANIFEST_CODE_SIZE_ERROR           ),
    MK_SY(MANIFEST_CORE_PMP_RESERVATION_ERROR),
    MK_SY(MANIFEST_DATA_SIZE_ERROR           ),
    MK_SY(MANIFEST_DEVICEMAP_BR_UNLOCK_ERROR ),
    MK_SY(MANIFEST_FAMILY_ID_ERROR           ),
    MK_SY(MANIFEST_MSPM_VALUE_ERROR          ),
    MK_SY(MANIFEST_PAD_INFO_MASK_ERROR       ),
    MK_SY(MANIFEST_REG_PAIR_ADDRESS_ERROR    ),
    MK_SY(MANIFEST_REG_PAIR_ENTRY_NUM_ERROR  ),
    MK_SY(MANIFEST_SECRET_MASK_ERROR         ),
    MK_SY(MANIFEST_SECRET_MASK_LOCK_ERROR    ),
    MK_SY(MANIFEST_SIGNATURE_ERROR           ),
    MK_SY(MANIFEST_UCODE_ID_ERROR            ),
    MK_SY(MANIFEST_UCODE_VERSION_ERROR       ),
    MK_SY(FMC_DIGEST_ERROR                   ),
    MK_SY(FUSEKEY_BAD_HEADER_ERROR           ),
    MK_SY(FUSEKEY_KEYGLOB_ILWALID_ERROR      ),
    MK_SY(FUSEKEY_PROTECT_INFO_ERROR         ),
    MK_SY(FUSEKEY_SIGNATURE_ERROR            ),
    MK_SY(KMEM_DISPOSE_KSLOT_ERROR           ),
    MK_SY(KMEM_KEY_SLOT_K3_ERROR             ),
    MK_SY(KMEM_LOAD_KSLOT2SCP_ERROR          ),
    MK_SY(KMEM_READ_ERROR                    ),
    MK_SY(KMEM_WRITE_ERROR                   ),
    MK_SY(IOPMP_ERROR                        ),
    MK_SY(MMIO_ERROR                         ),
    MK_SY(OK                                 ),
    {0, NULL}
};

#define MK_IN(X) {LW_PRISCV_RISCV_BR_RETCODE_INFO_##X, #X}

static reg_decodes br_infos[] = {
    MK_IN(INIT),
    MK_IN(DMA_WAIT_FOR_IDLE_HANG),
    MK_IN(SHA_HANG),
    MK_IN(DIO_READ_HANG),
    MK_IN(DIO_WAIT_FREE_ENTRY_HANG),
    MK_IN(DIO_WRITE_HANG),
    MK_IN(SE_ACQUIRE_MUTEX_HANG),
    MK_IN(SE_PDI_LOAD_HANG),
    MK_IN(SE_PKEYHASH_LOAD_HANG),
    MK_IN(PKA_POLL_RESULT_HANG),
    MK_IN(SCP_PIPELINE_RESET_HANG),
    MK_IN(TRAPPED_DMA_HANG),
    MK_IN(FUSEKEY_KEYGLOB_LOAD_HANG),
    MK_IN(KMEM_CMD_EXELWTE_HANG),
    {0, NULL}
};

static const char * lookupReg(LwU32 reg, reg_decodes *arr)
{
    while (arr->desc != NULL)
    {
        if (arr->val == reg)
            return arr->desc;
        arr++;
    }
    return "(UNKNOWN)";
}

static LwU64 rd64(LwU32 addr)
{
    return (LwU64)bar0Read(addr) | (((LwU64)bar0Read(addr + 4)) << 32);
}

LW_STATUS riscvBrStatus_GA10X(void)
{
    LwU32 reg;
    const char *DMACFG_TARGETS[] = {"localFB", "cohSysmem", "nonCohSysmem", "invalid"};

    reg = bar0Read(LW_PRISCV_RISCV_BCR_CTRL);
    dprintf("Bootrom Configuration: 0x%08x ", reg);
    dprintf("(%s) ", DRF_VAL(_PRISCV_RISCV, _BCR_CTRL, _VALID, reg) ? "VALID" : "INVALID");
    dprintf("core: %s ", FLD_TEST_DRF(_PRISCV_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV, reg) ? "RISC-V" : "FALCON");
    dprintf("brfetch: %s ", FLD_TEST_DRF(_PRISCV_RISCV, _BCR_CTRL, _BRFETCH, _TRUE, reg) ? "ENABLED" : "DISABLED");
    dprintf("\n");
    reg = bar0Read(LW_PRISCV_RISCV_BCR_DMACFG);
    dprintf("Bootrom DMA configuration: 0x%08x ", reg);
    dprintf("target: 0x%x(%s) ",
            DRF_VAL(_PRISCV_RISCV, _BCR_DMACFG, _TARGET, reg),
            DMACFG_TARGETS[DRF_VAL(_PRISCV_RISCV, _BCR_DMACFG, _TARGET, reg)]);
    dprintf("%s\n", FLD_TEST_DRF(_PRISCV_RISCV, _BCR_DMACFG, _LOCK, _LOCKED, reg) ? "LOCKED" : "UNLOCKED");
    dprintf("RISCV priv lockdown is %s\n",
            FLD_TEST_DRF(_PFALCON, _FALCON_HWCFG2, _RISCV_BR_PRIV_LOCKDOWN, _LOCK,
                         bar0ReadLegacy(LW_PFALCON_FALCON_HWCFG2)) ? "ENABLED" : "DISABLED");
    reg = bar0Read(LW_PRISCV_RISCV_BCR_DMACFG_SEC);
    dprintf("Bootrom DMA SEC configuration: 0x%08x ", reg);
    dprintf("wprid: 0x%x \n",
            DRF_VAL(_PRISCV_RISCV, _BCR_DMACFG_SEC, _WPRID, reg));
    reg = bar0Read(LW_PRISCV_RISCV_BR_RETCODE);
    dprintf("RETCODE: 0x%08x Result: %s Phase: %s Syndrome: %s Info:%s\n",
            reg,
            lookupReg(DRF_VAL(_PRISCV_RISCV, _BR_RETCODE, _RESULT, reg), br_result),
            lookupReg(DRF_VAL(_PRISCV_RISCV, _BR_RETCODE, _PHASE, reg), br_phase),
            lookupReg(DRF_VAL(_PRISCV_RISCV, _BR_RETCODE, _SYNDROME, reg), br_syndromes),
            lookupReg(DRF_VAL(_PRISCV_RISCV, _BR_RETCODE, _INFO, reg), br_infos)
            );
    dprintf("FMC Code addr: "LwU64_FMT"\n", rd64(LW_PRISCV_RISCV_BCR_DMAADDR_FMCCODE_LO));
    dprintf("FMC Data addr: "LwU64_FMT"\n", rd64(LW_PRISCV_RISCV_BCR_DMAADDR_FMCDATA_LO));
    dprintf("PKC addr     : "LwU64_FMT"\n", rd64(LW_PRISCV_RISCV_BCR_DMAADDR_PKCPARAM_LO));
    dprintf("PUBKEY addr  : "LwU64_FMT"\n", rd64(LW_PRISCV_RISCV_BCR_DMAADDR_PUBKEY_LO));

    return LW_OK;
}

LW_STATUS riscvBrBoot_GA10X(LwBool bWait)
{
    LwU32 bcr = 0;

    // Configure CPU, MK TODO: no interface in lwwatch to do it?
//    engineWrite(pEngine, pEngine->pInfo->fbif_ctl,
//                 (DRF_DEF(_PFALCON_FBIF, _CTL, _ENABLE, _TRUE) |
//                  DRF_DEF(_PFALCON_FBIF, _CTL, _ALLOW_PHYS_NO_CTX, _ALLOW)));
//    // Setup NACK as ACK that should help if we try invalid read
//    engineWrite(pEngine, pEngine->pInfo->fbif_ctl2,
//                 LW_PFALCON_FBIF_CTL2_NACK_MODE_NACK_AS_ACK);

    bcr = DRF_DEF(_PRISCV_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV)   |
          DRF_DEF(_PRISCV_RISCV, _BCR_CTRL, _VALID,       _TRUE)    |
          DRF_NUM(_PRISCV_RISCV, _BCR_CTRL, _BRFETCH,     0);

    bar0Write(LW_PRISCV_RISCV_BOOT_VECTOR_HI, 0);
    bar0Write(LW_PRISCV_RISCV_BOOT_VECTOR_LO, LW_RISCV_AMAP_IROM_START);

    // We actually don't need that.
    bar0Write(LW_PRISCV_RISCV_BCR_DMAADDR_FMCCODE_HI, 0);
    bar0Write(LW_PRISCV_RISCV_BCR_DMAADDR_FMCCODE_LO, LW_RISCV_AMAP_IMEM_START);

    bar0Write(LW_PRISCV_RISCV_BCR_CTRL, bcr);
    dprintf("Starting RISC-V in BROM mode...\n");
    bar0Write(LW_PRISCV_RISCV_CPUCTL, DRF_DEF(_PRISCV_RISCV, _CPUCTL, _STARTCPU, _TRUE));

    if (bWait)
    {
        LwU32 timeoutMs = 10000;
        LwU32 reg;

        dprintf("Waiting for BROM to finish...\n");
        do {
            reg = bar0Read(LW_PRISCV_RISCV_BR_RETCODE);
            if (FLD_TEST_DRF(_PRISCV, _RISCV_BR_RETCODE, _RESULT, _PASS, reg))
            {
                dprintf("BROM finished with result: PASS\n");
                break;
            } else if (FLD_TEST_DRF(_PRISCV, _RISCV_BR_RETCODE, _RESULT, _FAIL, reg))
            {
                dprintf("BROM finished with result: FAIL\n");
                break;
            }

            if (timeoutMs > 0)
            {
                if (timeoutMs >= 10)
                {
                    riscvDelay(10);
                    timeoutMs -= 10;
                } else
                {
                    riscvDelay(timeoutMs);
                    timeoutMs = 0;
                }
            } else
            {
                dprintf("Timeout waiting for BROM.\n");
                return LW_ERR_TIMEOUT;
            }
        } while (1);
    }
    return LW_OK;
}

// Not in falcon manuals :/
#define ENGINE_RESET_OFFSET         0x3c0
#define ENGINE_RESET_PLM_OFFSET     0x3c4

LW_STATUS riscvBrReset_GA10X(LwBool bWait)
{
    LwU32 bcr = 0;

    // Check if you can reset engine
    if ((bar0ReadLegacy(ENGINE_RESET_PLM_OFFSET) & 0xff) != 0xff)
    {
        dprintf("Can't reset %s - PLM prohibits ENGINE_RESET.\n", pRiscvInstance->name);
        return LW_FALSE;
    }

    dprintf("Issuing reset to %s... ", pRiscvInstance->name);
    bar0WriteLegacy(ENGINE_RESET_OFFSET, 1);
    riscvDelay(10);
    bar0WriteLegacy(ENGINE_RESET_OFFSET, 0);
    dprintf("Done.\n");
    if ((GPU_REG_RD32(LW_PSE_SE_CTRL_RESET__PRIV_LEVEL_MASK) & 0xff) == 0xff)
    {
        dprintf("Issuing reset to SE... ");
        GPU_REG_WR32(LW_PSE_SE_CTRL_RESET,
                    DRF_DEF(_PSE_SE, _CTRL_RESET, _SE_SOFT_RESET, _TRIGGER));
        if (bWait)
        {
            LwU32 timeoutMs = 10000;
            while(!FLD_TEST_DRF(_PSE_SE, _CTRL_RESET, _SE_SOFT_RESET, _DONE,
                                GPU_REG_RD32(LW_PSE_SE_CTRL_RESET)))
            {
                dprintf(".");
                if (timeoutMs > 0)
                {
                    if (timeoutMs >= 10)
                    {
                        riscvDelay(10);
                        timeoutMs -= 10;
                    } else
                    {
                        riscvDelay(timeoutMs);
                        timeoutMs = 0;
                    }
                } else
                {
                    dprintf("Timeout waiting for BROM.\n");
                    return LW_ERR_TIMEOUT;
                }
            }
            dprintf(" Done.\n");
        } else {
            dprintf("Not waiting for reset to complete. SE may be unusable.\n");
        }
    }
    else
    {
        dprintf("Can't reset %s - PLM prohibits SE_RESET. Boot may fail.\n", pRiscvInstance->name);
    }

    bcr = DRF_DEF(_PRISCV_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV)   |
          DRF_DEF(_PRISCV_RISCV, _BCR_CTRL, _VALID,       _TRUE)    |
          DRF_NUM(_PRISCV_RISCV, _BCR_CTRL, _BRFETCH,     0);
    dprintf("Configuring bcr...\n");
    bar0Write(LW_PRISCV_RISCV_BCR_CTRL, bcr);

    return LW_OK;
}
