#ifndef DCGM_LWSWITCH_INTERNAL_H
#define DCGM_LWSWITCH_INTERNAL_H

#ifdef  __cplusplus
extern "C" {
#endif

#include "dcgm_structs_internal.h"
#include "dcgm_uuid.h"
#include "dcgm_lwswitch_structs.h"

/*****************************************************************************
 *****************************************************************************/
/*****************************************************************************
 * DCGM Client Internal functions to be used by DCGMI or other LW tools
 *****************************************************************************/
/*****************************************************************************
 *****************************************************************************/

// GUIDS for internal APIs
DCGM_DEFINE_UUID(ETID_DCGMLwSwitchInternal,
                 0x59c938fd, 0x674e, 0x42cf, 0xaf, 0xd5, 0x83, 0x3e, 0x61, 0xb8, 0xaa, 0x3b);

typedef struct etblDCGMLwSwitchInternal_st {
    /// This export table supports versioning by adding to the end without changing
    /// the ETID.  The struct_size field will always be set to the size in bytes of
    /// the entire export table structure.
    size_t struct_size;

    // 1
    /**
     * This method is used to start the LwSwitch module within the host engine
     *
     * @param pLwcmHandle. Handle to the host engine
     * @param startMsg Parameter struct for this message
     * @return
     */
    dcgmReturn_t(*fpLwswitchStart)(dcgmHandle_t pLwcmHandle, dcgm_lwswitch_msg_start_t *startMsg);

    // 2
    /**
     * This method is used shut down the Lwswitch module within the host engine
     *
     * @param pLwcmHandle Handle to the host engine
     * @param shutdownMsg Parameter struct for this message
     * @return
     */
    dcgmReturn_t(*fpLwswitchShutdown)(dcgmHandle_t pLwcmHandle, dcgm_lwswitch_msg_shutdown_t *shutdownMsg);
} etblDCGMLwSwitchInternal;


#ifdef  __cplusplus
}
#endif

#endif //DCGM_LWSWITCH_INTERNAL_H
