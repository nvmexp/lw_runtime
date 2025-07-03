
DCGM_INT_ENTRY_POINT(dcgmLwSwitchStart, tsapiLwSwitchStart,
                     (dcgmHandle_t pDcgmHandle, dcgm_lwswitch_msg_start_t *startMsg),
                     "(%p %p)",
                     pDcgmHandle, startMsg);

DCGM_INT_ENTRY_POINT(dcgmLwSwitchShutdown, tsapiLwSwitchShutdown,
                     (dcgmHandle_t pDcgmHandle, dcgm_lwswitch_msg_shutdown_t *shutdownMsg),
                     "(%p %p)",
                     pDcgmHandle, shutdownMsg);


