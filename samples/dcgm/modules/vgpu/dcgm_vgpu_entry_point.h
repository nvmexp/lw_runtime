DCGM_INT_ENTRY_POINT(dcgmVgpuStart, tsapiVgpuStart,
                     (dcgmHandle_t pDcgmHandle, dcgm_vgpu_msg_start_t *startMsg),
                     "(%p %p)",
                     pDcgmHandle, startMsg);

DCGM_INT_ENTRY_POINT(dcgmVgpuShutdown, tsapiVgpuShutdown,
                     (dcgmHandle_t pDcgmHandle, dcgm_vgpu_msg_shutdown_t *shutdownMsg),
                     "(%p %p)",
                     pDcgmHandle, shutdownMsg);
