#ifndef DCGM_GPU_HARDWARE_LIMITS_H
#define DCGM_GPU_HARDWARE_LIMITS_H

// 8 replays per minute is the maximum recommended by RM
#define DCGM_LIMIT_MAX_PCIREPLAY_RATE 8


// defined via bug 1665722
#define DCGM_LIMIT_MAX_RETIRED_PAGES 60
#define DCGM_LIMIT_MAX_RETIRED_PAGES_SOFT_LIMIT 15

// dummy value until a real one is determined. JIRA: DCGM-445
#define DCGM_LIMIT_MAX_SBE_RATE 10


#define DCGM_LIMIT_MAX_LWLINK_ERROR 1
// RM has informed us that CRC errors only matter at rates of 100+ per second
#define DCGM_LIMIT_MAX_LWLINK_CRC_ERROR 100.0

#endif // DCGM_GPU_HARDWARE_LIMITS_H
