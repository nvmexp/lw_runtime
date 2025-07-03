#ifndef DCGMFIELDS_INTERNAL_H
#define DCGMFIELDS_INTERNAL_H

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************/
/* Definition of all the internal DCGM fields                                   */
/********************************************************************************/


/********************************************************************************/
/* Field ID ranges                                                              */
/********************************************************************************/

/* Profiling field IDs */
#define DCGM_FI_PROF_FIRST_ID DCGM_FI_PROF_GR_ENGINE_ACTIVE
#define DCGM_FI_PROF_LAST_ID  DCGM_FI_PROF_LWLINK_RX_BYTES

#ifdef __cplusplus
}
#endif

#endif //DCGMFIELDS_INTERNAL_H

