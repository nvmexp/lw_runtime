/* 
 * File:   LwcmSettings.h
 */

#ifndef DCGMSETTINGS_H
#define	DCGMSETTINGS_H

#include "dcgm_structs_internal.h"
#include "dcgm_structs.h"
#include "lwos.h"
#include "lwml.h"
#include "logging.h"

#define DCGM_MIN_DRIVER_VERSION "384.00"

#ifdef DEBUG
#define DEBUG_STDERR(x) do { std::cerr << x << std::endl; } while (0)
#define DEBUG_STDOUT(x) do { std::cout << x << std::endl; } while (0)
#else 
#define DEBUG_STDERR(x)
#define DEBUG_STDOUT(x)
#endif

/* Logging-related elwironmental variables */
#define DCGM_ELW_DBG_LVL     "__DCGM_DBG_LVL"
#define DCGM_ELW_DBG_APPEND  "__DCGM_DBG_APPEND"
#define DCGM_ELW_DBG_FILE    "__DCGM_DBG_FILE"
#define DCGM_ELW_DBG_FILE_ROTATE  "__DCGM_DBG_FILE_ROTATE"

/* Elwironmental variable to bypass the white list */
#define DCGM_ELW_WL_BYPASS   "__DCGM_WL_BYPASS"

#define DCGM_MODE_EMBEDDED_HE    0  /* Mode when Host Engine is Embedded. ISV Agent Use Case */
#define DCGM_MODE_STANDALONE_HE  1  /* Mode when Host Engine is Standalone. LW Agent Use Case */

/*****************************************************************************/
/* Null/failure IDs to use as initializers and error returns */
#define DCGM_LWML_ID_BAD (-1)
#define DCGM_GPU_ID_BAD  ((unsigned int)-1)
#define DCGM_ENTITY_ID_BAD DCGM_GPU_ID_BAD

#endif	/* DCGMSETTINGS_H */
