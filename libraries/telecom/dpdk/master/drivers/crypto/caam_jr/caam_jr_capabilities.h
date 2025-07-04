/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2017-2018 NXP
 */

#ifndef CAAM_JR_CAPABILITIES_H
#define CAAM_JR_CAPABILITIES_H

#include <rte_cryptodev.h>
#include <rte_selwrity.h>

/* Get cryptodev capabilities */
const struct rte_cryptodev_capabilities *
caam_jr_get_cryptodev_capabilities(void);
/* Get security capabilities */
const struct rte_selwrity_capability *
caam_jr_get_selwrity_capabilities(void *device);

#endif
