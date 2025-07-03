#ifndef MEMORYBANDWIDTHSHARED_H
#define MEMORYBANDWIDTHSHARED_H

/*
 * Definitions that are shared between the bandwidth_calc.lw file and the rest of the plugin. These are defined
 * separately in this file to minimize dependencies of the lwca compile 
 */

#define REAL int

/* Identifiers for each test */
#define SI_TEST_TRIAD 0

#define SI_TEST_COUNT 1 /* 1 greater than largest number above */

#define MAXTIMES  10000

#define BLK0 64
#define BLK1 128
#define BLK2 256
#define BLK3 512

#define STR0 1
#define STR1 2
#define STR2 4
#define STR3 8
#define STR4 16
#define STR5 32
#define STR6 64
#define STR7 128

#endif //MEMORYBANDWIDTHSHARED_H