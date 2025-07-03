
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <string.h>

#include "explorer16common.h"

// Below extracted from HW spreadsheet

HWLink16 HWLinks16[EXPLORER16_NUM_USED_PORTS] = {
                                {GB1_SW1_G1,   4,   GB1__GPU4,     1},
                                {GB1_SW1_G1,   5,   GB1__GPU1,     1},
                                {GB1_SW1_G1,   12,  GB1__GPU8,     1},
                                {GB1_SW1_G1,   13,  GB1__GPU5,     1},
                                {GB1_SW1_G1,   14,  GB1__GPU6,     1},
                                {GB1_SW1_G1,   15,  GB1__GPU7,     1},
                                {GB1_SW1_G1,   16,  GB1__GPU2,     1},
                                {GB1_SW1_G1,   17,  GB1__GPU3,     1},
                                {GB1_SW2_G2,   4,   GB1__GPU3,     5},
                                {GB1_SW2_G2,   5,   GB1__GPU8,     5},
                                {GB1_SW2_G2,   6,   GB1__GPU1,     5},
                                {GB1_SW2_G2,   7,   GB1__GPU4,     4},
                                {GB1_SW2_G2,   12,  GB1__GPU6,     4},
                                {GB1_SW2_G2,   13,  GB1__GPU7,     4},
                                {GB1_SW2_G2,   16,  GB1__GPU5,     5},
                                {GB1_SW2_G2,   17,  GB1__GPU2,     4},
                                {GB1_SW3_G3,   4,   GB1__GPU6,     3},
                                {GB1_SW3_G3,   5,   GB1__GPU7,     3},
                                {GB1_SW3_G3,   6,   GB1__GPU2,     2},
                                {GB1_SW3_G3,   7,   GB1__GPU4,     2},
                                {GB1_SW3_G3,   13,  GB1__GPU5,     3},
                                {GB1_SW3_G3,   14,  GB1__GPU8,     4},
                                {GB1_SW3_G3,   15,  GB1__GPU1,     3},
                                {GB1_SW3_G3,   17,  GB1__GPU3,     2},
                                {GB1_SW4_G4,   4,   GB1__GPU8,     0},
                                {GB1_SW4_G4,   5,   GB1__GPU6,     0},
                                {GB1_SW4_G4,   12,  GB1__GPU7,     0},
                                {GB1_SW4_G4,   13,  GB1__GPU5,     0},
                                {GB1_SW4_G4,   14,  GB1__GPU3,     0},
                                {GB1_SW4_G4,   15,  GB1__GPU2,     0},
                                {GB1_SW4_G4,   16,  GB1__GPU1,     0},
                                {GB1_SW4_G4,   17,  GB1__GPU4,     0},
                                {GB1_SW5_G5,   5,   GB1__GPU7,     2},
                                {GB1_SW5_G5,   6,   GB1__GPU3,     4},
                                {GB1_SW5_G5,   7,   GB1__GPU2,     5},
                                {GB1_SW5_G5,   12,  GB1__GPU8,     3},
                                {GB1_SW5_G5,   13,  GB1__GPU5,     2},
                                {GB1_SW5_G5,   14,  GB1__GPU6,     5},
                                {GB1_SW5_G5,   15,  GB1__GPU1,     4},
                                {GB1_SW5_G5,   17,  GB1__GPU4,     5},
                                {GB1_SW6_G6,   5,   GB1__GPU6,     2},
                                {GB1_SW6_G6,   6,   GB1__GPU1,     2},
                                {GB1_SW6_G6,   7,   GB1__GPU4,     3},
                                {GB1_SW6_G6,   12,  GB1__GPU2,     3},
                                {GB1_SW6_G6,   13,  GB1__GPU3,     3},
                                {GB1_SW6_G6,   14,  GB1__GPU5,     4},
                                {GB1_SW6_G6,   15,  GB1__GPU8,     2},
                                {GB1_SW6_G6,   17,  GB1__GPU7,     5},
                                {GB2_SW1_G1,   4,   GB2__GPU4,     1},
                                {GB2_SW1_G1,   5,   GB2__GPU1,     1},
                                {GB2_SW1_G1,   12,  GB2__GPU8,     1},
                                {GB2_SW1_G1,   13,  GB2__GPU5,     1},
                                {GB2_SW1_G1,   14,  GB2__GPU6,     1},
                                {GB2_SW1_G1,   15,  GB2__GPU7,     1},
                                {GB2_SW1_G1,   16,  GB2__GPU2,     1},
                                {GB2_SW1_G1,   17,  GB2__GPU3,     1},
                                {GB2_SW2_G2,   4,   GB2__GPU3,     5},
                                {GB2_SW2_G2,   5,   GB2__GPU8,     5},
                                {GB2_SW2_G2,   6,   GB2__GPU1,     5},
                                {GB2_SW2_G2,   7,   GB2__GPU4,     4},
                                {GB2_SW2_G2,   12,  GB2__GPU6,     4},
                                {GB2_SW2_G2,   13,  GB2__GPU7,     4},
                                {GB2_SW2_G2,   16,  GB2__GPU5,     5},
                                {GB2_SW2_G2,   17,  GB2__GPU2,     4},
                                {GB2_SW3_G3,   4,   GB2__GPU6,     3},
                                {GB2_SW3_G3,   5,   GB2__GPU7,     3},
                                {GB2_SW3_G3,   6,   GB2__GPU2,     2},
                                {GB2_SW3_G3,   7,   GB2__GPU4,     2},
                                {GB2_SW3_G3,   13,  GB2__GPU5,     3},
                                {GB2_SW3_G3,   14,  GB2__GPU8,     4},
                                {GB2_SW3_G3,   15,  GB2__GPU1,     3},
                                {GB2_SW3_G3,   17,  GB2__GPU3,     2},
                                {GB2_SW4_G4,   4,   GB2__GPU8,     0},
                                {GB2_SW4_G4,   5,   GB2__GPU6,     0},
                                {GB2_SW4_G4,   12,  GB2__GPU7,     0},
                                {GB2_SW4_G4,   13,  GB2__GPU5,     0},
                                {GB2_SW4_G4,   14,  GB2__GPU3,     0},
                                {GB2_SW4_G4,   15,  GB2__GPU2,     0},
                                {GB2_SW4_G4,   16,  GB2__GPU1,     0},
                                {GB2_SW4_G4,   17,  GB2__GPU4,     0},
                                {GB2_SW5_G5,   5,   GB2__GPU7,     2},
                                {GB2_SW5_G5,   6,   GB2__GPU3,     4},
                                {GB2_SW5_G5,   7,   GB2__GPU2,     5},
                                {GB2_SW5_G5,   12,  GB2__GPU8,     3},
                                {GB2_SW5_G5,   13,  GB2__GPU5,     2},
                                {GB2_SW5_G5,   14,  GB2__GPU6,     5},
                                {GB2_SW5_G5,   15,  GB2__GPU1,     4},
                                {GB2_SW5_G5,   17,  GB2__GPU4,     5},
                                {GB2_SW6_G6,   5,   GB2__GPU6,     2},
                                {GB2_SW6_G6,   6,   GB2__GPU1,     2},
                                {GB2_SW6_G6,   7,   GB2__GPU4,     3},
                                {GB2_SW6_G6,   12,  GB2__GPU2,     3},
                                {GB2_SW6_G6,   13,  GB2__GPU3,     3},
                                {GB2_SW6_G6,   14,  GB2__GPU5,     4},
                                {GB2_SW6_G6,   15,  GB2__GPU8,     2},
                                {GB2_SW6_G6,   17,  GB2__GPU7,     5},
                                     };

// on Explorer 16, all switches use the same trunk ports, connected to their peers in the same order
uint32_t   trunkPortsNear[EXPLORER16_NUM_TRUNK_PORTS] = {
                                        0,
                                        1,
                                        2,
                                        3,
                                        8,
                                        9,
                                        10,
                                        11
                                    };

uint32_t   trunkPortsFar[EXPLORER16_NUM_TRUNK_PORTS] =  {
                                        3,
                                        2,
                                        1,
                                        0,
                                        11,
                                        10,
                                        9,
                                        8
                                    };                                            

#define MAX_SHARED_LWSWITCH_FABRIC_PARTITIONS 31

SharedPartInfoTable_t gSharedVMPartInfo[MAX_SHARED_LWSWITCH_FABRIC_PARTITIONS] = {

                             //partionId 0 - 16 GPUs, 12 Willows, 48 intra-trunk conn, 0 inter-trunk conn
                             { 0, 16, 12, 48, 0,
                               // all the GPUs, id, numlink, mask
                               {   {0,  6,  0x3F},
                               {1,  6,  0x3F},
                               {2,  6,  0x3F},
                               {3,  6,  0x3F},
                               {4,  6,  0x3F},
                               {5,  6,  0x3F},
                               {6,  6,  0x3F},
                               {7,  6,  0x3F},
                               {8,  6,  0x3F},
                               {9,  6,  0x3F},
                               {10, 6,  0x3F},
                               {11, 6,  0x3F},
                               {12, 6,  0x3F},
                               {13, 6,  0x3F},
                               {14, 6,  0x3F},
                               {15, 6,  0x3F} },

                               // all the Switches, id, numlink, mask
                               {   {0x08, 18, 0x3FFFF},
                                   {0x09, 18, 0x3FFFF},
                                   {0x0A, 18, 0x3FFFF},
                                   {0x0B, 18, 0x3FFFF},
                                   {0x0C, 18, 0x3FFFF},
                                   {0x0D, 18, 0x3FFFF},
                                   {0x18, 18, 0x3FFFF},
                                   {0x19, 18, 0x3FFFF},
                                   {0x1A, 18, 0x3FFFF},
                                   {0x1B, 18, 0x3FFFF},
                                   {0x1C, 18, 0x3FFFF},
                                   {0x1D, 18, 0x3FFFF}, },
                             },

                             //partionId 1 - 8 GPUs, 6 Willows base board 1 , 0 intra-trunk conn, 0 inter-trunk conn
                             { 1, 8, 6, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {0, 6, 0x3F},
                                   {1, 6, 0x3F},
                                   {2, 6, 0x3F},
                                   {3, 6, 0x3F},
                                   {4, 6, 0x3F},
                                   {5, 6, 0x3F},
                                   {6, 6, 0x3F},
                                   {7, 6, 0x3F}, },

                               // all the Switches, id, numlink, mask
                               {   {0x08, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x09, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x0A, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x0B, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x0C, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x0D, 10, 0x3F0F0}, }, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                             },

                             //partionId 2 - 8 GPUs, 6 Willows base board 2, 0 intra-trunk conn, 0 inter-trunk conn
                             { 2, 8, 6, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {8,  6, 0x3F},
                                   {9,  6, 0x3F},
                                   {10, 6, 0x3F},
                                   {11, 6, 0x3F},
                                   {12, 6, 0x3F},
                                   {13, 6, 0x3F},
                                   {14, 6, 0x3F},
                                   {15, 6, 0x3F}, },

                               // all the Switches, id, numlink, mask
                               {   {0x18, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x19, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x1A, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x1B, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x1C, 10, 0x3F0F0}, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                                   {0x1D, 10, 0x3F0F0}, }, // trunk links 0,1,2,3,8,9,10,11 are disabled, rest is enabled
                             },

                             //partionId 3 - 4 GPUs, 5 Willows base board 1, 0 intra-trunk conn, 0 inter-trunk conn
                             { 3, 4, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {0, 5, 0x37}, // link3 disabled
                                   {3, 5, 0x3B}, // link2 disabled
                                   {5, 5, 0x37}, // link3 disabled
                                   {6, 5, 0x37}, }, // link3 disabled

                               // all the Switches, id, numlink, mask (Switch 3 not used)
                               {   {0x08, 4, 0x0C030}, // link4,5,14,15 enabled.
                                   {0x09, 4, 0x030C0}, // link6,7,12,13 enabled
                                   {0x0B, 4, 0x31020}, // link5,12,16,17 enabled
                                   {0x0C, 4, 0x2C020}, // link5,14,15,17 enabled
                                   {0x0D, 4, 0x200E0}, }, // link5,6,7,17 enabled
                              },

                             //partionId 4 - 4 GPUs, 5 Willows base board 1, 0 intra-trunk conn, 0 inter-trunk conn
                             { 4, 4, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {1, 5, 0x3E}, // link0 disabled
                                   {2, 5, 0x3E}, // link0 disabled
                                   {4, 5, 0x3E}, // link0 disabled
                                   {7, 5, 0x3E}, }, // link0 disabled

                               // all the Switches, id, numlink, mask (Switch 4 not used)
                               {   {0x08, 4, 0x33000}, // link12,13,16,17 enabled.
                                   {0x09, 4, 0x30030}, // link4,5,16,17 enabled
                                   {0x0A, 4, 0x26040}, // link6,13,14,17 enabled
                                   {0x0C, 4, 0x030C0}, // link6,7,12,13 enabled
                                   {0x0D, 4, 0x0F000}, }, // link12,13,14,15 enabled
                             },

                             //partionId 5 - 4 GPUs, 5 Willows base board 2, 0 intra-trunk conn, 0 inter-trunk conn
                             { 5, 4, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {8,  5, 0x37}, // link3 disabled
                                   {11, 5, 0x3B}, // link2 disabled
                                   {13, 5, 0x37}, // link3 disabled
                                   {14, 5, 0x37}, }, // link3 disabled

                               // all the Switches, id, numlink, mask (Switch 3 not used)
                               {   {0x18, 4, 0x0C030}, // link4,5,14,15 enabled.
                                   {0x19, 4, 0x030C0}, // link6,7,12,13 enabled
                                   {0x1B, 4, 0x31020}, // link5,12,16,17 enabled
                                   {0x1C, 4, 0x2C020}, // link5,14,15,17 enabled
                                   {0x1D, 4, 0x200E0}, }, // link5,6,7,17 enabled
                             },
 
                             //partionId 6 - 4 GPUs, 5 Willows base board 2, 0 intra-trunk conn, 0 inter-trunk conn
                             { 6, 4, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {9,  5, 0x3E}, // link0 disabled
                                   {10, 5, 0x3E}, // link0 disabled
                                   {12, 5, 0x3E}, // link0 disabled
                                   {15, 5, 0x3E}, }, // link0 disabled

                               // all the Switches, id, numlink, mask (Switch 4 not used)
                               {   {0x18, 4, 0x33000}, // link12,13,16,17 enabled.
                                   {0x19, 4, 0x30030}, // link4,5,16,17 enabled
                                   {0x1A, 4, 0x26040}, // link6,13,14,17 enabled
                                   {0x1C, 4, 0x030C0}, // link6,7,12,13 enabled
                                   {0x1D, 4, 0x0F000}, }, // link12,13,14,15 enabled
                             },

                             //partionId 7 - 2 GPUs, 5 Willows base board 1, 0 intra-trunk conn, 0 inter-trunk conn
                             { 7, 2, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {0, 5, 0x37}, // link3 disabled
                                   {3, 5, 0x3B}, }, // link2 disabled

                               // all the Switches, id, numlink, mask (Switch 3 not used)
                               {   {0x08, 2, 0x00030}, // link4,5 enabled.
                                   {0x09, 2, 0x000C0}, // link6,7 enabled.
                                   {0x0B, 2, 0x30000}, // link16,17 enabled.
                                   {0x0C, 2, 0x28000}, // link15,17 enabled.
                                   {0x0D, 2, 0x000C0}, }, // link6,7 enabled.
                             },

                             //partionId 8 - 2 GPUs, 5 Willows base board 1, 0 intra-trunk conn, 0 inter-trunk conn
                             { 8, 2, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {1, 5, 0x2F}, // link4 disabled
                                   {2, 5, 0x1F}, }, // link5 disabled

                               // all the Switches, id, numlink, mask (Switch 2 not used)
                               {   {0x08, 2, 0x30000}, // link16,17 enabled.
                                   {0x0A, 2, 0x20040}, // link6,17 enabled.
                                   {0x0B, 2, 0x0C000}, // link14,15 enabled.
                                   {0x0C, 2, 0x000C0}, // link6,7 enabled.
                                   {0x0D, 2, 0x03000}, }, // link12,13 enabled.
                             },

                             //partionId 9 - 2 GPUs, 5 Willows base board 1, 0 intra-trunk conn, 0 inter-trunk conn
                             { 9, 2, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {4, 5, 0x3E}, // link0 disabled
                                   {7, 5, 0x3E}, }, // link0 disabled

                               // all the Switches, id, numlink, mask (Switch 4 not used)
                               {   {0x08, 2, 0x03000}, // link12,13 enabled.
                                   {0x09, 2, 0x10020}, // link5,16 enabled.
                                   {0x0A, 2, 0x06000}, // link13,14 enabled.
                                   {0x0C, 2, 0x03000}, // link12,13 enabled.
                                   {0x0D, 2, 0x0C000}, }, // link14,15 enabled.
                             },

                             //partionId 10 - 2 GPUs, 5 Willows base board 1, 0 intra-trunk conn, 0 inter-trunk conn
                             { 10, 2, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {5, 5, 0x1F}, // link5 disabled
                                   {6, 5, 0x3B}, }, // link2 disabled

                               // all the Switches, id, numlink, mask (Switch 5 not used)
                               {   {0x08, 2, 0x0C000}, // link14,15 enabled.
                                   {0x09, 2, 0x03000}, // link12,13 enabled.
                                   {0x0A, 2, 0x00030}, // link4,5 enabled.
                                   {0x0B, 2, 0x01020}, // link5,12 enabled.
                                   {0x0D, 2, 0x20020}, }, // link5,17 enabled.
                             },

                             //partionId 11 - 2 GPUs, 5 Willows base board 2, 0 intra-trunk conn, 0 inter-trunk conn
                             { 11, 2, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {8,  5, 0x37}, // link3 disabled
                                   {11, 5, 0x3B}, }, // link2 disabled

                               // all the Switches, id, numlink, mask (Switch 3 not used)
                               {   {0x18, 2, 0x00030}, // link4,5 enabled.
                                   {0x19, 2, 0x000C0}, // link6,7 enabled.
                                   {0x1B, 2, 0x30000}, // link16,17 enabled.
                                   {0x1C, 2, 0x28000}, // link15,17 enabled.
                                   {0x1D, 2, 0x000C0}, }, // link6,7 enabled.
                             },

                             //partionId 12 - 2 GPUs, 5 Willows base board 2, 0 intra-trunk conn, 0 inter-trunk conn
                             { 12, 2, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {9,  5, 0x2F}, // link4 disabled
                                   {10, 5, 0x1F}, }, // link5 disabled

                               // all the Switches, id, numlink, mask (Switch 2 not used)
                               {   {0x18, 2, 0x30000}, // link16,17 enabled.
                                   {0x1A, 2, 0x20040}, // link6,17 enabled.
                                   {0x1B, 2, 0x0C000}, // link14,15 enabled.
                                   {0x1C, 2, 0x000C0}, // link6,7 enabled.
                                   {0x1D, 2, 0x03000}, }, // link12,13 enabled.
                             },
 
                             //partionId 13 - 2 GPUs, 5 Willows base board 2 , 0 intra-trunk conn, 0 inter-trunk conn
                             { 13, 2, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {12, 5, 0x3E}, // link0 disabled
                                   {15, 5, 0x3E}, }, // link0 disabled

                               // all the Switches, id, numlink, mask (Switch 4 not used)
                               {   {0x18, 2, 0x03000}, // link12,13 enabled.
                                   {0x19, 2, 0x10020}, // link5,16 enabled.
                                   {0x1A, 2, 0x06000}, // link13,14 enabled.
                                   {0x1C, 2, 0x03000}, // link12,13 enabled.
                                   {0x1D, 2, 0x0C000}, }, // link14,15 enabled.
                             },

                             //partionId 14 - 2 GPUs, 5 Willows base board 2, 0 intra-trunk conn, 0 inter-trunk conn
                             { 14, 2, 5, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   {13, 5, 0x1F}, // link5 disabled
                                   {14, 5, 0x3B}, }, // link2 disabled

                               // all the Switches, id, numlink, mask (Switch 5 not used)
                               {   {0x18, 2, 0x0C000}, // link14,15 enabled.
                                   {0x19, 2, 0x03000}, // link12,13 enabled.
                                   {0x1A, 2, 0x00030}, // link4,5 enabled.
                                   {0x1B, 2, 0x01020}, // link5,12 enabled.
                                   {0x1D, 2, 0x20020}, }, // link5,17 enabled.
                             },

                             //partionId 15 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 15, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 0, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 16 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 16, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 1, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 17 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 17, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 2, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 18 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 18, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 3, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 19 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 19, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 4, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 20 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 20, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 5, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 21 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 21, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 6, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 22 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 22, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 7, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 23 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 23, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 8, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 24 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 24, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 9, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 25 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 25, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 10, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 26 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 26, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 11, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 27 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 27, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 12, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 28 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 28, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 13, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },

                             //partionId 29 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 29, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 14, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },


                             //partionId 30 - 1 GPUs, 0 Willows, 0 intra-trunk conn, 0 inter-trunk conn
                             { 30, 1, 0, 0, 0,
                               // all the GPUs, id, numlink, mask
                               {   { 15, 0, 0}  },  // all links are disabled


                               // all the Switches, id, numlink, mask
                               {                  }, // No Switches are used
                             },
                           };



//gets Nth willow access port for given switch from the HWLinks array
//These ports are in conselwtive indexes
uint32_t exp16GetNthAccessPort(int switch_id, unsigned int n)
{
    return HWLinks16[switch_id * EXPLORER16_NUM_ACCESS_PORTS + n].willowPort;
}

//gets Nth willow trunk port for given switch from the trunkPortsNear/trunkPortsFar arrays
uint32_t exp16GetNthTrunkPort(int switch_id, unsigned int n)
{
    if (switch_id < EXPLORER16_NUM_SWITCH_PER_BASEBOARD)
        return trunkPortsNear[n];
    else
        return trunkPortsFar[n];
}

int exp16GetConnectedGPUID(int switch_id, uint32_t willow_port)
{
    HWLink16 *entry=NULL;
    int i;
    entry = &HWLinks16[switch_id * EXPLORER16_NUM_ACCESS_PORTS];
    for (i = 0; i < EXPLORER16_NUM_ACCESS_PORTS; i++)
    {
        if (entry[i].willowPort == willow_port)
            return entry[i].GPUIndex;
    }
    return EXPLORER16_NUM_GPU ;
}

int exp16ComputeReqLinkID(int switch_id, uint32_t willow_port)
{
    HWLink16 *entry=NULL;
    int i;
    entry = &HWLinks16[switch_id * EXPLORER16_NUM_ACCESS_PORTS];
    for (i = 0; i < EXPLORER16_NUM_ACCESS_PORTS; i++)
    {
        if (entry[i].willowPort == willow_port)
            return entry[i].GPUIndex * EXPLORER16_NUM_SWITCH_PER_BASEBOARD + entry[i].GPUPort;
    }
    return EXPLORER16_NUM_GPU ;
}

uint32_t exp16GetConnectedTrunkPortId(int switch_id, uint32_t willow_port)
{
    uint32_t *local_trunk_port, *connected_trunk_ports;
    if(switch_id < EXPLORER16_NUM_SWITCH_PER_BASEBOARD)
    {
        local_trunk_port = trunkPortsNear;
        connected_trunk_ports = trunkPortsFar;
    }
    else
    {
        local_trunk_port = trunkPortsFar;
        connected_trunk_ports = trunkPortsNear;
    }

    for(int i = 0; i < EXPLORER16_NUM_TRUNK_PORTS; i++)
    {
        if(local_trunk_port[i] == willow_port)
            return connected_trunk_ports[i];
    }
    return EXPLORER16_NUM_SWITCH_PORTS;
}

int exp16SwitchPhyIDtoSwitchID(int phy_id)
{
    if(phy_id >= 0x08 && phy_id <= 0x0d)
        return phy_id - 0x08;
    else if (phy_id >= 0x18 && phy_id <= 0x1d)
        return (phy_id - 0x18) + EXPLORER16_NUM_SWITCH_PER_BASEBOARD;
    else
        return -1;
}

bool exp16IsTrunkPort(unsigned int willow_port)
{
    int i;
    for( i = 0; i < EXPLORER16_NUM_TRUNK_PORTS; i++)
    {
        if (willow_port == trunkPortsNear[i])
        {
            return true;
        }
    }
    return false;
}

