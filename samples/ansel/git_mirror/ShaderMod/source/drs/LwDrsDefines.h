#pragma once

// This file has a copy of all the ANSEL_* defines found in the generated //sw/dev/gpu_drv/bugfix_main/drivers/common/inc/g_d3doglreg.h file.
// Any changes/additions to ANSEL_* regs should be propogated to this file alongside the driver change.
// The only difference between the definitions in this file and the one in drivers should be that all that ANSEL_*_STRING defines have been
// commented out, since those get colwerted to actual string defines in the LwApiDriverSettings.h which can be found in packman's lwapi package.

//#define ANSEL_ALLOW_STRING                                             "10682898"
#define ANSEL_ALLOW_ID                                                 0x1035db89
#define ANSEL_ALLOW_OVERINSTALL                                        0 // OVERRIDE
#define ANSEL_ALLOW_DISALLOWED                                         0
#define ANSEL_ALLOW_OFF                                                0
#define ANSEL_ALLOW_DISABLED                                           0
#define ANSEL_ALLOW_ALLOWED                                            1
#define ANSEL_ALLOW_ON                                                 1
#define ANSEL_ALLOW_ENABLED                                            1
#define ANSEL_ALLOW_DEFAULT                                            ANSEL_ALLOW_ALLOWED


//#define ANSEL_ALLOWLISTED_STRING                                       "14792591"
#define ANSEL_ALLOWLISTED_ID                                           0x1085da8a
#define ANSEL_ALLOWLISTED_OVERINSTALL                                  0 // OVERRIDE
#define ANSEL_ALLOWLISTED_DISALLOWED                                   0
#define ANSEL_ALLOWLISTED_OFF                                          0
#define ANSEL_ALLOWLISTED_DISABLED                                     0
#define ANSEL_ALLOWLISTED_ALLOWED                                      1
#define ANSEL_ALLOWLISTED_ON                                           1
#define ANSEL_ALLOWLISTED_ENABLED                                      1
#define ANSEL_ALLOWLISTED_DEFAULT                                      ANSEL_ALLOWLISTED_DISALLOWED


//#define ANSEL_ALLOW_FREESTYLE_MODE_STRING                              "33999624"
#define ANSEL_ALLOW_FREESTYLE_MODE_ID                                  0x101baaaf
#define ANSEL_ALLOW_FREESTYLE_MODE_OVERINSTALL                         0 // OVERRIDE
#define ANSEL_ALLOW_FREESTYLE_MODE_DISABLED                            0x0000
#define ANSEL_ALLOW_FREESTYLE_MODE_ENABLED                             0x0001
#define ANSEL_ALLOW_FREESTYLE_MODE_DEFAULT                             ANSEL_ALLOW_FREESTYLE_MODE_DISABLED


//#define ANSEL_ALLOW_OFFLINE_STRING                                     "93980749"
#define ANSEL_ALLOW_OFFLINE_ID                                         0x10d5c2db
#define ANSEL_ALLOW_OFFLINE_OVERINSTALL                                0 // OVERRIDE
#define ANSEL_ALLOW_OFFLINE_DISALLOWED                                 0
#define ANSEL_ALLOW_OFFLINE_OFF                                        0
#define ANSEL_ALLOW_OFFLINE_DISABLED                                   0
#define ANSEL_ALLOW_OFFLINE_ALLOWED                                    1
#define ANSEL_ALLOW_OFFLINE_ON                                         1
#define ANSEL_ALLOW_OFFLINE_ENABLED                                    1
#define ANSEL_ALLOW_OFFLINE_DEFAULT                                    ANSEL_ALLOW_OFFLINE_DISALLOWED


//#define ANSEL_BUFFERS_DEPTH_SETTINGS_STRING                            "16068746"
#define ANSEL_BUFFERS_DEPTH_SETTINGS_ID                                0x101314ce
#define ANSEL_BUFFERS_DEPTH_SETTINGS_OVERINSTALL                       0 // OVERRIDE
#define ANSEL_BUFFERS_DEPTH_SETTINGS_NONE                              0x00000000
#define ANSEL_BUFFERS_DEPTH_SETTINGS_USE_STATS                         0x00000001
#define ANSEL_BUFFERS_DEPTH_SETTINGS_USE_VIEWPORT                      0x00000002
#define ANSEL_BUFFERS_DEPTH_SETTINGS_VIEWPORT_SCALING                  0x00000004
#define ANSEL_BUFFERS_DEPTH_SETTINGS_ALL                               0xFFFFFFFF
#define ANSEL_BUFFERS_DEPTH_SETTINGS_DEFAULT                           ANSEL_BUFFERS_DEPTH_SETTINGS_NONE


//#define ANSEL_BUFFERS_DEPTH_WEIGHTS_STRING                             "16068747"
#define ANSEL_BUFFERS_DEPTH_WEIGHTS_ID                                 0x10079dbc
#define ANSEL_BUFFERS_DEPTH_WEIGHTS_OVERINSTALL                        0 // OVERRIDE
#define ANSEL_BUFFERS_DEPTH_WEIGHTS_DEFAULT                            L""


//#define ANSEL_BUFFERS_DISABLED_STRING                                  "16068745"
#define ANSEL_BUFFERS_DISABLED_ID                                      0x10e74421
#define ANSEL_BUFFERS_DISABLED_OVERINSTALL                             0 // OVERRIDE
#define ANSEL_BUFFERS_DISABLED_NONE                                    0x00000000
#define ANSEL_BUFFERS_DISABLED_DEPTH                                   0x00000001
#define ANSEL_BUFFERS_DISABLED_HDR                                     0x00000002
#define ANSEL_BUFFERS_DISABLED_HUDLESS                                 0x00000004
#define ANSEL_BUFFERS_DISABLED_FINAL_COLOR                             0x00000008
#define ANSEL_BUFFERS_DISABLED_ALL                                     0xFFFFFFFF
#define ANSEL_BUFFERS_DISABLED_DEFAULT                                 ANSEL_BUFFERS_DISABLED_NONE


//#define ANSEL_BUFFERS_HUDLESS_DRAWCALL_STRING                          "16068750"
#define ANSEL_BUFFERS_HUDLESS_DRAWCALL_ID                              0x101fd0c1
#define ANSEL_BUFFERS_HUDLESS_DRAWCALL_OVERINSTALL                     0 // OVERRIDE
#define ANSEL_BUFFERS_HUDLESS_DRAWCALL_DEFAULT                         2


//#define ANSEL_BUFFERS_HUDLESS_SETTINGS_STRING                          "16068748"
#define ANSEL_BUFFERS_HUDLESS_SETTINGS_ID                              0x10ad7f3b
#define ANSEL_BUFFERS_HUDLESS_SETTINGS_OVERINSTALL                     0 // OVERRIDE
#define ANSEL_BUFFERS_HUDLESS_SETTINGS_NONE                            0x00000000
#define ANSEL_BUFFERS_HUDLESS_SETTINGS_USE_STATS                       0x00000001
#define ANSEL_BUFFERS_HUDLESS_SETTINGS_ONLY_SINGLE_RTV_BINDS           0x00000002
#define ANSEL_BUFFERS_HUDLESS_SETTINGS_RESTRICT_FORMATS                0x00000004
#define ANSEL_BUFFERS_HUDLESS_SETTINGS_ALL                             0xFFFFFFFF
#define ANSEL_BUFFERS_HUDLESS_SETTINGS_DEFAULT                         ANSEL_BUFFERS_HUDLESS_SETTINGS_NONE


//#define ANSEL_BUFFERS_HUDLESS_WEIGHTS_STRING                           "16068749"
#define ANSEL_BUFFERS_HUDLESS_WEIGHTS_ID                               0x10c41bb5
#define ANSEL_BUFFERS_HUDLESS_WEIGHTS_OVERINSTALL                      0 // OVERRIDE
#define ANSEL_BUFFERS_HUDLESS_WEIGHTS_DEFAULT                          L""


//#define ANSEL_DENYLIST_ALL_PROFILED_STRING                             "43102335"
#define ANSEL_DENYLIST_ALL_PROFILED_ID                                 0x10f272b9
#define ANSEL_DENYLIST_ALL_PROFILED_OVERINSTALL                        0 // OVERRIDE
#define ANSEL_DENYLIST_ALL_PROFILED_DEFAULT                            L""


//#define ANSEL_DENYLIST_PER_GAME_STRING                                 "23776708"
#define ANSEL_DENYLIST_PER_GAME_ID                                     0x100d51f7
#define ANSEL_DENYLIST_PER_GAME_OVERINSTALL                            0 // OVERRIDE
#define ANSEL_DENYLIST_PER_GAME_DEFAULT                                L""


//#define ANSEL_ENABLE_STRING                                            "97373802"
#define ANSEL_ENABLE_ID                                                0x1075d972
#define ANSEL_ENABLE_OVERINSTALL                                       1 // MERGE
#define ANSEL_ENABLE_OFF                                               0
#define ANSEL_ENABLE_DISABLED                                          0
#define ANSEL_ENABLE_ON                                                1
#define ANSEL_ENABLE_ENABLED                                           1
#define ANSEL_ENABLE_DEFAULT                                           ANSEL_ENABLE_ON


//#define ANSEL_ENABLE_OPTIMUS_STRING                                    "97373801"
#define ANSEL_ENABLE_OPTIMUS_ID                                        0x1075d973
#define ANSEL_ENABLE_OPTIMUS_OVERINSTALL                               1 // MERGE
#define ANSEL_ENABLE_OPTIMUS_OFF                                       0
#define ANSEL_ENABLE_OPTIMUS_DISABLED                                  0
#define ANSEL_ENABLE_OPTIMUS_ON                                        1
#define ANSEL_ENABLE_OPTIMUS_ENABLED                                   1
#define ANSEL_ENABLE_OPTIMUS_DEFAULT                                   ANSEL_ENABLE_OPTIMUS_OFF


//#define ANSEL_FREESTYLE_MODE_STRING                                    "27152819"
#define ANSEL_FREESTYLE_MODE_ID                                        0x105e2a1d
#define ANSEL_FREESTYLE_MODE_OVERINSTALL                               0 // OVERRIDE
#define ANSEL_FREESTYLE_MODE_DISABLED                                  0x0000
#define ANSEL_FREESTYLE_MODE_ENABLED                                   0x0001
#define ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLED                      0x0002
#define ANSEL_FREESTYLE_MODE_APPROVED_ONLY                             0x0004
#define ANSEL_FREESTYLE_MODE_MULTIPLAYER_APPROVED_ONLY                 0x0008
#define ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLE_EXTRA_BUFFERS         0x0010
#define ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLE_DEPTH                 0x0020
#define ANSEL_FREESTYLE_MODE_DEFAULT                                   ANSEL_FREESTYLE_MODE_DISABLED

