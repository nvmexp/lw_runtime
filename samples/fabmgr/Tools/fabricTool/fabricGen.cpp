#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <g_lwconfig.h>

#include <google/protobuf/text_format.h>
#include "FMCommonTypes.h"
#include "fabricmanager.pb.h"
#include "basicE3600Config1.h"
#include "basicE3600Config2.h"
#include "basicE3600Config3.h"
#include "basicE3600Config4.h"
#include "basicE3600Config5.h"
#include "basicE3600Config6.h"
#include "basicE3600Config7.h"
#include "basicE3600Config8.h"
#include "basicE3600Config9.h"
#include "vcFlip.h"
#include "vanguard.h"
#include "explorer8Config.h"
#include "explorer16Config.h"
#include "explorer16TrunkSprayConfig.h"
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
#include "explorer16KTConfig.h"
#endif
#include "explorer2Config.h"
#include "explorer8LBConfig.h"
#include "explorerConfig.h"
#include "explorerSkylakeConfig.h"
#include "explorerLoop.h"
#include "hgx2LBConfig.h"
#include "lrEmulationConfig.h"
#include "lsFsfConfig.h"
#include "lsEmulationConfig.h"
#include "basicE4700Config1.h"
#include "basicE4700Config2.h"
#include "basicE4700Config3.h"
#include "deltaConfig.h"
#include "vulcanSurrogate.h"
#include "vulcan.h"
#include "basicE4840Config1.h"
#include "basicE4840Config2.h"

#include "emulationConfig.h"
#include "fm_log.h"

#define LINE_BUF_SIZE 512
/* Logging-related elwironmental variables */
#define FABRIC_TOOL_ELW_DBG_LVL         4
#define FABRIC_TOOL_ELW_DBG_APPEND      0
#define FABRIC_TOOL_ELW_DBG_FILE        "/var/log/fabricmanager.log"
#define FABRIC_TOOL_MAX_LOG_FILE_SIZE   2
#define FABRIC_TOOL_USE_SYS_LOG         0

static void decimalToHex ( const char *inFileName, const char *outFileName )
{
    FILE *inFile;
    FILE *outFile;
    char lineStr[LINE_BUF_SIZE];
    int64_t value;
    char hexStr[LINE_BUF_SIZE];
    char *ptr, *endptr;

    if ( ( inFileName == NULL ) || ( outFileName == NULL ) ) {
        printf("Invalid file names.\n");
        return;
    }

    inFile = fopen( inFileName, "r" );
    if ( inFile == NULL )
    {
        printf("Failed to open input file %s\n", inFileName);
        return;
    }

    outFile = fopen( outFileName, "w+" );
    if ( outFile == NULL )
    {
        printf("Failed to open output file %s\n", outFileName);
        fclose( inFile );
        return;
    }

    while ( fgets(lineStr, sizeof(lineStr), inFile) )
    {
        ptr = strstr( lineStr, ": " );
        if ( ptr == NULL )
        {
            fputs( lineStr, outFile );
            continue;
        }

        ptr += 2;
        if ( !isdigit(*ptr) )
        {
            fputs( lineStr, outFile );
            continue;
        }

        value = strtoll(ptr, &endptr, 10);
        sprintf(ptr, "0x%lx\n", value);
        fputs( lineStr, outFile );
    }

    fclose( inFile );
    fclose( outFile );
}

void genFabricConfig( fabricTopologyEnum topology,
                      const char *topoBinFileName,
                      const char *topoTextFileName,
                      const char *topoTextHexFileName)
{
    unsigned int fileLength;
    unsigned int bytesWritten;
    FILE *topoFile;
    FILE *topoTextFile;
    char *bufToWrite;
    int i, j, k;
    std::string configText;
    basicE3600Config1 *pConfig1;
    basicE3600Config2 *pConfig2;
    basicE3600Config3 *pConfig3;
    basicE3600Config4 *pConfig4;
    basicE3600Config5 *pConfig5;
    basicE3600Config6 *pConfig6;
    basicE3600Config7 *pConfig7;
    basicE3600Config8 *pConfig8;
    basicE3600Config9 *pConfig9;
    vcFlip            *pVcFlip;
    vanguardConfig    *pVanguardConfig;
    explorer8Config   *pExplorer8Config;
    explorer16Config  *pExplorer16Config;
    explorer16TrunkSprayConfig  *pExplorer16TrunkSprayConfig;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    explorer16KTConfig  *pExplorer16KTConfig;
#endif
    explorer2Config   *pExplorer2Config;
    explorer8LBConfig *pExplorer8LBConfig;
    explorerConfig    *pExplorer8BW;
    explorerSkylakeConfig *pExplorer8SL;
    explorerLoop      *pExplorerLoop;
    emulationConfig   *pConfigEmu;
    hgx2LBConfig      *pHgx2LBConfig;
    lrEmulationConfig *pLrEmulationConfig;
    lsFsfConfig *pLsFsfConfig;
    lsEmulationConfig *pLsEmulationConfig;
    basicE4700Config1 *pE4700Config1;
    basicE4700Config2 *pE4700Config2;
    basicE4700Config3 *pE4700Config3;
    deltaConfig       *pDeltaConfig;
    vulcanSurrogate   *pVulcanSurrogate;
    vulcan            *pVulcan;
    basicE4840Config1 *pE4840Config1;
    basicE4840Config2 *pE4840Config2;
    fabric *mFabric;
    nodeSystemPartitionInfo *pSystemPartInfo = NULL;

    if ( (topoBinFileName == NULL) ||
         (topoTextFileName == NULL) ||
         (topoTextHexFileName == NULL) ) {
        FM_LOG_ERROR("invalid input files.\n");
        return;
    }

    switch ( topology )
    {
    case BASIC_E3600_CONFIG1:

        pConfig1 = new basicE3600Config1( topology);
        pConfig1->makeNodes();
        mFabric = pConfig1->mFabric;
        break;

    case BASIC_E3600_CONFIG2:

        pConfig2 = new basicE3600Config2( topology);
        pConfig2->makeNodes();
        mFabric = pConfig2->mFabric;
        break;

    case BASIC_E3600_CONFIG3:

        pConfig3 = new basicE3600Config3( topology);
        pConfig3->makeNodes();
        mFabric = pConfig3->mFabric;
        break;

    case BASIC_E3600_CONFIG4:

        pConfig4 = new basicE3600Config4( topology);
        pConfig4->makeNodes();
        mFabric = pConfig4->mFabric;
        break;

    case BASIC_E3600_CONFIG5:

        pConfig5 = new basicE3600Config5( topology);
        pConfig5->makeNodes();
        mFabric = pConfig5->mFabric;
        break;

    case BASIC_E3600_CONFIG6:

        pConfig6 = new basicE3600Config6( topology);
        pConfig6->makeNodes();
        mFabric = pConfig6->mFabric;
        break;

    case BASIC_E3600_CONFIG7:

        pConfig7 = new basicE3600Config7( topology);
        pConfig7->makeNodes();
        mFabric = pConfig7->mFabric;
        break;

    case BASIC_E3600_CONFIG8:

        pConfig8 = new basicE3600Config8( topology);
        pConfig8->makeNodes();
        mFabric = pConfig8->mFabric;
        break;

    case BASIC_E3600_CONFIG9:

        pConfig9 = new basicE3600Config9( topology);
        pConfig9->makeNodes();
        mFabric = pConfig9->mFabric;
        break;

    case VC_FLIP:

        pVcFlip = new vcFlip( topology);
        pVcFlip->makeNodes();
        mFabric = pVcFlip->mFabric;
        break;

    case VANGUARD_CONFIG:

        pVanguardConfig = new vanguardConfig( topology);
        pVanguardConfig->makeNodes();
        mFabric = pVanguardConfig->mFabric;
        break;

    case EXPLORER8_CONFIG:

        pExplorer8Config = new explorer8Config( topology);
        pExplorer8Config->makeNodes();
        mFabric = pExplorer8Config->mFabric;
        break;

    case DGX2_CONFIG:

        pExplorer16Config = new explorer16Config( topology);
        pExplorer16Config->makeNodes();
        mFabric = pExplorer16Config->mFabric;
        break;

    case DGX2_TRUNK_SPRAY_CONFIG:

        pExplorer16TrunkSprayConfig = new explorer16TrunkSprayConfig( topology);
        pExplorer16TrunkSprayConfig->makeNodes();
        mFabric = pExplorer16TrunkSprayConfig->mFabric;
        break;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    case DGX2_KT_2VM_CONFIG:

        pExplorer16KTConfig = new explorer16KTConfig( topology);
        pExplorer16KTConfig->makeNodes();
        mFabric = pExplorer16KTConfig->mFabric;
        break;
#endif

    case EXPLORER2_CONFIG:

        pExplorer2Config = new explorer2Config( topology);
        pExplorer2Config->makeNodes();
        mFabric = pExplorer2Config->mFabric;
        break;

    case EXPLORER8LB_CONFIG:

        pExplorer8LBConfig = new explorer8LBConfig( topology);
        pExplorer8LBConfig->makeNodes();
        mFabric = pExplorer8LBConfig->mFabric;
        break;

    case EXPLORER8BW_CONFIG:

        pExplorer8BW = new explorerConfig( topology);
        pExplorer8BW->makeNodes();
        mFabric = pExplorer8BW->mFabric;
        break;

    case EXPLORER8SL_CONFIG:

        pExplorer8SL = new explorerSkylakeConfig( topology);
        pExplorer8SL->makeNodes();
        mFabric = pExplorer8SL->mFabric;
        break;

    case EXPLORER_LOOP:

        pExplorerLoop = new explorerLoop( topology);
        pExplorerLoop->makeNodes();
        mFabric = pExplorerLoop->mFabric;
        break;

    case EMULATION_CONFIG:

        pConfigEmu = new emulationConfig( topology);
        pConfigEmu->makeNodes();
        mFabric = pConfigEmu->mFabric;
        break;

    case HGX2_BASEBOARD1_LOOP:
    case HGX2_BASEBOARD2_LOOP:
    case HGX2_TWO_BASEBOARDS_LOOP:
        pHgx2LBConfig =  new hgx2LBConfig( topology);
        pHgx2LBConfig->makeNodes();
        mFabric = pHgx2LBConfig->mFabric;
        break;

    case LR_EMULATION_CONFIG:
        pLrEmulationConfig = new lrEmulationConfig(topology);
        pLrEmulationConfig->makeNodes();
        mFabric = pLrEmulationConfig->mFabric;
        break;

    case LS_FSF_CONFIG:
        pLsFsfConfig = new lsFsfConfig(topology);
        pLsFsfConfig->makeNodes();
        mFabric = pLsFsfConfig->mFabric;
        break;

    case LS_EMULATION_CONFIG:
        pLsEmulationConfig = new lsEmulationConfig(topology);
        // pLsEmulationConfig->setConfig(false, false, NULL);
        pLsEmulationConfig->makeNodes();
        mFabric = pLsEmulationConfig->mFabric;
        break;

    case DGXA100_HGXA100:
        pDeltaConfig = new deltaConfig(topology);
        pDeltaConfig->setConfig(false, false, NULL);
        pDeltaConfig->makeNodes();
        mFabric = pDeltaConfig->mFabric;
        break;

    case DELTA_SHARED_PARTITION:
    {
        std::string jsonFile;
        std::cout << "Enter Shared Fabric Partition JSON file: ";
        std::cin >> jsonFile;

        pDeltaConfig = new deltaConfig(topology);
        pDeltaConfig->setConfig(false, false, jsonFile.c_str());
        pDeltaConfig->makeNodes();
        mFabric = pDeltaConfig->mFabric;

        pSystemPartInfo = pDeltaConfig->mSystemPartInfo;
        break;
    }

    case BASIC_E4700_PG506_PG506:
        pE4700Config1 = new basicE4700Config1(topology);
        pE4700Config1->setE4700Config(basicE4700Config1::PG506,        // sxm4Slot0
                                      basicE4700Config1::PG506,        // sxm4Slot1
                                      basicE4700Config1::NotPopulated, // exaMAXslot0
                                      basicE4700Config1::NotPopulated, // exaMAXslot1
                                      false,                           // enableWarmup
                                      false,                           // enableSpray
                                      false);                          // useTrunkPort

        pE4700Config1->makeNodes();
        mFabric = pE4700Config1->mFabric;
        break;

    case BASIC_E4700_PG506_PG506_E4705:
        pE4700Config1 = new basicE4700Config1(topology);
        pE4700Config1->setE4700Config(basicE4700Config1::PG506,        // sxm4Slot0
                                      basicE4700Config1::PG506,        // sxm4Slot1
                                      basicE4700Config1::E4705,        // exaMAXslot0
                                      basicE4700Config1::E4705,        // exaMAXslot1
                                      false,                           // enableWarmup
                                      false,                           // enableSpray
                                      true);                           // useTrunkPort

        pE4700Config1->makeNodes();
        mFabric = pE4700Config1->mFabric;
        break;

    case BASIC_E4700_QUAD_PG506:
        pE4700Config2 = new basicE4700Config2(topology);
        pE4700Config2->setE4700Config(false, false);
        pE4700Config2->makeNodes();
        mFabric = pE4700Config2->mFabric;
        break;

    case BASIC_E4700_QUAD_PG506_2K_ENDPOINT:
        pE4700Config3 = new basicE4700Config3(topology);
        pE4700Config3->setE4700Config(false, false);
        pE4700Config3->makeNodes();
        mFabric = pE4700Config3->mFabric;
        break;

    case VULCAN_SURROGATE:
        pVulcanSurrogate = new vulcanSurrogate(topology);
        pVulcanSurrogate->setConfig(false, false, NULL);
        pVulcanSurrogate->makeNodes();
        mFabric = pVulcanSurrogate->mFabric;
        break;

    case VULCAN_SURROGATE_EXTERNAL_LOOPBACK:
        pVulcanSurrogate = new vulcanSurrogate(topology);
        pVulcanSurrogate->setConfig(true, false, NULL);
        pVulcanSurrogate->makeNodes();
        mFabric = pVulcanSurrogate->mFabric;
        break;

    case VULCAN:
        pVulcan = new vulcan(topology);
        pVulcan->setConfig(false, false, NULL);
        pVulcan->makeNodes();
        mFabric = pVulcan->mFabric;
        break;

    case BASIC_E4840_DUAL_PG520:
        pE4840Config1 = new basicE4840Config1(topology);
        pE4840Config1->setE4840Config(basicE4840Config1::PG520,        // sxm5Slot0
                                      basicE4840Config1::PG520,        // sxm5Slot1
                                      false,                           // enableWarmup
                                      false,                           // enableSpray
                                      false);                          // useTrunkPort

        pE4840Config1->makeNodes();
        mFabric = pE4840Config1->mFabric;
        break;

    case BASIC_E4840_QUAD_PG520:
        pE4840Config2 = new basicE4840Config2(topology);
        pE4840Config2->setE4840Config(false,                           // enableWarmup
                                      false,                            // enableSpray
                                      false);                           // useTrunkPort

        pE4840Config2->makeNodes();
        mFabric = pE4840Config2->mFabric;
        break;

    default:
        FM_LOG_ERROR("unknown topology %d.\n", topology);
        return;
    }

    topoFile = fopen( topoBinFileName, "w+" );
    if ( topoFile == NULL )
    {
        FM_LOG_ERROR("Failed to open topology file %s\n", topoBinFileName);
        return;
    }

    // write the binary topology file
    if ( pSystemPartInfo )
    {
        // partitions only
        fileLength = pSystemPartInfo->ByteSize();
    }
    else
    {
        // entire topology
        fileLength = mFabric->ByteSize();
    }

    bufToWrite = new char[fileLength];

    if ( pSystemPartInfo )
    {
        // partitions only
        pSystemPartInfo->SerializeToArray( bufToWrite, fileLength );
    }
    else
    {
        // entire topology
        mFabric->SerializeToArray( bufToWrite, fileLength );
    }

    bytesWritten = fwrite( bufToWrite, 1, fileLength, topoFile );
    if ( bytesWritten != fileLength )
    {
        FM_LOG_ERROR("Incomplete write of %d bytes. %d bytes requested\n",
                    bytesWritten, fileLength);
    } else
    {
        FM_LOG_DEBUG("Complete write of %d bytes.\n", bytesWritten);
    }

    fclose( topoFile );

    // write the text topology file
    topoTextFile = fopen( topoTextFileName, "w+" );
    if ( topoTextFile == NULL )
    {
        printf("%s: Failed to open topology text file %s\n",
               __FUNCTION__, topoTextFileName);
        return;
    }

    if ( pSystemPartInfo )
    {
        // partitions only
        google::protobuf::TextFormat::PrintToString(*pSystemPartInfo, &configText);
    }
    else
    {
        // entire topology
        google::protobuf::TextFormat::PrintToString(*mFabric, &configText);
    }

    bytesWritten = fwrite( configText.c_str(), 1,
                           (int)configText.length(), topoTextFile );
    if ( bytesWritten != configText.length() )
    {
        FM_LOG_ERROR("Incomplete write of %d bytes. %d bytes requested\n",
                     bytesWritten, (int)configText.length());
    } else
    {
        FM_LOG_DEBUG("Complete write of %d bytes.\n", bytesWritten);
    }

    fclose( topoTextFile );

    // colwert decimal value in the text topology file to hex
    decimalToHex ( topoTextFileName, topoTextHexFileName );
}

static void generateTopo(int option)
{
        switch (option) {
            case 0: {
                std::cout << " Exiting" << std::endl;
                exit(0);
            }

            case 1: {
                std::cout << "creating vanguard.topo.bin" << std::endl;
                genFabricConfig(VANGUARD_CONFIG,
                                "vanguard.topo.bin",
                                "vanguard.topo.txt",
                                "vanguard.topo_hex.txt");
                break;
            }

            case 2: {
                std::cout << "creating dgx2.topology" << std::endl;
                genFabricConfig(DGX2_CONFIG,
                                "dgx2.topology",
                                "dgx2.topology.txt",
                                "dgx2.topology_hex.txt");
                break;
            }

            case 3: {
                std::cout << "creating explorer8BW.topo.bin" << std::endl;
                genFabricConfig(EXPLORER8BW_CONFIG,
                                "explorer8BW.topo.bin",
                                "explorer8BW.topo.txt",
                                "explorer8BW.topo_hex.txt");
                break;
            }

            case 4: {
                std::cout << "creating explorer8SL.topo.bin" << std::endl;
                genFabricConfig(EXPLORER8SL_CONFIG,
                                "explorer8SL.topo.bin",
                                "explorer8SL.topo.txt",
                                "explorer8SL.topo_hex.txt");
                break;
            }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            case 5: {
                std::cout << "creating dgx2KT.topology" << std::endl;
                genFabricConfig(DGX2_KT_2VM_CONFIG,
                                "dgx2KT.topology",
                                "dgx2KT.topology.txt",
                                "dgx2KT.topology_hex.txt");
                break;
            }
#endif
            case 6: {
                std::cout << "creating deltaConfig.topology" << std::endl;
                genFabricConfig(DGXA100_HGXA100,
                                "deltaConfig.topology",
                                "deltaConfig.topology.txt",
                                "deltaConfig.topology_hex.txt");
                break;
            }

            case 7: {
                std::cout << "creating deltaSharedPartition.topology" << std::endl;
                genFabricConfig(DELTA_SHARED_PARTITION,
                                "deltaSharedPartition.topology",
                                "deltaSharedPartition.topology.txt",
                                "deltaSharedPartition.topology_hex.txt");
                break;
            }

            case 8: {
                std::cout << "creating E4700_QUAD_PG506 config" << std::endl;
                genFabricConfig(BASIC_E4700_QUAD_PG506,
                                "E4700_QUAD_PG506.topology",
                                "E4700_QUAD_PG506.topology.txt",
                                "E4700_PG506_PG506_E4705.topology_hex.txt");
                break;
            }

            case 9: {
                std::cout << "creating lsFsfConfig.topology" << std::endl;
                genFabricConfig(LS_FSF_CONFIG,
                                "lsFsfConfig.topology",
                                "lsFsfConfig.topology.txt",
                                "lsFsfConfig.topology_hex.txt");
                break;
            }

            case 10: {
                std::cout << "creating lsEmulationConfig.topology" << std::endl;
                genFabricConfig(LS_EMULATION_CONFIG,
                                "lsEmulationConfig.topology",
                                "lsEmulationConfig.topology.txt",
                                "lsEmulationConfig.topology_hex.txt");
                break;
            }

            case 11: {
                std::cout << "creating vulcanSurrogate.topology" << std::endl;
                genFabricConfig(VULCAN_SURROGATE,
                                "vulcanSurrogate.topology",
                                "vulcanSurrogate.topology.txt",
                                "vulcanSurrogate.topology_hex.txt");
                break;
            }

            case 12: {
                std::cout << "creating vulcanSurrogateExtLoopback.topology" << std::endl;
                genFabricConfig(VULCAN_SURROGATE_EXTERNAL_LOOPBACK,
                                "vulcanSurrogateExtLoopback.topology",
                                "vulcanSurrogateExtLoopback.topology.txt",
                                "vulcanSurrogateExtLoopback.topology_hex.txt");
                break;
            }

            case 13: {
                std::cout << "creating vulcan.topology" << std::endl;
                genFabricConfig(VULCAN,
                                "vulcan.topology",
                                "vulcan.topology.txt",
                                "vulcan.topology_hex.txt");
                break;
            }

            case 14: {
                std::cout << "creating E4840_DUAL_PG520 config" << std::endl;
                genFabricConfig(BASIC_E4840_DUAL_PG520,
                                "E4840_DUAL_PG520.topology",
                                "E4840_DUAL_PG520.topology.txt",
                                "E4840_DUAL_PG520.topology_hex.txt");
                break;
            }

            case 15: {
                std::cout << "creating E4840_QUAD_PG520 config" << std::endl;
                genFabricConfig(BASIC_E4840_QUAD_PG520,
                                "E4840_QUAD_PG520.topology",
                                "E4840_QUAD_PG520.topology.txt",
                                "E4840_QUAD_PG520.topology_hex.txt");
                break;
            }
        }
}
int main()
{
    int option;

    fabricManagerInitLog(FABRIC_TOOL_ELW_DBG_LVL, (char *)FABRIC_TOOL_ELW_DBG_FILE,
                         FABRIC_TOOL_ELW_DBG_APPEND, FABRIC_TOOL_MAX_LOG_FILE_SIZE,
                         FABRIC_TOOL_USE_SYS_LOG);

    while (1) {
        std::cout <<"\n";
        std::cout <<" 1. Generate VANGUARD_CONFIG \n";
        std::cout <<" 2. Generate DGX2_CONFIG \n";
        std::cout <<" 3. Generate EXPLORER8 BROADWELLL CONFIG \n";
        std::cout <<" 4. Generate EXPLORER8 SKYLAKE CONFIG \n";

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        std::cout <<" 5. Generate DGX2_KT_CONFIG \n";
#endif
        std::cout <<" 6. Generate DGXA100_HGXA100 (DELTA_CONFIG) \n";
        std::cout <<" 7. Generate DGXA100_HGXA100_SHARED_PARTITION (DELTA) \n";
        std::cout <<" 8. Generate BASIC_E4700_QUAD_PG506 \n";
        std::cout <<" 9. Generate LS_FSF_CONFIG \n";
        std::cout <<" 10. Generate LS_EMULATION_CONFIG \n";

        std::cout <<" 11. Generate VULCAN_SURROGATE \n";
        std::cout <<" 12. Generate VULCAN_SURROGATE_EXTERNAL_LOOPBACK\n";

        std::cout <<" 13. Generate VULCAN \n";
        std::cout <<" 14. Generate E4840_DUAL_PG520\n";
        std::cout <<" 15. Generate E4840_QUAD_PG520\n";

        std::cout <<" 0. Exit \n";

        std::cout << "Enter your choice ";
        std::cin >> option; 

        generateTopo(option);
    }

    return 0;
}
