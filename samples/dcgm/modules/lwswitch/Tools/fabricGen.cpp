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
#include "DcgmFMCommon.h"
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
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
#include "lrEmulationConfig.h"
#endif

#include "emulationConfig.h"

#define LINE_BUF_SIZE 512
/* Logging-related elwironmental variables */
#define FABRIC_GEN_ELW_DBG_LVL     "__FABRIC_GEN_DBG_LVL"
#define FABRIC_GEN_ELW_DBG_APPEND  "__FABRIC_GEN_DBG_APPEND"
#define FABRIC_GEN_ELW_DBG_FILE    "__FABRIC_GEN_DBG_FILE"

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
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    lrEmulationConfig *pLrEmulationConfig;
#endif
    fabric *mFabric;

    if ( (topoBinFileName == NULL) ||
         (topoTextFileName == NULL) ||
         (topoTextHexFileName == NULL) ) {
        PRINT_ERROR("","invalid input files.\n");
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

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    case LR_EMULATION_CONFIG:
        pLrEmulationConfig = new lrEmulationConfig(topology);
        pLrEmulationConfig->makeNodes();
        mFabric = pLrEmulationConfig->mFabric;
        break;
#endif

    default:
        PRINT_ERROR("%d", "unknown topology %d.\n", topology);
        return;
    }

    topoFile = fopen( topoBinFileName, "w+" );
    if ( topoFile == NULL )
    {
        PRINT_ERROR("%s", "Failed to open topology file %s\n", topoBinFileName);
        return;
    }

    // write the binary topology file
    fileLength = mFabric->ByteSize();

    bufToWrite = new char[fileLength];
    mFabric->SerializeToArray( bufToWrite, fileLength );

    bytesWritten = fwrite( bufToWrite, 1, fileLength, topoFile );
    if ( bytesWritten != fileLength )
    {
        PRINT_ERROR("%d,%d", "Incomplete write of %d bytes. %d bytes requested\n",
                    bytesWritten, fileLength);
    } else
    {
        PRINT_DEBUG("%d", "Complete write of %d bytes.\n", bytesWritten);
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

    google::protobuf::TextFormat::PrintToString(*mFabric, &configText);
    bytesWritten = fwrite( configText.c_str(), 1,
                           (int)configText.length(), topoTextFile );
    if ( bytesWritten != configText.length() )
    {
        PRINT_ERROR("%d,%d", "Incomplete write of %d bytes. %d bytes requested\n",
                     bytesWritten, (int)configText.length());
    } else
    {
        PRINT_DEBUG("%d", "Complete write of %d bytes.\n", bytesWritten);
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
                std::cout << "creating twoPG503onDev0.topo.bin" << std::endl;
                genFabricConfig(BASIC_E3600_CONFIG1,
                                "twoPG503onDev0.topo.bin",
                                "twoPG503onDev0.topo.txt",
                                "twoPG503onDev0.topo_hex.txt");
                break;
            }
            case 2: {
                std::cout << "creating BasicE3600Config2.topo.bin" << std::endl;
                genFabricConfig(BASIC_E3600_CONFIG2,
                                "BasicE3600Config2.topo.bin",
                                "BasicE3600Config2.topo.txt",
                                "BasicE3600Config2.topo_hex.txt");
                break;
            }
            case 3: {
                std::cout << "creating BasicE3600Config3.topo.bin" << std::endl;
                genFabricConfig(BASIC_E3600_CONFIG3,
                                "BasicE3600Config3.topo.bin",
                                "BasicE3600Config3.topo.txt",
                                "BasicE3600Config3.topo_hex.txt");
                break;
            }
            case 4: {
                std::cout << "creating EmulationConfig.topo.bin" << std::endl;         
                genFabricConfig(EMULATION_CONFIG,
                                "EmulationConfig.topo.bin",
                                "EmulationConfig.topo.txt",
                                "EmulationConfig.topo_hex.txt");
                break;
            }
            case 5: {
                std::cout << "creating oneE3620OneGpuOnDev0P16P17.topo.bin" << std::endl;
                genFabricConfig(BASIC_E3600_CONFIG4,
                                "oneE3620OneGpuOnDev0P16P17.topo.bin",
                                "oneE3620OneGpuOnDev0P16P17.topo.txt",
                                "oneE3620OneGpuOnDev0P16P17.topo_hex.txt");
                break;
            }
            case 6: {
                std::cout << "creating onePG503OneGpuOnDev0P0To5.topo.bin" << std::endl;
                genFabricConfig(BASIC_E3600_CONFIG5,
                                "onePG503OneGpuOnDev0P0To5.topo.bin",
                                "onePG503OneGpuOnDev0P0To5.topo.txt",
                                "onePG503OneGpuOnDev0P0To5.topo_hex.txt");
                break;
            }
            case 7: {
                std::cout << "creating onePG503OneGpuOnDev0P0To5Dev1P6P7.topo.bin" << std::endl;
                genFabricConfig(BASIC_E3600_CONFIG6,
                                "onePG503OneGpuOnDev0P0To5Dev1P6P7.topo.bin",
                                "onePG503OneGpuOnDev0P0To5Dev1P6P7.topo.txt",
                                "onePG503OneGpuOnDev0P0To5Dev1P6P7.topo_hex.txt");
                break;
            }
            case 8: {
                std::cout << "creating onePG503OneGpuOnDev0P0To5onePG503OneGpuOnDev1P0To5.topo.bin" << std::endl;
                genFabricConfig(BASIC_E3600_CONFIG7,
                                "onePG503OneGpuOnDev0P0To5onePG503OneGpuOnDev1P0To5.topo.bin",
                                "onePG503OneGpuOnDev0P0To5onePG503OneGpuOnDev1P0To5.topo.txt",
                                "onePG503OneGpuOnDev0P0To5onePG503OneGpuOnDev1P0To5.topo_hex.txt");
                break;
            }
            case 9: {
                std::cout << "creating vanguard.topo.bin" << std::endl;
                genFabricConfig(VANGUARD_CONFIG,
                                "vanguard.topo.bin",
                                "vanguard.topo.txt",
                                "vanguard.topo_hex.txt");
                break;
            }
            case 10: {
                std::cout << "creating explorer8.topo.bin" << std::endl;
                genFabricConfig(EXPLORER8_CONFIG,
                                "explorer8.topo.bin",
                                "explorer8.topo.txt",
                                "explorer8.topo_hex.txt");
                break;
            }
            case 11: {
                std::cout << "creating dgx2.topology" << std::endl;
                genFabricConfig(DGX2_CONFIG,
                                "dgx2.topology",
                                "dgx2.topology.txt",
                                "dgx2.topology_hex.txt");
                break;
            }
            case 12: {
                std::cout << "creating explorer2.topo.bin" << std::endl;
                genFabricConfig(EXPLORER2_CONFIG,
                                "explorer2.topo.bin",
                                "explorer2.topo.txt",
                                "explorer2.topo_hex.txt");
                break;
            }
            case 13: {
                std::cout << "creating explorer8LB.topo.bin" << std::endl;
                genFabricConfig(EXPLORER8LB_CONFIG,
                                "explorer8LB.topo.bin",
                                "explorer8LB.topo.txt",
                                "explorer8LB.topo_hex.txt");
                break;
            }
           case 14: {
                std::cout << "creating explorer8BW.topo.bin" << std::endl;
                genFabricConfig(EXPLORER8BW_CONFIG,
                                "explorer8BW.topo.bin",
                                "explorer8BW.topo.txt",
                                "explorer8BW.topo_hex.txt");
                break;
            }
           case 15: {
                std::cout << "creating explorer8SL.topo.bin" << std::endl;
                genFabricConfig(EXPLORER8SL_CONFIG,
                                "explorer8SL.topo.bin",
                                "explorer8SL.topo.txt",
                                "explorer8SL.topo_hex.txt");
                break;
            }
           case 16: {
                std::cout << "creating hgx2BaseBoard1LB.topo.bin" << std::endl;
                genFabricConfig(HGX2_BASEBOARD1_LOOP,
                                "hgx2BaseBoard1LB.topo.bin",
                                "hgx2BaseBoard1LB.topo.txt",
                                "hgx2BaseBoard1LB.topo_hex.txt");
                break;
            }
           case 17: {
                std::cout << "creating hgx2BaseBoard2LB.topo.bin" << std::endl;
                genFabricConfig(HGX2_BASEBOARD2_LOOP,
                                "hgx2BaseBoard2LB.topo.bin",
                                "hgx2BaseBoard2LB.topo.txt",
                                "hgx2BaseBoard2LB.topo_hex.txt");
                break;
            }
           case 18: {
                std::cout << "creating hgx2BothBaseBoardLB.topo.bin" << std::endl;
                genFabricConfig(HGX2_TWO_BASEBOARDS_LOOP,
                                "hgx2BothBaseBoardLB.topo.bin",
                                "hgx2BothBaseBoardLB.topo.txt",
                                "hgx2BothBaseBoardLB.topo_hex.txt");
                break;
            }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            case 19: {
                std::cout << "creating dgx2KT.topology" << std::endl;
                genFabricConfig(DGX2_KT_2VM_CONFIG,
                                "dgx2KT.topology",
                                "dgx2KT.topology.txt",
                                "dgx2KT.topology_hex.txt");
                break;
            }
#endif

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
            case 20: {
                std::cout << "creating lrEmulationConfig.topology" << std::endl;
                genFabricConfig(LR_EMULATION_CONFIG,
                                "lrEmulationConfig.topology",
                                "lrEmulationConfig.topology.txt",
                                "lrEmulationConfig.topology_hex.txt");
                break;
            }
#endif
            case 21: {
                std::cout << "creating dgx2_trunk_spray.topology" << std::endl;
                genFabricConfig(DGX2_TRUNK_SPRAY_CONFIG,
                                "dgx2_trunk_spray.topology",
                                "dgx2_trunk_spray.topology.txt",
                                "dgx2_trunk_spray.topology_hex.txt");
                break;
            }
        }
}
int main()
{
    int option;

    loggingInit((char *)FABRIC_GEN_ELW_DBG_LVL, (char *)FABRIC_GEN_ELW_DBG_APPEND,
                (char *)FABRIC_GEN_ELW_DBG_FILE);

    while (1) {
        std::cout <<"\n";
        std::cout <<" 1. Generate BASIC_E3600_CONFIG1 \n";
        std::cout <<" 2. Generate BASIC_E3600_CONFIG2 \n";
        std::cout <<" 3. Generate BASIC_E3600_CONFIG3 \n";
        std::cout <<" 4. Generate EMULATION_CONFIG \n";
        std::cout <<" 5. Generate BASIC_E3600_CONFIG4 \n";
        std::cout <<" 6. Generate BASIC_E3600_CONFIG5 \n";
        std::cout <<" 7. Generate BASIC_E3600_CONFIG6 \n";
        std::cout <<" 8. Generate BASIC_E3600_CONFIG7 \n";
        std::cout <<" 9. Generate VANGUARD_CONFIG \n";
        std::cout <<" 10. Generate EXPLORER8_CONFIG \n";
        std::cout <<" 11. Generate DGX2_CONFIG \n";
        std::cout <<" 12. Generate EXPLORER2_CONFIG \n";
        std::cout <<" 13. Generate EXPLORER8_LOOPBACK_CONFIG \n";
        std::cout <<" 14. Generate EXPLORER8 BROADWELLL CONFIG \n";
        std::cout <<" 15. Generate EXPLORER8 SKYLAKE CONFIG \n";
        std::cout <<" 16. Generate HGX2_BASEBOARD1_LOOP \n";
        std::cout <<" 17. Generate HGX2_BASEBOARD2_LOOP \n";
        std::cout <<" 18. Generate HGX2_TWO_BASEBOARDS_LOOP \n";
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        std::cout <<" 19. Generate DGX2_KT_CONFIG \n";
#endif

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        std::cout <<" 20. Generate LR_EMULATION_CONFIG \n";
#endif
        std::cout <<" 21. Generate DGX2_TRUNK_SPRAY_CONFIG \n";
        std::cout <<" 0. Exit \n";

        std::cout << "Enter your choice ";
        std::cin >> option; 

        generateTopo(option);
    }

    return 0;
}
