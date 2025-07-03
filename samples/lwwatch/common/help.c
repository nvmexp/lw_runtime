/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwwatch.h"
#include "help.h"
#include "elpg.h"
#include "vic.h"
#include "msenc.h"
#include "ofa.h"
#include "hda.h"
#include "ce.h"
#include "lwdec.h"
#include "lwjpg.h"
#include "acr.h"
#include "falcphys.h"
#include "vpr.h"
#include "intr.h"

void printHelpMenu()
{
#if LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
        printUsage_win_standalone();
#endif

    // INIT ROUTINES
    dprintf("INIT Routines:\n");
    dprintf(" init [lwBar0] [lwBar1]    - Searches for LW devices and sets lwBar0 and lwBar1\n");
    dprintf("                             + if no LW device is found, user must specify lwBar0 and lwBar1\n");
    dprintf(" multigpu [d0Bar0] [d1Bar0] [d2Bar0] ...\n");
    dprintf("                           - When exelwting cmds, loop over all GPUs supplied\n");

#if !defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
    dprintf(" modsinit [DebugLevel]     - Initializes lwwatchMods\n");
    dprintf("                             + 0 - all messages (default), 1 - level 1, 2 - level 2, 10 - none\n");
    dprintf("                             + debug file: %s\n", LWWATCH_OC_DEBUG_FILE);
#endif

#if defined(LWDEBUG_SUPPORTED)
    dprintf(" dumpinit <zip file> [inner file]\n");
    dprintf("                           - Initializes LwWatch from a protobuf dump file.\n");
    dprintf("                             + default inner file: rm_00.pb\n");
#endif

    dprintf(" msi                       - Prints the MSI enable status for each GPU\n");
    dprintf(" verbose <DebugLevel>      - Sets the output verbosity\n");
    dprintf("                             + 0 is less verbose\n");
    dprintf("                             + 1 is more verbose - default\n");
    dprintf("                             + 2 is most verbose\n");
    dprintf(" g_elw <elw>               - Prints the value of the given elw\n");
    dprintf(" s_elw <elw> <value>       - Sets the elw to the given value\n");
    dprintf("\n");

#if defined(WIN32) && !defined(USERMODE)
    // SW STATE ROUTINES
    dprintf("SW STATE Routines:\n");
    dprintf(" rmjournal [maxRecords]    - Dumps the RM Journal for all GPUs\n");
    dprintf(" rmrc [maxRecords]         - Dumps the RM RC Error List for all GPUs\n");
    dprintf(" rmlocks [-t]              - Dumps the RM Lock state\n");
    dprintf("                             -t dumps trace record information\n");
    dprintf(" rmthread                  - Dumps the RM ThreadState Database\n");
    dprintf(" lwlog                     - Dumps the RM LwLog debug output\n");
    dprintf("\n");

    // OBJECT DATABASE ROUTINES
    dprintf("OBJECT DATABASE Routines:\n");
    dprintf(" odbdump                   - Dumps the object database hierarchy\n");
    dprintf(" pdbdump -class <OdbClass> [-handle <ObjHandle>]\n");
    dprintf("                           - Dumps the property database for a specific object\n");
    dprintf("                             -class <OdbClass> specifies the object class\n");
    dprintf("                             -handle <ObjHandle> optionally specifies the object handle\n");
    dprintf("\n");
#endif // WIN32 && !USERMODE

    // SIGDUMP ROUTINES
    dprintf("SIGDUMP Routines:\n");
    dprintf(" sigdump                   - Print out a signal dump into file sigdump.txt\n");
    dprintf("\n");

    // REGISTER AND MEMORY ROUTINES
    dprintf("REGISTER and MEMORY Routines:\n");
    dprintf(" rd [-p] [-a] [-d] [-l] [-grIdx <grIdx>] <addr> [length] - Reads a dword from lwBar0 + addr\n");
    dprintf("                         -p Prints a dword from lwBar0 + addr in parsed format\n");
    dprintf("                         -a Used with -p to print all entries in an index register;\n");
    dprintf("                            Only has an effect if addr is an indexed register entry\n");
    dprintf("                         -d Used only for T124 right now to specify the CheetAh device to read from\n");
    dprintf("                         -l Used with the -d option to list the available CheetAh devices\n");
    dprintf("                         + default length is 4 bytes\n");
    dprintf("                         -grIdx Used only when SMC is enabled to specify the GR engine ID to which the addr belogs to\n");
    dprintf(" wr [-d] [-grIdx <grIdx>] <addr> <val32>   - Writes a dword to lwBar0 + addr\n");
    dprintf("                        -d Used only for T124 right now to specify the CheetAh device to write to\n");
    dprintf("                        -grIdx Used only when SMC is enabled to specify the GR engine ID to which the addr belogs to\n");
    dprintf(" rb [-grIdx <grIdx>] <addr>   - Reads a byte from lwBar0 + addr\n");
    dprintf("                              -grIdx Used only when SMC is enabled to specify the GR engine ID to which the addr belogs to\n");
    dprintf(" wb [-grIdx <grIdx>] <addr> <val08>  - Writes a byte to lwBar0 + addr\n");
    dprintf("                              -grIdx Used only when SMC is enabled to specify the GR engine ID to which the addr belogs to\n");
    dprintf(" rw [-grIdx <grIdx>] <addr>   - Reads a word from lwBar0 + addr\n");
    dprintf("                              -grIdx Used only when SMC is enabled to specify the GR engine ID to which the addr belogs to\n");
    dprintf(" ww [-grIdx <grIdx>] <addr> <val16>  - Writes a word to lwBar0 + addr\n");
    dprintf("                              -grIdx Used only when SMC is enabled to specify the GR engine ID to which the addr belogs to\n");

#if !defined(USERMODE)
    dprintf(" pd [-grIdx <grIdx>] <name>  - Dumps a Register using its name or address\n");
    dprintf("                              -grIdx Used only when SMC is enabled\n");
    dprintf(" pe <name>.<field> <data>  - Writes to a Register/field given its name\n");
#endif

#if !defined(USERMODE)
    if (!LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX))
    {
        dprintf(" ptov <physAddr> [optFlags]\n");
        dprintf("                           - Gives the virt addr of the phys addr supplied\n");
        dprintf("                             + set optFlags BIT(0) to give all virt addrs\n");
    }
    if (!LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(UNIX_HWSNOOP))
    {
        dprintf(" vtop <virtAddr>           - Gives the phys addr of the virt addr supplied\n");
    }
    dprintf("\n");
#endif

    // G80+ MEM and FB ROUTINES
    dprintf("G80+ MEM and FB Routines:\n");
    dprintf(" gvrd -smmu/iommu <asId> <vAddr> [length] OR\n");
    dprintf(" gvrd -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> [length] OR\n");
    dprintf(" gvrd -gmmu <-ch <rlid> <chid>> <vAddr> [length] OR\n");
    dprintf(" gvrd -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> [length]\n");
    dprintf("                           - Display data at specified chId's GPU virtual adddress for length in bytes\n");
    dprintf(" gvwr -smmu/iommu <asId> <vAddr> <data> [data] ... OR\n");
    dprintf(" gvwr -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> <data> [data] ... OR\n");
    dprintf(" gvwr -gmmu <-ch <rlid> <chid>> <vAddr> <data> [data] ... OR\n");
    dprintf(" gvwr -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> <data> [data] ...\n");
    dprintf("                           - Writes data at specified chId's GPU virtual adddress\n");
    dprintf(" gvfill -smmu <asId> <vAddr> <data> <length> OR\n");
    dprintf(" gvfill -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> <data> <length> OR\n");
    dprintf(" gvfill -gmmu <-ch <rlid> <chid>> <vAddr> <data> <length> OR\n");
    dprintf(" gvfill -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr> <data> <length>\n");
    dprintf("                           - Fills data at specified chId's GPU virtual adddress\n");
    dprintf(" gvcp -smmu/iommu <asId> <vSrcAddr> <vDstAddr> [length] OR\n");
    dprintf(" gvcp -gmmu <-bar1/-bar2/-ifb/<chid> <vSrcAddr> <vDstAddr> [length] OR\n");
    dprintf(" gvcp -gmmu <-ch <rlid> <chid>> <vSrcAddr> <vDstAddr> [length] OR\n");
    dprintf(" gvcp -gmmu -instptr <instptr> <-vidmem/-sysmem> <vSrcAddr> <vDstAddr> [length]\n");
    dprintf("                           - Copies from [vSrcAddr, vSrcAddr + length) to [vDstAddr, vDstAddr + length) in specified chId's GPU virtual address space\n");
    dprintf(" gpde -smmu/iommu <asId> <begin> [end] OR\n");
    dprintf(" gpde -gmmu <-bar1/-bar2/-ifb/<chid> <begin> [end] OR\n");
    dprintf(" gpde -gmmu <-ch <rlid> <chid>> <begin> [end] OR\n");
    dprintf(" gpde -gmmu -instptr <instptr> <-vidmem/-sysmem> <begin> [end]\n");
    dprintf("                           - Dumps the PDE table for channel\n");
    dprintf("                             + from entry <begin> to entry [end]\n");
    dprintf(" gpte -smmu/iommu <asId> <pde_id> <begin> [end] OR\n");
    dprintf(" gpte -gmmu <-bar1/-bar2/-ifb/<chid> <pde_id> <begin> [end] OR\n");
    dprintf(" gpte -gmmu <-ch <rlid> <chid>> <pde_id> <begin> [end] OR\n");
    dprintf(" gpte -gmmu -instptr <instptr> <-vidmem/-sysmem> <pde_id> <begin> [end]\n");
    dprintf("                           - Dumps the PTE table for channel\n");
    dprintf("                             + from directory <pdeId>\n");
    dprintf("                             + between <begin> and [end]\n");
    dprintf(" gvpte -smmu/iommu <asId> <vAddr> OR\n");
    dprintf(" gvpte -gmmu <-bar1/-bar2/-ifb/<chid> <vAddr> OR\n");
    dprintf(" gvpte -gmmu <-ch <rlid> <chid>> <vAddr> OR\n");
    dprintf(" gvpte -gmmu -instptr <instptr> <-vidmem/-sysmem> <vAddr>\n");
    dprintf("                           - Displays the page table entrie corresponding to the specified virtual address\n");
    dprintf(" gvtop -smmu/iommu <asId> <virtAddr> OR\n");
    dprintf(" gvtop -gmmu <-bar1/-bar2/-fla/-ifb/<chid> [flaImbAddr (hex)] [isSysMem] <virtAddr> OR\n");
    dprintf(" gvtop -gmmu <-ch <rlid> <chid>> <virtAddr> OR\n");
    dprintf(" gvtop -gmmu -instptr <instptr> <-vidmem/-sysmem> <virtAddr>\n");
    dprintf("                           - LW41+: Find physical address given a gart addr\n");
    dprintf("                           - G80+: Find 40-bit physical address for <gpuVirtAddr>\n");
    dprintf("                           - Fermi: For BAR1, BAR2, IFB pass 128, 129, 130 for chId\n");
    dprintf(" gptov -smmu/iommu <asId> <physAddr> <-vidmem/-sysmem> OR\n");
    dprintf(" gptov -gmmu <-bar1/-bar2/-ifb/<chid> <physAddr> <-vidmem/-sysmem> OR\n");
    dprintf(" gptov -gmmu <-ch <rlid> <chid>> <physAddr> <-vidmem/-sysmem> OR\n");
    dprintf(" gptov -gmmu -instptr <instptr> <-vidmem/-sysmem> <physAddr> <-vidmem/-sysmem>\n");
    dprintf("                           - G80+: Find gpuVirtAddr address for <physAddr>\n");
    dprintf("                           - Fermi: For BAR1, BAR2, IFB pass 128, 129, 130 for chId\n");

#if defined(WIN32) && !defined(USERMODE)
    dprintf(g_szGvDispUsage);
    dprintf(g_szGCo2blUsage);
    dprintf(g_szGBl2coUsage);
#endif
#if !defined(USERMODE)
    dprintf(" gvdiss <chId> <vAddr> <length> <shaderType>\n");
    dprintf("                           - Disassembles the G80 ucode at the specified offset using sass\n");
    dprintf("                           - For shader type, use the following enum:\n");
    dprintf("                             - 0 - vertex shader\n");
    dprintf("                             - 1 - geometry shader\n");
    dprintf("                             - 2 - pixel shader\n");
    dprintf("                             - 3 - compute shader\n");
    dprintf(" fbrd <fbOffset> [fbLength] [size]\n");
    dprintf("                           - Reads dwords from <fbOffset> to <fbOffset> + [fbLength]\n");
    dprintf("                             + Displays in LwU8/LwU16/LwU32 format depending on <size>\n");
    dprintf("                             + Size should be 1, 2, or 4.  If not, it defaults to 4\n");
    dprintf(" fbwr <fboffset> <val32> [val32] ...\n");
    dprintf("                           - Writes data at (FB base + fbOsffset)\n");
    dprintf(" fbfill <fbOffset> <val32> <fbLength>\n");
    dprintf("                           - Fills data at (FB base + fbOsffset)\n");
    dprintf(" fbcp <fbSrcOffset> <fbDstOffset> [fbLength]\n");
    dprintf("                           - Copies from [fbSrcOffset, fbSrcOffset + fbLength) to [fbDstOffset, fbDstOffset + fbLength)\n");
#endif //!USERMODE

    dprintf(" l2ilwalidate              - Forces a l2 ilwalidation\n");
    dprintf(" fbmonitor              - Checks whether there has been a fb access. Use -read to read the counts, -setup to setup counter for n fbps\n");
    dprintf(" l2state <nPartitions>  - Dumps l2 state for given number of fb partitions\n");
    dprintf(" ismemreq <nPartitions>  - Checks whether there have been pending mem requests\n");

#if !defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX)
    dprintf(" heap <-s sortOption> <-o owner> [optHeapAddr]\n");
    dprintf("                           - Dumps the heap\n");
    dprintf("                             + default optHeapAddr is pGpu->pFb->pHeap if available\n");
    dprintf("                             + sortOption: OWNER_UP = 1, OWNER_DOWN = 2, TYPE_UP = 3, TYPE_DOWN = 4\n");
    dprintf("                             +             ADDR_UP = 5,  ADDR_DOWN = 6,  SIZE_UP = 7, SIZE_DOWN = 8\n");
    dprintf("                             + owner:      String name of owner.  Can be \"FREE\"\n");

    dprintf(" pma\n");
    dprintf("                           - Dumps the pma object\n");
#endif

    dprintf(" tiling                    - LW4x and Prior: Print out tiling registers\n");
    dprintf(" ecc [-a]                  - Dumps ECC status information.\n");
    dprintf("                             + -a Prints more detailed information\n");

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) && !defined(CLIENT_SIDE_RESMAN)
    // BR04 ROUTINES
    dprintf("BR04 Routines:\n");
    dprintf(" br04init <PcieConfigSpaceBase>  - Initialized BR04 functions.  PcieConfigSpaceBase default = 0xe0000000\n");
    dprintf(" br04topology                    - Prints info on the topology of BR04s, showing what is connected on each port\n");
    dprintf(" br04dump <bid>                  - Prints info on the indicated BR04\n");
    dprintf(" br04port <bid> <port>           - Prints details on a specific port of the indicated BR04\n");
    dprintf("\n");
#endif

    // BUS ROUTINES
    dprintf("BUS Interface Routines:\n");
    dprintf(" pcieinfo                  - Get PCI-E info for the GPU link\n");
    dprintf(" pcieinfo all              - Get PCI-E info for the entire hierarchy above the GPU\n");
    dprintf(" pcieinfo <bus> <dev> <func>\n");
    dprintf("                           - Get PCI-E info for the specified link\n");
    dprintf(" pcie <bus> <dev> <func> [offset] [length]\n");
    dprintf("                           - Displays memory values in the PCI Config Space Specified by bus, dev, func\n");
    dprintf("                             + [offset] is the offset to start from\n");
    dprintf("                             + [length] is the number of bytes to display\n");
    dprintf(" pcie3evtlogdmp            - Dump PEX Gen3 event log\n");
    dprintf("\n");

    // CLOCK AND PERF ROUTINES
    dprintf("CLOCK and PERF Routines:\n");
    dprintf(" cntrfreq                  - Computes various clock frequencies based on clock counters\n");
    dprintf(" clocks [-disp]            - Print out mem and lw clock values\n");
    dprintf("                             + -disp Print out all display clock settings (DISP_CLK, SOR_CLK and RG_PCLK).\n");
    dprintf(" clocksregs                - Print out raw clock registers\n");
    dprintf(" plls                      - Print out PLL values\n");
    dprintf(" pllregs                   - Print out raws PLL registers\n");
    dprintf(" powergate                 - Print out powergate status of components\n");
    dprintf(" cluster                   - Print out CPU cluster related registers on T30+\n");
    dprintf(" perfmon                   - Prints out the perfmon registers\n");
    dprintf(" clklutread                - Prints out the requested LUT values for a given NAFLL\n");
    dprintf("\n");

    // DISPLAY ROUTINES
    dprintf("DISPLAY Routines:\n");
    dprintf(" dcb [-f <flags>] [-m <mode> <data>] - Print out the device control block\n");
    dprintf("                                       + -f Send <flags> to DCBCHK. Use 0x0000003F as <flags> to get more information from DCBCHK\n");
    dprintf("                                       + -m Specify <mode> to dump DCB contents using image from the following:\n");
    dprintf("                                       <mode> 0: Dump DCB from vbios image located at bar0+0x7e000.(default if no option given, and try other modes if no valid shadow image existed)\n");
    dprintf("                                       <mode> 1: Dump image using source symbols to get from lwrrently selected GPU's pVbios->pImage (need source symbols)\n");
    dprintf("                                       <mode> 2 <data>: Dump DCB from vbios image located at address <data>\n");
    dprintf(" slivb                               - Dump any registers affected by the SLI video bridge\n");
    dprintf(" getcr <reg> [optHd]                 - Read a cr register\n");
    dprintf("                                       + default head is 0\n");
    dprintf(" setcr <reg> <val> [optHd]           - Write a cr register to the given value\n");
    dprintf("                                       + default head is 0\n");
    dprintf(" tmdsrd <link> <reg>                 - Reads indexed tmds register\n");
    dprintf(" tmdswr <link> <reg> <val>           - Writes indexed tmds register\n");
    dprintf(" dsli <-v>                           - Prints SLI Configuration Information\n");
    dprintf("                                       + -v will turn on verbose option.   \n");
    dprintf(" msa                                 - Bring up the Main Stream Attribute (MSA) menu\n");

#if !defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX)
    dprintf(" i2c                       - Bring up the I2C menu\n");
#endif

    dprintf(" dpauxrd [-d] <phy port>[.concat port 1][.concat port 2][...] <offset> [length]\n"
            "                           - Performs DPAUX read transaction\n"
            "                             + -d Decode the numeric value into string based on dpcd.h\n");
    dprintf(" dpauxwr <phy port>[.concat port 1][.concat port 2][...] <offset> <data 1> [data 2] ...\n"
            "                           - Performs DPAUX write transaction\n");
    dprintf(" dpinfo [physical port] [<sorIndex> <dpIndex>]\n"
            "                           - Prints out display port info\n");
    dprintf(" dpauxlog [GPU instance index]\n"
            "                           - Prints out DpAuxlog\n");
    dprintf("\n");

    // EVO DISPLAY ROUTINES
    dprintf("EVO DISPLAY Routines:\n");
    dprintf(" ddispowner                        - Prints out the owner\n");
    dprintf(" dchnstate <chName> [-h<hd>/-w<wd>]        - Dumps out channel states in readable string\n");
    dprintf(" dchnnum <chName> [-h<hd>/-w<wd>]          - Prints out channel number\n");
    dprintf(" dchnname <chNum>                          - Prints out channel name\n");
    dprintf(" dchnexcept <core/win> -n<errorNum>        - Prints _mfs.c file info based on error number or pending exceptions\n");
    dprintf(" dchnmstate <chName> [-h<hd>/-w<wd>]       - Prints out the ARM and ASSY values for a channel\n");
    dprintf(" dchlwal <chName> <head/win/sor number> <offset> [-assy/-armed]\n");
    dprintf("                                   - Prints display channel method value by address offset in class file(clc57d.h)\n");
    dprintf("                                   - chName can either be core, win, winimm, base, ovly.\n");
    dprintf(" dgetdbgmode <chName> [-h<hd>/-w<wd>]      - Prints out debug mode\n");
    dprintf(" dsetdbgmode <chName> [-h<hd>/-w<wd>] 1/0  - Sets the debug mode\n");
    dprintf(" dinjectmethod <chName> -h<hd>/-w<winNum> mthd1 data1 mthd2 data2\n");
    dprintf("                                   - Inject methods\n"
            "                                     mthd format : METHOD_NAME[@n[@n]]\n"
            "                                     mthd format : Method Address\n"
            "                                     try adding '*' at the end of method or field name\n"
            "                                     NOTE: 1.  -h# can be ignored if it's core\n"
            "                                           2.  Auto DBG restore mode can be ignored\n"
            "                                               if method contains * or !.\n" );
    dprintf(" ddumppb <channelName> [-p] -h<headNum>/-w<winNum> [numDwords/-f] [-o<OffsetDwords>]\n");
    dprintf("                                   - Prints out the display pushbuffer\n");
    dprintf(" danalyzeblank -h<hd> 1/0          - Prints out blank display state\n");
    dprintf(" dhdorconn [-ascii]                - Prints out what OR each head is driving\n");
    dprintf(" dchlwars                          - Prints the hw variables\n");
    dprintf(" danalyzehang                      - Prints analyzing information to debug display hang\n");
    dprintf(" dintr                             - Dumps all the interrupt registers\n");
    //The arguments for dintr are not used in the pre-LwDisplay implementation. They are only of relevance for Volta and later.
    dprintf(" dtim                              - Prints display timing information\n");
    dprintf(" ddesc [handle] [chName] [-h<0|1>]\n");
    dprintf("                                   - Prints out the ctx dma description of the specified handle from display instance memory\n");
    dprintf(" drdhdmidecoder <orNum> <address>  - Reads the hdmi_decoder register with specified address\n");
    dprintf("                                   - To be used only on emulation\n");
    dprintf(" dwrhdmidecoder <orNum> <address> <data>\n");
    dprintf("                                   - Writes given data to hdmi_decoder register with specified address\n");
    dprintf("                                   - To be used only on emulation\n");
    dprintf(" dsorpadlinkconn                   - Prints out the connections of ORs with the associated Padlinks\n");
    dprintf(" ddsc [-h<hd>]\n");
    dprintf("                                   - Prints out Display Stream Compression related information\n");
    dprintf(" dlowpower [-analyze] [-clear] [-counters]  - Print out Display Low Power related information\n");
    dprintf("                                   - [-analyze]    Polls mscg_ok signal for 5 seconds and prints average mscg_ok assert time per frame\n");
    dprintf("                                   - [-clear]      Clears all MSCG related counters\n");
    dprintf("                                   - [-counters]   Prints all MSCG related counters\n");
    dprintf("\n");

    // FIFO ROUTINES
    dprintf("FIFO Routines:\n");
    dprintf(" fifoinfo                  - Gets the current fifo info\n");
    dprintf(" pbinfo                    - Gets pb current info\n");
    dprintf(" runlist                   - Gets runlist info. Use -a to dump whole (only Fermi)\n");
    dprintf(" pbdma                     - Gets pbdma register values for given pbdma unit\n");
    dprintf(" channelram                - Gets channel ram register values for active channels\n");

#ifndef USERMODE
    dprintf(" pb [-p] [optStart] [optEnd]\n");
    dprintf("                           - Dumps the push buffer for the current chid\n");
    dprintf("                             -p Dumps the push buffer in parsed format for the current chid\n");
    dprintf(" pbch [-p] <chid> [optStart] [optEnd]\n");
    dprintf("                           - Dumps the push buffer for the given chid\n");
    dprintf("                             -p Dumps the push buffer in parsed format for the given chid\n");
#endif // USERMODE
    dprintf("\n");

    // GR ROUTINES
    dprintf("GR Routines:\n");
    dprintf(" grinfo                    - Print out gr info\n");
    dprintf(" grstatus [-a] [-grIdx] <grIdx>\n");
    dprintf("                             - Prints a gr state summary, skipping registers/fields == 0\n");
    dprintf("                             -a Prints all registers/fields (including regs for each active GPC / TPC)\n");
    dprintf("                             + SW should include grstatus -a output when reporting HW bugs\n");
    dprintf("                             - grIdx Used only when SMC is enabled to specify the GR engine ID\n");
    dprintf(" grwarppc [repeat] [gpcId tpcId] [-grIdx] <grIdx>\n");
    dprintf("                           - Prints warp PC dumps, read PC for each warp \"repeat\" times.\n");
    dprintf("                             + It can optionally read only from a single TPC\n");
    dprintf("                             + repeat is 8 by default\n");
    dprintf("                             + Set repeat to 0 will only print warp valid / pause / trap masks\n");
    dprintf("                             -grIdx Used only when SMC is enabled to specify the GR engine ID\n");
    dprintf(" gr [-p] [-grIdx <grIdx>]    - Print out gr fifo\n");
    dprintf("                             -p Print out gr fifo in parsed format\n");
    dprintf("                             -grIdx Used only when SMC is enabled to specify the GR engine ID\n");
    dprintf(" subch <subCh>             - Print out gr ctx for a given subCh\n");
    dprintf("                             + for current ctx send a subCh of 0xff\n");
    dprintf("                             + to dump all subch send a subCh of 0xffff\n");
    dprintf(" grctx <chId>              - Print out gr ctx for a given chId\n");
    dprintf(" ssclass <class> [optDis]  - Enables single step for the given class\n");
    dprintf("                             + specify optDis of 1 to disable single step for the given class\n");
    dprintf(" surface                   - Prints out tiling and surface regs (BOFFSET, BLIMIT, BPITCH)\n");
    dprintf(" limiterror                - Analyze a gr color buffer limit error\n");
    dprintf(" launchcheck               - Analyze a primitive launch check error\n");
    dprintf(" zlwll [-grIdx <grIdx>]    - Print out zlwll registers\n");
    dprintf("                           -grIdx Used only when SMC is enabled to specify the GR engine ID\n");
    dprintf(" zlwllram <select> <addr> <size>\n");
    dprintf("                           - Dump out zlwll ram for <size> dwords\n");
    dprintf("                             + starting at <addr>, selected by <select>\n");
    dprintf("\n");
    dprintf(" texparity [-grIdx <grIdx>] - Dumps all the tex parity retry counts per GPC/TPC/PIPE.\n");
    dprintf("                            -grIdx Used only when SMC is enabled to specify the GR engine ID\n");
    dprintf("\n");

    // INSTMEM ROUTINES
    dprintf("INSTMEM Routines:\n");
    dprintf(" fifoctx <chId>                       - Print out fifo ctx for a given chId. (Deprecated Ampere+)\n");
    dprintf(" fifoctx -ch <eng_str> <chid>         - Print out fifo ctx for a given engine and chId\n");
    dprintf(" fifoctx -ch <type> <inst_id> <chid>  - Print out fifo ctx for a given engine and chId\n");
    dprintf(" insttochid <inst> [target]\n");
    dprintf("                           - Prints out the chid for the given engine instance and target\n");
    dprintf(" veidinfo <chId> <veid>    - Prints out valid PDB for a given subcontext on a given channel\n");
    dprintf("\n");

    // MSDEC ROUTINES
    dprintf("MSDEC Routines:\n");
    dprintf(" msdec <engine>            - Private registers <engine>=\"0=vld\",\"1=pdec\",\"2=ppp\" 0<=iSize<=4K\n");
    dprintf(" msdec [-p] <engine> <iSize>\n");
    dprintf("                           - dmem contents <engine>=\"0=vld\",\"1=pdec\",\"2=ppp\" 0<=iSize<=4K\n");
    dprintf("                             -p Dumps the common and app methods in parsed format\n");
    dprintf("                             !lw.msdec 0 dumps VLD  registers\n");
    dprintf("                             !lw.msdec 1 dumps PDEC registers\n");
    dprintf("                             !lw.msdec 2 dumps PPP  registers\n");
    dprintf("                             !lw.msdec 0 20, dumps 20 bytes of DMEM for VLD\n");
    dprintf("                             !lw.msdec 1 20, dumps 20 bytes of DMEM for PDEC\n");
    dprintf("                             !lw.msdec 2 20, dumps 20 bytes of DMEM for PPP\n");
    dprintf(" msdec -f <chId>           - Dumps out the flow control buffer contents in parsed format\n");
    dprintf("                             + where chId is the channel Id for PDEC\n");
    dprintf("\n");

    // VIC ROUTINES
    vicDisplayHelp();
    dprintf("\n");

    // MISC ROUTINES
    dprintf("MISC Routines:\n");
    dprintf(" diag [-grIdx <grIdx>]         - Reports the general status of the GPU\n");
    dprintf("                               - grIdx Used only when SMC is enabled to specify the GR engine ID\n");
    dprintf(" classname <classNum>          - Prints out the class name\n");
#ifdef USERMODE
    dprintf(" chex <hex value>          - Colwert hex value to other formats\n");
    dprintf(" cdec <decimal value>      - Colwert decimal value to other formats\n");
#endif
    dprintf("\n");
    dprintf("\n");


// DPU ROUTINES
    dprintf("DPU Routines:\n");
    dprintf(" dpudmemrd <offset> [length(bytes)] [port] [size(bytes)]\n");
    dprintf("                           - Dump 'length' bytes of the DPU DMEM at the given offset\n");
    dprintf("                             + size sets the width of values that are dumped (1=8-bit,\n");
    dprintf("                               2=16-bit,else=32-bit.)\n");
    dprintf(" dpuudmemwr <offset> <value> [-w <width>(byes)] [-l <length>(units of width)] [-p <port>] [-s <size>]\n");
    dprintf("                           - Write 'value' of width 'width', 'length' times starting at\n");
    dprintf("                             DMEM offset 'offset'.\n");
    dprintf(" dpuimemrd <offset> [length(bytes)] [port]\n");
    dprintf("                           - Dump 'length' bytes of the dpu IMEM at the given offset\n");
    dprintf(" dpuimemwr <offset> <value> [-w <width>(byes)] [length(units of width)]"\
                                         "[-p <port>][-s <size>]\n");
    dprintf("                           - Write 'value' of width 'width', 'length' times starting at\n");
    dprintf("                             IMEM offset 'offset'.\n");
    dprintf(" dpuqueues [queueId]       - Dump out the DPU queues\n");
    dprintf(" dpuimblk [block index]    - Dump status of DPU IMEM code block\n");
    dprintf("                             + dumps all blocks if no block index is specified\n");
    dprintf(" dpuimmap [-s] [start tag] [end tag]\n");
    dprintf("                           - Dumps the IMEM tag to block mapping for a range of tags\n");
    dprintf("                             + automatically tries to determine range if none given\n");
    dprintf("                             + '-s' skip unmapped tags\n");
    dprintf(" dpuimtag <code addr>      - Dumps IMEM code block the code address tag is mapped to\n");
    dprintf(" dpumutex [mutex id]       - Dump the status of one or all DPU mutices\n");
    dprintf(" dputcb [-v] [-c] [-a] [-l] [tcbAddress] [port] [size].\n");
    dprintf("                           - Dump the TCB at address 'tcbAddress'\n");
    dprintf("                             + '-v' : verbose\n");
    dprintf("                             + size sets the width of the stack values that are dumped\n");
    dprintf("                               1=8-bit,2=16-bit,else=32-bit.\n");
    dprintf("                             + '-c' current TCB\n");
    dprintf("                             + '-a' : print all OsTasks' TCBs\n");
    dprintf("                             + '-l' : same as -a\n");
    dprintf("                             + tcbAddress specify particular tcb at tcbAddress to print.\n");
    dprintf("                             + only one of the -a -c or tcbAddress options should present\n");
    dprintf(" dpusym [-l [fileName]] [-u] [-n <addr>] [-i <symbol>]");
    dprintf("                            - Dump the value of a DPU symbol or dump the symbol table\n");
    dprintf("                             + '-l' [filename]   : load symbols from a specific nm-file\n");
    dprintf("                             + '-n' <address>    : resolve an address to a symbol\n");
    dprintf("                             + '-u'              : unload\n");
    dprintf("                             + '-i <symbol>'     : lookup a particular symbol\n");
    dprintf("                             + only one of these options should be used\n");
    dprintf(" dpusched [-h] [-l]        - Dump the dpu RTOS scheduler information\n");
    dprintf("                             + '-h' : print usage\n");
    dprintf("                             + '-l' : list instead of table\n");
    dprintf(" dpuevtq [-h] [-a] [-s <symbol>] [-n <address>]\n");
    dprintf("                           - Dump out the DPU RTOS event queues\n");
    dprintf("                             + '-h' : print usage\n");
    dprintf("                             + '-a' : dump verbose info on all known event queues\n");
    dprintf("                             + '-s' : dump info on a specific queue (identified by symbol name)\n");
    dprintf("                             + '-n' : dump info on a specific queue (identified by queue address)\n");
    dprintf("                             + only one of the -a -s -n options should be used\n");
    dprintf(" dpuqboot <program>        - DPU Quick Boot - boots a DPU app in Simple DPU Binary format.\n");
    dprintf("                             <program> : name of application in Simple DPU Binary format.\n");
    dprintf(" dpusanitytest [options]   - Runs DPU sanity test.\n");
    dprintf("                             + '-t' <test#> : execute command on a single test program.\n");
    dprintf("                             + '-v' <level> : verbose level. 0-3 where 0 - mute (default), 3 - noisy.\n");
    dprintf("                             + '-i'         : prints description of available pmu sanity tests\n");
    dprintf("                             + '-n'         : returns the number of tests available. [testnum] ignored\n");
    dprintf("\n");
    dprintf("\n");



    // PMU ROUTINES
    dprintf("PMU Routines:\n");
    dprintf(" pmudmemrd <offset> [length(bytes)] [port] [size(bytes)]\n");
    dprintf("                           - Dump 'length' bytes of the PMU DMEM at the given offset\n");
    dprintf("                             + size sets the width of values that are dumped (1=8-bit,\n");
    dprintf("                               2=16-bit,else=32-bit.\n");
    dprintf(" pmudmemwr <offset> <value> [length(units of width)] [-p <port>] [-s <size>]\n");
    dprintf("                           - Write 'value' of width 'width', 'length' times starting at\n");
    dprintf("                             DMEM offset 'offset'.\n");
    dprintf(" pmuimemrd <offset> [length(bytes)] [port]\n");
    dprintf("                           - Dump 'length' bytes of the PMU IMEM at the given offset\n");
    dprintf(" pmuimemwr <offset> <value> [length(units of width)] [-p <port>] [-s <size>]\n");
    dprintf("                           - Write 'value' of width 'width', 'length' times starting at\n");
    dprintf("                             IMEM offset 'offset'.\n");
    dprintf(" pmuqueues [queueId]       - Dump out the PMU queues\n");
    dprintf(" pmuevtq [-h] [-a] [-s <symbol>] [-n <address>]\n");
    dprintf("                           - Dump out the PMU RTOS event queues\n");
    dprintf("                             + '-h' : print usage\n");
    dprintf("                             + '-a' : dump info on all known event queues\n");
    dprintf("                             + '-s' : dump info on a specific queue (identified by symbol name)\n");
    dprintf("                             + '-n' : dump info on a specific queue (identified by queue address)\n");
    dprintf(" pmuimblk [block index]    - Dump status of PMUOD IMEM code block\n");
    dprintf("                             + dumps all blocks if no block index is specified\n");
    dprintf(" pmuimtag <code addr>      - Dumps IMEM code block the code address tag is mapped to\n");
    dprintf(" pmuimmap [start tag] [end tag]\n");
    dprintf("                           - Dumps the IMEM tag to block mapping for a range of tags\n");
    dprintf("                             + automatically tries to determine range if none given\n");
    dprintf(" pmumutex [mutex id]       - Dump the status of one or all PMU mutices\n");
    dprintf(" pmutcb <tcbAddress> [port] [size(bytes)]\n");
    dprintf("                           - Dump the TCB at address 'tcbAddress'\n");
    dprintf("                             + size sets the width of the stack values that are dumped\n");
    dprintf("                               1=8-bit,2=16-bit,else=32-bit.\n");
    dprintf(" pmusched [-h] [-l]        - Dump the PMU RTOS scheduler information\n");
    dprintf("                             + '-h' : print usage\n");
    dprintf("                             + '-l' : print information in list form instead of table form\n");
    dprintf(" pmusym <options> <symbol> - Dump the value of a PMU symbol or dump the symbol table\n");
    dprintf("                             + '-l' <nm-file>    : load symbols from a specific nm-file\n");
    dprintf("                             + '-n' [address]    : resolve an address to a symbol\n");
    dprintf(" pmust [options]             - Dump the stack trace for PMU \n");
    dprintf("                             + '-l' <objdump-file>    : load symbols from a specific objdump-file\n");
    dprintf("                             + '-u'                   : unload the objdump file \n");
    dprintf(" pmusanitytest [options]   - Runs PMU sanity test.\n");
    dprintf("                             + '-t' <test#> : execute command on a single test program.\n");
    dprintf("                             + '-v' <level> : verbose level. 0-3 where 0 - mute (default), 3 - noisy.\n");
    dprintf("                             + '-i'         : prints description of available pmu sanity tests\n");
    dprintf("                             + '-n'         : returns the number of tests available. [testnum] ignored\n");
    dprintf(" pmuqboot <program>        - PMU Quick Boot - boots a PMU app in Simple PMU Binary format.\n");
    dprintf("                             <program> : name of application in Simple PMU Binary format.\n");
    dprintf("\n");


    // Falcon (PMU/DPU) ROUTINES
    dprintf("FALCON Routines:\n");
    dprintf(" falctrace <engine> -i <addr> <size> <-vir|-phy> <-vid|-sys>\n");
    dprintf("                           - Initialize the falcon-trace buffer for a particular engine \n");
    dprintf("                             + <engine> = '-pmu', '-dpu', '-msenc'\n");
    dprintf("                             + '-i' <addr> <size> <-vir|-phy> <-vid|-sys> : Initialize the buffer parameters \n");
    dprintf("                               <-vir|phy> : The address space - 'virtual(PMU only)' or 'physical(DPU only)'\n");
    dprintf("                               <-vid|sys> : The aperture - 'video memory' or 'system memory'\n");
    dprintf("                               <addr>     : The address\n");
    dprintf("                               <size>     : The size in bytes\n");
    dprintf(" falctrace <engine> [-n]   - Dump the last 16 entries (default) in the buffer\n");
    dprintf("                             + '-n' : Dump the last 'n' entries\n");
    dprintf(" flcn <engine|command>     - General Falcon debug extension.  Use '!flcn help' to get more information  \n");
    dprintf("\n");

    // RISC-V ROUTINES
    dprintf("RISC-V Routines:\n");
    dprintf("rv <command> [args]        - General RISC-V debug extension.\n");
    dprintf("\n");

    // BSI ROUTINES
    dprintf("BSI Routines:\n");
    dprintf(" bsiramrd <offset> [length(bytes)] \n");


// FECS ROUTINES
    dprintf("FECS FALCON Routines:\n");
    dprintf(" fecsdmemrd <offset> [length(bytes)] [port] [size(bytes)]\n");
    dprintf("                           - Dump 'length' bytes of the FECS DMEM at the given offset\n");
    dprintf("                             + size sets the width of values that are dumped (1=8-bit,\n");
    dprintf("                               2=16-bit,else=32-bit.)\n");
    dprintf(" fecsdmemwr <offset> <value> [-w <width>(byes)] [-l <length>(units of width)] [-p <port>] [-s <size>]\n");
    dprintf("                           - Write 'value' of width 'width', 'length' times starting at\n");
    dprintf("                             DMEM offset 'offset'.\n");
    dprintf(" fecsimemrd <offset> [length(bytes)] [port]\n");
    dprintf("                           - Dump 'length' bytes of the fecs IMEM at the given offset\n");
    dprintf(" fecsimemwr <offset> <value> [-w <width>(byes)] [length(units of width)]"\
                                         "[-p <port>][-s <size>]\n");
    dprintf("                           - Write 'value' of width 'width', 'length' times starting at\n");
    dprintf("                             IMEM offset 'offset'.\n");
    dprintf("                             + only one of the -a -s -n options should be used\n");
    dprintf("\n");
    dprintf("\n");


    // SMBus PostBox Interface
    dprintf("SMBus PostBox Interface Routines:\n");
    dprintf(" smbpbi\n");
    dprintf("\n");

    // Sequencer
    dprintf("Sequencer Routines:\n");
    dprintf(" seq\n");

    // STATE TEST ROUTINES
    dprintf("STATE TEST Routines:\n");
    dprintf(" gpuanalyze [-grIdx <grIdx>] - Comprehensive GPU test (all tests listed below, excluding ptevalidate)\n");
    dprintf("                             -grIdx Used only when SMC is enabled to specify the GR engine ID\n");
    dprintf(" fbstate                   - R/W test of Frame Buffer\n");
    dprintf(" ptevalidate               - Validate PTEs\n");
    dprintf(" pdecheck <chId> <pdeID>   - Check that all PTEs for given <chid, pdeid> are valid\n");
    dprintf(" hoststate                 - Test host state\n");
    dprintf(" grstate [-grIdx <grIdx>]  - Test graphics state\n");
    dprintf("                           -grIdx Used only when SMC is enabled to specify the GR engine ID\n");
    dprintf(" msdecstate                - Test MSDEC {VLD, DEC, PPP} state\n");
    dprintf(" elpgstate                 - Test ELPG state\n");
    dprintf(" cestate                   - Test Copy Engine state\n");
    dprintf(" dispstate                 - Test Display state\n");
#ifndef USERMODE
    dprintf(" vicstate                  - Test VIC state\n");
    dprintf(" msencstate                - Test MSENC state\n");
#endif
#ifndef USERMODE
    dprintf(" hdastate                  - Test HDA state\n");
#endif
    dprintf("\n");

    // PRIV ROUTINES
    dprintf("PRIV Routines:\n");
    dprintf(" privhistory               - Dump the Priv History Buffer (GF119, Kepler)\n");
    dprintf("\n");

    // ZBC COLOR AND DEPTH READ ROUTINES
    dprintf("ZBC COLOR AND DEPTH READ Routines:\n");
    dprintf(" zbc [index]               - Dump ZBC Color and Depth DS/L2 Table data for gievn index [index -> 1 to 15]\n");
    dprintf("                           - Dumps the whole ZBC Color and Depth DS/L2 Table for index == 0\n");
    dprintf("                           - On iGT21A [index] is comptag line address.  "
                                          "Prints color, z, stencil values and reference counts\n");
    dprintf("\n");

    // MSENC ROUTINES
    msencDisplayHelp();
    dprintf("\n");

    // OFA ROUTINES
    ofaDisplayHelp();
    dprintf("\n");

    // HDA ROUTINES
    hdaDisplayHelp();
    dprintf("\n");

    // MC ROUTINES
    dprintf(" pgob                      - Read PG On Boot specific fuse and registers\n");
    dprintf("\n");

    // ELPG ROUTINES
    elpgDisplayHelp();
    dprintf("\n");

    // CE ROUTINES
    ceDisplayHelp();
    dprintf("\n");

    // LWDEC ROUTINES
    lwdecDisplayHelp();
    dprintf("\n");

    // LWJPG ROUTINES
    lwjpgDisplayHelp();
    dprintf("\n");

    // MEMSYS ROUTINES
    acrDisplayHelp();

    // PRIV ROUTINES
    dprintf("Priv Sec Debug license:\n");
    dprintf(" psdl                      - Handle Priv sec debug license (GM20X and later)\n");
    dprintf("\n");

    // FALCON PHYSICAL DMA CHECK ROUTINES
    falcphysDisplayHelp();

    // DEVICE INFO ROUTINES
    dprintf(" deviceinfo                - Displays device info contents \n");

    // LWLINK ROUTINES
    dprintf("\n");
    dprintf(" lwlink [-v.-vv,-dumpuphys] - Display lwlink status\n");
    dprintf("                            -v for verbose mode\n");
    dprintf("                            -vv for pretty verbose mode\n");
    dprintf("                            -dumpuphys for dumping UPHY\n");
    dprintf(" lwlink -progCntrs  [link#] - Programs the lwlink TL counters\n");
    dprintf(" lwlink -resetCntrs [link#] - Reset the lwlink TL counters\n");
    dprintf(" lwlink -readCntrs  [link#] - Read the lwlink TL counters. In order to count again reset the counters first !\n");
    dprintf(" lwlink -dumpAlt            - Dump the registers used for ALT\n"); 

    // HSHUB ROUTINES
    dprintf("\n");
    dprintf(" Usage: lwv hshub <# args> arg1 arg2 ... \n");
    dprintf(" Commands:\n");
    dprintf(" -config          - Dump the current HSHUB sysmem and p2p config\n");
    dprintf(" -idleStatus      - Dump the idle status of each unit of HSHUB\n");
    dprintf(" -logErrors       - Dump the HSHUB error log registers\n");
    dprintf(" -reqTimeoutInfo  - Dump the HSHUB request timeout registers\n");
    dprintf(" -enableLogging   - Set HSHUB logging mode\n");
    dprintf(" -readTimeoutInfo - Dump the HSHUB read timeout registers\n");
    dprintf(" -connCfg         - Dump the HSHUB_CONNECTION_CFG registers for SYS & PEERs\n");
    dprintf(" -muxCfg          - Dump the MUX registers for both the HSHUBs\n");

    // IBMNPU ROUTINES
    dprintf("\n");
    dprintf(" ibmnpu <command> [option]\n");
    dprintf("Commands:\n");
    dprintf(" '-links'              : Displays a list of NPU hardware link ids\n");
    dprintf(" '-devices'            : Displays all IBMNPU PCI device info\n");
    dprintf(" '-ctrl [link#] <proc>': Utilizes the procedure ctrl registers to perform devinit.\n");
    dprintf(" '-dumpdlpl [link#]    : Dumps the DL/PL register set of the selected device\n");
    dprintf(" '-dumpntl [link#]     : Dumps the NTL register set of the selected device.\n");
    dprintf(" '-dumpuphys [link#]   : Dumps the PHY register set of the selected device.\n");
    dprintf("Options:\n");
    dprintf(" '-v'      : Verbose mode - Displays more useful info.\n");


    if (LWWATCHCFG_IS_PLATFORM(UNIX))
    {
        dprintf("\n");
        dprintf("Shell commands:\n");
        dprintf(" Anything to the right of a pipe ('|') will be passed to");
        dprintf(" `/bin/sh -c`, receiving its stdin from lwwatch's stdout.\n");
        dprintf(" Examples:\n");
        dprintf("    lw> dchnmstate | grep \"HEAD_SET_CONTROL\\[\"\n");
        dprintf("    lw> pd LW_PDISP_RG_STATUS(2) | grep LOCKED | grep TRUE\n");
        dprintf("    lw> help | less\n");
        dprintf("    lw> help | cat > /tmp/help.txt\n");
    }

     // HDCP ROUTINES
    dprintf("HDCP: ");
    dprintf("hdcp                       - Dumps HDCP 1.x and 2.x status\n");
    dprintf("\n");

    // LWSR ANALYZE ROUTINES
    dprintf("LWSR ANALYZE Routines:\n");
    dprintf(" lwsrmutex <port> <option> <16 byte key>    - Provides 2 subfunctions\n");
    dprintf("                                                (option = 1) Check SPRN randomness\n");
    dprintf("                                                (option = 2) Compute and unlock LWSR mutex\n");
    dprintf(" lwsrcap <port>                             - Prints the parsed LWSR capability registers\n");
    dprintf(" lwsrinfo <port> <verbose 0~1>              - Prints the parsed LWSR info registers\n");
    dprintf(" lwsrtiming <port>                          - Prints the LWSR GPU-SRC and SRC-Panel timings\n");
    dprintf(" lwsrsetrr <port> <refresh_rate>            - Sets user defined SRC-Panel min refresh rate\n");
    dprintf("\n");

    // SMC ROUTINES
    dprintf("SMC Routines:\n");
    dprintf(" smcpartitioninfo [-swizzid <swizzId>] - Gives partition info of the specified swizzId\n");
    dprintf("                                       - With no swizzId provided, partition info of all swizzIds is printed\n");
    dprintf(" smcengineinfo [-grIdx <grIdx>]        - Gives partition info of the specified grIdx\n");

    // VPR ROUTINES
    vprDisplayHelp();

// INTR ROUTINES
    intrPrintHelp();

    // DFD PLUGINS
    dprintf("DFD Plugins:\n");
    dprintf(" l2ila [-v] [-k <keepfile>] [-o <outfile>] <command> <inputscriptfile>\n");
    dprintf("                           - Plugin to control l2ila (Level 2 cache internal logic analyzer)\n");
    dprintf("                             l2ila is used to assist post silicon debugging by providing the ability to monitor l2 traffic, sample\n");
    dprintf("                             request information or generate global trigger.\n");
    dprintf("                             Confluence page: https://confluence.lwpu.com/display/GPUPSIDBG/L2-iLA+Config+Tool+V2\n");
    dprintf("                             + -v: Verbose\n");
    dprintf("                             + -k <keepfile>: Store read/write commands issued to <keepfile>\n");
    dprintf("                             + -o <outfile>: Store captured result in <outfile>. Default to l2ila.json\n");
    dprintf("                             + command: Available commands are config, arm, disarm, capture, status\n");
    dprintf("                             + inputscriptfile: Script generated by web gui to pass in\n");
    dprintf("\n");
    dprintf(" dfdasm [-h] [-v] [-vl] [-t] [-o <logfile>] <dfdasm> <command>\n");
    dprintf("                           - Plugin to execute DFD Assembly code\n");
    dprintf("                             dfdasm is an abstraction layer between dfd tools and hardware environment\n");
    dprintf("                             More info: https://confluence.lwpu.com/display/GPUPSIDBG/Lwwatch+DFD+Assembly\n");
    dprintf("                             + -h: Print this help message and exit\n");
    dprintf("                             + -v: Verbose\n");
    dprintf("                             + -vl: Verbose log. This will print all log method and command calls to the console\n");
    dprintf("                             + -t: Test mode. Under this mode all PRI traffic will be handled internally as variables and no actual PRI requests will be sent to HW\n");
    dprintf("                             + -o <logfile>: Store log in <outfile>. Default to dfdasm.log\n");
    dprintf("                             + dfdasm: DFD asm file to execute from\n");
    dprintf("                             + command: Command in dfd asm file to execute\n");
    dprintf("\n");
}
