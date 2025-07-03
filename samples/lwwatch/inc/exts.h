/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2000-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// Each os may have a slightly different entrypoint prototype, so define all
// that first
// DECLARE_API(NAME) is used for the actual functions in exts.c
// LWWATCH_API(NAME) is used below to declare the list of entrypoints.
// in the default case, LWWATCH_API() == DECLARE_API();  (note the ';') but in
// UserMode lwwatch, we need to include this list to create a function table in
// that case, LWWATCH_API() is already defined differently, to create this table
//
//  *** WARNING: THIS FILE IS INCLUDED IN A DATA TABLE  ***
//
#include "lwwatch.h"

#if defined(WIN32)
#include "lwwatch.h"   // defines DECLARE_API(NAME) for win32
#ifdef __cplusplus
#define DECLARE_API_DEFN(NAME) \
    extern "C" void NAME \
    (IN PDEBUG_CLIENT Client, IN OPTIONAL PCSTR args)
#define DECLARE_API_IMPL(NAME) \
    extern "C" void NAME##_IMPL \
    (IN PDEBUG_CLIENT Client, IN OPTIONAL PSTR args)
#else
#define DECLARE_API_DEFN(NAME) \
    void NAME \
    (IN PDEBUG_CLIENT Client, IN OPTIONAL PCSTR args)
#define DECLARE_API_IMPL(NAME) \
    void NAME##_IMPL \
    (IN PDEBUG_CLIENT Client, IN OPTIONAL PSTR args)
#endif

#elif defined(CLIENT_SIDE_RESMAN) || LWWATCHCFG_IS_PLATFORM(UNIX)
#define DECLARE_API_DEFN(NAME)   void NAME(char *args)

#endif

#if defined(USERMODE)
#ifndef DECLARE_API
#define DECLARE_API(NAME)   extern void NAME()
#endif

#elif defined (WIN32)
#undef DECLARE_API
#define DECLARE_API(NAME) \
    DECLARE_API_IMPL(NAME);\
    DECLARE_API_DEFN(NAME) \
    {\
        char localArgs[256];\
        strcpy(localArgs, args);\
        osCmdProlog();\
        NAME##_IMPL(Client, localArgs);\
        osCmdEpilog();\
    }\
    DECLARE_API_IMPL(NAME)

#elif defined(CLIENT_SIDE_RESMAN) || LWWATCHCFG_IS_PLATFORM(UNIX)
#define DECLARE_API(NAME) \
    DECLARE_API_DEFN(NAME##_IMPL);\
    DECLARE_API_DEFN(NAME) \
    {\
        NAME##_IMPL(args);\
    }\
    DECLARE_API_DEFN(NAME##_IMPL)
#else
#error need to implement DECLARE_API(NAME) for this platform!
#endif

#if defined(USERMODE)
#if !defined(LWWATCH_API)
#define EXTS_LWWATCH_API
#define LWWATCH_API(NAME)       DECLARE_API(NAME);

#if defined(DECLARE_API_CPP)
#define LWWATCH_API_CPP(NAME)   DECLARE_API_CPP(NAME);
#else
#define LWWATCH_API_CPP(NAME)   DECLARE_API(NAME);
#endif
#endif

#else // defined(USERMODE)
#if !defined(LWWATCH_API)
#define EXTS_LWWATCH_API
#define LWWATCH_API(NAME)       DECLARE_API_DEFN(NAME);

#if defined(DECLARE_API_DEFN_CPP)
#define LWWATCH_API_CPP(NAME)   DECLARE_API_DEFN_CPP(NAME);
#else
#define LWWATCH_API_CPP(NAME)   DECLARE_API_DEFN(NAME);
#endif
#endif

#endif // defined(USERMODE)

LWWATCH_API( init )

LWWATCH_API( multigpu )

#if !defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
LWWATCH_API( modsinit )
#endif

LWWATCH_API( verbose )

#if defined(USERMODE)
LWWATCH_API( chex )
LWWATCH_API( cdec )
#endif

LWWATCH_API( rb )
LWWATCH_API( wb )
LWWATCH_API( rw )
LWWATCH_API( ww )
LWWATCH_API( rd )
LWWATCH_API( wr )
LWWATCH_API( tmdsrd )
LWWATCH_API( tmdswr )

#if defined(LWDEBUG_SUPPORTED)
LWWATCH_API( dump )
LWWATCH_API( prbdec )

LWWATCH_API( dumpinit )
LWWATCH_API( dumpprint )
LWWATCH_API( dumpfeedback )
#endif

LWWATCH_API( fbrd )
LWWATCH_API( fbwr )

#if !defined(USERMODE)
LWWATCH_API( gvrd )
LWWATCH_API( gvwr )
LWWATCH_API( gvfill )
LWWATCH_API( gvcp )
LWWATCH_API( gco2bl )
LWWATCH_API( gbl2co )
LWWATCH_API( gvdiss )
LWWATCH_API( fbfill )
LWWATCH_API( fbcp )
#endif

LWWATCH_API( gvdisp )

LWWATCH_API( clocks )
LWWATCH_API( cntrfreq )
LWWATCH_API( clklutread )
LWWATCH_API( clkread )
LWWATCH_API( classname )

LWWATCH_API( gpde )
LWWATCH_API( gpte )
LWWATCH_API( gvpte )
LWWATCH_API( gvtop )
LWWATCH_API( gptov )


#if !defined (USERMODE)

#if !LWWATCHCFG_IS_PLATFORM(UNIX)
LWWATCH_API( ptov )
#endif
#if !LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(UNIX_HWSNOOP)
LWWATCH_API( vtop )
#endif

#endif

//cheetah
#ifdef _ARM_
LWWATCH_API( clockregs)
LWWATCH_API( plls)
LWWATCH_API( pllregs)
LWWATCH_API( dsiinfo)
LWWATCH_API( powergate)
LWWATCH_API(mcinfo)
LWWATCH_API(cluster)
#endif //_ARM_

#if defined(USERMODE) && defined(LW_WINDOWS)
LWWATCH_API(g_elw)
LWWATCH_API(s_elw)
LWWATCH_API(ecc)
#endif

LWWATCH_API( getcr )
LWWATCH_API( setcr )
LWWATCH_API( pbinfo )
LWWATCH_API( runlist )
LWWATCH_API( fifoinfo )
LWWATCH_API( pbdma )
LWWATCH_API( channelram )

#if !defined(USERMODE)
LWWATCH_API( pb )
LWWATCH_API( pbch )

LWWATCH_API( i2c )
#endif

LWWATCH_API( gr )
LWWATCH_API( grinfo )
LWWATCH_API( dcb )
LWWATCH_API( diag )
LWWATCH_API( fifoctx )
LWWATCH_API( veidinfo )
LWWATCH_API( launchcheck )
LWWATCH_API( perfmon )
LWWATCH_API( subch )
LWWATCH_API( grctx )
LWWATCH_API( tiling )
LWWATCH_API( zlwll )
LWWATCH_API( grstatus )
LWWATCH_API( grwarppc )
LWWATCH_API( ssclass )
LWWATCH_API( limiterror )
LWWATCH_API( surface )
LWWATCH_API( zbc )
LWWATCH_API( l2ilwalidate )
LWWATCH_API( fbmonitor )
LWWATCH_API( l2state )
LWWATCH_API( ismemreq )
LWWATCH_API( msa )
LWWATCH_API( msi )

#if !defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX)
LWWATCH_API( heap )
LWWATCH_API( pma )
#endif

LWWATCH_API( pcieinfo )
LWWATCH_API( pcie )
LWWATCH_API( pcie3evtlogdmp )
LWWATCH_API( slivb )

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) && !defined(CLIENT_SIDE_RESMAN)
LWWATCH_API( br04init )
LWWATCH_API( br04topology )
LWWATCH_API( br04dump )
LWWATCH_API( br04port )
#endif

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) && !defined(USERMODE) && \
!defined(CLIENT_SIDE_RESMAN)
LWWATCH_API ( stack )
#endif

#if defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) && !defined(WIN32)
LWWATCH_API( tests )
#endif

LWWATCH_API( dchnexcept )
LWWATCH_API( dchnstate )
LWWATCH_API( dchnnum )
LWWATCH_API( dchnname )
LWWATCH_API( dchnmstate )
LWWATCH_API( dchlwal )
LWWATCH_API( dgetdbgmode )
LWWATCH_API( dsetdbgmode )
LWWATCH_API( dinjectmethod )
LWWATCH_API( dreadasy )
LWWATCH_API( dreadarm )
LWWATCH_API( ddumppb )
LWWATCH_API( ddispowner )
LWWATCH_API( dorstate )
LWWATCH_API( dsetorstate )
LWWATCH_API( danalyzeblank )
LWWATCH_API( danalyzehang )
LWWATCH_API( dintr )
LWWATCH_API( dtim )
LWWATCH_API( dchlwars )
LWWATCH_API( drdhdmidecoder )
LWWATCH_API( dwrhdmidecoder )
LWWATCH_API( dsorpadlinkconn )
LWWATCH_API( ddsc )
LWWATCH_API( dlowpower )

LWWATCH_API( cpudisp )
LWWATCH_API( sigdump )
LWWATCH_API( checkprodvals )

LWWATCH_API( dpauxrd )
LWWATCH_API( dpauxwr )
LWWATCH_API( dpinfo )

LWWATCH_API( msdec )
LWWATCH_API( vic )
LWWATCH_API( msenc )
LWWATCH_API( ofa )
LWWATCH_API( lwdec )
LWWATCH_API( lwjpg )
LWWATCH_API( sec )

LWWATCH_API( ddesc )
LWWATCH_API( dhdorconn )
LWWATCH_API( ce )

LWWATCH_API( insttochid )

LWWATCH_API( pmusanitytest )
LWWATCH_API( pmusched )
LWWATCH_API( zlwllram )

#if !defined(USERMODE)
LWWATCH_API( pe )
LWWATCH_API( pd )
#endif

LWWATCH_API( pmu       )
LWWATCH_API( pmuimemrd )
LWWATCH_API( bsiramrd  )
LWWATCH_API( pmudmemrd )
LWWATCH_API( pmudmemwr )
LWWATCH_API( pmuqueues )
LWWATCH_API( pmuimblk  )
LWWATCH_API( pmuimtag  )
LWWATCH_API( pmuimmap  )
LWWATCH_API( pmumutex  )
LWWATCH_API( pmutcb    )
LWWATCH_API( pmusym    )
LWWATCH_API( pmust     )
LWWATCH_API( pmuevtq   )
LWWATCH_API( pmuimemwr )
LWWATCH_API( pmuqboot  )

LWWATCH_API( dpudmemrd )
LWWATCH_API( dpudmemwr )
LWWATCH_API( dpuimemrd )
LWWATCH_API( dpuimemwr )
LWWATCH_API( dpuqueues )
LWWATCH_API( dpuimblk )
LWWATCH_API( dpuimtag )
LWWATCH_API( dpuimmap )
LWWATCH_API( dpusym )
LWWATCH_API( dputcb )
LWWATCH_API( dpusched )
LWWATCH_API( dpuevtq )

LWWATCH_API( fecsdmemrd )
LWWATCH_API( fecsdmemwr )
LWWATCH_API( fecsimemrd )
LWWATCH_API( fecsimemwr )

LWWATCH_API( smbpbi )

LWWATCH_API( seq )

LWWATCH_API( fbstate )
LWWATCH_API( ptevalidate )
LWWATCH_API( pdecheck )
LWWATCH_API( hoststate )
LWWATCH_API( grstate )
LWWATCH_API( msdecstate )
LWWATCH_API( elpgstate )
LWWATCH_API( lpwrstate )
LWWATCH_API( cestate )
LWWATCH_API( dispstate )
LWWATCH_API( gpuanalyze )
LWWATCH_API( lpwrfsmstate )

LWWATCH_API( falctrace )
LWWATCH_API( flcn )
LWWATCH_API( rv )
LWWATCH_API( rvgdb )

LWWATCH_API( pgob )
LWWATCH_API( privhistory )
LWWATCH_API( elpg )
LWWATCH_API( dsli )
LWWATCH_API( help )
LWWATCH_API( hdcp )
LWWATCH_API( lwlink )
LWWATCH_API( ibmnpu )
LWWATCH_API( hshub )

#if defined(WIN32) && !defined(USERMODE)
LWWATCH_API( dumpclientdb )
LWWATCH_API( odbdump )
LWWATCH_API( objgpumgr )
LWWATCH_API( objdevice )
LWWATCH_API( sliconfig )
LWWATCH_API( objgpu )
LWWATCH_API( devicemappings )
LWWATCH_API( clientdb )
LWWATCH_API( objheap )
LWWATCH_API( memblock )
LWWATCH_API( membank )
LWWATCH_API( texinfo )
LWWATCH_API( objfb )
LWWATCH_API( fbramsettings )
LWWATCH_API( infolist )
LWWATCH_API( pdbdump )
LWWATCH_API( lwlog )
LWWATCH_API( lwsym )
#endif

LWWATCH_API( acr )
LWWATCH_API( vpr )
LWWATCH_API( psdl )
LWWATCH_API( falcphysdmacheck )
LWWATCH_API( deviceinfo )

/*LWSR Analyze entrypoints*/
LWWATCH_API( lwsrinfo )
LWWATCH_API( lwsrcap )
LWWATCH_API( lwsrtiming )
LWWATCH_API( lwsrmutex )
LWWATCH_API( lwsrsetrr )

/*Partition info*/
LWWATCH_API(smcpartitioninfo)
LWWATCH_API(smcengineinfo)

/*Interrupt info*/
LWWATCH_API( intr )

/*DFD Tools*/
LWWATCH_API( l2ila )
LWWATCH_API( dfdasm )

#if defined(EXTS_LWWATCH_API)
/* Limit the scope of the LWWATCH_API definitions to this file if we defined them here */
#undef LWWATCH_API
#undef LWWATCH_API_CPP
#undef EXTS_LWWATCH_API
#endif
