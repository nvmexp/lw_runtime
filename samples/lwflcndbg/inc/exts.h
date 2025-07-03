/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2000-2013 by LWPU Corporation.  All rights reserved.  All
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
#include "lwwatch.h"

#if defined(WIN32)
#include "lwwatch.h"   // defines DECLARE_API(NAME) for win32
#ifdef __cplusplus
#define DECLARE_API_DEFN(NAME) \
    extern "C" VOID NAME \
    (IN PDEBUG_CLIENT Client, IN OPTIONAL PCSTR args)
#else
#define DECLARE_API_DEFN(NAME) \
    VOID NAME \
    (IN PDEBUG_CLIENT Client, IN OPTIONAL PCSTR args)
#endif

#elif defined(CLIENT_SIDE_RESMAN) || defined(vxworks) || LWWATCHCFG_FEATURE_ENABLED(UNIX_USERMODE)
#define DECLARE_API_DEFN(NAME)   void NAME(char *args)

#endif

#if defined(USERMODE)
#ifndef DECLARE_API
#define DECLARE_API(NAME)   extern void NAME()
#endif

#elif defined(EFI_APP)
#define DECLARE_API(NAME)   void NAME()

#elif LWWATCHCFG_IS_PLATFORM(OSX)
#if defined(LW_MAC_KEXT)
#define DECLARE_API(NAME)   void klww_##NAME(char *args)
#else
#define DECLARE_API(NAME)   void lw_##NAME(char *args)
#endif

#ifdef __cplusplus
#define DECLARE_API_CPP(NAME)   extern "C" DECLARE_API(NAME)
#else
#define DECLARE_API_CPP(NAME)   DECLARE_API(NAME)
#endif

#elif defined (WIN32)
#undef DECLARE_API
#define DECLARE_API(NAME) \
    DECLARE_API_DEFN(NAME##_IMPL);\
    DECLARE_API_DEFN(NAME) \
    {\
        SaveStateBeforeExtensionExec();\
        NAME##_IMPL(Client, args);\
        RestoreStateAfterExtensionExec();\
    }\
    DECLARE_API_DEFN(NAME##_IMPL)

#elif defined(CLIENT_SIDE_RESMAN) || defined(vxworks) || LWWATCHCFG_FEATURE_ENABLED(UNIX_USERMODE)
#define DECLARE_API(NAME) \
    DECLARE_API_DEFN(NAME##_IMPL);\
    DECLARE_API_DEFN(NAME) \
    {\
        SaveStateBeforeExtensionExec();\
        NAME##_IMPL(args);\
        RestoreStateAfterExtensionExec();\
    }\
    DECLARE_API_DEFN(NAME##_IMPL)
#else
#error need to implement DECLARE_API(NAME) for this platform!
#endif

#if !LWWATCHCFG_IS_PLATFORM(OSX) && !defined(USERMODE) && !defined(EFI_APP)
#ifdef __cplusplus
extern "C" void SaveStateBeforeExtensionExec();
extern "C" void RestoreStateAfterExtensionExec();
#endif
#endif

#if LWWATCHCFG_IS_PLATFORM(OSX) || defined(USERMODE) || defined(EFI_APP)
#if !defined(LWWATCH_API)
#define EXTS_LWWATCH_API
#define LWWATCH_API(NAME)       DECLARE_API(NAME);

#if defined(DECLARE_API_CPP)
#define LWWATCH_API_CPP(NAME)   DECLARE_API_CPP(NAME);
#else
#define LWWATCH_API_CPP(NAME)   DECLARE_API(NAME);
#endif
#endif

#else //LWWATCHCFG_IS_PLATFORM(OSX) || defined(USERMODE) || defined(EFI_APP)
#if !defined(LWWATCH_API)
#define EXTS_LWWATCH_API
#define LWWATCH_API(NAME)       DECLARE_API_DEFN(NAME);

#if defined(DECLARE_API_DEFN_CPP)
#define LWWATCH_API_CPP(NAME)   DECLARE_API_DEFN_CPP(NAME);
#else
#define LWWATCH_API_CPP(NAME)   DECLARE_API_DEFN(NAME);
#endif
#endif

#endif //LWWATCHCFG_IS_PLATFORM(OSX) || defined(USERMODE) || defined(EFI_APP)

LWWATCH_API( init )

LWWATCH_API( multigpu )

#if !defined(USERMODE) && !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_IS_PLATFORM(OSX) && !defined(EFI_APP)
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

#if !defined(EFI_APP)
LWWATCH_API( fbrd )
LWWATCH_API( fbwr )
#endif

#if !defined(USERMODE) && !defined(EFI_APP)
LWWATCH_API( gvrd )
LWWATCH_API( gvwr )
LWWATCH_API( gvfill )
LWWATCH_API( gvcp )
LWWATCH_API( gco2bl )
LWWATCH_API( gbl2co )
LWWATCH_API( gvtextureheader )
LWWATCH_API( gvdiss )
LWWATCH_API( fbfill )
LWWATCH_API( fbcp )
#endif

LWWATCH_API( gvdisp )

LWWATCH_API( agpinfo )
LWWATCH_API( clocks )
LWWATCH_API( cntrfreq )
LWWATCH_API( classname )

#if !defined(EFI_APP)
LWWATCH_API( gpde )
LWWATCH_API( gpte )
LWWATCH_API( gvpte )
LWWATCH_API( gvtop )
LWWATCH_API( gptov )

#if !defined (USERMODE)

#if !LWWATCHCFG_FEATURE_ENABLED(UNIX_USERMODE)
LWWATCH_API( ptov )
#endif
#if !LWWATCHCFG_FEATURE_ENABLED(UNIX_USERMODE) || LWWATCHCFG_FEATURE_ENABLED(UNIX_HWSNOOP)
LWWATCH_API( vtop )
#endif

LWWATCH_API( gtlbvsvirtual )
LWWATCH_API( fbvsgtlb )
#endif
#endif

//cheetah
#ifdef _ARM_
LWWATCH_API( clockregs)
LWWATCH_API( plls)
LWWATCH_API( pllregs)
LWWATCH_API( dccmdinfo)
LWWATCH_API(  dccominfo)
LWWATCH_API(dcdispinfo)
LWWATCH_API( dsiinfo)
LWWATCH_API(winainfo )
LWWATCH_API( winbinfo)
LWWATCH_API( wincinfo)
LWWATCH_API( tvdacinfo)
LWWATCH_API(pinmuxinfo)
LWWATCH_API( dispclocks )
LWWATCH_API(dispallinfo)
LWWATCH_API(intrctlrinfo)
LWWATCH_API(gpioinfo  )
LWWATCH_API(gpioallinfo)
LWWATCH_API( powergate)
LWWATCH_API(hdmiregs)
LWWATCH_API(mcinfo)
LWWATCH_API(mpeinfo)
LWWATCH_API(smmuinfo)
LWWATCH_API(cluster)
#endif //_ARM_

#if defined(USERMODE) && defined(LW_WINDOWS)
LWWATCH_API(g_elw)
LWWATCH_API(s_elw)
LWWATCH_API(cachestatus)
LWWATCH_API(ecc)
LWWATCH_API(crd)
#endif

LWWATCH_API( getcr )
LWWATCH_API( setcr )
LWWATCH_API( crtcfifo )
LWWATCH_API( pbinfo )
LWWATCH_API( c1info )
LWWATCH_API( runlist )
LWWATCH_API( fifoinfo )
LWWATCH_API( pbdma )
LWWATCH_API( channelram )

#if !defined(USERMODE) && !defined(EFI_APP)
LWWATCH_API( pb )
LWWATCH_API( pbch )
LWWATCH_API( c1 )

#if !LWWATCHCFG_IS_PLATFORM(OSX)
LWWATCH_API( i2c )
#endif
#endif

LWWATCH_API( dispinfo )
LWWATCH_API( vga )
LWWATCH_API( vgat )
LWWATCH_API( palette )
LWWATCH_API( gr )
LWWATCH_API( grinfo )
LWWATCH_API( fp )
LWWATCH_API( hwseq )
LWWATCH_API( rdtv )
LWWATCH_API( wrtv )
LWWATCH_API( tv )
LWWATCH_API( dcb )
LWWATCH_API( diag )
LWWATCH_API( hash )
LWWATCH_API( shash )
LWWATCH_API( sctxdma )
LWWATCH_API( inst )
LWWATCH_API( fifoctx )
LWWATCH_API( c1ctx )
LWWATCH_API( launchcheck )
LWWATCH_API( perfmon )
LWWATCH_API( subch )
LWWATCH_API( grctx )
LWWATCH_API( tiling )
LWWATCH_API( zlwll )
LWWATCH_API( grstatus )
LWWATCH_API( ssclass )
LWWATCH_API( limiterror )
LWWATCH_API( surface )
LWWATCH_API( scompaction )
LWWATCH_API( stutter )
LWWATCH_API( zbc )
LWWATCH_API( l2ilwalidate )
LWWATCH_API( fbmonitor )
LWWATCH_API( l2state )
LWWATCH_API( ismemreq )
LWWATCH_API( msa )
LWWATCH_API( msi )

#if !defined(USERMODE) && !LWWATCHCFG_FEATURE_ENABLED(UNIX_USERMODE)
LWWATCH_API( heap )
#endif

LWWATCH_API( vpinfo )
LWWATCH_API( vpctx )
LWWATCH_API( meinfo )
LWWATCH_API( mectx )
LWWATCH_API( pcieinfo )
LWWATCH_API( pcie )
LWWATCH_API( pcie3evtlogdmp )
LWWATCH_API( slivb )
LWWATCH_API( vpfifo )
LWWATCH_API( lbinfo )

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_IS_PLATFORM(OSX) && !defined(CLIENT_SIDE_RESMAN)
LWWATCH_API( br04init )
LWWATCH_API( br04topology )
LWWATCH_API( br04dump )
LWWATCH_API( br04port )
#endif

LWWATCH_API( save )
LWWATCH_API( restore )

#if !LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_IS_PLATFORM(OSX) && !defined(USERMODE) && \
!defined(vxworks) && !defined(EFI_APP) && !defined(CLIENT_SIDE_RESMAN)
LWWATCH_API ( stack )
#endif

#if defined(USERMODE) && !defined(MINIRM) && !LWWATCHCFG_IS_PLATFORM(OSX) && !LWWATCHCFG_IS_PLATFORM(UNIX) && !defined(WIN32)
LWWATCH_API( tests )
#endif

#if LWWATCHCFG_FEATURE_ENABLED(UNIX_USERMODE) || (defined(USERMODE) && LWWATCHCFG_IS_PLATFORM(UNIX))
LWWATCH_API( bspinfo )
#endif

#if LWWATCHCFG_IS_PLATFORM(OSX) && !defined(USERMODE)
LWWATCH_API( debug_level )
LWWATCH_API( iotop )
#if !defined(LW_MAC_KEXT)
LWWATCH_API( has_writephys )
#endif
#endif

LWWATCH_API( dchnstate )
LWWATCH_API( dchnnum )
LWWATCH_API( dchnname )
LWWATCH_API( dchnmstate )
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
LWWATCH_API( danalyzestateexcept )
LWWATCH_API( drdhdmidecoder )
LWWATCH_API( dwrhdmidecoder )

LWWATCH_API( checkhwinit )

LWWATCH_API( cpudisp )
LWWATCH_API( sigdump )
LWWATCH_API( smuinfo )
LWWATCH_API( checkprodvals )

LWWATCH_API( dpauxrd )
LWWATCH_API( dpauxwr )
LWWATCH_API( dpinfo )

LWWATCH_API( msdec )
LWWATCH_API( vic )
LWWATCH_API( msenc )
LWWATCH_API( lwdec )
LWWATCH_API( sec )

LWWATCH_API( ddesc )
LWWATCH_API( dhdorconn )
LWWATCH_API( ce )

#if !LWWATCHCFG_IS_PLATFORM(OSX)
LWWATCH_API( bspctx )

LWWATCH_API( cipher )
LWWATCH_API( compute )
LWWATCH_API( insttochid )

LWWATCH_API( pmusanitytest )
LWWATCH_API( pmusched )
LWWATCH_API( zlwllram )
#endif

#if !defined(MINIRM) && !defined(EFI_APP) && !defined(USERMODE)
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
LWWATCH_API( fbinfo )
LWWATCH_API( ptevalidate )
LWWATCH_API( pdecheck )
LWWATCH_API( hoststate )
LWWATCH_API( grstate )
LWWATCH_API( msdecstate )
LWWATCH_API( elpgstate )
LWWATCH_API( cestate )
LWWATCH_API( dispstate )
LWWATCH_API( gpuanalyze )

LWWATCH_API( falctrace )
LWWATCH_API( flcn )

LWWATCH_API( pgob )
LWWATCH_API( privhistory )
LWWATCH_API( elpg )
LWWATCH_API( dsli )
LWWATCH_API( help )
LWWATCH_API( hdcp )

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
LWWATCH_API( impdump )
LWWATCH_API( pep )
LWWATCH_API( therm )
LWWATCH_API( lwlog )
LWWATCH_API( lwsym )
#endif

#if defined(EXTS_LWWATCH_API)
/* Limit the scope of the LWWATCH_API definitions to this file if we defined them here */
#undef LWWATCH_API
#undef LWWATCH_API_CPP
#undef EXTS_LWWATCH_API
#endif
