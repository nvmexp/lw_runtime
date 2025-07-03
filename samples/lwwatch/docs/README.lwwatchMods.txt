-------------------------------------
Testing instructions for lwwatchMods: 
------------------------------------- 

set _NT_DEBUGGER_EXTENSION_PATH=%_NT_DEBUGGER_EXTENSION_PATH%;<Full path of
directory containing lw50.dll Something like
E:\src1\sw\main\tools\resman\lwwatch\objchk\i386>

The debugger engine looks at a list of directories by default. The dll can be
copied to any of those locations if you don't want to set the
_NT_DEBUGGER_EXTENSION_PATH environment variable. "C:\Program Files\Debugging
Tools for Windows\winext" is one these locations.

cd to your HW tree's mods directory. Something like E:\src1\hw\lwdiag\mods

Issue the command from a dos shell (one line):

prompt> "c:\Program Files\Debugging Tools for Windows\windbg.exe" mods -bios lw50n.rom -chip lw50_debug_fmodel.dll "mdiag.js" " -tpc_mask" "0x0001" "-serial" "-noFileIo" "-e" "trace_3d -i e:\src1\arch\traces\lw50_tesla\ogtest\simple_quad\test.hdr -o fixthis -A8R8G8B8"

or whichever regression you want to run.

Some other useful commandlines and aliases (for cygwin users) are in a section below.

When windbg loads up, set any breakpoints you want, e.g. 

bp mods!C_Gpu_Initialize+0x20 (end of the C_Gpu_Initialize function)
(Important note: The breakpoint mods!C_Gpu_Initialize+0x20 is the earliest in
the simulation, the extension seems to work. From the name of the function
- C_Gpu_Initialize, it looks like the GPU/fmodel objects finish initializing 
themselves.)
    OR
any other resman function that is exelwted after this.
(e.g. bp librm!lwHalFifoSetUpChannelDma_LW50)

(See docs/windbg.help.txt for helpful info)

but do not load the extension yet.

Run to those breakpoints. At that point, you can test lwwatchMods as follows: 

>!lw.modsinit 

Some examples:

>!lw.rd <register offset from bar0>
>!lw.fifoctx 1

Run 
>!lw.help for a list of commands


-------------------------------------
Important note regarding lwwatchMods:
-------------------------------------
You can also try running these commands from other parts of the code as well but
make sure that the windbg is not stopped in simulator, i.e.  the call stack does
not contain fmodel functions. I have successfully tried calling these functions
from various resman Hal functions.  

Make sure that all the basic GPU init routines have been exelwted before you
can load and use the extension. For example, if you are calling !lw.rd before
C_Gpu_Initialize+0x20, make sure that MODs/fmodels are able to provide those
values. As mentioned above,  mods!C_Gpu_Initialize+0x20 looks like such a point.

Here's a note from //sw/pvt/main_lw5x/diag/mods/docs/sim.txt: 
 "WARNING: our simulators are not reentrant; never call these functions if your
 call stack is lwrrently inside the simulator."

which basically translates to  - Do not call extension routines that interact
with the simulator when stopped in a simulator routine. It is probably safe to
call !lw.classname etc. at any point since it does not call a MODs/simulator
function.


--------------------------------------------------
Useful aliases and other lwwatchMods commandlines:
--------------------------------------------------
If you use cygwin to run lwwatchMods, use following commands (you may have to
change the path) to set the paths. Put them in your ~/.bashrc file

# _NT_DEBUGGER_EXTENSION_PATH: used by debugger engine to locate lw.dll
export _NT_DEBUGGER_EXTENSION_PATH=`cygpath -w /cygdrive/e/src1/sw/main/tools/resman/lwwatch/objchk/i386`

TEST_HRD_PATH=`cygpath -w /cygdrive/e/src1/arch/traces/lw50_tesla/ogtest/simple_quad/test.hdr`

alias 2='cd /cygdrive/e/src1/hw/lwdiag/mods'

alias w1='(pushd .; 2 ; "/cygdrive/c/Program Files/Debugging Tools for Windows/windbg.exe" -c "bp RmFindDevices; g;bp mods!Trace3DTest::RealSetup+0xb6d;g;bm *Hal*LW50;g" mods -bios lw50n.rom -chip lw50_debug_fmodel.dll mdiag.js -A8R8G8B8  -e trace_3d -i $TEST_HRD_PATH;popd)&'

The interesting option above is the -c option which is a list of commands which
are exelwted by windbg at the very start. You may want to modify it as suited to
your needs. You may want to run to the start of, say,
lwHalFifoSetUpChannelDma_LW50 to start running the extension commands.


----------------
Troubleshooting:
----------------
1) If you are not able to read registers from MODs, grep for "WINDBG VERSION
CAUTION" in the codebase and read and follow those comments.

2) Do not set a radix other than the default (16). If you think, you have a
different radix than 16, use the following windbg command to change it back to
16: 
n (Set Number Base) - The n command sets the default number base (radix) to the
specified value, or displays the current number base.


--------------------------------------------------
Cautions and notice to maintainers of lwwatchMods:
--------------------------------------------------
1) Microsoft presentations (see them in docs/ directory) says that we should use
64 bit addresses when communicating addresses to windbg. This is not a problem
lwrrently because we are not reading symbols from MODS/fmodel user-mode
program. Our way of reading registers from MODs is dolwmented in
docs/lwwatchMods-windbg.txt

2) lwwatch2.0 uses dbgeng.h while lwwatch1.0 didn't. Lwrrently, dbgeng.h that
is checked in at inc/dbgeng.h has about 11 lines commented out since it fails
to compile when compiled with the DDK in the perforce tree. Find the comment by
grepping for LWWATCH in inc/dbgeng.h. This is because of a missing macro
definition(DBG_COMMAND_EXCEPTION) required by dbgeng.h which is not being
provided by the DDK in the p4 tree. The comment in dbgeng.h is not required if
you use the DDK version 3790. In future, we may want to install the latest
version of DDK to the perforce tree. BTW, The DDK 3790 version did not have
dbgeng.h shipped with it. It comes with windbg SDK.

3) The extension gvtop is written with 64-bit code. It needs a routine like
FB_RD32_64(LwU64 reg) to return dwords read from 64-bit addresses in the FB.
FB_RD32_64 lwrrently works with only 32bit addresses since MODs does not yet
provide a way to read memory from 64-bit addresses. To complete that
implementation, Grep for "64BIT CAUTION" in the codebase and follow from there. 

General Note regarding 64-bit safety of the codebase: We do not have a FB read
routine that accepts 64bit addresses. Resman experts have assured us that we do
not need such a routine for a long time into the future.

4) fmodel is not correctly reporting the result of REG_RD08(). So, The MODs
function GpuRegRd08() is not correctly working. When fmodel developers fix that
bug, undefine the macro FMODEL_REG08_BUG from sources file and all associated
code in codebase.


------------
Error Codes:
------------
1] You get a message like:
"Filename: funcName1: funcName2 returned error.  Register reading/writing *MAY*
be incorrect. Error code 1. See error codes in docs/README.lwwatchMods.txt"

Reason 1: This condition can happen if you use old version of windbg. You should
upgrade your windbg to the one mentioned in Requirements section of README.txt. 

Reason 2: There have been some cases when you can get this message even if you
are using the latest windbg. These are usually triggered if register write
extensions are used.

Checking the log at c:\lwwatch.txt might indicate the problem:
 - You might see 
   "Text: (bf8.398): Break instruction exception - code 80000003 (first chance)"
   This indicates that this exception oclwred. Entering gh (go exception
   handled) / g (Go) on the windbg command-line might help.
 - You might see 
   "Thread already has call in progress error in '.call GpuRegRd32(reg number)"
   Entering g(Go) on the windbg command-line might help.

Reason 3: If you call
.call someModsFunction(arg1, arg2)
then, complete the call by exelwting the windbg command g after it. If you fail
to do this, then subsequent .calls will fail. If subsequent .calls are initiated
by our lw.dll extension commands, then they will fail and you will see error
message above.

Last resort: If none of these seem to be oclwring/solving the problem, then
restart windbg and reload the extension.
