========================================================================
    MAKEFILE PROJECT : CoprocStatistics Project Overview
========================================================================

AppWizard has created this CoprocStatistics project for you.  

This file contains a summary of what you will find in each of the files that
make up your CoprocStatistics project.


CoprocStatistics.vcxproj
    This is the main project file for VC++ projects generated using an Application Wizard. 
    It contains information about the version of Visual C++ that generated the file, and 
    information about the platforms, configurations, and project features selected with the
    Application Wizard.

CoprocStatistics.vcxproj.filters
    This is the filters file for VC++ projects generated using an Application Wizard. 
    It contains information about the association between the files in your project 
    and the filters. This association is used in the IDE to show grouping of files with
    similar extensions under a specific node (for e.g. ".cpp" files are associated with the
    "Source Files" filter).

This project allows you to build/clean/rebuild from within Visual Studio by calling the commands you have input 
in the wizard. The build command can be nmake or any other tool you use.

This project does not contain any files, so there are none displayed in Solution Explorer.

/////////////////////////////////////////////////////////////////////////////

If the user runs into this error:
gmake: *** No rule to make target `E:/code/PRIYANKJ-DT/sw/tools/win32/msvc110/vc', needed by `build'.  Stop.
gmake: Leaving directory `E:/code/PRIYANKJ-DT/sw/dev/gpu_drv/bugfix_main/apps/CoprocStatistics/CoprocStatistics'

>> You don't have //sw/tools/win32/msvc110/ checked out on your local machine (since the make file uses this compiler). once you check it out, the code should be able to compile and build.
Hope this helps.
/Priyank
***************************************************************************************************************

On the off chance the reader is trying to compile this project, and gets an error something like
CoprocStatistics.cpp
cl : Command line error D8027 : cannot execute 'C:\sb4\sw\tools\win32\msvc110\vc\bin\c2.dll'
gmake: *** [_out/win7_x86_debug/CoprocStatistics.obj] Error 1

This is a known issue, and Nilesh suggested a WAR to Dean, which I (Matt Radecki) will pass along.
And kudos to you for reading the readme.txt file, well played!

Hi Dean,
The issue youre facing is known VC11 issue. This normally oclwrs if you have other version of VC11 installed on your system (beta, developer preview). The KMDTEST builds uses MSVC11 RTM (Visual Studio 2012 Professional) as the compiler. The fix is to install VS2012 professional removing all its previous versions.

If you dont want reinstall, then there is a war 
For your case, rename C:\sb4\sw\tools\win32\msvc110\vc\bin folder to something else & copy the bin folder C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin to C:\sb4\sw\tools\win32\msvc110\vc\ location. This should resolve the mismatched compilers issue.

Hi Dmitry,
One more thing, KMDTEST wont build on WindowsXP machine as there is no MSVC11 support for XP. You have to use Vista+ for that.

Hope this helps.

Regards,
Nilesh
