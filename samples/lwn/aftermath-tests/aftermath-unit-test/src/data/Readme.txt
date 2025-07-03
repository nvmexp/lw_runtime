How to generate new NXGCD test dump data files
-----------------------------------------------

Preparations
------------
1) p4 sync //sw/mobile/odin/devtools/Aftermath/GameWorks/trunk
2) Open CMD
3) cd sw/mobile/odin/devtools/Aftermath/GameWorks/trunk/Samples/applications/build/xpj
4) create_all_sample_projects.cmd
5) In ViualStudio open sw/mobile/odin/devtools/Aftermath/GameWorks/trunk/Samples/applications/build/vs2017All/AllLWNSamples.sln
6) Build the CascadedShadowMapping project (platform=NX64)
7) In DevMenu (on the SDEV) got to the "Debug" menu and make sure "GPU Crash Automatic Dump" is enabled
8) Make sure you have a copy of bin2c available, e.g. from the LwdaToolkit (//sw/devtools/Agora/Prebuilt/Windows/LwdaToolkit/xxx/bin/bin2c.exe)

Example: Generate fragment_shader_hang_nxgcd_v1.h
---------------------------------------------------------------------------
1) Run "CascadedShadowMapping.nspd tdr FragmentShaderHang" on the SDEV
2) Wait for the application to crash
3) Open CMD
4) cd %USERPROFILE%/Dolwments/Nintendo/NXDMP
5) bin2c -c -n fragment_shader_hang_nxgcd_v1 -st <your device's id>_<most recent time stamp>.nxgcd > fragment_shader_hang_nxgcd_v1.h
6) Delete the empty last line in fragment_shader_hang_nxgcd_v1.h to make gerrit git patching happy :(
7) copy fragment_shader_hang_nxgcd_v1.h gpu\drv\apps\lwn\aftermath-test\tests\data

Example: Generate compute_shader_pagefault_nxgcd_v1.h
---------------------------------------------------------------------------
1) Run "CascadedShadowMapping.nspd tdr OutOfBoundsBufferAccess" on the SDEV
2) Wait for the application to crash
3) Open CMD
4) cd %USERPROFILE%/Dolwments/Nintendo/NXDMP
5) bin2c -c -n compute_shader_pagefault_nxgcd_v1 -st <your device's id>_<most recent time stamp>.nxgcd > compute_shader_pagefault_nxgcd_v1.h
6) Delete the empty last line in compute_shader_pagefault_nxgcd_v1.h to make gerrit git patching happy :(
7) copy compute_shader_pagefault_nxgcd_v1.h gpu\drv\apps\lwn\aftermath-test\tests\data
