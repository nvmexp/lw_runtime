glslc=/d/p4/sw/devtools/Agora/Rel/LWNGD_Lima/Prebuilt/Windows/x86/BinaryLwnGlslc/17.18.1.15/Release/BinaryLwnGlslc.exe
glslc_dll=/d/nxsdk/810_rc/NintendoSDK/Tools/Graphics/LwnTools/LwnGlslc32.dll
$glslc -glslc $glslc_dll -o null_vs -cpp nullVs -vs null.vs
$glslc -glslc $glslc_dll -o grey_fs -cpp greyFs -fs grey.fs
$glslc -glslc $glslc_dll -o red_fs -cpp redFs -fs red.fs
$glslc -glslc $glslc_dll -o zero_cs -cpp zeroCs -cs zero.cs
$glslc -glslc $glslc_dll -o one_cs -cpp oneCs -cs one.cs
