This is a subset of the shaderc and glslang projects in order to support GLSL -> SPIR-V compilation cross-platform. Standard shaderc instructions are at:

https://github.com/google/shaderc

Note that the glslang version is from the KHR_vulkan_glsl branch:

git clone git@gitlab.khronos.org:GLSL/glslang.git -b KHR_vulkan_glsl
git checkout 38616fa321e6beea9816ee2fe9817f6d79cd1766

Modifications:
1) using lwogtest Android.mk, makefile.lwmk etc instead of the prepackaged build systems
2) before copying in glslang, do local build using its build system in order to generate glslang_tab.cpp[.h]
3) remove test and tools directories to save space
4) rename shaderc/third_party/glslang/glslang/MachineIndependent/limits.cpp to mi_limits.cpp
   - due to lwmake name conflict with tests/bitmap/limits.c
5) Apply WinBuildFixes.diff (temporary; being upstreamed)
6) Modified a preprocessor conditional to avoid a build failure on HOS.
   - patch -p6 < 0002-Disable-fancy-pthreads-on-HOS.patch