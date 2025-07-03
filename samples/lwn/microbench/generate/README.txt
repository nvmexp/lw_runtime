mkshaders.py is used to generate header files from GLSL code.
These header files contain a const char * string of the GLSL
code which can be referenced directly in microbenchmarks.

-----
USAGE
-----
usage: mkshaders.py [-h] [--manifest MANIFEST] [-o OUTPUT]
                    [filename [filename ...]]

positional arguments:
  filename

optional arguments:
  -h, --help            show this help message and exit
  --manifest MANIFEST   Path to manifest xml
  -o OUTPUT, --output OUTPUT
                        Output .h file

--------
EXAMPLE:
--------
from $TOP/gpu/drv/apps/lwn/microbench/:

1) python generate/mkshaders.py cases/shaderperf/shaders/dce_lmemaccess.glsl -o cases/shaderperf/shaders/g_dce_lmemaccess.h

2) Then in your microbenchmark include the header file:
#include "shaders/g_dce_lmemaccess.h"

3) Access the (const char *) shader string named "shader_dce_lmemaccess"
