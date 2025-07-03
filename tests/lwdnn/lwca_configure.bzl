"""Build rule generator for locally installed LWCA toolkit and lwDNN SDK."""

def _get_elw_var(repository_ctx, name, default):
    if name in repository_ctx.os.elwiron:
        return repository_ctx.os.elwiron[name]
    return default

def _impl(repository_ctx):
    lwda_path = _get_elw_var(repository_ctx, "LWDA_PATH", "/usr/local/lwca")
    lwdnn_path = _get_elw_var(repository_ctx, "LWDNN_PATH", lwda_path)

    print("Using LWCA from %s\n" % lwda_path)
    print("Using lwDNN from %s\n" % lwdnn_path)

    repository_ctx.symlink(lwda_path, "lwca")
    repository_ctx.symlink(lwdnn_path, "lwdnn")

    repository_ctx.file("lwcc.sh", """
#! /bin/bash
repo_path=%s
compiler=${CC:+"--compiler-bindir=$CC"}
$repo_path/lwca/bin/lwcc $compiler --compiler-options=-fPIC --include-path=$repo_path $*
""" % repository_ctx.path("."))

    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "lwcc",
    srcs = ["lwcc.sh"],
)

# The *_headers cc_library rules below aren't cc_inc_library rules because
# dependent targets would only see the first one.

cc_library(
    name = "lwda_headers",
    hdrs = glob(
        include = ["lwca/include/**/*.h*"],
        exclude = ["lwca/include/lwdnn.h"]
    ),
    # Allows including LWCA headers with angle brackets.
    includes = ["lwca/include"],
)

cc_library(
    name = "lwca",
    srcs = ["lwca/lib64/stubs/liblwda.so"],
    linkopts = ["-ldl"],
)

cc_library(
    name = "lwda_runtime",
    srcs = ["lwca/lib64/liblwdart_static.a"],
    deps = [":lwca"],
    linkopts = ["-lrt"],
)

cc_library(
    name = "lwrand_static",
    srcs = [
        "lwca/lib64/liblwrand_static.a",
    ],
    deps = [
        ":lwlibos",
    ],
)

cc_library(
    name = "lwpti_headers",
    hdrs = glob(["lwca/extras/LWPTI/include/**/*.h"]),
    # Allows including LWPTI headers with angle brackets.
    includes = ["lwca/extras/LWPTI/include"],
)

cc_library(
    name = "lwpti",
    srcs = glob(["lwca/extras/LWPTI/lib64/liblwpti.so*"]),
)

cc_library(
    name = "lwdnn",
    srcs = [
        "lwdnn/lib64/liblwdnn_static.a",
        "lwca/lib64/liblwblas_static.a",
    ] + glob(["lwca/lib64/liblwblasLt_static.a"]),
    hdrs = ["lwdnn/include/lwdnn.h"],
    deps = [
        ":lwca",
        ":lwda_headers",
        ":lwlibos",
    ],
)

cc_library(
    name = "lwlibos",
    srcs = ["lwca/lib64/liblwlibos.a"],
)

cc_library(
    name = "lwda_util",
    deps = [":lwda_util_compile"],
)
""")

lwda_configure = repository_rule(
    implementation = _impl,
    elwiron = ["LWDA_PATH", "LWDNN_PATH"],
)
