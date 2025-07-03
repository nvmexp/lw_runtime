licenses(["notice"])

package(default_visibility = ["//visibility:private"])

exports_files(["LICENSE"])

cc_library(
    name = "all_pairs",
    hdrs = ["all_pairs.h"],
    deps = [
        ":glog",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/utility",
    ],
)

cc_test(
    name = "all_pairs_test",
    size = "small",
    srcs = [
        "all_pairs_test.cc",
    ],
    deps = [
        ":all_pairs",
        ":glog",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "lwda_util",
    srcs = [
        "lwda_util.cc",
        "lwda_util_lw",
        "status.h",
    ],
    hdrs = [
        "lwda_util.h",
    ],
    deps = [
        ":glog",
        "@com_google_absl//absl/types:optional",
        "@local_config_lwda//:lwda_headers",
        "@local_config_lwda//:lwda_runtime",
        "@local_config_lwda//:lwpti",
        "@local_config_lwda//:lwpti_headers",
        "@local_config_lwda//:lwrand_static",
    ],
)

cc_library(
    name = "kernel_timer",
    srcs = ["kernel_timer.cc"],
    hdrs = ["kernel_timer.h"],
    deps = [
        ":lwda_util",
        ":glog",
        "@local_config_lwda//:lwda_headers",
        "@local_config_lwda//:lwpti_headers",
    ],
)

proto_library(
    name = "lwdnn_proto",
    srcs = ["lwdnn.proto"],
)

cc_proto_library(
    name = "lwdnn_cc_proto",
    deps = [":lwdnn_proto"],
)

cc_library(
    name = "lwdnn_util",
    srcs = [
        "lwdnn_util.cc",
    ],
    hdrs = ["lwdnn_util.h"],
    deps = [
        ":lwda_util",
        ":lwdnn_cc_proto",
        ":glog",
        "@com_google_absl//absl/types:variant",
        "@local_config_lwda//:lwda_headers",
        "@local_config_lwda//:lwda_runtime",
        "@local_config_lwda//:lwdnn",
    ],
)

# Wraps glog and provides a work-around header.
cc_library(
    name = "glog",
    hdrs = ["ostream_nullptr.h"],
    deps = ["@com_google_glog//:glog"],
)

# See README.md for bazel command lines to build and run benchmarks.
cc_binary(
    name = "lwdnn_benchmark",
    srcs = [
        "lwdnn_benchmark.cc",
        "load_textproto.h",
        "status.h",
    ],
    data = glob(["*.textproto"]),
    deps = [
        ":lwda_util",
        ":lwdnn_cc_proto",
        ":lwdnn_util",
        ":glog",
        ":kernel_timer",
        "@com_google_absl//absl/types:optional",
        "@com_google_benchmark//:benchmark",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

# See README.md for bazel command lines to build and run tests.
cc_test(
    name = "lwdnn_test",
    size = "enormous",
    srcs = [
        "lwdnn_colw_test.cc",
        "lwdnn_test.cc",
        "lwdnn_test.h",
        "load_textproto.h",
        "status.h",
        "test_util.cc",
        "test_util.h",
    ],
    data = glob(["*.textproto"]),
    linkstatic = 1,
    deps = [
        ":all_pairs",
        ":lwda_util",
        ":lwdnn_cc_proto",
        ":lwdnn_util",
        ":glog",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf_lite",
        "@local_config_lwda//:lwda_headers",
        "@local_config_lwda//:lwdnn",
    ],
)

genrule(
    name = "lwda_util_lw",
    srcs = ["lwda_util.lw"],
    outs = ["lwda_util.o"],
    cmd = "$(location @local_config_lwda//:lwcc) --compile $(SRCS) --output-file $(OUTS)",
    tools = ["@local_config_lwda//:lwcc"],
)
