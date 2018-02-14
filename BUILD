# Copyright Vertex.AI.

package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

exports_files([
    "requirements.txt",
    "venv.txt",
])

filegroup(
    name = "golden",
    srcs = glob(["golden/**"]),
)

filegroup(
    name = "networks",
    srcs = glob(["networks/**"]),
)

filegroup(
    name = "cifar16",
    srcs = ["cifar16.npy"],
)

pkg_tar(
    name = "pkg",
    srcs = glob(["**/*"]),
    package_dir = "plaidbench",
    strip_prefix = ".",
)

py_library(
    name = "plaidbench",
    srcs = ["plaidbench.py"],
    data = glob([
        "cifar16.npy",
        "golden/**",
        "networks/**",
    ]),
)

py_binary(
    name = "bin",
    srcs = ["plaidbench.py"],
    data = glob([
        "cifar16.npy",
        "golden/**",
        "networks/**",
    ]),
    main = "plaidbench.py",
    deps = ["@vertexai_plaidml//plaidml/keras"],
)
