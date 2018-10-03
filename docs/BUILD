# Copyright Vertex.AI.

load("@com_intel_plaidml//bzl:conda.bzl", "conda_binary")
load("@com_intel_plaidml//bzl:docs.bzl", "sphinx")

package(default_visibility = ["//visibility:public"])

conda_binary(
    name = "sphinx",
    srcs = ["@com_intel_plaidml//bzl:sphinx.py"],
    env = "@com_intel_plaidml_conda_sphinx//:env",
    deps = ["@com_intel_plaidbench//:py"],
)

sphinx(
    name = "docs",
    srcs = glob(["**/*"]),
    out = "html",
    conf = "conf.py",
    sphinx = ":sphinx",
    tags = ["docs"],
)
