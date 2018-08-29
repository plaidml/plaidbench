# Copyright Vertex.AI.

load("@vertexai_plaidml//bzl:conda.bzl", "conda_binary")
load("@vertexai_plaidml//bzl:docs.bzl", "sphinx")

package(default_visibility = ["//visibility:public"])

conda_binary(
    name = "sphinx",
    srcs = ["@vertexai_plaidml//bzl:sphinx.py"],
    env = "@vertexai_plaidml_conda_sphinx//:env",
    deps = ["@vertexai_plaidbench//:py"],
)

sphinx(
    name = "docs",
    srcs = glob(["**/*"]),
    out = "html",
    conf = "conf.py",
    sphinx = ":sphinx",
    tags = ["docs"],
)
