# Copyright Vertex.AI.

package(default_visibility = ["//visibility:public"])

genrule(
    name = "docs",
    local = 1,
    srcs = ["conf.py", "requirements.txt"] +
        glob(["**/*.rst"], exclude = ["venv/**"]),
    outs = ["docs.zip"],
    cmd = """
        CWD=`pwd`
        cd $$(dirname $(location conf.py))
        virtualenv --clear venv
        venv/bin/python -m pip install -r requirements.txt
        venv/bin/python -m sphinx -a -E -b html . _build/html
        cd _build/html
        zip -q -r $$CWD/$@ ./
    """,
)
