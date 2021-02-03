#!/bin/bash
jq --indent 1 \
    '
    (.cells[] | select(has("outputs")) | .outputs) = []
    | (.cells[] | select(has("execution_count")) | .execution_count) = null
    | .metadata = {}
    | .cells[].metadata = {}
    ' $1 > /tmp/$(basename $1)
cat /tmp/$(basename $1) > $1

# https://github.com/holoviz/holoviews/pull/2507
# for f in **/**/*.ipynb; do ./strip_kernel.sh $f; done
