#!/bin/bash

node --perf-basic-prof transpiled.js &
perf record -F 99 -p `pgrep -n node` -g -- sleep 30
perf script > out.nodestacks01
pushd FlameGraph
./stackcollapse-perf.pl < ../out.nodestacks01 | ./flamegraph.pl > ../out.nodestacks01.svg
popd
