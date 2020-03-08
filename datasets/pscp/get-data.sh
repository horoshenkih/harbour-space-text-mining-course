#!/bin/bash
set -exo pipefail

# download data
for year in $(seq 1991 2017); do
  wget -N https://raw.githubusercontent.com/paperscape/paperscape-data/master/pscp-${year}.csv
done

# construct the reversed graph: paper -> list of papers that reference to it
cat pscp-*.csv \
  | grep -vE '^#' \
  | cut -d ';' -f 1,5 \
  | grep -vE ';$' \
  | perl -lne '($a, $b) = split(";"); print join("\n", map {$_.";".$a} split(",", $b))' \
  | sort \
  > pscp-reversed-graph.txt

# construct the mapping: arXiv ID -> first category
cat pscp-*.csv \
  | grep -vE '^#' \
  | cut -d ';' -f1,2 \
  | cut -d, -f1 \
  > pscp-categories.txt
