#!/bin/bash

TEX_FILE=$1
OUTPUT_DIR=$2

if [[ -z "$TEX_FILE" ]] ; then
  echo "latextopdf.sh: usage: <TEX_FILE> [OUTPUT_DIR]" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]] ; then
  OUTPUT_DIR=$(dirname "$TEX_FILE")/output
fi

mkdir -p "${OUTPUT_DIR}"

exec pdflatex -bibtex -deps -interaction=nonstopmode -output-directory="${OUTPUT_DIR}" "${TEX_FILE}"
