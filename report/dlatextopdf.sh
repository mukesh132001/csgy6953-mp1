#!/bin/bash

PROG=dlatextopdf
TEX_FILE=$1

if [[ -z "$TEX_FILE" ]] ; then
   TEX_FILE=mp1-report.tex
fi

MTIME_REF=""

for _ in $(seq 3600) ; do
  MTIME_CURRENT=$(stat --format "%Y" "${TEX_FILE}")
  if [[ "x${MTIME_REF}" != "x${MTIME_CURRENT}" ]] ; then
    MTIME_REF=${MTIME_CURRENT}
    echo "${PROG}: regenerating pdf from ${TEX_FILE}"
    ./latextopdf.sh "${TEX_FILE}" >& /dev/null
    CODE=$?
    if [[ $CODE -ne 0 ]] ; then
      echo "${PROG}: ${CODE} is latexmk return code"
    fi
  fi
  sleep 1
done
