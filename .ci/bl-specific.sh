#!/bin/bash

export AZURE_TESTING=1

sudo mkdir -v -p /NSLS2/xf18id1/DATA/FXI_log/
sudo chown -Rv $USER: /NSLS2
sudo chown -Rv $USER: /nsls2/data

echo -e "Current directory: $PWD"

cp -v .ci/calib_new.csv /NSLS2/xf18id1/DATA/FXI_log/calib_new.csv

files_to_touch=$(grep '/nsls2/data/' startup/99-umacro.py | cut -d\" -f 2 | grep '\.txt$')

for f in $files_to_touch; do
    mkdir -v -p $(dirname $f)
    touch $f
done
