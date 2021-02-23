#!/bin/bash

export AZURE_TESTING=1

sudo mkdir -v -p /NSLS2/xf18id1/DATA/FXI_log/
sudo chown -Rv $USER: /NSLS2

echo -e "Current directory: $PWD"

cp -v .ci/calib_new.csv /NSLS2/xf18id1/DATA/FXI_log/calib_new.csv
