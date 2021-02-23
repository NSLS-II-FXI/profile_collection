#!/bin/bash

export AZURE_TESTING=1

sudo mkdir -v -p /NSLS2/xf18id1/DATA/FXI_log/
sudo chown -rv $USER: /NSLS2
touch /NSLS2/xf18id1/DATA/FXI_log/calib_new.csv

