import os

if not os.environ.get("AZURE_TESTING") and not is_re_worker_active():
    new_user()
show_global_para()
run_pdf()
read_calib_file_new()  # read file from: /nsls2/data/fxi-new/legacy/log/calib_new.csv
# check_latest_scan_id(init_guess=60000, search_size=100)
