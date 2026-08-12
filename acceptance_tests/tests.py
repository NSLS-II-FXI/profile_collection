# Acceptance tests
# Running the tests from IPython
# %run -i ~/.ipython/profile_collection/acceptance_tests/tests.py


def test_xanes_scan2():
    """
    Xanes scan test.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting xanes scan test")
    (uid,) = RE(xanes_scan2([8.37, 8.36, 8.35], simu=True))
    print("Fly scan complete")
    print("Reading scan from tiled...")
    # db[uid].table(fill=True)
    db[uid].baseline.data
    print("Exporting scan ...")
    export_scan(db[uid].start["scan_id"])
    print("Test is complete")


def test_fly_scan3():
    """
    Fly scan test 3.
    If db.table() and export scan complete without errors then it was successful.
    """
    print("Starting fly scan test 3")
    (uid,) = RE(
        fly_scan(
            exposure_time=0.05,
            start_angle=0,
            relative_rot_angle=180,
            period=0.05,
            out_x=None,
            out_y=-10,
            out_z=None,
            out_r=0,
            rs=6,
            relative_move_flag=True,
            rot_first_flag=1,
            filters=[],
            add_bkg_filt_only=False,
            rot_back_velo=30,
            binning=None,
            move_to_ini_pos=True,
            simu=True,
            take_bkg_img=True,
            take_dark_img=True,
            close_shutter_finish=True,
            note="None",
        )
    )
    print("Fly scan complete")
    print("Reading scan from tiled...")
    # db[uid].table(fill=True)
    db[uid].baseline.data
    print("Exporting scan ...")
    export_scan(db[uid].start["scan_id"])
    print("Test is complete")


def test_tomo_zfly_scan():
    """
    Tomo zfly scan test
    If db.table() and export scan complete without errors then it was successful.
    """
    print("Starting tomo zfly scan")
    (uid,) = RE(
        tomo_zfly(
            scn_mode=0,
            exp_t=0.05,
            acq_p=0.052,
            ang_s=0,
            ang_e=180,
            vel=3,
            acc_t=1,
            num_swing=1,
            out_pos=[None, -200, None, None],
            rel_out_flag=True,
            rot_back_velo=30,
            bin_fac=1,
            roi={"min_x": 576, "size_x": 2048, "min_y": 576, "size_y": 2048},
            note="test",
            simu=True,
        )
    )
    print("Tomo zfly scan complete")
    print("Reading scan from tiled...")
    # db[uid].table(fill=True)
    db[uid].baseline.data
    print("Exporting scan ...")
    export_scan(db[uid].start["scan_id"])
    print("Test is complete")


def test_multi_edge_xanes_zebra_scan():
    """
    Multi edge xanes zebra scan test
    If db.table() and export scan complete without errors then it was successful.
    """
    print("Starting multi edge xanes zebra scan")
    uid1, uid2 = RE(
        multi_edge_xanes_zebra(
            edge_list={"Ni": [8.339, 8.338]},
            scan_type="3D",
            flts={"Ni": []},
            exp_t={"Ni": 0.03},
            acq_p={"Ni": 0.032},
            ang_s=0,
            ang_e=180,
            vel=6,
            acc_t=1,
            in_pos_list=[[None, None, None, None]],
            bin_fac=1,
            out_pos=[None, None, None, None],
            rel_out_flag=1,
            note="test",
            simu=True,
        )
    )
    print("Multi edge xanes zebra scan complete")
    print("Reading scan from tiled...")
    # db[uid].table(fill=True)
    db[uid1].baseline.data
    print("Exporting scan ...")
    export_scan(db[uid1].start["scan_id"])
    print("Test is complete")


# ===========================================================================================
#                              test_xanes_scan2
test_xanes_scan2()

# ===========================================================================================
#                              test_fly_scan3
test_fly_scan3()

# ===========================================================================================
#                              test_tomo_zfly_scan
test_tomo_zfly_scan()

# ===========================================================================================
#                              test_multi_edge_xanes_zebra_scan
test_multi_edge_xanes_zebra_scan()
