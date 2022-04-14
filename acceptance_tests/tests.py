# Acceptance tests


def test_fly_scan2():
    """
    Fly scan test 2.
    If db.table() and export scan complete without errors than it was successful.
    """
    
    print("Starting fly scan test 2")
    uid, = RE(fly_scan(relative_rot_angle=20, out_x=10, out_y=20, out_z=30, simu=True))
    print("Fly scan complete")
    print("Reading scan from databroker")
    db[uid].table()
    print("Exporting scan")
    export_scan(db[uid].start['scan_id'])


def test_xanes_scan2():
    """
    Xanes scan test.
    If db.table() and export scan complete without errors than it was successful.
    """
    
    print("Starting xanes scan test")
    uid, = RE(xanes_scan2([8.35, 8.36, 8.37], simu=True))
    print("Fly scan complete")
    print("Reading scan from databroker")
    db[uid].table()
    print("Exporting scan")
    export_scan(db[uid].start['scan_id'])
