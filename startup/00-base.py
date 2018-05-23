import nslsii
from datetime import datetime

# Register bluesky IPython magics.
from bluesky.magics import BlueskyMagics
get_ipython().register_magics(BlueskyMagics)

from bluesky.preprocessors import stage_decorator, run_decorator

nslsii.configure_base(get_ipython().user_ns, 'fxi', mpl=False)
from databroker.assets.handlers import AreaDetectorHDF5TimestampHandler
import pandas as pd


EPICS_EPOCH = datetime(1990, 1, 1, 0, 0)


def convert_AD_timestamps(ts):
    return pd.to_datetime(ts, unit='s', origin=EPICS_EPOCH, utc=True).dt.tz_convert('US/Eastern')


#nslsii.configure_base(get_ipython().user_ns, 'fxi', bec=False)

'''
def ts_msg_hook(msg):
    t = '{:%H:%M:%S.%f}'.format(datetime.now())
    msg_fmt = '{: <17s} -> {!s: <15s} args: {}, kwargs: {}'.format(
        msg.command,
        msg.obj.name if hasattr(msg.obj, 'name') else msg.obj,
        msg.args,
        msg.kwargs)
    print('{} {}'.format(t, msg_fmt))

RE.msg_hook = ts_msg_hook
'''
