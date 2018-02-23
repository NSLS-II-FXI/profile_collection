import nslsii
import datetime

# Register bluesky IPython magics.
from bluesky.magics import BlueskyMagics
get_ipython().register_magics(BlueskyMagics)

from bluesky.preprocessors import stage_decorator, run_decorator

nslsii.configure_base(get_ipython().user_ns, 'fxi')


#nslsii.configure_base(get_ipython().user_ns, 'fxi', bec=False)

'''
def ts_msg_hook(msg):
    t = '{:%H:%M:%S.%f}'.format(datetime.datetime.now())
    msg_fmt = '{: <17s} -> {!s: <15s} args: {}, kwargs: {}'.format(
        msg.command,
        msg.obj.name if hasattr(msg.obj, 'name') else msg.obj,
        msg.args,
        msg.kwargs)
    print('{} {}'.format(t, msg_fmt))

RE.msg_hook = ts_msg_hook
'''
