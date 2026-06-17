import nslsii
import os

###############################################################################
# TODO: remove this block once https://github.com/bluesky/ophyd/pull/959 is
# merged/released.
print(f"Loading {__file__}...")

from datetime import datetime
from ophyd.signal import EpicsSignalBase, EpicsSignal, DEFAULT_CONNECTION_TIMEOUT
from bluesky.callbacks.tiled_writer import TiledWriter
from tiled.client import from_profile, from_uri

try:
    from bluesky_queueserver import is_re_worker_active, parameter_annotation_decorator

except ImportError:
    # Remove the try..except once 'bluesky_queueserver' is included in the collection environment

    def is_re_worker_active():
        return False

    import functools

    def parameter_annotation_decorator(annotation):
        def function_wrap(func):
            if inspect.isgeneratorfunction(func):

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    return (yield from func(*args, **kwargs))

            else:

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)

            return wrapper

        return function_wrap


def print_now():
    return datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")


###############################################################################

from ophyd.signal import EpicsSignalBase

# EpicsSignalBase.set_default_timeout(timeout=10, connection_timeout=10)  # old style
EpicsSignalBase.set_defaults(timeout=10, connection_timeout=10)  # new style
EpicsSignal.set_defaults(timeout=10, connection_timeout=10)  # new style


from datetime import datetime

# Register bluesky IPython magics.
if not is_re_worker_active():
    from bluesky.magics import BlueskyMagics

    get_ipython().register_magics(BlueskyMagics)

from bluesky.preprocessors import stage_decorator, run_decorator

tiled_writing_client = from_profile("nsls2", api_key=os.getenv("TILED_BLUESKY_WRITING_API_KEY_FXI", ""))["fxi"]["raw"]
tiled_writing_client.context.http_client.headers['tiled-qos'] = 'acquisition'

class TiledInserter:
    name = 'fxi'

    def insert(self, name, doc):
        tiled_writing_client.post_document(name, doc)

tiled_inserter = TiledInserter()

if not is_re_worker_active():
    db = tiled_reading_client = from_profile("nsls2", include_data_sources=True)["fxi"]["raw"]
    db.context.http_client.headers['tiled-qos'] = 'acquisition'

nslsii.configure_base(get_ipython().user_ns,
                      tiled_inserter,
                      bec=True,
                      publish_documents_with_kafka=True,
                      redis_url='xf18id1-fxi-redis1.nsls2.bnl.gov',
                      redis_port=6380,
                      redis_ssl=True
                      )


# The following plan stubs should not be imported directly in the global namespace.
#   Otherwise Queue Server will not be able to load the startup files.
del one_1d_step
del one_nd_step
del one_shot


# disable plotting from best effort callback
bec.disable_plots()

from databroker.assets.handlers import AreaDetectorHDF5TimestampHandler
import pandas as pd


EPICS_EPOCH = datetime(1990, 1, 1, 0, 0)


def convert_AD_timestamps(ts):
    return pd.to_datetime(ts, unit="s", origin=EPICS_EPOCH, utc=True).dt.tz_convert(
        "US/Eastern"
    )


# subscribe the zmq plotter

from bluesky.callbacks.zmq import Publisher

publisher = Publisher("xf18id-srv1:5577")
RE.subscribe(publisher)

# nslsii.configure_base(get_ipython().user_ns, 'fxi', bec=False)



## HACK HACK


def rd(obj, *, default_value=0):
    """Reads a single-value non-triggered object

    This is a helper plan to get the scalar value out of a Device
    (such as an EpicsMotor or a single EpicsSignal).

    For devices that have more than one read key the following rules are used:

    - if exactly 1 field is hinted that value is used
    - if no fields are hinted and there is exactly 1 value in the
      reading that value is used
    - if more than one field is hinted an Exception is raised
    - if no fields are hinted and there is more than one key in the reading an
      Exception is raised

    The devices is not triggered and this plan does not create any Events

    Parameters
    ----------
    obj : Device
        The device to be read
    default_value : Any
        The value to return when not running in a "live" RunEngine.
        This come ups when ::

           ret = yield Msg('read', obj)
           assert ret is None

        the plan is passed to `list` or some other iterator that
        repeatedly sends `None` into the plan to advance the
        generator.

    Returns
    -------
    val : Any or None
        The "single" value of the device

    """
    hints = getattr(obj, "hints", {}).get("fields", [])
    if len(hints) > 1:
        msg = (
            f"Your object {obj} ({obj.name}.{getattr(obj, 'dotted_name', '')}) "
            f"has {len(hints)} items hinted ({hints}).  We do not know how to "
            "pick out a single value.  Please adjust the hinting by setting the "
            "kind of the components of this device or by rd ing one of it's components"
        )
        raise ValueError(msg)
    elif len(hints) == 0:
        hint = None
        if hasattr(obj, "read_attrs"):
            if len(obj.read_attrs) != 1:
                msg = (
                    f"Your object {obj} ({obj.name}.{getattr(obj, 'dotted_name', '')}) "
                    f"and has {len(obj.read_attrs)} read attrs.  We do not know how to "
                    "pick out a single value.  Please adjust the hinting/read_attrs by "
                    "setting the kind of the components of this device or by rd ing one "
                    "of its components"
                )

                raise ValueError(msg)
    # len(hints) == 1
    else:
        (hint,) = hints

    ret = yield from read(obj)

    # list-ify mode
    if ret is None:
        return default_value

    if hint is not None:
        return ret[hint]["value"]

    # handle the no hint 1 field case
    try:
        (data,) = ret.values()
    except ValueError as er:
        msg = (
            f"Your object {obj} ({obj.name}.{getattr(obj, 'dotted_name', '')}) "
            f"and has {len(ret)} read values.  We do not know how to pick out a "
            "single value.  Please adjust the hinting/read_attrs by setting the "
            "kind of the components of this device or by rd ing one of its components"
        )

        raise ValueError(msg) from er
    else:
        return data["value"]

# monkey batch bluesky.plans_stubs to fix bug.
bps.rd = rd


def get_proposal_type(proposal_id=None):
    import httpx
    nslsii_api_client = httpx.Client(base_url="https://api.nsls2.bnl.gov")
    if (proposal_id is None):
        proposal_id = RE.md["proposal"]["proposal_id"]

    proposal_response = nslsii_api_client.get(f"/v1/proposal/{RE.md['proposal']['proposal_id']}")
    proposal_response.raise_for_status()
    proposal = proposal_response.json()["proposal"]

    return proposal["type"]
