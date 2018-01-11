import nslsii

# Register bluesky IPython magics.
from bluesky.magics import BlueskyMagics
get_ipython().register_magics(BlueskyMagics)

from bluesky.preprocessors import stage_decorator, run_decorator


nslsii.configure_base(get_ipython().user_ns, 'fxi')

