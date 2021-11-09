from bluesky.bundlers import RunBundler as _OrigBundler
import bluesky

class PatchedRunBundler(_OrigBundler):

    async def configure(self, msg):
        """Configure an object

        Expected message object is ::

            Msg('configure', object, *args, **kwargs)

        which results in this call ::

            object.configure(*args, **kwargs)
        """
        obj = msg.obj
        # Invalidate any event descriptors that include this object.
        # New event descriptors, with this new configuration, will
        # be created for any future event documents.
        for name in list(self._descriptors):
            obj_set, _ = self._descriptors[name]
            if obj in obj_set:
                del self._descriptors[name]
    

        if obj in self._describe_cache:
            del self._describe_cache[obj]
            del self._config_desc_cache[obj]
            del self._config_values_cache[obj]
            del self._config_ts_cache[obj]

bluesky.run_engine.RunBundler = PatchedRunBundler
bluesky.bundlers.RunBundler = PatchedRunBundler