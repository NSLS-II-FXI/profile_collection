print(f"Loading {__file__}...")

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


class TiledRefAssetResolver:
    """
    Live resolver for flat/dark external assets in a Bluesky + Tiled workflow.

    This does not use databroker registry APIs.

    It listens directly to the RunEngine document stream and collects:
      - start      -> run uid
      - descriptor -> stream names, e.g. "flat", "dark"
      - event      -> datum_id stored in event["data"][image_key]
      - datum      -> maps datum_id to resource uid
      - resource   -> maps resource uid to root/resource_path/spec

    The results are grouped by RunStart UID, which is the same UID you use
    to access the run from Tiled.
    """

    def __init__(
        self,
        *,
        tiled_client=None,
        image_key=None,
        streams=("flat", "dark"),
        verbose=True,
    ):
        self.client = tiled_client
        self.image_key = image_key
        self.streams = set(streams)
        self.verbose = verbose

        self.current_run_uid = None

        self.descriptor_to_run = {}
        self.descriptor_to_stream = {}

        self.resources = {}
        self.datums = {}

        self.pending = {}

        # Main result:
        #   self.resolved_by_run[run_uid]["flat"]
        #   self.resolved_by_run[run_uid]["dark"]
        self.resolved_by_run = defaultdict(lambda: defaultdict(list))

    def __call__(self, name, doc):
        if name == "start":
            self.current_run_uid = doc["uid"]
            if self.verbose:
                print(f"[start] run_uid = {self.current_run_uid}")

        elif name == "descriptor":
            descriptor_uid = doc["uid"]
            run_uid = doc["run_start"]
            stream = doc["name"]

            self.descriptor_to_run[descriptor_uid] = run_uid
            self.descriptor_to_stream[descriptor_uid] = stream

            if self.verbose and stream in self.streams:
                print(
                    f"[descriptor] stream={stream!r}, "
                    f"descriptor={descriptor_uid}, run={run_uid}"
                )

        elif name == "event":
            self._handle_event(doc)

        elif name == "resource":
            self.resources[doc["uid"]] = doc
            self._try_resolve_all()

        elif name == "datum":
            self.datums[doc["datum_id"]] = doc
            self._try_resolve_all()

        elif name == "event_page":
            # Some callbacks receive event_page instead of individual event docs.
            for event_doc in self._unpack_event_page(doc):
                self._handle_event(event_doc)

        elif name == "datum_page":
            # Some callbacks receive datum_page instead of individual datum docs.
            for datum_doc in self._unpack_datum_page(doc):
                self.datums[datum_doc["datum_id"]] = datum_doc
            self._try_resolve_all()

    def _handle_event(self, doc):
        descriptor_uid = doc["descriptor"]
        stream = self.descriptor_to_stream.get(descriptor_uid)

        if stream not in self.streams:
            return

        run_uid = self.descriptor_to_run.get(descriptor_uid)

        for key, value in doc["data"].items():
            if self.image_key is not None and key != self.image_key:
                continue

            # For external assets, the Event data value is normally a datum_id.
            # This is usually a string. Scalar readings should be ignored.
            if isinstance(value, str):
                self.pending[value] = {
                    "run_uid": run_uid,
                    "stream": stream,
                    "event_uid": doc["uid"],
                    "seq_num": doc["seq_num"],
                    "data_key": key,
                    "time": doc["time"],
                }

                if self.verbose:
                    print(
                        f"[{stream}] found datum_id: "
                        f"run={run_uid}, key={key}, datum_id={value}"
                    )

        self._try_resolve_all()

    def _try_resolve_all(self):
        for datum_id, info in list(self.pending.items()):
            datum = self.datums.get(datum_id)
            if datum is None:
                continue

            resource_uid = datum["resource"]
            resource = self.resources.get(resource_uid)
            if resource is None:
                continue

            root = resource.get("root", "")
            resource_path = resource.get("resource_path", "")
            spec = resource.get("spec", "")
            datum_kwargs = datum.get("datum_kwargs", {})

            # This is the physical resource reference, not always the final
            # frame-specific path. For HDF5, datum_kwargs may include frame index
            # or point number; the handler/spec determines exact interpretation.
            file_reference = str(Path(root) / resource_path)

            record = {
                "run_uid": info["run_uid"],
                "stream": info["stream"],
                "event_uid": info["event_uid"],
                "seq_num": info["seq_num"],
                "data_key": info["data_key"],
                "datum_id": datum_id,
                "resource_uid": resource_uid,
                "resource_spec": spec,
                "root": root,
                "resource_path": resource_path,
                "file_reference": file_reference,
                "datum_kwargs": datum_kwargs,
                "resource": resource,
                "datum": datum,
            }

            self.resolved_by_run[info["run_uid"]][info["stream"]].append(record)

            if self.verbose:
                print(
                    f"[{info['stream']}] resolved asset: "
                    f"run={info['run_uid']}, "
                    f"key={info['data_key']}, "
                    f"datum_id={datum_id}, "
                    f"spec={spec}, "
                    f"file={file_reference}, "
                    f"datum_kwargs={datum_kwargs}"
                )

            del self.pending[datum_id]

    def latest_run_uid(self):
        return self.current_run_uid

    def latest(self, stream):
        run_uid = self.current_run_uid
        if run_uid is None:
            return None
        records = self.resolved_by_run[run_uid][stream]
        return records[-1] if records else None

    def latest_flat(self):
        return self.latest("flat")

    def latest_dark(self):
        return self.latest("dark")

    def get_run(self, run_uid=None):
        """
        Return the Tiled run node for a run UID.

        This requires a Tiled client that supports lookup by UID. Depending on
        your site configuration, this may be client[uid], client[run_uid], or
        a search query. Keep this method small so you can adapt it locally.
        """
        if self.client is None:
            raise RuntimeError("No tiled_client was supplied.")

        run_uid = run_uid or self.current_run_uid

        try:
            return self.client[run_uid]
        except Exception as exc:
            raise RuntimeError(
                f"Could not get run {run_uid!r} from Tiled using client[uid]. "
                "Your site's Tiled tree may require a search query instead."
            ) from exc

    def clear_old_runs(self, keep_latest=True):
        if keep_latest and self.current_run_uid is not None:
            latest = self.current_run_uid
            latest_data = self.resolved_by_run[latest]
            self.resolved_by_run.clear()
            self.resolved_by_run[latest] = latest_data
        else:
            self.resolved_by_run.clear()

    @staticmethod
    def _unpack_event_page(doc):
        """
        Convert an event_page document into individual event-like dicts.

        This is enough for the fields used by the resolver.
        """
        count = len(doc["uid"])
        events = []

        for i in range(count):
            data = {key: values[i] for key, values in doc["data"].items()}
            timestamps = {
                key: values[i] for key, values in doc.get("timestamps", {}).items()
            }

            events.append(
                {
                    "uid": doc["uid"][i],
                    "time": doc["time"][i],
                    "seq_num": doc["seq_num"][i],
                    "descriptor": doc["descriptor"],
                    "data": data,
                    "timestamps": timestamps,
                }
            )

        return events

    @staticmethod
    def _unpack_datum_page(doc):
        """
        Convert a datum_page document into individual datum-like dicts.
        """
        count = len(doc["datum_id"])
        datums = []

        for i in range(count):
            datum_kwargs = {
                key: values[i] for key, values in doc.get("datum_kwargs", {}).items()
            }

            datums.append(
                {
                    "datum_id": doc["datum_id"][i],
                    "resource": doc["resource"],
                    "datum_kwargs": datum_kwargs,
                }
            )

        return datums


resolver = TiledRefAssetResolver(
    image_key="KinetixU_image",
    streams=("flat", "dark"),
    verbose=False,
)

token = RE.subscribe(resolver)
