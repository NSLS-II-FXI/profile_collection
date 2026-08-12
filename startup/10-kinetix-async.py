print(f"Loading {__file__}...")

import asyncio

from ophyd_async.core import AsyncStatus, init_devices
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.adcore import ADHDFWriter
from nslsii.ophyd_async.providers import NSLS2PathProvider


RUNNING_IN_NSLS2_CI = False


class FXIADHDFWriter(ADHDFWriter):
    async def begin_capture(self):
        await super().begin_capture()
        await self.fileio.swmr_mode.set(False)


class FXIKinetixDetector(KinetixDetector):
    """Override base StandardDetector unstage class to reset into continuous mode after scan/abort"""

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        # Stop data writing.
        await asyncio.gather(self._writer.close(), self._controller.disarm())

        # Set to continuous internal trigger, and start acquiring
        await self.driver.trigger_mode.set("Internal")
        await self.driver.image_mode.set("Continuous")
        await self._controller.arm()


def connect_to_kinetix():

    print(f"Connecting to kinetix...")
    with init_devices(mock=RUNNING_IN_NSLS2_CI):
        kinetix_path_provider = NSLS2PathProvider(RE.md, tla_suffix="-new")
        prefix = "XF:18ID1-ES{Kinetix-Det:1}"
        kinetix = FXIKinetixDetector(
            prefix,
            kinetix_path_provider,
            name="kinetix",
            writer_cls=FXIADHDFWriter,
        )

    print("Done.")

    return kinetix
