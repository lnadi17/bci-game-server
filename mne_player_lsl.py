import time
import uuid

from matplotlib import pyplot as plt
from mne import set_log_level

from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream

set_log_level("WARNING")

source_id = uuid.uuid4().hex
fname = sample.data_path() / "sample-ant-raw.fif"
player = Player(fname, chunk_size=200, source_id=source_id).start()
print(player.info)

stream = Stream(bufsize=2, source_id=source_id).connect()
print(stream.info)

ch_types = stream.get_channel_types(unique=True)
print(f"Channel types included: {', '.join(ch_types)}")