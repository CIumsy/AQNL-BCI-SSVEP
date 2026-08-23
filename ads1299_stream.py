#!/usr/bin/env python3
"""
ads1299_stream.py

Talks to the ADS1299 EEG board: opens the WebSocket, writes the chip's
configuration registers, unpacks the binary frames it sends back, and
hands each sample to a callback.

Split into functions so the calibration and live-decoding scripts can
hook into every incoming sample without having to touch any of the
protocol code.

Run it on its own to just stream:
    python ads1299_stream.py
"""

import json
import socket
import time
from typing import Callable, List, Optional

import websocket
from pylsl import StreamInfo, StreamOutlet

# ── Stream IDs ───────────────────────────────────────────────────────────────
STREAM_ADC = 0x01
STREAM_BAT = 0x02
STREAM_MPU = 0x03

# ── Protocol constants ───────────────────────────────────────────────────────
ADC_BLOCK_SIZE = 56
ADC_CHANNELS = 16
ADC_HEADER_BYTES = 8

MPU_BLOCK_SIZE = 20
MPU_CHANNELS = 6
MPU_HEADER_BYTES = 8

# ── ADS1299 config ───────────────────────────────────────────────────────────
# The one and only place the sample rate is set. CONFIG_1 just below
# works out the chip's data-rate bits from it, and run_ssvep_detection.py
# passes it to the classifier, so changing this single number
# reconfigures the board, the buffer length and the reference waves all
# together.
#
# It used to be 1000. Dropping to 250 loses nothing, because the
# classifier was already downsampling 1000 to 250 internally: same 250 Hz
# analysis rate, same 500-sample 2-second window, same 0.5 Hz resolution,
# same sub-bands.
#
# What it buys is four times fewer per-sample callbacks, which is 75% less
# work fighting for the interpreter lock on the UNO Q -- the board where
# the display and the EEG stream were starving each other.
#
# The tradeoff: with no downsampling left to do, the software anti-alias
# filter is skipped, so band-limiting is left to the chip's own sinc3
# filter. That rolls off from about 65 Hz at 250 SPS, so the 4th harmonic
# of the 15 and 17 Hz targets (60 and 68 Hz) is somewhat reduced. Minor,
# since real 4th harmonics are weak and lightly weighted anyway, but not
# nothing.
SAMPLING_RATE = 250
DAISY_EN = 0
CLK_EN = 1
CONFIG_1 = (1 << 7 | DAISY_EN << 6 | CLK_EN << 5 | 0x2 << 3 | int(SAMPLING_RATE / 250).bit_length() ^ 7)
CHANNEL_REGS = [0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C]
CHANNEL_VAL = 0x60


def parse_frame(raw: bytes):
    stream_id = raw[0]
    length = int.from_bytes(raw[1:3], "little")
    payload = raw[3:3 + length]
    return stream_id, payload


class SampleTracker:
    def __init__(self, name: str):
        self.name = name
        self.prev = -1

    def check(self, current: int) -> bool:
        if self.prev == -1:
            self.prev = current
            return True
        diff = current - self.prev
        if diff > 1:
            print(f"[{self.name}] Sample lost -- prev: {self.prev}, current: {current}")
        elif diff == 0:
            print(f"[{self.name}] Duplicate sample -- prev: {self.prev}, current: {current}")
        elif diff < 0:
            print(f"[{self.name}] Out of order -- prev: {self.prev}, current: {current}")
        self.prev = current
        return diff == 1


class ThroughputMonitor:
    def __init__(self, interval_sec: float = 1.0):
        self.interval = interval_sec
        self.start_time = time.time()
        self.bytes = 0
        self.samples = 0
        self.packets = 0

    def update(self, new_bytes: int, new_samples: int):
        self.bytes += new_bytes
        self.samples += new_samples
        self.packets += 1
        elapsed = time.time() - self.start_time
        if elapsed >= self.interval:
            # print(f"[ADC] {self.packets/elapsed:.2f} pkt/s  {self.samples/elapsed:.2f} smp/s  {self.bytes/elapsed:.2f} B/s")
            self.bytes = self.samples = self.packets = 0
            self.start_time = time.time()


def parse_adc_payload(payload: bytes, tracker: SampleTracker, monitor: ThroughputMonitor,
                       outlet: StreamOutlet,
                       on_adc_sample: Optional[Callable[[List[int]], None]] = None):
    monitor.update(len(payload), len(payload) // ADC_BLOCK_SIZE)

    for offset in range(0, len(payload), ADC_BLOCK_SIZE):
        block = payload[offset:offset + ADC_BLOCK_SIZE]
        timestamp = int.from_bytes(block[0:4], "little")
        sample_number = int.from_bytes(block[4:8], "little")

        channel_data = [
            int.from_bytes(block[ADC_HEADER_BYTES + ch * 3: ADC_HEADER_BYTES + ch * 3 + 3],
                            "big", signed=True)
            for ch in range(ADC_CHANNELS)
        ]

        tracker.check(sample_number)
        outlet.push_sample(channel_data)

        if on_adc_sample:
            on_adc_sample(channel_data)  # <-- SSVEP hook: fires once per 16-channel sample

        if all(v == 0 for v in channel_data[:3]) and all(v > 0 for v in channel_data[4:]):
            print(f"[ADC] Blank data at ts={timestamp} sample={sample_number}: {channel_data[:8]}")
            raise SystemExit("Blank data detected")


def parse_mpu_payload(payload: bytes, tracker: SampleTracker, outlet: StreamOutlet):
    for offset in range(0, len(payload), MPU_BLOCK_SIZE):
        block = payload[offset:offset + MPU_BLOCK_SIZE]
        sample_number = int.from_bytes(block[4:8], "little")
        channel_data = [
            int.from_bytes(block[MPU_HEADER_BYTES + ch * 2: MPU_HEADER_BYTES + ch * 2 + 2],
                            "little", signed=True)
            for ch in range(MPU_CHANNELS)
        ]
        tracker.check(sample_number)
        outlet.push_sample(channel_data)


def connect_and_configure(sampling_rate: int) -> websocket.WebSocket:
    ws = websocket.WebSocket()
    ws.connect("ws://" + socket.gethostbyname("oric.local") + ":81")

    def cmd(command, parameters=[]):
        ws.send_text(json.dumps({"command": command, "parameters": parameters}))

    cmd("sdatac")
    cmd("reset")
    cmd("wreg", [0x01, CONFIG_1])
    cmd("wreg", [0x03, 0xEC])
    cmd("wreg", [0x15, 0x20]) # for srb1
    for reg in CHANNEL_REGS:
        cmd("wreg", [reg, CHANNEL_VAL])
    cmd("status")
    cmd("rdatac")
    cmd("start")
    return ws


def create_outlets(sampling_rate: int):
    adc_info = StreamInfo("ORIC-16", "EEG", ADC_CHANNELS, sampling_rate, "float32", "uid007")
    mpu_info = StreamInfo("ORIC-16", "AUX", MPU_CHANNELS, 50, "float32", "uid067")
    return StreamOutlet(adc_info), StreamOutlet(mpu_info)


def run_stream(on_adc_sample: Optional[Callable[[List[int]], None]] = None,
               on_mpu_sample: Optional[Callable[[List[int]], None]] = None,
               stop_event=None):
    """Blocking loop, identical behavior to your original main(), plus hooks.
    Pass a threading.Event as stop_event to allow another thread to request a
    graceful stop (checked between messages -- see the calibration script for
    how this is used).
    """
    adc_outlet, mpu_outlet = create_outlets(SAMPLING_RATE)
    ws = connect_and_configure(SAMPLING_RATE)

    adc_tracker = SampleTracker("ADC")
    mpu_tracker = SampleTracker("MPU")
    monitor = ThroughputMonitor(interval_sec=1.0)

    print("Setup done -- streaming...")

    while stop_event is None or not stop_event.is_set():
        raw = ws.recv()

        if isinstance(raw, str):
            # print("JSON:", json.loads(raw))
            continue

        stream_id, payload = parse_frame(raw)

        if stream_id == STREAM_ADC:
            parse_adc_payload(payload, adc_tracker, monitor, adc_outlet, on_adc_sample=on_adc_sample)
        elif stream_id == STREAM_MPU:
            parse_mpu_payload(payload, mpu_tracker, mpu_outlet)
        # elif stream_id == STREAM_BAT:
            # print(f"[BAT] {int.from_bytes(payload, 'little')}%")


if __name__ == "__main__":
    run_stream()
