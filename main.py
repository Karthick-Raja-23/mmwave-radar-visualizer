import serial
import struct
import time
import math
import sys
from visuals import Visualizer

HEADER_FMT  = '<Q8I'
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 40 bytes

NUM_RX          = 4
NUM_TX          = 2
NUM_VIRTUAL_ANT = NUM_RX * NUM_TX   # 8 virtual antennas
NUM_RANGE_BINS  = 256
NUM_DOPPLER_BINS = 16

# ── Serial ports ───────────────────────────────────────────────────────────────
CONFIG_PORT = "COM11"
DATA_PORT   = "COM10"
CFG_FILE    = "mmwave_config.cfg"


# =============================================================================
class ConfigSender:
    """Sends .cfg lines to the sensor's control UART."""

    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud

    def send(self, cfg_file: str) -> None:
        with serial.Serial(self.port, self.baud, timeout=1) as ser:
            print("[INFO] Sending config …")
            with open(cfg_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue
                    ser.write((line + '\n').encode())
                    time.sleep(0.05)
            print("[INFO] Config sent")


# =============================================================================
class TLVDecoder:
    """Decode a single TLV payload into a Python object."""

    def decode(self, tlv_type: int, data: bytes):
        try:
            if   tlv_type == 1: return self._points(data)
            elif tlv_type == 2: return self._uint16_array(data)   # range profile
            elif tlv_type == 3: return self._uint16_array(data)   # noise profile
            elif tlv_type == 6: return {"stats_bytes": len(data)}
            elif tlv_type == 7: return self._side_info(data)
            else:               return data   # raw bytes → visuals handles 4, 5, 8
        except Exception as e:
            return {"error": str(e)}

    # TLV-1 — detected points: 4 × float32 per point (x, y, z, velocity)
    def _points(self, data: bytes) -> list:
        pts = []
        for i in range(0, len(data) - 15, 16):
            x, y, z, v = struct.unpack_from('<ffff', data, i)
            pts.append({
                "x": x, "y": y, "z": z,
                "velocity": v,
                "range": math.sqrt(x*x + y*y + z*z),
            })
        return pts

    # TLV-2 / TLV-3 — uint16 array
    def _uint16_array(self, data: bytes) -> list:
        n = len(data) // 2
        return list(struct.unpack_from(f'<{n}H', data))

    # TLV-7 — side info: SNR + noise as uint16 pairs (×0.1 dB)
    def _side_info(self, data: bytes) -> list:
        info = []
        for i in range(0, len(data) - 3, 4):
            snr, noise = struct.unpack_from('<HH', data, i)
            info.append({"snr": snr * 0.1, "noise": noise * 0.1})
        return info


# =============================================================================
class FrameParser:
    """Parse a full binary frame packet → (frame_info, [tlv, …])."""

    def __init__(self):
        self.decoder = TLVDecoder()

    def parse(self, packet: bytes):
        if len(packet) < HEADER_SIZE:
            return None, []
        try:
            hdr = struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
        except struct.error:
            return None, []

        frame_info = {
            "frame_num": hdr[4],
            "num_obj":   hdr[6],
            "num_tlv":   hdr[7],
        }

        offset = HEADER_SIZE
        tlvs   = []

        for _ in range(frame_info["num_tlv"]):
            if offset + 8 > len(packet):
                break
            tlv_type, tlv_len = struct.unpack_from('<II', packet, offset)
            offset += 8
            if offset + tlv_len > len(packet):
                break
            raw    = packet[offset: offset + tlv_len]
            offset += tlv_len
            tlvs.append({
                "type":    tlv_type,
                "length":  tlv_len,
                "decoded": self.decoder.decode(tlv_type, raw),
            })

        return frame_info, tlvs


# =============================================================================
class RadarUART:
    """
    Robust UART reader with automatic reconnection.

    Fixes "ClearCommError failed (Access is denied)":
      • Every read is wrapped in try/except serial.SerialException.
      • On any port error the port is closed and reopened after a short delay.
      • Buffer is cleared on reconnect to discard stale data.
    """

    MAX_BUF = 1 << 17   # 128 kB safety cap

    def __init__(self, port: str, baud: int = 921600):
        self.port   = port
        self.baud   = baud
        self.ser    = None
        self.buffer = bytearray()
        self._open()

    # ── Port lifecycle ────────────────────────────────────────────────────────
    def _open(self) -> None:
        self._close_quietly()
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.buffer.clear()
            print(f"[INFO] Opened {self.port} @ {self.baud}")
        except serial.SerialException as e:
            print(f"[WARN] Cannot open {self.port}: {e}")
            self.ser = None

    def _close_quietly(self) -> None:
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def _reconnect(self) -> None:
        print("[WARN] Serial error — attempting reconnect in 2 s …")
        time.sleep(2)
        self._open()

    # ── Frame extraction ──────────────────────────────────────────────────────
    def read_frame(self):
        """
        Returns one complete binary packet, or None if unavailable yet.
        Never raises — all serial errors trigger a reconnect.
        """
        if self.ser is None or not self.ser.is_open:
            self._reconnect()
            return None

        try:
            chunk = self.ser.read(4096)
        except serial.SerialException as e:
            print(f"[ERROR] UART read failed: {e}")
            self._reconnect()
            return None

        if chunk:
            self.buffer.extend(chunk)

        # Prevent unbounded growth from a malformed stream
        if len(self.buffer) > self.MAX_BUF:
            self.buffer = self.buffer[-self.MAX_BUF:]

        # Locate magic word
        idx = self.buffer.find(MAGIC_WORD)
        if idx < 0:
            return None
        if idx > 0:
            del self.buffer[:idx]

        if len(self.buffer) < HEADER_SIZE:
            return None

        # Read total packet length from byte offset 12 of the header
        # (after 8-byte magic + 4-byte version = offset 12)
        total_len = struct.unpack_from('<I', self.buffer, 12)[0]
        if total_len < HEADER_SIZE or total_len > 65536:
            del self.buffer[:8]   # skip corrupted magic, search again
            return None

        if len(self.buffer) < total_len:
            return None   # wait for more bytes

        packet = bytes(self.buffer[:total_len])
        del self.buffer[:total_len]
        return packet

    def close(self) -> None:
        self._close_quietly()


# =============================================================================
def main():
    # Send configuration to sensor
    sender = ConfigSender(CONFIG_PORT)
    sender.send(CFG_FILE)

    uart   = RadarUART(DATA_PORT)
    parser = FrameParser()
    viz    = Visualizer(
        num_range_bins   = NUM_RANGE_BINS,
        num_doppler_bins = NUM_DOPPLER_BINS,
        num_virtual_ant  = NUM_VIRTUAL_ANT,
    )

    print("\n[INFO] Reading frames … (Ctrl-C to stop)\n")
    try:
        while True:
            packet = uart.read_frame()
            if packet is None:
                continue

            frame_info, tlvs = parser.parse(packet)
            if frame_info is None:
                continue

            viz.update(tlvs)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        uart.close()


if __name__ == "__main__":
    main()
