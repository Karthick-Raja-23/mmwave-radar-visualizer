import struct
import time
import sys
import threading
import collections
import traceback

import serial
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # explicit backend – avoids Qt/Tk conflicts
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

CONFIG_PORT = 'COM11'              # CLI / config UART (lower COM#)
DATA_PORT   = 'COM10'              # Data UART         (higher COM#)
CONFIG_FILE = 'mmwave_config.cfg'  # path to your config file

NUM_ANGLE_BINS = 64       # azimuth FFT bins (power-of-2; try 64 or 128)
LATERAL_RANGE  = (-5, 5)  # x-axis  [m]
LONG_RANGE     = (0, 10)  # y-axis  [m]
SMOOTH_SIGMA   = 0.6      # Gaussian blur (0 = off)
UPDATE_MS      = 100      # animation refresh [ms]
DEBUG          = True     # print TLV info per frame
# =============================================================================


# ---------------------------------------------------------------------------
#  Protocol constants
# ---------------------------------------------------------------------------
MAGIC_WORD     = bytes([2, 1, 4, 3, 6, 5, 8, 7])
FRAME_HDR_FMT  = '<8sIIIIIIII'   # magic(8) + 8×uint32 = 40 bytes
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)  # 40
TLV_HDR_FMT    = '<II'           # type(4) + length(4) = 8 bytes
TLV_HDR_SIZE   = struct.calcsize(TLV_HDR_FMT)    # 8

TLV_DETECTED_POINTS = 1
TLV_RANGE_PROFILE   = 2
TLV_NOISE_PROFILE   = 3
TLV_AZIMUTH_HEATMAP = 4
TLV_RANGE_DOPPLER   = 5
TLV_STATS           = 6
TLV_DETECTED_XYZ    = 7

MAX_VALID_TLV_TYPE   = 200
MAX_VALID_TLV_LEN    = 65535
MAX_VALID_FRAME_LEN  = 131072   # 128 KB
MIN_VALID_FRAME_LEN  = FRAME_HDR_SIZE + TLV_HDR_SIZE


# =============================================================================
#  1. CONFIG PARSER
# =============================================================================

def parse_config(cfg_file: str) -> dict:
    p = {}
    with open(cfg_file) as fh:
        for raw in fh:
            tok = raw.strip().split()
            if not tok:
                continue
            cmd = tok[0]

            if cmd == 'channelCfg':
                p['numRx'] = bin(int(tok[1])).count('1')
                p['numTx'] = bin(int(tok[2])).count('1')

            elif cmd == 'profileCfg':
                p['rampEndTime']    = float(tok[5]) * 1e-6   # µs -> s
                p['freqSlopeConst'] = float(tok[8]) * 1e12   # MHz/µs -> Hz/s
                p['numAdcSamples']  = int(tok[10])
                p['adcSampleRate']  = float(tok[11]) * 1e3   # ksps -> sps

            elif cmd == 'frameCfg':
                p['numLoops'] = int(tok[3])

    c               = 3e8
    bw              = p['freqSlopeConst'] * p['rampEndTime']
    p['rangeRes']   = c / (2.0 * bw)
    p['numRangeBins'] = p['numAdcSamples']
    p['maxRange']   = p['rangeRes'] * p['numRangeBins']
    p['numVirtAnt'] = p['numRx'] * p['numTx']

    print("=" * 55)
    print(f"  Config : {cfg_file}")
    print(f"  RX={p['numRx']}  TX={p['numTx']}  VirtualAnt={p['numVirtAnt']}")
    print(f"  ADC samples : {p['numAdcSamples']}")
    print(f"  Range res   : {p['rangeRes']*100:.2f} cm")
    print(f"  Max range   : {p['maxRange']:.2f} m")
    print(f"  Bandwidth   : {bw/1e9:.3f} GHz")
    print("=" * 55)
    return p


# =============================================================================
#  2. CONFIG SENDER
# =============================================================================

def send_config(cfg_port: str, cfg_file: str) -> None:
    print(f"\nOpening config port {cfg_port} ...")
    try:
        ser = serial.Serial(cfg_port, 115200, timeout=1)
    except serial.SerialException as e:
        sys.exit(f"[ERROR] Cannot open config port: {e}")

    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    with open(cfg_file) as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith('%')]

    print(f"Sending {len(lines)} commands ...")
    for line in lines:
        ser.write((line + '\n').encode())
        time.sleep(0.08)
        resp = ser.read_all().decode(errors='ignore').strip()
        tag  = 'OK' if ('Done' in resp or resp == '') else f'?? ({resp[:40]})'
        if DEBUG:
            print(f"  [{tag}]  {line}")

    ser.close()
    print("Config uploaded – sensor running.\n")


# =============================================================================
#  3. FRAME READER  (background thread)
# =============================================================================

class FrameReader(threading.Thread):
    """
    Reads TLV frames from the data UART and pushes heatmap arrays to
    self.queue.  Uses a validated-header approach to avoid false syncs.
    """

    def __init__(self, port: str, params: dict):
        super().__init__(daemon=True)
        self.port         = port
        self.params       = params
        self.queue        = collections.deque(maxlen=2)
        self.frame_count  = 0
        self.tlv_counters = collections.Counter()
        # NOTE: cannot name this _stop – that's a method on threading.Thread
        self._running     = threading.Event()
        self._running.set()

    def stop(self):
        self._running.clear()

    # -------------------------------------------------------------------------
    #  Serial helpers
    # -------------------------------------------------------------------------

    def _read_exact(self, ser: serial.Serial, n: int) -> bytes:
        """Block until exactly n bytes are available."""
        buf = b''
        while len(buf) < n:
            if not self._running.is_set():
                raise EOFError("Reader stopped")
            chunk = ser.read(n - len(buf))
            if chunk:
                buf += chunk
        return buf

    def _sync_to_magic(self, ser: serial.Serial) -> bool:
        """
        Slide one byte at a time until MAGIC_WORD is found.
        Returns True when found, False if reader was stopped.
        """
        window = bytearray(self._read_exact(ser, 8))
        while bytes(window) != MAGIC_WORD:
            if not self._running.is_set():
                return False
            window = window[1:] + bytearray(self._read_exact(ser, 1))
        return True

    # -------------------------------------------------------------------------
    #  Header validation  (guards against false magic-word hits in data)
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_valid_header(version, total_len, platform, frame_num,
                         cpu_cyc, num_obj, num_tlvs, subframe) -> bool:
        """
        Sanity-check header fields.  Any obviously-wrong value means we
        synced onto the magic-word pattern inside a TLV payload.
        """
        if not (MIN_VALID_FRAME_LEN <= total_len <= MAX_VALID_FRAME_LEN):
            return False
        if num_tlvs == 0 or num_tlvs > 20:
            return False
        if num_obj > 200:
            return False
        if subframe > 4:     # max sub-frames in any mmWave config
            return False
        return True

    # -------------------------------------------------------------------------
    #  TLV parsers
    # -------------------------------------------------------------------------

    def _parse_tlv4(self, payload: bytes):
        """
        TLV type 4 – Azimuth Static Heatmap.
        Payload: int16 pairs (real, imag) for [rangeBin][virtAnt].
        Returns complex ndarray (numRangeBins, numVirtAnt).
        """
        nr       = self.params['numRangeBins']
        nv       = self.params['numVirtAnt']
        expected = nr * nv * 4   # 2 × int16 per element

        if len(payload) < expected:
            print(f"  [WARN] TLV4 payload {len(payload)} B < expected {expected} B")
            return None

        raw = np.frombuffer(
            payload[:expected], dtype=np.int16
        ).reshape(nr, nv, 2).astype(np.float32)

        return raw[..., 0] + 1j * raw[..., 1]   # complex (nr, nv)

    def _parse_tlv1_float(self, payload: bytes):
        """
        TLV type 1 – Detected Points  (SDK ≥ 3.x float format).
        Each point: x(f32) y(f32) z(f32) doppler(f32) = 16 bytes.
        Returns list of (x, y) tuples with NaN/Inf filtered out.
        """
        n   = len(payload) // 16
        pts = []
        fmt = struct.Struct('<ffff')
        for i in range(n):
            x, y, z, d = fmt.unpack_from(payload, i * 16)
            if np.isfinite(x) and np.isfinite(y) and (x != 0.0 or y != 0.0):
                pts.append((float(x), float(y)))
        return pts

    # -------------------------------------------------------------------------
    #  Frame processor
    # -------------------------------------------------------------------------

    def _process_frame(self, header_bytes: bytes, body: bytes):
        """
        Walk the TLV list.  Stops immediately on any invalid TLV header.
        Returns a magnitude heatmap ndarray or None.
        """
        (_, version, total_len, platform, frame_num, cpu_cyc,
         num_obj, num_tlvs, subframe) = struct.unpack(FRAME_HDR_FMT, header_bytes)

        if DEBUG:
            print(f"  Frame #{frame_num:05d}  tlvs={num_tlvs}  objs={num_obj}"
                  f"  total_len={total_len}")

        offset  = 0
        heatmap = None
        points  = []

        for t_idx in range(num_tlvs):
            # Check we have enough bytes for a TLV header
            if offset + TLV_HDR_SIZE > len(body):
                if DEBUG:
                    print(f"    [WARN] Body exhausted after {t_idx} TLVs")
                break

            tlv_type, tlv_len = struct.unpack_from(TLV_HDR_FMT, body, offset)
            offset += TLV_HDR_SIZE

            # ---- Validity gate -------------------------------------------------
            if tlv_type > MAX_VALID_TLV_TYPE or tlv_len > MAX_VALID_TLV_LEN:
                print(f"    [WARN] Invalid TLV type={tlv_type} len={tlv_len} "
                      f"at body offset {offset-TLV_HDR_SIZE} – frame corrupt, discarding rest")
                break
            if offset + tlv_len > len(body):
                if DEBUG:
                    print(f"    [WARN] TLV {tlv_type} extends past frame body")
                break
            # -------------------------------------------------------------------

            payload = body[offset: offset + tlv_len]
            offset += tlv_len

            self.tlv_counters[tlv_type] += 1
            if DEBUG:
                print(f"    TLV type={tlv_type:2d}  len={tlv_len:6d} B")

            if tlv_type in (TLV_AZIMUTH_HEATMAP, 8):
                heatmap = self._parse_tlv4(payload)
                print("HEATMAP RECEIVED")

            elif tlv_type in (TLV_DETECTED_POINTS, TLV_DETECTED_XYZ):
                points = self._parse_tlv1_float(payload)

        # ---- Build output heatmap -------------------------------------------
        if heatmap is not None:
            # Remove DC / static clutter across antennas
            heatmap = heatmap - np.mean(heatmap, axis=1, keepdims=True)

            af = np.fft.fftshift(
                np.fft.fft(heatmap, n=NUM_ANGLE_BINS, axis=1), axes=1)

            return np.abs(af)

        return None

    def _points_to_heatmap(self, points):
        """Project (x, y) detections onto a polar-to-Cartesian grid."""
        nr  = self.params['numRangeBins']
        na  = NUM_ANGLE_BINS
        hm  = np.zeros((nr, na), dtype=np.float32)
        mr  = self.params['maxRange']
        dr  = mr / nr
        da  = np.pi / na

        for x, y in points:
            r = float(np.sqrt(x**2 + y**2))
            if r <= 0 or r > mr:
                continue
            angle = float(np.arctan2(x, y)) + np.pi / 2
            ri = int(r / dr)
            ai = int(angle / da)
            if 0 <= ri < nr and 0 <= ai < na:
                hm[ri, ai] += 1.0

        return hm

    # -------------------------------------------------------------------------
    #  Thread main loop
    # -------------------------------------------------------------------------

    def run(self):
        print(f"Opening data port {self.port} (921600 baud) ...")
        try:
            ser = serial.Serial(self.port, 921600, timeout=2)
        except serial.SerialException as e:
            print(f"[ERROR] Cannot open data port {self.port}: {e}")
            return

        # Drain stale UART bytes
        time.sleep(0.4)
        ser.reset_input_buffer()
        print("Data port ready – listening for frames ...\n")

        consec_no_hm = 0

        try:
            while self._running.is_set():

                # ------ Step 1: sync on magic word --------------------------
                if not self._sync_to_magic(ser):
                    break

                # ------ Step 2: read rest of header (32 more bytes) ---------
                rest = self._read_exact(ser, FRAME_HDR_SIZE - 8)
                header = MAGIC_WORD + rest

                (_, version, total_len, platform, frame_num, cpu_cyc,
                 num_obj, num_tlvs, subframe) = struct.unpack(FRAME_HDR_FMT, header)

                # ------ Step 3: validate header (false-sync guard) -----------
                if not self._is_valid_header(version, total_len, platform,
                                             frame_num, cpu_cyc, num_obj,
                                             num_tlvs, subframe):
                    if DEBUG:
                        print(f"  [SYNC] Bad header "
                              f"(total_len={total_len} num_tlvs={num_tlvs} "
                              f"num_obj={num_obj} subframe={subframe}) – re-syncing")
                    continue   # discard, go back to magic-word search

                # ------ Step 4: read body -----------------------------------
                body_len = total_len - FRAME_HDR_SIZE
                if body_len <= 0:
                    continue
                body = self._read_exact(ser, body_len)

                # ------ Step 5: parse TLVs ----------------------------------
                hm = self._process_frame(header, body)
                if hm is not None:
                    self.frame_count += 1
                    self.queue.append(hm)
                    consec_no_hm = 0
                else:
                    consec_no_hm += 1
                    if consec_no_hm == 10 and DEBUG:
                        print("  [INFO] 10 frames without a heatmap TLV.")
                        print(f"         TLV types seen: {dict(self.tlv_counters)}")
                        print("         Ensure guiMonitor parameter 4 = 1 in config.")

        except EOFError:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            ser.close()
            print(f"\nData port closed.  Frames OK: {self.frame_count}")
            print(f"TLV type counts: {dict(self.tlv_counters)}")


# =============================================================================
#  4. CARTESIAN COORDINATE GRID
# =============================================================================

def build_cartesian_grid(params: dict):
    """
    Pre-compute X, Y Cartesian meshes (numRangeBins × NUM_ANGLE_BINS).
    ULA with d = λ/2  →  azimuth FFT bins map to sin(θ) uniformly.
    """
    nr     = params['numRangeBins']
    na     = NUM_ANGLE_BINS
    r      = np.linspace(0.0, params['maxRange'], nr)
    sin_az = np.linspace(-0.95, 0.95, na, endpoint=False)
    az     = np.arcsin(np.clip(sin_az, -1.0, 1.0))

    R, A = np.meshgrid(r, az, indexing='ij')   # (nr, na)
    return R * np.sin(A), R * np.cos(A)         # X, Y


# =============================================================================
#  5. MAIN
# =============================================================================

def main():
    params = parse_config(CONFIG_FILE)
    send_config(CONFIG_PORT, CONFIG_FILE)

    reader = FrameReader(DATA_PORT, params)
    reader.start()
    time.sleep(0.5)

    X, Y = build_cartesian_grid(params)
    nr, na = params['numRangeBins'], NUM_ANGLE_BINS

    # -------------------------------------------------------------------------
    #  Figure – styled to match the mmWave Demo Visualiser
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#020B30')

    mesh = ax.pcolormesh(
        X, Y, np.zeros((nr, na)),
        cmap='jet',
        shading='gouraud',
        vmin=-40,
        vmax=0,
        rasterized=True,
    )

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalised Power', fontsize=9)
    cbar.ax.yaxis.set_tick_params(labelsize=8)

    ax.set_xlim(*LATERAL_RANGE)
    ax.set_ylim(*LONG_RANGE)
    ax.set_xlabel('Distance along lateral axis (meters)', fontsize=11, labelpad=8)
    ax.set_ylabel('Distance along longitudinal axis (meters)', fontsize=11, labelpad=8)
    ax.set_title('Azimuth-Range Heatmap', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(axis='both', labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor('#aaaaaa')

    info = ax.text(
        0.01, 0.99, 'Waiting for data ...',
        transform=ax.transAxes, va='top', ha='left',
        fontsize=8, color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#020B30', alpha=0.75),
    )

    fig.tight_layout()

    # -------------------------------------------------------------------------
    #  Animation
    # -------------------------------------------------------------------------
    ani_holder = [None]   # mutable container so on_close can reference it

    def update(_):
        if not reader.queue:
            return mesh, info

        hm = reader.queue.pop()

        if SMOOTH_SIGMA > 0:
            hm = gaussian_filter(hm, sigma=SMOOTH_SIGMA)

        hm = 20 * np.log10(hm + 1e-6)
        hm = hm - np.max(hm)
        hm = np.clip(hm, -40, 0)

        mesh.set_array(hm.ravel())
        info.set_text(
            f"Frame: {reader.frame_count}   "
            f"TLVs: {dict(reader.tlv_counters)}"
        )
        return mesh, info

    def on_close(event):
        """Stop the animation cleanly when the window is closed."""
        ani = ani_holder[0]
        if ani is not None:
            ani.event_source.stop()
        reader.stop()

    fig.canvas.mpl_connect('close_event', on_close)

    ani = FuncAnimation(
        fig, update,
        interval=UPDATE_MS,
        blit=True,
        cache_frame_data=False,
    )
    ani_holder[0] = ani

    plt.show()

    # Cleanup after window closes
    reader.stop()
    reader.join(timeout=3)
    print("Done.")


if __name__ == '__main__':
    main()