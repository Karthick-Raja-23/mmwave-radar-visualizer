# visuals.py — TI mmWave Visualizer (4-panel layout)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants from mmwave_config.cfg ─────────────────────────────────────────
C                = 3e8
FREQ_SLOPE       = 70e12
RAMP_END         = 57.14e-6
ADC_START        = 7 * 0.1e-6
BW               = FREQ_SLOPE * (RAMP_END - ADC_START)
RANGE_RES        = C / (2.0 * BW)
NUM_RANGE_BINS   = 256
MAX_RANGE        = NUM_RANGE_BINS * RANGE_RES   # ≈ 9.75 m

NUM_TX           = 2
NUM_RX           = 4
NUM_VIRTUAL_ANT  = NUM_TX * NUM_RX              # 8
NUM_DOPPLER_BINS = 16

LAMBDA           = C / 77e9
IDLE_TIME        = 429 * 10e-9
TC_EFF           = NUM_TX * (IDLE_TIME + RAMP_END)
MAX_VEL          = LAMBDA / (4.0 * TC_EFF)

RANGE_AXIS = np.linspace(0, MAX_RANGE, NUM_RANGE_BINS)

# ── Angle FFT ─────────────────────────────────────────────────────────────────
ANGLE_FFT_SIZE = 64
_bins          = np.arange(-ANGLE_FFT_SIZE // 2, ANGLE_FFT_SIZE // 2)
_sin_theta     = (2.0 * _bins) / ANGLE_FFT_SIZE
_valid_mask    = np.abs(_sin_theta) <= 1.0
VALID_THETA    = np.arcsin(_sin_theta[_valid_mask])
VALID_IDX      = np.where(_valid_mask)[0]

GRID_W = 300
GRID_H = 300
ALPHA  = 0.35   # EMA smoothing

# ── Dynamic range window for azimuth heatmap (matches TI demo) ────────────────
AZ_DYNAMIC_RANGE_DB = 40   # dB below peak that maps to lowest color


def _build_cartesian_lut():
    xi = np.linspace(-MAX_RANGE, MAX_RANGE, GRID_W)
    yi = np.linspace(0,           MAX_RANGE, GRID_H)
    GX, GY   = np.meshgrid(xi, yi)
    R        = np.sqrt(GX**2 + GY**2)
    TH       = np.arctan2(GX, GY)
    r_lut    = np.clip((R / MAX_RANGE * NUM_RANGE_BINS).astype(int), 0, NUM_RANGE_BINS - 1)
    th_min, th_max = VALID_THETA[0], VALID_THETA[-1]
    n_valid  = len(VALID_THETA)
    a_norm   = (TH - th_min) / (th_max - th_min) * (n_valid - 1)
    a_lut    = np.clip(np.round(a_norm).astype(int), 0, n_valid - 1)
    fov_mask = (R <= MAX_RANGE) & (TH >= th_min) & (TH <= th_max)
    return r_lut.astype(np.int32), a_lut.astype(np.int32), fov_mask


class Visualizer:

    def __init__(self,
                 num_range_bins:   int = NUM_RANGE_BINS,
                 num_doppler_bins: int = NUM_DOPPLER_BINS,
                 num_virtual_ant:  int = NUM_VIRTUAL_ANT):

        self.NR = num_range_bins
        self.ND = num_doppler_bins
        self.NV = num_virtual_ant

        self.az_win  = np.hanning(self.NV).astype(np.float32)
        self.r_lut, self.a_lut, self.fov_mask = _build_cartesian_lut()
        self.az_ema  = np.zeros((GRID_H, GRID_W),       dtype=np.float32)
        self.rd_ema  = np.zeros((self.ND, self.NR),     dtype=np.float32)

        self._build_figure()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 8), facecolor='white')
        gs = gridspec.GridSpec(2, 3, figure=self.fig, hspace=0.45, wspace=0.38)

        self.ax_xy = self.fig.add_subplot(gs[0, 0])
        self.scatter = self.ax_xy.scatter([], [], c='lime', s=20, edgecolors='none')
        self.ax_xy.set(title="Detected Points (TLV-1)",
                       xlabel="X (m)", ylabel="Y (m)",
                       xlim=(-5, 5), ylim=(0, MAX_RANGE))
        self.ax_xy.grid(True, lw=0.4, alpha=0.5)

        self.ax_rp = self.fig.add_subplot(gs[0, 1:])
        self.rp_line,    = self.ax_rp.plot(RANGE_AXIS, np.zeros(self.NR),
                                           color='royalblue', lw=1.2, label='Range Profile')
        self.noise_line, = self.ax_rp.plot(RANGE_AXIS, np.zeros(self.NR),
                                           color='limegreen', lw=1.2, label='Noise Profile')
        self.det_pts,    = self.ax_rp.plot([], [], 'o', color='orange', ms=6,
                                           label='Detected Points', linestyle='None')
        self.ax_rp.set(title="Range Profile for Zero Doppler",
                       xlabel="Range (m)", ylabel="Relative Power",
                       xlim=(0, MAX_RANGE), ylim=(0, 1000))
        self.ax_rp.legend(loc='upper right', fontsize=9)
        self.ax_rp.grid(True, lw=0.4, alpha=0.5)

        self.ax_az = self.fig.add_subplot(gs[1, 0])
        self.az_img = self.ax_az.imshow(
            self.az_ema, origin='lower', aspect='auto', cmap='jet',
            extent=[-MAX_RANGE, MAX_RANGE, 0, MAX_RANGE],
            interpolation='bilinear')
        self.fig.colorbar(self.az_img, ax=self.ax_az,
                          label='dB (angle FFT magnitude)', fraction=0.046)
        self.ax_az.set(title="Azimuth-Range Heatmap",
                       xlabel="Distance along lateral axis (meters)",
                       ylabel="Distance along longitudinal axis (meters)")

        self.ax_rd = self.fig.add_subplot(gs[1, 1:])
        self.rd_img = self.ax_rd.imshow(
            self.rd_ema, origin='lower', aspect='auto', cmap='jet',
            extent=[0, MAX_RANGE, -MAX_VEL, MAX_VEL],
            interpolation='bilinear')
        self.fig.colorbar(self.rd_img, ax=self.ax_rd, label='dB', fraction=0.046)
        self.ax_rd.axhline(0, color='white', lw=0.8, linestyle='--', alpha=0.7)
        self.ax_rd.set(title="Range-Doppler Heatmap (TLV-5)",
                       xlabel="Range (m)",
                       ylabel=f"Velocity (m/s)  [max ±{MAX_VEL:.1f} m/s]")

        plt.ion()
        plt.show()

    def _process_azimuth(self, raw_bytes):
        raw = np.frombuffer(raw_bytes, dtype=np.int16)
        if raw.size % (self.NR * 2) != 0:
            return None
        nv  = raw.size // (self.NR * 2)
        win = np.hanning(nv).astype(np.float32)

        iq   = raw.reshape(self.NR, nv, 2).astype(np.float32)
        cplx = (iq[..., 0] + 1j * iq[..., 1]) * win[np.newaxis, :]

        af   = np.fft.fftshift(np.fft.fft(cplx, n=ANGLE_FFT_SIZE, axis=1), axes=1)

        mag  = np.abs(af[:, VALID_IDX]).astype(np.float32)
        mag[:8, :] = 0.0                                          # suppress near-field clutter
        mag  = 20.0 * np.log10(mag + 1.0)

        cart = mag[self.r_lut, self.a_lut].astype(np.float32)
        cart[~self.fov_mask] = 0.0
        return cart

    def _process_rd(self, raw_bytes):
        vals = np.frombuffer(raw_bytes, dtype=np.uint16)
        if vals.size != self.ND * self.NR:
            return None
        rd = vals.reshape(self.ND, self.NR).astype(np.float32)
        rd = np.fft.fftshift(rd, axes=0)
        return 20.0 * np.log10(rd + 1.0)

    def update(self, tlvs):
        points = []; range_prof = None; noise_prof = None
        heatmap_raw = None; rd_raw = None

        for tlv in tlvs:
            t, d = tlv["type"], tlv["decoded"]
            if   t == 1: points      = d
            elif t == 2: range_prof  = np.asarray(d, dtype=np.float32)
            elif t == 3: noise_prof  = np.asarray(d, dtype=np.float32)
            elif t == 4: heatmap_raw = d
            elif t == 8: heatmap_raw = d
            elif t == 5: rd_raw      = d

        # Scatter
        if points:
            xs = np.array([p["x"] for p in points], dtype=np.float32)
            ys = np.array([p["y"] for p in points], dtype=np.float32)
            self.scatter.set_offsets(np.c_[xs, ys])
        else:
            self.scatter.set_offsets(np.empty((0, 2)))

        # Range profile
        if range_prof is not None and len(range_prof) == self.NR:
            self.rp_line.set_ydata(range_prof)
            if noise_prof is not None:
                n = len(noise_prof)
                if n != self.NR:
                    noise_prof = np.interp(
                        np.linspace(0, 1, self.NR),
                        np.linspace(0, 1, n), noise_prof).astype(np.float32)
                self.noise_line.set_ydata(noise_prof)
            combined = range_prof if noise_prof is None else np.concatenate([range_prof, noise_prof])
            p2  = float(np.percentile(combined, 2))
            p98 = float(np.percentile(combined, 98))
            pad = max((p98 - p2) * 0.1, 50.0)
            self.ax_rp.set_ylim(max(0.0, p2 - pad), p98 + pad)
            if points:
                rs  = np.array([p["range"] for p in points], dtype=np.float32)
                idx = np.clip((rs / MAX_RANGE * self.NR).astype(int), 0, self.NR - 1)
                self.det_pts.set_data(rs, range_prof[idx])
            else:
                self.det_pts.set_data([], [])

        # Azimuth heatmap — fixed 40 dB window anchored to peak
        if isinstance(heatmap_raw, (bytes, bytearray)) and heatmap_raw:
            cart = self._process_azimuth(heatmap_raw)
            if cart is not None:
                self.az_ema = ALPHA * cart + (1.0 - ALPHA) * self.az_ema
                self.az_img.set_data(self.az_ema)
                fv   = self.az_ema[self.fov_mask]
                vmax = float(np.percentile(fv, 99.5))
                vmin = 0.0
                self.az_img.set_clim(vmin, vmax)

        # Range-Doppler
        if isinstance(rd_raw, (bytes, bytearray)) and rd_raw:
            rd = self._process_rd(rd_raw)
            if rd is not None:
                self.rd_ema = ALPHA * rd + (1.0 - ALPHA) * self.rd_ema
                self.rd_img.set_data(self.rd_ema)
                self.rd_img.set_clim(float(np.percentile(self.rd_ema, 2)),
                                     float(np.percentile(self.rd_ema, 98)))

        self.fig.canvas.flush_events()
        plt.pause(0.001)