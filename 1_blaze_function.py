import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ---------- theory ----------
d_A   = 1.89e5                 # groove spacing [Å]  (18.9 micron)
delta = np.deg2rad(69.74)      # blaze angle [rad]

m_theory = np.arange(40, 95)   # orders 40..94
lambda_theory = (2 * d_A * np.sin(delta)) / m_theory

plt.plot(m_theory, lambda_theory, "o")
plt.xlabel("Diffraction order m")
plt.ylabel("Theoretical blaze wavelength [Å]")
plt.tight_layout()
plt.show()


# ---------- measured ----------
ff_file = "/home/agungnegara/hermesRun/20251009/reduced/01153948_HRF_FF_ext.fits"
wl_file = "/home/agungnegara/hermesRun/20251009/reduced/01153968_HRF_TH_ext_wavelengthScale.fits"

ff = fits.getdata(ff_file)
wl = fits.getdata(wl_file)

# make sure arrays are (pix, orders)
if ff.shape[0] < ff.shape[1]:
    ff = ff.T
if wl.shape[0] < wl.shape[1]:
    wl = wl.T

n_pix, n_orders = ff.shape
m0 = 40
m = m0 + np.arange(n_orders)   # 40..94

blaze_pix = np.argmax(ff, axis=0)                       # per order
lambda_meas = wl[blaze_pix, np.arange(n_orders)]        # Å
lambda_th   = (2 * d_A * np.sin(delta)) / m             # Å

# compare
plt.figure(figsize=(7,4))
plt.plot(m, lambda_th, "r--", label="Theory")
plt.plot(m, lambda_meas, "bx", label="Measured")
plt.xlabel("Spectral order m")
plt.ylabel("Blaze wavelength [Å]")
plt.legend()
plt.tight_layout()
plt.show()

# residuals
plt.figure(figsize=(7,4))
plt.plot(m, lambda_meas - lambda_th, "o")
plt.axhline(0, ls="--", color="k", alpha=0.4)
plt.xlabel("Spectral order m")
plt.ylabel("Residual (meas - theory) [Å]")
plt.tight_layout()
plt.show()


#%%%%%%%%%%%
#this is for flat field blaze profiles
def fsr_edges(flux, wave, frac=0.05):
    if len(flux) < 10 or np.all(np.isnan(flux)):
        return None
    peak_flux = np.nanmax(flux)
    threshold = frac * peak_flux
    above_thresh = flux > threshold
    if not np.any(above_thresh):
        return None
    valid_idx = np.where(above_thresh)[0]
    return (wave[valid_idx[0]], wave[valid_idx[-1]])

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle("Flat-field Blaze Profiles (HERMES) - All Spectral Orders",
             fontsize=14, fontweight='bold')

windows = [
    (3800, 4900, "UV-Blue (orders 74-94)"),
    (4800, 6200, "Green-Yellow (orders 54-74)"),
    (6200, 9000, "Red-NIR (orders 40-54)"),
]

for ax, (wmin, wmax, label) in zip(axes, windows):
    orders_plotted = 0
    ff_label_added = False  # Track if we've added "FF spectrum" label
    edge_label_added = False  # Track if we've added "FSR edges" label

    # Plot ALL orders in BLUE
    for ord_idx in range(n_orders):
        wave = wl[:, ord_idx]
        flux = ff[:, ord_idx]

        # Check overlap with window
        wave_min_ord = np.nanmin(wave)
        wave_max_ord = np.nanmax(wave)

        if wave_max_ord < wmin or wave_min_ord > wmax:
            continue

        # Crop to window
        mask = (wave >= wmin) & (wave <= wmax) & np.isfinite(flux) & np.isfinite(wave)
        if np.sum(mask) < 10:
            continue

        wave_plot = wave[mask]
        flux_plot = flux[mask]

        # Plot in BLUE - only add label once
        if not ff_label_added:
            ax.plot(wave_plot, flux_plot, color='blue', lw=1,
                   label='FF spectrum', alpha=0.7)
            ff_label_added = True
        else:
            ax.plot(wave_plot, flux_plot, color='blue', lw=1, alpha=0.7)

        orders_plotted += 1

        # Add FSR edges in GREEN - only add label once
        edges = fsr_edges(flux_plot, wave_plot, frac=0.05)
        if edges:
            left, right = edges
            if not edge_label_added:
                ax.axvline(left, color='green', ls='--', lw=1,
                          label='FSR edges', alpha=0.6)
                ax.axvline(right, color='green', ls='--', lw=1, alpha=0.6)
                edge_label_added = True
            else:
                ax.axvline(left, color='green', ls='--', lw=1, alpha=0.6)
                ax.axvline(right, color='green', ls='--', lw=1, alpha=0.6)

    ax.set_xlim(wmin, wmax)
    ax.set_ylabel("Flux (e$^-$)", fontweight='bold', fontsize=11)
    ax.set_title(label, fontsize=11, loc='left', fontweight='bold')
    ax.grid(alpha=0.3)

    # Add text showing number of orders
    ax.text(0.98, 0.95, f'{orders_plotted} orders shown',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add legend
    ax.legend(loc='best', fontsize=10)

axes[-1].set_xlabel("Wavelength (Å)", fontweight='bold', fontsize=12)
plt.tight_layout()
plt.show()

#blaze angle
# Calculate blaze angle from measurements
delta_m_rad = np.arcsin(m * lambda_meas / (2 * d_A))
delta_m_deg = np.rad2deg(delta_m_rad)

# Statistics
mu = np.mean(delta_m_deg)
sig = np.std(delta_m_deg, ddof=1)

print(f"Measured blaze angle: δ = {mu:.2f} ± {sig:.2f}°")
print(f"Theoretical value: δ = {delta_theory}°")
print(f"Difference: {mu - delta_theory:.2f}°")
print(f"\nGroove spacing used: d = {d_A/1e4:.2f} μm")

# Plot
plt.figure(figsize=(9, 5))
plt.plot(m, delta_m_deg, 'o', markersize=5, color='steelblue',
         label=f"Measured per order", alpha=0.7)
plt.axhline(delta_theory, ls="--", color='red', lw=2,
           label=f"Theoretical ({delta_theory}°)")
plt.axhline(mu, ls="-", color='green', lw=2,
           label=f"Mean ({mu:.2f}°)")

# Shade ±1σ region
plt.fill_between([m.min(), m.max()], mu - sig, mu + sig,
                 alpha=0.2, color='green',
                 label=f"±1σ ({sig:.2f}°)")

plt.xlabel("Spectral order m", fontweight='bold', fontsize=12)
plt.ylabel("Blaze angle δ [degrees]", fontweight='bold', fontsize=12)
plt.title("Blaze Angle Determination from Flat Field",
         fontweight='bold', fontsize=13)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.ylim(69, 71)  # Zoom in to relevant range
plt.tight_layout()
plt.show()
