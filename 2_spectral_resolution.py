
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Load ThArNe calibration data
th_file = "/home/agungnegara/hermesRun/20251009/reduced/01153968_HRF_TH_ext.fits"
wl_file = "/home/agungnegara/hermesRun/20251009/reduced/01153968_HRF_TH_ext_wavelengthScale.fits"

th = fits.getdata(th_file)
wl = fits.getdata(wl_file)

# Ensure (pixels, orders) format
if th.shape[0] < th.shape[1]:
    th = th.T
if wl.shape[0] < wl.shape[1]:
    wl = wl.T

n_pix, n_orders = th.shape

def gaussian(x, amplitude, center, sigma, background):
    """Gaussian function for line fitting."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2) + background

def fit_emission_line(pixels, flux, wavelength, peak_pixel, halfwin=6):
    """
    Fit Gaussian to emission line and calculate resolution.
    Returns: (center_pix, center_wave, FWHM_wave, R, FWHM_pix) or None
    """
    lo = max(0, peak_pixel - halfwin)
    hi = min(len(pixels), peak_pixel + halfwin + 1)

    x_pix = pixels[lo:hi].astype(float)
    y_flux = flux[lo:hi].astype(float)
    x_wave = wavelength[lo:hi].astype(float)

    if len(x_pix) < 5:
        return None

    # Initial guesses
    bg_guess = np.median(y_flux)
    amp_guess = np.max(y_flux) - bg_guess
    center_guess = float(peak_pixel)
    sigma_guess = 2.0

    if amp_guess <= 0:
        return None

    try:
        popt, _ = curve_fit(
            gaussian, x_pix, y_flux,
            p0=[amp_guess, center_guess, sigma_guess, bg_guess],
            maxfev=5000
        )

        amp, center_pix, sigma_pix, bg = popt
        sigma_pix = abs(sigma_pix)

        if sigma_pix < 0.5 or sigma_pix > 10:
            return None

        center_wave = np.interp(center_pix, x_pix, x_wave)
        FWHM_pix = 2.355 * sigma_pix

        dispersion = abs(np.gradient(x_wave, x_pix).mean())
        FWHM_wave = FWHM_pix * dispersion

        if FWHM_wave <= 0:
            return None

        R = center_wave / FWHM_wave

        return center_pix, center_wave, FWHM_wave, R, FWHM_pix

    except:
        return None

# Find and fit emission lines
orders_to_analyze = range(5, 50)
peak_height_sigma = 6
peak_prominence = 5000
min_peak_distance = 8

all_center_pixels = []
all_resolutions = []
all_fwhm_pixels = []
all_center_wavelengths = []

for order_idx in orders_to_analyze:
    flux = th[:, order_idx]
    wave = wl[:, order_idx]
    pixels = np.arange(n_pix)

    # Find peaks
    median_flux = np.median(flux)
    std_flux = np.std(flux)
    height_threshold = median_flux + peak_height_sigma * std_flux

    peaks, _ = find_peaks(
        flux,
        height=height_threshold,
        prominence=peak_prominence,
        distance=min_peak_distance
    )

    # Fit each peak
    for peak_pix in peaks:
        result = fit_emission_line(pixels, flux, wave, peak_pix, halfwin=6)

        if result is None:
            continue

        center_pix, center_wave, fwhm_wave, R, fwhm_pix = result

        # Quality filter
        if 30000 < R < 120000 and 1.5 < fwhm_pix < 5.0:
            all_center_pixels.append(center_pix)
            all_center_wavelengths.append(center_wave)
            all_resolutions.append(R)
            all_fwhm_pixels.append(fwhm_pix)

# Convert to arrays
all_center_pixels = np.array(all_center_pixels)
all_resolutions = np.array(all_resolutions)
all_fwhm_pixels = np.array(all_fwhm_pixels)
all_center_wavelengths = np.array(all_center_wavelengths)

# Calculate statistics
median_R = np.median(all_resolutions)
mean_R = np.mean(all_resolutions)
std_R = np.std(all_resolutions)
median_sampling = np.median(all_fwhm_pixels)

print(f"Fitted {len(all_resolutions)} emission lines")
print(f"Spectral resolution: R = {median_R:.0f} ± {std_R:.0f}")
print(f"Sampling: {median_sampling:.2f} pixels/FWHM")

# Plot 1: Resolution vs pixel position
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(all_center_pixels, all_resolutions, s=20, alpha=0.6,
          color='steelblue', edgecolors='navy', linewidth=0.5)

ax.axhline(85000, ls='-', color='black', lw=2, label='Design: R = 85,000')
ax.axhline(80000, ls='--', color='red', lw=1.5, label='Goal: R > 80,000')
ax.axhline(median_R, ls=':', color='green', lw=2,
          label=f'Measured: R = {median_R:.0f}')
ax.fill_between([0, n_pix], median_R - std_R, median_R + std_R,
                alpha=0.15, color='green')

ax.set_xlabel('Pixel Position', fontsize=11)
ax.set_ylabel('Spectral Resolution R', fontsize=11)
ax.set_title('HERMES HRF Spectral Resolution', fontsize=12)
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(25000, 105000)

plt.tight_layout()
plt.savefig('spectral_resolution_analysis.png', dpi=300)

# Plot 2: Sampling distribution
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(all_fwhm_pixels, bins=40, color='steelblue',
        edgecolor='navy', alpha=0.7)

ax.axvline(median_sampling, color='green', ls='--', lw=2,
          label=f'Median: {median_sampling:.2f} pix')
ax.axvline(2.0, color='red', ls='--', lw=1.5, label='Nyquist: 2.0 pix')

ax.set_xlabel('FWHM (pixels)', fontsize=11)
ax.set_ylabel('Number of Lines', fontsize=11)
ax.set_title('HERMES HRF Sampling Distribution', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sampling_distribution.png', dpi=300)
