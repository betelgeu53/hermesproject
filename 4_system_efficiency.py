
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Star information and ETC predictions
star_info = {
    'HD182488': {
        'file': '01153979',
        'vmag': 6.15,
        'exptime': 30.0,
        'airmass': 1.003,
        'spectral_type': 'K',
        'etc_snr_550nm': 60.6,
        'etc_flux_550nm': 4016.1
    },
    'HD185395': {
        'file': '01153980',
        'vmag': 4.48,
        'exptime': 360.0,
        'airmass': 1.075,
        'spectral_type': 'F',
        'etc_snr_550nm': 471.6,
        'etc_flux_550nm': 222749.5
    },
    'HD168797': {
        'file': '01153981',
        'vmag': 6.13,
        'exptime': 550.0,
        'airmass': 1.145,
        'spectral_type': 'B',
        'etc_snr_550nm': 271.2,
        'etc_flux_550nm': 73925.2
    }
}

def measure_snr(file_number, night='20251009',
                wavelength_center=5500, wavelength_width=50):
    """
    Measure S/N using photon statistics.
    Returns S/N per resolution element at specified wavelength.
    """
    base_path = f'/home/agungnegara/hermesRun/{night}/reduced/'
    merged_file = f'{base_path}{file_number}_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits'

    try:
        with fits.open(merged_file) as hdul:
            flux = hdul[0].data
            header = hdul[0].header

            # Get wavelength array
            crval1 = header.get('CRVAL1')
            cdelt1 = header.get('CDELT1')
            naxis1 = header.get('NAXIS1')

            if not all([crval1, cdelt1, naxis1]):
                return None

            log_wavelength = crval1 + np.arange(naxis1) * cdelt1
            wavelength = np.exp(log_wavelength)

            # Select measurement region
            wl_min = wavelength_center - wavelength_width/2
            wl_max = wavelength_center + wavelength_width/2
            mask = (wavelength >= wl_min) & (wavelength <= wl_max)

            if np.sum(mask) == 0:
                return None

            flux_region = flux[mask]
            wl_region = wavelength[mask]

            # Clean data
            valid = np.isfinite(flux_region) & (flux_region > 0)
            flux_clean = np.asarray(flux_region[valid], dtype=np.float64)
            wl_clean = np.asarray(wl_region[valid], dtype=np.float64)

            if len(flux_clean) < 10:
                return None

            # Calculate S/N using photon statistics
            continuum_flux = np.percentile(flux_clean, 90)  # electrons per pixel

            # Noise components
            noise_photon = np.sqrt(np.abs(continuum_flux))
            readnoise = 2.9  # electrons (from HERMES specs)
            noise_total = np.sqrt(continuum_flux + readnoise**2)

            # S/N per pixel
            snr_per_pixel = continuum_flux / noise_total

            # Convert to S/N per resolution element
            resolution = 85000
            fwhm_angstrom = wavelength_center / resolution
            pixel_size = np.median(np.diff(wl_clean))
            pixels_per_resel = fwhm_angstrom / pixel_size
            snr_per_resel = snr_per_pixel * np.sqrt(pixels_per_resel)

            # Alternative using median
            median_flux = np.median(flux_clean)
            noise_median = np.sqrt(median_flux + readnoise**2)
            snr_median_pixel = median_flux / noise_median
            snr_median_resel = snr_median_pixel * np.sqrt(pixels_per_resel)

            return {
                'snr_per_pixel': snr_per_pixel,
                'snr_per_resel': snr_per_resel,
                'snr_median_resel': snr_median_resel,
                'continuum_flux': continuum_flux,
                'median_flux': median_flux,
                'noise_total': noise_total,
                'pixels_per_resel': pixels_per_resel
            }

    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def calculate_efficiency(star_name, info, measured):
    """
    Calculate system efficiency from measured vs predicted S/N.
    Efficiency = (Measured S/N / ETC S/N)²
    """
    if measured is None:
        return None

    etc_snr = info['etc_snr_550nm']
    measured_snr = measured['snr_per_resel']

    ratio = measured_snr / etc_snr
    efficiency = ratio ** 2 * 100

    return {
        'star': star_name,
        'etc_snr': etc_snr,
        'measured_snr': measured_snr,
        'ratio': ratio,
        'efficiency': efficiency,
        'vmag': info['vmag'],
        'exptime': info['exptime'],
        'airmass': info['airmass'],
        'spectral_type': info['spectral_type']
    }

def plot_efficiency(results):
    """Create efficiency comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    stars = [r['star'] for r in results]
    etc_snr = np.array([r['etc_snr'] for r in results])
    measured_snr = np.array([r['measured_snr'] for r in results])
    efficiency = np.array([r['efficiency'] for r in results])

    x = np.arange(len(stars))
    width = 0.35

    # Panel A: S/N comparison
    ax1.bar(x - width/2, etc_snr, width, label='ETC Predicted',
            color='#3498db', edgecolor='black', alpha=0.85)
    ax1.bar(x + width/2, measured_snr, width, label='Measured',
            color='#e74c3c', edgecolor='black', alpha=0.85)

    ax1.set_ylabel('Signal-to-Noise Ratio', fontsize=11)
    ax1.set_title('(a) ETC Prediction vs Measured S/N', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stars)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel B: Efficiency
    colors = ['#e74c3c' if e < 15 else '#f39c12' if e < 35 else
              '#27ae60' if e <= 60 else '#9b59b6' for e in efficiency]

    bars = ax2.bar(x, efficiency, color=colors, edgecolor='black', alpha=0.85)

    ax2.axhline(y=100, color='green', linestyle='--', linewidth=2,
                label='100% (Ideal)', alpha=0.5)
    ax2.axhline(y=30, color='orange', linestyle=':', linewidth=2,
                label='~30% (HERMES Typical)', alpha=0.7)

    ax2.set_ylabel('System Efficiency (%)', fontsize=11)
    ax2.set_title('(b) Measured System Efficiency', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stars)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('system_efficiency_analysis.png', dpi=300)
    plt.close()

    return fig

# Main analysis
if __name__ == "__main__":
    results = []

    for star_name, info in star_info.items():
        measured = measure_snr(info['file'])

        if measured:
            result = calculate_efficiency(star_name, info, measured)
            if result:
                results.append(result)
                print(f"{star_name}: {result['efficiency']:.1f}% "
                      f"(S/N: {result['measured_snr']:.1f} vs {result['etc_snr']:.1f})")

    # Create plots
    if len(results) > 0:
        plot_efficiency(results)

        avg_eff = np.mean([r['efficiency'] for r in results])
        std_eff = np.std([r['efficiency'] for r in results])

        print(f"\nAverage efficiency: {avg_eff:.1f}% ± {std_eff:.1f}%")
