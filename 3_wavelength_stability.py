
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ThArNe calibration measurements
file_numbers = [1153968, 1153993, 1154031]
times_ut = ['16:21:59', '00:31:52', '07:19:46']
vrad = [0.0096, 0.0340, 0.1226]  # km/s
err_vrad = [0.0022, 0.0023, 0.0024]  # km/s

def time_to_hours(time_str, reference_time):
    """Convert time to hours since reference."""
    t = datetime.strptime(time_str, '%H:%M:%S')
    ref = datetime.strptime(reference_time, '%H:%M:%S')
    hours = (t.hour - ref.hour) + (t.minute - ref.minute)/60.0 + (t.second - ref.second)/3600.0
    if hours < 0:
        hours += 24
    return hours

time_hours = [time_to_hours(t, times_ut[0]) for t in times_ut]

# Convert to m/s and make relative to first measurement
vrad_ms = np.array(vrad) * 1000
vrad_ms_relative = vrad_ms - vrad_ms[0]
err_vrad_ms = np.array(err_vrad) * 1000

# Plot wavelength drift
fig, ax = plt.subplots(figsize=(12, 7))

ax.errorbar(time_hours, vrad_ms_relative, yerr=err_vrad_ms,
            fmt='o-', markersize=10, capsize=5, capthick=2,
            color='#2E86AB', ecolor='#A23B72', linewidth=2.5,
            label='ThArNe wavelength drift')

# Reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5,
           alpha=0.7, label='Reference (first ThArNe)')
ax.axhline(y=5, color='green', linestyle='--', linewidth=2,
           alpha=0.6, label='Short-term goal (±5 m/s)')
ax.axhline(y=-5, color='green', linestyle='--', linewidth=2, alpha=0.6)
ax.axhline(y=79, color='orange', linestyle=':', linewidth=2,
           alpha=0.6, label='Long-term stability (79 m/s)')

ax.axhspan(-5, 5, alpha=0.1, color='green')

ax.set_xlabel('Time since first ThArNe (hours)', fontsize=12)
ax.set_ylabel('Wavelength Drift (m/s)', fontsize=12)
ax.set_title('HERMES Wavelength Stability - Night 20251009', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10, loc='upper left')

# Annotate measurements
for i, (x, y, fn) in enumerate(zip(time_hours, vrad_ms_relative, file_numbers)):
    offset_y = 15 if i % 2 == 0 else -35
    ax.annotate(f'{fn}\n{times_ut[i]} UT',
                xy=(x, y), xytext=(0, offset_y), textcoords='offset points',
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                         edgecolor='gray', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# Summary box
textstr = f'Total drift: {vrad_ms_relative[-1]:.1f} m/s\n' \
          f'Drift rate: {vrad_ms_relative[-1]/time_hours[-1]:.1f} m/s/hr\n' \
          f'Duration: {time_hours[-1]:.1f} hours'
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('wavelength_stability_analysis.png', dpi=300)
# plt.close()

# Print results
print(f"\nWavelength Stability Analysis - Night 20251009")
print(f"Total drift over {time_hours[-1]:.1f} hours: {vrad_ms_relative[-1]:.1f} m/s")
print(f"Drift rate: {vrad_ms_relative[-1]/time_hours[-1]:.2f} m/s/hour")
print(f"Maximum deviation: {max(abs(vrad_ms_relative)):.1f} m/s")
