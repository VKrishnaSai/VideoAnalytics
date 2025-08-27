import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# -------------------------------
# Parameters and Time Base
# -------------------------------
fs = 2000            # Sampling frequency in Hz
T = 1                # Signal duration in seconds
t = np.linspace(0, T, int(fs*T), endpoint=False)

# -------------------------------
# 1. Generate a Baseband Signal
# -------------------------------
# For example, use a simple cosine (e.g., a 5 Hz tone)
f_base = 5  # Baseband frequency in Hz
x = np.sin(2 * np.pi * f_base * t)  # Baseband signal

# -------------------------------
# 2. Compute the Analytic Signal via Hilbert Transform
# -------------------------------
# The analytic signal x_a(t) = x(t) + j*H{x(t)}
x_analytic = hilbert(x)  # scipy returns the analytic signal
# Separate real and imaginary parts:
x_real = np.real(x_analytic)  # should equal x(t)
x_imag = np.imag(x_analytic)  # this is the Hilbert transform of x(t)

# -------------------------------
# 3. Define the Carrier and Form the Modulated Signal
# -------------------------------
f_c = 100  # Carrier frequency in Hz
# Create a complex exponential carrier: exp(j2pi f_c t)
carrier = np.exp(1j * 2 * np.pi * f_c * t)

# Option 1: Modulate via the Analytic Signal
# Multiply the analytic signal by the carrier:
ssb_analytic = x_analytic * carrier
# The actual SSB signal transmitted is the real part:
ssb = np.real(ssb_analytic)

# Option 2: IQ Modulation Approach
# Define In-phase (I) and Quadrature (Q) components:
I = x              # In-phase: original baseband
Q = x_imag         # Quadrature: Hilbert transform of x(t)
# Mix with cosine and sine carriers respectively:
ssb_IQ = I * np.cos(2 * np.pi * f_c * t) - Q * np.sin(2 * np.pi * f_c * t)
# ssb and ssb_IQ should be nearly identical.
error = np.max(np.abs(ssb - ssb_IQ))
print("Max error between analytic modulation and IQ modulation:", error)

# -------------------------------
# 4. Plotting: Step-by-Step Visualization
# -------------------------------
plt.figure(figsize=(14, 12))

# Plot A: Baseband Signal and its Hilbert Transform
plt.subplot(4, 1, 1)
plt.plot(t, x, label='x(t) (Baseband)')
plt.plot(t, x_imag, label='Hilbert{x(t)}', linestyle='--')
plt.title('Baseband Signal and its Hilbert Transform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot B: Analytic Signal: Real and Imaginary Parts
plt.subplot(4, 1, 2)
plt.plot(t, x_real, label='Real part of x_a(t)', color='C0')
plt.plot(t, x_imag, label='Imaginary part of x_a(t)', color='C1', linestyle='--')
plt.title('Analytic Signal (Baseband)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot C: SSB Signal via Analytic Modulation
plt.subplot(4, 1, 3)
plt.plot(t, ssb, label='SSB Signal (Real part of x_a(t)*exp(j2πf_ct))')
plt.title('SSB Modulated Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot D: Complex Modulated Signal: Real & Imaginary Parts
plt.subplot(4, 1, 4)
plt.plot(t, np.real(ssb_analytic), label='Real part of modulated analytic signal')
plt.plot(t, np.imag(ssb_analytic), label='Imaginary part of modulated analytic signal', linestyle='--')
plt.title('Complex Modulated (SSB) Signal: Real & Imaginary Parts')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# 5. Frequency Domain Visualization
# -------------------------------
# Compute the FFT of the analytic modulated signal (SSB)
N = len(ssb_analytic)
f_axis = np.linspace(-fs/2, fs/2, N)
SSB_fft = np.fft.fftshift(np.fft.fft(ssb_analytic)) / N

plt.figure(figsize=(12, 5))
plt.plot(f_axis, np.abs(SSB_fft), label='Magnitude Spectrum')
plt.title('Spectrum of the Analytic SSB Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Magnitude')
plt.legend()
plt.grid(True)
plt.show()
