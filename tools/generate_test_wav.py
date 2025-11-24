#!/usr/bin/env python3
"""Generate a 2.5s 16kHz mono WAV that can be used to test /predict endpoints.

Usage:
  python3 tools/generate_test_wav.py sample.wav --duration 2.5

Produces a simple voiced-like chirp with modest amplitude.
"""
import argparse
import numpy as np
from scipy.io import wavfile

DEFAULT_SR = 16000


def generate_chirp(duration=2.5, sr=DEFAULT_SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # chirp from 120Hz to 400Hz (voiced-like), plus light noise
    freqs = np.linspace(120.0, 400.0, t.size)
    sig = 0.6 * 0.5 * (np.sin(2 * np.pi * freqs * t))
    # add gentle amplitude modulation to mimic syllables
    amp_env = 0.5 * (1.0 + 0.4 * np.sin(2 * np.pi * 2.5 * t))
    sig = sig * amp_env
    # add small broadband noise
    sig = sig + 0.01 * np.random.randn(t.size)
    # normalize to int16 range
    sig = sig / max(1e-9, np.max(np.abs(sig))) * 0.95
    return (sig * 32767).astype(np.int16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', help='output wav path')
    p.add_argument('--duration', type=float, default=2.5)
    p.add_argument('--sr', type=int, default=DEFAULT_SR)
    args = p.parse_args()
    data = generate_chirp(args.duration, args.sr)
    wavfile.write(args.out, args.sr, data)
    print(f"Wrote {args.out} ({args.duration}s @ {args.sr}Hz)")

if __name__ == '__main__':
    main()
