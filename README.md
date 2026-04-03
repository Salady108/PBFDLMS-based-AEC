# PBFDLMS-based Acoustic Echo Cancellation (AEC)

A robust, modular implementation of **acoustic echo cancellation** using **time-aligned Partitioned Block Frequency-Domain Adaptive Filtering (PBFDLMS)** with GCC-PHAT delay estimation.

![AEC Pipeline](https://img.shields.io/badge/DSP-Audio%20Processing-blue) ![Python](https://img.shields.io/badge/python-3.8+-green)

## 📋 Project Overview

This project implements a acoustic echo cancellation system designed for real-time and offline audio processing. It addresses the challenge of removing echo from speech signals by combining signal processing techniques:

- **GCC-PHAT**: Generalized Cross-Correlation with Phase Transform for robust delay estimation
- **PBFDLMS**: Partitioned Block Frequency-Domain LMS adaptive filtering for efficient echo path modeling
- **Adaptive Step-Size Control**: MSC-based (Magnitude-Squared Coherence) dynamic adjustment
- **Error Constraint**: Beta-constraint mechanism for stability

## ✨ Features

- **Dual Operation Modes**:
  - Synthetic demo with controllable echo/noise
  - Real-audio restoration from AAC/WAV files
- **Robust Delay Estimation**: GCC-PHAT algorithm handles misalignment automatically
- **Efficient Adaptive Filtering**: Frequency-domain LMS reduces computational complexity
- **Configurable Parameters**: Full CLI control over algorithm hyperparameters
- **Audio Visualization**: Generated waveform plots comparing input/output
- **Modular Architecture**: Clean separation of concerns for extensibility

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PBFDLMS-based-AEC
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy pydub imageio-ffmpeg
   ```

   **Or install individually:**
   ```bash
   # Core numerical computing
   pip install numpy
   
   # Audio file I/O and format conversion
   pip install pydub
   
   # Audio codec support (AAC, MP3, etc.)
   pip install imageio-ffmpeg
   ```

## 🚀 Quick Start

### Run Synthetic Demo

Execute a demonstration with synthetic clean speech, injected echo, and additive noise:

```bash
python3 main.py --demo
```

This will:
- Generate synthetic speech signal
- Add controlled echo and noise
- Apply AEC filtering
- Output performance metrics and waveform plot

### Process Real Audio File

Restore a clean audio file by removing echo artifacts:

```bash
python3 main.py --real-audio --input-audio your_audio.aac --seconds 7 --output-prefix output_name
```

**Generated outputs:**
- `output_name_clean_reference.wav` - Recovered clean signal
- `output_name_noisy_echo.wav` - Input with echo and noise
- `output_name_recovered_clean.wav` - Algorithm output
- `output_name_waveforms.png` - Visualization

## 📖 Usage Examples

### Example 1: Custom Filter Length
```bash
python3 main.py --demo --filter-len 4800 --block-len 160
```

### Example 2: Different Noise Level
```bash
python3 main.py --demo --noise-db -20 --seed 42
```

### Example 3: Real Audio with Parameters
```bash
python3 main.py --real-audio \
  --input-audio speech.aac \
  --seconds 10 \
  --echo-delay-ms 100 \
  --echo-decay 0.5 \
  --output-prefix my_result
```

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--demo` | flag | - | Run synthetic AEC demo |
| `--real-audio` | flag | - | Process real audio file |
| `--input-audio` | str | `Myvoice.aac` | Path to input audio file |
| `--output-prefix` | str | `myvoice` | Prefix for output files |
| `--fs` | int | 16000 | Sampling frequency (Hz) |
| `--seconds` | float | 5.0 | Duration for processing |
| `--block-len` | int | 160 | Block/hop length samples |
| `--filter-len` | int | 3840 | Adaptive filter length |
| `--mu` | float | 0.5 | Step-size parameter |
| `--beta` | float | 2.0 | Error magnitude constraint |
| `--max-delay-ms` | float | 700.0 | Max delay for GCC-PHAT (ms) |
| `--noise-db` | float | -30.0 | Noise level (dB) |
| `--seed` | int | 0 | Random seed for reproducibility |

## 📁 Project Structure

```
PBFDLMS-based-AEC/
├── main.py                          # Entry point / CLI dispatcher
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── aec/                             # Core algorithm package
    ├── __init__.py                  # Package exports
    ├── cli.py                       # Command-line argument parser
    ├── audio.py                     # Audio I/O and resampling utilities
    ├── delay.py                     # GCC-PHAT delay estimation algorithm
    ├── stft.py                      # Short-Time Fourier Transform utilities
    ├── filter.py                    # TimeAlignedPBFDAF main filter class
    ├── metrics.py                   # Performance metrics (SNR, PESQ, etc.)
    ├── plotting.py                  # Waveform visualization
    ├── pipelines.py                 # Synthetic demo & real-audio workflows
    └── utils.py                     # Shared numerical/signal helpers
```

### Module Descriptions

- **main.py**: Command-line entry point that routes to synthetic or real-audio modes
- **cli.py**: Argument parser with all configurable hyperparameters
- **audio.py**: Audio file loading, resampling, and WAV writing
- **delay.py**: GCC-PHAT implementation for robust echo path delay detection
- **stft.py**: STFT/iSTFT kernels for frequency-domain processing
- **filter.py**: Core `TimeAlignedPBFDAF` class implementing the adaptive filter
- **metrics.py**: SNR, ERLE, and other evaluation metrics
- **plotting.py**: Matplotlib-based waveform visualization
- **pipelines.py**: High-level workflows for demo and real-audio restoration
- **utils.py**: Numerical utilities (correlation, windowing, etc.)

## 💻 Software Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | Latest | Numerical computing, signal processing |
| **pydub** | Latest | Audio file I/O and format conversion |
| **imageio-ffmpeg** | Latest | Audio codec support (AAC, MP3, etc.) |
| **matplotlib** | (optional) | Waveform plotting and visualization |

### System Requirements

- Python 3.8+
- ~200 MB disk space for virtual environment
- Multi-core CPU recommended for real-time processing

## 🔧 Algorithm Details

### AEC Pipeline

1. **Delay Estimation** (delay.py):
   - GCC-PHAT computes cross-correlation
   - Identifies time alignment between reference and microphone signals
   - Searches within configurable max-delay window

2. **Time Alignment**:
   - Reference signal is time-shifted to align with echo path
   - Improves convergence of adaptive filter

3. **Partitioned Block Frequency-Domain LMS**:
   - Divides filter into P partitions for efficiency
   - Processes in frequency domain (FFT-based)
   - Reduces complexity from O(N²) to O(N log N)
   - Faster convergence than time-domain LMS

4. **Adaptive Step-Size**:
   - MSC-based control adjusts learning rate dynamically
   - Prevents instability and divergence
   - Improves convergence speed

5. **Error Constraint**:
   - Beta constraint: |Error| ≤ β|Desired|
   - Prevents filter divergence
   - Maintains perceptual quality

## 📊 Output & Results

The system generates:

1. **Audio Files** (WAV format):
   - Clean reference signal
   - Noisy+echo corrupted version
   - Recovered/enhanced output

2. **Metrics** (printed to console):
   - SNR (Signal-to-Noise Ratio)
   - ERLE (Echo Return Loss Enhancement)
   - MSE (Mean Squared Error)

3. **Visualization**:
   - 3-panel waveform plot showing input/output/recovery



## 📚 References

- [1] Peng et al., **ICASSP 2021 AEC Challenge**, Proc. ICASSP, pp. 146-150, 2021.
- [2] Eneman and Moonen, **Iterated partitioned block frequency-domain AEC**, IEEE Trans. Speech Audio Process., 11(2):143-158, 2003, doi:10.1109/TSA.2003.809194.
- [3] Knapp and Carter, **Generalized correlation for time-delay estimation**, IEEE Trans. Acoust., Speech, Signal Process., 24(4):320-327, 1976, doi:10.1109/TASSP.1976.1162830.
- [4] Enzner and Vary, **Frequency-domain adaptive Kalman filter for AEC**, Signal Processing, 86(6):1140-1156, 2006.
- [5] Farhang-Boroujeny, **Adaptive Filters: Theory and Applications**, Wiley, 2013.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## 💡 Tips & Troubleshooting

### Audio File Issues
- **Supported formats**: AAC, MP3, WAV (via pydub + imageio-ffmpeg)
- **Encoding**: Ensure audio is 16-bit PCM mono or stereo
- **Sample rate**: Auto-resamples to target fs (default 16 kHz)


---

**Project**: EC208 DSP - Theory Project  

## Authors

- [Sri Prahlad Mukunthan](https://github.com/Salady108)
- [Vamshikrishna V Bidari](https://github.com/VamshikrishnaBidari)
