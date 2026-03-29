# Time-Aligned Partitioned Block Frequency Domain Adaptive Filtering for Robust Acoustic Echo Cancellation

Modular implementation of Stage-1 acoustic echo cancellation using:

- GCC-PHAT time alignment
- Partitioned Block Frequency-Domain LMS (PBFDLMS/MDF-style)
- MSC-based step-size control
- Error magnitude constraint with beta=2

## Project Structure

```text
PBFDLMS-based-AEC/
	main.py                  # Entry point
	aec/
		__init__.py
		cli.py                 # Command-line arguments
		audio.py               # Audio loading/resampling/writing
		delay.py               # GCC-PHAT delay estimator
		stft.py                # STFT helper
		filter.py              # TimeAlignedPBFDAF class
		metrics.py             # Evaluation metrics
		plotting.py            # Waveform plotting
		pipelines.py           # Synthetic + real-audio runners
		utils.py               # Shared numerical helpers
```

## Run

Synthetic demo:

```bash
python3 main.py --demo
```

Real-audio restoration:

```bash
python3 main.py --real-audio --input-audio Myvoice.aac --seconds 7 --output-prefix myvoice_test
```