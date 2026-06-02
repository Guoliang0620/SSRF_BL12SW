# 2theta Fitting Tool

PyQt5 desktop tool for SSRF BL12SW energy-dispersive XRD data. It imports two-column text data, applies channel-to-energy calibration, fits selected ROIs with a Gaussian plus linear baseline, and estimates fixed 2theta from `d` and `E` pairs using Bragg's law.

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python "2theta SSRF_250404_final.py"
```

On Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python "2theta SSRF_250404_final.py"
```

## Build Locally

PyInstaller builds must be created on the target operating system. Build Windows packages on Windows, and macOS packages on macOS.

```bash
python -m pip install -r requirements.txt
pyinstaller --noconfirm --clean --windowed --name "2theta-fitting-tool" "2theta SSRF_250404_final.py"
```

Artifacts are written under `dist/`.

## GitHub Actions

The workflow in `.github/workflows/build.yml` can build macOS and Windows artifacts in GitHub Actions. Push this folder to a GitHub repository, then run the workflow manually or push a version tag like `v0.1.0`.

## Notes

- `Skip last row` is enabled by default for SSRF-style files that include a non-data trailer row. Disable it when the last row is real data.
- 2theta calibration now validates positive `d/E` values and the Bragg-law arcsin domain.
- Fit results are quick-look peak fits, not a full profile refinement.
- `SSRFtest/` contains small raw example files for manual import testing. Converted outputs are generated files and should not be committed.
- A future fitting upgrade can add pseudo-Voigt/Voigt profiles, but the current release keeps Gaussian + linear baseline for predictable quick-look behavior.
