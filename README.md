# STS-map

A desktop GUI tool for analysing **scanning tunnelling spectroscopy (STS) line-scan data** (Omicron `.dat` format).
It plots dI/dV maps, automatically detects the conduction band minimum (Ec / CBM) and valence band maximum (Ev / VBM), and calculates the band gap Eg along the line.

[中文使用说明](使用说明_中文.txt) · [English User Guide](User_Guide_EN.txt)

## Features

- dI/dV map of a full line scan, with optional log scale and quick normalisation (gN)
- Four analysis modes:
  | Mode | Output |
  |---|---|
  | `Ec_log` | Conduction band edge Ec only (positive-bias data) |
  | `newBG` | Ec, Ev and band gap Eg |
  | `line` | Stacked dI/dV curves, one per point |
  | `Line CBM VBM` | Stacked curves with CBM / VBM marked |
- Manual correction of Ec / Ev in the results table, with live re-plotting
- Export figures (SVG / PDF / PNG) and results (TXT / CSV)
- Chinese / English interface

## Download (Windows)

No Python needed: download `STS_map.exe` from the [Releases](https://github.com/RinoWu96/STS-map/releases) page and double-click it.

## Run from source

Requires Python 3.9+.

```bash
git clone https://github.com/RinoWu96/STS-map.git
cd STS-map
pip install -r requirements.txt
python STS_map.py
```

### Build the executable yourself

```bash
pip install pyinstaller
pyinstaller STS_map.spec
# output: dist/STS_map.exe
```

## Project structure

```
STS_map.py          # GUI entry point
band_edge_core.py   # Shared band-edge detection and normalisation
Ec_log.py           # Ec_log analysis mode
newBG.py            # newBG analysis mode (Ec, Ev, Eg)
Line_Log.py         # line mode / .dat file reader
Line_CBM_VBM.py     # Line CBM VBM mode
STS_map.spec        # PyInstaller build config
```

## Input data

Select a folder containing the `.dat` files of **one** line scan. The program reads the column headers and tries to pick the Bias, Current and LI Demod columns automatically; you can change them in the drop-down lists.

## Citation

If this tool helps your research, please cite:

```
Rino. STS-map: a GUI tool for STS line-scan band-edge analysis. GitHub, 2026. https://github.com/RinoWu96/STS-map
```

## Contact

Rino
