# MolFrag Data Preprocessing Scripts

This directory contains scripts for converting tabular molecular data (`CID / SMILES / Description`) into fragment-enhanced data for training and evaluation.

The main preprocessing steps are:

- Fragmenting molecules with **RDKit BRICS** (`fragments`)
- Extracting chemical entity keywords from descriptions with **ChemDataExtractor** (`keywords`)
- Writing output as individual JSON objects with support for resuming interrupted runs

## Directory Contents

- `build_dataset.py`: Main script that reads the input data and generates the output file
- `count_records.py`: Counts the number of samples in an output file
- `frag.py`: Minimal example of BRICS fragmentation
- `chem.py`: Minimal example of chemical entity extraction with ChemDataExtractor
- `frag_train.jsonl / frag_valid.jsonl / frag_test.jsonl`: Example output files

## Requirements

Python 3.9 or later is recommended.

Install the dependencies, for example:

```bash
pip install rdkit-pypi chemdataextractor tqdm
```

> Note: The recommended RDKit installation method may vary by environment. Installing it with conda is often more reliable.

## Input Data Format

The input and output paths are currently configured through constants at the top of `build_dataset.py`:

```python
INPUT_PATH = "/path/to/your/input.txt"
OUTPUT_JSONL_PATH = "frag_test.jsonl"
```

The input file can be one of the following:

- `.txt` / `.tsv` (tab-separated with `\t`)
- `.csv` (comma-separated with `,`)

It must contain the following columns (the accepted capitalization follows the matching logic in the script):

- `CID` or `cid`
- `SMILES` or `smiles`
- `Description` or `description`

Samples missing any of these fields are skipped.

## Usage

### 1. Generate the Data

First, update `INPUT_PATH` and `OUTPUT_JSONL_PATH` in `build_dataset.py`, and then run:

```bash
python build_dataset.py
```

The script displays a progress bar and writes the processed samples to the output file.

### 2. Count the Samples

Update `INPUT_PATH` in `count_records.py`, and then run:

```bash
python count_records.py
```

## Output Format

Each sample in the output file has the following structure:

```json
{
  "cid": "12345",
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "description": "...",
  "fragments": ["...", "..."],
  "keywords": ["...", "..."]
}
```

The file is written as follows:

- Each sample is serialized as indented JSON (`indent=2`).
- An additional blank line is inserted between samples.

Therefore, the output is more accurately described as **blank-line-separated JSON records**, rather than strict JSON Lines with one object per line.

## Resuming an Interrupted Run

Before processing begins, `build_dataset.py` reads the `cid` values from an existing output file. When the script is run again, it automatically skips samples that have already been processed.

This is useful for:

- Resuming a long-running task after an interruption
- Building a dataset incrementally in batches

## Implementation Details

- BRICS decomposition has a timeout (5 seconds by default). If processing times out or raises an exception, the sample's `fragments` field is set to an empty list so that the full run does not stall.
- Duplicate `keywords` are removed while preserving their original order.

## Troubleshooting

1. **RDKit installation fails**
   - Try installing RDKit in a conda environment.

2. **The output file does not contain one JSON object per line**
   - This is intentional: the current format prioritizes readability.
   - The accompanying `count_records.py` script parses this format correctly.

3. **Some samples have an empty `fragments` list**
   - The SMILES string may be invalid, or BRICS decomposition may have timed out or raised an exception.
