# =========================================
# Configuration (set all options here)
# =========================================

INPUT_PATH = "/home/hejiawei/data/ChEBI-20_data/valid.txt"
OUTPUT_JSONL_PATH = "frag_valid.jsonl"

# =========================================
# Dependency imports
# =========================================

import csv
import json
import os
import signal
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Recap
from chemdataextractor import Document
from tqdm import tqdm

BRICS_TIMEOUT_SECONDS = 5
RECAP_TIMEOUT_SECONDS = 5

# =========================================
# Fragment extraction
# =========================================

def fragments_with_star(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        raw_frags = _recap_decompose_with_timeout(mol)
    except TimeoutError:
        return []
    except Exception:
        return []
    fragments = []

    for frag in raw_frags:
        frag_mol = Chem.MolFromSmiles(frag)
        if frag_mol is None:
            continue

        rw_mol = Chem.RWMol(frag_mol)
        for atom in rw_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(0)
                atom.SetIsotope(0)
                atom.SetFormalCharge(0)

        frag_smiles = Chem.MolToSmiles(rw_mol)
        if frag_mol.GetNumAtoms() >= 1:
            fragments.append(frag_smiles)

    return fragments


def _brics_decompose_with_timeout(mol):
    def _handle_timeout(signum, frame):
        raise TimeoutError("BRICSDecompose timeout")

    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        signal.alarm(BRICS_TIMEOUT_SECONDS)
        return list(BRICS.BRICSDecompose(mol))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def _recap_decompose_with_timeout(mol):
    def _handle_timeout(signum, frame):
        raise TimeoutError("RECAPDecompose timeout")

    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        signal.alarm(RECAP_TIMEOUT_SECONDS)
        tree = Recap.RecapDecompose(mol)
        if tree is None:
            return []
        return list(tree.GetLeaves().keys())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)       

# =========================================
# Keyword (chemical entity) extraction
# =========================================

def extract_keywords(text):
    doc = Document(text)
    keywords = [cem.text for cem in doc.cems]
    # Remove duplicates while preserving order
    return list(dict.fromkeys(keywords))

# =========================================
# Read processed CIDs to support resuming interrupted runs
# =========================================

def load_processed_cids(jsonl_path):
    processed_cids = set()

    if not os.path.exists(jsonl_path):
        return processed_cids

    with open(jsonl_path, "r", encoding="utf-8") as f:
        buffer = ""
        for line in f:
            if line.strip() == "":
                if buffer:
                    try:
                        obj = json.loads(buffer)
                        cid = obj.get("cid")
                        if cid is not None:
                            processed_cids.add(str(cid))
                    except json.JSONDecodeError:
                        pass
                    buffer = ""
            else:
                buffer += line

    return processed_cids

# =========================================
# CSV/TXT -> JSONL (with a progress bar and resume support)
# =========================================

def _read_rows(input_path):
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    delimiter = "\t" if ext in {".txt", ".tsv"} else ","
    with open(input_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _get_value(row, *keys):
    for key in keys:
        if key in row:
            return row.get(key, "")
    return ""


def csv_to_jsonl(input_path, output_jsonl):
    processed_cids = load_processed_cids(output_jsonl)
    reader = _read_rows(input_path)

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        for row in tqdm(reader, desc="Processing molecules"):
            cid = str(_get_value(row, "CID", "cid")).strip()
            smiles = _get_value(row, "SMILES", "smiles").strip()
            description = _get_value(row, "Description", "description").strip()

            if not cid or not smiles or not description:
                continue

            if cid in processed_cids:
                continue  # Skip samples that have already been processed

            fragments = fragments_with_star(smiles)
            keywords = extract_keywords(description)

            sample = {
                "cid": cid,
                "smiles": smiles,
                "description": description,
                "fragments": fragments,
                "keywords": keywords
            }

            out_f.write(
                json.dumps(sample, ensure_ascii=False, indent=2)
            )
            out_f.write("\n\n")  # Add a blank line between samples for readability

# =========================================
# Entry point
# =========================================

if __name__ == "__main__":
    print("[INFO] Start building dataset")
    print(f"[INFO] Input File  : {INPUT_PATH}")
    print(f"[INFO] Output JSONL : {OUTPUT_JSONL_PATH}")

    csv_to_jsonl(INPUT_PATH, OUTPUT_JSONL_PATH)

    print("[INFO] Done.")
