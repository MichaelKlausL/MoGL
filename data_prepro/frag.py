# This script fragments molecules using the BRICS algorithm
from rdkit import Chem
from rdkit.Chem import BRICS

def brics_fragments_with_star(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    raw_frags = list(BRICS.BRICSDecompose(mol))
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



smiles = "O=C(O)C1=NC(C(=O)O)CC=C1"
fragments = brics_fragments_with_star(smiles)
print(fragments)
