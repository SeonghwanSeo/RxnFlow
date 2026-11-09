"""SMILES cleaning with RDKit and ChEMBL salt/solvent patterns."""

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def _load_patterns(entries):
    patterns = {}
    for smarts in entries:
        query = Chem.MolFromSmarts(smarts)
        size = (query.GetNumAtoms(), query.GetNumBonds())
        patterns.setdefault(size, []).append(query)
    return patterns


def _matches_component(fragment, patterns):
    size = (fragment.GetNumAtoms(), fragment.GetNumBonds())
    return any(fragment.HasSubstructMatch(query) for query in patterns.get(size, ()))


class ParentCleaner:
    """Prepare scoring inputs and extract parents using frozen removal rules."""

    def __init__(self):
        rules = json.loads(Path(__file__).with_name("cleaning.json").read_text())
        self.solvents = _load_patterns(rules["solvents"])
        self.salts = _load_patterns(rules["salts"])
        self.uncharger = rdMolStandardize.Uncharger(canonicalOrder=True)

    def clean(self, smiles):
        """Parse, remove atom maps, and return a sanitized single parent.

        Invalid or empty SMILES return None; unresolved mixtures raise ValueError.
        """
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            return None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        parent = self.fragment_parent(molecule)
        Chem.SanitizeMol(parent)
        if len(Chem.GetMolFrags(parent)) != 1:
            raise ValueError("unresolved mixture of distinct parents")
        return parent

    def neutralize(self, molecule):
        parent = self.uncharger.uncharge(molecule)
        parent.UpdatePropertyCache(strict=False)
        return parent

    def _component_identity(self, fragment):
        comparison = Chem.RemoveHs(self.neutralize(fragment), sanitize=False)
        comparison.ClearComputedProps()
        comparison.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            comparison,
            sanitizeOps=Chem.SANITIZE_SYMMRINGS
            | Chem.SANITIZE_FINDRADICALS
            | Chem.SANITIZE_SETAROMATICITY
            | Chem.SANITIZE_ADJUSTHS,
        )
        return Chem.MolToSmiles(comparison, canonical=True, isomericSmiles=True)

    def fragment_parent(self, molecule):
        originals = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
        if not originals:
            raise ValueError("Cannot extract a parent from an empty molecule")
        matching = []
        for original in originals:
            fragment = Chem.RemoveHs(original, sanitize=False)
            fragment.UpdatePropertyCache(strict=False)
            Chem.SetAromaticity(fragment)
            matching.append(fragment)

        # Roll back a stage if its removal list would remove every component.
        remaining = [
            i
            for i, fragment in enumerate(matching)
            if not _matches_component(fragment, self.solvents)
        ]
        if not remaining:
            return self.neutralize(molecule)
        retained = [
            i for i in remaining if not _matches_component(matching[i], self.salts)
        ]
        if not retained:
            retained = remaining

        # Collapse repeated copies only when all surviving components have the
        # same neutralized isomeric identity. Distinct components stay unresolved.
        identities = {self._component_identity(originals[i]) for i in retained}
        parent = Chem.Mol(originals[retained[0]])
        if len(identities) > 1:
            for i in retained[1:]:
                parent = Chem.CombineMols(parent, originals[i])
        return self.neutralize(parent)
