import os
import random
import re

from rdkit import Chem
from tqdm import tqdm


def randomize_smi(smi):
    random_equivalent_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(smi, doRandom=True))
    return random_equivalent_smiles


class SmileTokenizer():
    def __init__(self):
        self.pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def __call__(self, smi):
        tokens = [token for token in self.regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)

def randomSmiles(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


def tokenzie_smile(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def train_test_split(datafile, file_base_paths='data', sampler=1.0, test_size=0.0002, rseed=42, total=-1):
    if not os.path.exists(file_base_paths):
        os.makedirs(file_base_paths)

    random.seed(rseed)
    tokenizer = SmileTokenizer()
    with open(datafile, 'r') as fin:
        with open(f"{file_base_paths}/src-train.txt", 'w') as src_train:
            with open(f"{file_base_paths}/tgt-train.txt", 'w') as tgt_train:
                with open(f"{file_base_paths}/src-val.txt", 'w') as src_val:
                    with open(f"{file_base_paths}/tgt-val.txt", 'w') as tgt_val:
                        for line in tqdm(fin, total=total if total != -1 else None):
                            if random.random() <= sampler:
                                molecule, scaffold = line.strip().split("\t")
                                if random.random() > test_size:  # goes into train
                                    try:
                                        mol_molcule, mol_scaffold = Chem.MolFromSmiles(
                                            molecule), Chem.MolFromSmiles(scaffold)
                                    except:
                                        continue
                                    if mol_molcule is None or mol_scaffold is None:
                                        continue
                                    for i in range(5):
                                        molecule, scaffold = randomSmiles(mol_molcule), randomSmiles(mol_scaffold)
                                        molecule_tokens, scaffold_tokens = tokenizer(molecule), tokenizer(scaffold)

                                    src_train.write(f"{scaffold_tokens}\n")
                                    tgt_train.write(f"{molecule_tokens}\n")
                                else:
                                    molecule_tokens, scaffold_tokens = tokenizer(molecule), tokenizer(scaffold)
                                    src_val.write(f"{scaffold_tokens}\n")
                                    tgt_val.write(f"{molecule_tokens}\n")


def getgargs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='smiles tab separated', type=str, required=True)
    parser.add_argument('-o', help='output folder', type=str, required=True)
    parser.add_argument('--sampler', type=float, required=True)
    parser.add_argument('--test_size', type=float, required=True)
    parser.add_argument('--total', type=int, required=False, default=-1)
    return parser.parse_args()

if __name__ == '__main__':
    args = getgargs()
    filename = args.i
    train_test_split(filename, file_base_paths=args.o, sampler=args.sampler, test_size=args.test_size, total=args.total)
