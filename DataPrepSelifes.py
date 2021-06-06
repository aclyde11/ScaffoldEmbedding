import os
import random
import re

import selfies
from rdkit import Chem
from tqdm import tqdm

import scaffoldgraph as sg


def randomize_smi(smi):
    random_equivalent_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(smi, doRandom=True))
    return random_equivalent_smiles



class SmileTokenizer():
    def __init__(self):
        pass

    def __call__(self, smi):
        tokens = selfies.split_selfies(smi)
        return ' '.join(tokens)

def randomSmiles(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


def tokenzie_smile(smi):
    tokens = selfies.split_selfies(smi)
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
                                try:
                                    op , molecule, scaffold = line.strip().split(" ")
                                except ValueError as e:
                                    print(e, f"Failed wit this line: {line}")
                                    continue
                                if random.random() > test_size:  # goes into train
                                    molecule_tokens, scaffold_tokens = tokenizer(molecule), tokenizer(scaffold)
                                    if len(molecule_tokens) != 0 and len(scaffold_tokens) != 0:
                                        src_train.write(f"{op} {molecule_tokens}\n")
                                        tgt_train.write(f"{scaffold_tokens}\n")
                                else:
                                    molecule_tokens, scaffold_tokens = tokenizer(molecule), tokenizer(scaffold)
                                    if len(molecule_tokens) != 0 and len(scaffold_tokens) != 0:
                                        src_val.write(f"{op} {molecule_tokens}\n")
                                        tgt_val.write(f"{scaffold_tokens}\n")


# try:
#     mol_molcule, mol_scaffold = Chem.MolFromSmiles(
#         molecule), Chem.MolFromSmiles(scaffold)
# except:
#     continue
# if mol_molcule is None or mol_scaffold is None:
#     continue
# for i in range(20):
#     molecule, scaffold = randomSmiles(mol_molcule), randomSmiles(mol_scaffold)

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
