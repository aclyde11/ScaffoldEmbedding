import sys
from rdkit import Chem
import random
from tqdm import tqdm
import multiprocessing


def randomSmiles(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)

def getRandom(data):
    smi1, smi2 = data
    mol1, mol2 = None, None
    try:
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    except KeyboardInterrupt:
        exit()
    except:
        return None
    if mol1 is None or mol2 is None:
        return None

    smiles1, smiles2 = set(), set()
    smiles1.add(smi1)
    smiles2.add(smi2)
    for _ in range(5):
        smiles1.add(randomSmiles(mol1))
        smiles2.add(randomSmiles(mol2))
    lsize = min(len(smiles1), len(smiles2))
    return list(smiles1)[:lsize], list(smiles2)[:lsize]

if __name__ == '__main__':
    filein = sys.argv[1]
    fileout = sys.argv[2]

    with open(filein) as fin:
        with open(fileout, 'w') as fout:
            with multiprocessing.Pool(8) as p:
                iterm = map(lambda x : x.strip().split('\t'), fin)
                resm = p.imap_unordered(getRandom, iterm)
                for res in tqdm(resm):
                    if res is not None:
                        for i in range(len(res[0])):
                            fout.write(f"{res[0][i]}\t{res[1][i]}\n")
