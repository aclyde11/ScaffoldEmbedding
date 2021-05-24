from rdkit import Chem
import rdkit.Chem.Scaffolds.MurckoScaffold
import scaffoldgraph as sg
from tqdm import tqdm
import multiprocessing
import random

def getlines(filename):
    with open(filename, 'r') as f:
        counter = 0
        for _ in f:
            counter += 1
    return counter

def randomSmiles(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)

def getRset(mol1, lim=3):
    s = set()
    s.add(Chem.MolToSmiles(mol1))
    for i in range(lim):
        s.add(randomSmiles(mol1))
    return list(s)

def generate_ops(smi):
    smi = smi.strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"None value: {smi}\n")
        return []
    scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    frags = sg.get_next_murcko_fragments(scaffold)

    smi = getRset(mol)
    scaffold_smi = getRset(scaffold)

    data = []
    for smi_ in smi:
        for scaffold_smi_ in scaffold_smi:
            data += [('SCAFFOLD', smi_, scaffold_smi_),
            ('EXPAND', scaffold_smi_, smi_)]
            for frag in frags:
                for frag_smi in getRset(frag):
                    data.append(('LOWER', scaffold_smi_, frag_smi))
                    data.append(('UPPER', frag_smi, scaffold_smi_))
    return data

if __name__ == '__main__':
    import sys

    filein = sys.argv[1]
    fileout = sys.argv[2]
    linecount = getlines(filein)
    print(f"Loaded {filein} with {linecount} lines.")

    cs = 0
    with open(filein, 'r') as fin:
        with open(fileout, 'w') as fout:
            with multiprocessing.Pool(7) as pool:
                resiter = pool.imap_unordered(generate_ops, fin)
                for data in tqdm(resiter, total=linecount):
                    if len(data) != 0:
                        for op, s1, s2 in data:
                            fout.write(f"{op} {s1} {s2}\n")
                    cs += 1
                    # if cs > 100:
                    #     break
