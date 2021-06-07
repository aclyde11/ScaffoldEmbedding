import multiprocessing

import rdkit.Chem.Scaffolds.MurckoScaffold
import scaffoldgraph as sg
import selfies
from rdkit import Chem
from tqdm import tqdm


def getlines(filename):
    with open(filename, 'r') as f:
        counter = 0
        for _ in f:
            counter += 1
    return counter


def generate_ops(smi):
    smi = smi.strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"None value: {smi}\n")
        return []
    scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    frags = sg.get_next_murcko_fragments(scaffold)

    smi_selfies = selfies.encoder(smi)
    scaffold_selfies = selfies.encoder(Chem.MolToSmiles(scaffold))

    data = []
    data += [('SCAFFOLD', smi_selfies, scaffold_selfies),
             ('EXPAND', scaffold_selfies, smi_selfies)]
    for frag in frags:
        frag_seflies = selfies.encoder(Chem.MolToSmiles(frag))
        data.append(('LOWER', scaffold_selfies, frag_seflies))
        data.append(('UPPER', frag_seflies, scaffold_selfies))
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
            with multiprocessing.Pool(24) as pool:
                resiter = pool.imap_unordered(generate_ops, fin)
                for data in tqdm(resiter, total=linecount):
                    if len(data) != 0:
                        for op, s1, s2 in data:
                            fout.write(f"{op} {s1} {s2}\n")
                    cs += 1
                    # if cs > 100:
                    #     break
