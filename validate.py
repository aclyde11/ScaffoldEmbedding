from rdkit import Chem
import rdkit.Chem.Scaffolds.MurckoScaffold
from rdkit import RDLogger
from tqdm import tqdm

import multiprocessing

RDLogger.DisableLog('rdApp.info')

def mol_equal(s,t):
    return s.HasSubstructMatch(t) and t.HasSubstructMatch(s)

def check_scaffold(s,t):
    s = Chem.MolFromSmiles(s)
    t = Chem.MolFromSmiles(t)

    if s is None or t is None:
        return -1

    scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(s)

    if mol_equal(scaffold, t):
        return 1
    else:
        return 0

def check_lower(s,t):
    s = Chem.MolFromSmiles(s)
    t = Chem.MolFromSmiles(t)

    if s is None or t is None:
        return -1
    if t.HasSubstructMatch(s):
        return 1
    else:
        return 0

def check_upper(s,t):
    s = Chem.MolFromSmiles(s)
    t = Chem.MolFromSmiles(t)

    if s is None or t is None:
        return -1
    if s.HasSubstructMatch(t):
        return 1
    else:
        return 0
    
def check_expand(s,t):
    s = Chem.MolFromSmiles(s)
    t = Chem.MolFromSmiles(t)

    if s is None or t is None:
        return -1
    scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(t)

    if mol_equal(scaffold, s):
        return 1
    else:
        return 0
    

def get_result(t):
    s,t = t
    op = s.split(" ")
    s = " ".join(op[1:])
    op = op[0]
    s = s.strip().replace(' ', '')
    t = t.strip().replace(' ', '')

    if op == 'SCAFFOLD':
        return 's', check_scaffold(s, t)
    elif op == 'EXPAND':
        return 'e', check_expand(s, t)
    elif op == 'LOWER':
        return 'l', check_lower(s, t)
    elif op == 'UPPER':
        return 'u', check_upper(s, t)
    else:
        print(f"Error {op} did not match anything. The whole line is {s, t}")
        exit()

def main(fsrc, ftgt, threads=4):

    total = 0
    total_t = {'l':0, 'u': 0, 'e': 0, 's': 0}
    correct_t = {'l':0, 'u': 0, 'e': 0, 's': 0}
    correct = 0
    correct_syntax = 0
    correct_syntax_t = {'l':0, 'u': 0, 'e': 0, 's': 0}
    with open(fsrc, 'r') as src:
        with open(ftgt, 'r') as tgt:
            with multiprocessing.Pool(threads) as p:
                iterr = p.imap(get_result, zip(src, tgt))
                pbar = tqdm(iterr, postfix="")
                for idx, (t,r) in enumerate(pbar):
                    total_t[t] += 1
                    total += 1
                    if r == 0: #could parse, but not right
                        correct_syntax += 1
                        correct_syntax_t[t] =+ 1
                    if r > 0:
                        correct += 1
                        correct_t[t] += 1
                if idx > 0 and  idx % 1000 == 0:
                    pbar.set_postfix(f"Total {total}, Syntax good {correct_syntax} ({correct_syntax / total}%), Correct {correct} ({correct / total}%)")
        print(f"Total {total}, Syntax good {correct_syntax} ({correct_syntax /total}%), Correct {correct} ({correct/total}%)")
        for key in total_t.keys():
            print(f"{key}: {correct_t[key]} ({0 if total_t[key] == 0 else (correct_t[key] / total_t[key])}%) ({0 if correct_syntax_t[key] == 0 else (correct_t[key] / correct_syntax_t[key])}%))")

if  __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', type=str, required=True)
    parser.add_argument('-tgt', type=str, required=True)
    parser.add_argument('-threads', type=int, required=False, default=4)
    args = parser.parse_args()

    main(args.src, args.tgt, args.threads)