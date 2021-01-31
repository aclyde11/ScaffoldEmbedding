from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import multiprocessing as mp
from multiprocessing import Pool
from rdkit import RDLogger

import sys


# def otherfrag(smi):
#     mol = oechem.OEGraphMol()
#     oechem.OESmilesToMol(mol, smi)
#
#     adjustHCount = True
#     for frag in oemedchem.OEGetBemisMurcko(mol):
#         fragment = oechem.OEGraphMol()
#         oechem.OESubsetMol(fragment, mol, frag, adjustHCount)
#         print(".".join(r.GetName() for r in frag.GetRoles()), oechem.OEMolToSmiles(fragment))
#
#
# def fragsmiles(smi):
#     mol = oechem.OEGraphMol()
#     oechem.OESmilesToMol(mol, smi)
#     options = oemedchem.OEBemisMurckoOptions()
#     options.SetUnsaturatedHeteroBonds(True)
#
#     adjustHCount = True
#
#     result = {'original' : smi}
#     for frag in oemedchem.OEGetBemisMurcko(mol, options):
#         fragment = oechem.OEGraphMol()
#         oechem.OESubsetMol(fragment, mol, frag, adjustHCount)
#         fsmi = oechem.OEMolToSmiles(fragment)
#         for r in frag.GetRoles():
#             name = r.GetName()
#             if r.GetName() in result:
#                 result[name].append(fsmi)
#             else:
#                 result[name] = [fsmi]
#     return result

def getscaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smi = Chem.MolToSmiles(scaffold)
    smi = Chem.MolToSmiles(mol)
    if len(scaffold_smi) == 0:
        scaffold_smi = '()'
    return smi, scaffold_smi

if __name__ == '__main__':

    infile = sys.argv[1]
    outfile = sys.argv[2]

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    mp.set_start_method('fork')

    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            smilesIncoming = map(lambda x : x.strip(), fin)
            with Pool(64) as pool:
                outputIter = pool.imap_unordered(getscaffold, smilesIncoming)
                for output in tqdm(outputIter):
                    if output is not None:
                        s1, scaffold = output
                        fout.write(f"{s1}\t{scaffold}\n")
