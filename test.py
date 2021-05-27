import subprocess
from DataPrep import tokenzie_smile
from rdkit import Chem

def make_call(cmd, config):
    with open('t1.txt', 'w') as f:
        op, smiles = cmd.strip().split(' ')
        smiles = tokenzie_smile(smiles)
        smiles = f"{op} {smiles}\n"
        f.write(smiles)
        f.write(smiles)
        f.write(smiles)
        f.write(smiles)


    cmdstr = f"onmt_translate -model {config} -src t1.txt -output t2.txt --batch_size 4 --random_sampling_topk -1".split(" ")
    subprocess.run(cmdstr, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open('t2.txt', 'r') as f:
        results = list(map(lambda x : x.strip().replace(' ', ''), f))
        results_valid = set()
        for result in results:
            m =  Chem.MolFromSmiles(result)
            if m is not None:
                results_valid.add(Chem.MolToSmiles(m))
    return results_valid


def main(config):


    #x = input()
    x = '[SCAFFOLD] O=C(O)c1ccc2c(c1)nc(-c1ccccn1)n2C1CCCCCC1'
    print(f"Input {x}")
    results = make_call(x, config)
    print(f"Results {results}")





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=False)
    args = parser.parse_args()

    main('algebraDataSmall_step_199000.pt')