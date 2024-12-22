import numpy as np
import pandas as pd
import pickle
import os


def get_all_filenames(directory):
    entries = os.listdir(directory)
    files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
    return files

def CRISPR_HNN_coding(guide_seq):
    code_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    direction_dict = {'A': 2, 'C': 3, 'G': 4, 'T': 5}

    tlen = 23

    gRNA_list = list(guide_seq)
    pair_code = []
    guide_encoded_integers = np.zeros((tlen,), dtype=int)

    for i in range(len(gRNA_list)):
        gRNA_base_code = code_dict[gRNA_list[i].upper()]
        pair_code.append(gRNA_base_code)
        guide_encoded_integers[i] = direction_dict[gRNA_list[i]]

    guide_encoded_integers = np.insert(guide_encoded_integers, 0, 1)
    pair_code_matrix = np.array(pair_code, dtype=np.float32).reshape(1, 1, 23, 4)

    return pair_code_matrix, guide_encoded_integers


modelfile = 'CRISPR_HNNcoded'
codingname = CRISPR_HNN_coding

directory_path = "./datasets"
if not os.path.exists(f"./{modelfile}"):
    os.makedirs(f"./{modelfile}")

file_names = get_all_filenames(directory_path)
for name in file_names:
    path = f'./datasets/{name}'
    dataname = name.split('.')[0].split("（")[0]
    if os.path.exists(f"./{modelfile}/{dataname}.pkl"):
        continue

    df = pd.read_csv(path)
    X_onehot = []
    X_on = []
    for _, row in df.iterrows():
        print(f"{dataname}  Procing {row['sgRNA']}")
        on_hot_encoded, gRNA_encoded = codingname(row['sgRNA'])
        X_onehot.append(on_hot_encoded)
        X_on.append(gRNA_encoded)

    X_onehot = np.array(X_onehot, dtype=np.float32).reshape((len(X_onehot), 1, 23, 4)).astype('float32')
    X_on = np.array(X_on, dtype=np.int32)
    y = df['indel'].values.astype('float32')

    # Save encoded data and labels into a pickle file
    output_path = f'./{modelfile}/{dataname}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump((X_onehot, X_on, y), f)

    print(f"Encoded data saved to {output_path}")

    X_onehot = []
    for _, row in df.iterrows():
        print(f"{dataname}  Procing {row['sgRNA']}")
        on_hot_encoded = codingname(row['sgRNA'])
        X_onehot.append(on_hot_encoded)

    X_onehot = np.array(X_onehot, dtype=np.float32).reshape((len(X_onehot), 1, 23, 4)).astype('float32')
    y = df['indel'].values.astype('float32')

    output_path = f'./{modelfile}/{dataname}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump((X_onehot, y), f)

    print(f"Encoded data saved to {output_path}")
