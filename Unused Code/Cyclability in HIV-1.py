#%%
''' HIV-1 '''
import random
import re
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from Bio import SeqIO      
from Bio.Seq import Seq
from scipy.stats import pearsonr
from scipy.stats import linregress
import csv

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
'''Genes'''
df_HIV = pd.read_table("HIV-1 genes.tsv", delimiter= "\t")

'''Genome'''
fasta_file = "HIV-1 genome.fna"
for record in SeqIO.parse(fasta_file, "fasta"):
    HIV_genome = str(record.seq)

#%%
def sequencer1000(position, direction, chromosome):
    position = position - 1
    Left_pos = position - 1000
    Right_pos = position + 1001 
    Nucleosome_sequence = chromosome[Left_pos:Right_pos]
    if direction == "+":
        Sequence = Nucleosome_sequence
    elif direction == "-":
        Nucleosome_seq_obj = Seq(Nucleosome_sequence)
        compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
        Sequence = str(compliment_Nuclesome)
    return Sequence

def average_Nucleosome(Nucleosomes):
    average_cyclability = np.array(Nucleosomes)
    average_cyclability = np.average(Nucleosomes, axis=0)
    return average_cyclability

'''
Loading Model and Functions
'''
def generate_random_dna_sequence(length):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(bases) for _ in range(length))

def pred(model, pool):
    input = np.zeros((len(pool), 200), dtype = np.single)
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50): 
            input[i][j*4 + temp[pool[i][j]]] = 1
    A = model.predict(input, batch_size=128, verbose = 0)
    A.resize((len(pool),))
    return A

def load_model(modelnum: int):
    return keras.models.load_model(f"./adapter-free-Model/C{modelnum}free")


def run_model(seqs):
    accumulated_cyclability = []
    option = "C0free prediction"
    modelnum = int(re.findall(r'\d+', option)[0])
    model = load_model(modelnum)
    x =1
    for seq in seqs:
        if x%200 == 0:
            print(x)
        x = x+1
        
        list50 = [seq[i:i+50] for i in range(len(seq) - 50 + 1)]
        cNfree = pred(model, list50)
        prediction = list(cNfree)
        accumulated_cyclability.append(prediction)
    
    return accumulated_cyclability

def plot_cyclability1000(values):
    x_values = np.linspace(-975, 975, len(values))  
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, values)
    plt.xlabel('Distance from TSS (BP)')
    plt.ylabel('Cyclability')
    plt.title('Cyclability Around TSS in HIV-1')
    plt.xlim(-1000, 1000) 
    plt.ylim(-0.27, 0)
    plt.show()
    
def sliding_window_average(arr, window_size=50):
    result = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i+window_size]
        result.append(np.mean(window))
    return np.array(result)

def plot_sliding_window_average(data, window_size=50):
    averaged_data = sliding_window_average(data, window_size)
    x_avg_values = np.linspace(-1000, 1000, len(averaged_data))
    plt.figure(figsize=(12, 6))
    plt.plot(x_avg_values, averaged_data, label='Sliding Window Average', linestyle='--')
    plt.ylim([min(data), max(data)])
    plt.ylim(-0.8, 0.95)
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')
    plt.title('Cyclability Around TSS in HIV-1')
    plt.legend()
    plt.show()
    
#%%
TSS_positions = df_HIV["Begin"]
TSS_sequences = []
for position in TSS_positions:
    sequence = sequencer1000(position, "+", HIV_genome)
    TSS_sequences.append(sequence)
    
#%%
HIV_cyclabilities = run_model(TSS_sequences)
HIV_cyclability = average_Nucleosome(HIV_cyclabilities)

plot_sliding_window_average(HIV_cyclability)

#%%
for item in HIV_cyclabilities:
    plot_sliding_window_average(item)

