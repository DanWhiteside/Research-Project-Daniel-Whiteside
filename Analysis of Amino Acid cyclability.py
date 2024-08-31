'''Amino Acid Research'''
''' Ploting Stuff '''
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
from liftover import ChainFile
import matplotlib.colors as mcolors
from scipy import stats
import seaborn as sns
import colorcet as cc

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
'''Generating random but set 100,000 47 bp sequences'''

def generate_dna_sequence(length):
    nucleotides = ['A', 'C', 'G', 'T']
    sequence = ''.join(random.choice(nucleotides) for _ in range(length))
    return sequence

random_seqs_47 = []
for i in range(100000):
    random.seed(i)
    seq = generate_dna_sequence(47)
    random_seqs_47.append(seq)

if random_seqs_47[0] == "TTAGTTGTGCCGCAGCGAAGTAGTGCTTGAAATATGCGACCCCTAAG":
    print("The random list has remained the same")
    
#%%
codon_table = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 
    'TAT': 'Y', 'TAC': 'Y', 
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 
    'TGT': 'C', 'TGC': 'C', 'TGG': 'W', 
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

def insert_codon(sequence, codon):
    halfway_index = len(sequence) // 2 #Calculating the halfway point of the sequence
    new_sequence = sequence[:halfway_index] + codon + sequence[halfway_index:]
    return new_sequence


def pred(model, pool):
    input = np.zeros((len(pool), 200), dtype = np.single)
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50): 
            input[i][j*4 + temp[pool[i][j]]] = 1
    A = model.predict(input, batch_size=128)
    A.resize((len(pool),))
    return A

def load_model(modelnum: int):
    return keras.models.load_model(f"./adapter-free-Model/C{modelnum}free")


def run_model50bp(seqs):
    
    if not isinstance(seqs, list): #Checking if the input is a list of sequences or a single input
        seqs = [seqs]

    accumulated_cyclability = []

    #Get the model number, here the model for C0
    option = "C0free prediction"
    modelnum = int(re.findall(r'\d+', option)[0])

    #Loading Model
    model = load_model(modelnum)

    # Process each sequence
    for seq in seqs:
        #Counter to keep track of progress
        x = 1
        if x % 200 == 0:
            print(x)
        x += 1
        

        subseq_length = 50
        step_size = subseq_length
        
        #Non-overlapping list of 50 bp sequences made from input
        list50 = [seq[i:i+subseq_length] for i in range(0, len(seq) - subseq_length + 1, step_size)]
        cNfree = pred(model, list50)
        prediction = list(cNfree)
    
        #Add the values to a list
        accumulated_cyclability.append(prediction)
    return accumulated_cyclability


def codon_cyclability(codon, random_seq = random_seqs_47):
    amino_sequences = []
    for i in range(0,len(random_seq)):
        amino_seq = insert_codon(random_seq[i], codon)
        amino_sequences.append(amino_seq)
    amino_sequence = "".join(amino_sequences)
    amino_cyclabilities = run_model50bp(amino_sequence)
    flatten_list = [item for sublist in amino_cyclabilities for item in sublist]
    average_cyclability = sum(flatten_list)/len(flatten_list)
    return average_cyclability, flatten_list


#%%
'''Running for all the codons'''

#1: Calculating the cyclizability for each codon and storing the results
codon_cyclabilities = {}
codon_cyclabilities_raw = {}
for codon in codon_table:
    print(codon)
    avg_cyclability, raw_cyclabilities = codon_cyclability(codon)
    codon_cyclabilities[codon] = avg_cyclability
    codon_cyclabilities_raw[codon] = raw_cyclabilities


#2: Grouping the codons by the amino acids they code for
amino_acid_groups = {}
amino_acid_groups_raw = {}
for codon, amino_acid in codon_table.items():
    if amino_acid not in amino_acid_groups:
        amino_acid_groups[amino_acid] = []
    if amino_acid not in amino_acid_groups_raw:
        amino_acid_groups_raw[amino_acid] = []
    amino_acid_groups[amino_acid].append((codon, codon_cyclabilities[codon]))
    amino_acid_groups_raw[amino_acid].append((codon, codon_cyclabilities_raw[codon]))
    
#%%
#3: Calculating the average flexibility of each amino acid

#This uses the already calculated codon average cyclizability 
amino_acid_flexibilities = {}

for amino_acid, codons in amino_acid_groups.items():
    total_flexibility = sum([cyclability for codon, cyclability in codons])
    average_flexibility = total_flexibility / len(codons)
    amino_acid_flexibilities[amino_acid] = average_flexibility

#Using the raw data
amino_acid_flexibilities_raw = {}

for amino_acid, codons_raw in amino_acid_groups_raw.items():
    #Average the raw cyclabilities for each codon first
    codon_averages = []
    for codon, raw_cyclabilities in codons_raw:
        codon_average = sum(raw_cyclabilities) / len(raw_cyclabilities)
        codon_averages.append(codon_average)
    
    #Then average across all codons for the amino acid
    total_flexibility_raw = sum(codon_averages)
    average_flexibility_raw = total_flexibility_raw / len(codon_averages)
    amino_acid_flexibilities_raw[amino_acid] = average_flexibility_raw

#4: Compare the average of averages vs. the average of raw data
for amino_acid in amino_acid_flexibilities:
    print(f"Amino Acid: {amino_acid}")
    print(f"Average Flexibility (using codon averages): {amino_acid_flexibilities[amino_acid]}")
    print(f"Average Flexibility (using raw data): {amino_acid_flexibilities_raw[amino_acid]}")
    print("\n")

#%%
fig, ax = plt.subplots(figsize=(24, 12))
base_colors = plt.cm.tab20.colors
def adjust_color(base_color, factor):
    return tuple(np.clip(np.array(base_color) * factor, 0, 1))
bar_width = 0.3
group_spacing = 1.0
group_positions = []
x_offset = 0

for i, (amino_acid, codons) in enumerate(amino_acid_groups.items()):
    group_positions.append(x_offset + len(codons) * bar_width / 2)
    num_codons = len(codons)
    x_positions = [x_offset + j * bar_width for j in range(num_codons)]
    
    base_color = base_colors[i % len(base_colors)]
    for j, (codon, cyclability) in enumerate(codons):
        color = adjust_color(base_color, 1 - j * 0.1)  
        ax.bar(x_positions[j], cyclability, width=bar_width, color=color, 
               edgecolor='black', label=amino_acid if j == 0 else "")
        ax.text(x_positions[j], cyclability + 0.01, f'{codon}', 
                ha='center', va='bottom', 
                fontsize=15, color='black', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))
    x_offset += (num_codons * bar_width) + group_spacing

ax.set_xlabel('Amino Acids', fontsize=24)
ax.set_ylabel('Average Cyclizability', fontsize=24)
ax.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')
ax.set_xticks(group_positions)
ax.set_xticklabels(list(amino_acid_groups.keys()), ha='right')
ax.tick_params(axis='both', which='major', labelsize=16)
plt.ylim(-0.24,-0.05)
plt.tight_layout()
plt.subplots_adjust(right=0.8)

plt.show()


#%%
'''Adjusting colour palette'''
import colorsys

base_colors = plt.cm.tab20.colors
def increase_saturation(rgb, factor=1.5):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s = np.clip(s * factor, 0, 1)
    return colorsys.hls_to_rgb(h, l, s)

vibrant_colors = [increase_saturation(color) for color in base_colors]

#%%
'''Boxplots of average amino, no weighting'''
#Ordering amino acids based on codon_table
amino_acid_order = sorted(set(codon_table.values()), key=lambda x: list(codon_table.values()).index(x))

plot_data = []

for amino_acid, codons_raw in amino_acid_groups_raw.items():
    for codon, raw_cyclabilities in codons_raw:
        for value in raw_cyclabilities:
            plot_data.append([amino_acid, value])

df = pd.DataFrame(plot_data, columns=["Amino Acid", "Cyclizability"])
df['Amino Acid'] = pd.Categorical(df['Amino Acid'], categories=amino_acid_order, ordered=True)
plt.figure(figsize=(12, 8))
sns.boxplot(x="Amino Acid", y="Cyclizability", data=df, showfliers=False, palette=vibrant_colors)
plt.title('Boxplot of Flexibility for Each Amino Acid')
plt.xlabel('Amino Acid')
plt.ylabel('Cyclizability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
'''Weighting time'''
import pickle
with open("Codon Percentages Pombe", "rb") as file:
    Pombe_weights = pickle.load(file)

with open("Codon Percentages Cerevisiae", "rb") as file:
    Cerevisiae_weights = pickle.load(file)

def calculate_weighted_flexibility_and_store_raw(codon_cyclabilities_raw, codon_percentages):
    weighted_amino_acid_flexibilities = {}
    raw_weighted_values = {}
    for amino_acid, codons in codon_percentages.items():
        weighted_sum = 0
        raw_values = []
        for codon, percentage in codons.items():
            codon_average_flexibility = sum(codon_cyclabilities_raw[codon]) / len(codon_cyclabilities_raw[codon])
            weighted_value = codon_average_flexibility * (percentage / 100)
            weighted_sum += weighted_value
            raw_values.extend([flexibility * (percentage / 100) for flexibility in codon_cyclabilities_raw[codon]])
        weighted_amino_acid_flexibilities[amino_acid] = weighted_sum
        # Store the raw weighted values for the amino acid
        raw_weighted_values[amino_acid] = raw_values

    return weighted_amino_acid_flexibilities, raw_weighted_values

# Example usage:
# Assuming codon_cyclabilities_raw and codon_percentages are defined
weighted_flexibilities_Pombe, raw_weighted_Pombe = calculate_weighted_flexibility_and_store_raw(codon_cyclabilities_raw, Pombe_weights)
weighted_flexibilities_Cerevisiae, raw_weighted_Cerevisiae = calculate_weighted_flexibility_and_store_raw(codon_cyclabilities_raw, Cerevisiae_weights)


#%%
'''Plotting Boxplots'''
def plot_boxplot_of_flexibility(raw_weights, codon_table):
    amino_acid_order = sorted(set(codon_table.values()), key=lambda x: list(codon_table.values()).index(x))
    palette = vibrant_colors
    plot_data = []
    for amino_acid, values in raw_weights.items():
        for value in values:
            plot_data.append([amino_acid, value])
    df = pd.DataFrame(plot_data, columns=["Amino Acid", "Weighted Flexibility"])
    df['Amino Acid'] = pd.Categorical(df['Amino Acid'], categories=amino_acid_order, ordered=True)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Amino Acid", y="Weighted Flexibility", data=df, showfliers=False, palette=palette)
    plt.title('Boxplot of Weighted Flexibility for Each Amino Acid')
    plt.xlabel('Amino Acid')
    plt.ylabel('Weighted Flexibility')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
plot_boxplot_of_flexibility(raw_weighted_Pombe, codon_table)
plot_boxplot_of_flexibility(raw_weighted_Cerevisiae, codon_table)

#%%
def plot_average_flexibility_barplot(average_flexibilities, codon_table, y_range=(-0.2, -0.1)):
    palette = vibrant_colors
    amino_acid_order = sorted(set(codon_table.values()), key=lambda x: list(codon_table.values()).index(x))
    plot_data = []
    for amino_acid in amino_acid_order:
        if amino_acid in average_flexibilities:
            plot_data.append([amino_acid, average_flexibilities[amino_acid]])

    df = pd.DataFrame(plot_data, columns=["Amino Acid", "Average Flexibility"])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Amino Acid", y="Average Flexibility", data=df, palette=palette
                )
    
    plt.xlabel('Amino Acid',fontsize=22, 
               fontname='Verdana')
    plt.ylabel('Average Cyclizability',fontsize=22, 
               fontname='Verdana')
    plt.ylim(y_range)
    
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')

    plt.show()

plot_average_flexibility_barplot(amino_acid_flexibilities, codon_table)
plot_average_flexibility_barplot(weighted_flexibilities_Pombe, codon_table)
plot_average_flexibility_barplot(weighted_flexibilities_Cerevisiae, codon_table)







#%%
'''Storing Codon Average data'''
import pickle

with open('codon_cyclabilities.pkl', 'wb') as file:
    pickle.dump(codon_cyclabilities, file)
    
#%%
'''Building Custom peptides'''
#Amino acids to their codons
codon_dict = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'C': ['TGT', 'TGC'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['TTT', 'TTC'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'K': ['AAA', 'AAG'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'M': ['ATG'],
    'N': ['AAT', 'AAC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC']
}

def generate_custom_peptides(letter, num, possible_letters):
    result = [letter] * num
    while len(result) < 16:
        result.append(random.choice(list(possible_letters.keys())))
    random.shuffle(result)
    return ''.join(result)

def sequences_of_peptide(peptide_sequence, extra_bases, codon_dict = codon_dict):
    sequence_list = []
    for i in range(100000):
        substituted_sequence = [extra_bases]
        for amino_acid in peptide_sequence:
            if amino_acid in codon_dict:
                codons = codon_dict[amino_acid]
                substituted_sequence.append(random.choice(codons))
            else:
                raise ValueError(f"Invalid amino acid: {amino_acid}")
        sequence = "".join(substituted_sequence)
        sequence_list.append(sequence)
    amino_sequence = "".join(sequence_list)
    amino_cyclabilities = run_model50bp(amino_sequence)
    flatten_list = [item for sublist in amino_cyclabilities for item in sublist]
    return flatten_list

def generate_random_dna_sequence(length):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(bases) for _ in range(length))

def random_for_comparison(extra_bases):
    sequence_list = []
    for i in range(100000):
        random_seq = generate_random_dna_sequence(48)
        substituted_sequence = [extra_bases, random_seq]
        sequence = "".join(substituted_sequence)
        sequence_list.append(sequence)
    amino_sequence = "".join(sequence_list)
    amino_cyclabilities = run_model50bp(amino_sequence)
    flatten_list = [item for sublist in amino_cyclabilities for item in sublist]
    return flatten_list

#Reusing the function that entrywise averages an array though not for nucleosomes
def average_Nucleosome(Nucleosomes):
    average_cyclability = np.array(Nucleosomes)
    average_cyclability = np.average(Nucleosomes, axis=0)
    return average_cyclability

#%%
Tyrosine_4 = []
Tyrosine_8 = []
for i in range(0,100):
    custom1 = generate_custom_peptides("Y", 4, codon_dict)
    custom2 = generate_custom_peptides("Y", 8, codon_dict)
    Tyrosine_4.append(custom1)
    Tyrosine_8.append(custom2)

cyc_Tyrosine_4_list = []
for seq in Tyrosine_4:
    cyc = sequences_of_peptide(seq, "AT")
    cyc_Tyrosine_4_list.append(cyc)
cyc_Tyrosine_4 = [item for sublist in cyc_Tyrosine_4_list for item in sublist]

cyc_Tyrosine_8_list = [] 
for seq in Tyrosine_8:
    cyc = sequences_of_peptide(seq, "AT")
    cyc_Tyrosine_8_list.append(cyc)
cyc_Tyrosine_8 = [item for sublist in cyc_Tyrosine_8_list for item in sublist]
cyc_random = random_for_comparison("AT")



#%%
def plot_violin_plots(list1, list2, list3, name1='List 1', name2='List 2', name3='List 3'):
    data = [list1, list2, list3]
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, inner = "quartile", palette=vibrant_colors)
    plt.xticks([0, 1, 2], [name1, name2, name3])
    plt.ylabel('Cyclizability', fontsize=22, 
                fontname='Verdana')
    plt.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')
    plt.tick_params(axis='both', which='major', labelsize=18)
    #plt.ylim(-0.5, 0)
    plt.tight_layout()
    plt.show()
def perform_t_test(list1, list2):
    t_statistic, p_value = stats.ttest_ind(list1, list2)
    print("T-test results between List 1 and List 2:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Interpret the p-value
    if p_value < 0.01:
        print("The difference between the two lists is statistically significant.")
    else:
        print("The difference between the two lists is not statistically significant.")

#%%
plot_violin_plots(cyc_Tyrosine_4, cyc_Tyrosine_8, cyc_random, "4 Tyrosine", "8 Tyrosine", "Random Peptide")

'''T-test of 4 tyrosine and Random'''
perform_t_test(cyc_Tyrosine_4, cyc_random) #Significant 
perform_t_test(cyc_Tyrosine_8, cyc_random) #Significant

#%%

def amino_custom_testing(amino, run_num):
    Amino_4 = []
    Amino_8 = []
    for i in range(0,run_num):
        custom1 = generate_custom_peptides(amino, 4, codon_dict)
        custom2 = generate_custom_peptides(amino, 8, codon_dict)
        Amino_4.append(custom1)
        Amino_8.append(custom2)
        
    cyc_Amino_4_list = []
    for seq in Amino_4:
        cyc = sequences_of_peptide(seq, "AT")
        cyc_Amino_4_list.append(cyc)
    cyc_Amino_4 = [item for sublist in cyc_Amino_4_list for item in sublist]

    cyc_Amino_8_list = [] 
    for seq in Amino_8:
        cyc = sequences_of_peptide(seq, "AT")
        cyc_Amino_8_list.append(cyc)
    cyc_Amino_8 = [item for sublist in cyc_Amino_8_list for item in sublist]
    cyc_random = random_for_comparison("AT")
    
    plot_violin_plots(cyc_Amino_4, cyc_Amino_8, cyc_random, 
                      name1= f"4 {amino}",name2= f"8 {amino}",
                      name3='Random Peptide')
    
    average_Amino_4 = sum(cyc_Amino_4)/len(cyc_Amino_4)
    average_Amino_8 = sum(cyc_Amino_8)/len(cyc_Amino_8)
    average_random = sum(cyc_random)/len(cyc_random)
    print("Amino Acid:", amino)
    print(average_Amino_4, average_Amino_8, average_random)
    print(perform_t_test(cyc_Amino_4, cyc_random))    
    print(perform_t_test(cyc_Amino_8, cyc_random))
    return average_Amino_4, average_Amino_8, average_random
    
    
#%%
Names,Amino4s, Amino8s, Randos = [], [], [], []
for keys in codon_dict.keys():
    print(keys)
    Names.append(keys)
    Amino4, Amino8, Rando =amino_custom_testing(keys, 10)
    Amino4s.append(Amino4)
    Amino8s.append(Amino8)
    Randos.append(Rando)

summary_dict = {"Amino Acid":Names, "25% content": Amino4s, "50% content":Amino8s, "Random Control":Randos}
df_summary = pd.DataFrame(summary_dict)
#%%
amino_custom_testing("E", 10)

