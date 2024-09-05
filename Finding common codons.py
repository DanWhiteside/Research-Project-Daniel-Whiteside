'''Finding common codons'''
#%%
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
import seaborn as sns
import matplotlib.colors as mcolors
from collections import defaultdict

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
'''
Storing the sacCer2 genome 
'''
def load_fasta_files_to_dict(directory):
    fasta_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            filepath = os.path.join(directory, filename)
            for record in SeqIO.parse(filepath, "fasta"):
                key = os.path.splitext(filename)[0]
                fasta_dict[key] = str(record.seq)
    return fasta_dict

directory = 'sacCer2 Chromosomes'
saccer2_genome = load_fasta_files_to_dict(directory)
#%%
print(saccer2_genome["chrI"])


#%%
'''
Loading datasets
'''
df_cerN = pd.read_csv("(CSV)41586_2012_BFnature11142_MOESM263_ESM.csv")
df_cer = pd.read_csv("41586_2021_3314_MOESM3_ESM (edited copy).csv")
df_cer = df_cer.drop(5378)
df_cerTSS = df_cer.filter(items = ["Chrom", "Strand"])



#%%
'''
Conversion of sacCer3 TS coodinates to sacCer2
'''
converter = ChainFile("sacCer3ToSacCer2.over.chain", one_based = True)

chr_to_roman = {'chr1': 'chrI', 'chr2': 'chrII', 'chr3': 'chrIII', 'chr4': 'chrIV', 
                'chr5': 'chrV', 'chr6': 'chrVI', 'chr7': 'chrVII', 'chr8': 'chrVIII', 
                'chr9': 'chrIX', 'chr10': 'chrX', 'chr11': 'chrXI', 'chr12': 'chrXII', 
                'chr13': 'chrXIII', 'chr14': 'chrXIV', 'chr15': 'chrXV', 'chr16': 'chrXVI'}

chrome_to_num = {"chromosome1": "1", "chromosome2": "2", "chromosome3": "3"
    }

def sacCer3tosacCer2(df, target_column_name, chrome_column_name, chr_to_roman):
    chromosomes = df[chrome_column_name]
    targets = df[target_column_name]
    outputs = []
    for i in range(0, len(targets)):
        chrom = chromosomes[i]
        chrom = chr_to_roman[chrom]
        target = targets[i]
        output = converter[chrom][target]
        output = output[0][1]
        outputs.append(output)
    return outputs
        
#%%
adjusted_left = sacCer3tosacCer2(df_cer, "Experiment_Left", "Chrom", chr_to_roman)
adjusted_right = sacCer3tosacCer2(df_cer, "Experiment_Right", "Chrom", chr_to_roman)
df_cerTSS["Left TS"] = adjusted_left
df_cerTSS["Right TS"]= adjusted_right


#Storing for genome mutation
saccer_2_ORF_Left = sacCer3tosacCer2(df_cer, "SGD_Left", "Chrom", chr_to_roman)
saccer_2_ORF_Right = sacCer3tosacCer2(df_cer, "SGD_Right", "Chrom", chr_to_roman)
df_cerTSS["Left ORF border"] = saccer_2_ORF_Left
df_cerTSS["Right ORF border"] = saccer_2_ORF_Right
saccer2_chrom = df_cer["Chrom"]
saccer2_direction = df_cer["Strand"]

#%%
'''

'''

def sequencer(position, direction, chromosome):
    position = int(position) - 1
    Left_pos = position - 200
    Right_pos = position + 200
    Nucleosome_sequence = chromosome[Left_pos:Right_pos]
    if direction == "+":
        Sequence = Nucleosome_sequence
    elif direction == "-":
        Nucleosome_seq_obj = Seq(Nucleosome_sequence)
        compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
        Sequence = str(compliment_Nuclesome)
    return Sequence


#%%
'''
Analysing amino acid composition / testing amino acid functions 
'''
#function for calculating the cyclability of a given nucleosome 

def gene_sequencer(Left, Right, direction, chromosome):
    try:
        Left = int(Left)
        Right = int(Right)
        gene_sequence = chromosome[Left:Right]
        
        if direction == "+":
            Sequence = gene_sequence
        elif direction == "-":
            gene_seq_obj = Seq(gene_sequence)
            compliment_gene = gene_seq_obj.reverse_complement()
            Sequence = str(compliment_gene)
    except:
        Sequence = "None"
    return Sequence



def gene_finder(dataframe, chromosomes, conversion_dict):
    Left = dataframe["Left ORF border"]
    Right = dataframe["Right ORF border"]
    Direction = dataframe["Strand"]
    Chromosome_id = dataframe["Chrom"]
    Char_to_check = "N"
    Sequences = []
    for i in range(0, len(Left)):
        chrom = Chromosome_id[i]
        chrom = conversion_dict[chrom]
        chromosome = chromosomes[chrom]
        sequence = gene_sequencer(Left[i], Right[i], Direction[i], chromosome)
        if Char_to_check not in sequence:
            Sequences.append(sequence)
        else:
            Sequences.append("None")
    return Sequences
        
'''
Codon table 
'''
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

def codon_counter(sequences, codon_table):
    codon_counts = defaultdict(lambda: defaultdict(int))
    amino_acid_counts = defaultdict(int)
    
    for sequence in sequences:
        if sequence == "None":
            continue
        codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        
        for codon in codons:
            if codon in codon_table:
                amino_acid = codon_table[codon]
                codon_counts[amino_acid][codon] += 1
                amino_acid_counts[amino_acid] += 1
    
    #Calculate percentages
    codon_percentages = {}
    for amino_acid, codons in codon_counts.items():
        codon_percentages[amino_acid] = {}
        for codon, count in codons.items():
            percentage = (count / amino_acid_counts[amino_acid]) * 100
            codon_percentages[amino_acid][codon] = percentage
    
    return codon_counts, codon_percentages







def heatmaps(nuc_num, amino_abundance):
    df = pd.DataFrame(amino_abundance)
    
    df = df.transpose()
    
    # Plot the heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(df, cmap="inferno", vmax=20, vmin=0 ,annot=False)
    
    plt.xlabel("Position")
    plt.ylabel("Amino Acid")
    plt.title(f"Heatmap of Amino Acid Abundance Across Positions in the plus {nuc_num} Nucleosome of Cerevisiae (%)")
    plt.tight_layout()
    plt.show()

def sliding_window_average(arr, window_size=7):
    result = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i+window_size]
        result.append(np.mean(window))
    return np.array(result)
    
def position_visualisation(amino_abundance):
    positional_prevalence = []
    for item in amino_abundance:
        x = (item/sum(amino_abundance))*100
        positional_prevalence.append(x)
    positional_prevalence = sliding_window_average(positional_prevalence)
    print(sum(positional_prevalence))
    return positional_prevalence
        

def pallet_plot(nuc_num, data_dict):
    palette = sns.color_palette("husl", len(data_dict))
    num_plots = len(data_dict)
    cols = 4  # You can adjust this number to change the grid's shape
    rows = (num_plots + cols - 1) // cols  # Ceiling division to determine rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
    
    axes = axes.flatten()
    
    for idx, (key, value) in enumerate(data_dict.items()):
        # Apply the function to the list
        transformed_data = position_visualisation(value)
        
        # Plot the data on the corresponding subplot
        ax = axes[idx]
        ax.plot(transformed_data, color=palette[idx])
        ax.set_title(f"{key}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Abundance (%)")
    
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle(f"Positional Abundance of Amino Acids in Cerevisiae plus {nuc_num} Nucleosome", fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
def darken_color(color, factor=0.8):
    rgb = mcolors.to_rgb(color)
    darkened_rgb = [x * factor for x in rgb]
    return mcolors.to_hex(darkened_rgb)


def pallet_plot2(dict1, dict2, organism1, organism2):
    num_plots = len(dict1)
    cols = 4 
    rows = (num_plots + cols - 1) // cols 

    fig, axes = plt.subplots(rows, cols, figsize=(15, 11))

    axes = axes.flatten()

    handles = []
    labels = []

    for idx, (key, value) in enumerate(dict1.items()):
        transformed_data1 = position_visualisation(value)
        transformed_data2 = position_visualisation(dict2[key])

        ax = axes[idx]
        line1, = ax.plot(transformed_data1, label=f"{organism1}", linestyle='-', linewidth=2)
        line2, = ax.plot(transformed_data2, linewidth=1, label=f"{organism2}")

        ax.set_title(f"{key}", fontsize=15)
        ax.set_xlabel("Position", fontsize=15)

        if idx % cols == 0:
            ax.set_ylabel("Abundance (%)", fontsize=12)
            ax.yaxis.set_tick_params(labelsize=10)
        else:

            ax.set_ylabel("")
            ax.yaxis.set_tick_params(labelsize=10) 

        ax.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')

        if idx == 0:
            handles.extend([line1, line2])
            labels.extend([f"{organism1}", f"{organism2}"])

    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Positional Abundance of Amino Acids in plus 5-9 Nucleosomes", fontsize=14, fontweight='bold')

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2,
               fontsize=10, markerscale=1)

    plt.tight_layout()  # Adjust bottom to make room for the global legend
    plt.show()

def sliding_window_heatmap(dictionary):
    for idx, (key, value) in enumerate(dictionary.items()):
        transformed_data = sliding_window_average(value)
        dictionary[key] = transformed_data
    return dictionary

def average_Nucleosome(Nucleosomes):
    average_cyclability = np.array(Nucleosomes)
    average_cyclability = np.average(Nucleosomes, axis=0)
    return average_cyclability

#%%

sequences = gene_finder(df_cerTSS, saccer2_genome, chr_to_roman)
codon_counts, codon_percentages = codon_counter(sequences, codon_table)

#%%
def plot_codon_percentages(codon_percentages):
    # Create a plot for each amino acid
    for amino_acid, codons in codon_percentages.items():
        codon_names = list(codons.keys())
        percentages = list(codons.values())
        
        plt.figure(figsize=(10, 5))
        plt.bar(codon_names, percentages, color='skyblue')
        plt.title(f'Codon Usage for Amino Acid: {amino_acid}')
        plt.xlabel('Codons')
        plt.ylabel('Percentage')
        plt.ylim(0, 100)  # Percentage is out of 100
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

plot_codon_percentages(codon_percentages)

#%%
import pickle
with open("Codon Percentages Cerevisiae", 'wb') as file:
    pickle.dump(codon_percentages, file)
    
#%%
'''Again for Pombe'''
df_TSS = pd.read_csv("41594_2010_BFnsmb1741_MOESM9_ESM (COPY).csv")
df_TSS["Strand"] = df_TSS["Orientation"]
df_TSS["Chrom"] = df_TSS["Chromosome"]

chromosomes = {}
fasta_file = "GCA_000002945.2_ASM294v2_genomic.fna"

chromosome_keys = ['1', '2', '3', 'mitochondria']


for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
    if idx < len(chromosome_keys):
        key = chromosome_keys[idx]
    else:
        key = f'chromosome_{idx + 1}'
    chromosomes[key] = str(record.seq).upper()
    
#%%
sequencesP = gene_finder(df_TSS, chromosomes, chrome_to_num)
codon_countsP, codon_percentagesP = codon_counter(sequencesP, codon_table)
plot_codon_percentages(codon_percentagesP)

#%%
import pickle
with open("Codon Percentages Pombe", 'wb') as file:
    pickle.dump(codon_percentagesP, file)
