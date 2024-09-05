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
saccer_2_ORF_Left = sacCer3tosacCer2(df_cer, "SGD_Left", "Chrom", chr_to_roman)
saccer_2_ORF_Right = sacCer3tosacCer2(df_cer, "SGD_Right", "Chrom", chr_to_roman)
df_cerTSS["Left ORF border"] = saccer_2_ORF_Left
df_cerTSS["Right ORF border"] = saccer_2_ORF_Right
saccer2_chrom = df_cer["Chrom"]
saccer2_direction = df_cer["Strand"]

#%%
'''
Finding all nucleosomes within each gene 
'''

def Nfinder(dfN, dfT, conversion_dict):
    output = []
    Left = dfT["Left TS"]
    Right = dfT["Right TS"]
    Chromosome = dfT["Chrom"]
    for i in range(0, len(dfT)):
        lower_threshold = Left[i]
        upper_threshold = Right[i]
        chrom = conversion_dict[Chromosome[i]]
        Nucleosomes = dfN.loc[dfN["Chromosome"]== chrom, ["Position"]]
        Nucleosomes = Nucleosomes["Position"]
        filtered_data = Nucleosomes[(Nucleosomes > lower_threshold) & (Nucleosomes < upper_threshold)]
        filtered_data = sorted(filtered_data)
        output.append(filtered_data)
    return output

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

def Nucleosome_N_Seq(Nuc_Number,df, genome, conversion_dict):
    Position = df["Nucleosome Positions"]
    Direction = df["Strand"]
    Chromosome_id = df["Chrom"]
    Sequences = []
    Char_to_check = "N"
    for i in range(0,len(Position)):
        if i%50 == 0:
            print(i)
        #try:
        Current_nucleosomes = Position[i]
        #print(Current_nucleosomes)
        if len(Current_nucleosomes) >= Nuc_Number:
            if Direction[i] == "+":
                Nucleosome_position = Current_nucleosomes[Nuc_Number-1]
            elif Direction[i] == "-":
                Nucleosome_position = Current_nucleosomes[-Nuc_Number]
            Chrom_id = conversion_dict[Chromosome_id[i]]
            chromosome = genome[Chrom_id]
            #print(Nucleosome_position)
            #print(str(chromosome))
            sequence = sequencer(Nucleosome_position, Direction[i], chromosome)
            if Char_to_check not in sequence:
                Sequences.append(sequence)
            else:
                Sequences.append("None")
        else:
            Sequences.append("None")
        #except:
            #Sequences.append("None")
    return Sequences


#%%
NucleosomeByGene = Nfinder(df_cerN, df_cerTSS, chr_to_roman)
df_cerTSS["Nucleosome Positions"] = NucleosomeByGene
#%%
N_NucleosomeSequence = Nucleosome_N_Seq(1, df_cerTSS, saccer2_genome, chr_to_roman)

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
#Codon to amino acid mapping
codon_table = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 
    'TAT': 'Y', 'TAC': 'Y', 'TAA': 'Stop', 'TAG': 'Stop', 
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 
    'TGT': 'C', 'TGC': 'C', 'TGA': 'Stop', 'TGG': 'W', 
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}



def extract_subsequence(full_gene, desired_sequence):
    for i in range(len(desired_sequence)):
        adjusted_sequence = desired_sequence[i:]
        start = full_gene.find(adjusted_sequence)
        if start != -1:
            aligned_subsequence = full_gene[start:start + len(adjusted_sequence)]
            position = i
            return aligned_subsequence, start, position
    
    raise ValueError("Desired sequence not found in the full gene.")


def dna_to_amino(sequence, codon_table):
    protein = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        if codon in codon_table:
            amino_acid = codon_table[codon]
            if amino_acid == 'Stop':
                break
            protein.append(amino_acid)
    return ''.join(protein)



def amino_acid_nucleosomes(full_gene, desired_sequence):
    subsequence, start_index, N_position = extract_subsequence(full_gene, desired_sequence)
    adjusted_start_index = start_index % 3
    aligned_subsequence = full_gene[start_index - adjusted_start_index:start_index + len(desired_sequence)]
    protein_sequence = dna_to_amino(aligned_subsequence, codon_table)
    amino_acids_to_remove = adjusted_start_index // 3
    final_protein_sequence = protein_sequence[amino_acids_to_remove:]
    return final_protein_sequence, start_index, N_position

def amino_acid_sequencer(dataframe):
    Gene = dataframe["Gene Sequences"]
    Nucleosome = dataframe["Nucleosome Sequences"]
    Amino_Acid_seq = []
    adjusted_start_seq = []
    for i in range(0,len(dataframe["Gene Sequences"])):
        print(i)
        if Gene[i] != "None" and Nucleosome[i] != "None":
            Seq, start, adj_n = amino_acid_nucleosomes(Gene[i], Nucleosome[i]) 
            
        else:
            Seq = "None" 
            adj_n = "None"
        Amino_Acid_seq.append(Seq)
        adjusted_start_seq.append(adj_n)
    return Amino_Acid_seq, adjusted_start_seq

amino_acid_dict = {"Alanine": "A", "Arginine": "R", "Asparagine": "N", "Aspartic acid": "D",
                   "Cysteine": "C", "Glutamic acid": "E", "Glutamine": "Q", "Glycine": "G",
                   "Histidine": "H", "Isoleucine": "I", "Leucine": "L", "Lysine": "K",
                   "Methionine": "M", "Phenylalanine": "F", "Proline": "P", "Serine": "S",
                   "Threonine": "T", "Tryptophan": "W", "Tyrosine": "Y", "Valine": "V", 
}

amino_acids = ["Alanine", "Arginine", "Asparagine", "Aspartic acid",
               "Cysteine", "Glutamic acid", "Glutamine", "Glycine",
               "Histidine", "Isoleucine", "Leucine", "Lysine",
               "Methionine", "Phenylalanine", "Proline", "Serine",
               "Threonine", "Tryptophan", "Tyrosine", "Valine"]


def amino_by_position(amino_acid, dataframe, amino_acid_dict):
    amino = amino_acid_dict[amino_acid]
    seq = dataframe["Nucleosome Amino Acids"]
    seq = [item for item in seq if item != 'None']
    pos = dataframe["N position on the gene"]
    pos = [item for item in pos if item != 'None']
    
    padded_seqs = []
    for i in range(len(seq)):
        offset = int(pos[i]) // 3
        padded_seq = "x" * offset + seq[i]
        padded_seqs.append(padded_seq)
        
    max_length = len(max(padded_seqs, key=len))

    position_counts = [0] * max_length
    total_counts = [0] * max_length
    
    for seq in padded_seqs:
        for i, aa in enumerate(seq):
            if i < max_length:
                if aa == amino:
                    position_counts[i] += 1
                if aa != 'x':  # Count only valid amino acids
                    total_counts[i] += 1
    
    #Percentage Abundance
    abundance_percentages = [(position_counts[i] / total_counts[i]) * 100 if total_counts[i] > 0 else 0 for i in range(max_length)]
    abundance_percentages = abundance_percentages[:133]
    return abundance_percentages


def count_all_amino(dataframe, amino_acid_dict, amino_list):
    amino_acid_abundances = {}
    for amino in amino_list:
        abundance = amino_by_position(amino, dataframe, amino_acid_dict) 
        amino_acid_abundances[amino] = abundance
    return amino_acid_abundances


def sum_abundances(amino_acid_abundances):
    sequence_length = len(next(iter(amino_acid_abundances.values())))
    summed_abundances = [0] * sequence_length
    for amino_acid, abundances in amino_acid_abundances.items():
        for i in range(sequence_length):
            summed_abundances[i] += abundances[i]
    
    return summed_abundances

def count_all_by_nucleosome(nuc_num, dataframe, amino_acid_dict, amino_list, chromosomes, conversion_dict):
    Nucleosome_TS = Nucleosome_N_Seq(nuc_num, dataframe, chromosomes, conversion_dict)  #edit this to change the target nucleosome
    dataframe["Nucleosome Sequences"] = Nucleosome_TS

    Gene_Sequences = gene_finder(dataframe, chromosomes, conversion_dict)
    dataframe["Gene Sequences"] = Gene_Sequences

    Nucleosome_aminos, Nucleosome_start = amino_acid_sequencer(dataframe)
    dataframe["Nucleosome Amino Acids"] = Nucleosome_aminos 
    dataframe["N position on the gene"] = Nucleosome_start
    
    amino_abundance = count_all_amino(dataframe, amino_acid_dict, amino_list)
    return amino_abundance

amino_acid_mapping = {
    'Alanine': 'A', 'Cysteine': 'C', 'Aspartic acid': 'D', 'Glutamic acid': 'E',
    'Phenylalanine': 'F', 'Glycine': 'G', 'Histidine': 'H', 'Isoleucine': 'I',
    'Lysine': 'K', 'Leucine': 'L', 'Methionine': 'M', 'Asparagine': 'N',
    'Proline': 'P', 'Glutamine': 'Q', 'Arginine': 'R', 'Serine': 'S',
    'Threonine': 'T', 'Valine': 'V', 'Tryptophan': 'W', 'Tyrosine': 'Y'
}

def replace_keys_with_single_letters(amino_abundance):
    updated_dict = {amino_acid_mapping.get(k, k): v for k, v in amino_abundance.items()}
    return updated_dict

def heatmaps(nuc_num, amino_abundance):
    amino_abundance = replace_keys_with_single_letters(amino_abundance)
    df = pd.DataFrame(amino_abundance)
    df = df.transpose()
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(df, cmap="inferno", vmax=20, vmin=0 ,annot=False)
    
    plt.xlabel("Amino Acid Position",fontsize=18, 
               fontname='Verdana')
    plt.ylabel("Amino Acid",fontsize=18, 
               fontname='Verdana')
   
    plt.tick_params(axis='y', which='major', labelsize=15)
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
        
#Plots all the amino acids positional abundances
def pallet_plot(nuc_num, data_dict):
    palette = sns.color_palette("husl", len(data_dict))
    num_plots = len(data_dict)
    cols = 4  
    rows = (num_plots + cols - 1) // cols  
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
    
    axes = axes.flatten()
    
    for idx, (key, value) in enumerate(data_dict.items()):
        transformed_data = position_visualisation(value)
        ax = axes[idx]
        ax.plot(transformed_data, color=palette[idx])
        ax.set_title(f"{key}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Abundance (%)")
    

    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f"Positional Abundance of Amino Acids in {nuc_num} plus 5 - 9 Nucleosomes", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
def darken_color(color, factor=0.8):
    rgb = mcolors.to_rgb(color)
    darkened_rgb = [x * factor for x in rgb]
    return mcolors.to_hex(darkened_rgb)

#Plots all the amino acids of both species 
def pallet_plot2(dict1, dict2, organism1, organism2):
    
    cols = 4  
    rows = 5
    fig, axes = plt.subplots(rows, cols, figsize=(30, 22))

    axes = axes.flatten()
    handles = []
    labels = []

    for idx, (key, value) in enumerate(dict1.items()):
        transformed_data1 = position_visualisation(value)
        transformed_data2 = position_visualisation(dict2[key])

        ax = axes[idx]
        line1, = ax.plot(transformed_data1, label=f"{organism1}", linestyle='-', linewidth=4)
        line2, = ax.plot(transformed_data2, linewidth=2, label=f"{organism2}")
        
        ax.set_title(f"{key}", fontsize=28)
        if idx % cols == 0:
            ax.set_ylabel("Abundance (%)", fontsize=28)
            ax.yaxis.set_tick_params(labelsize=26)
        else:
            ax.set_ylabel("")
            ax.yaxis.set_tick_params(labelsize=26)
        
        if idx  > 15:
            ax.set_xlabel("Position", fontsize=28)
            ax.yaxis.set_tick_params(labelsize=26)
        else:
            ax.set_xlabel("")
            ax.yaxis.set_tick_params(labelsize=26)
        ax.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')

        if idx == 0:
            handles.extend([line1, line2])
            labels.extend([f"{organism1}", f"{organism2}"])


    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2,
               fontsize=31, markerscale=1)


    plt.tight_layout() 
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
CerN5_aminos = count_all_by_nucleosome(5,df_cerTSS, amino_acid_dict, amino_acids, saccer2_genome, chr_to_roman)
CerN6_aminos = count_all_by_nucleosome(6,df_cerTSS, amino_acid_dict, amino_acids, saccer2_genome, chr_to_roman)
CerN7_aminos = count_all_by_nucleosome(7,df_cerTSS, amino_acid_dict, amino_acids, saccer2_genome, chr_to_roman)
CerN8_aminos = count_all_by_nucleosome(8,df_cerTSS, amino_acid_dict, amino_acids, saccer2_genome, chr_to_roman)
CerN9_aminos = count_all_by_nucleosome(9,df_cerTSS, amino_acid_dict, amino_acids, saccer2_genome, chr_to_roman)

#%%
CerN5to9 = [CerN5_aminos, CerN6_aminos, CerN7_aminos, CerN8_aminos, CerN9_aminos]

CerAverage_amino = {}

for key in CerN5_aminos.keys():
    aggregated_values = zip(*(d[key] for d in CerN5to9))
    CerAverage_amino[key] = [sum(values) / len(values) for values in aggregated_values]

#%%
pallet_plot("5 - 9", CerAverage_amino)
heatmaps("Cerevisiae", CerAverage_amino)

#%%
#Loading Pombe data here to plot them together 
import pickle 
with open('Pombe Amino 5-9.pkl', 'rb') as file:
    pombeAverage_Amino = pickle.load(file)
    
#%%
pallet_plot2(CerAverage_amino, pombeAverage_Amino, "Cerevisiae", "Pombe")
heatmaps("Pombe", pombeAverage_Amino)
