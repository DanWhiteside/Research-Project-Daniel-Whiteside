'''
Finding the most common amino acids in chromosome 1 
'''
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
from collections import Counter
import seaborn as sns


os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
#POMBE
#df_nucleosome = pd.read_csv("sd02.csv")
#df_nucleosome.head

#Importing the relevent data and storing it 
df_nucleosome = pd.read_csv("sd01.csv")
df_TSS = pd.read_csv("41594_2010_BFnsmb1741_MOESM9_ESM (COPY).csv")
#df_TSS.head

#%%
#POMBE
Left_ORF = list(df_TSS["Left ORF border"])
Right_ORF = list(df_TSS["Right ORF border"])
Left_txn = list(df_TSS["Left txn border"])
Right_txn = list(df_TSS["Right txn border"]) 
Nucleosome_chromosome = list(df_nucleosome["Chromosome"])
TS_chromosome = list(df_TSS["Chromosome"])
TS_directions = list(df_TSS["Orientation"])

#%%

# Initialize an empty dictionary to store chromosomes
chromosomes = {}
fasta_file = "GCA_000002945.2_ASM294v2_genomic.fna"

# Define the keys for the first three chromosomes and the mitochondria
chromosome_keys = ['1', '2', '3', 'mitochondria']

# Parse the FASTA file and store each chromosome sequence in the dictionary
for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
    if idx < len(chromosome_keys):
        key = chromosome_keys[idx]
    else:
        key = f'chromosome_{idx + 1}'
    chromosomes[key] = str(record.seq).upper()
    
    
#%%
#function for calculating the cyclability of a given nucleosome 
def sequencer(position, direction, chromosome):
    position = position - 1
    Left_pos = position - 200
    Right_pos = position + 201
    Nucleosome_sequence = chromosome[Left_pos:Right_pos]
    if direction == "+":
        Sequence = Nucleosome_sequence
    elif direction == "-":
        Nucleosome_seq_obj = Seq(Nucleosome_sequence)
        compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
        Sequence = str(compliment_Nuclesome)
    return Sequence

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

#List of Nucleosome finder 
def Nucleosome_TSS_finder(TSS, TTS, Nucleosome_sites, 
                             chromosome_number):
    position_lists = []
    #Nucleosome_sites = Nucleosome_sites.loc[Nucleosome_sites["score 1"]>10]
    for i in range(0, len(TSS)):
        lower_threshold = TSS[i]
        upper_threshold = TTS[i]
        try: 
            if chromosome_number[i] == 	"chromosome1":
                Nuc_chr1 = Nucleosome_sites.loc[Nucleosome_sites["Chromosome"]=="chrI", ["Position"]]
                Nuc_chr1 = Nuc_chr1["Position"]
                filtered_data = list(Nuc_chr1[(Nuc_chr1 > lower_threshold) & (Nuc_chr1 < upper_threshold)])
                filtered_data = sorted(filtered_data)
            elif chromosome_number[i] == "chromosome2":
                Nuc_chr2 = Nucleosome_sites.loc[Nucleosome_sites["Chromosome"]=="chrII", ["Position"]]
                Nuc_chr2 = Nuc_chr2["Position"]
                filtered_data = list(Nuc_chr2[(Nuc_chr2 > lower_threshold) & (Nuc_chr2 < upper_threshold)])
                filtered_data = sorted(filtered_data)
            elif chromosome_number[i] == "chromosome3":
                Nuc_chr3 = Nucleosome_sites.loc[Nucleosome_sites["Chromosome"]=="chrIII", ["Position"]]
                Nuc_chr3 = Nuc_chr3["Position"]
                filtered_data = list(Nuc_chr3[(Nuc_chr3 > lower_threshold) & (Nuc_chr3 < upper_threshold)])
                filtered_data = sorted(filtered_data)
            else:
                filtered_data = ["None"]
        except:
            filtered_data = ["None"]
        position_lists.append(filtered_data)
        print(i)
    return position_lists


#extract specific nucleosome sequence, input dataframe that already has Nucleosome column 
def NucX_sequencer(Nuc_Number, dataframe, chromosomes):
    Position = dataframe["Nucleosome"]
    Direction = dataframe["Orientation"]
    Chromosome_id = dataframe["Chromosome"]
    Sequences = []
    Char_to_check = "N"
    for i in range(0,len(Position)):
        print(i)
        try:
            Current_nucleosomes = Position[i]
            if len(Current_nucleosomes) >= Nuc_Number:
                if Direction[i] == "+":
                    Nucleosome_position = Current_nucleosomes[Nuc_Number-1]
                elif Direction[i] == "-":
                    Nucleosome_position = Current_nucleosomes[-Nuc_Number]
                if Chromosome_id[i] == "chromosome1":
                    chromosome = chromosomes["1"]
                elif Chromosome_id[i] == "chromosome2":
                    chromosome = chromosomes["2"]
                elif Chromosome_id[i] == "chromosome3":
                    chromosome = chromosomes["3"]
                sequence = sequencer(Nucleosome_position, Direction[i], chromosome)
                if Char_to_check not in sequence:
                    Sequences.append(sequence)
                else:
                    Sequences.append("None")
            else:
                Sequences.append("None")
        except:
            Sequences.append("None")
    return Sequences

def gene_finder(dataframe, chromosomes):
    Left = dataframe["Left ORF border"]
    Right = df_TSS["Right ORF border"]
    Direction = dataframe["Orientation"]
    Chromosome_id = dataframe["Chromosome"]
    Char_to_check = "N"
    Sequences = []
    for i in range(0, len(Left)):
        print(i)
        if Chromosome_id[i] == "chromosome1":
            chromosome = chromosomes["1"]
        elif Chromosome_id[i] == "chromosome2":
            chromosome = chromosomes["2"]
        elif Chromosome_id[i] == "chromosome3":
            chromosome = chromosomes["3"]
        sequence = gene_sequencer(Left[i], Right[i], Direction[i], chromosome)
        if Char_to_check not in sequence:
            Sequences.append(sequence)
        else:
            Sequences.append("None")
    return Sequences
        
'''
Generating code to silently mutate DNA sequence  
'''
# Define the codon to amino acid mapping
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
    # Iterate over potential start positions of the desired sequence
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
    # Extract the correctly aligned subsequence
    subsequence, start_index, N_position = extract_subsequence(full_gene, desired_sequence)
    # Adjust the start index to the nearest codon boundary
    adjusted_start_index = start_index % 3
    # Extract the aligned DNA sequence for translation
    aligned_subsequence = full_gene[start_index - adjusted_start_index:start_index + len(desired_sequence)]
    protein_sequence = dna_to_amino(aligned_subsequence, codon_table)
    # Determine how many amino acids to remove from the beginning
    amino_acids_to_remove = adjusted_start_index // 3
    # Remove the unnecessary amino acids from the start
    final_protein_sequence = protein_sequence[amino_acids_to_remove:]
    return final_protein_sequence, start_index, N_position

def amino_acid_sequencer(dataframe):
    Gene = df_TSS["Gene Sequences"]
    Nucleosome = df_TSS["Nucleosome Sequences"]
    Amino_Acid_seq = []
    adjusted_start_seq = []
    for i in range(0,len(df_TSS["Gene Sequences"])):
        print(i)
        if Gene[i] != "None" and Nucleosome[i] != "None":
            Seq, start, adj_n = amino_acid_nucleosomes(Gene[i], Nucleosome[i]) 
            
        else:
            Seq = "None" 
            adj_n = "None"
        Amino_Acid_seq.append(Seq)
        adjusted_start_seq.append(adj_n)
    return Amino_Acid_seq, adjusted_start_seq

def amino_acid_counter(dataframe):
    protein_sequences = df_TSS["Nucleosome Amino Acids"]
    amino_acid_counter = Counter()
    
    for sequence in protein_sequences:
        amino_acid_counter.update(sequence)
    
    return dict(amino_acid_counter)

def plot_amino_acid_tally(amino_acid_tally):
    # Sort the amino acids alphabetically
    amino_acids = sorted(amino_acid_tally.keys())
    counts = [amino_acid_tally[aa] for aa in amino_acids]
    
    plt.figure(figsize=(10, 6))
    plt.bar(amino_acids, counts, color='skyblue')
    
    plt.xlabel('Amino Acids')
    plt.ylabel('Counts')
    plt.title('Amino Acid Tally')
    plt.xticks(rotation=90)
    
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    
    plt.tight_layout()
    plt.show()

amino_acid_dict = {
    "Alanine": "A",
    "Arginine": "R",
    "Asparagine": "N",
    "Aspartic acid": "D",
    "Cysteine": "C",
    "Glutamic acid": "E",
    "Glutamine": "Q",
    "Glycine": "G",
    "Histidine": "H",
    "Isoleucine": "I",
    "Leucine": "L",
    "Lysine": "K",
    "Methionine": "M",
    "Phenylalanine": "F",
    "Proline": "P",
    "Serine": "S",
    "Threonine": "T",
    "Tryptophan": "W",
    "Tyrosine": "Y",
    "Valine": "V", 
}

amino_acids = [
    "Alanine",
    "Arginine",
    "Asparagine",
    "Aspartic acid",
    "Cysteine",
    "Glutamic acid",
    "Glutamine",
    "Glycine",
    "Histidine",
    "Isoleucine",
    "Leucine",
    "Lysine",
    "Methionine",
    "Phenylalanine",
    "Proline",
    "Serine",
    "Threonine",
    "Tryptophan",
    "Tyrosine",
    "Valine", 
]

'''
First version 

def amino_by_position(amino_acid, dataframe, amino_acid_dict):
    amino = amino_acid_dict[amino_acid]
    seq = dataframe["Nucleosome Amino Acids"]
    seq = [item for item in seq if item != 'None']
    pos = dataframe["N position on the gene"]
    pos = [item for item in pos if item != 'None']
    
    #offsetting the sequences that aren't aligned 
    padded_seqs = []
    for i in range(0,len(seq)-1):
        offset = pos[i]//3
        padded_seq = "x" * offset + seq[i]
        padded_seqs.append(padded_seq)
    
    print(len(max(padded_seqs)))
    #finding the thing 
    # Step 2: Iterate over all sequences
    sequence_length = 275
    position_counts = [0] * sequence_length
    na_counts = [0] * sequence_length
    for seq in padded_seqs:
        for i, aa in enumerate(seq):
            if aa == amino:
                position_counts[i] += 1
            if aa == "x":
                na_counts[i] += 1
    
    # Step 3: Calculate abundance percentages
    total_sequences = len(padded_seqs)
    abundance_percentages = [(position_counts[i] / (total_sequences - na_counts[i])) * 100 for i in range(0,sequence_length)]
    return abundance_percentages
'''
def amino_by_position(amino_acid, dataframe, amino_acid_dict):
    amino = amino_acid_dict[amino_acid]
    seq = dataframe["Nucleosome Amino Acids"]
    seq = [item for item in seq if item != 'None']
    pos = dataframe["N position on the gene"]
    pos = [item for item in pos if item != 'None']
    
    # Offset the sequences that aren't aligned 
    padded_seqs = []
    for i in range(len(seq)):
        offset = int(pos[i]) // 3
        padded_seq = "x" * offset + seq[i]
        padded_seqs.append(padded_seq)
    
    # Determine the maximum length of sequences for proper alignment
    max_length = len(max(padded_seqs, key=len))
    
    # Initialize counts for each position
    position_counts = [0] * max_length
    total_counts = [0] * max_length
    
    # Count the occurrences of the target amino acid and valid sequences at each position
    for seq in padded_seqs:
        for i, aa in enumerate(seq):
            if i < max_length:
                if aa == amino:
                    position_counts[i] += 1
                if aa != 'x':  # Count only valid amino acids
                    total_counts[i] += 1
    
    # Calculate abundance percentages
    abundance_percentages = [(position_counts[i] / total_counts[i]) * 100 if total_counts[i] > 0 else 0 for i in range(max_length)]
    abundance_percentages = abundance_percentages[:133]
    return abundance_percentages

                
    
    # Calculate abundance percentages
    position_counts = position_counts[:133]
    abundance_percentages = [(position_counts[i] / sum(position_counts)) * 100 for i in range(0,133)]
    return abundance_percentages

def count_all_amino(dataframe, amino_acid_dict, amino_list):
    amino_acid_abundances = {}
    # Loop through each amino acid and calculate its abundance
    for amino in amino_list:
        abundance = amino_by_position(amino, dataframe, amino_acid_dict)  # Use the first letter of each amino acid
        amino_acid_abundances[amino] = abundance
    return amino_acid_abundances


def sum_abundances(amino_acid_abundances):
    # Find the length of sequences (assuming all have the same length)
    sequence_length = len(next(iter(amino_acid_abundances.values())))
    
    # Initialize a list of zeros to store the sum of abundances at each position
    summed_abundances = [0] * sequence_length
    
    # Loop through each amino acid's abundance
    for amino_acid, abundances in amino_acid_abundances.items():
        for i in range(sequence_length):
            summed_abundances[i] += abundances[i]
    
    return summed_abundances

def count_all_by_nucleosome(nuc_num, dataframe, amino_acid_dict, amino_list, chromosomes):
    Nucleosome_TS = NucX_sequencer(nuc_num, dataframe, chromosomes)  #edit this to change the target nucleosome
    dataframe["Nucleosome Sequences"] = Nucleosome_TS

    Gene_Sequences = gene_finder(dataframe, chromosomes)
    dataframe["Gene Sequences"] = Gene_Sequences

    Nucleosome_aminos, Nucleosome_start = amino_acid_sequencer(dataframe)
    dataframe["Nucleosome Amino Acids"] = Nucleosome_aminos 
    dataframe["N position on the gene"] = Nucleosome_start
    
    amino_abundance = count_all_amino(dataframe, amino_acid_dict, amino_list)
    return amino_abundance


def heatmaps(nuc_num, amino_abundance):
    df = pd.DataFrame(amino_abundance)
    
    # Transpose the DataFrame to have positions as rows and amino acids as columns
    df = df.transpose()
    
    # Plot the heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(df, cmap="inferno", vmax=20, vmin=0 ,annot=False)
    
    # Adding labels and title
    plt.xlabel("Position")
    plt.ylabel("Amino Acid")
    plt.title(f"Heatmap of Amino Acid Abundance Across Positions in the plus {nuc_num} Nucleosome of Pombe (%)")
    
    # Display the plot
    plt.tight_layout()
    plt.show()


def sliding_window_average(arr, window_size=10):
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
    # Set the color palette
    palette = sns.color_palette("husl", len(data_dict))
    
    # Determine the number of rows and columns for the grid
    num_plots = len(data_dict)
    cols = 4  # You can adjust this number to change the grid's shape
    rows = (num_plots + cols - 1) // cols  # Ceiling division to determine rows
    
    # Create a smaller figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Loop through the dictionary and plot each list in a separate subplot
    for idx, (key, value) in enumerate(data_dict.items()):
        # Apply the function to the list
        transformed_data = position_visualisation(value)
        
        # Plot the data on the corresponding subplot
        ax = axes[idx]
        ax.plot(transformed_data, color=palette[idx])
        ax.set_title(f"{key}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Abundance (%)")
    
    # Hide any unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])
    
    # Add a super title for the entire grid
    fig.suptitle(f"Positional Abundance of Amino Acids in Pombe plus {nuc_num} Nucleosome", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

#%%
Nucleosome_Positions = Nucleosome_TSS_finder(Left_txn, Right_txn, df_nucleosome, TS_chromosome)
df_TSS["Nucleosome"] = Nucleosome_Positions

#%%
Nucleosome_TS = NucX_sequencer(1, df_TSS, chromosomes)  #edit this to change the target nucleosome
df_TSS["Nucleosome Sequences"] = Nucleosome_TS

Gene_Sequences = gene_finder(df_TSS, chromosomes)
df_TSS["Gene Sequences"] = Gene_Sequences

#%%
Nucleosome_aminos, Nucleosome_start = amino_acid_sequencer(df_TSS)
df_TSS["Nucleosome Amino Acids"] = Nucleosome_aminos 
df_TSS["N position on the gene"] = Nucleosome_start

#%%
amino_tally = amino_acid_counter(df_TSS)

plot_amino_acid_tally(amino_tally)


#%%
Leucine_N = amino_by_position("Leucine", df_TSS, amino_acid_dict)
#Valine_N2 = 

#%%
x_axis = np.linspace(-200, 200, num = 133)

plt.figure(figsize=(14, 8))
plt.plot(x_axis, Leucine_N)
plt.xlabel("Position Along Nucleosome (bp)")
plt.ylabel("Abundance (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
N_amino_abundance = count_all_amino(df_TSS, amino_acid_dict, amino_acids)
print(sum_abundances(N_amino_abundance))


#%%
'''
Heatmaps of amino across Pombe 1 - 9 
'''
#1
plus1_aminos = count_all_by_nucleosome(1, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#2
plus2_aminos = count_all_by_nucleosome(2, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#3
plus3_aminos = count_all_by_nucleosome(3, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#4
plus4_aminos = count_all_by_nucleosome(4, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#5
plus5_aminos = count_all_by_nucleosome(5, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#6
plus6_aminos = count_all_by_nucleosome(6, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#7
plus7_aminos = count_all_by_nucleosome(7, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#8
plus8_aminos = count_all_by_nucleosome(8, df_TSS, amino_acid_dict, amino_acids, chromosomes)

plus9_aminos = count_all_by_nucleosome(9, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#%%
'''
positional prevelence
'''
#testing 
test = amino_by_position2("Leucine", df_TSS, amino_acid_dict)
print(sum(test))

#%%
plt.figure(figsize=(14, 8))
plt.plot(range(1, len(test)+1), test)
plt.xlabel("Position")
plt.ylabel("Relative Abundance (%)")
plt.title("Leucine prevalence along nucleosome")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
plus2_aminos = count_all_by_nucleosome(2, df_TSS, amino_acid_dict, amino_acids, chromosomes)
pallet_plot(2, plus2_aminos)
#3
plus3_aminos = count_all_by_nucleosome(3, df_TSS, amino_acid_dict, amino_acids, chromosomes)
pallet_plot(3, plus3_aminos)
#4
plus4_aminos = count_all_by_nucleosome(4, df_TSS, amino_acid_dict, amino_acids, chromosomes)
pallet_plot(4, plus4_aminos)
#%%
#5-9
plus5_aminos = count_all_by_nucleosome(5, df_TSS, amino_acid_dict, amino_acids, chromosomes)
plus6_aminos = count_all_by_nucleosome(6, df_TSS, amino_acid_dict, amino_acids, chromosomes)
plus7_aminos = count_all_by_nucleosome(7, df_TSS, amino_acid_dict, amino_acids, chromosomes)
plus8_aminos = count_all_by_nucleosome(8, df_TSS, amino_acid_dict, amino_acids, chromosomes)
plus9_aminos = count_all_by_nucleosome(9, df_TSS, amino_acid_dict, amino_acids, chromosomes)

#%%
pombeN5to9 = [plus5_aminos, plus6_aminos, plus7_aminos, plus8_aminos, plus9_aminos]

# Initialize the result dictionary
Average_amino = {}

# Iterate over each key in the dictionaries
for key in plus5_aminos.keys():
    # Use zip to aggregate the values across all dictionaries for each key
    # For example: key 'a' -> zip([1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7])
    aggregated_values = zip(*(d[key] for d in pombeN5to9))
    
    # Calculate the average for each position and store it in the result dictionary
    Average_amino[key] = [sum(values) / len(values) for values in aggregated_values]

#%%
pallet_plot("5 - 9", Average_amino)

#%%
'''Storing Average Amino'''
import pickle


with open('Pombe Amino 5-9.pkl', 'wb') as file:
    pickle.dump(Average_amino, file)
