'''D. Melanogaster amino acid composition'''
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
from scipy.signal import find_peaks

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#%%
''' Combining DNA mnase results '''
# Replace 'your_file.bedgraph' with the path to your BEDGRAPH file
df1 = pd.read_csv('GSM1694819_S2_cells_genomic_DNA_MNase_rep1.bedGraph', sep='\t', header=None, names=['chrom', 'start', 'end', 'value'])
df2 = pd.read_csv('GSM1694819_S2_cells_genomic_DNA_MNase_rep1.bedGraph', sep='\t', header=None, names=['chrom', 'start', 'end', 'value'])

# Combine dataframes
df_combined = pd.concat([df1, df2])

df_combined = df_combined.groupby(['chrom', 'start', 'end']).mean().reset_index()

#%%
def find_nucleosome_centers(df_combined, height=2, distance=200, prominence=1):
    """
    Finds nucleosome centers by detecting peaks in MNase data for each chromosome.
    
    Parameters:
    - df_combined: DataFrame with columns ['chrom', 'start', 'end', 'value']
    - height: float, minimum height of peaks to detect
    - distance: int, minimum distance between peaks
    - prominence: float, minimum prominence of peaks
    
    Returns:
    - DataFrame containing nucleosome centers with columns: chrom, start, end, value, center
    """
    # List to store results for each chromosome
    all_peaks = []

    # Get unique chromosomes
    chroms = df_combined['chrom'].unique()
    print(chroms)
    
    for chrom in chroms:
        # Filter data for the current chromosome
        df_chrom = df_combined[df_combined['chrom'] == chrom]
        
        # Apply find_peaks to detect peaks
        peaks, _ = find_peaks(df_chrom['value'], height=height, distance=distance, prominence=prominence)
        
        # Extract peak data
        df_chrom_peaks = df_chrom.iloc[peaks].copy()  # Use .copy() to avoid SettingWithCopyWarning
        df_chrom_peaks['center'] = (df_chrom_peaks['start'] + df_chrom_peaks['end']) / 2
        df_chrom_peaks["Length"] = (df_chrom_peaks['end'] - df_chrom_peaks['start'])
        
        # Append to the list of results
        all_peaks.append(df_chrom_peaks)
    
    # Combine all chromosome-specific peak DataFrames
    nucleosome_centers = pd.concat(all_peaks, ignore_index=True)
    
    return nucleosome_centers

df_flyN = find_nucleosome_centers(df_combined)

#%%
'''Fly TSS and TTS?'''

# Define the column names as per GTF format
column_names = ["chrom", "source", "feature", "Left TS", "Right TS", "score", "strand", "frame", "attribute"]

# Load the GTF file into a DataFrame
gtf_file = 'dm3.ensGene.gtf'
df = pd.read_csv(gtf_file, sep='\t', comment='#', names=column_names)

# Filter for transcript features
transcripts_df = df[df['feature'] == 'transcript']



df_flyTS = transcripts_df

print(df_flyTS["chrom"].unique())

#%%
'''Filtering Genes by size'''
df_flyTS["Length"] = df_flyTS["Right TS"] - df_flyTS["Left TS"]
pd.to_numeric(df_flyTS["Length"])
df_flyTS = df_flyTS[df_flyTS["Length"] > 100] 
df_flyTS = df_flyTS[df_flyTS["Length"] < 15000].reset_index()
    

#%%
''' Filtering by chromosomes '''
target_chromosomes = ["chr2R", "chr3R","chr4"]
df_flyChrT = df_flyTS[df_flyTS["chrom"].isin(target_chromosomes)].reset_index()


'''storing columns for codon randomisation '''
dm3_Left = df_flyChrT["Left TS"]
dm3_Right = df_flyChrT["Right TS"]
dm3_chrom = df_flyChrT["chrom"]
dm3_strand = df_flyChrT["strand"]


#%%
'''Loading Genome'''

def load_fasta_files_to_dict(directory):
    """
    Load all FASTA files from a directory into a dictionary containing only sequences.
    
    Parameters:
    - directory: The path to the directory containing FASTA files.
    
    Returns:
    - A dictionary where keys are filenames (without extensions) and values are sequences as strings.
    """
    fasta_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            filepath = os.path.join(directory, filename)
            # Read the FASTA file
            for record in SeqIO.parse(filepath, "fasta"):
                # Use the filename (without extension) as the key
                key = os.path.splitext(filename)[0]
                # Store only the sequence as a string
                fasta_dict[key] = str(record.seq)
    return fasta_dict

directory = "BDGP5dm3 Chromosomes"
dm3_genome = load_fasta_files_to_dict(directory)

#%%
'''
Finding all nucleosomes within each gene 
'''

def Nfinder(dfN, dfT):
    output = []
    Left = dfT["Left TS"]
    Right = dfT["Right TS"]
    Chromosome = dfT["chrom"]
    for i in range(0, len(dfT)):
        if i % 500  == 0:
            print(i)
        lower_threshold = Left[i]
        upper_threshold = Right[i]
        try:
            chrom = Chromosome[i]
            Nucleosomes = dfN.loc[dfN["chrom"]== chrom, ["center"]]
            #print(Nucleosomes)
            Nucleosomes = Nucleosomes["center"]
            #print(Nucleosomes)
            filtered_data = Nucleosomes[(Nucleosomes > lower_threshold) & (Nucleosomes < upper_threshold)]
            #print(filtered_data)
            filtered_data = sorted(filtered_data)
        except:
            filtered_data = "None"
        output.append(filtered_data)    
    return output

def sequencer(position, direction, chromosome):
    position = int(position) - 1
    Left_pos = position - 200
    Right_pos = position + 200
    Nucleosome_sequence = chromosome[Left_pos:Right_pos]
    if direction == "+":
        Sequence = Nucleosome_sequence.upper()
    elif direction == "-":
        Nucleosome_seq_obj = Seq(Nucleosome_sequence)
        compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
        Sequence = str(compliment_Nuclesome).upper()
    return Sequence

def Nucleosome_N_Seq(Nuc_Number,df, genome):
    Position = df["Nucleosome Positions"]
    Direction = df["strand"]
    Chromosome_id = df["chrom"]
    Sequences = []
    Char_to_check = "N"
    for i in range(0,len(Position)):
        # if i%50 == 0:
        #     print(i)
        #try:
        Current_nucleosomes = Position[i]
        #print(Current_nucleosomes)
        if Current_nucleosomes != "None":
            if len(Current_nucleosomes) >= Nuc_Number:
                if Direction[i] == "+":
                    Nucleosome_position = Current_nucleosomes[Nuc_Number-1]
                elif Direction[i] == "-":
                    Nucleosome_position = Current_nucleosomes[-Nuc_Number]
                Chrom_id = Chromosome_id[i]
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
        else:
            Sequences.append("None")
        #except:
            #Sequences.append("None")
    return Sequences

def average_Nucleosome(Nucleosomes):
    average_cyclability = np.array(Nucleosomes)
    average_cyclability = np.average(Nucleosomes, axis=0)
    return average_cyclability

#%%

NucleosomeByGene = Nfinder(df_flyN, df_flyChrT)
df_flyChrT["Nucleosome Positions"] = NucleosomeByGene

#%%
N_NucleosomeSequence = Nucleosome_N_Seq(1, df_flyChrT, dm3_genome)

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



def gene_finder(dataframe, chromosomes):
    Left = dataframe["Left TS"]
    Right = dataframe["Right TS"]
    Direction = dataframe["strand"]
    Chromosome_id = dataframe["chrom"]
    Char_to_check = "N"
    Sequences = []
    for i in range(0, len(Left)):
        chrom = Chromosome_id[i]
        #chrom = conversion_dict[chrom]
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
    try:
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
    except:
        final_protein_sequence, start_index, N_position = "None", "None", "None"

def amino_acid_sequencer(dataframe):
    Gene = dataframe["Gene Sequences"]
    Nucleosome = dataframe["Nucleosome Sequences"]
    Amino_Acid_seq = []
    adjusted_start_seq = []
    for i in range(0,len(dataframe["Gene Sequences"])):
        print(i)
        if Gene[i] != "None" and Nucleosome[i] != "None":
            try:
                Seq, start, adj_n = amino_acid_nucleosomes(Gene[i], Nucleosome[i]) 
            except:
                Seq = "None"
                adj_n = "None"
        else:
            Seq = "None" 
            adj_n = "None"
        Amino_Acid_seq.append(Seq)
        adjusted_start_seq.append(adj_n)
    return Amino_Acid_seq, adjusted_start_seq

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

                
    
    # # Calculate abundance percentages
    # position_counts = position_counts[:133]
    # abundance_percentages = [(position_counts[i] / sum(position_counts)) * 100 for i in range(0,133)]
    # return abundance_percentages

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
    Nucleosome_TS = Nucleosome_N_Seq(nuc_num, dataframe, chromosomes)  #edit this to change the target nucleosome
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
    plt.title(f"Heatmap of Amino Acid Abundance Across Positions in the plus {nuc_num} Nucleosome of Cerevisiae (%)")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def sliding_window_average(arr, window_size=15):
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
    fig.suptitle(f"Positional Abundance of Amino Acids in M. Musculus plus {nuc_num} Nucleosome", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def sliding_window_heatmap(dictionary):
    for idx, (key, value) in enumerate(dictionary.items()):
        # Apply the function to the list
        transformed_data = sliding_window_average(value)
        dictionary[key] = transformed_data
    return dictionary
    
#%%
dm3N1_aminos = count_all_by_nucleosome(1,df_flyChrT, amino_acid_dict, amino_acids, dm3_genome)
heatmaps(3, dm3N1_aminos)
pallet_plot(1, dm3N1_aminos)

sliding_window_aminos = sliding_window_heatmap(dm3N1_aminos)
heatmaps(3, sliding_window_aminos)

#%%
dm3N5_aminos = count_all_by_nucleosome(5,df_flyChrT, amino_acid_dict, amino_acids, dm3_genome)
dm3N6_aminos = count_all_by_nucleosome(6,df_flyChrT, amino_acid_dict, amino_acids, dm3_genome)
dm3N7_aminos = count_all_by_nucleosome(7,df_flyChrT, amino_acid_dict, amino_acids, dm3_genome)
dm3N8_aminos = count_all_by_nucleosome(8,df_flyChrT, amino_acid_dict, amino_acids, dm3_genome)
dm3N9_aminos = count_all_by_nucleosome(9,df_flyChrT, amino_acid_dict, amino_acids, dm3_genome)
#%%
dm3N5to9 = [dm3N5_aminos, dm3N6_aminos, dm3N7_aminos, dm3N8_aminos, dm3N9_aminos]

# Initialize the result dictionary
fly_Average_amino = {}

# Iterate over each key in the dictionaries
for key in dm3N5_aminos.keys():
    # Use zip to aggregate the values across all dictionaries for each key
    # For example: key 'a' -> zip([1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7])
    aggregated_values = zip(*(d[key] for d in dm3N5to9))
    
    # Calculate the average for each position and store it in the result dictionary
    fly_Average_amino[key] = [sum(values) / len(values) for values in aggregated_values]

#%%
pallet_plot("5 - 9", fly_Average_amino)

#%%
'''Storing Average Amino'''
import pickle


with open('Fly Amino 5-9.pkl', 'wb') as file:
    pickle.dump(fly_Average_amino, file)
