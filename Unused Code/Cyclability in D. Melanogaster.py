'''
D melanogaster genome analysis 
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
from scipy.stats import pearsonr
from scipy.stats import linregress
from difflib import SequenceMatcher
from collections import Counter
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
target_chromosomes = ["chr2R", "chr4"]
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


def run_model(seqs): #Allows it to take a list of sequences as input
    #num_seqs = len(seqs) 
    #seq_length = len(seqs[0])
    #subseq_length = 50
    #num_subseqs = seq_length - subseq_length + 1
    #the variables above enabled checking of functionality 
    
    # Initialize an array to accumulate the cyclability values
    accumulated_cyclability = []
    # Extract the model number from the option string
    option = "C0free prediction"
    modelnum = int(re.findall(r'\d+', option)[0])
    # Load the model
    model = load_model(modelnum)
    x =1
    # Process each sequence
    for seq in seqs:
        #A simple counter to keep track of progress
        if x%200 == 0:
            print(x)
        x = x+1
        
        # Create a list of subsequences of length 50
        list50 = [seq[i:i+50] for i in range(len(seq) - 50 + 1)]
        # Make predictions using the model
        cNfree = pred(model, list50)
        prediction = list(cNfree)
        # Accumulate the cyclability values
        accumulated_cyclability.append(prediction)
        
    
    return accumulated_cyclability

#%%
'''
Loading Plotting Functions 
'''
def plot_cyclability(values, nuc_num, name):
    x_values = np.linspace(-175, 175, len(values))  # Generate x-values from -175 to 200
    plt.figure(figsize=(12, 6))  # Create figure with specified size
    plt.plot(x_values, values, color="blue")
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')
    plt.title(f'Cyclability Around Nucleosome {nuc_num} {name}')
    plt.xlim(-200, 200)  # Set x-axis limits
    plt.ylim(-0.26, -0.1)
    
    plt.show()

def plot_cyclability2(values1, values2, title=None):
    # Generate x-values for both sequences (same for both sequences)
    x_values = np.linspace(-175, 175, len(values1))
    
    # Ensure that both sequences have the same length
    if len(values1) != len(values2):
        raise ValueError("Both sequences must have the same length.")
    
    plt.figure(figsize=(12, 6))  # Create figure with specified size
    
    # Plot both sequences with different colors
    plt.plot(x_values, values1, label='Natural', color='blue')
    plt.plot(x_values, values2, label='Mutated', color='red')
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')
    # Customize plot
    plt.xlim(-200, 200)  # Set x-axis limits
    plt.ylim(-0.26, -0.1)
    plt.legend()  # Add a legend

    # Add title if provided
    if title is not None:
        plt.title(title)
    
    plt.show()


#%%
fly1_NucleosomeSequences = [item for item in N_NucleosomeSequence if item != 'None'] 
flyN1 = run_model(fly1_NucleosomeSequences)
dm3N1 = average_Nucleosome(flyN1)

#%%
plot_cyclability(dm3N1, 1, "Drosophila Melanogaster")

#%%
'''
Genome Mutation
'''

# code from pombe 
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

# Create a reverse dictionary mapping amino acids to their codons
amino_acid_to_codons = {}
for codon, amino_acid in codon_table.items():
    if amino_acid not in amino_acid_to_codons:
        amino_acid_to_codons[amino_acid] = []
    amino_acid_to_codons[amino_acid].append(codon)

        


def mutate_genome(chromosomes, gene_left, gene_right, chromosome_id, direction, 
                  codon_table, amino_acid_to_codons):
    # Create a copy of the original dictionary to ensure it is being updated
    updated_chromosomes = chromosomes.copy()
    
    for idx in range(len(gene_left)):
        if idx%500 == 0:
            print(idx)
        try:
            current_chr = chromosome_id[idx]
            #current_chr = conversion_dict[current_chr]
            #print(f"Processing chromosome: {current_chr}")
            
            # Directly use the current_chr as key since it's now in format 'chrI', 'chrII', etc.
            chromosome = updated_chromosomes.get(current_chr)
            if chromosome is None:
                print(f"Chromosome {current_chr} not found in the genome.")
                continue
            
            gene_seq = chromosome[int(gene_left[idx]):int(gene_right[idx])]
            
            if direction[idx] == "+":
                Sequence = gene_seq
            elif direction[idx] == "-":
                Nucleosome_seq_obj = Seq(gene_seq)
                compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
                Sequence = str(compliment_Nuclesome)
            
            # Doing the mutation
            mutated_sequence = []
            for j in range(0, len(Sequence), 3):
                codon = Sequence[j:j+3].upper()
                if len(codon) == 3:
                    amino_acid = codon_table.get(codon, None)
                    if amino_acid is None:
                        print(f"Codon {codon} not found in codon table.")
                        mutated_sequence.append(codon)
                        continue

                    possible_codons = amino_acid_to_codons.get(amino_acid, [])
                    if not possible_codons:
                        print(f"No possible codons found for amino acid {amino_acid}.")
                        mutated_sequence.append(codon)
                        continue

                    # If there is more than one codon for the amino acid, select a different one
                    if len(possible_codons) > 1:
                        new_codon = random.choice([c for c in possible_codons if c != codon])
                    else:
                        new_codon = codon

                    mutated_sequence.append(new_codon)
                else:
                    mutated_sequence.append(codon)

            mutated_sequence = "".join(mutated_sequence)
            # Reverse complement if necessary
            if direction[idx] == "-":
                x = Seq(mutated_sequence)
                mutated_sequence = str(x.reverse_complement())
            
            # Building replacement chromosome
            chrome_up = chromosome[:int(gene_left[idx])]
            chrome_down = chromosome[int(gene_right[idx]):]
            updated_chromosome = chrome_up + mutated_sequence + chrome_down
            updated_chromosomes[current_chr] = updated_chromosome
        
        except Exception as e:
            print(f"Fault: {e}")
    
    return updated_chromosomes
#%%
mutated_dm3 = mutate_genome(dm3_genome, dm3_Left, dm3_Right, 
                                dm3_chrom, dm3_strand, codon_table, 
                                amino_acid_to_codons)

test_chrome = dm3_genome["chr2R"]
test_Mchrome = mutated_dm3["chr2R"]      

import hashlib

str1 = test_chrome
str2 = test_Mchrome

hash1 = hashlib.md5(str1.encode()).hexdigest()
hash2 = hashlib.md5(str2.encode()).hexdigest()

if hash1 == hash2:
    print("The strings are identical.")
else:
    print("The strings are different.")

#%%
dm3M1_Nucleosome = Nucleosome_N_Seq(1, df_flyChrT, mutated_dm3)

#%%
dm3M1_NucleosomeSequences = [item for item in dm3M1_Nucleosome if item != 'None'] 
dm3M1 = run_model(dm3M1_NucleosomeSequences)
dm3M1 = average_Nucleosome(dm3M1)
#%%
print(len(dm3N1))
print(len(dm3M1))
#%%
plot_cyclability2(dm3N1, dm3M1, title= "Cyclability around Nucleosome 1 D. Melanogaster chr2L and chr4")


#%%
def model_a_nucleosome(nuc_num, df, genome, mutated_genome):
    N_seq = Nucleosome_N_Seq(nuc_num, df, genome)
    N_seq = [item for item in N_seq if item != 'None'] 
    print("Natural Cyclability")
    N_cyc = run_model(N_seq)
    N = average_Nucleosome(N_cyc)
    
    M_seq = Nucleosome_N_Seq(nuc_num, df, mutated_genome)
    M_seq = [item for item in M_seq if item != 'None'] 
    print("Codon Randomised Cyclability")
    M_cyc = run_model(M_seq)
    M = average_Nucleosome(M_cyc)
    plot_cyclability2(N, M, title= f"Cyclability around Natural and Mutated Ceverisiae Nucleosome {nuc_num}")
    return M

#%%
dm3N2 = model_a_nucleosome(2, df_flyTS, dm3_genome, mutated_dm3)

dm3N3 = model_a_nucleosome(3, df_flyTS, dm3_genome, mutated_dm3)

dm3N4 = model_a_nucleosome(4, df_flyTS, dm3_genome, mutated_dm3)

dm3N5 = model_a_nucleosome(5, df_flyTS, dm3_genome, mutated_dm3)

dm3N6 = model_a_nucleosome(6, df_flyTS, dm3_genome, mutated_dm3)

dm3N7 = model_a_nucleosome(7, df_flyTS, dm3_genome, mutated_dm3)

dm3N8 = model_a_nucleosome(8, df_flyTS, dm3_genome, mutated_dm3)

dm3N9 = model_a_nucleosome(9, df_flyTS, dm3_genome, mutated_dm3)


