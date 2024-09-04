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
            # Read the FASTA file
            for record in SeqIO.parse(filepath, "fasta"):
                # Use the filename (without extension) as the key
                key = os.path.splitext(filename)[0]
                # Store only the sequence as a string
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
        # if i%50 == 0:
        #     print(i)
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

def average_Nucleosome(Nucleosomes):
    average_cyclability = np.array(Nucleosomes)
    average_cyclability = np.average(Nucleosomes, axis=0)
    return average_cyclability


#%%
NucleosomeByGene = Nfinder(df_cerN, df_cerTSS, chr_to_roman)
df_cerTSS["Nucleosome Positions"] = NucleosomeByGene
#%%
N_NucleosomeSequence = Nucleosome_N_Seq(1, df_cerTSS, saccer2_genome, chr_to_roman)

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
    plt.ylim(-0.27, -0.1)
    plt.legend()  # Add a legend

    # Add title if provided
    if title is not None:
        plt.title(title)
    
    plt.show()

def sliding_window_average(arr, window_size=25):
    result = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i+window_size]
        result.append(np.mean(window))
    return np.array(result)

#%%
Cer1_NucleosomeSequences = [item for item in N_NucleosomeSequence if item != 'None'] 
CerN1 = run_model(Cer1_NucleosomeSequences)
CerN1 = average_Nucleosome(CerN1)

#%%
CerN1_windowed = sliding_window_average(CerN1)
plot_cyclability(CerN1, 1, "Cerevisiae")
plot_cyclability(CerN1_windowed, 1, "Cerevisiae")

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
                  codon_table, amino_acid_to_codons, conversion_dict):
    #Creates a copy of the original dictionary to ensure it is being updated
    updated_chromosomes = chromosomes.copy()
    
    for idx in range(len(gene_left)):
        try:
            current_chr = chromosome_id[idx]
            current_chr = conversion_dict[current_chr]
            #print(f"Processing chromosome: {current_chr}")
            
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
                codon = Sequence[j:j+3]
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

                    #Swaps codon for another if available 
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
mutated_saccer2 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman)

test_chrome = saccer2_genome["chrI"]
test_Mchrome = mutated_saccer2["chrI"]      
print(test_chrome[1799128:1799200])
print(test_Mchrome[1799128:1799200])

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
CerM1_NucleosomeSequence = Nucleosome_N_Seq(1, df_cerTSS, mutated_saccer2, chr_to_roman)

#%%
CerM1_NucleosomeSequences = [item for item in CerM1_NucleosomeSequence if item != 'None'] 
CerM1 = run_model(CerM1_NucleosomeSequences)
CerM1 = average_Nucleosome(CerM1)
#%%
print(len(CerN1))
print(len(CerM1))
#%%
plot_cyclability2(CerN1, CerM1, title= "Cyclability around Natural and Mutated Ceverisiae Nucleosome {}")


#%%
def model_a_nucleosome(nuc_num, df, genome, mutated_genome, chr_to_roman):
    N_seq = Nucleosome_N_Seq(nuc_num, df, genome, chr_to_roman)
    N_seq = [item for item in N_seq if item != 'None'] 
    print("Natural Cyclability")
    N_cyc = run_model(N_seq)
    N = average_Nucleosome(N_cyc)
    
    M_seq = Nucleosome_N_Seq(nuc_num, df, mutated_genome, chr_to_roman)
    M_seq = [item for item in M_seq if item != 'None'] 
    print("Codon Randomised Cyclability")
    M_cyc = run_model(M_seq)
    M = average_Nucleosome(M_cyc)
    plot_cyclability2(N, M, title= f"Cyclability around Natural and Mutated Ceverisiae Nucleosome {nuc_num}")
    return N,M

#%%
cerN1, cerM1 = model_a_nucleosome(1, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)

cerN2, cerM2 = model_a_nucleosome(2, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)

cerN3, cerM3 = model_a_nucleosome(3, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)

cerN4, cerM4 = model_a_nucleosome(4, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)

cerN5, cerM5 = model_a_nucleosome(5, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)
#%%
cerN6, cerM6 = model_a_nucleosome(6, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)

cerN7, cerM7 = model_a_nucleosome(7, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)

cerN8, cerM8 = model_a_nucleosome(8, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)

cerN9, cerM9 = model_a_nucleosome(9, df_cerTSS, saccer2_genome, mutated_saccer2, chr_to_roman)

#%%
cerNlist = [cerN1, cerN2, cerN3, cerN4, cerN5, cerN6, cerN7, cerN8, cerN9]
cerMlist = [cerM1, cerM2, cerM3, cerM4, cerM5, cerM6, cerM7, cerM8, cerM9]

#%%
'''Storing lists of cyclability'''

import pickle

with open('cerNlist.pkl', 'wb') as file:
    pickle.dump(cerNlist, file)

with open("cerMlist.pkl", "wb") as file:
    pickle.dump(cerMlist, file)
    


#%%
'''Loading Lists of cyclability'''

import pickle 
with open('cerNlist.pkl', 'rb') as file:
    cerNlist = pickle.load(file)
    
with open('cerMlist.pkl', 'rb') as file:
    cerMlist = pickle.load(file)
    


#%%
for i in range(0,len(cerNlist)):
    plot_cyclability2(cerNlist[i], cerMlist[i])




