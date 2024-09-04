"""Multiple Codon Randomisation effects"""
'''
Much of the code here is reused from predicting cerevisiae cyclizability
Now looking at 10 different codon randomisations 
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
from liftover import ChainFile

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
'''
Storing the sacCer2 genome 
'''
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

directory = 'sacCer2 Chromosomes'
saccer2_genome = load_fasta_files_to_dict(directory)


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
#Here focusing on the +5 Nucleosome
N_NucleosomeSequence = Nucleosome_N_Seq(5, df_cerTSS, saccer2_genome, chr_to_roman)

#%%
'''
Loading Model and reusing functions from previous script
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
        #A simple counter to keep track of progress
        if x%500 == 0:
            print(x)
        x = x+1
        
        list50 = [seq[i:i+50] for i in range(len(seq) - 50 + 1)]
        cNfree = pred(model, list50)
        prediction = list(cNfree)
        accumulated_cyclability.append(prediction)
        
    
    return accumulated_cyclability


#%%
'''Running model for Natural +5 nucleosome cerevisiae'''
Cer5_NucleosomeSequences = [item for item in N_NucleosomeSequence if item != 'None'] 
CerN5 = run_model(Cer5_NucleosomeSequences)
CerN5 = average_Nucleosome(CerN5)


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
                  codon_table, amino_acid_to_codons, conversion_dict, seed):
    # Create a copy of the original dictionary to ensure it is being updated
    updated_chromosomes = chromosomes.copy()
    
    for idx in range(len(gene_left)):
        try:
            current_chr = chromosome_id[idx]
            current_chr = conversion_dict[current_chr]
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

                    if len(possible_codons) > 1:
                        random.seed(idx+seed) #Makes the randomness determined by seed provided
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
                                amino_acid_to_codons, chr_to_roman, 1)

mutated_saccer2_1 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 1)

#Previously the difference to original was tested now testing that seeds maintain

test_chrome = mutated_saccer2["chrI"]
test_Mchrome = mutated_saccer2_1["chrI"]      


import hashlib

str1 = test_chrome
str2 = test_Mchrome

hash1 = hashlib.md5(str1.encode()).hexdigest()
hash2 = hashlib.md5(str2.encode()).hexdigest()

if hash1 == hash2:
    print("The strings are identical.")
else:
    print("The strings are different.")

#Adding the seed made the random genomes the same so can be reused if needed 

#%%
'''Generating more codon randomised genomes'''

mutated_saccer2_2 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 2)

mutated_saccer2_3 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 3)

mutated_saccer2_4 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 4)

mutated_saccer2_5 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 5)

#%%
mutated_saccer2_6 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 6)

mutated_saccer2_7 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 7)

mutated_saccer2_8 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 8)

mutated_saccer2_9 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 9)

mutated_saccer2_10 = mutate_genome(saccer2_genome, saccer_2_ORF_Left, saccer_2_ORF_Right, 
                                saccer2_chrom, saccer2_direction, codon_table, 
                                amino_acid_to_codons, chr_to_roman, 10)    

#%%
#checking mutated genomes are different
test_chrome = mutated_saccer2_3["chrI"]
test_Mchrome = mutated_saccer2_1["chrI"]      


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

def multiple_randomisations(genome):
    print("Randomised Cyclizability")
    CerM1_NucleosomeSequence = Nucleosome_N_Seq(5, df_cerTSS, genome, chr_to_roman)
    CerM1_NucleosomeSequences = [item for item in CerM1_NucleosomeSequence if item != 'None'] 
    CerM1 = run_model(CerM1_NucleosomeSequences)
    CerM1average = average_Nucleosome(CerM1)
    return CerM1, CerM1average

#%%
#Here M1 does not denote +1 nucleosome but the 1 genome randomisation
M1raw, M1av = multiple_randomisations(mutated_saccer2_1)
M2raw, M2av = multiple_randomisations(mutated_saccer2_2)
M3raw, M3av = multiple_randomisations(mutated_saccer2_3)
M4raw, M4av = multiple_randomisations(mutated_saccer2_4)
M5raw, M5av = multiple_randomisations(mutated_saccer2_5)

Mav1to5 = [M1av,M2av,M3av,M4av,M5av]
#%%
'''Storing lists of cyclability'''

import pickle

with open('cerevisiae randomisations 1-5', 'wb') as file:
    pickle.dump(Mav1to5, file)

Mraw1to5 = [M1raw, M2raw, M3raw, M4raw, M5raw]

with open('cerevisiae randomisations 1-5 raw', 'wb') as file:
    pickle.dump(Mraw1to5, file)

#%%
M6raw, M6av = multiple_randomisations(mutated_saccer2_6)
M7raw, M7av = multiple_randomisations(mutated_saccer2_7)
M8raw, M8av = multiple_randomisations(mutated_saccer2_8)
M9raw, M9av = multiple_randomisations(mutated_saccer2_9)
M10raw, M10av = multiple_randomisations(mutated_saccer2_10)

Mav6to10 = [M6av,M7av,M8av,M9av,M10av]
#%%
'''Storing lists of cyclability'''

import pickle

with open('cerevisiae randomisations 6-10', 'wb') as file:
    pickle.dump(Mav6to10, file)

#%%
N = [CerN5]
Plot_List5 = N + Mav1to5
Plot_List10 = N + Mav1to5 + Mav6to10
#%%
'''Ploting Natural against all the codon randomisations'''
#Making a colour pallete for the randomisations
import colorsys

base_colors = plt.cm.Set1.colors
def increase_saturation(rgb, factor=1.5):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s = np.clip(s * factor, 0, 1)
    return colorsys.hls_to_rgb(h, l, s)

vibrant_colors = [increase_saturation(color) for color in base_colors]


def plot_randomisations(lists):
    plt.figure(figsize=(12, 8))
    
    for i in range(len(lists)):
        x_values = np.linspace(-175, 175, len(lists[i]))
        if i == 0:
            plt.plot(x_values, lists[i], label='Natural', 
                     color='black', linewidth = 2)
        else:
            plt.plot(x_values, lists[i], label=f'Random {i}', 
                     color=vibrant_colors[i-1])
            
    plt.xlabel('Distance from Nucleosome Centre (BP)', fontsize=24, 
                fontname='Verdana')
    plt.ylabel('Cyclizability', fontsize=24, 
                fontname='Verdana')
    plt.xlim(-200, 200)
    plt.ylim(-0.26, -0.17)
    plt.legend(loc='best', fontsize=15, frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.show()

#%%
plot_randomisations(Plot_List5)

#%%
plot_randomisations(Plot_List10)
'''All of the randomisations were very similar to eachother but not natural'''
'''The plot of 5 was chosen purely for less visual clutter'''



#%%
'''Pombe'''

df_TSS = pd.read_csv("41594_2010_BFnsmb1741_MOESM9_ESM (COPY).csv")
df_TSS["Left TS"] = df_TSS["Left txn border"]
df_TSS["Right TS"] = df_TSS["Right txn border"]
df_TSS["Strand"] = df_TSS["Orientation"]
df_TSS["Chrom"] = df_TSS["Chromosome"]

df_nucleosome = pd.read_csv("sd01.csv")

#Store seperate variables for mutation
Pombe_ORF_Left = df_TSS["Left ORF border"]
Pombe_ORF_Right = df_TSS["Right ORF border"] 
Pombe_Chrome = df_TSS["Chrom"]
Pombe_direction = df_TSS["Strand"]


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
    
chrome_to_roman = {"chromosome1": "chrI", "chromosome2": "chrII", "chromosome3": "chrIII"
    }
chrome_to_num = {"chromosome1": "1", "chromosome2": "2", "chromosome3": "3"
    }
    
#%%
NucleosomeByGene = Nfinder(df_nucleosome, df_TSS, chrome_to_roman)
df_TSS["Nucleosome Positions"] = NucleosomeByGene

#%%
Pombe_NucleosomeSequence = Nucleosome_N_Seq(5, df_TSS, chromosomes, chrome_to_num)

#%%
print("Pombe")
Pombe2000_NucleosomeSequences = [item for item in Pombe_NucleosomeSequence if item != 'None'] 
PombeN2000raw = run_model(Pombe2000_NucleosomeSequences)
#%%
PombeN = average_Nucleosome(PombeN2000raw)

#%%
p_mutated_genome_1 = mutate_genome(chromosomes, Pombe_ORF_Left, Pombe_ORF_Right, 
                                Pombe_Chrome, Pombe_direction, codon_table, 
                                amino_acid_to_codons, chrome_to_num, 1)

p_mutated_genome_2 = mutate_genome(chromosomes, Pombe_ORF_Left, Pombe_ORF_Right, 
                                Pombe_Chrome, Pombe_direction, codon_table, 
                                amino_acid_to_codons, chrome_to_num, 2)

p_mutated_genome_3 = mutate_genome(chromosomes, Pombe_ORF_Left, Pombe_ORF_Right, 
                                Pombe_Chrome, Pombe_direction, codon_table, 
                                amino_acid_to_codons, chrome_to_num, 3)

p_mutated_genome_4 = mutate_genome(chromosomes, Pombe_ORF_Left, Pombe_ORF_Right, 
                                Pombe_Chrome, Pombe_direction, codon_table, 
                                amino_acid_to_codons, chrome_to_num, 4)

p_mutated_genome_5 = mutate_genome(chromosomes, Pombe_ORF_Left, Pombe_ORF_Right, 
                                Pombe_Chrome, Pombe_direction, codon_table, 
                                amino_acid_to_codons, chrome_to_num, 5)

#%%
def multiple_randomisations2(genome):
    print("Randomised Cyclizability")
    CerM1_NucleosomeSequence = Nucleosome_N_Seq(5, df_TSS, genome, chrome_to_num)
    CerM1_NucleosomeSequences = [item for item in CerM1_NucleosomeSequence if item != 'None'] 
    CerM1 = run_model(CerM1_NucleosomeSequences)
    CerM1average = average_Nucleosome(CerM1)
    return CerM1, CerM1average


#%%
#Here P1 does not denote +1 nucleosome but the 1 genome randomisation
P1raw, P1av = multiple_randomisations2(p_mutated_genome_1)
P2raw, P2av = multiple_randomisations2(p_mutated_genome_2)
P3raw, P3av = multiple_randomisations2(p_mutated_genome_3)
P4raw, P4av = multiple_randomisations2(p_mutated_genome_4)
P5raw, P5av = multiple_randomisations2(p_mutated_genome_5)

Pav1to5 = [P1av,P2av,P3av,P4av,P5av]

#%%
N = [PombeN] #Change this to just PombeN
Plot_List5 = N + Pav1to5

#%%
plot_randomisations(Plot_List5)

