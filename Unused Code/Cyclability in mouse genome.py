'''
M musculus genome analysis 
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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
column_names = ["chrom", "source", "feature", "Left TS", "Right TS", "score", "strand", "frame", "attribute"]
gtf_file = 'mm9.knownGene.gtf'
df = pd.read_csv(gtf_file, sep='\t', comment='#', names=column_names)
transcripts_df = df[df['feature'] == 'transcript']



df_mouseTS = transcripts_df

#%%
'''Filtering Genes by size'''
df_mouseTS["Length"] = df_mouseTS["Right TS"] - df_mouseTS["Left TS"]
pd.to_numeric(df_mouseTS["Length"])
df_mouseTS = df_mouseTS[df_mouseTS["Length"] > 100] 
df_mouseTS = df_mouseTS[df_mouseTS["Length"] < 15000].reset_index()
    

#%%
''' Filtering by chromosomes '''
target_chromosomes = ["chr1", "chr2", "chr3", "chr4", "Chr5", "chr6"]
df_mouseChr1 = df_mouseTS[df_mouseTS["chrom"].isin(target_chromosomes)].reset_index()


'''storing columns for codon randomisation '''
mm9_Left = df_mouseChr1["Left TS"]
mm9_Right = df_mouseChr1["Right TS"]
mm9_chrom = df_mouseChr1["chrom"]
mm9_strand = df_mouseChr1["strand"]

#%%
'''
Nucleosome Positions 
'''

# Define column names
column_names = ['chromosome', 'center_position']

# Load the text file into a DataFrame
df_mouseN = pd.read_csv('unique.map_95pc.txt', delim_whitespace=True, header=None, names=column_names)

#%%
'''
Mouse Genome
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

directory = "mm9 Chromosomes"
mm9_genome = load_fasta_files_to_dict(directory)

#%%
'''
Dictionary for matching chromosomes between datasets
'''
chr_to_num = {'chr1': '1', 'chr2': '2', 'chr3': '3', 'chr4': '4', 
                'chr5': '5', 'chr6': '6', 'chr7': '7', 'chr8': '8', 
                'chr9': '9', 'chr10': '10', 'chr11': '11', 'chr12': '12', 
                'chr13': '13', 'chr14': '14', 'chr15': '15', 'chr16': '16',
                'chr17': '17', 'chr18': '18', 'chr19': '19',} #'chrX': '20',
                #'chrY': '21', 'chrM': '22'}


num_to_chr = {'1': 'chr1', '2': 'chr2', '3': 'chr3', '4': 'chr4',
    '5': 'chr5', '6': 'chr6', '7': 'chr7', '8': 'chr8',
    '9': 'chr9', '10': 'chr10', '11': 'chr11', '12': 'chr12',
    '13': 'chr13', '14': 'chr14', '15': 'chr15', '16': 'chr16',
    '17': 'chr17', '18': 'chr18', '19': 'chr19', '20': 'chrX',
    '21': 'chrY', '22': 'chrM'
}

print(chr_to_num["chr7"])

print(df_mouseTS.columns)
print(df_mouseTS.index) 
print(df_mouseTS.dtypes)
#%%
'''
Finding all nucleosomes within each gene 
'''

def Nfinder(dfN, dfT, conversion_dict):
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
            chrom = int(conversion_dict[Chromosome[i]])
            Nucleosomes = dfN.loc[dfN["chromosome"]== chrom, ["center_position"]]
            #print(Nucleosomes)
            Nucleosomes = Nucleosomes["center_position"]
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

def Nucleosome_N_Seq(Nuc_Number,df, genome, conversion_dict):
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

NucleosomeByGene = Nfinder(df_mouseN, df_mouseChr1, chr_to_num)
df_mouseChr1["Nucleosome Positions"] = NucleosomeByGene

#%%
N_NucleosomeSequence = Nucleosome_N_Seq(1, df_mouseChr1, mm9_genome, chr_to_num)

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

#%%
'''
Loading Plotting Functions 
'''
def plot_cyclability(values, nuc_num, name):
    x_values = np.linspace(-175, 175, len(values))
    plt.figure(figsize=(12, 6)) 
    plt.plot(x_values, values, color="blue")
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')
    plt.title(f'Cyclability Around Nucleosome {nuc_num} {name}')
    plt.xlim(-200, 200)
    plt.ylim(-0.15, -0.1)
    
    plt.show()

def plot_cyclability2(values1, values2, title=None):
    x_values = np.linspace(-175, 175, len(values1))
    
    if len(values1) != len(values2):
        raise ValueError("Both sequences must have the same length.")
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(x_values, values1, label='Natural', color='blue')
    plt.plot(x_values, values2, label='Mutated', color='red')
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')

    plt.xlim(-200, 200)
    plt.ylim(-0.17, -0.1)
    plt.legend()  


    if title is not None:
        plt.title(title)
    
    plt.show()


#%%
Mouse1_NucleosomeSequences = [item for item in N_NucleosomeSequence if item != 'None'] 
MouseN1 = run_model(Mouse1_NucleosomeSequences)
mm9N1 = average_Nucleosome(MouseN1)

#%%
plot_cyclability(mm9N1, 1, "Mus Musculus")

#%%
'''
Genome Mutation
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
    'TAT': 'Y', 'TAC': 'Y', 'TAA': 'Stop', 'TAG': 'Stop', 
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 
    'TGT': 'C', 'TGC': 'C', 'TGA': 'Stop', 'TGG': 'W', 
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}


amino_acid_to_codons = {}
for codon, amino_acid in codon_table.items():
    if amino_acid not in amino_acid_to_codons:
        amino_acid_to_codons[amino_acid] = []
    amino_acid_to_codons[amino_acid].append(codon)

        


def mutate_genome(chromosomes, gene_left, gene_right, chromosome_id, direction, 
                  codon_table, amino_acid_to_codons, conversion_dict):
    updated_chromosomes = chromosomes.copy()
    
    for idx in range(len(gene_left)):
        if idx%500 == 0:
            print(idx)
        try:
            current_chr = chromosome_id[idx]
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


                    if len(possible_codons) > 1:
                        new_codon = random.choice([c for c in possible_codons if c != codon])
                    else:
                        new_codon = codon

                    mutated_sequence.append(new_codon)
                else:
                    mutated_sequence.append(codon)

            mutated_sequence = "".join(mutated_sequence)
            if direction[idx] == "-":
                x = Seq(mutated_sequence)
                mutated_sequence = str(x.reverse_complement())
            
            chrome_up = chromosome[:int(gene_left[idx])]
            chrome_down = chromosome[int(gene_right[idx]):]
            updated_chromosome = chrome_up + mutated_sequence + chrome_down
            updated_chromosomes[current_chr] = updated_chromosome
        
        except Exception as e:
            print(f"Fault: {e}")
    
    return updated_chromosomes
#%%
mutated_mm9 = mutate_genome(mm9_genome, mm9_Left, mm9_Right, 
                                mm9_chrom, mm9_strand, codon_table, 
                                amino_acid_to_codons, chr_to_num)

test_chrome = mm9_genome["chr1"]
test_Mchrome = mutated_mm9["chr1"]      

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
mm9M1_Nucleosome = Nucleosome_N_Seq(1, df_mouseChr1, mutated_mm9, chr_to_num)

#%%
mm9M1_NucleosomeSequences = [item for item in mm9M1_Nucleosome if item != 'None'] 
mm9M1 = run_model(mm9M1_NucleosomeSequences)
mm9M1 = average_Nucleosome(mm9M1)
#%%
print(len(mm9N1))
print(len(mm9M1))
#%%
plot_cyclability2(mm9N1, mm9M1, title= "Cyclability around Nucleosome 1 M. Musculus chr1-6")


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
    return M

#%%
mm9N2 = model_a_nucleosome(2, df_mouseChr1, mm9_genome, mutated_mm9, chr_to_num)

mm9N3 = model_a_nucleosome(3, df_mouseChr1, mm9_genome, mutated_mm9, chr_to_num)

mm9N4 = model_a_nucleosome(4, df_mouseChr1, mm9_genome, mutated_mm9, chr_to_num)

mm9N5 = model_a_nucleosome(5, df_mouseChr1, mm9_genome, mutated_mm9, chr_to_num)

mm9N6 = model_a_nucleosome(6, df_mouseChr1, mm9_genome, mutated_mm9, chr_to_num)

mm9N7 = model_a_nucleosome(7, df_mouseChr1, mm9_genome, mutated_mm9, chr_to_num)

mm9N8 = model_a_nucleosome(8, df_mouseChr1, mm9_genome, mutated_mm9, chr_to_num)

mm9N9 = model_a_nucleosome(9, df_mouseChr1, mm9_genome, mutated_mm9, chr_to_num)
