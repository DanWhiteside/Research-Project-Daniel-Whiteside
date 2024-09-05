
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
from matplotlib.gridspec import GridSpec
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK']='True' #Stops 
#%%
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
    #Combinging the cyclizability of each list item
    accumulated_cyclability = []
    #Getting model number from the option string
    option = "C0free prediction"
    modelnum = int(re.findall(r'\d+', option)[0])
    #Loading model
    model = load_model(modelnum)
    x =1
    #Predicting cyclizability for each sequence
    for seq in seqs:
        #A simple counter to keep track of progress
        if x%200 == 0:
            print(x)
        x = x+1
        #Breaks sequence down into 1-50, 2-51, etc.
        list50 = [seq[i:i+50] for i in range(len(seq) - 50 + 1)]
        #Makes Prediction
        cNfree = pred(model, list50)
        prediction = list(cNfree)
        #Combined all the cyclizabilities
        accumulated_cyclability.append(prediction)
        
    
    return accumulated_cyclability



#POMBE
#Importing the relevent data and storing it 
df_nucleosome = pd.read_csv("sd01.csv")
df_TSS = pd.read_csv("41594_2010_BFnsmb1741_MOESM9_ESM (COPY).csv")
#df_TSS.head

#CERIVISIAE
df_cer = pd.read_csv("41586_2021_3314_MOESM3_ESM (edited copy).csv")


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
#CERIVISIAE
cer_direction = list(df_cer["Strand"])
cer_chromosome = list(df_cer["Chrom"])
cer_left = list(df_cer["Experiment_Left"])
cer_right = list(df_cer["Experiment_Right"])
cer_nuc1 = list(df_cer["PlusOne_Dyad"])
cer_COD_left = list(df_cer["SGD_Left"])
cer_COD_right = list(df_cer["SGD_Right"])

#%%

#DIctionary for pombe chromosomes
chromosomes = {}
fasta_file = "GCA_000002945.2_ASM294v2_genomic.fna"
chromosome_keys = ['1', '2', '3', 'mitochondria']

#Parsing Fasta file 
for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
    if idx < len(chromosome_keys):
        key = chromosome_keys[idx]
    else:
        key = f'chromosome_{idx + 1}'
    chromosomes[key] = str(record.seq).upper()


#%%
'''This code was used to load cerevisiae chromosomes but the genome used has sinced been changed'''
cer_chromosomes = {}
cer_fasta_file = "GCF_000146045.2_R64_genomic.fna"
cer_chromosome_keys = ['1', '2', '3']

cer_mitochondria_key = "mitochondria"
for idx, record in enumerate(SeqIO.parse(cer_fasta_file, "fasta")):
    if cer_mitochondria_key in record.description.lower():
        key = cer_mitochondria_key
    elif idx < len(cer_chromosome_keys):
        key = cer_chromosome_keys[idx]
    else:
        key = f'{idx + 1}'
    cer_chromosomes[key] = str(record.seq).upper()


#%%
#function for calculating the cyclability of a given nucleosome 
def sequencer(position, direction, chromosome):
    position = position - 1
    Left_pos = position - 200
    Right_pos = position + 201 
    Nucleosome_sequence = chromosome[Left_pos:Right_pos]
    if direction == "+":
        Sequence = Nucleosome_sequence.upper()
    elif direction == "-":
        Nucleosome_seq_obj = Seq(Nucleosome_sequence)
        compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
        Sequence = str(compliment_Nuclesome).upper()
    return Sequence

def sequencer500(position, direction, chromosome):
    position = position - 1
    Left_pos = position - 500
    Right_pos = position + 501 
    Nucleosome_sequence = chromosome[Left_pos:Right_pos]
    if direction == "+":
        Sequence = Nucleosome_sequence
    elif direction == "-":
        Nucleosome_seq_obj = Seq(Nucleosome_sequence)
        compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
        Sequence = str(compliment_Nuclesome)
    return Sequence

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

def sequencerFilter(position, direction, chromosome, left, right):
    position = position - 1
    Left_pos = position - 1000
    Right_pos = position + 1001 
    if position < left + 100 and position > right + 100:
        Nucleosome_sequence = chromosome[Left_pos:Right_pos]
        if direction == "+":
            Sequence = Nucleosome_sequence
        elif direction == "-":
            Nucleosome_seq_obj = Seq(Nucleosome_sequence)
            compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
            Sequence = str(compliment_Nuclesome)
    else:
        Sequence = "None"
    return Sequence

#function that averages an array containing multiple lists 
def average_Nucleosome(Nucleosomes):
    average_cyclability = np.array(Nucleosomes)
    average_cyclability = np.average(Nucleosomes, axis=0)
    return average_cyclability


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

def NucX_sequencer1000(Nuc_Number, dataframe, chromosomes):
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
                sequence = sequencer1000(Nucleosome_position, Direction[i], chromosome)
                if Char_to_check not in sequence:
                    Sequences.append(sequence)
                else:
                    Sequences.append("None")
            else:
                Sequences.append("None")
        except:
            Sequences.append("None")
    return Sequences

def NucX_sequencer500(Nuc_Number, dataframe, chromosomes):
    Position = dataframe["Nucleosome"]
    Direction = dataframe["Orientation"]
    Chromosome_id = dataframe["Chromosome"]
    Sequences = []
    Char_to_check = "N"
    for i in range(0,10):
        try:
            Current_nucleosomes = Position[i]
            print(Current_nucleosomes)
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
                print(Nucleosome_position)
                sequence = sequencer500(Nucleosome_position, Direction[i], chromosome)
                if Char_to_check not in sequence:
                    Sequences.append(sequence)
                else:
                    Sequences.append("None")
            else:
                Sequences.append("None")
        except:
            Sequences.append("None")
    return Sequences

def Nuc1_sequencerFilter(Nuc_Number, dataframe, chromosomes, left, right):
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
                sequence = sequencerFilter(Nucleosome_position, Direction[i], chromosome, left[i], right[i])
                if Char_to_check not in sequence:
                    Sequences.append(sequence)
                else:
                    Sequences.append("None")
            else:
                Sequences.append("None")
        except:
            Sequences.append("None")
    return Sequences

#%%
'''
The main plotting functions 
'''
def plot_cyclability(values, nuc_num, name):
    x_values = np.linspace(-175, 175, len(values)) 
    plt.figure(figsize=(12, 6))  
    plt.plot(x_values, values, color="blue")
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')
    plt.title(f'Cyclability Around Nucleosome {nuc_num} {name}')
    plt.xlim(-200, 200)
    
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
    # Customize plot
    plt.xlim(-200, 200)
    plt.ylim(-0.27, -0.1)
    plt.legend()  
    if title is not None:
        plt.title(title)
    
    plt.show()

def plot_cyclability1000(values, nuc_num, name):
    x_values = np.linspace(-975, 975, len(values))  
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, values)
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')
    plt.title(f'Cyclability Around Nucleosome {nuc_num} {name}')
    plt.xlim(-1000, 1000) 
    plt.ylim(-0.27, -0.1)
    plt.show()

def plot_cyclability2_1000(values1, values2, title=None):
    x_values = np.linspace(-975, 975, len(values1))
    
    if len(values1) != len(values2):
        raise ValueError("Both sequences must have the same length.")
    
    plt.figure(figsize=(12, 6))      
    plt.plot(x_values, values1, label='Natural', color='blue')
    plt.plot(x_values, values2, label='Mutated', color='red')
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')
    plt.xlim(-1000, 1000)
    plt.ylim(-0.27, -0.1)
    plt.legend()

    if title is not None:
        plt.title(title)
    
    plt.show()

#%%
'''
Finding the Nucleosomes of all possible genes in Pombe
'''
Nucleosome_Positions = Nucleosome_TSS_finder(Left_txn, Right_txn, df_nucleosome, TS_chromosome)
df_TSS["Nucleosome"] = Nucleosome_Positions

#%%
#vaildating functions
def gene_checker(n):
    gene = Nucleosome_Positions[n]
    print(gene)
    print(TS_directions[n])
    if TS_directions[n] == '+':
        print(gene[1-1])
    else:
        print(gene[-1])
    print(Left_txn[n], Right_txn[n])

gene_checker(6)


#%%
'''
A range of checks for the accuracy of functions 

'''
#Checking the first 100 genes for any problems

for i in range(0,100):
    gene = Nucleosome_Positions[i]
    x = Left_txn[i]
    y = Right_txn[i]
    if x % 1 == 0 and y % 1 == 0: 
        print(gene)
        print(Left_txn[i], Right_txn[i])
        print(TS_directions[i])
        print("")

#Checking if a sequence produced by the function matches the results of seperately finding the sequnece
NucV = NucX_sequencer500(1, df_TSS, chromosomes)
gene6 = NucV[6]
print(gene6)

chromosome1 = chromosomes["1"]
position_left = 1809186 - 500
position_right = 1809186 + 500 
forward = Seq(chromosome1[position_left:position_right])
print(forward.reverse_complement())
print("")
print(sequencer500(1809186, "-", chromosome1))



#%%
'''
Collecting sequences for the relevent Nucleosome (here plus 1)
'''

Nucleosome1_TS = NucX_sequencer(1, df_TSS, chromosomes)  #400 bp
#Nucleosome1_TS = NucX_sequencer1000(1, df_TSS, chromosomes) #2000 bp
#Nucleosome1_TS = Nuc1_sequencerFilter(1, df_TSS, chromosomes, Left_txn, Right_txn)

#Removing any entries with missing values 
Nucleosome1_sequences = [item for item in Nucleosome1_TS if item != 'None'] 




#%%
'''
Running the model and seeing if all arays are of equal length
'''
Nuc1_cyc = run_model(Nucleosome1_sequences)

indices_different_length = [i for i, sublist in enumerate(Nuc1_cyc) if len(sublist) != len(Nuc1_cyc[0])]

#Checking if sequences are the same length 
if indices_different_length:
    print("Indices of sublists with different lengths:", indices_different_length)
else:
    print("All sublists have the same length.")
     

#%%
'''
Averaging across either the 400 or 2000 bp sorounding Pombe Nucleosome 1
'''
Nucleosome1 = average_Nucleosome(Nuc1_cyc)
 

plot_cyclability(Nucleosome1, 1, "Pombe")
#plot_cyclability1000(Nucleosome1, 1, "Pombe")


#sample_Nucleosome = Nuc1_cyc[5] #just to see how a single genes cyclability compares to average
#plot_cyclability1000(sample_Nucleosome)


#%%

def sliding_window_average(arr, window_size=50):
    result = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i+window_size]
        result.append(np.mean(window))
    return np.array(result)

def plot_sliding_window_average(data, Nuc_num, name, both = None, window_size=50):
    averaged_data = sliding_window_average(data, window_size)
    x_values = np.linspace(-1000, 1000, len(data))
    x_avg_values = np.linspace(-1000, 1000, len(averaged_data))
    if both == None:
        plt.figure(figsize=(12, 6))
        plt.plot(x_avg_values, averaged_data, label='Sliding Window Average', linestyle='--')
        plt.ylim([min(data), max(data)])
        plt.ylim(-0.25, -0.1)
        plt.xlabel('Distance from Nucleosome Centre (BP)')
        plt.ylabel('Cyclability')
        plt.title(f'Sliding Window Average: Nucleosome {Nuc_num} {name}')
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(x_avg_values, averaged_data, label='Sliding Window Average', linestyle='--')
        plt.plot(x_values, data, label="Raw Data")
        plt.xlabel('Distance from Nucleosome Centre (BP)')
        plt.ylabel('Cyclability')
        plt.title(f'Sliding Window Average: Nucleosome {Nuc_num} {name}')
        plt.ylim(-0.25, -0.1)
        plt.legend()
        plt.show()

def sliding_window_mutated(data, Mdata, Nuc_num, name, size, window_size=50):
    averaged_data = sliding_window_average(data, window_size)
    averaged_Mdata = sliding_window_average(Mdata, window_size)
    xmin = 25 - size
    xmax = size - 25
    x_avg_values = np.linspace(xmin, xmax, len(averaged_data))
    x_avg_valuesM = np.linspace(xmin, xmax, len(averaged_Mdata))
    plt.figure(figsize=(12, 6))
    plt.plot(x_avg_values, averaged_data, label='Natural', color = "Blue")
    plt.plot(x_avg_valuesM, averaged_Mdata, label="Mutated", color = "red")
    plt.xlabel('Distance from Nucleosome Centre (BP)')
    plt.ylabel('Cyclability')
    plt.title(f'Sliding Window Average: Nucleosome {Nuc_num} {name}')
    plt.xlim(-size, size)
    plt.ylim(-0.27, -0.1)
    plt.legend()
    plt.show()

    
    
#%%

plot_sliding_window_average(Nucleosome1, 1, "Pombe")



#%%
'''
Altering the NucX_sequencer code for CERIVISIAE +1

This was done before I had gained access to the full map of nucleosome positions in cerevisiae
'''
def cer_sequencer(dataframe, chromosomes, Nuc_Num):
    if Nuc_Num == 1:
        Position = dataframe["PlusOne_Dyad"]
    else:
        Position = dataframe["NFR/NDR_Midpoint"]
    Direction = dataframe["Strand"]
    Chromosome_id = dataframe["Chrom"]
    Sequences = []
    Char_to_check = "N"
    for i in range(0,len(Position)):
        print(i)   
        try:
            current_chr = str(Chromosome_id[i])
            chr_split = current_chr.split("chr")
            chr_num = chr_split[-1]
            chromosome = chromosomes[chr_num]
            try:
                sequence = sequencer(int(Position[i]), Direction[i], chromosome)
                
                if Char_to_check not in sequence:
                    Sequences.append(sequence)
                else:
                   Sequences.append("None")
            except:
                Sequences.append("None")
        except:
            Sequences.append("None")
    return Sequences

def cer_sequencer1000(dataframe, chromosomes, Nuc_Num):
    if Nuc_Num == 1:
        Position = dataframe["PlusOne_Dyad"]
    else:
        Position = dataframe["NFR/NDR_Midpoint"]
    Direction = dataframe["Strand"]
    Chromosome_id = dataframe["Chrom"]
    Sequences = []
    Char_to_check = "N"
    for i in range(0,len(Position)):
        print(i)   
        try:
            current_chr = str(Chromosome_id[i])
            chr_split = current_chr.split("chr")
            chr_num = chr_split[-1]
            chromosome = chromosomes[chr_num]
            try:
                sequence = sequencer1000(int(Position[i]), Direction[i], chromosome)
                
                if Char_to_check not in sequence:
                    Sequences.append(sequence)
                else:
                   Sequences.append("None")
            except:
                Sequences.append("None")
        except:
            Sequences.append("None")
    return Sequences
       
#%%
#cer_Nuc1_seq = cer_sequencer(df_cer, cer_chromosomes, 1)  
cer_Nuc1_seq = cer_sequencer1000(df_cer, cer_chromosomes, 1)          
cer_Nuc1_seq2 = [item for item in cer_Nuc1_seq if item != 'None'] 

#%%
cer_Nuc1_cyc = run_model(cer_Nuc1_seq2)
#%%
indices_different_length = [i for i, sublist in enumerate(cer_Nuc1_cyc) if len(sublist) != len(cer_Nuc1_cyc[0])]

# Print the indices of sublists with different lengths
if indices_different_length:
    print("Indices of sublists with different lengths:", indices_different_length)
else:
    print("All sublists have the same length.")

#%%
del cer_Nuc1_cyc[1017]
#%%
cer_PlusOne = average_Nucleosome(cer_Nuc1_cyc)
#%%
#plot_cyclability(cer_PlusOne)
plot_cyclability1000(cer_PlusOne, 1, "Cerevisiae")




#%%
#NFR/NDR midpoint instead of plus one dyad
cer_Nuc2_seq = cer_sequencer(df_cer, cer_chromosomes, 2)            
cer_Nuc2_seq2 = [item for item in cer_Nuc2_seq if item != 'None'] 

#%%
cer_Nuc2_cyc = run_model(cer_Nuc2_seq2)
#%%
indices_different_length = [i for i, sublist in enumerate(cer_Nuc2_cyc) if len(sublist) != len(cer_Nuc1_cyc[0])]

# Print the indices of sublists with different lengths
if indices_different_length:
    print("Indices of sublists with different lengths:", indices_different_length)
else:
    print("All sublists have the same length.")

#%%
cer_NFR = average_Nucleosome(cer_Nuc2_cyc)
#%%
plot_cyclability(cer_NFR)

#Conclusion is that it's not a nucleosome 












#%%
'''
Generating code to silently mutate DNA sequence  
'''
#Dictionary for replacing codons
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

#Dictionary maps amino acids to codons
amino_acid_to_codons = {}
for codon, amino_acid in codon_table.items():
    if amino_acid not in amino_acid_to_codons:
        amino_acid_to_codons[amino_acid] = []
    amino_acid_to_codons[amino_acid].append(codon)

        


def mutate_genome(chromosomes, gene_left, gene_right, chromosome_id, direction):
    updated_chromosomes = chromosomes.copy()
    
    for idx in range(len(gene_left)):
        try:
            current_chr = str(chromosome_id[idx])
            #print(f"Processing chromosome: {current_chr}")
            match = re.match(r"([a-z]+)([0-9]+)", current_chr, re.I)
            if match:
                items = match.groups()
                chr_num = items[-1]
                #print(f"Chromosome number extracted: {chr_num}")
                
                chromosome = updated_chromosomes.get(chr_num)
                if chromosome is None:
                    
                    continue
                
                
                gene_seq = chromosome[int(gene_left[idx]):int(gene_right[idx])]
                

                if direction[idx] == "+":
                    Sequence = gene_seq
                elif direction[idx] == "-":
                    Nucleosome_seq_obj = Seq(gene_seq)
                    compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
                    Sequence = str(compliment_Nuclesome)
                

                #Doing the mutation
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

                        #Selecting different amino acid when available
                        if len(possible_codons) > 1:
                            new_codon = random.choice([c for c in possible_codons if c != codon])
                        else:
                            new_codon = codon

                        mutated_sequence.append(new_codon)
                    else:
                        mutated_sequence.append(codon)

                mutated_sequence = "".join(mutated_sequence)
                #Reverse complement if necessary
                if direction[idx] == "-":
                    x = Seq(mutated_sequence)
                    mutated_sequence = str(x.reverse_complement())
                
                

                #Building replacement chromosome since strings are immuatable 
                chrome_up = chromosome[:int(gene_left[idx])]
                chrome_down = chromosome[int(gene_right[idx]):]
                updated_chromosome = chrome_up + mutated_sequence + chrome_down
                updated_chromosomes[chr_num] = updated_chromosome
                
                

        except Exception as e:
            print(f"Fault: {e}")

    
    return updated_chromosomes
        
        
        
        
#%%
'''
testing genome mutation function/ Producing Mutated Pombe
'''
mutated_chromosomes = mutate_genome(chromosomes, Left_ORF, Right_ORF, TS_chromosome, TS_directions)

#%%
test_chrome = chromosomes["1"]
test_Mchrome = mutated_chromosomes["1"]      
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
#comparing mutated and non mutated   
cer_mutated_chromosomes = mutate_genome(cer_chromosomes, cer_COD_left, cer_COD_right, cer_chromosome, cer_direction)

#%%
#cer_MNuc1_seq = cer_sequencer(df_cer, cer_mutated_chromosomes, 1)   
cer_MNuc1_seq = cer_sequencer1000(df_cer, cer_mutated_chromosomes, 1)         
cer_MNuc1_seq2 = [item for item in cer_MNuc1_seq if item != 'None'] 

#%%
cer_MNuc1_cyc = run_model(cer_MNuc1_seq2)
#%%
indices_different_length = [i for i, sublist in enumerate(cer_MNuc1_cyc) if len(sublist) != len(cer_MNuc1_cyc[0])]

#Print the indices of sublists with different lengths
if indices_different_length:
    print("Indices of sublists with different lengths:", indices_different_length)
else:
    print("All sublists have the same length.")

#%%
del cer_MNuc1_cyc[1018]

#%%
cer_MPlusOne = average_Nucleosome(cer_MNuc1_cyc)
#%%
plot_cyclability(cer_MPlusOne)

#%%
plot_cyclability2(cer_PlusOne, cer_MPlusOne, "Cerevisiae Plus 1 Nucleosome")
sliding_window_mutated(cer_PlusOne, cer_MPlusOne, 1, "Cerevisiae",1000)


#%%

def model_a_nucleosome(nuc_num, df, genome, mutated_genome):
    N_seq = NucX_sequencer(nuc_num, df, genome)
    N_seq = [item for item in N_seq if item != 'None'] 
    print("Natural Cyclability")
    N_cyc = run_model(N_seq)
    N = average_Nucleosome(N_cyc)
    
    M_seq = NucX_sequencer(nuc_num, df, mutated_genome)
    M_seq = [item for item in M_seq if item != 'None'] 
    print("Codon Randomised Cyclability")
    M_cyc = run_model(M_seq)
    M = average_Nucleosome(M_cyc)
    plot_cyclability2(N, M, title= f"Cyclability around Natural and Mutated Pombe Nucleosome {nuc_num}")
    return N,M




#%%
pombeN1, pombeM1 = model_a_nucleosome(1, df_TSS, chromosomes, mutated_chromosomes)

pombeN2, pombeM2 = model_a_nucleosome(2, df_TSS, chromosomes, mutated_chromosomes)

pombeN3, pombeM3 = model_a_nucleosome(3, df_TSS, chromosomes, mutated_chromosomes)

pombeN4, pombeM4 = model_a_nucleosome(4, df_TSS, chromosomes, mutated_chromosomes)

pombeN5, pombeM5 = model_a_nucleosome(5, df_TSS, chromosomes, mutated_chromosomes)

pombeN6, pombeM6 = model_a_nucleosome(6, df_TSS, chromosomes, mutated_chromosomes)
 
pombeN7, pombeM7 = model_a_nucleosome(7, df_TSS, chromosomes, mutated_chromosomes)

pombeN8, pombeM8 = model_a_nucleosome(8, df_TSS, chromosomes, mutated_chromosomes)

pombeN9, pombeM9 = model_a_nucleosome(9, df_TSS, chromosomes, mutated_chromosomes)

#%%
'''
Averages of Nucleosomes 
'''
'''Lists'''
pombeNlist = [pombeN1,pombeN2,pombeN3,pombeN4,pombeN5,pombeN6,pombeN7,pombeN8,pombeN9]
pombeMlist = [pombeM1,pombeM2,pombeM3,pombeM4,pombeM5,pombeM6,pombeM7,pombeM8,pombeM9]

#%%
'''2 - 4'''
pombeN2to4 = average_Nucleosome(pombeNlist[1:3])
pombeM2to4 = average_Nucleosome(pombeMlist[1:3])
plot_cyclability2(pombeN2to4, pombeM2to4, title= "Average Cyclability Pombe Nucleosomes 2 - 4")

'''5 - 9'''
pombeN5to9 = average_Nucleosome(pombeNlist[4:8])
pombeM5to9 = average_Nucleosome(pombeMlist[4:8])
plot_cyclability2(pombeN5to9, pombeM5to9, title= "Average Cyclability Pombe Nucleosomes 5 - 9")

'''2 - 9'''
pombeN2to9 = average_Nucleosome(pombeNlist[1:8])
pombeM2to9 = average_Nucleosome(pombeMlist[1:8])
plot_cyclability2(pombeN2to9, pombeM2to9, title= "Average Cyclability Pombe Nucleosomes 2 - 9")

#%%
def plot_cyclability_subplot(ax, values1, values2, title=None):
    x_values = np.linspace(-175, 175, len(values1))
    
    if len(values1) != len(values2):
        raise ValueError("Both sequences must have the same length.")
    
    ax.plot(x_values, values1, label='Natural', color='blue')
    ax.plot(x_values, values2, label='Mutated', color='red')
    ax.set_xlabel('Distance from Nucleosome Centre (BP)')
    ax.set_ylabel('Cyclability')
    ax.set_xlim(-200, 200)
    ax.set_ylim(-0.27, -0.1)
    ax.legend()
    
    if title is not None:
        ax.set_title(title)

#%%
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(9):
    row = i // 3
    col = i % 3
    plot_cyclability_subplot(axs[row, col], pombeNlist[i], pombeMlist[i], title=f"+{i+1} Nucleosome")
    
plt.tight_layout()
plt.show()

#%%
'''Storing lists of cyclability'''

import pickle

with open('pombeNlist.pkl', 'wb') as file:
    pickle.dump(pombeNlist, file)

with open("pombeMlist.pkl", "wb") as file:
    pickle.dump(pombeMlist, file)
    
#%%
'''Loading Lists of cyclability'''

import pickle 
with open('pombeNlist.pkl', 'rb') as file:
    pombeNlist = pickle.load(file)
    
with open('pombeMlist.pkl', 'rb') as file:
    pombeMlist = pickle.load(file)

#%%
'''Loading Cerevisiae'''

import pickle
'''Cerevisiae'''
with open('cerNlist.pkl', 'rb') as file:
    cerNlist = pickle.load(file)
    
with open('cerMlist.pkl', 'rb') as file:
    cerMlist = pickle.load(file)
    

#%%
'''Plotting all Nucleosomes together'''
'''Pombe'''
def plot_cyclability_subplot(ax, values1, values2, title=None):
    x_values = np.linspace(-175, 175, len(values1))
    
    if len(values1) != len(values2):
        raise ValueError("Both sequences must have the same length.")
    
    ax.plot(x_values, values1, label='Natural', color='royalblue', linestyle='-', linewidth=2)
    ax.plot(x_values, values2, label='Mutated', color='darkred', linestyle='--', linewidth=2)
    ax.set_xlabel('Distance from Nucleosome Centre (BP)', fontsize=12)
    ax.set_ylabel('Cyclizability', fontsize=12)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-0.27, -0.1)
    
    ax.legend(loc='best', fontsize=10)
    
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)


fig = plt.figure(figsize=(15, 15))
gs = fig.add_gridspec(24, 30)
ax_large = fig.add_subplot(gs[0:12, 0:18])
plot_cyclability_subplot(ax_large, pombeNlist[0], 
                         pombeMlist[0], title="+1 Nucleosome")

for i in range(0, 2):
    row1 = 7 * i
    row2 = 7 * i + 5
    col1 = 19
    col2 = 29
    ax = fig.add_subplot(gs[row1:row2, col1:col2])
    plot_cyclability_subplot(ax, pombeNlist[i + 1], 
                             pombeMlist[i + 1], title=f"+{i + 2} Nucleosome")
    ax.set_ylabel('')
    ax.set_yticklabels([])

for i in range(0, 3):
    row1 = 14
    row2 = 18
    col1 = 10 * i
    col2 = 10 * i + 9
    ax = fig.add_subplot(gs[row1:row2, col1:col2])
    plot_cyclability_subplot(ax, pombeNlist[i + 3], 
                             pombeMlist[i + 3], title=f"+{i + 4} Nucleosome")
    
    if i != 0:
        ax.set_ylabel('')  
        ax.set_yticklabels([])  
 
for i in range(0, 3):
    row1 = 20
    row2 = 24
    col1 = 10 * i
    col2 = 10 * i + 9
    ax = fig.add_subplot(gs[row1:row2, col1:col2])
    plot_cyclability_subplot(ax, pombeNlist[i + 6], pombeMlist[i + 6], title=f"+{i + 7} Nucleosome")
    if i != 0:
        ax.set_ylabel('')
        ax.set_yticklabels([])
plt.tight_layout()
plt.show()

#%%
'''Cerevisiae'''
fig = plt.figure(figsize=(15, 15))
gs = fig.add_gridspec(24, 30)
ax_large = fig.add_subplot(gs[0:12, 0:18])
plot_cyclability_subplot(ax_large, cerNlist[0], 
                         cerMlist[0], title="+1 Nucleosome")

for i in range(0, 2):
    row1 = 7 * i
    row2 = 7 * i + 5
    col1 = 19
    col2 = 29
    ax = fig.add_subplot(gs[row1:row2, col1:col2])
    plot_cyclability_subplot(ax, cerNlist[i + 1], 
                             cerMlist[i + 1], title=f"+{i + 2} Nucleosome")
    ax.set_ylabel('')
    ax.set_yticklabels([])

for i in range(0, 3):
    row1 = 14
    row2 = 18
    col1 = 10 * i
    col2 = 10 * i + 9
    ax = fig.add_subplot(gs[row1:row2, col1:col2])
    plot_cyclability_subplot(ax, cerNlist[i + 3], 
                             cerMlist[i + 3], title=f"+{i + 4} Nucleosome")
    
    if i != 0:
        ax.set_ylabel('')  
        ax.set_yticklabels([])  
 
for i in range(0, 3):
    row1 = 20
    row2 = 24
    col1 = 10 * i
    col2 = 10 * i + 9
    ax = fig.add_subplot(gs[row1:row2, col1:col2])
    plot_cyclability_subplot(ax, cerNlist[i + 6], cerMlist[i + 6], title=f"+{i + 7} Nucleosome")
    if i != 0:
        ax.set_ylabel('')
        ax.set_yticklabels([])
plt.tight_layout()
plt.show()

#%%
'''Storing the raw data for N1 and N5'''
def model_a_nucleosome_Raw(nuc_num, df, genome, mutated_genome):
    N_seq = NucX_sequencer(nuc_num, df, genome)
    N_seq = [item for item in N_seq if item != 'None'] 
    print("Natural Cyclability")
    N_cyc = run_model(N_seq)
    
    M_seq = NucX_sequencer(nuc_num, df, mutated_genome)
    M_seq = [item for item in M_seq if item != 'None'] 
    print("Codon Randomised Cyclability")
    M_cyc = run_model(M_seq)
    return N_cyc, M_cyc

#%%
pombeN1_Raw, pombeM1_Raw = model_a_nucleosome_Raw(1, df_TSS, chromosomes, mutated_chromosomes)

pombeN5_Raw, pombeM5_Raw = model_a_nucleosome_Raw(5, df_TSS, chromosomes, mutated_chromosomes)

#%%
'''Storing lists of cyclability'''

import pickle

with open('pombeN1_Raw.pkl', 'wb') as file:
    pickle.dump(pombeN1_Raw, file)

with open('pombeN5_Raw.pkl', 'wb') as file:
    pickle.dump(pombeN5_Raw, file)

with open('pombeM1_Raw.pkl', 'wb') as file:
    pickle.dump(pombeM1_Raw, file)

with open('pombeM5_Raw.pkl', 'wb') as file:
    pickle.dump(pombeM5_Raw, file)

