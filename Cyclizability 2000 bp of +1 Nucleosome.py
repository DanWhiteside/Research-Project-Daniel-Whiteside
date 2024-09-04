'''1000 base pairs upstream and downstream'''
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
from liftover import ChainFile

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
'''Loading Species'''

'''Pombe'''
df_TSS = pd.read_csv("41594_2010_BFnsmb1741_MOESM9_ESM (COPY).csv")
df_TSS["Left TS"] = df_TSS["Left txn border"]
df_TSS["Right TS"] = df_TSS["Right txn border"]
df_TSS["Strand"] = df_TSS["Orientation"]
df_TSS["Chrom"] = df_TSS["Chromosome"]

df_nucleosome = pd.read_csv("sd01.csv")


chromosomes = {}
fasta_file = "GCA_000002945.2_ASM294v2_genomic.fna"

chromosome_keys = ['1', '2', '3', 'mitochondria']
for idx, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
    if idx < len(chromosome_keys):
        key = chromosome_keys[idx]
    else:
        key = f'chromosome_{idx + 1}'
    chromosomes[key] = str(record.seq).upper()


'''Cerevisiae'''
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

chrome_to_roman = {"chromosome1": "chrI", "chromosome2": "chrII", "chromosome3": "chrIII"
    }
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

def sequencer1000(position, direction, chromosome):
    position = int(position) - 1
    Left_pos = position - 1000
    Right_pos = position + 1000
    Nucleosome_sequence = chromosome[Left_pos:Right_pos]
    if direction == "+":
        Sequence = Nucleosome_sequence
    elif direction == "-":
        Nucleosome_seq_obj = Seq(Nucleosome_sequence)
        compliment_Nuclesome = Nucleosome_seq_obj.reverse_complement()
        Sequence = str(compliment_Nuclesome)
    return Sequence

def Nucleosome_N_Seq1000(Nuc_Number,df, genome, conversion_dict):
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
            sequence = sequencer1000(Nucleosome_position, Direction[i], chromosome)
            if Char_to_check not in sequence:
                Sequences.append(sequence)
            else:
                Sequences.append("None")
        else:
            Sequences.append("None")
        #except:
            #Sequences.append("None")
    return Sequences

def TSS_Seq1000(df, genome, conversion_dict):
    Direction = df["Strand"]
    Chromosome_id = df["Chrom"]
    Left = df["Left TS"]
    Right = df["Right TS"]
    Sequences = []
    Char_to_check = "N"
    for i in range(0,len(Direction)):
        Chrom_id = conversion_dict[Chromosome_id[i]]
        chromosome = genome[Chrom_id]
        try:
            if Direction[i] == "+":
                position = int(Left[i])
            else:
                position = int(Right[i])
            
            sequence = sequencer1000(position, Direction[i], chromosome)
            if Char_to_check not in sequence:
                Sequences.append(sequence)
            else:
                Sequences.append("None")
        except:
            Sequences.append("None")
    return Sequences

def average_Nucleosome(Nucleosomes):
    #Was having errors so added functionallity if not all are the same size
    max_length = max(len(n) for n in Nucleosomes)
    filtered_nucleosomes = [n for n in Nucleosomes if len(n) == max_length]
    filtered_nucleosomes = np.array(filtered_nucleosomes)
    average_cyclability = np.average(filtered_nucleosomes, axis=0)
    return average_cyclability


#%%
NucleosomeByGene = Nfinder(df_cerN, df_cerTSS, chr_to_roman)
df_cerTSS["Nucleosome Positions"] = NucleosomeByGene

NucleosomeByGene = Nfinder(df_nucleosome, df_TSS, chrome_to_roman)
df_TSS["Nucleosome Positions"] = NucleosomeByGene
#%%
N_NucleosomeSequence = Nucleosome_N_Seq1000(1, df_cerTSS, saccer2_genome, chr_to_roman)

Pombe_NucleosomeSequence = Nucleosome_N_Seq1000(1, df_TSS, chromosomes, chrome_to_num)
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
    
def sliding_window_average(arr, window_size=50):
    result = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i+window_size]
        result.append(np.mean(window))
    return np.array(result)

def plot_cyclability2000(values1, values2, label1, label2):
    values1 = sliding_window_average(values1)
    values2 = sliding_window_average(values2)
    
    # Generate x-values for both sequences (same for both sequences)
    x_values = np.linspace(-975, 975, len(values1))
    
    
    plt.figure(figsize=(12, 6))  # Create figure with specified size
    
    # Plot both sequences with different colors
    plt.plot(x_values, values1, label=label1, color='blue')
    plt.plot(x_values, values2, label=label2, color='red')
    plt.xlabel('Distance from Nucleosome Centre (BP)', fontsize=24, 
                fontname='Verdana')
    plt.ylabel('Cyclizability', fontsize=24, 
                fontname='Verdana')
    plt.xlim(-1000, 1000)  
    plt.ylim(-0.27, -0.1)
    plt.legend(loc='best', fontsize=13, frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    
    plt.show()



#%%
print("Cerevisiae")
Cer2000_NucleosomeSequences = [item for item in N_NucleosomeSequence if item != 'None'] 
Cer2000_NucleosomeSequences = [item for i, item in enumerate(Cer2000_NucleosomeSequences) if i != 1109]
CerN2000raw = run_model(Cer2000_NucleosomeSequences)
print("Pombe")
Pombe2000_NucleosomeSequences = [item for item in Pombe_NucleosomeSequence if item != 'None'] 
PombeN2000raw = run_model(Pombe2000_NucleosomeSequences)

#%%
CerN2000 = average_Nucleosome(CerN2000raw)
PombeN2000 = average_Nucleosome(PombeN2000raw)

#%%
plot_cyclability2000(CerN2000, PombeN2000, "Cerevisiae", "Pombe")

#%%
'''TSS as centre'''
CerTSSsequences = TSS_Seq1000( df_cerTSS, saccer2_genome, chr_to_roman)
PombeTSSsequences = TSS_Seq1000(df_TSS, chromosomes, chrome_to_num)
cTSS2000sequences = [item for item in CerTSSsequences if item != "None"]
cTSS2000sequences = [item for i, item in enumerate(cTSS2000sequences) if i != 1129]
pTSS2000sequences = [item for item in PombeTSSsequences if item != "None"]

#%%
print("Cerevisiae")
cTSS2000raw = run_model(cTSS2000sequences)
print("Pombe")
pTSS2000raw = run_model(pTSS2000sequences)

#%%
cTSS2000 = average_Nucleosome(cTSS2000raw)
pTSS2000 = average_Nucleosome(pTSS2000raw)

#%%
plot_cyclability2000(cTSS2000, pTSS2000, "Cerevisiae", "Pombe")

