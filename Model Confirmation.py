'''Model Confirmation and Validation'''
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
from scipy import stats
import statsmodels.api as sm 
import pylab as py 

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%

def pred(model, pool):
    input = np.zeros((len(pool), 200), dtype = np.single)
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50): 
            input[i][j*4 + temp[pool[i][j]]] = 1
    A = model.predict(input, batch_size=128)
    A.resize((len(pool),))
    return A

def load_model(modelnum: int):
    return keras.models.load_model(f"./adapter-free-Model/C{modelnum}free")


def run_model50bp(seqs):
    if not isinstance(seqs, list):
        seqs = [seqs]
    accumulated_cyclability = []
    option = "C0free prediction"
    modelnum = int(re.findall(r'\d+', option)[0])
    model = load_model(modelnum)
    
    for seq in seqs:
        x = 1
        if x % 200 == 0:
            print(x)
        x += 1
        
        subseq_length = 50
        step_size = subseq_length
        list50 = [seq[i:i+subseq_length] for i in range(0, len(seq) - subseq_length + 1, step_size)]
        cNfree = pred(model, list50)
        prediction = list(cNfree)
        accumulated_cyclability.append(prediction)
    
    return accumulated_cyclability



#%%
#Validation of random sequences 
df_RandomSeq = pd.read_csv("41586_2020_3052_MOESM6_ESM(copy).csv")

#%%
#Building function to collect the central 50bp from the Random 100bp 
def Central50(dna_sequence):
    #Checking if the sequence length is 100 bp
    if len(dna_sequence) != 100:
        raise ValueError("The DNA sequence must be exactly 100 bp long.")
    start_index = 25
    end_index = start_index + 50
    central_50bp = dna_sequence[start_index:end_index]
    return central_50bp

#%%
df_RandomSeq['Central_50bp'] = df_RandomSeq['Sequence'].apply(Central50) 

#%%
Central_50list = list(df_RandomSeq['Central_50bp'])
Central_50string = "".join(Central_50list)
test_cyc = run_model50bp(Central_50string)
test_cyc = np.concatenate(test_cyc)

#%%
df_outliers = df_RandomSeq.loc[df_RandomSeq[" C0"] > 1.9, ["Sequence"]]
df_outliers = df_outliers["Sequence"].apply(Central50)

#%%

#Pearson correlation
C0 = np.array(df_RandomSeq[" C0"])
if len(df_RandomSeq[" C0"]) == len(test_cyc):
    correlation_coefficient, p_value_corr = pearsonr(df_RandomSeq[" C0"], test_cyc)
    print("Correlation coefficient is:", correlation_coefficient)
    print(f"P value is: {p_value_corr:.30f}")  # Print p-value with higher precision
else:
    print("Not the same length")

#Line of best fit
slope, intercept, r_value, p_value_reg, std_err = linregress(test_cyc, C0)    
line = slope * test_cyc + intercept

#R-squared
r_squared = r_value**2
print("R-squared is:", r_squared)

#Residuals
residuals = C0 - line

#Regular outliers (Residuals > 4.5 standard deviations)
threshold1 = 4.5 * np.std(residuals)
outliers = np.abs(residuals) > threshold1
outliers_test = test_cyc[outliers]
outliers_exp = C0[outliers]

#Extreme outliers (C0 > 1.9)
extreme_outliers = C0 > 1.9
extreme_outliers_test = test_cyc[extreme_outliers]
extreme_outliers_exp = C0[extreme_outliers]

#Plot
plt.figure(figsize=(10, 6))
plt.scatter(test_cyc, C0, alpha=0.75, label="Data Points", 
            edgecolor='black', s=70, marker='o', color='#3498db')
plt.plot(test_cyc, intercept + slope * test_cyc, color='#e74c3c', 
         linewidth=2.5, linestyle='-.', label='Regression Line')
plt.scatter(outliers_test, outliers_exp, color='#e67e22', 
            edgecolor='black', s=85, label='Outliers', marker='D')
plt.scatter(extreme_outliers_test, extreme_outliers_exp, 
            color='#9b59b6', edgecolor='black', s=100, 
            label='Extreme Outliers', marker='P')
plt.xlabel('Predicted Cyclizability', fontsize=22, 
            fontname='Verdana')
plt.ylabel('Experimental Cyclizability', fontsize=22, 
           fontname='Verdana')
plt.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')
plt.legend(fontsize=16, loc='best', frameon=True, shadow=True, borderpad=1)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.show()

#%%
df_ex_outliers = df_RandomSeq.loc[df_RandomSeq[" C0"] > 1.9, ["Sequence"]]
df_ex_outliers = df_outliers["Sequence"].apply(Central50)
'''
Sequences greater than 2 C0
571	    CCCGATGGTCCACATGCTCCTTAGAAGAGCTAGCCGTCGATAGACCATCC
8160	GTTCTGGGTTAATACTGATCGGAAGAGCAAGTGGGCTCAGTCAACAGACT

Sequences greater than 1.9 C0
571	    CCCGATGGTCCACATGCTCCTTAGAAGAGCTAGCCGTCGATAGACCATCC
2085	AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC
3084	CCTGGTCTGGTACAGTGAGGCTCTTCGTAGAGTCACAGAGGAGGGTGACC
4434	GGAAAGCGTTAGGAACTCGATTGACTGTCGTCACAGGAAGAGCACCGGAT
8160	GTTCTGGGTTAATACTGATCGGAAGAGCAAGTGGGCTCAGTCAACAGACT


'''
#%%
#Threshold for regular outliers (e.g., residuals > 4.5 standard deviations)
threshold1 = 4.5 * np.std(residuals)
outliers = np.abs(residuals) > threshold1
outliers_indices = np.where(outliers)[0]
df_outliers_regular = df_RandomSeq.iloc[outliers_indices]["Sequence"]
df_outliers_regular = df_outliers_regular.apply(Central50)
outlier_sequences = df_outliers_regular.tolist()

for seq in outlier_sequences:
    print(seq)
    
'''
All outlier sequences 

ACTAGGCTCTTGGACAAATACAGTTACATCTCATGTAGAAGAGCACCCAG
TATCTGTAATTGATTTAGTAAGCTCTTCGGTAAGCATCTCGTCGTCAGCA
CAATAGAACTCAAGGAGCAGTGCTCTTCGTAGGACTTACGAGGAATATGC
CCCGATGGTCCACATGCTCCTTAGAAGAGCTAGCCGTCGATAGACCATCC
AGTCCCTCGTTGGGGTGTGAGCTCTTCATTCGTTAGTTCTCAAAGGACAG
TTGACGTAGCATTTAGAGTCCCGGAAAGCGAGGCCAGAAGAGCCGGCCGA
GTAGACACGCTGACTAACAATCAATCCTGGAAGAGCGCTAACATCCTTCG
AAACGAACTCTCACTCCTGAGTGAGCACTATATTCTGAAGAGCGCAACTG
GGAGGTTTAGGCCGTTCCAGGGCGGTATAGAAGAGCGAGCCAAAAAAGGA
AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC
GGGGACGCTCTTCATCTCGAACAGTTTAGAAGGGATATATCCGTACGTAC
CCTGGTCTGGTACAGTGAGGCTCTTCGTAGAGTCACAGAGGAGGGTGACC
GGAAAGCGTTAGGAACTCGATTGACTGTCGTCACAGGAAGAGCACCGGAT
GCAACTCCTGAAAATTACACAGAGACTGCGGGGCAGGAAGAGCTTCTATT
CGCCGCCCCGCAGAAGAGCGGTGCCGGTATTATTTACAAACCACGGAGCT
CGCTTAAGTGACCTAGCACATTGTATTGACAGGAAGAGCCTCCTTCCACC
GGGGTATTATATCTCGTTTGCGCTCTTCTGGATCTTTCCCCCTTTTTCTG
GATGTACAAGAAACTATTTCCTAACGAAGAGCCACTTCGCGTTCCCCTGG
GTTCTGGGTTAATACTGATCGGAAGAGCAAGTGGGCTCAGTCAACAGACT
TTTCGTCATAGCTACATGTGGTACCAGAATAGTACGAAGAGCCAACGAAA
AATATCTTGAGTCTACGATGGCGGGATGAAGAGCAGCACGTACTATCGAG
GGTGGCGAACCATACACGTTCTACTGAAGAGCGTGTGTTCTATGACAAGT
CTTGTGAGAGGGATTGCTCTTCTGCACTCATGGCAGTTATACATAGCTGA
TTGATGTCTGGAAACCTGGCCTTGCTCTTCCAAGGTAGATCGAGAATAGA
TCTGCTCTTCCGCCTACCGACTACTCCAGCATGGAGTACACCATAAGGGC
ACAGACATCCACTCATAGTTGGTCCCTACTGGAAGAGCCCCATGTGACAC
GTGGACGTGGTAGCTGCTCCACGATTACATGCACGAAGAGCCCCTGTGTC
GGGTTGTCGCCATGGAAGAGCGTCGTTTCGATCTCTCATGAAAAGAAGTA
CCCTTTCATAGCGGTCAAAACCGGCTCCGCACGAAGAGCCCATCACGAGG
GGTCGGGCTCTCCACTGCTGCCCTGTGAAGAGCCGGCAAGAAGGGAAGCC
TGTATCCCATGCCCTCTCAGAAGAGCGAATGCTTAACTACATAGAGTTAC
AATGATTAGACGTCTCGCAATGACAGCTCTTCCGCACCTTAAAAGTCCTC
GAGAGCCGTAGTATTACCGAAGAGCATCTCCCCGGTTATAATTGTTCGTT
'''

#%%
'''
Finding GC content of extreme outliers 
'''
from Bio.SeqUtils import GC

sequences = ["CCCGATGGTCCACATGCTCCTTAGAAGAGCTAGCCGTCGATAGACCATCC", "AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC", 
             "CCTGGTCTGGTACAGTGAGGCTCTTCGTAGAGTCACAGAGGAGGGTGACC", "GGAAAGCGTTAGGAACTCGATTGACTGTCGTCACAGGAAGAGCACCGGAT", 
             "GTTCTGGGTTAATACTGATCGGAAGAGCAAGTGGGCTCAGTCAACAGACT"]
gc_contents1 = [GC(seq) for seq in sequences]
print(gc_contents1)
print(np.mean(gc_contents1))

#%%
'''Finding GC content of all outliers'''
gc_contents2 = [GC(seq) for seq in outlier_sequences]
print(gc_contents2)
print(np.mean(gc_contents2))


#%%
'''Analysing significance of the motif sequence GAAGAGC'''

'''Generating random but set 100,000 43 bp sequences'''

def generate_dna_sequence(length):
    nucleotides = ['A', 'C', 'G', 'T']
    sequence = ''.join(random.choice(nucleotides) for _ in range(length))
    return sequence

random_seqs_43 = []
for i in range(100000):
    random.seed(i)
    seq = generate_dna_sequence(43)
    random_seqs_43.append(seq)

def insert_codon(sequence, codon):
    halfway_index = len(sequence) // 2
    new_sequence = sequence[:halfway_index] + codon + sequence[halfway_index:]
    return new_sequence

def codon_cyclability(codon, random_seq):
    amino_sequences = []
    for i in range(0,len(random_seq)):
        amino_seq = insert_codon(random_seq[i], codon)
        amino_sequences.append(amino_seq)
    amino_sequence = "".join(amino_sequences)
    amino_cyclabilities = run_model50bp(amino_sequence)
    flatten_list = [item for sublist in amino_cyclabilities for item in sublist]
    average_cyclability = sum(flatten_list)/len(flatten_list)
    return average_cyclability, flatten_list

def generate_random_dna_sequence(length):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(bases) for _ in range(length))

def random_for_comparison():
    sequence_list = []
    for i in range(100000):
        random_seq = generate_random_dna_sequence(50)
        substituted_sequence = [random_seq]
        sequence = "".join(substituted_sequence)
        if i == 1:
            print(len(sequence))
        sequence_list.append(sequence)
    amino_sequence = "".join(sequence_list)
    amino_cyclabilities = run_model50bp(amino_sequence)
    flatten_list = [item for sublist in amino_cyclabilities for item in sublist]
    return flatten_list


#%%
motif_average, motif_raw = codon_cyclability("GAAGAGC", random_seqs_43)

comp_motif_average, comp_motif_raw = codon_cyclability("GCTCTTC", random_seqs_43)

cyc_random = random_for_comparison()
average_cyc_random = sum(cyc_random)/len(cyc_random)

#%%
'''Visulising distributions of raw data and testing normality'''
plt.hist(motif_raw)
plt.title("GAAGAGC")
plt.show
print(stats.shapiro(motif_raw))

array_motif_raw = np.array(motif_raw)
sm.qqplot(array_motif_raw, line='45') 
py.show()

#%%
plt.hist(comp_motif_raw)
plt.title("GCTCTTC")
plt.show
print(stats.shapiro(comp_motif_raw))

array_comp_motif_raw = np.array(comp_motif_raw)
sm.qqplot(array_comp_motif_raw, line='45') 
py.show()

#%%
plt.hist(cyc_random)
plt.title("Random")
plt.show
print(stats.shapiro(cyc_random))

array_cyc_random = np.array(cyc_random)
sm.qqplot(array_cyc_random, line='45') 
py.show()

#%%
print(motif_average, comp_motif_average, average_cyc_random)

#%%
def plot_violin_plots(list1, list2, list3, name1='List 1', name2='List 2', name3='List 3'):
    data = [list1, list2, list3]
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, palette="deep")
    plt.xticks([0, 1, 2], [name1, name2, name3])
    plt.ylabel('Cyclizability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def perform_t_test(list1, list2):
    t_statistic, p_value = stats.ttest_ind(list1, list2)
    print("T-test results between List 1 and List 2:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.01:
        print("The difference between the two lists is statistically significant.")
    else:
        print("The difference between the two lists is not statistically significant.")
    
#%%
plot_violin_plots(motif_raw, comp_motif_raw, cyc_random)

perform_t_test(motif_raw, cyc_random)
perform_t_test(comp_motif_raw, cyc_random)

