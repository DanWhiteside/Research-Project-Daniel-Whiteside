''' Ploting Stuff '''
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
import matplotlib.colors as mcolors
from scipy import stats
import statsmodels.api as sm 
import pylab as py 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
'''Previous Plotting functions'''
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
    plt.ylim(-0.27, -0.1)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()
    
def average_Nucleosome(Nucleosomes):
    average_cyclability = np.array(Nucleosomes)
    average_cyclability = np.average(Nucleosomes, axis=0)
    return average_cyclability


#%%
'''Loading Lists of cyclability'''
import pickle
'''Cerevisiae'''
with open('cerNlist.pkl', 'rb') as file:
    cerNlist = pickle.load(file)
    
with open('cerMlist.pkl', 'rb') as file:
    cerMlist = pickle.load(file)

'''Pombe'''
with open('pombeNlist.pkl', 'rb') as file:
    pombeNlist = pickle.load(file)
    
with open('pombeMlist.pkl', 'rb') as file:
    pombeMlist = pickle.load(file)


#%%
'''Comparison between Pombe and Cerevisae'''
def compare_cyclability(org1_name, values1_nat, values1_mut, org2_name, values2_nat, values2_mut, title=None):
    if len(values1_nat) != len(values1_mut):
        raise ValueError(f"Natural and mutated data for {org1_name} must have the same length.")
    if len(values2_nat) != len(values2_mut):
        raise ValueError(f"Natural and mutated data for {org2_name} must have the same length.")
    x_values1 = np.linspace(-175, 175, len(values1_nat))
    x_values2 = np.linspace(-175, 175, len(values2_nat))
    plt.figure(figsize=(12, 6))
    deep_blue = '#00008B'  
    vibrant_orange = '#FF4500'  
    mutated_blue = '#4169E1'  
    mutated_orange = '#FF6347'  
    plt.plot(x_values1, values1_nat, label=f'{org1_name} Natural', color=deep_blue, linestyle='-', linewidth=2)
    plt.plot(x_values1, values1_mut, label=f'{org1_name} Mutated', color=mutated_blue, linestyle='--', linewidth=2)
    plt.plot(x_values2, values2_nat, label=f'{org2_name} Natural', color=vibrant_orange, linestyle='-', linewidth=2)
    plt.plot(x_values2, values2_mut, label=f'{org2_name} Mutated', color=mutated_orange, linestyle='--', linewidth=2)
    plt.xlabel('Distance from Nucleosome Centre (BP)', fontsize=24, fontname='Verdana')
    plt.ylabel('Cyclizability', fontsize=24, 
                fontname='Verdana')
    plt.xlim(-200, 200)
    plt.ylim(-0.25, -0.1)
    plt.legend(loc='best', fontsize=13, frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.show()
    

compare_cyclability("Pombe",pombeNlist[0], pombeMlist[0], 
                    "Cerevisiae",cerNlist[0], cerMlist[0], 
                    title= "Nucleosome 1")

#%%
''' Averaging over downstream nucleosomes'''
pombeN2to4 = average_Nucleosome(pombeNlist[1:3])
pombeM2to4 = average_Nucleosome(pombeMlist[1:3])
cerN2to4 = average_Nucleosome(cerNlist[1:3])
cerM2to4 = average_Nucleosome(cerMlist[1:3])
compare_cyclability("Pombe",pombeN2to4, pombeM2to4, 
                    "Cerevisiae",cerN2to4, cerM2to4, 
                    title= "Nucleosomes 2 - 4")

pombeN5to9 = average_Nucleosome(pombeNlist[4:8])
pombeM5to9 = average_Nucleosome(pombeMlist[4:8])
cerN5to9 = average_Nucleosome(cerNlist[4:8])
cerM5to9 = average_Nucleosome(cerMlist[4:8])
compare_cyclability("Pombe",pombeN5to9, pombeM5to9, 
                    "Cerevisiae",cerN5to9, cerM5to9, 
                    title= "Nucleosomes 5 - 9")

pombeN2to9 = average_Nucleosome(pombeNlist[1:8])
pombeM2to9 = average_Nucleosome(pombeMlist[1:8])
cerN2to9 = average_Nucleosome(cerNlist[1:8])
cerM2to9 = average_Nucleosome(cerMlist[1:8])
compare_cyclability("Pombe",pombeN2to9, pombeM2to9, 
                    "Cerevisiae",cerN2to9, cerM2to9, 
                    title= "Nucleosomes 2 - 9")


#%%
''' Sum of Squared differences '''
def sum_of_squared_differences(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    sum_squared_diff = sum((a - b) ** 2 for a, b in zip(list1, list2))
    return sum_squared_diff

#Difference in pombe N1 mutated natural
SSD_pombeNM1 = sum_of_squared_differences(pombeNlist[0], pombeMlist[0])
print(SSD_pombeNM1)
'''0.08'''

#Difference in pombe N5 mutated natural
SSD_pombeNM5to9 = sum_of_squared_differences(pombeN5to9, pombeM5to9)
print(SSD_pombeNM5to9)
'''0.465'''

#Difference in cerevisiae N1
SSD_cerNM1 = sum_of_squared_differences(cerNlist[0], cerMlist[0])
print(SSD_cerNM1)
'''0.023889'''

#Difference in cerevisiae N5
SSD_cerNM5to9 = sum_of_squared_differences(cerN5to9, cerM5to9)
print(SSD_cerNM5to9)
'''0.105'''

#Between Pombe N1 and Cerevisiae N1
pombeN1 = np.delete(pombeNlist[0], -1)
SSD_compN1 = sum_of_squared_differences(pombeN1, cerNlist[0])
print(SSD_compN1)
'''0.4787955'''

#Between Pombe N1 and Cerevisiae N1 codon randomised
pombeM1 = np.delete(pombeMlist[0], -1)
SSD_compM1 = sum_of_squared_differences(pombeM1, cerMlist[0])
print(SSD_compM1)
'''0.409 so the mutated plus 1 just as different probably because of how similar each is to natural'''

#Between Pombe N5-9 and Cerevisiae N5-9
pombeN59 = np.delete(pombeN5to9, -1)
SSD_compN5 = sum_of_squared_differences(pombeN59, cerN5to9)
print(SSD_compN5)
'''0.2647'''

#Between Pombe N5-9 and Cerevisiae N5-9 codon randomised
pombeM59 = np.delete(pombeM5to9, -1)
SSD_compM5 = sum_of_squared_differences(pombeM59, cerM5to9)
print(SSD_compM5)
'''0.0201 Mutated on downstream really quite similar'''

#%%
#Quantifying difference between nucleosomes using the last nucleosome as reference
'''Difference to the +5'''
pSSDs = []
cSSDs = []
for i in range(0,len(pombeNlist)):
    x = sum_of_squared_differences(pombeNlist[i], pombeNlist[4])
    pSSDs.append(x)

for i in range(0,len(cerNlist)):
    x = sum_of_squared_differences(cerNlist[i], cerNlist[4])
    cSSDs.append(x)
    
conditions = list(range(1, len(pSSDs) + 1))
line_color1 = '#00008B'  
marker_color1 = '#4169E1'  
marker_style1 = 'o'  
line_color2 = '#FF4500'
marker_color2 = '#4169E1'  
marker_style2 = 'D'  
plt.figure(figsize=(12, 6))
plt.plot(conditions, pSSDs, color=line_color1, marker=marker_style1, 
         markersize=8, linestyle='-', linewidth=2, label = "Pombe")
plt.plot(conditions, cSSDs, color=line_color2, marker=marker_style2, 
         markersize=8, linestyle='-', linewidth=2, label = "Cerevisiae")
plt.xlabel('Nucleosome (+)', fontsize=26,fontname='Verdana')
plt.ylabel('SSD to +5 Nucleosome', fontsize=26,fontname='Verdana')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='#bdc3c7')
plt.legend(loc='best', fontsize=20, frameon=True, shadow=True, borderpad=1)
plt.tight_layout()
plt.show()


#%%
def distribution_at_position(lists, position):
    #Checking the position is in range
    if position < 0:
        raise ValueError("Position must be within the length of the list")
    values = [lst[position] for lst in lists if position < len(lst)]
    print(len(values))
    #Histogram
    plt.hist(values)
    plt.show()
    
    array = np.array(values)
    sm.qqplot(array, line="45") 
    py.show()
    
    #Shapiro Wilkes test for normal distribution
    print(stats.shapiro(values))
    
    
#%%
#Load raw data for t tests
import pickle

with open('pombeN1_Raw.pkl', 'rb') as file:
   pombeN1_Raw = pickle.load(file)

with open('pombeN5_Raw.pkl', 'rb') as file:
    pombeN5_Raw = pickle.load(file)

with open('pombeM1_Raw.pkl', 'rb') as file:
    pombeM1_Raw = pickle.load(file)

with open('pombeM5_Raw.pkl', 'rb') as file:
    pombeM5_Raw = pickle.load(file)  
    
#%%
distribution_at_position(pombeN5_Raw, 176)


#%%
'''T test differences between Pombe and Cerevisiae '''
def U_test_at_position(lists_condition1, lists_condition2, position):
    #Getting values at the specified position
    values1 = [lst[position] for lst in lists_condition1 if position < len(lst)]
    values2 = [lst[position] for lst in lists_condition2 if position < len(lst)]
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    #Performing the t-test
    t_statistic, p_value = stats.mannwhitneyu(values1, values2)
    
    return t_statistic, p_value


    
#%%
'''1: Pombe Plus 1 nucleosome difference from codon randomisation by position'''
#Centre
t1_centre, p1_centre = U_test_at_position(pombeN1_Raw, pombeM1_Raw, 176)
print(t1_centre, p1_centre)
'''P = 0.03 so significantly different at the centre'''

#far left (-150)
t1_Fleft, p1_Fleft = t_test_at_position(pombeN1_Raw, pombeM1_Raw, 26)
print(t1_Fleft, p1_Fleft)
'''P = ~ 0.9 so near identical at the far left'''

#far right (+150)
t1_Fright, p1_Fright = t_test_at_position(pombeN1_Raw, pombeM1_Raw, 326)
print(t1_Fright, p1_Fright)
'''P = ~ 0.3 so not significantly different but still a little'''

#near left (-75)
t1_Nleft, p1_Nleft = t_test_at_position(pombeN1_Raw, pombeM1_Raw, 101)
print(t1_Nleft, p1_Nleft)
'''P = ~ 0.6 so not significantly different at near left'''

#near right (+75)
t1_Nright, p1_Nright = t_test_at_position(pombeN1_Raw, pombeM1_Raw, 251)
print(p1_Nright)
'''P = ~ 0.5 e-9 so very much significantly different at near right'''


'''2: Pombe Plus 5 nucleosome difference from codon randomisation by position'''
# Centre
t2_centre, p2_centre = U_test_at_position(pombeN5_Raw, pombeM5_Raw, 176)
print(t2_centre, p2_centre)
'''P = 0.002 so significantly different at the centre'''

# Far left (-150)
t2_Fleft, p2_Fleft = U_test_at_position(pombeN5_Raw, pombeM5_Raw, 26)
print(t2_Fleft, p2_Fleft)
'''P = ~ 0.03 so significantly different at the far left'''

# Far right (+150)
t2_Fright, p2_Fright = U_test_at_position(pombeN5_Raw, pombeM5_Raw, 326)
print(t2_Fright, p2_Fright)
'''P = ~ 6 e-7 so significantly different at the far right'''

# Near left (-75)
t2_Nleft, p2_Nleft = t_test_at_position(pombeN5_Raw, pombeM5_Raw, 101)
print(t2_Nleft, p2_Nleft)
'''P = ~ 1 e-20 so very much significantly different at near left'''

# Near right (+75)
t2_Nright, p2_Nright = t_test_at_position(pombeN5_Raw, pombeM5_Raw, 251)
print(p2_Nright)
'''P = ~ 5 e-17 so very much significantly different at the far right '''


'''3: Difference between Nucleosome +1 and +5 in pombe'''
# Centre
t3_centre, p3_centre = t_test_at_position(pombeN1_Raw, pombeN5_Raw, 176)
print(t3_centre, p3_centre)
'''P = 4.9586 e-7 so significantly different at the centre'''

# Far left (-150)
t3_Fleft, p3_Fleft = t_test_at_position(pombeN1_Raw, pombeN5_Raw, 26)
print(t3_Fleft, p3_Fleft)
'''P = ~ 0.2 so not significantly different'''

# Far right (+150)
t3_Fright, p3_Fright = t_test_at_position(pombeN1_Raw, pombeN5_Raw, 326)
print(t3_Fright, p3_Fright)
'''P = ~ 0.015 so significantly different'''

# Near left (-75)
t3_Nleft, p3_Nleft = t_test_at_position(pombeN1_Raw, pombeN5_Raw, 101)
print(t3_Nleft, p3_Nleft)
'''P = ~ 0.0005 so significantly different at near left'''

# Near right (+75)
t3_Nright, p3_Nright = t_test_at_position(pombeN1_Raw, pombeN5_Raw, 251)
print(p3_Nright)
'''P = ~ 0.8 so not significantly different'''




'''
To be done once Cerevisiae is ready; will need to get GPT to change t3 and p3
'''

# '''4: Cerevisiae PLus 1 Nucleosome difference from codon randomisation by position'''
# # Centre
# t3_centre, p3_centre = t_test_at_position(cerN1_Raw, cerM1_Raw, 176)
# print(t3_centre, p3_centre)
# '''P = 0.03 so significantly different at the centre'''

# # Far left (-150)
# t3_Fleft, p3_Fleft = t_test_at_position(cerN1_Raw, cerM1_Raw, 26)
# print(t3_Fleft, p3_Fleft)
# '''P = ~ 0.9 so near identical at the far left'''

# # Far right (+150)
# t3_Fright, p3_Fright = t_test_at_position(cerN1_Raw, cerM1_Raw, 326)
# print(t3_Fright, p3_Fright)
# '''P = ~ 0.3 so not significantly different but still a little'''

# # Near left (-75)
# t3_Nleft, p3_Nleft = t_test_at_position(cerN1_Raw, cerM1_Raw, 101)
# print(t3_Nleft, p3_Nleft)
# '''P = ~ 0.6 so not significantly different at near left'''

# # Near right (+75)
# t3_Nright, p3_Nright = t_test_at_position(cerN1_Raw, cerM1_Raw, 251)
# print(p3_Nright)
# '''P = ~ 0.5 e-9 so very much significantly different but still a little'''

#%%
'''Ploting data with errorbar region'''
def plot_with_shaded_sem(list1, list2, label1, label2):
    array1 = np.array(list1)
    array2 = np.array(list2)
    mean1 = np.mean(array1, axis=0)
    sem1 = np.std(array1, axis=0) / np.sqrt(array1.shape[0])
    mean2 = np.mean(array2, axis=0)
    sem2 = np.std(array2, axis=0) / np.sqrt(array2.shape[0])
    x = np.arange(len(mean1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean1, label=label1, color='blue')
    plt.fill_between(x, mean1 - sem1, mean1 + sem1, color='blue', alpha=0.2)
    plt.plot(x, mean2, label=label2, color='red')
    plt.fill_between(x, mean2 - sem2, mean2 + sem2, color='red', alpha=0.2)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Mean with SEM Shaded Region')
    plt.legend()
    plt.show()

plot_with_shaded_sem(pombeN1_Raw, pombeM5_Raw, "Plus 1 Nucleosome", "Plus 5 Nucleosome")