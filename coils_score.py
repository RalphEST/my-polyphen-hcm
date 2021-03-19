# Using 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import shutil
import requests
import json
from tqdm import tqdm
import subprocess




aa3to1 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Glu":"E","Gln":"Q","Gly":"G","His":"H","Ile":"I",
"Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}
aa1to3 = {aa3to1[k]: k for k in aa3to1}


class COILS():
    '''
    *  Evaluates the COILS score for the non-synonymous single nucleotide variants of a given protein
    *  Their are two possible versions of the score returned:
          -  maxdelta (from Jordan et al. 2011): the feauture is the (signed) magnitude of the largest single-residue change
          -  normdelta: the feature is the norm of the difference vector divided by the sequence length
    '''
    
    def __init__(self, df, gene_name_col, prot_change_col, protein_dict, analysis_name, coilsdir, ncoils_abs_path = "ncoils-osf", overwrite = True):
        '''
        *  Initialize the class
        *  Create necessary files and folders
        INPUTS:
        df              | PANDAS DATAFRAME of variants to be COILS-scored 
        prot_change_col | STRING that is the protein change column in the variants dataframe.
                        | The colum should have strings of the form AA_ref+Pos+AA_var, such as 'E1934K'    
        protein_dict    | DICTIONARY where each key:value pair is gene_name:accession code 
        analysis_name   | STRING. Just a name for the analysis in case you want to run many
        ncoils_abs_path | STRING and ABSOLUTE PATH to the ncoils executable. If the executable is in 
                        | the PATH variable
        ''' 
        self.df = df.copy()
        self.protein_dict = protein_dict
        self.analysis_name = analysis_name
        self.ncoils_abs_path = ncoils_abs_path
        self.gene_name_col = gene_name_col
        self.prot_change_col = prot_change_col

        self.sequence_dict = {}
        self.length_dict = {}
        self.dir_dict = {}
        self.num_variants_dict = {}
        self.df['COILS_maxdelta'] = [np.nan] * df.shape[0]
        self.df['COILS_normdelta'] = [np.nan] * df.shape[0]
        self.df['COILS_sumdelta'] = [np.nan] * df.shape[0]


        self.top_dir = 'COILS_analysis_' + analysis_name

        if os.path.exists(self.top_dir) and overwrite:
            shutil.rmtree(self.top_dir)

        os.mkdir(self.top_dir)
        os.environ["COILSDIR"] = coilsdir
        
        print('importing sequences')
        for gene, acc in self.protein_dict.items():
            url = f"https://www.ebi.ac.uk/proteins/api/proteins?&accession={acc}"
            r = requests.get(url, headers={ "Accept" : "text/x-fasta"})
            sequence = ''.join(r.text.split('\n')[1:])      
            self.sequence_dict[gene] = sequence
            self.length_dict[gene] = len(sequence)

            gene_dir = os.path.join(self.top_dir, f'{gene}({acc})_analysis')
            self.dir_dict[gene] = gene_dir
            os.mkdir(gene_dir) 

        
    def generate_variant_seqs(self, gene, var_list):
        '''

        '''
        var_seqs_list = []
        for var in var_list:
            # Decipher variant string
            aa_ref, pos, aa_var = var[0], int(var[1:-1]), var[-1]

            seq = list(self.sequence_dict[gene])
            
            if not seq[pos-1] == aa_ref:
                print('WARNING: ' + seq[pos-1] + ' (ref) is not ' + aa_ref + ' (variant ref)')

            seq[pos-1] = aa_var
            var_seqs_list.append(''.join(seq))

        return var_seqs_list
        
    def generate_fasta_input(self, gene, var_list):
        '''

        '''
        input_file = os.path.join(self.dir_dict[gene], 'input.fasta')

        # FASTA files require both sequence names ...
        input_info = ['> ' + gene + '_WT\n'] + ['> ' + gene + '_' + var+'\n' for var in var_list]
        # ... and the sequences themselves
        input_seqs  = [self.sequence_dict[gene]+'\n'] + [x+'\n' for x in self.generate_variant_seqs(gene, var_list)]


        fasta_lines = [None] * (2 * len(input_info))
        fasta_lines[::2] = input_info
        fasta_lines[1::2] = input_seqs

        # Write to input file
        with open(input_file, 'w') as inputFile:
            inputFile.writelines(fasta_lines)
            
    def read_txt_output(self, gene):
        '''

        '''
        outfile = os.path.join(self.dir_dict[gene], 'output.txt')

        probs, scores = [],[]

        with open(outfile, 'r') as out:
            for line in out:
                probs.append(float(line.split()[4]))
                scores.append(float(line.split()[3]))
                
        probs = np.array(probs).reshape(self.num_variants_dict[gene] + 1, self.length_dict[gene])
        scores = np.array(scores).reshape(self.num_variants_dict[gene] + 1, self.length_dict[gene])

        return probs, scores

    def run_ncoils(self, gene):
        '''
        
        '''
        # run ncoils 
        bashCommand = self.ncoils_abs_path + " -win 14"
        input_file = os.path.join(self.dir_dict[gene], 'input.fasta')
        output_file = os.path.join(self.dir_dict[gene], 'output.txt')
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            proc = subprocess.Popen(bashCommand.split(), 
                stdin=infile, stdout=outfile, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        print(err.decode().strip(), end=' ... ')


    def score(self):
        '''

        '''
    
        for gene in self.protein_dict.keys():
            print(f'|| {gene} ||', end=' ')
            var_list = list(self.df[self.df[self.gene_name_col] == gene][self.prot_change_col])
            self.num_variants_dict[gene] = len(var_list)
            print('generating input fasta file ...', end=' ')
            self.generate_fasta_input(gene, var_list)
            print('running ncoils ...', end=' ')
            self.run_ncoils(gene)
            probs, scores = self.read_txt_output(gene)

            # compute the scores
            print('Scoring ...', end=' ')
            diff = scores - scores[0, :]
            diff = diff[1:, :]
            maxdelta = diff[np.arange(len(var_list)), np.abs(diff).argmax(axis = 1)]
            sumdelta = diff.sum(axis = 1)
            normdelta = np.linalg.norm(diff, axis = 1)

            # fill in the df
            self.df.loc[self.df[self.gene_name_col] == gene, 'COILS_maxdelta'] = maxdelta 
            self.df.loc[self.df[self.gene_name_col] == gene, 'COILS_normdelta'] = normdelta 
            self.df.loc[self.df[self.gene_name_col] == gene, 'COILS_sumdelta'] = sumdelta 

            print('Done!')

        return self.df



