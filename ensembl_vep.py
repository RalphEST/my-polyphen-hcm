# Using the Ensembl VEP API to Featurize Variants

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import requests
import json
import re

class vep_features: 
    def __init__(self, df, gene_name_col, hgvs_cols, id_cols, conseqs):
        '''
        INPUT: 
        df         | DATAFRAME with the variants to featurize
        hgvs_cols  | LIST of column names with HGVS notations per variant
        id_cols    | LIST of column names with ID notation per variant 
        '''
        self.df = df.copy()
        self.hgvs_cols = hgvs_cols
        self.id_cols = id_cols
        self.gene_name_col = gene_name_col
        self.conseqs = conseqs

        self.n_variants = df.shape[0]
        self.variant_genes = set([z for y in [x.split('|') for x in set(df[gene_name_col])] for z in y])
        
        self.notation_col_dict = {x:'hgvs' for x in hgvs_cols}
        self.notation_col_dict.update({x:'id' for x in id_cols})

        self.df['blosum62'] = ['None'] * self.n_variants
        self.df['cadd_raw'] = ['None'] * self.n_variants
        self.df['cadd_phred'] = ['None'] * self.n_variants
        self.df['polyphen2_score'] = ['None'] * self.n_variants
        self.df['polyphen2_pred'] = ['None'] * self.n_variants
        self.df['sift_score'] = ['None'] * self.n_variants
        self.df['sift_pred'] = ['None'] * self.n_variants
        
    def vep_request(self, notation_list, notation_type):
        '''
        INPUT:
        hgvs_list  | LIST of HGVS notations to featurize
        '''
        server = "https://rest.ensembl.org"
        ext = "/vep/human/" + notation_type
        headers = { "Content-Type" : "application/json", "Accept" : "application/json"}
        params = {"Blosum62" : "1", "CADD" : "1"}

        string_input = ("\", \"").join(notation_list)
        data_type = 'hgvs_notations' if notation_type == 'hgvs' else 'ids'
        print('sending request ...', end=' ')
        r = requests.post(server+ext, headers=headers, data='{ \"'+ data_type +'\" : [\"'+ string_input + '\"] }', params = params)
        decoded = r.json()
        print('Done!')

        inputs = []
        pphen2_score = []
        pphen2_pred = []
        blosum62 = []
        cadd_raw = []
        cadd_phred = []
        sift_pred = []
        sift_score = []

        for var in decoded:
            inputs.append(var['input'])
            transcripts = var['transcript_consequences']

            input_var = var['input']

            tc_pphen2_score = []
            tc_pphen2_pred = []
            tc_blosum62 = []
            tc_cadd_raw = []
            tc_cadd_phred = []
            tc_sift_pred = []
            tc_sift_score = []

            for tc in transcripts:

                is_right_gene = (tc['gene_symbol'] in self.variant_genes)
                is_protein_coding = (tc['biotype'] == 'protein_coding')
                is_right_consequence = any([conseq in tc["consequence_terms"] for conseq in self.conseqs])

                if is_right_gene and is_protein_coding and is_right_consequence:
                    tc_pphen2_score.append(tc.get('polyphen_score'))
                    tc_pphen2_pred.append(tc.get('polyphen_prediction'))
                    tc_blosum62.append(tc.get('blosum62'))
                    tc_cadd_raw.append(tc.get('cadd_raw'))
                    tc_cadd_phred.append(tc.get('cadd_phred'))
                    tc_sift_score.append(tc.get('sift_score'))
                    tc_sift_pred.append(tc.get('sift_prediction'))
            
            if None in tc_pphen2_score: tc_pphen2_score.remove(None)
            if None in tc_pphen2_pred: tc_pphen2_pred.remove(None)
            if None in tc_blosum62: tc_blosum62.remove(None)
            if None in tc_cadd_raw: tc_cadd_raw.remove(None)
            if None in tc_cadd_phred: tc_cadd_phred.remove(None)
            if None in tc_sift_pred: tc_sift_pred.remove(None)
            if None in tc_sift_score: tc_sift_score.remove(None)

            pphen2_score.append('not_found' if len(tc_pphen2_score)==0 else np.mean(tc_pphen2_score))
            pphen2_pred.append('not_found' if len(tc_pphen2_pred)==0 else tc_pphen2_pred[0])
            blosum62.append('not_found' if len(tc_blosum62)==0 else np.mean(tc_blosum62))
            cadd_raw.append('not_found' if len(tc_cadd_raw)==0 else np.mean(tc_cadd_raw))
            cadd_phred.append('not_found' if len(tc_cadd_phred)==0 else np.mean(tc_cadd_phred))
            sift_pred.append('not_found' if len(tc_sift_pred)==0 else tc_sift_pred[0])
            sift_score.append('not_found' if len(tc_sift_score)==0 else tc_sift_score[0])
        
        return (inputs, pphen2_score, pphen2_pred, blosum62, cadd_raw, cadd_phred, sift_score, sift_pred)

    def featurize_id_col(self, col_name):
        '''
        INPUT:
        col_name   | STRING name of the column based on which we will featurize

        Based on a column, divide the variants in batches of 200
        (the API requests cannot request more than 200 items), and featurize (while
        updating self.df)
        '''

        unfeat_vars = list(self.featurized[(self.featurized['featurized'] == 0) & ~(self.featurized[col_name].isna())][col_name])
        print(f'Featurizing {len(unfeat_vars)} variants!')
        N = len(unfeat_vars)
        batch_idx = list(range(0, N, 200)) + [N]
        unfeat_vars_batches = [unfeat_vars[batch_idx[i]: batch_idx[i+1]] for i in range(N//200 + 1)]

        for batch in unfeat_vars_batches:
            inputs, pphen2_score, pphen2_pred, blosum62, cadd_raw, cadd_phred, sift_score, sift_pred = self.vep_request(batch, self.notation_col_dict[col_name])
            self.current = (inputs, pphen2_score, pphen2_pred, blosum62, cadd_raw, cadd_phred, sift_score, sift_pred)
            self.df.loc[(self.df[col_name].isin(inputs)) & (self.featurized['featurized']==0), 'blosum62'] = blosum62
            self.df.loc[(self.df[col_name].isin(inputs)) & (self.featurized['featurized']==0), 'cadd_raw'] = cadd_raw
            self.df.loc[(self.df[col_name].isin(inputs)) & (self.featurized['featurized']==0), 'cadd_phred'] = cadd_phred
            self.df.loc[(self.df[col_name].isin(inputs)) & (self.featurized['featurized']==0), 'polyphen2_score'] = pphen2_score
            self.df.loc[(self.df[col_name].isin(inputs)) & (self.featurized['featurized']==0), 'polyphen2_pred'] = pphen2_pred
            self.df.loc[(self.df[col_name].isin(inputs)) & (self.featurized['featurized']==0), 'sift_score'] = sift_score
            self.df.loc[(self.df[col_name].isin(inputs)) & (self.featurized['featurized']==0), 'sift_pred'] = sift_pred
            self.featurized.loc[self.featurized[col_name].isin(inputs), 'featurized'] = 1

    def featurize_variants(self):

        '''
        Main function: used to start the process of adding ENSEMBL VEP features to the variant dataframe
        '''
        
        # Create a dataframe to keep track of which variants have been featurized and which have not.
        self.featurized = self.df[[self.gene_name_col] + self.hgvs_cols + self.id_cols].copy()
        self.featurized['featurized'] = 0

        # Iterate over the notation columns, and progressively featurize what has not been featurized yet.
        for col in self.notation_col_dict.keys():
            self.featurize_id_col(col)
        
        return self.df
