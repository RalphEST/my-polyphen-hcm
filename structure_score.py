# use python class conventions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os 
import subprocess
import collections
import itertools
from scipy import stats
from scipy.special import logsumexp
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import shutil


class StructScore():
    
    def __init__(self, name, sets, window, chain_dict, folder, lovoalign_path, precomputed=False):
        
        '''
        Initiate the StructScore class
        ------------------------------
        sets         | array of arrays (corresponding to protein states) with the structure names
        file         | file in which the structures are
        chain        | the PDB chain we are interested in aligning
        folder       | the folder in which both the PDB files and B-factor files are
        precomputed  | whether the alignments have already been done (to avoid recomputing)
        name         | appended to StructScore folder to make it unique
        window       | size of sliding alignment window (in residues)
        
        Notes:
        ------
        *  This computation assumes that the B-factors have been already computed and that they are in 
           filees called "{4-letter PDB file}_bfactors.txt"
        '''
        
        # store the original set, PDB files folder, and list the relevant PDB IDs
        self.sets = sets
        self.pdb_folder = folder
        self.pdb_ids = [y for x in sets for y in x]
        self.chain = chain_dict
        self.window = window
        self.lovoalign_path = lovoalign_path
        
        # create folder for all alignment files
        self.struct_folder = os.path.join(folder, 'StructScore_'+name)
        
        # list the alignments that need to be made
        self.alignments = []
        
        # dictionary of alignment bundles for combining p-values
        self.bundles = {}
        
        # iterate through the states and list the alignments
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                a = list(itertools.product(sets[i], sets[j]))
                a = [x+":"+y for x,y in a]
                self.alignments += a
                self.bundles[str(i) + ":" + str(j)] = a
        
        # compute everything from scratch?
        if not precomputed: 
            
            # create 'StructScore_{name}' folder
            if not os.path.exists(self.struct_folder):
                os.mkdir(self.struct_folder)
            else:
                shutil.rmtree(self.struct_folder)
                os.mkdir(self.struct_folder)
        
            # create sliding-window alignment folders
            for f in self.alignments:
                os.mkdir(os.path.join(self.struct_folder, f))
                    
        
        # list the names of B-factor files
        b_files = {x:os.path.join(folder, x+"_bfactors.txt") for x in self.pdb_ids}
        
        # create B-factors table
        b_dict = {k: pd.read_csv(b_files[k],
                                 sep = "\t", 
                                 header = None, 
                                 names = ["res", "chain_"+k, "b_"+k]).set_index("res") for k in self.pdb_ids}
                    
        self.B = pd.DataFrame()
        for pdb in self.pdb_ids:
            table = b_dict[pdb]
            self.B = self.B.join(table[table['chain_'+pdb]==self.chain[pdb]], how ='outer')
        
        self.B = self.B.filter(regex = ("^b"))
        
        # impute missing values with column mean and create variances dataframe
        self.variances = self.B.fillna(self.B.mean(0)) /8/np.pi**2
        
        # rename the columns of 
        self.variances.columns = [x.replace('b_', '') for x in self.variances.columns]
        
        # recover the length of the protein and adjust indices
        self.length = self.B.index.max()
        self.B = self.B.reindex(pd.RangeIndex(self.length + 1))
        self.variances = self.variances.reindex(pd.RangeIndex(self.length + 1))
        
        if precomputed:
            self.readPrecomputed()
        
        
    def windowAlign(self, alignment):
        '''
        *  Helper function, used in align()
        *  Window-aligns the two proteins using LovoAlign
        '''
        # folder to store the alignment files
        folder = os.path.join(self.struct_folder, alignment)
        
        p1, p2 = alignment.split(":")
        p1path = os.path.join(self.pdb_folder, p1 + ".pdb")
        p2path = os.path.join(self.pdb_folder, p2 + ".pdb")
        
        for i in np.arange(1, self.length + self.window - 1):
            pdbout = os.path.join(folder, str(i))
            
            rmin = np.max([i - self.window, 0])
            rmax = np.min([self.length, i + self.window])
            
            bashCommand = f"{self.lovoalign_path} -p1 {p1path} -p2 {p2path} -c1 {self.chain[p1]} -c2 {self.chain[p2]} -rmin1 {rmin} -rmin2 {rmin} -rmax1 {rmax} -rmax2 {rmax} -o {pdbout}.pdb -rmsf {pdbout}.dat"
            process = subprocess.Popen(bashCommand.split(),stdout=subprocess.PIPE)
            out, err = process.communicate()
            
    def readDatFile(self, file):
        '''
        *  Helper function, used in readAlignment()
        *  Reads .dat files with alignment RMSF values
        '''
        return(pd.read_csv(file, 
                           delim_whitespace = True, 
                           skiprows=[0], 
                           header = None, 
                           names = ["res", "rmsf"]).set_index("res"))
    
    def readAlignment(self, alignment):
        '''
        *  Helper function, used in align() 
        *  Reads all the window-alignments for alignment 'alignment'
        '''
        print(f"Collecting alignment {alignment}")
        
        folder = os.path.join(self.struct_folder, alignment)
        
        fileList = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f[-3:]=='dat']
        
        df = pd.DataFrame()
        
        for f in fileList:
            file = os.path.join(folder, f)
            df = df.join(self.readDatFile(file),
                                          rsuffix = f.replace('.dat', ''), 
                                          how = "outer")
            df = df.reindex(pd.RangeIndex(self.length + 1))
            df.index.name = 'res'
        return(df)
    
    def align(self):
        '''
        *  One of the main functions in this class
        *  Does the alignments
        '''
       
        self.alRMSF = {}
        self.alAvgRMSF = {}
        
        for a in self.alignments:
            p1, p2 = a.split(":")
            print(f"Aligning {p1} and {p2} with a window of {self.window}")
            
            self.windowAlign(a) # align
            df = self.readAlignment(a) # read alignment
            self.alRMSF[a] = df
            self.alAvgRMSF[a] = df.mean(axis = 1)
                         
            # save alignment for later (when precomputed = True)
            df.to_csv(os.path.join(self.struct_folder, a + "_df.csv"))
            
    def readPrecomputed(self):
        '''
        *  Helper function, used in __init__
        *  Reads saved alignment .csv files when precomputed=True
        '''
        self.alRMSF = {}
        self.alAvgRMSF = {}
        
        for a in self.alignments:
            p1, p2 = a.split(":")
            df = pd.read_csv(os.path.join(self.struct_folder, a + "_df.csv")).set_index("res")
            self.alRMSF[a] = df
            self.alAvgRMSF[a] = df.mean(axis = 1)
            
    def plotAvgRMSF(self, a):
        '''
        Plots the average RMSF profile of an alignment
        '''                         
        fig, ax = plt.subplots(figsize = (6,3))
        ax.plot(self.alAvgRMSF[a], color='royalblue', lw = 1)
                         
        ax.set_xlabel("Residue number")
        ax.set_ylabel("Window-estimated RMSF")
        ax.set_title(f"{a} alignment profile")
        ax.legend([r"$\overline{RMSF}$"], fontsize = 8)
        
    def plotQntlRMSF(self, a):
        '''
        Plots the 0.25, 0.5, and 0.75 quantiles of the alignment profile
        '''
        fig, ax = plt.subplots(figsize = (6,3))
                         
        x = np.arange(0, self.length+1)
        y = np.array(self.alRMSF[a].quantile(q = 0.5, axis = 1))
        y1 = np.array(self.alRMSF[a].quantile(q = 0.25, axis = 1))
        y2 = np.array(self.alRMSF[a].quantile(q = 0.75, axis = 1))
                         
        ax.plot(x, y, lw = 0.5, color = "black")
        ax.fill_between(x, y1, y2, color = 'tab:blue', alpha = 0.5)
    
        ax.set_xlabel("Residue number")
        ax.set_ylabel("Window-estimated RMSF")
        ax.set_title(f"{a} alignment profile")
        ax.legend(["Median", "IQR"], fontsize = 8)
        
        
    def Fisher(self, df):
        '''
        *  Helper function, used in pValues()
        *  Applies Fisher's method to combine p-values
        '''
        p = df.apply(lambda row : stats.combine_pvalues(row[~np.isnan(row)]+1e-06, method='fisher')[1], axis = 1)
        return(p)
        
    def pValues(self):
        '''
        *  Combine the p-values using Fisher's method
        '''
        # gather variances per alignment
        self.alVariance = {}
        for a in self.alignments:
            p1, p2 = a.split(":")
            self.alVariance[a] = self.variances[p1] + self.variances[p2]
        
        # get the Boltzmann 1 - CDF values for each 
        self.alCDF = {}
        for a in self.alignments:
            var = np.sqrt(self.alVariance[a])
            Xbar = self.alAvgRMSF[a]
            
            F = stats.maxwell.cdf(Xbar, scale = var)
            mu = stats.maxwell.mean(scale = var)
            
            less = Xbar < mu
            F_ = less * F + (1-less) * (1-F)
            
            self.alCDF[a] = pd.DataFrame({a: F_})
            
        # bundle the 1 - CDF values
        self.bundledCDF = {}
        for b in self.bundles.keys():
            df = pd.DataFrame()
            for a in self.bundles[b]:
                df = df.join(self.alCDF[a], how = 'outer')

            self.bundledCDF[b] = df
        
        # combine the p-values 
        self.p = {}
        for b in self.bundles.keys():
            self.p[b] = self.Fisher(self.bundledCDF[b])
            
    def plotPValues(self, b):
        '''
        *  Plot the p-values
        '''
        fig, ax = plt.subplots(figsize = (6,3))
        
        ax.plot(self.p[b], lw = 1, color = 'black')
        
        ax.set_xlabel('Residue number')
        ax.set_ylabel("Fisher's combined $\chi^2$ p-values")
        
    def pymolSel(self, b, thresh):
        
        aa = self.p[b][self.p[b]<=thresh].index.tolist()
        s = f"sele 'important', resi {aa[0]} " + " ".join(['or resi '+str(x) for x in aa[1:]])
        return s
        