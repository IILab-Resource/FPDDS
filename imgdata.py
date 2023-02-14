from torch.utils.data import Dataset
from PIL import Image
import requests
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import dgl.backend as F
import numpy as np 
from tqdm import tqdm 
import torch 
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from molfeaturizer import *
# feature_extractor = AutoFeatureExtractor.from_pretrained("newoutputs/checkpoint-76582")


nbits = 1024
longbits = 16384
# RDKit descriptors -->
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
# dictionary
fpFunc_dict = {}
fpFunc_dict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpFunc_dict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpFunc_dict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpFunc_dict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpFunc_dict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpFunc_dict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpFunc_dict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpFunc_dict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpFunc_dict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits) # 1024
fpFunc_dict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits) # 1024
fpFunc_dict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits) # 1024
fpFunc_dict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits) # 1024
fpFunc_dict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2) # 1024
fpFunc_dict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2) # 1024
fpFunc_dict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2) # 1024
fpFunc_dict['rdkDes'] = lambda m: calc.CalcDescriptors(m)
class ImgDataset(Dataset):
    def __init__(self, drug1, drug2, drug2smiles, contexts, contexts_dict, labels, dataset_name, feature_extractor, fp_name='hashap'):
        self.drug1 = drug1
        self.drug2 = drug2
        self.contexts = contexts
        self.contexts_dict = contexts_dict
        self.labels = F.zerocopy_from_numpy(labels.astype(np.float32))
        self.length = len(self.labels)
        self.drug2smiles = drug2smiles
        self.drug2graphs = {}
        self.drug2fps = {}
        self.feature_extractor = feature_extractor
        self.fp_name = fp_name
        self._pre_process()

    def _pre_process(self):
        for key, smile in tqdm(self.drug2smiles.items()):
            mol = Chem.MolFromSmiles( smile.strip() )
            Draw.MolToFile( mol,  "tmp.png" )
            image = Image.open("tmp.png")
            inputs = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"][0]
            self.drug2graphs[key] = inputs
            fp = fpFunc_dict[self.fp_name](mol)
            self.drug2fps[key] = np.asarray(fp)
            # print(np.asarray(fp).shape)
             

    def __getitem__(self, idx):
        
        graph_drug1 = self.drug2graphs[ str(self.drug1[idx]) ]
        graph_drug2 = self.drug2graphs[ str(self.drug2[idx]) ]
        context_feature = self.contexts_dict[ self.contexts[idx] ]
        label = self.labels[idx]
        fp1 = self.drug2fps[str(self.drug1[idx])]
        fp2 = self.drug2fps[str(self.drug2[idx])]
        

        return graph_drug1,graph_drug2,torch.FloatTensor(fp1), torch.FloatTensor(fp2), torch.FloatTensor(context_feature), label
 

    def __len__(self):
        return self.length

# simple:8
# uncorrelated: 200
# fragment: 101
# graph: 19
# surface: 49
# druglikeness: 24

class FPDataset(Dataset):
    def __init__(self, drug1, drug2, drug2smiles, contexts, contexts_dict, labels, dataset_name, fp_name='hashap', descript_set='logp'):
        self.drug1 = drug1
        self.drug2 = drug2
        self.contexts = contexts
        self.contexts_dict = contexts_dict
        self.labels =  labels 
        self.length = len(self.labels)
        self.drug2smiles = drug2smiles
        self.drug2fps = {}
        self.fp_name = fp_name
        descriptors = get_descriptor_subset(descript_set)
        self.calc = MolecularDescriptorCalculator(descriptors)
        self.drug2descriptor = {}
        distributions_path =  'physchem_distributions.json'

        with open(distributions_path) as fp:
            self.distributions = json.load(fp)
        self.scaler = PhyschemScaler(descriptor_list=descriptors, dists=self.distributions)
        self._pre_process()

        
    def _pre_process(self):
        for key, smile in tqdm(self.drug2smiles.items()):
            mol = Chem.MolFromSmiles( smile.strip() )
            fp = fpFunc_dict[self.fp_name](mol)
            self.drug2fps[key] = np.asarray(fp)
            fp = transform_mol(self.calc, mol)
            fp = self.scaler.transform_single(fp)
            self.drug2descriptor[key] = fp
 
    def __getitem__(self, idx):
        
        
        context_feature = self.contexts_dict[ self.contexts[idx] ]
        label = self.labels[idx]
        fp1_1 = self.drug2fps[str(self.drug1[idx])]
        fp2_1 = self.drug2fps[str(self.drug2[idx])]

        descript1 = self.drug2descriptor[ str(self.drug1[idx]) ]
        descript2 = self.drug2descriptor[ str(self.drug2[idx]) ]
       
        

        return torch.FloatTensor(fp1_1), torch.FloatTensor(fp2_1), torch.FloatTensor(descript1), torch.FloatTensor(descript2), torch.FloatTensor(context_feature),  torch.FloatTensor([label])
 

    def __len__(self):
        return self.length