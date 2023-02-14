import json
import logging
import os
from abc import abstractmethod, ABC
from typing import List, Tuple, Dict, Sequence, Optional

import numpy as np
import pandas as pd
import scipy.stats as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.stats import rv_continuous
from numpy import (where, arange, putmask, ravel, sum, shape,
                   log, sqrt, exp, arctanh, tan, sin, arcsin, arctan,
                   tanh, cos, cosh, sinh, log1p, expm1)


class frechet_r_gen(rv_continuous):
    """A Frechet right (or Weibull minimum) continuous random variable.
    %(before_notes)s
    See Also
    --------
    weibull_min : The same distribution as `frechet_r`.
    frechet_l, weibull_max
    Notes
    -----
    The probability density function for `frechet_r` is::
        frechet_r.pdf(x, c) = c * x**(c-1) * exp(-x**c)
    for ``x > 0``, ``c > 0``.
    %(example)s
    """
    def _pdf(self, x, c):
        return c*pow(x, c-1)*exp(-pow(x, c))

    def _logpdf(self, x, c):
        return log(c) + (c-1)*log(x) - pow(x, c)

    def _cdf(self, x, c):
        return -expm1(-pow(x, c))

    def _ppf(self, q, c):
        return pow(-log1p(-q), 1.0/c)

    def _munp(self, n, c):
        return special.gamma(1.0+n*1.0/c)

    def _entropy(self, c):
        return -_EULER / c - log(c) + _EULER + 1
frechet_r = frechet_r_gen(a=0.0, name='frechet_r')
weibull_min = frechet_r_gen(a=0.0, name='weibull_min')


class frechet_l_gen(rv_continuous):
    """A Frechet left (or Weibull maximum) continuous random variable.
    %(before_notes)s
    See Also
    --------
    weibull_max : The same distribution as `frechet_l`.
    frechet_r, weibull_min
    Notes
    -----
    The probability density function for `frechet_l` is::
        frechet_l.pdf(x, c) = c * (-x)**(c-1) * exp(-(-x)**c)
    for ``x < 0``, ``c > 0``.
    %(example)s
    """
    def _pdf(self, x, c):
        return c*pow(-x, c-1)*exp(-pow(-x, c))

    def _cdf(self, x, c):
        return exp(-pow(-x, c))

    def _ppf(self, q, c):
        return -pow(-log(q), 1.0/c)

    def _munp(self, n, c):
        val = special.gamma(1.0+n*1.0/c)
        if (int(n) % 2):
            sgn = -1
        else:
            sgn = 1
        return sgn * val

    def _entropy(self, c):
        return -_EULER / c - log(c) + _EULER + 1
frechet_l = frechet_l_gen(b=0.0, name='frechet_l')
weibull_max = frechet_l_gen(b=0.0, name='weibull_max')


MetricDictType = Dict[str, Tuple[str, Sequence[float], float, float, float, float]]


class PhyschemScaler:
    def __init__(self, descriptor_list: List[str], dists: MetricDictType):
        self.descriptor_list = descriptor_list
        self.dists = dists
        self.cdfs = self.prepare_cdfs()

    def prepare_cdfs(self):
        cdfs = {}

        dist_subset = dict(filter(lambda elem: elem[0] in self.descriptor_list, self.dists.items()))

        for descriptor_name, (dist, params, minV, maxV, avg, std) in dist_subset.items():
            arg = params[:-2]  # type: ignore
            loc = params[-2]  # type: ignore
            scale = params[-1]  # type: ignore
            if dist == 'frechet_l':
                dist = frechet_l
            elif dist == 'frechet_r':
                dist = frechet_r 
            else:
                dist = getattr(st, dist)

            # make the cdf with the parameters
            def cdf(v, dist=dist, arg=arg, loc=loc, scale=scale, minV=minV, maxV=maxV):
                v = dist.cdf(np.clip(v, minV, maxV), loc=loc, scale=scale, *arg)
                return np.clip(v, 0.0, 1.0)

            cdfs[descriptor_name] = cdf

        return cdfs

    def transform(self, X):
        # transform each column with the corresponding descriptor
        transformed_list = [
            self.cdfs[descriptor](X[:, idx])[..., np.newaxis] for idx, descriptor in enumerate(self.descriptor_list)
        ]
        transformed = np.concatenate(transformed_list, axis=1)

        # make sure the shape is intact
        assert X.shape == transformed.shape

        return transformed

    def transform_single(self, X):
        assert len(X.shape) == 1, 'When using transform_single, input should have a 1-dimensional shape (e.g. (200,))'

        X = X[np.newaxis, :]
        transformed = self.transform(X)
        transformed = transformed.squeeze(axis=0)
        return transformed        

  
def get_descriptor_subset(subset: str) -> List[str]:
    if subset == 'all':
        return  get_all_descriptor_names() 
    elif subset == 'simple':
        return  get_simple_descriptor_subset() 
    elif subset == 'uncorrelated':
        return  get_uncorrelated_descriptor_subset()
    elif subset == 'fragment':
        return  get_fragment_descriptor_subset() 
    elif subset == 'graph':
        return  get_graph_descriptor_subset() 
    elif subset == 'surface':
        return  get_surface_descriptor_subset() 
    elif subset == 'druglikeness':
        return  get_druglikeness_descriptor_subset() 
    elif subset == 'logp':
        return  get_logp_descriptor_subset() 
    elif subset == 'refractivity':
        return  get_refractivity_descriptor_subset() 
    elif subset == 'estate':
        return  get_estate_descriptor_subset() 
    elif subset == 'charge':
        return  get_charge_descriptor_subset() 
    elif subset == 'general':
        return  get_general_descriptor_subset() 
    elif subset == 'surface-logp':
        return  get_logp_descriptor_subset() + get_surface_descriptor_subset()
    else:
        raise ValueError(
            f'Unrecognised descriptor subset: {subset} (should be "all", "simple",'
            f'"uncorrelated", "fragment", "graph", "logp", "refractivity",'
            f'"estate", "druglikeness", "surface", "charge", "general").'
        )

     

def rdkit_dense_array_to_np(dense_fp, dtype=np.int32):
    """
    Converts RDKit ExplicitBitVect to 1D numpy array with specified dtype.
    Args:
        dense_fp (ExplicitBitVect or np.ndarray): fingerprint
        dtype: dtype of the returned array
    Returns:
        Numpy matrix with shape (fp_len,)
    """
    dense_fp = np.array(dense_fp, dtype=dtype)
    if len(dense_fp.shape) == 1:
        pass
    elif len(dense_fp.shape) == 2 and dense_fp.shape[0] == 1:
        dense_fp = np.squeeze(dense_fp, axis=0)
    else:
        raise ValueError("Input matrix should either have shape of (fp_size, ) or (1, fp_size).")

    return np.array(dense_fp)
   

def transform_mol(calc, molecule):
    fp = calc.CalcDescriptors(molecule)
    fp = np.array(fp)
    mask = np.isfinite(fp)
    fp[~mask] = 0
    fp = rdkit_dense_array_to_np(fp, dtype=float)
    return fp

     

def get_simple_descriptor_subset() -> List[str]:
    return [
        'FpDensityMorgan2',
        'FractionCSP3',
        'MolLogP',
        'MolWt',
        'NumHAcceptors',
        'NumHDonors',
        'NumRotatableBonds',
        'TPSA',
    ]

    
def get_refractivity_descriptor_subset() -> List[str]:
    return [
        'MolMR',
        'SMR_VSA1',
        'SMR_VSA10',
        'SMR_VSA2',
        'SMR_VSA3',
        'SMR_VSA4',
        'SMR_VSA5',
        'SMR_VSA6',
        'SMR_VSA7',
        'SMR_VSA8',
        'SMR_VSA9',
    ]

     
def get_logp_descriptor_subset() -> List[str]:
    """LogP descriptors and VSA/LogP descriptors
    SlogP_VSA: VSA of atoms contributing to a specified bin of SlogP
    """

    return [
        'MolLogP',
        'SlogP_VSA1',
        'SlogP_VSA10',
        'SlogP_VSA11',
        'SlogP_VSA12',
        'SlogP_VSA2',
        'SlogP_VSA3',
        'SlogP_VSA4',
        'SlogP_VSA5',
        'SlogP_VSA6',
        'SlogP_VSA7',
        'SlogP_VSA8',
        'SlogP_VSA9',
    ]

     
def get_graph_descriptor_subset() -> List[str]:
    """ Graph descriptors (https://www.rdkit.org/docs/source/rdkit.Chem.GraphDescriptors.html) """
    return [
        'BalabanJ',
        'BertzCT',
        'Chi0',
        'Chi0n',
        'Chi0v',
        'Chi1',
        'Chi1n',
        'Chi1v',
        'Chi2n',
        'Chi2v',
        'Chi3n',
        'Chi3v',
        'Chi4n',
        'Chi4v',
        'HallKierAlpha',
        'Ipc',
        'Kappa1',
        'Kappa2',
        'Kappa3',
    ]


def get_surface_descriptor_subset() -> List[str]:
    """MOE-like surface descriptors
    EState_VSA: VSA (van der Waals surface area) of atoms contributing to a specified bin of e-state
    SlogP_VSA: VSA of atoms contributing to a specified bin of SlogP
    SMR_VSA: VSA of atoms contributing to a specified bin of molar refractivity
    PEOE_VSA: VSA of atoms contributing to a specified bin of partial charge (Gasteiger)
    LabuteASA: Labute's approximate surface area descriptor
    """
    return [
        'SlogP_VSA1',
        'SlogP_VSA10',
        'SlogP_VSA11',
        'SlogP_VSA12',
        'SlogP_VSA2',
        'SlogP_VSA3',
        'SlogP_VSA4',
        'SlogP_VSA5',
        'SlogP_VSA6',
        'SlogP_VSA7',
        'SlogP_VSA8',
        'SlogP_VSA9',
        'SMR_VSA1',
        'SMR_VSA10',
        'SMR_VSA2',
        'SMR_VSA3',
        'SMR_VSA4',
        'SMR_VSA5',
        'SMR_VSA6',
        'SMR_VSA7',
        'SMR_VSA8',
        'SMR_VSA9',
        'EState_VSA1',
        'EState_VSA10',
        'EState_VSA11',
        'EState_VSA2',
        'EState_VSA3',
        'EState_VSA4',
        'EState_VSA5',
        'EState_VSA6',
        'EState_VSA7',
        'EState_VSA8',
        'EState_VSA9',
        'LabuteASA',
        'PEOE_VSA1',
        'PEOE_VSA10',
        'PEOE_VSA11',
        'PEOE_VSA12',
        'PEOE_VSA13',
        'PEOE_VSA14',
        'PEOE_VSA2',
        'PEOE_VSA3',
        'PEOE_VSA4',
        'PEOE_VSA5',
        'PEOE_VSA6',
        'PEOE_VSA7',
        'PEOE_VSA8',
        'PEOE_VSA9',
        'TPSA',
    ]

 
def get_druglikeness_descriptor_subset() -> List[str]:
    """ Descriptors commonly used to assess druglikeness"""
    return [
        'TPSA',
        'MolLogP',
        'MolMR',
        'ExactMolWt',
        'FractionCSP3',
        'HeavyAtomCount',
        'MolWt',
        'NHOHCount',
        'NOCount',
        'NumAliphaticCarbocycles',
        'NumAliphaticHeterocycles',
        'NumAliphaticRings',
        'NumAromaticCarbocycles',
        'NumAromaticHeterocycles',
        'NumAromaticRings',
        'NumHAcceptors',
        'NumHDonors',
        'NumHeteroatoms',
        'NumRotatableBonds',
        'NumSaturatedCarbocycles',
        'NumSaturatedHeterocycles',
        'NumSaturatedRings',
        'RingCount',
        'qed',
    ]

 
def get_fragment_descriptor_subset() -> List[str]:
    return [
        'NHOHCount',
        'NOCount',
        'NumAliphaticCarbocycles',
        'NumAliphaticHeterocycles',
        'NumAliphaticRings',
        'NumAromaticCarbocycles',
        'NumAromaticHeterocycles',
        'NumAromaticRings',
        'NumHAcceptors',
        'NumHDonors',
        'NumHeteroatoms',
        'NumRotatableBonds',
        'NumSaturatedCarbocycles',
        'NumSaturatedHeterocycles',
        'NumSaturatedRings',
        'RingCount',
        'fr_Al_COO',
        'fr_Al_OH',
        'fr_Al_OH_noTert',
        'fr_ArN',
        'fr_Ar_COO',
        'fr_Ar_N',
        'fr_Ar_NH',
        'fr_Ar_OH',
        'fr_COO',
        'fr_COO2',
        'fr_C_O',
        'fr_C_O_noCOO',
        'fr_C_S',
        'fr_HOCCN',
        'fr_Imine',
        'fr_NH0',
        'fr_NH1',
        'fr_NH2',
        'fr_N_O',
        'fr_Ndealkylation1',
        'fr_Ndealkylation2',
        'fr_Nhpyrrole',
        'fr_SH',
        'fr_aldehyde',
        'fr_alkyl_carbamate',
        'fr_alkyl_halide',
        'fr_allylic_oxid',
        'fr_amide',
        'fr_amidine',
        'fr_aniline',
        'fr_aryl_methyl',
        'fr_azide',
        'fr_azo',
        'fr_barbitur',
        'fr_benzene',
        'fr_benzodiazepine',
        'fr_bicyclic',
        'fr_diazo',
        'fr_dihydropyridine',
        'fr_epoxide',
        'fr_ester',
        'fr_ether',
        'fr_furan',
        'fr_guanido',
        'fr_halogen',
        'fr_hdrzine',
        'fr_hdrzone',
        'fr_imidazole',
        'fr_imide',
        'fr_isocyan',
        'fr_isothiocyan',
        'fr_ketone',
        'fr_ketone_Topliss',
        'fr_lactam',
        'fr_lactone',
        'fr_methoxy',
        'fr_morpholine',
        'fr_nitrile',
        'fr_nitro',
        'fr_nitro_arom',
        'fr_nitro_arom_nonortho',
        'fr_nitroso',
        'fr_oxazole',
        'fr_oxime',
        'fr_para_hydroxylation',
        'fr_phenol',
        'fr_phenol_noOrthoHbond',
        'fr_phos_acid',
        'fr_phos_ester',
        'fr_piperdine',
        'fr_piperzine',
        'fr_priamide',
        'fr_prisulfonamd',
        'fr_pyridine',
        'fr_quatN',
        'fr_sulfide',
        'fr_sulfonamd',
        'fr_sulfone',
        'fr_term_acetylene',
        'fr_tetrazole',
        'fr_thiazole',
        'fr_thiocyan',
        'fr_thiophene',
        'fr_unbrch_alkane',
        'fr_urea',
    ]

 
def get_estate_descriptor_subset() -> List[str]:
    """Electrotopological state (e-state) and VSA/e-state descriptors
    EState_VSA: VSA (van der Waals surface area) of atoms contributing to a specified bin of e-state
    VSA_EState: e-state values of atoms contributing to a specific bin of VSA
    """
    return [
        'EState_VSA1',
        'EState_VSA10',
        'EState_VSA11',
        'EState_VSA2',
        'EState_VSA3',
        'EState_VSA4',
        'EState_VSA5',
        'EState_VSA6',
        'EState_VSA7',
        'EState_VSA8',
        'EState_VSA9',
        'VSA_EState1',
        'VSA_EState10',
        'VSA_EState2',
        'VSA_EState3',
        'VSA_EState4',
        'VSA_EState5',
        'VSA_EState6',
        'VSA_EState7',
        'VSA_EState8',
        'VSA_EState9',
        'MaxAbsEStateIndex',
        'MaxEStateIndex',
        'MinAbsEStateIndex',
        'MinEStateIndex',
    ]

 
def get_charge_descriptor_subset() -> List[str]:
    """
    Partial charge and VSA/charge descriptors
    PEOE: Partial Equalization of Orbital Electronegativities (Gasteiger partial atomic charges)
    PEOE_VSA: VSA of atoms contributing to a specific bin of partial charge
    """
    return [
        'PEOE_VSA1',
        'PEOE_VSA10',
        'PEOE_VSA11',
        'PEOE_VSA12',
        'PEOE_VSA13',
        'PEOE_VSA14',
        'PEOE_VSA2',
        'PEOE_VSA3',
        'PEOE_VSA4',
        'PEOE_VSA5',
        'PEOE_VSA6',
        'PEOE_VSA7',
        'PEOE_VSA8',
        'PEOE_VSA9',
        'MaxAbsPartialCharge',
        'MaxPartialCharge',
        'MinAbsPartialCharge',
        'MinPartialCharge',
    ]

 
def get_general_descriptor_subset() -> List[str]:
    """ Descriptors from https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html """
    return [
        'MaxAbsPartialCharge',
        'MaxPartialCharge',
        'MinAbsPartialCharge',
        'MinPartialCharge',
        'ExactMolWt',
        'MolWt',
        'FpDensityMorgan1',
        'FpDensityMorgan2',
        'FpDensityMorgan3',
        'HeavyAtomMolWt',
        'NumRadicalElectrons',
        'NumValenceElectrons',
    ]

 
def get_uncorrelated_descriptor_subset() -> List[str]:
    """
    Column names are sorted starting with the non-informative descriptors, then the rest are ordered
    from most correlated to least correlated. This will return the n least correlated descriptors.
    Args:
        subset_size: how many to return
    Returns:
        List of descriptors
    """
    columns_sorted_by_correlation = [
        'fr_sulfone',
        'MinPartialCharge',
        'fr_C_O_noCOO',
        'fr_hdrzine',
        'fr_Ndealkylation2',
        'NumAromaticHeterocycles',
        'fr_N_O',
        'fr_piperdine',
        'fr_HOCCN',
        'fr_Nhpyrrole',
        'NumHAcceptors',
        'NumHeteroatoms',
        'fr_C_O',
        'VSA_EState5',
        'fr_Al_OH',
        'SlogP_VSA9',
        'fr_benzodiazepine',
        'VSA_EState6',
        'fr_Ar_N',
        'VSA_EState7',
        'fr_COO2',
        'VSA_EState3',
        'fr_Imine',
        'fr_sulfide',
        'FractionCSP3',
        'fr_imidazole',
        'fr_azo',
        'NumHDonors',
        'fr_COO',
        'fr_ether',
        'fr_nitro',
        'NumSaturatedHeterocycles',
        'fr_lactam',
        'fr_aniline',
        'NumAliphaticCarbocycles',
        'fr_para_hydroxylation',
        'SMR_VSA2',
        'MaxAbsPartialCharge',
        'fr_thiocyan',
        'NHOHCount',
        'fr_ester',
        'fr_aldehyde',
        'SMR_VSA8',
        'fr_halogen',
        'fr_NH0',
        'fr_furan',
        'fr_tetrazole',
        'HeavyAtomCount',
        'NumRotatableBonds',
        'NumSaturatedCarbocycles',
        'fr_SH',
        'fr_Ar_NH',
        'SlogP_VSA7',
        'fr_ketone',
        'fr_alkyl_halide',
        'fr_NH1',
        'NumRadicalElectrons',
        'MaxPartialCharge',
        'fr_ArN',
        'fr_imide',
        'fr_priamide',
        'fr_hdrzone',
        'fr_azide',
        'NumAromaticCarbocycles',
        'NOCount',
        'fr_isocyan',
        'RingCount',
        'fr_nitroso',
        'EState_VSA11',
        'MinAbsPartialCharge',
        'fr_Ar_COO',
        'fr_prisulfonamd',
        'fr_sulfonamd',
        'VSA_EState4',
        'fr_quatN',
        'fr_NH2',
        'fr_epoxide',
        'fr_allylic_oxid',
        'fr_piperzine',
        'VSA_EState1',
        'NumAliphaticHeterocycles',
        'fr_Ndealkylation1',
        'fr_Al_OH_noTert',
        'fr_aryl_methyl',
        'NumAromaticRings',
        'fr_bicyclic',
        'fr_methoxy',
        'fr_oxazole',
        'fr_barbitur',
        'NumAliphaticRings',
        'fr_Ar_OH',
        'fr_phos_ester',
        'fr_thiophene',
        'fr_nitrile',
        'fr_dihydropyridine',
        'VSA_EState2',
        'fr_nitro_arom',
        'SlogP_VSA11',
        'fr_thiazole',
        'fr_ketone_Topliss',
        'fr_term_acetylene',
        'fr_isothiocyan',
        'fr_urea',
        'fr_nitro_arom_nonortho',
        'fr_lactone',
        'fr_diazo',
        'fr_amide',
        'fr_alkyl_carbamate',
        'fr_Al_COO',
        'fr_amidine',
        'fr_phos_acid',
        'fr_oxime',
        'fr_guanido',
        'fr_C_S',
        'NumSaturatedRings',
        'fr_benzene',
        'fr_phenol',
        'fr_unbrch_alkane',
        'fr_phenol_noOrthoHbond',
        'fr_pyridine',
        'fr_morpholine',
        'MaxAbsEStateIndex',
        'ExactMolWt',
        'MolWt',
        'Chi0',
        'LabuteASA',
        'Chi0n',
        'NumValenceElectrons',
        'Chi3n',
        'Chi0v',
        'Chi3v',
        'Chi1',
        'Chi1n',
        'Chi1v',
        'FpDensityMorgan2',
        'HeavyAtomMolWt',
        'Kappa1',
        'SMR_VSA7',
        'Chi2n',
        'Chi2v',
        'Kappa2',
        'Chi4n',
        'SMR_VSA5',
        'MolMR',
        'EState_VSA10',
        'BertzCT',
        'MinEStateIndex',
        'SMR_VSA1',
        'FpDensityMorgan1',
        'VSA_EState10',
        'SlogP_VSA2',
        'SMR_VSA10',
        'HallKierAlpha',
        'VSA_EState9',
        'TPSA',
        'MaxEStateIndex',
        'Chi4v',
        'SMR_VSA4',
        'MolLogP',
        'qed',
        'VSA_EState8',
        'EState_VSA2',
        'SMR_VSA6',
        'PEOE_VSA1',
        'EState_VSA1',
        'SlogP_VSA8',
        'SlogP_VSA6',
        'SlogP_VSA5',
        'SlogP_VSA10',
        'BalabanJ',
        'Kappa3',
        'EState_VSA4',
        'PEOE_VSA6',
        'EState_VSA9',
        'PEOE_VSA2',
        'PEOE_VSA5',
        'SMR_VSA3',
        'SlogP_VSA3',
        'EState_VSA7',
        'EState_VSA3',
        'PEOE_VSA7',
        'SlogP_VSA1',
        'SMR_VSA9',
        'EState_VSA8',
        'EState_VSA6',
        'PEOE_VSA3',
        'MinAbsEStateIndex',
        'PEOE_VSA14',
        'FpDensityMorgan3',
        'PEOE_VSA12',
        'SlogP_VSA4',
        'PEOE_VSA9',
        'PEOE_VSA13',
        'PEOE_VSA10',
        'PEOE_VSA8',
        'EState_VSA5',
        'SlogP_VSA12',
        'PEOE_VSA4',
        'Ipc',
        'PEOE_VSA11',
    ]

    return columns_sorted_by_correlation

     