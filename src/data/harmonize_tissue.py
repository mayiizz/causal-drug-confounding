"""Tissue harmonization using DepMap model_list as the single source of truth."""

import pandas as pd
import numpy as np

TISSUE_HARMONIZATION_MAP = {
    # GDSC2 CANCER_TYPE values (underscore format)
    'non-small_cell_lung_carcinoma': 'Lung',
    'small_cell_lung_carcinoma': 'Lung',
    'lung_nsclc': 'Lung',
    'lung_sclc': 'Lung',
    'lung_carcinoma': 'Lung',
    'lung_adenocarcinoma': 'Lung',
    'lung_squamous_cell_carcinoma': 'Lung',
    'breast_carcinoma': 'Breast',
    'breast_ductal_carcinoma': 'Breast',
    'breast_lobular_carcinoma': 'Breast',
    'colorectal_carcinoma': 'Colorectal',
    'colon_carcinoma': 'Colorectal',
    'rectal_carcinoma': 'Colorectal',
    'melanoma': 'Skin',
    'skin_carcinoma': 'Skin',
    'pancreatic_carcinoma': 'Pancreas',
    'pancreas_carcinoma': 'Pancreas',
    'ovarian_carcinoma': 'Ovary',
    'ovary_carcinoma': 'Ovary',
    'gastric_carcinoma': 'Gastric',
    'stomach_carcinoma': 'Gastric',
    'prostate_carcinoma': 'Prostate',
    'renal_cell_carcinoma': 'Kidney',
    'kidney_carcinoma': 'Kidney',
    'bladder_carcinoma': 'Bladder',
    'urinary_tract_carcinoma': 'Bladder',
    'cns_cancer': 'CNS_Brain',
    'glioma': 'CNS_Brain',
    'glioblastoma': 'CNS_Brain',
    'medulloblastoma': 'CNS_Brain',
    'neuroblastoma': 'CNS_Brain',
    'head_and_neck_carcinoma': 'Head_and_Neck',
    'head_and_neck_squamous_cell_carcinoma': 'Head_and_Neck',
    'aero_digestive_tract': 'Head_and_Neck',
    'aerodigestive tract': 'Head_and_Neck',
    'aerodigestive_tract': 'Head_and_Neck',
    'esophageal_carcinoma': 'Esophagus',
    'esophagus_carcinoma': 'Esophagus',
    'liver_cancer': 'Liver',
    'hepatocellular_carcinoma': 'Liver',
    'liver_carcinoma': 'Liver',
    'hepatoma': 'Liver',
    'thyroid_carcinoma': 'Thyroid',
    'cervical_carcinoma': 'Cervix',
    'cervix_carcinoma': 'Cervix',
    'endometrial_carcinoma': 'Uterus',
    'endometrium_carcinoma': 'Uterus',
    'uterus_carcinoma': 'Uterus',
    'bone_cancer': 'Bone',
    'ewing_sarcoma': 'Bone',
    'osteosarcoma': 'Bone',
    'biliary_tract_cancer': 'Biliary_Tract',
    'biliary_tract_carcinoma': 'Biliary_Tract',
    'cholangiocarcinoma': 'Biliary_Tract',
    'biliary tract': 'Biliary_Tract',
    'biliary_tract': 'Biliary_Tract',
    'lymphoid_leukemia': 'Lymphoid',
    'lymphoid': 'Lymphoid',
    'lymphoma': 'Lymphoid',
    'myeloid_leukemia': 'Myeloid',
    'myeloid': 'Myeloid',
    'aml': 'Myeloid',
    'all': 'Lymphoid',
    'cll': 'Lymphoid',
    'cml': 'Myeloid',
    'multiple_myeloma': 'Plasma_Cell',
    'plasma_cell': 'Plasma_Cell',
    'plasma_cell_myeloma': 'Plasma_Cell',
    'rhabdomyosarcoma': 'Soft_Tissue',
    'soft_tissue': 'Soft_Tissue',
    'soft tissue': 'Soft_Tissue',
    'fibroblast': 'Fibroblast',
    'mesothelioma': 'Pleura',
    'pleura': 'Pleura',
    'vulva': 'Vulva',
    'eye': 'Eye',
    'eye_cancer': 'Eye',
    'retinoblastoma': 'Eye',
    'uveal_melanoma': 'Eye',
    
    # CCLE/DepMap values (space format)
    'bile duct': 'Biliary_Tract',
    'bone marrow': 'Bone',
    'brain': 'CNS_Brain',
    'breast': 'Breast',
    'central nervous system': 'CNS_Brain',
    'cervical': 'Cervix',
    'colorectal': 'Colorectal',
    'endometrium': 'Uterus',
    'esophagus': 'Esophagus',
    'eye': 'Eye',
    'fibroblast': 'Fibroblast',
    'gastrointestinal tract': 'Gastric',
    'head and neck': 'Head_and_Neck',
    'kidney': 'Kidney',
    'liver': 'Liver',
    'lung': 'Lung',
    'lymphocyte': 'Lymphoid',
    'myelocyte': 'Myeloid',
    'ovary': 'Ovary',
    'pancreas': 'Pancreas',
    'plasma cell': 'Plasma_Cell',
    'pleura': 'Pleura',
    'prostate': 'Prostate',
    'skin': 'Skin',
    'soft tissue': 'Soft_Tissue',
    'stomach': 'Gastric',
    'thyroid': 'Thyroid',
    'upper aerodigestive tract': 'Head_and_Neck',
    'urinary tract': 'Urogenital',
    'uterus': 'Uterus',
    
    # Generic fallbacks
    'blood': 'Blood',
    'bone': 'Bone',
    'cns': 'CNS_Brain',
    'cns/brain': 'CNS_Brain',
    'cns brain': 'CNS_Brain',
    'gastric': 'Gastric',
    'stomach': 'Gastric',
    'large_intestine': 'Colorectal',
    'urogenital_system': 'Urogenital',
    'urogenital system': 'Urogenital',
}


def harmonize_tissue(series: pd.Series) -> pd.Series:
    """Map raw tissue names to harmonized ontology."""
    # Step 1: lowercase, strip, normalize spaces
    cleaned = series.astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
    
    # Step 2: map via dictionary
    mapped = cleaned.map(TISSUE_HARMONIZATION_MAP)
    
    # Step 3: try replacing underscores with spaces and re-map
    unmapped_mask = mapped.isna()
    if unmapped_mask.any():
        spaced = cleaned[unmapped_mask].str.replace('_', ' ')
        remapped = spaced.map(TISSUE_HARMONIZATION_MAP)
        mapped.loc[unmapped_mask] = remapped
    
    # Step 4: final fallback — title-case with underscores
    still_unmapped = mapped.isna()
    if still_unmapped.any():
        fallback = cleaned[still_unmapped].str.title().str.replace(' ', '_')
        mapped.loc[still_unmapped] = fallback
        
    return mapped


def validate_tissue_coverage(df: pd.DataFrame, tissue_col: str = 'tissue_type') -> dict:
    total = len(df)
    unmapped = df[tissue_col].isna().sum()
    return {
        'total_records': total,
        'unmapped_records': int(unmapped),
        'coverage_pct': round((total - unmapped) / total * 100, 2),
        'unique_harmonized_tissues': int(df[tissue_col].nunique()),
        'tissue_distribution': df[tissue_col].value_counts().to_dict()
    }