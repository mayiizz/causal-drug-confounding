"""Tissue harmonization using DepMap model_list as the single source of truth."""

import re
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
    # Squamous histology of lung — same organ lineage as other lung carcinomas
    'squamous_cell_lung_carcinoma': 'Lung',
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
    # Oral cavity is upper aerodigestive / head-and-neck anatomic site
    'oral_cavity_carcinoma': 'Head_and_Neck',
    'aero_digestive_tract': 'Head_and_Neck',
    'aerodigestive tract': 'Head_and_Neck',
    'aerodigestive_tract': 'Head_and_Neck',
    'esophageal_carcinoma': 'Esophagus',
    'esophagus_carcinoma': 'Esophagus',
    # Squamous histology of esophagus — same organ as esophageal carcinoma
    'esophageal_squamous_cell_carcinoma': 'Esophagus',
    'liver_cancer': 'Liver',
    'hepatocellular_carcinoma': 'Liver',
    'liver_carcinoma': 'Liver',
    'hepatoma': 'Liver',
    'thyroid_carcinoma': 'Thyroid',
    'thyroid_gland_carcinoma': 'Thyroid',
    'cervical_carcinoma': 'Cervix',
    'cervix_carcinoma': 'Cervix',
    'endometrial_carcinoma': 'Uterus',
    'endometrium_carcinoma': 'Uterus',
    'uterus_carcinoma': 'Uterus',
    'bone_cancer': 'Bone',
    'ewing_sarcoma': 'Bone',
    'osteosarcoma': 'Bone',
    # Cartilage-derived sarcoma of bone; DepMap/TCGA group with bone lineage
    'chondrosarcoma': 'Bone',
    'biliary_tract_cancer': 'Biliary_Tract',
    'biliary_tract_carcinoma': 'Biliary_Tract',
    'cholangiocarcinoma': 'Biliary_Tract',
    'biliary tract': 'Biliary_Tract',
    'biliary_tract': 'Biliary_Tract',
    'lymphoid_leukemia': 'Lymphoid',
    'lymphoid': 'Lymphoid',
    'lymphoma': 'Lymphoid',
    # Explicit lymphoma / lymphoid leukemia subtypes (WHO hematopoietic lineage)
    'b_cell_non_hodgkin_lymphoma': 'Lymphoid',
    't_cell_non_hodgkin_lymphoma': 'Lymphoid',
    'burkitt_lymphoma': 'Lymphoid',
    'hodgkin_lymphoma': 'Lymphoid',
    'b_lymphoblastic_leukemia': 'Lymphoid',
    't_lymphoblastic_leukemia': 'Lymphoid',
    'myeloid_leukemia': 'Myeloid',
    'myeloid': 'Myeloid',
    'aml': 'Myeloid',
    'acute_myeloid_leukemia': 'Myeloid',
    'all': 'Lymphoid',
    'cll': 'Lymphoid',
    'cml': 'Myeloid',
    'chronic_myelogenous_leukemia': 'Myeloid',
    'multiple_myeloma': 'Plasma_Cell',
    'plasma_cell': 'Plasma_Cell',
    'plasma_cell_myeloma': 'Plasma_Cell',
    'rhabdomyosarcoma': 'Soft_Tissue',
    'soft_tissue': 'Soft_Tissue',
    'soft tissue': 'Soft_Tissue',
    # Unspecified sarcoma basket → soft-tissue sarcoma lineage (not bone-specified)
    'other_sarcomas': 'Soft_Tissue',
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

    # Generic / residual hematologic
    'blood': 'Blood',
    'other_blood_cancers': 'Blood',
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


# Labels excluded from causal analyses (explicit biological decision, not fallback).
# Keys are stored in raw spelling; lookup uses normalize_tissue_label.
TISSUE_EXCLUDE_LABELS = {
    # Non-malignant / immortalized lines — not cancer tissue confounding
    'non-cancerous',
    'non_cancerous',
    # Heterogeneous multi-organ catch-all — not a single lineage confounder
    'other solid cancers',
    'other_solid_cancers',
}


"""
Stage 1A.1 ontology decisions for former GDSC2 fallbacks
--------------------------------------------------------
Raw                              → Canonical / Action   Reason
Acute Myeloid Leukemia           → Myeloid (Map)        WHO myeloid neoplasm; aligns with aml/cml
B-Cell Non-Hodgkin's Lymphoma    → Lymphoid (Map)       Mature B-cell lymphoma lineage
B-Lymphoblastic Leukemia         → Lymphoid (Map)       B-ALL; lymphoid precursor neoplasm
Burkitt's Lymphoma               → Lymphoid (Map)       Aggressive B-cell lymphoma
Chondrosarcoma                   → Bone (Map)           Cartilage sarcoma of bone (DepMap bone)
Chronic Myelogenous Leukemia     → Myeloid (Map)        BCR-ABL1 myeloid neoplasm; aligns with cml
Esophageal Squamous Cell Ca.     → Esophagus (Map)      Organ site esophagus; histology only
Hodgkin's Lymphoma               → Lymphoid (Map)       Classical lymphoid neoplasm
Non-Cancerous                    → Exclude              Non-malignant models
Oral Cavity Carcinoma            → Head_and_Neck (Map)  Oral cavity = H&N / aerodigestive site
Other Blood Cancers              → Blood (Map)          Unspecified hematologic; existing Blood
Other Sarcomas                   → Soft_Tissue (Map)    Non-bone sarcoma basket
Other Solid Cancers              → Exclude              Multi-organ catch-all, not one lineage
Squamous Cell Lung Carcinoma     → Lung (Map)           Lung organ lineage (like NSCLC/SCLC)
T-Cell Non-Hodgkin's Lymphoma    → Lymphoid (Map)       Mature T-cell lymphoma
T-Lymphoblastic Leukemia         → Lymphoid (Map)       T-ALL; lymphoid precursor neoplasm
Thyroid Gland Carcinoma          → Thyroid (Map)        Same as thyroid_carcinoma
No new canonical tissues created (Option B unused).
"""


def normalize_tissue_label(label) -> str:
    """
    Canonical lookup key for tissue labels.

    Handles capitalization, leading/trailing whitespace, repeated whitespace,
    spaces, underscores, and hyphens by collapsing separator runs to '_'.
    """
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return ''
    text = str(label).strip().lower()
    if text in ('', 'nan', 'none', 'null'):
        return ''
    # Possessive / apostrophes: "ewing's sarcoma" → "ewing sarcoma"
    text = re.sub(r"'s\b", '', text)
    text = text.replace("'", '')
    # Collapse any run of whitespace / underscore / hyphen to a single underscore
    text = re.sub(r'[\s_\-]+', '_', text)
    # Drop remaining punctuation that blocks matching
    text = re.sub(r'[^a-z0-9_/]+', '', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text


def _build_normalized_lookup(tissue_map: dict = None) -> dict:
    """
    Single-source lookup: normalize every map key once.
    If two raw key spellings normalize to the same key and disagree on target,
    keep the first and warn via assertion in tests — ontology must be consistent.
    """
    tissue_map = tissue_map if tissue_map is not None else TISSUE_HARMONIZATION_MAP
    lookup = {}
    for raw_key, target in tissue_map.items():
        norm = normalize_tissue_label(raw_key)
        if not norm:
            continue
        if norm in lookup and lookup[norm] != target:
            raise ValueError(
                f"Normalized key '{norm}' maps to both '{lookup[norm]}' and '{target}'"
            )
        lookup[norm] = target
    return lookup


def _build_normalized_exclude(exclude_labels=None) -> set:
    exclude_labels = exclude_labels if exclude_labels is not None else TISSUE_EXCLUDE_LABELS
    return {normalize_tissue_label(x) for x in exclude_labels if normalize_tissue_label(x)}


# Built once from TISSUE_HARMONIZATION_MAP (no duplicated scientific entries)
_NORMALIZED_LOOKUP = _build_normalized_lookup()
_NORMALIZED_EXCLUDE = _build_normalized_exclude()


def lookup_tissue(label, lookup: dict = None, exclude: set = None):
    """
    Look up a raw label.

    Returns
    -------
    mapped_label : str or None
    mapping_status : {'Mapped', 'Excluded', 'Fallback', 'Unmapped'}
    normalized_label : str
    """
    lookup = lookup if lookup is not None else _NORMALIZED_LOOKUP
    exclude = exclude if exclude is not None else _NORMALIZED_EXCLUDE
    norm = normalize_tissue_label(label)
    if not norm:
        return None, 'Unmapped', norm

    if norm in exclude:
        return None, 'Excluded', norm

    if norm in lookup:
        return lookup[norm], 'Mapped', norm

    # Title-case underscore fallback (should be unused after ontology completion)
    fallback = '_'.join(part.capitalize() for part in norm.split('_'))
    return fallback, 'Fallback', norm


def harmonize_tissue(series: pd.Series) -> pd.Series:
    """Map raw tissue names to harmonized ontology via normalized lookup.

    Excluded and unmapped labels become NaN (drop from causal analysis frames).
    """
    lookup = _NORMALIZED_LOOKUP
    exclude = _NORMALIZED_EXCLUDE

    def _map_one(val):
        mapped, status, _ = lookup_tissue(val, lookup=lookup, exclude=exclude)
        if status in ('Unmapped', 'Excluded'):
            return np.nan
        return mapped

    return series.map(_map_one)


def audit_tissue_mapping(raw_labels) -> pd.DataFrame:
    """
    Build audit table for unique raw tissue labels.

    Columns: raw_label, normalized_label, mapped_label, mapping_status
    """
    labels = pd.Series(raw_labels).dropna().astype(str).str.strip()
    labels = labels[~labels.str.lower().isin(['', 'nan', 'none', 'null'])]
    unique = sorted(labels.unique(), key=lambda x: x.lower())

    rows = []
    for raw in unique:
        mapped, status, norm = lookup_tissue(raw)
        rows.append({
            'raw_label': raw,
            'normalized_label': norm,
            'mapped_label': mapped if mapped is not None else '',
            'mapping_status': status,
        })
    return pd.DataFrame(rows)


def validate_tissue_coverage(df: pd.DataFrame, tissue_col: str = 'tissue_type') -> dict:
    total = len(df)
    unmapped = df[tissue_col].isna().sum()
    return {
        'total_records': total,
        'unmapped_records': int(unmapped),
        'coverage_pct': round((total - unmapped) / total * 100, 2) if total else 0.0,
        'unique_harmonized_tissues': int(df[tissue_col].nunique()),
        'tissue_distribution': df[tissue_col].value_counts().to_dict()
    }
