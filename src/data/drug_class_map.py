"""Drug-to-class mapping."""

import pandas as pd

DRUG_CLASS_MAP = {
    'Erlotinib': 'EGFR_Inhibitor',
    'Gefitinib': 'EGFR_Inhibitor',
    'Afatinib': 'EGFR_Inhibitor',
    'Lapatinib': 'EGFR_HER2_Inhibitor',
    'AZD3759': 'EGFR_Inhibitor',
    'Osimertinib': 'EGFR_T790M_Inhibitor',
    'Alpelisib': 'PI3K_Inhibitor',
    'Buparlisib': 'PI3K_Inhibitor',
    'Taselisib': 'PI3K_Inhibitor',
    'MK-2206': 'AKT_Inhibitor',
    'Ipatasertib': 'AKT_Inhibitor',
    'Everolimus': 'mTOR_Inhibitor',
    'Sirolimus': 'mTOR_Inhibitor',
    'Temsirolimus': 'mTOR_Inhibitor',
    'Trametinib': 'MEK_Inhibitor',
    'Cobimetinib': 'MEK_Inhibitor',
    'Selumetinib': 'MEK_Inhibitor',
    'Dabrafenib': 'BRAF_Inhibitor',
    'Vemurafenib': 'BRAF_Inhibitor',
    'Encorafenib': 'BRAF_Inhibitor',
    'Olaparib': 'PARP_Inhibitor',
    'Rucaparib': 'PARP_Inhibitor',
    'Niraparib': 'PARP_Inhibitor',
    'Talazoparib': 'PARP_Inhibitor',
    'AZD6738': 'ATR_Inhibitor',
    'Berzosertib': 'ATR_Inhibitor',
    'Prexasertib': 'CHK1_Inhibitor',
    'Navitoclax': 'BCL2_Inhibitor',
    'Venetoclax': 'BCL2_Inhibitor',
    'ABT-737': 'BCL2_Inhibitor',
    'Cisplatin': 'Platinum_Based',
    'Carboplatin': 'Platinum_Based',
    'Oxaliplatin': 'Platinum_Based',
    'Paclitaxel': 'Taxane',
    'Docetaxel': 'Taxane',
    'Crizotinib': 'ALK_ROS1_Inhibitor',
    'Alectinib': 'ALK_Inhibitor',
    'Ceritinib': 'ALK_Inhibitor',
    'Lorlatinib': 'ALK_Inhibitor',
    'Palbociclib': 'CDK4_6_Inhibitor',
    'Ribociclib': 'CDK4_6_Inhibitor',
    'Abemaciclib': 'CDK4_6_Inhibitor',
    'Belinostat': 'HDAC_Inhibitor',
    'Panobinostat': 'HDAC_Inhibitor',
    'Vorinostat': 'HDAC_Inhibitor',
    'Linsitinib': 'IGF1R_Inhibitor',
    'BMS-754807': 'IGF1R_Inhibitor',
    'Ruxolitinib': 'JAK_Inhibitor',
    'Tofacitinib': 'JAK_Inhibitor',
    'Erdafitinib': 'FGFR_Inhibitor',
    'Pemigatinib': 'FGFR_Inhibitor',
    'Sorafenib': 'Multi_Kinase',
    'Sunitinib': 'Multi_Kinase',
    'Regorafenib': 'Multi_Kinase',
    'Pazopanib': 'Multi_Kinase',
    'Axitinib': 'Multi_Kinase',
}


def map_drug_to_class(drug_name_series: pd.Series) -> pd.Series:
    cleaned = drug_name_series.astype(str).str.strip()
    mapped = cleaned.map(DRUG_CLASS_MAP)
    unmapped = mapped.isna()
    if unmapped.any():
        lower_map = {k.lower(): v for k, v in DRUG_CLASS_MAP.items()}
        lower_mapped = cleaned[unmapped].str.lower().map(lower_map)
        mapped.loc[unmapped] = lower_mapped
    return mapped.fillna('Other')


def get_drug_class_stats(df: pd.DataFrame, class_col: str = 'drug_class') -> pd.DataFrame:
    stats = df.groupby(class_col).size().reset_index(name='n_records')
    return stats.sort_values('n_records', ascending=False)