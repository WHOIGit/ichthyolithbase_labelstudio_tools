


def cleanup_df(df):
    print('Cleaning-up dataframe')
    # Define fields
    numerical_annot_fields = ['Z1',
       'A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1',
       'E1', 'E2', 'E3', 'E4', 'F1', 'G1', 'G2', 'G3', 'H1',
       'I1', 'I2', 'I3', 'J1', 'J2', 'K1', 'K2', 'L1', 'L2', 'L3',
       'L4', 'M1', 'M2', 'M3', 'M4', 'N1', 'N2', 'N3',
       'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10']
    extra_annot_fields = ['Type']
    cache_fields = ['cache_'+col for col in extra_annot_fields+numerical_annot_fields]
    data_fields = ['pid', 'site', 'slide', 'sediment', 'iodp', 'fraction',
                   'Slide Hole', 'Slide Hole Max', 'obj', 'magnification',
                   'Notes on Coding', 'Sample #', 'Object #',
                   'Depth', 'core', 'section', 'interval', 'image']
    export_fields = ['id','annotation_id','annotator','created_at','updated_at']

    ordered_fields = export_fields + data_fields + extra_annot_fields + numerical_annot_fields

    # Note excluded fields
    excluded_fields = sorted(set(df.columns) - set(ordered_fields))
    if excluded_fields:
        #print(f"Excluded fields: {excluded_fields}")
        ...

    # Cleaning up taxonomies
    for col in numerical_annot_fields+extra_annot_fields:
        if col in df.columns:
            df[col] = df[col].str.split('"').str[3] if isinstance(df[col].iloc[0], str) else df[col]

    # Create a copy of the dataframe with ordered fields
    df = df[ordered_fields].copy()

    # Add numerical matrix
    for col in numerical_annot_fields:
        df[col+'.1'] = df[col].str.split('.').str[1]

    return df