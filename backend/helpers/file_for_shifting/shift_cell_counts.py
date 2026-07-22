import pandas as pd

def shift_cell_counts(input_filename, new_starting_log=8.0):
    """
    Reads a dataset, applies a new starting cell count, and recalculates 
    the final cell count to maintain the exact same log reduction difference
    using a for loop. Clamps the final cell count at 0.9.
    Returns the modified DataFrame.
    """
    print(f"Processing data from {input_filename} for starting count {new_starting_log}...")
    df = pd.read_csv(input_filename)
    
    new_startings = []
    new_finals = []
    
    # Iterate through each row using a for loop
    for index, row in df.iterrows():
        starting = row['Starting Cell Count(Log CFU/g)']
        final = row['Cell Count(Log CFU/g)']
        
        original_reduction = starting - final
        new_final = round(new_starting_log - original_reduction, 2)
        
        # Enforce the 0.9 limit of detection 
        if  new_final < 0.9:
            new_final = 0.9
            
        new_startings.append(new_starting_log)
        new_finals.append(new_final)
        
    # Assign the newly calculated lists back to the dataframe
    df['Starting Cell Count(Log CFU/g)'] = new_startings
    df['Cell Count(Log CFU/g)'] = new_finals
    
    return df

if __name__ == "__main__":
    # Define a list of multiple starting cell counts you want to generate
    combined_data = []
    
    for count in range(2,10):
        # Collect the dataframe for each starting count
        shifted_df = shift_cell_counts(
            input_filename='Texas_A&M_internship_Salmonilla_S_modeling.csv',
            new_starting_log=count
        )
        combined_data.append(shifted_df)
        
    # Combine all the dataframes into one and save to a single CSV
    final_df = pd.concat(combined_data, ignore_index=True)
    final_output = 'adjusted_Texas_A&M_internship_Salmonilla_S_modeling.csv'
    final_df.to_csv(final_output, index=False)
    
    print(f"Success! All adjusted datasets combined and saved to {final_output}\n")