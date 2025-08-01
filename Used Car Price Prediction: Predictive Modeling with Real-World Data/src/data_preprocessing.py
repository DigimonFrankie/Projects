import numpy as np
import pandas as pd
import os
import re
import pickle
import logging
from collections import defaultdict, Counter

class DataPreprocessor:
    def __init__(self, df, assets_dir):
        self.df = df.copy()
        self.assets_dir = assets_dir

        ## set up logging
        log_path = os.path.join(self.assets_dir, 'data_preprocessing.log')

        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info("DataPreprocessor initialized. Logging to : %s", log_path)

    '''
    ---
    ### 🔧 Brand Name Corrections Needed

    As predicted, these three brands require their brand and model names to be fixed for consistency:

    - **Alfa** → _Alfa Romeo_
    - **Aston** → _Aston Martin_
    - **Land** → _Land Rover_
    '''
    def correct_brand_names(self):
        logging.info("START: correct_brand_names")

        ## Replace inconsistent brand names with full names
        self.df['brand'] = self.df['brand'].replace({
            'Alfa': 'Alfa Romeo',
            'Aston': 'Aston Martin',
            'Land': 'Land Rover'
        })

        ## Remove first word from model names for these brands
        brand_name = ['Alfa Romeo', 'Aston Martin', 'Land Rover']

        for brand in brand_name:

            self.df.loc[self.df['brand'] == brand, 'model'] = self.df.loc[self.df['brand'] == brand, 'model'].str.split().apply(lambda x: ' '.join(x[1:]) if isinstance(x, list) and len(x) > 1 else '')

        logging.info("END: correct_brand_names")
    
    '''
    Clean model names
    '''
    def clean_model_names(self):
        logging.info("START: clean_model_names")

        ## particularly for BMW models, clean up mutant model names with duplicated numeric badges
        def clean_bmw_model(model):
            """
            Cleans up mutant BMW model names with duplicated numeric badges.
            Examples:
                '330 330i xDrive'         -> '330 i xDrive'
                'M550 M550i xDrive'       -> 'M550 i xDrive'
                '440 Gran Coupe 440i xDrive' -> '440 Gran Coupe i xDrive'
                '428 Gran Coupe 428i xDrive SULEV' -> '428 Gran Coupe i xDrive SULEV'
            Only works if the second badge starts with the first (e.g., '330' in '330i').
            """
            tokens = str(model).split()
            if len(tokens) < 2:
                return model
            for i in range(1, len(tokens)):
                if tokens[i].startswith(tokens[0]) and tokens[i].endswith('i'):
                    suffix = tokens[i][len(tokens[0]):]  # Should just be 'i'
                    tokens = tokens[:i] + [suffix] + tokens[i+1:]
                    # Clean up empty tokens and extra spaces, just in case
                    return ' '.join([t for t in tokens if t])
            return model
        
        # Remove duplicate model names
        # For example, 'RX 350 RX 350' should be 'RX 350'
        self.df['model'] = self.df['model'].apply(lambda x: ' '.join(dict.fromkeys(x.split())))

        ## Apply the cleaning function to BMW models
        mask = self.df['brand'] == 'BMW'
        self.df.loc[mask, 'model'] = self.df.loc[mask, 'model'].apply(clean_bmw_model)

        logging.info("END: clean_model_names")

    '''
    ### Extract numbers columns
    '''
    def extract_numerical_columns(self):
        logging.info("START: extract_numerical_columns")

        col_names = ['milage', 'price']

        def clean_numeric_col(series):
            """
            Clean numeric columns by removing non-numeric characters and converting to float.
            """
            return (
                series.astype(str)  # Ensure the series is of string type
                .str.replace(r'[^\d.]', '', regex=True)  # Remove non-numeric characters except digits and "."
                .astype(float)  # Convert to float
            )

        # Apply the cleaning function to the specified columns
        for col in col_names:
            self.df[col] = clean_numeric_col(self.df[col])

        logging.info("END: extract_numerical_columns")

    '''
    #### ⛽ Fuel Type: Data Cleaning Needed
    '''
    def get_fuel_type(self):
        """
        Impute the fuel type based on the brand and model.
        """
        logging.info("START: get_fuel_type")

        ## Check if same models already have fuel_type
        ## Load it back
        map_path = os.path.join(self.assets_dir, 'fuel_type_map.pkl')
        
        try:
            with open(map_path, 'rb') as f:
                fuel_type_map = pickle.load(f)
        except FileNotFoundError:
            logging.warning("File not found, creating new fuel_type_map.")
            fuel_type_map = {}

        mask_electric = (self.df['brand'].isin(['Tesla', 'Lucid', 'Rivian']) & 
                (self.df['fuel_type'].isin(['–', 'not supported']) |
                self.df['fuel_type'].isna()))

        # Fill missing fuel_type for electric brands
        self.df.loc[mask_electric, 'fuel_type'] = 'Electric'
        
        mask_valid = (~self.df['fuel_type'].isin(['–', 'not supported'])) & (~self.df['fuel_type'].isna())
        fuel_type_map_new = (
            self.df[mask_valid]
            .groupby(['brand', 'model'])['fuel_type']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            .to_dict()
        )

        ## update the fuel_type_map with the current DataFrame
        fuel_type_map.update(fuel_type_map_new)

        ## Save the fuel_type_map    
        with open(map_path, 'wb') as f:
            pickle.dump(fuel_type_map, f)
            logging.info("Completed: Updated fuel_type_map saved to %s", map_path)

        ## Impute fuel_type

        def impute_fuel_type(row, fuel_type_map):
            """
            Impute fuel_type based on brand and model.

            Args:
                row (pd.Series): A row of the DataFrame.
                
            Returns:
                str: The imputed transmission speeds.
            """
            mask = (row['fuel_type'] in ['–', 'not supported']) | (pd.isna(row['fuel_type']))
            if mask:
                key = (row['brand'], row['model'])
                return fuel_type_map.get(key, row['fuel_type'])
            else:
                return row['fuel_type']
            
        ## Apply the imputation function to the DataFrame
        self.df['fuel_type'] = self.df.apply(lambda row: impute_fuel_type(row, fuel_type_map), axis=1)

        logging.info("END: get_fuel_type")

    '''
    #### ⚙️ Transmission Data: Standardization & Feature Extraction

    The `transmission` column contains various types and naming conventions for similar transmissions.  
    To ensure consistency and improve analysis, we’ll standardize these values and extract two separate features:
    - **Transmission Type** (e.g., Automatic, Manual, CVT)
    - **Number of Speeds** (e.g., 6, 8, Single-Speed)
    '''

    ## Identify and Standardize Transmission Types
    """
    This step will focus on standardizing the values with both speed and mannual or automatic transmission.
    Two new columns will be created:
    - **Transmission Type** (e.g., Automatic, Manual, CVT)
    - **Number of Speeds** (e.g., 6, 8, Single-Speed
    """
    def get_transmission_type(self):
        logging.info("START: get_transmission_type")

        def extract_transmission_type(val):
            """
            Extracts the transmission type from the given value.

            Args:
                val (str): The transmission value to extract from.

            Returns:
                str: The transmission type ('Automatic', 'Manual', 'CVT', or 'Other').
            """
            v = str(val).lower()
            if any(keyword in v for keyword in ['automatic', 'a/t', 'auto', 'dual-clutch',
                                                'steptronic', 'dct', 'pdk', 'at', 'dual shift mode', 
                                                'overdrive switch', 'single-speed']):
                return 'Automatic'
            if any(keyword in v for keyword in ['manual', 'm/t', 'mt']):
                return 'Manual'
            if any(keyword in v for keyword in ['cvt', 'variable']):
                return 'CVT'
            return 'Other'
        
        self.df['transmission_type'] = self.df['transmission'].apply(extract_transmission_type)

        logging.info("END: get_transmission_type")


    def get_transmission_speeds(self):
        logging.info("START: get_transmission_speeds")

        def extract_transmission_speeds(val):
            """
            Extracts the number of speeds from the given transmission value.

            Args:
                val (str): The transmission value to extract from.

            Returns:
                int: The number of speeds (e.g., 6), or 1 for single-speed, or None if not applicable.
            """
            v = str(val).lower()
            match = re.search(r'(\d+)[-\s]?(?:speed|spd)', v)
            if match:
                return int(match.group(1))
            if 'single-speed' in v or 'single speed' in v:
                return 1
            numbers = re.search(r'(\d+)', v)
            if numbers:
                return int(numbers.group(1))
            return None
        
        self.df['transmission_speeds'] = self.df['transmission'].apply(extract_transmission_speeds)

        logging.info("END: get_transmission_speeds")
    """
    💡 **Why Keep “CVT” as Its Own Category?**

    1. **Mechanically Different**  
    - **Traditional Automatic:** Uses a set of gears, shifts through them automatically.  
    - **CVT:** No gears—uses pulleys and belts for infinite gear ratios.

    2. **User Experience is Different**  
    - CVTs drive differently. No gear shifts, “rubber band” feel.  
    - Impacts consumer reviews, performance, and pricing.

    3. **Manufacturer & Industry Reporting**  
    - Specs, consumer guides, and data vendors *always* break out “CVT” separately from “Automatic”.  
    - Lumping them together can hide meaningful patterns (pricing, reliability, satisfaction, etc).

    4. **Modeling & Analytics**  
    - Some buyers specifically want to avoid (or seek out) CVTs.  
    - Resale value, repair costs, and reliability trends can differ significantly.  
    - **Keeping “CVT” as its own class maintains transparency and analytic flexibility.**
    """


    ## Impute transmission_type
    def impute_transmission(self):
        logging.info("START: impute_transmission")

        def impute_transmission_type(row, transmission_type_map):
            """
            Impute the transmission type based on the brand and model.
            
            Args:
                row (pd.Series): A row of the DataFrame.
                
            Returns:
                str: The imputed transmission type.
            """
            if row['transmission_type'] == 'Other':
                key = (row['model_year'], row['brand'], row['model'])
                return transmission_type_map.get(key, row['transmission_type'])
            return row['transmission_type']

        ## Fix transmission speeds
        def impute_transmission_speeds(row, transmission_speeds_map):
            """
            Impute the transmission speeds based on the brand and model.
            
            Args:
                row (pd.Series): A row of the DataFrame.
                
            Returns:
                int or None: The imputed transmission speeds.
            """
            if pd.isna(row['transmission_speeds']):
                key = (row['model_year'], row['brand'], row['model'])
                return transmission_speeds_map.get(key, row['transmission_speeds'])
            else:
                return row['transmission_speeds']
            
        ## Check if same models already have transmission types
        map_path = os.path.join(self.assets_dir, 'transmission_type_map.pkl')
        try:
            with open(map_path, 'rb') as f:
                transmission_type_map = pickle.load(f)
        except FileNotFoundError:
            logging.warning("File not found, creating new transmission_type_map.")
            transmission_type_map = {}
        
        transmission_type_map_new = (
            self.df[self.df['transmission_type'] != 'Other']
            .groupby(['model_year', 'brand', 'model'])['transmission_type']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Other')
            .to_dict()
        )
        ## Update the transmission_type_map with the current DataFrame
        transmission_type_map.update(transmission_type_map_new)

        ## Save the transmission_type_map
        with open(map_path, 'wb') as f:
            pickle.dump(transmission_type_map, f)

        ## Check if same models already have transmission speeds
        map_path = os.path.join(self.assets_dir, 'transmission_speeds_map.pkl')

        try:
            with open(map_path, 'rb') as f:
                transmission_speeds_map = pickle.load(f)
        except FileNotFoundError:
            logging.warning("File not found, creating new transmission_speeds_map.")
            transmission_speeds_map = {}
        
        transmission_speeds_map_new = (
            self.df[self.df['transmission_speeds'].notna()]
            .groupby(['model_year', 'brand', 'model'])['transmission_speeds']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            .to_dict()
        )

        ## Update the transmission_speeds_map with the current DataFrame
        transmission_speeds_map.update(transmission_speeds_map_new)

        ## Save the transmission_speeds_map
        with open(map_path, 'wb') as f:
            pickle.dump(transmission_speeds_map, f)
        
        # Apply the imputation function
        self.df['transmission_type'] = self.df.apply(lambda row: impute_transmission_type(row, transmission_type_map), axis=1)
        # Apply the imputation function
        self.df['transmission_speeds'] = self.df.apply(lambda row: impute_transmission_speeds(row, transmission_speeds_map), axis=1)

        logging.info("END: impute_transmission")

    '''
    ### Engine Data
    Engine data has rich info. it can be extract to various features. e.g. hoursepower (HP), Liter, Cylinder
    '''
    def extract_engine_data(self):
        logging.info("START: extract_engine_data")

        def engine_extract(row):
            """
            Extracts engine features from the 'engine' column.
            
            Args:
                row (pd.Series): A row of the DataFrame.
                
            Returns:
                dict: A dictionary with extracted engine features.
            """
            ## Handle Electric Vehicles
            if row['fuel_type'] in ['Electric', 'Hydrogen']:
                match = re.search(r'(\d+\.?\d*)\s*hp', row['engine'], re.IGNORECASE)
                return {
                    'hp': float(match.group(1)) if match else None,
                    'liters': 0.0,
                    'cylinders': 0.0
                }

            match = re.search(
                r'(?:([\d.]+)\s*hp\s+)?([\d.]+)?\s*(?:l|liter)?\s*(?:v|i|flat|straight|rotary)?\s*(\d+)?(?:\s*cylinder)?',
                row['engine'],
                re.IGNORECASE
            )

            if match:
                hp = float(match.group(1)) if match.group(1) else None
                liters = float(match.group(2)) if match.group(2) else None
                cylinders = float(match.group(3)) if match.group(3) else None
                
                ## Double check if the parse value is correct
                if liters is not None and not (0<= liters <= 10):
                    liters = None
                
                if cylinders is not None and not (0 <= cylinders <= 16):
                    cylinders = None

                return {'hp': hp, 'liters': liters, 'cylinders': cylinders}

            return {'hp': None, 'liters': None, 'cylinders': None}
        
        ## Apply engine extraction to the DataFrame
        self.df[['hp', 'liters', 'cylinders']] = self.df.apply(engine_extract, axis=1, result_type='expand')

        logging.info("END: extract_engine_data")

    """
    #### 🛠️ Engine Data: Imputation & Standardization

    Engine specs (`hp`, `liters`, `cylinders`) can be missing or inconsistent across similar vehicles.
    To ensure data completeness and accuracy, I’ll impute missing values using a prioritized approach:

    1. **Build Engine Spec Mapping**  
    - Create a dictionary mapping of available engine specs for each unique combo of model, year, brand, liters, and cylinders.

    2. **Identify Models With All Engine Specs Missing**  
    - Find vehicles with all three specs (`hp`, `liters`, `cylinders`) missing.
    - Impute these by looking up the records (by brand/model/year/engine config).

    3. **Impute Partially Missing Specs**  
    - For vehicles with one or two missing engine specs:
        - Impute **liters** first, then **cylinders**, then **hp**, following the order of most complete -> least complete.
        - For each missing value, search for matching vehicles (brand, model, liters, cylinders) using the closest current/prior model year.
        - If multiple candidates exist, use the most frequent value (mode). If still tied, use the highest value.

    4. **Update the Engine Map After Each Imputation**  
    - If a row is completed with all specs, add it to the engine spec mapping for future reference.

    **Example:**
    > If a 2019 Ford F-150 XLT has 3.5L, 6 cylinders, but missing `hp`, look up the most similar prior records.  
    > If find `325` and `375` as options and both are equally frequent, impute with `375`.

    ##### 🛠️ Create Engine Spec Maps

    To efficiently impute missing engine specs (horsepower, liters, cylinders), I build a mapping structure that summarizes all known engine configurations for each (model_year, brand, model) combo in the dataset.

    The resulting map is a nested dictionary that looks like this:

    ```
    {
    (2019, 'Ford', 'F-150 XLT'): {
        (325.0, 2.7, 6.0): 1,
        (335.0, 2.7, 6.0): 2,
        (375.0, 3.5, 6.0): 3,
        (395.0, 5.0, 8.0): 4
    },
    (2021, 'BMW', 'X5 xDrive40i'): {
        (335.0, 3.0, 6.0): 5
    },
    ...
    }
    ```
    """
    def impute_engine_specs(self):
        """
        Impute missing engine specs (hp, liters, cylinders) using a mapping of known engine configurations.
        """
        logging.info("START: impute_engine_specs")

        ## Function to impute all 3 null engine specs
        def impute_full_null_specs(row, engine_specs):
            """
            Impute missing engine specs (hp, liters, cylinders) when ALL THREE are missing.
            Looks for the most common engine spec for the (model_year, brand, model).
            If there’s a tie, picks the biggest (hp > liters > cylinders).
            If not found for this year, checks +/- 1, 2, 3 years out.
            """
            key = (row['model_year'], row['brand'], row['model'])
            # Grab all engine spec combos/counts for this exact year-brand-model
            spec_cnt = engine_specs.get(key, {})

            # If nothing for this year-brand-model, search ±3 years
            if not spec_cnt:
                for delta in range(1, 4):
                    for shift in [row['model_year']-delta, row['model_year']+delta]:
                        alt_key = (shift, row['brand'], row['model'])
                        spec_cnt = engine_specs.get(alt_key, {})
                        if spec_cnt:
                            break  # Stop looking further if you found any
                    if spec_cnt:
                        break

            if spec_cnt:  # Got some data, time to pick the winner
                max_cnt = max(spec_cnt.values())  # Highest count (mode)
                # Grab all engine specs with that highest count
                mode_specs = [spec for spec, cnt in spec_cnt.items() if cnt == max_cnt]
                # If tie, pick biggest hp, then liters, then cylinders
                best_spec = max(mode_specs, key=lambda x: (x[0], x[1], x[2]))
                return {'hp': best_spec[0], 'liters': best_spec[1], 'cylinders': best_spec[2]}

            # Fallback: couldn't impute, return all None
            return {'hp': None, 'liters': None, 'cylinders': None}


        ## Function to impute partial null engine specs
        def impute_partial_null_specs(row, engine_specs):
            """
            Impute engine specs when one or two are missing.
            Looks for the most common spec matching the other columns.
            Prioritizes same MY first, then checks ±3 years.
            Mode wins; if tie, picks biggest value.
            """
            key = (row['model_year'], row['brand'], row['model'])
            spec_cnt = engine_specs.get(key, {})

            # Put current values into a list so we can index it
            target = [row['hp'], row['liters'], row['cylinders']]
            nulls = [idx for idx, v in enumerate(target) if pd.isna(v)]  # Indices of missing values
            filled = target.copy()  # Start with what we have

            # If nothing for this year, search ±3 years
            if not spec_cnt:
                for delta in range(1, 4):
                    for shift in [row['model_year']-delta, row['model_year']+delta]:
                        alt_key = (shift, row['brand'], row['model'])
                        spec_cnt = engine_specs.get(alt_key, {})
                        if spec_cnt:
                            break
                    if spec_cnt:
                        break

            # If we found possible specs and not all three are missing
            if spec_cnt and len(nulls) < 3:
                for idx in nulls:  # For each missing spec
                    # Build up all (value, count) that match the KNOWN specs
                    specs = []
                    for spec, cnt in spec_cnt.items():
                        match = True
                        for j in range(3):
                            if j != idx and pd.notna(target[j]) and spec[j] != target[j]:
                                match = False
                                break
                        if match:
                            specs.append((spec[idx], cnt))
                    if specs:
                        # mode or max if tie
                        max_cnt = max(cnt for val, cnt in specs)
                        # Grab all values with highest count, pick the biggest one
                        filled[idx] = max([val for val, cnt in specs if cnt == max_cnt])

                return {'hp': filled[0], 'liters': filled[1], 'cylinders': filled[2]}

            # Return the original if nothing is found
            return {'hp': row['hp'], 'liters': row['liters'], 'cylinders': row['cylinders']}
        
        def assign_engine_specs(row, engine_specs):
            if row[['hp', 'liters', 'cylinders']].isna().sum() == 3:
                return impute_full_null_specs(row, engine_specs)
            elif row[['hp', 'liters', 'cylinders']].isna().sum() in [1,2]:
                return impute_partial_null_specs(row, engine_specs)
            else:
                return {'hp': row['hp'], 'liters': row['liters'], 'cylinders': row['cylinders']}
            
        
        ## Check if same models already have transmission speeds
        map_path = os.path.join(self.assets_dir, 'engine_specs.pkl')

        try:
            with open(map_path, 'rb') as f:
                engine_specs = pickle.load(f)
        except FileNotFoundError:
            logging.warning("File not found, creating new engine_specs.")
            engine_specs = {}
            
        engine_specs_new = defaultdict(Counter)

        for row in self.df[['model_year', 'brand', 'model', 'hp', 'liters', 'cylinders']].itertuples(index=False):
            model_year, brand, model, hp, liters, cylinders = row
            if pd.notna(hp) and pd.notna(liters) and pd.notna(cylinders):
                key = (model_year, brand, model)
                spec = (hp, liters, cylinders)
                engine_specs[key][spec] += 1
        
        engine_specs.update(engine_specs_new)

        ## Save the engine_specs
        with open(map_path, 'wb') as f:
            pickle.dump(engine_specs, f)

        self.df[['hp', 'liters', 'cylinders']] = self.df.apply(lambda row: assign_engine_specs(row, engine_specs), axis=1, result_type='expand')

        logging.info("END: impute_engine_specs")

    """
    ## 🚗 Accident History Imputation

    There are three types of accident values in the dataset:
    - **'None reported'** - will be labeled as 0.
    - **'At least 1 accident or damage reported'** - will be labeled as 1.
    - **NaN (missing value)** - will be considered as 'had accident' and be labeled as 1.

    For modeling purposes, *missing values (`NaN`)* will be **imputed as 'At least 1 accident or damage reported'**.  

    This is because accident history is one of the most critical factors in used car evaluation. If a seller (or previous owner) leaves this field blank, it raises a major red flag—potentially indicating an attempt to hide accident history.

    Treating missing data as “damage reported” makes the model more aggressive and risk-averse, which ultimately protects both buyers and sellers from misrepresentation or hidden issues.  

    Leaving this field blank could easily mislead the model and users. If you’re not willing to admit “none,” we assume the worst.
    """
    def impute_accident_history(self):
        logging.info("START: impute_accident_history")

        self.df['accident'] = self.df['accident'].fillna('At least 1 accident or damage reported')

        ## Convert 'accident' column to categorical type
        self.df['accident_reported'] = self.df['accident'].map({
            'None reported': 0,
            'At least 1 accident or damage reported': 1
        })

        logging.info("END: impute_accident_history")

    """
    ## 🚗 Clean_title Imputation

    Just like with the `Accident` column, there are only two possible values for clean title in this dataset:
    - **'Yes'** - will be labeled as 1
    - **NaN** (missing) - will be labeled as 0

    For modeling, every *missing value (`NaN`)* will be **imputed as 'No'**.  

    Why? Because clean title status is one of the most critical damn factors for evaluating a used car. If someone leaves this blank, they’re probably trying to hide something, and that is a massive red flag. 

    Imputing missing values as “No” (not a clean title) forces the model to be more aggressive and risk-averse—just the way it should be when there’s potential for someone to cover up damage or dirty history. This approach protects both the buyer and the seller from shady surprises.

    Bottom line: If you can't claim that it’s a clean title, we’re gonna assume the worst.  This way, we avoid misleading the model and users. If you’re not willing to say “yes,” we’re gonna treat it as “no.”
    """
    def impute_clean_title(self):
        logging.info("START: impute_clean_title")

        self.df['clean_title'] = self.df['clean_title'].fillna('No')

        self.df['clean_title'] = self.df['clean_title'].map({
                'Yes': 1,
                'No': 0
                })

        logging.info("END: impute_clean_title")

    """
    ## Checking for Missing Values
    After all preprocessing steps, we should check for any remaining missing values in the DataFrame.
    This will help ensure that the data is clean and ready for modeling.
    """
    def check_missing_values(self):
        """
        Check for missing values in the DataFrame.
        
        Returns:
            pd.Series: A Series with the count of missing values for each column.
        """
        logging.info("START: check_missing_values")

        missing = self.df.isnull().sum()
        total_missing = missing[missing > 0]
        
        if len(total_missing) > 0:
            logging.warning("Missing values found:\n%s", total_missing)
            print("Missing values found:\n", total_missing)
        else:
            logging.info("No missing values found.")
            print("No missing values found.")

        logging.info("END: check_missing_values")


    def preprocess(self):
        """
        Run all preprocessing steps in order.
        """
        logging.info("START: preprocess pipeline")

        self.correct_brand_names()
        self.clean_model_names()
        self.extract_numerical_columns()
        self.get_fuel_type()
        self.get_transmission_type()
        self.get_transmission_speeds()
        self.impute_transmission()
        self.extract_engine_data()
        self.impute_engine_specs()
        self.impute_accident_history()
        self.impute_clean_title()
        self.check_missing_values()

        logging.info("END: preprocess pipeline")
        print("Data preprocessing completed successfully.")

        return self.df.reset_index(drop=True)
    