import pandas as pd
import acquire

def prep_iris(iris_df):
    iris_data = iris_df
    col_drop = ['species_id', 'measurement_id']
    iris_data = iris_data.drop(columns=col_drop)
    

    iris_data = iris_data.rename(columns={'species_name': 'species'})
    print(iris_data.head())
    
    dummy_df = pd.get_dummies(iris_data['species'], dummy_na=False, drop_first=True)
    iris_data = pd.concat([iris_data, dummy_df], axis=1)
    
    return iris_data
    
def acquire_prepare_iris():
    iris_data = acquire.get_iris_data()
    iris_data = prep_iris(iris_data)
    return iris_data


def prep_titanic(titanic_df):
    
    titanic_dummy = pd.get_dummies(titanic_df[['embarked', 'sex', ]], dummy_na=False, drop_first=[True, True])
    titanic_df = pd.concat([titanic_df, titanic_dummy], axis=1)
    col_drop = ['class', 'deck', 'Unnamed: 0', 'embark_town', 'passenger_id','sex','embarked']
    titanic_df = titanic_df.drop(columns=col_drop)
    print(titanic_df)

    categories = []
    quant_cols = []
    for col in titanic_df.columns:
        if titanic_df[col].nunique() < 10:
            categories.append(col)
        else:
            quant_cols.append(col)
    print(quant_cols)
    return titanic_df, categories, quant_cols

def acquire_prep_titanic():
    titanic_df = acquire.get_titanic_data()
    titanic_df, categories, quant_cols = prep_titanic(titanic_df)

    return titanic_df, categories, quant_cols

def contains_yes_no(df):
    categories_to_map = []
    for col in df.columns:
        if 'Yes' in df[col].unique():
            if 'No' in df[col].unique():
                if len(df[col].unique()) <= 3:
                    categories_to_map.append(col)
                
    return categories_to_map

def map_yes_nos(df):
    categories_to_map = contains_yes_no(df)
    
    for col in categories_to_map:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    return df


def prep_telco(df):
    """
    Takes in Telco_Churn Dataframe.
    Arguments: drops unnecessary columns, converts categorical data.
    Returns cleaned data.
    """
    #drop unneeded columns
    df.drop(columns=['internet_service_type_id',
                 'payment_type_id', 'contract_type_id', 'customer_id'], inplace=True)
    #drop null values stored as whitespace:
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != ""]
    #convert to correct data type:
    df['total_charges'] = df.total_charges.astype(float)
    #Convert binary categorical to numeric
    df = map_yes_nos(df)
    df = df.rename(columns={'gender': 'is_female'})
    df.is_female = df.is_female.map({'Female': 1, 'Male': 0})
    #Turn NaNs to 'None' 
    #All NaNs result from lacking a service
    # ex: online_sec: None because they have no
    # internet service. 
    # ex2: multiple_lines: None because they have
    # no phone service.
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].astype('object')
            df[col] = df[col].fillna('None')
    
    #Turning all quantitative dtypes to float64
    #So I can loop and get them in a list
    df.tenure = df.tenure.astype('float64')

    quant_cols = []
    categories= []

    for col in df.columns:
        if len(df[col].unique()) < 5:
            categories.append(col)
        elif df[col].dtype == 'float64':
            quant_cols.append(col)

    
    #Get dummies for non-binary categorical variables:
    # dummy_df = pd.get_dummies(df[['contract_type', 'payment_type',
    #                           'internet_service_type']],
    #                           dummy_na = False,
    #                           drop_first=[True, True, True])
    #concatenate the two dataframes
    # df = pd.concat([df, dummy_df], axis=1)
    return df, categories, quant_cols

def object_columns_to_encode(train_df):
    object_type = []
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            object_type.append(col)

    return object_type

def encode_object_columns(train_df, drop_encoded=True):
    
    col_to_encode = object_columns_to_encode(train_df)
    dummy_df = pd.get_dummies(train_df[col_to_encode],
                              dummy_na=False,
                              drop_first=[True for col in col_to_encode])
    train_df = pd.concat([train_df, dummy_df], axis=1)
    train_df = train_df.drop(columns='Unnamed: 0')
    
    if drop_encoded:
        train_df = drop_encoded_columns(train_df, col_to_encode)

    return train_df

def drop_encoded_columns(train_df, col_to_encode):
    train_df = train_df.drop(columns=col_to_encode)
    return train_df

def acquire_prep_telco():
    telco_df = acquire.get_telco_data()
    telco_df = prep_telco(telco_df)

    return telco_df

if __name__ == '__main__':
    print(acquire_prepare_iris().head())
    print(acquire_prep_titanic()[0].head())
    print(acquire_prep_telco()[0].head())
