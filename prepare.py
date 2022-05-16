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
    
    col_drop = ['class', 'deck', 'Unnamed: 0', 'embark_town']
    titanic_df = titanic_df.drop(columns=col_drop)
    print(titanic_df)
    titanic_dummy = pd.get_dummies(titanic_df[['embarked', 'sex']], dummy_na=False, drop_first=[True, True])
    titanic_df = pd.concat([titanic_df, titanic_dummy], axis=1)

    return titanic_df

def acquire_prep_titanic():
    titanic_df = acquire.get_titanic_data()
    titanic_df = prep_titanic(titanic_df)

    return titanic_df

def prep_telco(telco_df):
    duplicates = ['contract_type_id', 'payment_type_id', 'internet_service_type_id', 'customer_id']
    categorical = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'internet_service_type', 'payment_type', 'contract_type']
    lst_true = [True] * len(categorical)
    blanks = telco_df['total_charges'] != ' '
    telco_df = telco_df[blanks]

    telco_df = telco_df.drop(columns=duplicates)
    dummies_df = pd.get_dummies(telco_df[categorical], dummy_na=False, drop_first=lst_true)
    dummies_df

    telco_df = pd.concat([telco_df, dummies_df], axis=1)
    return telco_df

def acquire_prep_telco():
    telco_df = acquire.get_telco_data()
    telco_df = prep_telco(telco_df)

    return telco_df

if __name__ == '__main__':
    print(acquire_prepare_iris().head())
    print(acquire_prep_titanic().head())
    print(acquire_prep_telco().head())
