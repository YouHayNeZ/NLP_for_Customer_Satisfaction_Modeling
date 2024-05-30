# Preprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Import data
def import_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Clean data
def clean_data(data):
    # Clean 'Trip_verified' column
    data['Trip_verified'] = data['Trip_verified'].replace({'NotVerified': 'Not Verified', 'Unverified': 'Not Verified'})

    # Drop 'Unnamed: 0' column
    data = data.drop(columns=['Unnamed: 0'])

    # Clean 'Origin' column (same cities with different airports are combined)
    data['Origin'] = data['Origin'].replace({
        'London Stansted': 'London', 'Stansted': 'London', 'LTN': 'London', 'Gatwick': 'London', 'LGW': 'London',
        'Heathrow': 'London', 'London Gatwick': 'London', 'London Luton': 'London', 'LHR': 'London', 'London Stamsted': 'London',
        'STN': 'London', 'London (Stansted)': 'London', 'London, Stansted': 'London', 'Stansted, Uk': 'London', 'Stanstred': 'London',
        'London Stanstead': 'London', 'London Southend': 'London', 'Southend': 'London', 'SEN': 'London',
        'Stanstead': 'London', 'Luton': 'London', 'Stansted (London)': 'London', 'London Stansted Airport': 'London',
        'Stansted via Aalborg': 'London', 'Stansted via Dublin': 'London', 'Stansted via Eindhoven': 'London', 'Stansted via Gdansk': 'London',
        'Paris Beauvais': 'Paris', 'Beauvais': 'Paris', 'BVA': 'Paris', 'Paris': 'Paris',
        'Ciampino Rome': 'Rome', 'Rome Ciampino': 'Rome', 'Rome-Ciampino': 'Rome', 'Rome Fiumicino': 'Rome', 'FCO': 'Rome',
        'from Rome (CIA)': 'Rome', 'CIA': 'Rome', 'Roma': 'Rome',
        'Frankfurt Hahn': 'Frankfurt', 'Frankfurt-Hahn': 'Frankfurt', 'Frankfurt Hanh': 'Frankfurt', 'HHN': 'Frankfurt', 'FMM': 'Frankfurt',
        'Hahn': 'Frankfurt', 'Düsseldorf': 'Dusseldorf', 'DUS': 'Dusseldorf',
        'Brussels Charleroi': 'Brussels', 'Charleroi': 'Brussels', 'Bruxelles': 'Brussels', 'Brussel': 'Brussels', 'CRL': 'Brussels',
        'Tel Aviv-Yafo': 'Tel Aviv', 'TLV': 'Tel Aviv', 'Tel-Aviv': 'Tel Aviv',
        'Milan Bergamo': 'Milan', 'Bergamo': 'Milan', 'BGY': 'Milan', 'Milano': 'Milan', 'Milano Malpensa': 'Milan',
        'Warsaw Modlin': 'Warsaw', 'Warszawa': 'Warsaw', 'OPO': 'Porto', 'LIS': 'Lisbon', 'Lisbon, Portugal': 'Lisbon', 'MAN': 'Manchester', 'CPH': 'Copenhagen',
        'DUB': 'Dublin', 'BUD': 'Budapest', 'Hungary': 'Budapest', 'Cologne Bonn': 'Cologne', 'Cologne-Bonn': 'Cologne', 'CGN': 'Cologne', 'Berlin TXL': 'Berlin', 'SXF': 'Berlin', 'AMS': 'Amsterdam', 'ATH': 'Athens', 'BCN': 'Barcelona', 'EDI': 'Edinburgh',
        'Tenerife South': 'Tenerife', 'TFS': 'Tenerife', 'Palma': 'Palma de Mallorca', 'Palma Majorca': 'Palma de Mallorca', 'Majorca': 'Palma de Mallorca', 'PMI': 'Palma de Mallorca',
        'Mallorca': 'Palma de Mallorca', 'FAO': 'Faro', 'Faro Portugal': 'Faro',
        'Edinbrough': 'Edinburgh', 'Leeds Bradford': 'Leeds', 'Leeds/Bradford': 'Leeds', 'Leeds bradford': 'Leeds',
        'Gran Carnia': 'Gran Canaria', 'Ponta DelGada': 'Ponta Delgada', 'PDL': 'Ponta Delgada',
        'Glasgow Prestwick': 'Glasgow', 'GLA': 'Glasgow', 'Prestwick': 'Glasgow',
        'Beziers, France': 'Beziers', 'Beauvais, France': 'Paris',
        'Memmingnem': 'Memmingen', 'Perugia/Pisa': 'Pisa', 'Bucharest Otopeni': 'Bucharest',
        'Nürnberg': 'Nuremberg', 'NUE': 'Nuremberg', 'Jasionka': 'Rzeszow',
        'Santiago': 'Santiago de Compostela', 'Ancona Falconara': 'Ancona',
        'Lamezia': 'Lamezia Terme', 'Bati': 'Batumi', 'Sandefjord, Torp': 'Oslo',
        'Landvetter': 'Gothenburg', 'GOT': 'Gothenburg', 'BTS': 'Bratislava', 'BRE': 'Bremen',
        'BFS': 'Belfast', 'RAK': 'Marrakech', 'ALC': 'Alicante', 'BRS': 'Bristol', 'EGC': 'Bergerac',
        'IBZ': 'Ibiza', 'ACE': 'Lanzarote', 'EMA': 'East Midlands', 'MAD': 'Madrid', 
        'BLQ': 'Bologna', 'MJV': 'Murcia', 'CRK': 'Cluj Napoca', 'PIK': 'Glasgow', 
        'ORK': 'Cork', 'SNN': 'Shannon', 'LPL': 'Liverpool', 'FUE': 'Fuerteventura',
        'SKG': 'Thessaloniki', 'PRG': 'Prague', 'BOH': 'Bournemouth', 'BRU': 'Brussels', 
        'LBA': 'Leeds', 'GRO': 'Barcelona', 'SVQ': 'Seville', 'BHX': 'Birmingham', 
        'TSF': 'Venice', 'FEZ': 'Fes', 'BLL': 'Billund', 'XRY': 'Jerez', 'LPA': 'Las Palmas',
        'NRN': 'Dusseldorf', 'Arrecife': 'Lanzarote', 'BSL': 'Basel', 'AGP': 'Malaga', 'VLG': 'Vilnius',
        'AC': 'Lanzarote', 'Skavsta': 'Stockholm', 'Weeze': 'Dusseldorf',
        'Krakow return': 'Krakow', 'AHO': 'Alghero', 'Bologna Italy': 'Bologna', 'BRE via DUB': 'Bremen',
        'Rzeszow, Jasionka': 'Rzeszow', 'Glasgow via Stansted': 'Glasgow', 'Gerona': 'Barcelona', 'Berlin-Schonefeld': 'Berlin',
        'ATH via BGY': 'Athens', 'GLA via DUB': 'Glasgow', 'EDI via LON': 'Edinburgh', 'CHQ': 'Chania', 'BUD via Stansted': 'Budapest',
        'Rriga': 'Riga', 'E.Midlands': 'East Midlands', 'Knock, Ireland': 'Knock', 'Lanzorote': 'Lanzarote', 'Barcelona El Prat': 'Barcelona',
        'POZ': 'Poznan', 'PSA': 'Pisa', 'Toulouse via Stansted': 'Toulouse', 'Bristol, UK': 'Bristol', 'Bristol, UK': 'Bristol',
        'Brindisi via Barcelona': 'Brindisi', 'Bristol via Dublin': 'Bristol', 'Bristol via Stansted': 'Bristol',
        'Stockholm-Skavsta': 'Stockholm', 'Stockholm Skavsta': 'Stockholm', 'Stockholm Arlanda': 'Stockholm',
        'Madrid via Berlin': 'Madrid', 'Madrid via Stansted': 'Madrid', 'Madrid via Dublin': 'Madrid', 'Madrid via London': 'Madrid',
        'Venice Treviso': 'Venice', 'Venice Marco Polo': 'Venice', 'Venice Marco Polo Airport': 'Venice',
        'Cluj-Napoca': 'Cluj Napoca', 'Cluj Napoca via Stansted': 'Cluj Napoca', 'Cluj Napoca via Dublin': 'Cluj Napoca',
        'Tirana via Stansted london': 'Tirana', 'Vasteras': 'Stockholm', 'Stockholm Vasteras': 'Stockholm', 'Funchal': 'Madeira',
        'Newcastle via Dublin': 'Newcastle', 'Bergamo via Bristol': 'Milan', 'Málaga': 'Malaga', 'Gdańsk': 'Gdansk',
        'Kaunas via Stansted': 'Kaunas', 'Fuerventura': 'Fuerteventura', 'Fuertavenura': 'Fuerteventura', 'Rīga': 'Riga',
        'Kraków': 'Krakow', 'Marrakech via Madrid': 'Marrakesh', 'Milan (Linate)': 'Milan', 'Bolonia via Mykonos': 'Bologna',
        'Wien': 'Vienna', 'Bergamo Italy': 'Milan', 'Brussels South': 'Brussels', 'Malta': 'Valletta', 'Milán': 'Milan',
        'Dubrosnik': 'Dubrovnik', 'Budapest via Stansted': 'Budapest', 'Gothenburg via Stansted': 'Gothenburg', 'Shannon via Gadwick': 'Shannon',
        'Cracow': 'Krakow', 'Tenerife south': 'Tenerife', 'Treviso': 'Venice', 'Sarajevo via Cologne': 'Sarajevo', 
        'Marrakech': 'Marrakesh', 'Fuertuventura': 'Fuerteventura', 'Milano Bergamo': 'Milan', 'Berlin Schönefeld': 'Berlin',
        'Cluj-Napoca via Luton': 'Cluj Napoca', 'Łódź': 'Lodz', 'Prague via Stansted': 'Prague', 'Nis via Weeze': 'Nis', 
        'Brindisi via Stansted': 'Brindisi', 'Barcelona Reus': 'Barcelona', 'Malta and return': 'Valletta', 'Sophia': 'Sofia',
        'Las Palmas': 'Palma de Mallorca', 'Wrowclaw': 'Wroclaw', 'Toulouse via London Stansted': 'Toulouse', 'Vilnius via Orio al Serio': 'Vilnius',
        'Sevilla': 'Seville', 'Oporto': 'Porto', 'Newcastle-upon-Tyne': 'Newcastle', 'Rome Ciampnio': 'Rome', 'Brussels CRL': 'Brussels', 
        'Santiago de Compustelo': 'Santiago de Compostela', 'Fuertenventura': 'Fuerteventura', 'Gasglow': 'Glasgow', 
        'VNO': 'Vilnius', 'MXP': 'Milan', 'NYO': 'Stockholm', 'TLL': 'Tallinn', 'SDR': 'Santander', 'VRN': 'Verona', 'FNI': 'Nimes',
        'FKB': 'Karlsruhe', 'KUN': 'Kaunas', 'JMK': 'Mykonos', 'OTP': 'Bucharest', 'MLA': 'Valletta', 'WRO': 'Wroclaw', 'LIG': 'Limoges',
        'PUY': 'Pula', 'HAM': 'Hamburg', 'WAW': 'Warsaw', 'PMO': 'Palermo', 'JTR': 'Santorini', 'Fes': 'Fes', 'TNG': 'Tangier',
        'RHO': 'Rhodes', 'ZAZ': 'Zaragoza', 'CHP': 'Copenhagen', 'Treviso': 'Venice', 'Girona': 'Barcelona', 'Chania': 'Crete', 'Tirana via Stansted London': 'Tirana',
        'Chania': 'Crete', 'Heraklion': 'Crete', 'Bergamo': 'Milan', 'Rodez': 'Rzeszow', 'Marrakech': 'Marrakesh', 'Crete (Chania)': 'Crete', 'Kyiv': 'Kiev',
        'Malta': 'Valletta'
    })

    # Clean 'Destination' column
    data['Destination'] = data['Destination'].replace({
        
        'London Stansted': 'London', 'Stansted': 'London', 'LTN': 'London', 'Gatwick': 'London', 'LGW': 'London',
        'Heathrow': 'London', 'London Gatwick': 'London', 'London Luton': 'London', 'LHR': 'London', 'London Stamsted': 'London',
        'STN': 'London', 'London (Stansted)': 'London', 'London, Stansted': 'London', 'Stansted, Uk': 'London', 'Stanstred': 'London',
        'London Stanstead': 'London', 'London Southend': 'London', 'Southend': 'London', 'SEN': 'London',
        'Stanstead': 'London', 'Luton': 'London', 'Stansted (London)': 'London', 'London Stansted Airport': 'London',
        'Stansted via Aalborg': 'London', 'Stansted via Dublin': 'London', 'Stansted via Eindhoven': 'London', 'Stansted via Gdansk': 'London',
        'Paris Beauvais': 'Paris', 'Beauvais': 'Paris', 'BVA': 'Paris', 'Paris': 'Paris',
        'Ciampino Rome': 'Rome', 'Rome Ciampino': 'Rome', 'Rome-Ciampino': 'Rome', 'Rome Fiumicino': 'Rome', 'FCO': 'Rome',
        'from Rome (CIA)': 'Rome', 'CIA': 'Rome', 'Roma': 'Rome',
        'Frankfurt Hahn': 'Frankfurt', 'Frankfurt-Hahn': 'Frankfurt', 'Frankfurt Hanh': 'Frankfurt', 'HHN': 'Frankfurt', 'FMM': 'Frankfurt',
        'Hahn': 'Frankfurt', 'Düsseldorf': 'Dusseldorf', 'DUS': 'Dusseldorf',
        'Brussels Charleroi': 'Brussels', 'Charleroi': 'Brussels', 'Bruxelles': 'Brussels', 'Brussel': 'Brussels', 'CRL': 'Brussels',
        'Tel Aviv-Yafo': 'Tel Aviv', 'TLV': 'Tel Aviv', 'Tel-Aviv': 'Tel Aviv',
        'Milan Bergamo': 'Milan', 'Bergamo': 'Milan', 'BGY': 'Milan', 'Milano': 'Milan', 'Milano Malpensa': 'Milan',
        'Warsaw Modlin': 'Warsaw', 'Warszawa': 'Warsaw', 'OPO': 'Porto', 'LIS': 'Lisbon', 'Lisbon, Portugal': 'Lisbon', 'MAN': 'Manchester', 'CPH': 'Copenhagen',
        'DUB': 'Dublin', 'BUD': 'Budapest', 'Hungary': 'Budapest', 'Cologne Bonn': 'Cologne', 'Cologne-Bonn': 'Cologne', 'CGN': 'Cologne', 'Berlin TXL': 'Berlin', 'SXF': 'Berlin', 'AMS': 'Amsterdam', 'ATH': 'Athens', 'BCN': 'Barcelona', 'EDI': 'Edinburgh',
        'Tenerife South': 'Tenerife', 'TFS': 'Tenerife', 'Palma': 'Palma de Mallorca', 'Palma Majorca': 'Palma de Mallorca', 'Majorca': 'Palma de Mallorca', 'PMI': 'Palma de Mallorca',
        'Mallorca': 'Palma de Mallorca', 'FAO': 'Faro', 'Faro Portugal': 'Faro',
        'Edinbrough': 'Edinburgh', 'Leeds Bradford': 'Leeds', 'Leeds/Bradford': 'Leeds', 'Leeds bradford': 'Leeds',
        'Gran Carnia': 'Gran Canaria', 'Ponta DelGada': 'Ponta Delgada', 'PDL': 'Ponta Delgada',
        'Glasgow Prestwick': 'Glasgow', 'GLA': 'Glasgow', 'Prestwick': 'Glasgow',
        'Beziers, France': 'Beziers', 'Beauvais, France': 'Paris',
        'Memmingnem': 'Memmingen', 'Perugia/Pisa': 'Pisa', 'Bucharest Otopeni': 'Bucharest',
        'Nürnberg': 'Nuremberg', 'NUE': 'Nuremberg', 'Jasionka': 'Rzeszow',
        'Santiago': 'Santiago de Compostela', 'Ancona Falconara': 'Ancona',
        'Lamezia': 'Lamezia Terme', 'Bati': 'Batumi', 'Sandefjord, Torp': 'Oslo',
        'Landvetter': 'Gothenburg', 'GOT': 'Gothenburg', 'BTS': 'Bratislava', 'BRE': 'Bremen',
        'BFS': 'Belfast', 'RAK': 'Marrakech', 'ALC': 'Alicante', 'BRS': 'Bristol', 'EGC': 'Bergerac',
        'IBZ': 'Ibiza', 'ACE': 'Lanzarote', 'EMA': 'East Midlands', 'MAD': 'Madrid', 
        'BLQ': 'Bologna', 'MJV': 'Murcia', 'CRK': 'Cluj Napoca', 'PIK': 'Glasgow', 
        'ORK': 'Cork', 'SNN': 'Shannon', 'LPL': 'Liverpool', 'FUE': 'Fuerteventura',
        'SKG': 'Thessaloniki', 'PRG': 'Prague', 'BOH': 'Bournemouth', 'BRU': 'Brussels', 
        'LBA': 'Leeds', 'GRO': 'Barcelona', 'SVQ': 'Seville', 'BHX': 'Birmingham', 
        'TSF': 'Venice', 'FEZ': 'Fes', 'BLL': 'Billund', 'XRY': 'Jerez', 'LPA': 'Las Palmas',
        'NRN': 'Dusseldorf', 'Arrecife': 'Lanzarote', 'BSL': 'Basel', 'AGP': 'Malaga', 'VLG': 'Vilnius',
        'AC': 'Lanzarote', 'Skavsta': 'Stockholm', 'Weeze': 'Dusseldorf',
        'Krakow return': 'Krakow', 'AHO': 'Alghero', 'Bologna Italy': 'Bologna', 'BRE via DUB': 'Bremen',
        'Rzeszow, Jasionka': 'Rzeszow', 'Glasgow via Stansted': 'Glasgow', 'Gerona': 'Barcelona', 'Berlin-Schonefeld': 'Berlin',
        'ATH via BGY': 'Athens', 'GLA via DUB': 'Glasgow', 'EDI via LON': 'Edinburgh', 'CHQ': 'Chania', 'BUD via Stansted': 'Budapest',
        'Rriga': 'Riga', 'E.Midlands': 'East Midlands', 'Knock, Ireland': 'Knock', 'Lanzorote': 'Lanzarote', 'Barcelona El Prat': 'Barcelona',
        'POZ': 'Poznan', 'PSA': 'Pisa', 'Toulouse via Stansted': 'Toulouse', 'Bristol, UK': 'Bristol', 'Bristol, UK': 'Bristol',
        'Brindisi via Barcelona': 'Brindisi', 'Bristol via Dublin': 'Bristol', 'Bristol via Stansted': 'Bristol',
        'Stockholm-Skavsta': 'Stockholm', 'Stockholm Skavsta': 'Stockholm', 'Stockholm Arlanda': 'Stockholm',
        'Madrid via Berlin': 'Madrid', 'Madrid via Stansted': 'Madrid', 'Madrid via Dublin': 'Madrid', 'Madrid via London': 'Madrid',
        'Venice Treviso': 'Venice', 'Venice Marco Polo': 'Venice', 'Venice Marco Polo Airport': 'Venice',
        'Cluj-Napoca': 'Cluj Napoca', 'Cluj Napoca via Stansted': 'Cluj Napoca', 'Cluj Napoca via Dublin': 'Cluj Napoca',
        'Tirana via Stansted london': 'Tirana', 'Vasteras': 'Stockholm', 'Stockholm Vasteras': 'Stockholm', 'Funchal': 'Madeira',
        'Newcastle via Dublin': 'Newcastle', 'Bergamo via Bristol': 'Milan', 'Málaga': 'Malaga', 'Gdańsk': 'Gdansk',
        'Kaunas via Stansted': 'Kaunas', 'Fuerventura': 'Fuerteventura', 'Fuertavenura': 'Fuerteventura', 'Rīga': 'Riga',
        'Kraków': 'Krakow', 'Marrakech via Madrid': 'Marrakesh', 'Milan (Linate)': 'Milan', 'Bolonia via Mykonos': 'Bologna',
        'Wien': 'Vienna', 'Bergamo Italy': 'Milan', 'Brussels South': 'Brussels', 'Malta': 'Valletta', 'Milán': 'Milan',
        'Dubrosnik': 'Dubrovnik', 'Budapest via Stansted': 'Budapest', 'Gothenburg via Stansted': 'Gothenburg', 'Shannon via Gadwick': 'Shannon',
        'Cracow': 'Krakow', 'Tenerife south': 'Tenerife', 'Treviso': 'Venice', 'Sarajevo via Cologne': 'Sarajevo', 
        'Marrakech': 'Marrakesh', 'Fuertuventura': 'Fuerteventura', 'Milano Bergamo': 'Milan', 'Berlin Schönefeld': 'Berlin',
        'Cluj-Napoca via Luton': 'Cluj Napoca', 'Łódź': 'Lodz', 'Prague via Stansted': 'Prague', 'Nis via Weeze': 'Nis', 
        'Brindisi via Stansted': 'Brindisi', 'Barcelona Reus': 'Barcelona', 'Malta and return': 'Valletta', 'Sophia': 'Sofia',
        'Las Palmas': 'Palma de Mallorca', 'Wrowclaw': 'Wroclaw', 'Toulouse via London Stansted': 'Toulouse', 'Vilnius via Orio al Serio': 'Vilnius',
        'Sevilla': 'Seville', 'Oporto': 'Porto', 'Newcastle-upon-Tyne': 'Newcastle', 'Rome Ciampnio': 'Rome', 'Brussels CRL': 'Brussels', 
        'Santiago de Compustelo': 'Santiago de Compostela', 'Fuertenventura': 'Fuerteventura', 'Gasglow': 'Glasgow', 
        'VNO': 'Vilnius', 'MXP': 'Milan', 'NYO': 'Stockholm', 'TLL': 'Tallinn', 'SDR': 'Santander', 'VRN': 'Verona', 'FNI': 'Nimes',
        'FKB': 'Karlsruhe', 'KUN': 'Kaunas', 'JMK': 'Mykonos', 'OTP': 'Bucharest', 'MLA': 'Valletta', 'WRO': 'Wroclaw', 'LIG': 'Limoges',
        'PUY': 'Pula', 'HAM': 'Hamburg', 'WAW': 'Warsaw', 'PMO': 'Palermo', 'JTR': 'Santorini', 'Fes': 'Fes', 'TNG': 'Tangier',
        'RHO': 'Rhodes', 'ZAZ': 'Zaragoza', 'CHP': 'Copenhagen', 'Treviso': 'Venice', 'Girona': 'Barcelona', 'Chania': 'Crete', 'Tirana via Stansted London': 'Tirana',
        'Chania': 'Crete', 'Heraklion': 'Crete', 'Bergamo': 'Milan', 'Rodez': 'Rzeszow', 'Marrakech': 'Marrakesh', 'Crete (Chania)': 'Crete', 'Kyiv': 'Kiev',
        'Malta': 'Valletta'
    })

    def clean_aircraft_name(name):
        if pd.isna(name):
            return np.nan
        
        name = name.lower()
        
        # Handling specific aircraft variations
        if '737' in name:
            if 'max' in name:
                return 'Boeing 737 MAX'
            elif '800' in name or '8' in name:
                return 'Boeing 737-800'
            elif '700' in name:
                return 'Boeing 737-700'
            elif '900' in name:
                return 'Boeing 737-900'
            elif '300' in name:
                return 'Boeing 737-300'
            elif '400' in name:
                return 'Boeing 737-400'
            else:
                return 'Boeing 737'
        
        # Other specific aircraft
        if '747' in name:
            return 'Boeing 747'
        if 'a320' in name:
            return 'Airbus A320'
        if 'a340' in name:
            return 'Airbus A340'
        if 'a319' in name:
            return 'Airbus A319'
        
        return name.title()
    data['Aircraft'] = data['Aircraft'].apply(clean_aircraft_name)

    # Clean 'Passenger Country' column
    data['Passenger Country'] = data['Passenger Country'].replace({
        'Steven Bouchere16th September 2013': np.nan
    })

    # Bin categories that make up less than 0.5% of observations
    for column in ['Aircraft', 'Origin', 'Destination', 'Passenger Country']:
        country_counts = data[column].value_counts().to_dict()
        for country, count in country_counts.items():
            if count < 5:
                data[column] = data[column].replace({country: 'Other'})

    return data

# Create datetime object & variables
def create_datetime(data):
    data['Date Published'] = pd.to_datetime(data['Date Published'])
    data['Date Flown'] = pd.to_datetime(data['Date Flown'])
    
    data['Year Published'] = data['Date Published'].dt.year
    data['Month Published'] = data['Date Published'].dt.month
    data['Day Published'] = data['Date Published'].dt.day
    
    data['Year Flown'] = data['Date Flown'].dt.year
    data['Month Flown'] = data['Date Flown'].dt.month
    data['Day Flown'] = data['Date Flown'].dt.day
    return data

# # Encode categorical variables
# def encode_categoricals(data):
#     # encode all columns except for Overall rating, Comment title, Comment
#     for col in data.columns:
#         if col not in ['Comment title', 'Comment', 'Overall Rating', 'Date Published', 'Date Flown', 'exclamation_marks', 'question_marks', 'comment_length']:
#             data = pd.get_dummies(data, columns=[col], prefix=col)
#     return data

# One-hot encoding of categorical variables
def one_hot_encode(X_train, X_val, X_test):
    # Save non-categorical columns
    non_categorical_columns = ['Comment title', 'Comment', 'Overall Rating', 'Date Published', 'Date Flown', 'exclamation_marks', 'question_marks', 'comment_length']
    X_train_non_categorical = X_train[non_categorical_columns]
    X_val_non_categorical = X_val[non_categorical_columns]
    X_test_non_categorical = X_test[non_categorical_columns]
    
    # Drop non-categorical columns
    X_train = X_train.drop(columns=non_categorical_columns)
    X_val = X_val.drop(columns=non_categorical_columns)
    X_test = X_test.drop(columns=non_categorical_columns)

    # One-hot encode categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train_encoded = encoder.fit_transform(X_train)
    X_val_encoded = encoder.transform(X_val)
    X_test_encoded = encoder.transform(X_test)

    # Combine one-hot encoded columns with non-categorical columns
    X_train_encoded = pd.DataFrame(X_train_encoded, index=X_train.index)
    X_val_encoded = pd.DataFrame(X_val_encoded, index=X_val.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, index=X_test.index)
    X_train = pd.concat([X_train_encoded, X_train_non_categorical], axis=1)
    X_val = pd.concat([X_val_encoded, X_val_non_categorical], axis=1)
    X_test = pd.concat([X_test_encoded, X_test_non_categorical], axis=1)
    
    return X_train, X_val, X_test

# Drop highly correlated column (see exploration.py) -> nothing to be dropped b/c all correlations < abs(0.8)

# Drop missing values in target 'Overall Rating'
def drop_missing_target(data):
    data = data.dropna(subset=['Overall Rating'])
    return data

# Normalization of continuous variables
def normalize_continuous(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train[['exclamation_marks', 'question_marks', 'comment_length']] = scaler.fit_transform(X_train[['exclamation_marks', 'question_marks', 'comment_length']])
    X_val[['exclamation_marks', 'question_marks', 'comment_length']] = scaler.transform(X_val[['exclamation_marks', 'question_marks', 'comment_length']])
    X_test[['exclamation_marks', 'question_marks', 'comment_length']] = scaler.transform(X_test[['exclamation_marks', 'question_marks', 'comment_length']])
    return X_train, X_val, X_test

# Create pipeline
def create_pipeline(file_path):
    data = import_data(file_path)
    data = clean_data(data)
    data = create_datetime(data)
    #data = encode_categoricals(data)
    data = drop_missing_target(data)

    # Splitting the dataset into the Training set, Validation set, and Test set using stratified sampling
    X = data.drop(columns=['Overall Rating'])
    y = data['Overall Rating']
    
    # Stratified split to maintain class balance
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Normalize continuous variables
    X_train, X_val, X_test = normalize_continuous(X_train, X_val, X_test)

    # One-hot encode categorical variables
    X_train, X_val, X_test = one_hot_encode(X_train, X_val, X_test)
    
    # Adjust the target labels to start from 0 instead of 1
    y_train = y_train.astype(int) - 1
    y_val = y_val.astype(int) - 1
    y_test = y_test.astype(int) - 1

    # Extract dates from train, validation, and test sets
    datetime_train = X_train[['Date Flown']]
    datetime_val = X_val[['Date Flown']]
    datetime_test = X_test[['Date Flown']]

    # Remove dates from train, validation, and test sets
    X_train = X_train.drop(columns=['Date Published', 'Date Flown'])
    X_val = X_val.drop(columns=['Date Published', 'Date Flown'])
    X_test = X_test.drop(columns=['Date Published', 'Date Flown'])

    # Remove 'Comment title' and 'Comment' columns
    X_train = X_train.drop(columns=['Comment title', 'Comment'])
    X_val = X_val.drop(columns=['Comment title', 'Comment'])
    X_test = X_test.drop(columns=['Comment title', 'Comment'])

    return X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data