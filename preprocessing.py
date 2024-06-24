# Preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier


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

    #### BINNING IS TURNED ON!!! ####

    # Bin categories that make up less than 0.5% of observations
    for column in ['Aircraft', 'Origin', 'Destination', 'Passenger Country']:
        country_counts = data[column].value_counts().to_dict()
        for country, count in country_counts.items():
            if count < 5:
                data[column] = data[column].replace({country: 'Other'})

    # Drop columns that are not used
    data = data.drop(columns=['Comment', 'Comment title'])

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

    data = data.drop(columns=['Date Flown'])

    return data

# Drop missing values in target 'Overall Rating'
def drop_missing_target(data):
    data = data.dropna(subset=['Overall Rating'])
    return data

def splitting_data(data):
    # Splitting the dataset into the Training set, Validation set, and Test set using stratified sampling
    X = data.drop(columns=['Overall Rating'])
    y = data['Overall Rating']
    
    # Stratified split to maintain class balance
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Scale of continuous variables
def scale_continuous(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    continuous_vars = ['exclamation_marks', 'question_marks', 'comment_length']
    X_train[continuous_vars] = scaler.fit_transform(X_train[continuous_vars])
    X_val[continuous_vars] = scaler.transform(X_val[continuous_vars])
    X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])
    return X_train, X_val, X_test

# Impute missing values
def impute_often_missing_values(X_train, X_val, X_test):
    
    # Dataframes to store imputed values
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()
    X_test_imputed = X_test.copy()

    # High missing value rate > 0.30: encode N/A as a new category '-1'
    high_missing_columns_str = ['Aircraft', 'Trip_verified']
    high_missing_columns_int = ['Inflight Entertainment', 'Wifi & Connectivity', 'Food & Beverages']
    for col in high_missing_columns_str:
        X_train_imputed[col] = X_train[col].fillna('Missing')
        X_val_imputed[col] = X_val[col].fillna('Missing')
        X_test_imputed[col] = X_test[col].fillna('Missing')
    for col in high_missing_columns_int:
        X_train_imputed[col] = X_train[col].fillna(-1)
        X_val_imputed[col] = X_val[col].fillna(-1)
        X_test_imputed[col] = X_test[col].fillna(-1)

    return X_train_imputed, X_val_imputed, X_test_imputed

# Impute missing values
def impute_missing_values_with_knn(X_train, X_val, X_test):
    
    # Dataframes to store imputed values
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()
    X_test_imputed = X_test.copy()
  
    # Moderate missing value rate < 0.30 and > 0.05: KNN Classifier
    knn_impute_columns = ['Year Flown', 'Month Flown', 'Day Flown', 'Destination', 'Origin', 'Ground Service', 'Cabin Staff Service', 'Type Of Traveller']
    
    # Extract 'Date Published' column
    train_date_published = X_train['Date Published']
    val_date_published = X_val['Date Published']
    test_date_published = X_test['Date Published']

    # Drop 'Date Published' column
    X_train = X_train.drop(columns=['Date Published'])
    X_val = X_val.drop(columns=['Date Published'])
    X_test = X_test.drop(columns=['Date Published'])

    for col in knn_impute_columns:
        train_target = X_train.copy()[col]
        val_target = X_val.copy()[col]
        test_target = X_test.copy()[col]

        # Label encode target column
        le = LabelEncoder()
        train_target = le.fit_transform(train_target)

        # Create dictionary to map target values to original values
        target_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        val_target = val_target.map(target_dict)
        test_target = test_target.map(target_dict)

        # Drop target column from training, validation, and test sets
        X_train_rest = X_train.copy().drop(columns=[col])
        X_val_rest = X_val.copy().drop(columns=[col])
        X_test_rest = X_test.copy().drop(columns=[col])

        # Save non-categorical columns
        non_categorical_columns = ['exclamation_marks', 'question_marks', 'comment_length']
        X_train_non_categorical = X_train_rest[non_categorical_columns]
        X_val_non_categorical = X_val_rest[non_categorical_columns]
        X_test_non_categorical = X_test_rest[non_categorical_columns]

        # Drop non-categorical columns
        X_train_rest = X_train_rest.drop(columns=non_categorical_columns)
        X_val_rest = X_val_rest.drop(columns=non_categorical_columns)
        X_test_rest = X_test_rest.drop(columns=non_categorical_columns)

        # Encode categorical features
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_train_rest_encoded = encoder.fit_transform(X_train_rest)
        X_val_rest_encoded = encoder.transform(X_val_rest)
        X_test_rest_encoded = encoder.transform(X_test_rest)

        # Combine one-hot encoded columns with non-categorical columns
        X_train_rest_encoded = pd.DataFrame(X_train_rest_encoded.toarray(), index=X_train_rest.index)
        X_val_rest_encoded = pd.DataFrame(X_val_rest_encoded.toarray(), index=X_val_rest.index)
        X_test_rest_encoded = pd.DataFrame(X_test_rest_encoded.toarray(), index=X_test.index)
        X_train_rest = pd.concat([X_train_rest_encoded, X_train_non_categorical], axis=1)
        X_val_rest = pd.concat([X_val_rest_encoded, X_val_non_categorical], axis=1)
        X_test_rest = pd.concat([X_test_rest_encoded, X_test_non_categorical], axis=1)
        
        # Convert column names to strings
        X_train_rest.columns = X_train_rest.columns.astype(str)
        X_val_rest.columns = X_val_rest.columns.astype(str)
        X_test_rest.columns = X_test_rest.columns.astype(str)
        
        # KNN fitting
        knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
        fitted_X_train_rest = X_train_rest.copy()
        fitted_train_target = train_target.copy()
        nan_value = None

        for key, value in target_dict.items():
            if pd.isna(key):
                nan_value = value
                break

        # Create a mask to filter out rows where train_target has the value 11
        mask = fitted_train_target != nan_value

        # Filter the DataFrame and the NumPy array using the mask
        fitted_X_train_rest = fitted_X_train_rest[mask]
        fitted_train_target = fitted_train_target[mask]

        knn.fit(fitted_X_train_rest, fitted_train_target)
        
        train_pred = knn.predict(X_train_rest)
        val_pred = knn.predict(X_val_rest)
        test_pred = knn.predict(X_test_rest)

        # Create dataframe with target and predictions columns
        train_df = pd.DataFrame({col: train_target, 'pred': train_pred})
        val_df = pd.DataFrame({col: val_target, 'pred': val_pred})
        test_df = pd.DataFrame({col: test_target, 'pred': test_pred})
 
        # Apply the transformation directly
        train_df["fixed"] = train_df.apply(lambda x: x['pred'] if x[col] == nan_value else x[col], axis=1)
        val_df["fixed"] = val_df.apply(lambda x: x['pred'] if x[col] == nan_value else x[col], axis=1)
        test_df["fixed"] = test_df.apply(lambda x: x['pred'] if x[col] == nan_value else x[col], axis=1)
        train_df.index = X_train_imputed.index
        val_df.index = X_val_imputed.index
        test_df.index = X_test_imputed.index       
       
        # Add target to X_train_imputed, X_val_imputed, X_test_imputed
        X_train_imputed[col] = train_df["fixed"]
        X_val_imputed[col] = val_df["fixed"]
        X_test_imputed[col] = test_df["fixed"]
  
    # Rename dataframes back to original
    X_train = X_train_imputed
    X_val = X_val_imputed
    X_test = X_test_imputed

    # Add 'Date Published' column back to X_train, X_val, X_test
    X_train['Date Published'] = train_date_published
    X_val['Date Published'] = val_date_published
    X_test['Date Published'] = test_date_published
    
    return X_train, X_val, X_test

# Impute low missing values
def impute_low_missing_values(X_train, X_val, X_test):
        
    # Low missing value rate < 0.05: mode and mean imputation
    low_missing_columns = ['Value For Money', 'Seat Comfort']
    imputer = SimpleImputer(strategy='most_frequent')
    
    for col in low_missing_columns:
        X_train[col] = imputer.fit_transform(X_train[[col]])
        X_val[col] = imputer.transform(X_val[[col]])
        X_test[col] = imputer.transform(X_test[[col]])
    
    return X_train, X_val, X_test

# Convert specified columns in a DataFrame to string type
def convert_columns_to_string(X_train, X_val, X_test):
    columns = ['Year Flown', 'Month Flown', 'Day Flown', 'Destination', 'Origin', 'Ground Service', 'Cabin Staff Service', 'Type Of Traveller', 'Aircraft', 'Trip_verified', 'Inflight Entertainment', 'Wifi & Connectivity', 'Food & Beverages', 'Value For Money', 'Seat Comfort']
    for column in columns:
        if column in X_train.columns:
            X_train[column] = X_train[column].astype(str)
        if column in X_val.columns:
            X_val[column] = X_val[column].astype(str)
        if column in X_test.columns:
            X_test[column] = X_test[column].astype(str)
    return X_train, X_val, X_test

# One-hot encoding of categorical variables
def one_hot_encode(X_train, X_val, X_test):

    # Identify categorical columns (type 'object')
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Select only categorical columns for encoding
    X_train_categorical = X_train[categorical_columns]
    X_val_categorical = X_val[categorical_columns]
    X_test_categorical = X_test[categorical_columns]

    # One-hot encode categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output = False)
    X_train_encoded = encoder.fit_transform(X_train_categorical)
    X_val_encoded = encoder.transform(X_val_categorical)
    X_test_encoded = encoder.transform(X_test_categorical)
    
    # Convert encoded arrays to DataFrames
    X_train_encoded = pd.DataFrame(X_train_encoded, index=X_train.index, columns=encoder.get_feature_names_out(categorical_columns))
    X_val_encoded = pd.DataFrame(X_val_encoded, index=X_val.index, columns=encoder.get_feature_names_out(categorical_columns))
    X_test_encoded = pd.DataFrame(X_test_encoded, index=X_test.index, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop original categorical columns
    X_train = X_train.drop(columns=categorical_columns)
    X_val = X_val.drop(columns=categorical_columns)
    X_test = X_test.drop(columns=categorical_columns)

    # Concatenate encoded columns with original data
    X_train_encoded = pd.concat([X_train, X_train_encoded], axis=1)
    X_val_encoded = pd.concat([X_val, X_val_encoded], axis=1)
    X_test_encoded = pd.concat([X_test, X_test_encoded], axis=1)

    return X_train_encoded, X_val_encoded, X_test_encoded

# Create pipeline
def create_pipeline(file_path, feature_selection=True):
    data = import_data(file_path)
    data = clean_data(data)
    data = create_datetime(data)
    data = drop_missing_target(data)

    # Splitting the dataset into the Training set, Validation set, and Test set using stratified sampling
    X = data.drop(columns=['Overall Rating'])
    y = data['Overall Rating']
    
    # Stratified split to maintain class balance
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Normalize continuous variables
    X_train, X_val, X_test = scale_continuous(X_train, X_val, X_test)

    # Impute missing values
    X_train, X_val, X_test = impute_often_missing_values(X_train, X_val, X_test)
    X_train, X_val, X_test = impute_missing_values_with_knn(X_train, X_val, X_test)
    X_train, X_val, X_test = impute_low_missing_values(X_train, X_val, X_test)

    # Prepare data for one-hot-encoding
    X_train, X_val, X_test = convert_columns_to_string(X_train, X_val, X_test)

    # One-hot encode categorical variables
    X_train, X_val, X_test = one_hot_encode(X_train, X_val, X_test)

    # One-hot encode categorical variables
    X_train, X_val, X_test = one_hot_encode(X_train, X_val, X_test)
    
    # Adjust the target labels to start from 0 instead of 1
    y_train = y_train.astype(int) - 1
    y_val = y_val.astype(int) - 1
    y_test = y_test.astype(int) - 1

    # Extract dates from train, validation, and test sets
    datetime_train = X_train[['Date Published']]
    datetime_val = X_val[['Date Published']]
    datetime_test = X_test[['Date Published']]

    # Remove dates from train, validation, and test sets
    X_train = X_train.drop(columns=['Date Published'])
    X_val = X_val.drop(columns=['Date Published'])
    X_test = X_test.drop(columns=['Date Published'])

    if feature_selection:
        # Feature selection: only keep 100 features with highest feature importance score (according to optimized random forest model based on all features)
        all_features = pd.read_csv('outputs/predictive_modeling/classification/feature_selection/feature_importance_scores.csv')
        top_100_features = all_features['feature'][:100].tolist()
        X_train = X_train[top_100_features]
        X_val = X_val[top_100_features]
        X_test = X_test[top_100_features]

    return X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data