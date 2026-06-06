# Internal python libraries
import pathlib as path
import time

# External libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Internal libraries

# Code

def format_x(value, pos):
    # Formatera x-axeln utan exponentform även om det är riktigt stora tal (då funkar inte ScalarFormatter)
    return f'{int(value):,}'  # Lägger till tusentalsseparator

CURRENT_PATH = path.Path()
DATA_PATH = CURRENT_PATH / "data"
MODEL_DIR = CURRENT_PATH / "models"
DATA_FILE_PATH = DATA_PATH / "used_cars_data.csv"
CACHE_PATH = CURRENT_PATH / "cache" 
CACHE_FILE_PATH = CACHE_PATH / "used_cars_data.pkl" 
MODEL_FILE_PATH = MODEL_DIR / "linear-regression-model.joblib"

# 1. Ladda in orginaldata eller cache:ad data om det finns
seconds_start = time.time()

if CACHE_FILE_PATH.exists():
    original_car_sales_info = pd.read_pickle(CACHE_FILE_PATH)   # Laddning tar ca 5 sekunder
else:
    original_car_sales_info = pd.read_csv(DATA_FILE_PATH)       # Laddning tar ca 90+ sekunder
    original_car_sales_info.to_pickle(CACHE_FILE_PATH)

seconds_finished = time.time()
print(f"Import time: {seconds_finished - seconds_start}")       


# 2. Analysera data och rensa bort felaktig eller icke relavant data

## Gå datat översiktligt
print(f"Example: {original_car_sales_info.hist}")
print(f"Info: {original_car_sales_info.info}")
print(f"Size: {original_car_sales_info.size}")
print(f"Data info: {original_car_sales_info.dtypes}")
print(f"Describe: {original_car_sales_info.describe()}")

pd.set_option('display.max_rows', None)

## Spara endast kolumner vi vill börja titta vidare på
seconds_start = time.time()

relevant_car_info = pd.DataFrame()
relevant_car_info['sp_id'] = original_car_sales_info['sp_id']                   # Unikt ID
relevant_car_info['make_name'] = original_car_sales_info['make_name']           # Märke
relevant_car_info['model_name'] = original_car_sales_info['model_name']         # Modell
relevant_car_info['fuel_type'] = original_car_sales_info['fuel_type']           # Bränsletyp
relevant_car_info['price'] = original_car_sales_info['price']                   # Pris
relevant_car_info['year'] = original_car_sales_info['year']                     # Årsmodell
relevant_car_info['is_new'] = original_car_sales_info['is_new']                 # Om den är mindre än 2 år
relevant_car_info['horsepower'] = original_car_sales_info['horsepower']         # Hästkrafter
relevant_car_info['has_accidents'] = original_car_sales_info['has_accidents']   # Varit med om olycka?
relevant_car_info['mileage'] = original_car_sales_info['mileage']               # Mätarställning
#relevant_car_info['back_legroom'] = original_car_sales_info['back_legroom']
#relevant_car_info['body_type'] = original_car_sales_info['body_type']
#relevant_car_info['city'] = original_car_sales_info['city']
#relevant_car_info['city_fuel_economy'] = original_car_sales_info['city_fuel_economy']
#relevant_car_info['combine_fuel_economy'] = original_car_sales_info['combine_fuel_economy']
#relevant_car_info['daysonmarket'] = original_car_sales_info['daysonmarket']
#relevant_car_info['engine_cylinders'] = original_car_sales_info['engine_cylinders']
#relevant_car_info['engine_type'] = original_car_sales_info['engine_type']
#relevant_car_info['exterior_color'] = original_car_sales_info['exterior_color']
#relevant_car_info['frame_damaged'] = original_car_sales_info['frame_damaged']
#relevant_car_info['front_legroom'] = original_car_sales_info['front_legroom']
#relevant_car_info['fuel_tank_volume'] = original_car_sales_info['fuel_tank_volume']
#relevant_car_info['height'] = original_car_sales_info['height']
#relevant_car_info['highway_fuel_economy'] = original_car_sales_info['highway_fuel_economy']
#relevant_car_info['interior_color'] = original_car_sales_info['interior_color']
#relevant_car_info['isCab'] = original_car_sales_info['isCab']
#relevant_car_info['length'] = original_car_sales_info['length']
#relevant_car_info['listed_date'] = original_car_sales_info['listed_date']
#relevant_car_info['main_picture_url'] = original_car_sales_info['main_picture_url']
#relevant_car_info['maximum_seating'] = original_car_sales_info['maximum_seating']
#relevant_car_info['owner_count'] = original_car_sales_info['owner_count']
#relevant_car_info['power'] = original_car_sales_info['power']
#relevant_car_info['seller_rating'] = original_car_sales_info['seller_rating']
#relevant_car_info['sp_name'] = original_car_sales_info['sp_name']
#relevant_car_info['torque'] = original_car_sales_info['torque']
#relevant_car_info['wheelbase'] = original_car_sales_info['wheelbase']
#relevant_car_info['width'] = original_car_sales_info['width']
print(relevant_car_info.columns)

## Kolla vilka kolumner som saknar data
pd.set_option('display.max_rows', None)
print(f"Saknad data:\n {relevant_car_info.isnull().sum()}")
print(f"Number of rows: {relevant_car_info.size}")

#Vi tar bort alla rader där någon av följande kolumner saknar data
#sp_id                 96
#make_name              0
#model_name             0
#fuel_type          82724
#price                  0
#year                   0
#is_new                 0
#horsepower        172386
#has_accidents    1426595
#mileage           144387

clean_relevant_car_info = relevant_car_info.dropna()
print(f"Saknad data:\n {clean_relevant_car_info.isnull().sum()}")

#sp_id            0
#make_name        0
#model_name       0
#fuel_type        0
#price            0
#year             0
#is_new           0
#horsepower       0
#has_accidents    0
#mileage          0

# 2.3 Analysera viss data närmare
print(f"make_name: {clean_relevant_car_info['make_name'].unique()}")


#Märke
#[   'Land Rover',    'Alfa Romeo',           'BMW',       'Hyundai',
#     'Chevrolet',         'Lexus',          'Jeep',      'Cadillac',
#      'Chrysler',         'Dodge', 'Mercedes-Benz',        'Nissan',
#         'Honda',           'Kia',          'Ford',       'Lincoln',
#        'Subaru',          'Audi',    'Volkswagen',       'Porsche',
#        'Jaguar',         'Mazda',        'Toyota',           'GMC',
#         'Acura',      'INFINITI',      'Maserati',           'RAM',
#          'FIAT',         'Volvo',    'Mitsubishi',         'Buick',
#       'Mercury',         'Scion',          'Saab',          'MINI',
#       'Ferrari',       'Genesis',        'Saturn',       'Bentley',
#        'Suzuki',        'Fisker',       'Pontiac',   'Lamborghini',
#         'smart',        'Hummer',   'Rolls-Royce',         'Lotus',
#       'McLaren',  'Aston Martin',       'Maybach',         'Isuzu',
#    'Oldsmobile',      'Plymouth',         'Karma',        'Datsun',
#   'Pininfarina',           'Geo',           'SRT',         'Eagle',
#       'Bugatti',        'Daewoo',    'AM General',      'DeLorean']
# Ser bra ut, ingen skräpdata vad jag kan se

# Modell
print(f"model_name: {clean_relevant_car_info['model_name'].unique().tolist()}")
#model_name: ['Range Rover Velar', 'Range Rover Evoque', '4C', '3 Series', 'Elantra', 'Malibu', 'RC 350', 'Traverse', 'Grand Cherokee', 
#             'Compass', 'Veloster', 'XT4', '200', 'Equinox', 'Renegade', 'Wrangler Unlimited', 'Durango', 'CLA-Class', 'Charger', 'Silverado 1500', 
#             'Rogue', 'Civic', 'RX 350', 'GLC-Class', 'Optima', 'Explorer', 'Navigator', 'Outback', '4 Series', 'GLE-Class', 'Escalade', 'Tucson', 
#             '2 Series', 'Pacifica', 'Suburban', 'Camaro', 'Cruze', 'Trax', 'SQ5', 'Maxima', 'Cherokee', 'Tiguan', 'F-350 Super Duty', 'Colorado', 
#             'Pathfinder', 'Tahoe', 'Blazer', 'Sorento', 'SRX', 'Mustang', 'X5', 'Range Rover', 'Impala', 'Cayenne', 'F-PACE', 'GLK-Class', 

# Datat ser bra ut (men det finns måna modeller så vi visar bara ett fåtal)


# Bränsletyp
print(f"fuel_type: {clean_relevant_car_info['fuel_type'].unique()}")
#fuel_type: <ArrowStringArray>
#[              'Gasoline',              'Biodiesel',      'Flex Fuel Vehicle',
#                 'Hybrid',                 'Diesel', 'Compressed Natural Gas',
#                'Propane',               'Electric']


## Gör om bränsletyp strängar till numeriska enum värden för att kunna användas om man bygger en modell
fuel_mapping = {
    'Gasoline': 0,
    'Biodiesel': 1,
    'Flex Fuel Vehicle': 2,
    'Hybrid': 3,
    'Diesel': 4,
    'Compressed Natural Gas': 5,
    'Propane': 6,
    'Electric': 7,
}

## Gör om Namn och Märle till IDn
le_make = LabelEncoder()
clean_relevant_car_info['make_encoded'] = le_make.fit_transform(clean_relevant_car_info['make_name'])

le_model = LabelEncoder()
clean_relevant_car_info['model_encoded'] = le_model.fit_transform(clean_relevant_car_info['model_name'])


clean_relevant_car_info['fuel_type_encoded'] = clean_relevant_car_info['fuel_type'].map(fuel_mapping)
print(f"fuel_type_encoded: {clean_relevant_car_info['fuel_type_encoded'].unique()}")

# is_new känns inte relevant längee iom att vi har årsmodell så vi tar bort den
clean_relevant_car_info.drop('is_new', axis=1, inplace=True)

# Vi tar fram en bargraf och tittar på prisspannet på bilarna
# Först kollar vi max och min-värdena
pd.set_option('display.float_format', '{:.0f}'.format)      # Visa nummer sutan exponentform
# Vi gör om priset till SEK (USD/SEK sätts till 10. Det är inte relevant för beräkningarna vad kursen var)
clean_relevant_car_info['price'] = clean_relevant_car_info['price'] * 10
print(f"Describe\n: {clean_relevant_car_info['price'].describe()}")
print(f"Number of rows: {clean_relevant_car_info.size}")


## Skapa ett barchart diagram över priset, uppdelat i 50 000 kr block, för samtliga bilar
plt.figure(figsize=(10, 6))
plt.hist(clean_relevant_car_info['price'], bins=range(0, 33000000, 50000), edgecolor='black', alpha=0.7)
plt.yscale('log')  # Log-skalning på y-axeln
plt.xlabel('Pris (SEK)')
plt.ylabel('Antal (log-skala)')
plt.title('Spridning av pris')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x))
plt.grid(True, alpha=0.3)
#plt.show()


## Skapa ett scatterplot diagram för att se korrelationen mellan årsmodell och pris, för Volvo XC60
volvo_XC60 = clean_relevant_car_info[(clean_relevant_car_info['make_name'] == 'Volvo') & (clean_relevant_car_info['model_name'] == 'XC60')]

plt.figure(figsize=(10, 6))
plt.scatter(volvo_XC60['year'], volvo_XC60['price'], alpha=0.5, s=10)
plt.xlabel('Årsmodell')
plt.ylabel('Pris (SEK)')
plt.title('Samband mellan årsmodell och pris för Volvo XC60')
plt.grid(True, alpha=0.3)
#plt.show()

## Skapa ett linjediagram för att se korrelationen mellan årsmodell och medelvärde på pris, för Volvo XC60
avg = volvo_XC60.groupby('year')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(avg['year'], avg['price'], marker='o', linewidth=2, markersize=6)
plt.xlabel('Årsmodell')
plt.ylabel('Medel pris')
plt.title('Medel pris per årsmodell')
plt.grid(True, alpha=0.3)
#plt.show()


## Skapa ett scetterplot diagram för att se korrelationen mellan antal körda mil och pris, för Volvo XC60
## Här ser vi en intressant sak. Det finns ett klart samband men för låga antal körda mil finns det två nivåer
## Tittar vidar epå vad det kan bero på om det finns tid
plt.figure(figsize=(10, 6))
plt.scatter(volvo_XC60['mileage'], volvo_XC60['price'], alpha=0.5, s=10)
plt.xlabel('Mätarställning')
plt.ylabel('Pris (SEK)')
plt.title('Samband mellan mätarställning och pris')
plt.grid(True, alpha=0.3)
#plt.show()

## Titta på korrelation mellan alla mätarställningsvärden per årsmodell
plt.figure(figsize=(10, 6))
plt.scatter(volvo_XC60['mileage'], volvo_XC60['year'], alpha=0.5, s=10)
plt.xlabel('Mätarställning')
plt.ylabel('Årsmodell')
plt.title('Samband mellan mätarställning och årsmodell')
plt.grid(True, alpha=0.3)
#plt.show()

# Titta på korrelation mellan medelvärde av mätarställning per år vs årsmodell
avg_per_year = volvo_XC60.groupby('year')['mileage'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(avg_per_year['year'], avg_per_year['mileage'], marker='o', linewidth=2, markersize=6)
plt.xlabel('Årsmodell')
plt.ylabel('Medel mätarställning (km)')
plt.title('Medel mätarställning per årsmodell')
plt.grid(True, alpha=0.3)
#plt.show()


## Summering inför att modellen byggs
## Eftersom det inte är så mycket tid innan inlämning så väljer jag ut två av parametrarna och bygger modellen på det analyserar och bygger sen vidare
## Då kan jag lämna in något grundläggande och riskerar inte att fastna i detaljerna
## Väljer: Märke, Modell, Års och Mätarställning för att beräkna pris

# 3. Bygg modell
feature_columns = ["year", "mileage", "has_accidents"]
# feature_columns = ["make_encoded", "model_encoded", "year", "mileage", "has_accidents"]   # Doesnt work well to include make and model
target_column = "price"

#X = clean_relevant_car_info[feature_columns]
#y = clean_relevant_car_info[target_column]
X = volvo_XC60[feature_columns]
y = volvo_XC60[target_column]

## Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,  # Columns to send in to model, that is known
    y,  # Column to get out from model, that is estimated
    test_size=0.25, 
    random_state=42)

## Create an untrained model of LinearRegression type 
model = LinearRegression()

## Train model
model.fit(X_train, y_train)

predictions = model.predict(X_test)

results = X_test.copy()
results['actual_price'] = y_test.values
results['predicted_prices'] = predictions

print(results.head())

## Save outputs and export model
results.to_csv("output", index=False)
joblib.dump(model, "model.joblib")

## Beräkna R², dvs hur bra modellen är på att estimera (0-1 bäst)
r2 = r2_score(y_test, predictions)
print(f"R² = {r2:.4f} ({r2*100:.2f}%)")

print(f"Modellen testades på {X_test.size} och tränades på {X_train.size} bilförsäljningar")
# R² = 0.7868 (78.68%)

# Spara modellen med utförd träningsdata
joblib.dump(model, MODEL_FILE_PATH)