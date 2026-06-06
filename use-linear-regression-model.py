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

# Ladda in orginaldata eller cache:ad data om det finns

if CACHE_FILE_PATH.exists():
    original_car_sales_info = pd.read_pickle(CACHE_FILE_PATH)   # Laddning tar ca 5 sekunder
else:
    original_car_sales_info = pd.read_csv(DATA_FILE_PATH)       # Laddning tar ca 90+ sekunder
    original_car_sales_info.to_pickle(CACHE_FILE_PATH)  

# Förbered data att skicka in till modellen
feature_columns = ["year", "mileage", "has_accidents"]
car_info = pd.DataFrame(
    data = [
        ['2019', 10000, True]
    ],
    columns = feature_columns
)

# Läs in modellen
linear_regression_model = joblib.load(MODEL_FILE_PATH)

# Anropa modellen
predicted_prices = linear_regression_model.predict(car_info)

# Presentera resultat
print(f"Uppskattat pris på bilen är {predicted_prices[0]:,.0f}".replace(",", " ").replace(".", ",")) 
