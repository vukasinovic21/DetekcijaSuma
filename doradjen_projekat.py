import numpy as np
import pandas as pd
import cv2  # OpenCV
import rasterio
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import ee  # Earth Engine API
import geemap
import os
import leaflet
import ipywidgets as widgets

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import serial

#os.remove(os.path.expanduser('~/.config/earthengine/credentials'))
ee.Authenticate()
ee.Initialize(project='nikolavukasinovic')

# Definišemo područje interesa (AOI) - Stavio da bude moje selo - to ce da bude test
aoi = ee.Geometry.Polygon([
    [
        [20.611538, 43.045487],  # Ugao 1 (Jugo Zapadna granica)
        [20.619155, 43.045487],  # Ugao 2 (Jugo Istočna granica)
        [20.619155, 43.051737],  # Ugao 3 (Severo Istočna granica)
        [20.611538, 43.051737]   # Ugao 4 (Severo Zapadna granica)
    ]
])

# Definišemo drugo područje interesa (AOI) - Stavio da bude pokrcena suma u Amazonu - train
aoi2 = ee.Geometry.Polygon([
    [
        [-60.230, -3.540],  # Ugao 1 (Jugo Zapadna granica)
        [-60.220, -3.540],  # Ugao 2 (Jugo Istočna granica)
        [-60.220, -3.530],  # Ugao 3 (Severo Istočna granica)
        [-60.230, -3.530]   # Ugao 4 (Severo Zapadna granica)
    ]
])

# Definišemo drugo područje interesa (AOI) - Stavio da bude gusta vegetacija iz Severne amerike - train
aoi3 = ee.Geometry.Polygon([
    [
        [-83.540, 35.625],  # Ugao 1 (Jugo Zapadna granica)
        [-83.530, 35.625],  # Ugao 2 (Jugo Istočna granica)
        [-83.530, 35.635],  # Ugao 3 (Severo Istočna granica)
        [-83.540, 35.635]   # Ugao 4 (Severo Zapadna granica)
    ]
])

def calculate_ndvi_evi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }).rename('EVI')
    return image.addBands([ndvi, evi])


datasets = [
    ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate('2015-01-01', '2022-12-31').filterBounds(aoi2).map(calculate_ndvi_evi),
    ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate('2015-01-01', '2022-12-31').filterBounds(aoi3).map(calculate_ndvi_evi),
]

combined_dataset = ee.ImageCollection(datasets[0])

for dataset in datasets[1:]:
    combined_dataset = combined_dataset.merge(dataset)
images = combined_dataset.median()


dataset_test = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate('2015-01-01', '2022-12-31').filterBounds(aoi).map(calculate_ndvi_evi)
image = ee.Image(dataset_test.first())

vis_params = {
    'bands': ['B4', 'B3', 'B2'],  # RGB bands
    'min': 0,
    'max': 3000,
    'gamma': 1.4
}

# Pravimo mapu za vizuelizaciju
Map = geemap.Map()
Map.centerObject(aoi, 13)
Map.addLayer(image, vis_params, 'RGB Image')
Map.addLayer(aoi, {}, 'AOI')
Map.addLayerControl()

first_map = 'first_map.html'
Map.save(first_map)


def calculate_ndvi(image):
    # Racunamo NDVI: (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Uzimamo prosecnu sliku za period 2015-2022 za NDVI i EVI
median_ndvi = combined_dataset.select('NDVI').median() 
median_evi = combined_dataset.select('EVI').median() 

# Selo
median_ndvi_test = dataset_test.select('NDVI').median().clip(aoi)
median_evi_test = dataset_test.select('EVI').median().clip(aoi)

# Amazon
median_ndvi2 = datasets[0].select('NDVI').median() 
median_evi2 = datasets[0].select('EVI').median() 

# Amerika
median_ndvi3 = datasets[1].select('NDVI').median() 
median_evi3 = datasets[1].select('EVI').median() 

# Postavljamo parametre za NDVI (Normalized Difference Vegetation Index)
ndvi_params = {
    'min': -1,
    'max': 1,
    'palette': ['blue', 'white', 'green']
}

# Visualize NDVI
Map = geemap.Map()
Map.centerObject(aoi, 13)
Map.addLayer(median_ndvi_test, ndvi_params, 'Median NDVI')
Map.addLayer(aoi, {}, 'AOI')
Map.addLayerControl()

second_map = 'second_map.html'
Map.save(second_map)


#images_folder = os.path.join(os.getcwd(), 'images')
images_folder = "K:\\DetekcijaSuma\\images"
export_path = os.path.join(images_folder, 'median_ndvi_test.tif')
export_path1 = os.path.join(images_folder, 'median_evi_test.tif')
export_path2 = os.path.join(images_folder, 'median_ndvi_train1.tif')
export_path3 = os.path.join(images_folder, 'median_evi_train1.tif')
export_path4 = os.path.join(images_folder, 'median_ndvi_train2.tif')
export_path5 = os.path.join(images_folder, 'median_evi_train2.tif')

# Exportujemo test slike na lokal
if not os.path.exists(export_path):
    geemap.ee_export_image(
        median_ndvi_test,
        filename=export_path,
        scale=5,
        region=aoi.getInfo()['coordinates']
    )

if not os.path.exists(export_path1):
    geemap.ee_export_image(
        median_evi_test,
        filename=export_path1,
        scale=5,
        region=aoi.getInfo()['coordinates']
    )

# Exportujemo train slike na lokal
if not os.path.exists(export_path2):
    geemap.ee_export_image(
        median_ndvi2,
        filename=export_path2,
        scale=10,
        region=aoi2.getInfo()['coordinates']
    )

if not os.path.exists(export_path3):
    geemap.ee_export_image(
        median_evi2,
        filename=export_path3,
        scale=10,
        region=aoi2.getInfo()['coordinates']
    )

if not os.path.exists(export_path4):
    geemap.ee_export_image(
        median_ndvi3,
        filename=export_path4,
        scale=10,
        region=aoi3.getInfo()['coordinates']
    )

if not os.path.exists(export_path5):
    geemap.ee_export_image(
        median_evi3,
        filename=export_path5,
        scale=10,
        region=aoi3.getInfo()['coordinates']
    )


# Ucitavamo exportovanu NDVI test sliku
with rasterio.open(export_path) as src:
    ndvi_data_test = src.read(1)  # Citamo prvi band
    profile = src.profile

# Ucitavamo exportovanu EVI test sliku
with rasterio.open(export_path1) as src:
    evi_data_test = src.read(1)  # Citamo prvi band
    profile = src.profile

# Ucitavamo exportovanu NDVI train1 sliku
with rasterio.open(export_path2) as src:
    ndvi_data1 = src.read(1)  # Citamo prvi band
    profile = src.profile

# Ucitavamo exportovanu EVI train1 sliku
with rasterio.open(export_path3) as src:
    evi_data1 = src.read(1)  # Citamo prvi band
    profile = src.profile

# Ucitavamo exportovanu NDVI train2 sliku
with rasterio.open(export_path4) as src:
    ndvi_data2 = src.read(1)  # Citamo prvi band
    profile = src.profile

# Ucitavamo exportovanu EVI train2 sliku
with rasterio.open(export_path5) as src:
    evi_data2 = src.read(1)  # Citamo prvi band
    profile = src.profile



assert ndvi_data1.shape == ndvi_data2.shape, "NDVI slike moraju da budu istih dimenzija"
assert evi_data1.shape == evi_data2.shape, "EVI slike moraju da budu istih dimenzija"



ndvi_data = np.median([ndvi_data1, ndvi_data2], axis=0)
evi_data = np.median([evi_data1, evi_data2], axis=0)

# Priprema feature-a za trening, kombinovanje NDVI i EVI u feature
X = np.vstack([
    ndvi_data.flatten(), 
    evi_data.flatten() 
]).T 

# Gde su ndvi i evi manji od 0.21 racunacemo da je pokrceno(deforestation) a ostali ce da upadnu u sume
labels = np.where((ndvi_data < 0.21) & (evi_data < 0.21), 1, 0)
y = labels.flatten()  # Labels (0 i dalje sume, 1 pokrceno)

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Deljenje podataka na train i test skupove u odnosu 75/25
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Inicijalizacija RandomForestClassifier sa 100 jedinica
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Treniranje modela
rf_classifier.fit(X_train, y_train)

# Prediktovanje
y_pred = rf_classifier.predict(X_test)


# Vise se ne salje mejl 
# Sada se salje poruka na mikrokontroler preko uart.com koriscenjem pyserial biblioteke
# Posto nemam uredjaj testiracu kao simulaciju 
def send_email(message):
    try:
        ser = serial.Serial('COM3', 9600, timeout=1)
        ser.write(message.encode())
        ser.close()
        print(f"Alarm: '{message}' poslat na mikrokontroler.")

    except serial.SerialException as e:
        print("Serijski port nije dostupan. Simuliracu slanje alarma na mikrokontroler.")
        print(f"Simulirani alarm: '{message}' poslat na mikrokontroler.")

    except Exception as e:
        print(f"Greška pri slanju alarma: {e}")



def load_data():
    aoinew = ee.Geometry.Polygon([
    [
        [20.615000, 43.046000],  # Ugao 1 (Jugo Zapadna granica)
        [20.619000, 43.046000],  # Ugao 2 (Jugo Istočna granica)
        [20.619000, 43.050000],  # Ugao 3 (Severo Istočna granica)
        [20.615000, 43.050000]   # Ugao 4 (Severo Zapadna granica)
    ]
    ])

    # Trebalo bi da se uzimaju datumi u zadnjih nedelju dana ili samo za juce npr, ali nemaju jos podaci za taj period
    dataset_new = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate('2024-01-01', '2024-03-31').filterBounds(aoi).map(calculate_ndvi_evi)

    median_ndvi_new = dataset_new.select('NDVI').median().clip(aoinew)
    median_evi_new = dataset_new.select('EVI').median().clip(aoinew)

    export_path_new = os.path.join(images_folder, 'median_ndvi_new.tif')
    export_path_new2 = os.path.join(images_folder, 'median_evi_new.tif')

    geemap.ee_export_image(
        median_ndvi_new,
        filename=export_path_new,
        scale=10,
        region=aoi.getInfo()['coordinates']
    )

    geemap.ee_export_image(
        median_evi_new,
        filename=export_path_new2,
        scale=10,
        region=aoi.getInfo()['coordinates']
    )

    with rasterio.open(export_path_new) as src:
        ndvi_data_new = src.read(1) 

    with rasterio.open(export_path_new2) as src:
        evi_data_new = src.read(1) 

    X_new = np.vstack([
        ndvi_data_new.flatten(), 
        evi_data_new.flatten()    
    ]).T

    X_scaled_new = scaler.transform(X_new)

    y_pred_new = rf_classifier.predict(X_scaled_new)
    predicted_map = y_pred_new.reshape(ndvi_data_new.shape)
        
    return predicted_map   



# Proveravamo da li ima vise pokrcenih piksela od navedenog broja piksela (5% za sad).
def check_deforestation_total(predicted_map):
    total_deforested_pixels = np.sum(predicted_map == 1)  

    # Ako ima ukupno vise od 5% pokrcenih piksela onda se aktivira alarm
    if total_deforested_pixels >= (predicted_map.size * 0.05):
        return True  
    return False

def monitor_deforestation_and_notify():
    predicted_map = load_data()

    # Mozemo da vidimo koliko ukupno ima piksela kako bi promenili treshold 
    #print(predicted_map.size)

    # Ako postoji vise pokrcenih piksela od navedenog broja piksela navedenih u check_deforestation_total
    if check_deforestation_total(predicted_map):
        send_email('Bespravna seca je zabelezena u tvojoj oblasti.')

monitor_deforestation_and_notify()