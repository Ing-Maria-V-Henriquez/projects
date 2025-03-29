# Parte 1: Importación de Librerías y Definición de la Función para Leer Datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from pyproj import Transformer
import os
from folium.plugins import MarkerCluster
import numpy as np

ruta_carpeta = 'D:/Nueva carpeta/Curso de google analisis de datos/Proyecto - Caso de Estudio Bike/divvy-tripdata'

def leer_archivos_csv_en_carpeta(ruta_carpeta):
    """
    Lee todos los archivos CSV en una carpeta y los combina en un solo DataFrame.
    """
    lista_archivos = os.listdir(ruta_carpeta)
    lista_dataframes = []

    for archivo in lista_archivos:
        if archivo.endswith('.csv'):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            try:
                if archivo == '202412-divvy-tripdata.csv':
                    try:
                        df = pd.read_csv(ruta_archivo, sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)
                    except UnicodeDecodeError:
                        df = pd.read_csv(ruta_archivo, sep=';', encoding='windows-1252', on_bad_lines='skip', low_memory=False)
                    df['start_lat'] = pd.to_numeric(df['start_lat'], errors='coerce') / 10000
                    df['start_lng'] = pd.to_numeric(df['start_lng'], errors='coerce') / 10000
                    df['end_lat'] = pd.to_numeric(df['end_lat'], errors='coerce') / 10000
                    df['end_lng'] = pd.to_numeric(df['end_lng'], errors='coerce') / 10000
                    df['ride_id'] = df['ride_id'].str.replace(',', '.')
                    df['started_at'] = pd.to_datetime(df['started_at'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                    df['ended_at'] = pd.to_datetime(df['ended_at'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                else:
                    df = pd.read_csv(ruta_archivo)
                lista_dataframes.append(df)
            except Exception as e:
                print(f"Error al leer {archivo}: {e}")

    if lista_dataframes:
        df = pd.concat(lista_dataframes, ignore_index=True)
        return df
    else:
        return None

df = leer_archivos_csv_en_carpeta(ruta_carpeta)

# Parte 2: Limpieza y Preprocesamiento de Datos

if df is not None:
    # Manejo de valores nulos
    numeric_cols = df.select_dtypes(include=['number']).columns
    object_cols = df.select_dtypes(include=['object']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[object_cols] = df[object_cols].fillna('Desconocido')

    # Conversión de tipos de datos de fechas
    df['started_at'] = pd.to_datetime(df['started_at'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

    # Eliminar filas con fechas nulas después de la conversión
    df = df.dropna(subset=['started_at', 'ended_at'])

    # Conversión de coordenadas
    transformer = Transformer.from_crs("EPSG:32616", "EPSG:4326")
    df['lat'], df['lng'] = transformer.transform(df['start_lng'], df['start_lat'])

    # Filtrado de coordenadas
    df = df[(df['lat'] >= -90) & (df['lat'] <= 90) & (df['lng'] >= -180) & (df['lng'] <= 180)]

    # Cálculo de la duración del viaje
    df['duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

    # Eliminar filas con duraciones negativas
    df = df[df['duration'] >= 0]

# Parte 3: Análisis Exploratorio de Datos (EDA) y Visualizaciones Básicas

# Estadísticas de la duración
print("Estadísticas de la duración:")
print(df['duration'].describe())

# Conteo de tipos de usuarios
print("\nConteo de tipos de usuarios:")
print(df['member_casual'].value_counts())

sns.countplot(x='member_casual', data=df)
plt.xlabel('Tipo de usuario')
plt.ylabel('Número de viajes')
plt.title('Distribución de tipos de usuarios')
plt.show()

# Conteo de tipos de bicicletas
print("\nConteo de tipos de bicicletas:")
print(df['rideable_type'].value_counts())

sns.countplot(x='rideable_type', data=df)
plt.xlabel('Tipo de bicicleta')
plt.ylabel('Número de viajes')
plt.title('Distribución de tipos de bicicletas')
plt.show()

# Tendencias temporales
df['hour'] = df['started_at'].dt.hour
df['day_of_week'] = df['started_at'].dt.day_name()
df['month'] = df['started_at'].dt.month_name()

# Tendencias por hora (conteo)
print("\nTendencias por hora:")
print(df['hour'].value_counts().sort_index())

sns.countplot(x='hour', data=df)
plt.xlabel('Hora del día')
plt.ylabel('Número de viajes')
plt.title('Tendencias de uso por hora')
plt.show()

# Tendencias por día de la semana (conteo)
print("\nTendencias por día de la semana:")
print(df['day_of_week'].value_counts())

sns.countplot(x='day_of_week', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xlabel('Día de la semana')
plt.ylabel('Número de viajes')
plt.title('Tendencias de uso por día de la semana')
plt.show()

# Tendencias por mes (conteo)
print("\nTendencias por mes:")
print(df['month'].value_counts())

sns.countplot(x='month', data=df, order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
plt.xlabel('Mes')
plt.ylabel('Número de viajes')
plt.title('Tendencias de uso por mes')
plt.show()

# Parte 4: Análisis de Estaciones y Visualización con Folium

# Análisis de las estaciones más concurridas por hora:
hourly_station_counts = df.groupby(['start_station_name', 'hour']).size().unstack(fill_value=0)
most_popular_hourly_stations = hourly_station_counts.idxmax()
plt.figure(figsize=(16, 10))
sns.heatmap(hourly_station_counts.T, cmap='YlGnBu')
plt.title('Estaciones más concurridas por hora')
plt.xlabel('Estación de inicio')
plt.ylabel('Hora del día')
plt.show()

# Análisis de las estaciones más concurridas por día de la semana:
weekly_station_counts = df.groupby(['start_station_name', 'day_of_week']).size().unstack(fill_value=0)
most_popular_weekly_stations = weekly_station_counts.idxmax()
plt.figure(figsize=(16, 10))
sns.heatmap(weekly_station_counts.T, cmap='YlGnBu')
plt.title('Estaciones más concurridas por día de la semana')
plt.xlabel('Estación de inicio')
plt.ylabel('Día de la semana')
plt.show()


# Parte 5: Análisis por Tipo de Usuario y Visualizaciones Adicionales

# Visualización con Folium
map_chicago = folium.Map(location=[41.88, -87.63], zoom_start=12)
for lat, lng, name in zip(df['lat'].head(100), df['lng'].head(100), df['start_station_name'].head(100)):
    folium.Marker([lat, lng], popup=name).add_to(map_chicago)
map_chicago

# Análisis por tipo de usuario (hora del día)
print("\nTendencias por hora (por tipo de usuario):")
print(df.groupby(['hour', 'member_casual']).size().unstack())

sns.countplot(x='hour', hue='member_casual', data=df)
plt.xlabel('Hora del día')
plt.ylabel('Número de viajes')
plt.title('Tendencias de uso por hora (por tipo de usuario)')
plt.show()

# Análisis por día de la semana y mes, y por tipo de bicicleta
print("\nTendencias por día de la semana (por tipo de usuario):")
print(df.groupby(['day_of_week', 'member_casual']).size().unstack())
sns.countplot(x='day_of_week', hue='member_casual', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xlabel('Día de la semana')
plt.ylabel('Número de viajes')
plt.title('Tendencias de uso por día de la semana (por tipo de usuario)')
plt.show()

# Análisis por mes (por tipo de usuario):
print("\nTendencias por mes (por tipo de usuario):")
print(df.groupby(['month', 'member_casual']).size().unstack())
sns.countplot(x='month', hue='member_casual', data=df, order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
plt.xlabel('Mes')
plt.ylabel('Número de viajes')
plt.title('Tendencias de uso por mes (por tipo de usuario)')
plt.show()

# Análisis por tipo de bicicleta (por tipo de usuario):
print("\nTipos de bicicleta (por tipo de usuario):")
print(df.groupby(['rideable_type', 'member_casual']).size().unstack())
sns.countplot(x='rideable_type', hue='member_casual', data=df)
plt.xlabel('Tipo de bicicleta')
plt.ylabel('Número de viajes')
plt.title('Tipos de bicicleta (por tipo de usuario)')
plt.show()

# Parte 6: Visualizaciones de Folium por Día, Mes y Hora

# Visualizaciones de Folium por día, mes y hora:
# Visualizaciones de Folium por día, mes y hora:
    # Por día de la semana:
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
    map_day = folium.Map(location=[41.88, -87.63], zoom_start=12)
    day_df = df[df['day_of_week'] == day]
    for lat, lng, name in zip(day_df['lat'].head(100), day_df['lng'].head(100), day_df['start_station_name'].head(100)):
            folium.Marker([lat, lng], popup=name).add_to(map_day)
    map_day.save(f'map_chicago_{day}.html')  # Guarda el mapa en un archivo HTML

    # Por mes:
    for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
        map_month = folium.Map(location=[41.88, -87.63], zoom_start=12)
        month_df = df[df['month'] == month]
        for lat, lng, name in zip(month_df['lat'].head(100), month_df['lng'].head(100), month_df['start_station_name'].head(100)):
            folium.Marker([lat, lng], popup=name).add_to(map_month)
        map_month.save(f'map_chicago_{month}.html')  # Guarda el mapa en un archivo HTML

    # Por hora:
    for hour in range(24):
        map_hour = folium.Map(location=[41.88, -87.63], zoom_start=12)
        hour_df = df[df['hour'] == hour]
        for lat, lng, name in zip(hour_df['lat'].head(100), hour_df['lng'].head(100), hour_df['start_station_name'].head(100)):
            folium.Marker([lat, lng], popup=name).add_to(map_hour)
        map_hour.save(f'map_chicago_hour_{hour}.html')  # Guarda el mapa en un archivo HTML


# Parte 7: Identificaciíon de valores atípicos

# Convierte las columnas a tipo datetime
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])

# Calcula la duración del viaje en minutos
df['duracion_viaje'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

# Estadísticas descriptivas de la duración del viaje
print("Estadísticas de la duración del viaje:")
print(df['duracion_viaje'].describe())

# Diagrama de caja y bigotes para identificar valores atípicos
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['duracion_viaje'])
plt.title('Diagrama de Caja y Bigotes de la Duración del Viaje')
plt.xlabel('Duración del Viaje (minutos)')
plt.show()

# Identificación de valores atípicos usando el rango intercuartílico (IQR)
Q1 = df['duracion_viaje'].quantile(0.25)
Q3 = df['duracion_viaje'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

viajes_cortos_atipicos = df[df['duracion_viaje'] < limite_inferior]
viajes_largos_atipicos = df[df['duracion_viaje'] > limite_superior]

print("\nViajes cortos atípicos:")
print(viajes_cortos_atipicos)

print("\nViajes largos atípicos:")
print(viajes_largos_atipicos)


# Parte 8: Análisis de Valores Atípicos

# Reemplaza 'nombre_columna_duracion' con el nombre correcto de tu columna
nombre_columna_duracion = 'duration' # ejemplo

# Estadísticas descriptivas de la duración del viaje
print("Estadísticas de la duración del viaje:")
print(df[nombre_columna_duracion].describe())

# Diagrama de caja y bigotes para identificar valores atípicos
plt.figure(figsize=(8, 6))
sns.boxplot(x=df[nombre_columna_duracion])
plt.title('Diagrama de Caja y Bigotes de la Duración del Viaje')
plt.xlabel('Duración del Viaje (minutos)')
plt.show()

# Identificación de valores atípicos usando el rango intercuartílico (IQR)
Q1 = df[nombre_columna_duracion].quantile(0.25)
Q3 = df[nombre_columna_duracion].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

viajes_cortos_atipicos = df[df[nombre_columna_duracion] < limite_inferior]
viajes_largos_atipicos = df[df[nombre_columna_duracion] > limite_superior]

print("\nViajes cortos atípicos:")
print(viajes_cortos_atipicos)

print("\nViajes largos atípicos:")
print(viajes_largos_atipicos)

# Reemplaza 'nombre_columna_tipo_usuario' con el nombre correcto de tu columna (si aplica)
nombre_columna_tipo_usuario = 'member_casual' # ejemplo

# Distribución de tipos de usuario en viajes atípicos
print("\nDistribución de tipos de usuario en viajes cortos atípicos:")
print(viajes_cortos_atipicos[nombre_columna_tipo_usuario].value_counts())

print("\nDistribución de tipos de usuario en viajes largos atípicos:")
print(viajes_largos_atipicos[nombre_columna_tipo_usuario].value_counts())


# Parte 9: Creación de una Variable de Indicador:
 # Crea la columna 'es_atipico'
df['es_atipico'] = 0  # Inicializa con 0 (no atípico)
df.loc[df['duracion_viaje'] > limite_superior, 'es_atipico'] = 1  # Marca como 1 (atípico)

# Verifica la creación de la columna
print(df[['duracion_viaje', 'es_atipico']].head())

# Ahora puedes realizar análisis comparativos
# Estadísticas descriptivas con y sin atípicos
print("\nEstadísticas con atípicos:")
print(df['duracion_viaje'].describe())

print("\nEstadísticas sin atípicos:")
print(df[df['es_atipico'] == 0]['duracion_viaje'].describe())

# Análisis de la distribución de tipos de usuario
print("\nDistribución de tipos de usuario (todos los viajes):")
print(df['member_casual'].value_counts())

print("\nDistribución de tipos de usuario (viajes atípicos):")
print(df[df['es_atipico'] == 1]['member_casual'].value_counts())

duracion_promedio_mayo_todos = df[df['month'] == 'May']['duracion_viaje'].mean()
duracion_promedio_mayo_sin_atipicos = df[(df['month'] == 'May') & (df['es_atipico'] == 0)]['duracion_viaje'].mean()

print(f"\nDuración promedio en mayo (todos los viajes): {duracion_promedio_mayo_todos}")
print(f"Duración promedio en mayo (sin atípicos): {duracion_promedio_mayo_sin_atipicos}")

# Parte 10: Identificar datos atipicos viajes largos


# 1. Investigación Detallada de los Viajes Largos Atípicos:

# Filtra los viajes largos atípicos
viajes_largos_atipicos = df[df['es_atipico'] == 1]

# Patrones y Características:

# Distribución por día de la semana
print("\nDistribución de viajes largos atípicos por día de la semana:")
print(viajes_largos_atipicos['day_of_week'].value_counts())

# Distribución por hora del día
print("\nDistribución de viajes largos atípicos por hora del día:")
print(viajes_largos_atipicos['hour'].value_counts())

# Distribución por tipo de bicicleta
print("\nDistribución de viajes largos atípicos por tipo de bicicleta:")
print(viajes_largos_atipicos['rideable_type'].value_counts())

# Correlación con el mes
print("\nCorrelación de viajes largos atípicos por mes:")
print(viajes_largos_atipicos['month'].value_counts())

# Contexto Geográfico:

# Mapa de viajes largos atípicos (requiere columnas 'start_lat', 'start_lng', 'end_lat', 'end_lng')
# Nota: Este código básico solo muestra los puntos de inicio y fin. Se puede personalizar más.

plt.figure(figsize=(10, 8))
plt.scatter(viajes_largos_atipicos['start_lng'], viajes_largos_atipicos['start_lat'], label='Inicio', alpha=0.5)
plt.scatter(viajes_largos_atipicos['end_lng'], viajes_largos_atipicos['end_lat'], label='Fin', alpha=0.5)
plt.title('Mapa de Viajes Largos Atípicos')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend()
plt.show()

# Tipos de Usuarios:

# Distribución por tipo de usuario
print("\nDistribución de tipos de usuario en viajes largos atípicos:")
print(viajes_largos_atipicos['member_casual'].value_counts())

# 2. Análisis Comparativo:

# Viajes Largos Atípicos vs. Viajes Largos No Atípicos:

# Filtra los viajes largos no atípicos
viajes_largos_no_atipicos = df[(df['es_atipico'] == 0) & (df['duracion_viaje'] > df['duracion_viaje'].quantile(0.75))]

# Comparación de duración promedio
print("\nDuración promedio de viajes largos atípicos:")
print(viajes_largos_atipicos['duracion_viaje'].mean())
print("\nDuración promedio de viajes largos no atípicos:")
print(viajes_largos_no_atipicos['duracion_viaje'].mean())

# Comparación de tipos de usuario
print("\nDistribución de tipos de usuario en viajes largos atípicos:")
print(viajes_largos_atipicos['member_casual'].value_counts())
print("\nDistribución de tipos de usuario en viajes largos no atípicos:")
print(viajes_largos_no_atipicos['member_casual'].value_counts())

# Viajes Largos en Mayo vs. Otros Meses:

# Distribución de viajes largos atípicos por mes
print("\nDistribución de viajes largos atípicos por mes:")
print(viajes_largos_atipicos['month'].value_counts())

# 3. Posibles Explicaciones y Hipótesis: 
#Viajes Recreativos de Fin de Semana: La alta frecuencia de viajes largos atípicos los fines de semana y durante las horas de la tarde sugiere que podrían estar relacionados con actividades recreativas en parques, playas u otras atracciones turísticas.
#Turismo de Verano: La alta frecuencia de viajes largos atípicos durante los meses de verano sugiere que podrían estar relacionados con el turismo.
#Comportamiento Inusual de Usuarios Casuales: La alta proporción de usuarios casuales en los viajes largos atípicos sugiere que podrían estar utilizando las bicicletas para fines no convencionales, como explorar la ciudad o realizar viajes de larga distancia.
#Errores de Registro: Aunque es menos probable, aún es posible que algunos de los viajes largos atípicos sean errores de registro en la hora de inicio o finalización.
#Datos "Vacios" en la columna de tipo de usuario: Es importante investigar el por que hay tantos datos con el tipo de usuario vacio, esto podria generar información importante.

# Parte 11

# Filtra los viajes largos atípicos
viajes_largos_atipicos = df[df['es_atipico'] == 1]

# Estaciones de inicio más frecuentes
print("\nEstaciones de inicio más frecuentes de viajes largos atípicos:")
print(viajes_largos_atipicos['start_station_name'].value_counts().head(10))

# Estaciones de fin más frecuentes
print("\nEstaciones de fin más frecuentes de viajes largos atípicos:")
print(viajes_largos_atipicos['end_station_name'].value_counts().head(10))

 # Filtra los viajes largos atípicos
viajes_largos_atipicos = df[df['es_atipico'] == 1]

 # Función para calcular la distancia entre dos puntos geográficos (en kilómetros)
def calcular_distancia(lat1, lon1, lat2, lon2):
     R = 6371  # Radio de la Tierra en kilómetros
     dlat = np.radians(lat2 - lat1)
     dlon = np.radians(lon2 - lon1)
     a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
     distancia = R * c
     return distancia

 # Calcula la distancia recorrida para cada viaje largo atípico
viajes_largos_atipicos['distancia_km'] = viajes_largos_atipicos.apply(
     lambda row: calcular_distancia(row['start_lat'], row['start_lng'], row['end_lat'], row['end_lng']), axis=1)

 # Correlación entre distancia y duración del viaje
print("\nCorrelación entre distancia y duración del viaje largo atípico:")
print(viajes_largos_atipicos[['distancia_km', 'duracion_viaje']].corr())

# Filtra los viajes largos atípicos
viajes_largos_atipicos = df[df['es_atipico'] == 1]

# Distribución por hora del día para diferentes días de la semana
plt.figure(figsize=(12, 6))
sns.countplot(x='hour', hue='day_of_week', data=viajes_largos_atipicos)
plt.title('Distribución de viajes largos atípicos por hora del día y día de la semana')
plt.xlabel('Hora del día')
plt.ylabel('Cantidad de viajes')
plt.show()

# Distribución por hora del día para diferentes meses del año
plt.figure(figsize=(12, 6))
sns.countplot(x='hour', hue='month', data=viajes_largos_atipicos)
plt.title('Distribución de viajes largos atípicos por hora del día y mes del año')
plt.xlabel('Hora del día')
plt.ylabel('Cantidad de viajes')
plt.show()

# Filtra los registros con "Vacío" en la columna "member_casual"
registros_vacios = df[df['member_casual'] == 'Vacío']

# Análisis de los registros con "Vacío"
print("\nAnálisis de registros con 'Vacío' en la columna 'member_casual':")
print(registros_vacios.describe())

