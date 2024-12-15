import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Veri yükleme ve kontroller

data = pd.read_csv("co2.csv")
print(data.head())
print("*" * 45)
print(f"Veri setinin boyutu: {data.shape}")
print("*" * 45)
print(data.describe())
print("*" * 45)
print("Eksik Veri Sayıları:")
print(data.isnull().sum())
print("*" * 45)

# Eksik veri oranlarını görmek için:
print("Eksik Veri Oranları (%):")
print((data.isnull().mean() * 100).round(2))

# İlk 15 markayı al, diğerlerini 'Diğer' olarak topla
marka_15 = data['Make'].value_counts().head(15)
diger_markalar = data['Make'].value_counts().iloc[15:].sum()

top_10_makes = pd.concat([marka_15, pd.Series({'Diğer': diger_markalar})])
plt.figure(figsize=(10, 8))
plt.pie(top_10_makes, labels=top_10_makes.index, autopct='%1.1f%%', startangle=90)
plt.title('Araba Markaları')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Engine Size(L)'], kde=True, bins=30, color='skyblue', alpha=0.7, label='Engine Size (L)')
plt.title('Motor Hacmi Dağılımı', fontsize=14)
plt.xlabel('Motor Hacmi (L)', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# CO2 Emisyon Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data['CO2 Emissions(g/km)'], kde=True, bins=30, color='lightgreen', alpha=0.7, label='CO2 Emissions (g/km)')
plt.title('CO2 Emisyon Dağılımı', fontsize=14)
plt.xlabel('CO2 Emisyonu (g/km)', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(x='Fuel Type', y='CO2 Emissions(g/km)', data=data, hue='Fuel Type', palette="Set2", legend=False)
plt.title("Yakıt Türü ve CO2 Emisyonları Dağılımı(g/km)", fontsize=14)
plt.xlabel("Fuel Type")
plt.ylabel("Ortalama CO2 Emissions (g/km)")
plt.show()

arac_markalari = data.groupby('Make')['CO2 Emissions(g/km)'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
arac_markalari.plot(kind='bar', color='orange')
plt.title("En Yüksek Ortalama CO2 Emisyonuna(g/km) Sahip İlk 10 Marka", fontsize=14)
plt.xlabel("Make")
plt.ylabel("Average CO2 Emissions (g/km)")
plt.xticks(rotation=45)
plt.show()

arac_sinifi = data.groupby('Vehicle Class')['CO2 Emissions(g/km)'].mean().sort_values()
plt.figure(figsize=(12, 6))
arac_sinifi.plot(kind='barh', color='skyblue')
plt.title("Araç Sınıfı ve Ortalama CO2 Emisyonları(g/km)", fontsize=12)
plt.xlabel("Average CO2 Emissions (g/km)")
plt.ylabel("Vehicle Class")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Engine Size(L)', y='CO2 Emissions(g/km)', hue='Fuel Type', data=data, palette='Set1')
plt.title("Motor Boyutu ve CO2 Emisyonları(g/km)", fontsize=14)
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend(title="Fuel Type")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Fuel Consumption City (L/100 km)', y='CO2 Emissions(g/km)', hue='Fuel Type', data=data, palette='coolwarm')
plt.title("Şehir Yakıt Tüketimi(L/100 km) ve CO2 Emisyonları(g/km)", fontsize=14)
plt.xlabel("Fuel Consumption City (L/100 km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend(title="Fuel Type")
plt.show()

fuel_make_co2 = data.pivot_table(values='CO2 Emissions(g/km)', index='Make', columns='Fuel Type', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(fuel_make_co2, cmap='coolwarm', annot=True, fmt='.1f')
plt.title("Marka ve Yakıt Türüne Göre Ortalama CO2 Emisyonları(g/km)", fontsize=14)
plt.xlabel("Fuel Type")
plt.ylabel("Make")
plt.show()

veri_arac_marka = pd.crosstab(data['Make'], data['Vehicle Class'])
veri_arac_marka.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Marka ve Araç Sınıfı İlişkisi')
plt.xlabel('Make')
plt.ylabel('Count')
plt.show()

sutunlar = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)', 'CO2 Emissions(g/km)']
kolerasyon = data[sutunlar].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(kolerasyon, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Continuous Variables", fontsize=14)
plt.show()

vehicle_fuel = pd.crosstab(data['Vehicle Class'], data['Fuel Type'])
vehicle_fuel.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title("Araç Sınıfı ve Yakıt Türü", fontsize=14)
plt.xlabel("Vehicle Class")
plt.ylabel("Count")
plt.legend(title="Fuel Type")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Engine Size(L)', y='CO2 Emissions(g/km)', hue='Fuel Type', style='Fuel Type', data=data, palette='tab10')
plt.title("Yakıt Tipine Göre Motor Boyutu(L) ve CO2 Emisyonları(g/km)", fontsize=14)
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend(title="Fuel Type")
plt.show()

araclarin_ortalama_emisyonlari = data.groupby('Make')['CO2 Emissions(g/km)'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
araclarin_ortalama_emisyonlari.plot(kind='bar', color='orange')
plt.title("En Yüksek Ortalama CO2 Emisyonuna(g/km) Sahip İlk 10 Marka", fontsize=14)
plt.xlabel("Make")
plt.ylabel("Average CO2 Emissions (g/km)")
plt.xticks(rotation=45)
plt.show()

scaler = MinMaxScaler()
data[['Engine Size(L)', 'CO2 Emissions(g/km)']] = scaler.fit_transform(data[['Engine Size(L)', 'CO2 Emissions(g/km)']])
# MinMaxScaler ile 'Engine Size(L)' ve 'CO2 Emissions(g/km)' sütunlarını 0 ile 1 arasında normalleştiriyoruz.

plt.figure(figsize=(12, 6))
sns.countplot(x='Fuel Type', data=data, hue='Vehicle Class')
plt.title('Araç Sınıfı ve Yakıt Türü Arasındaki İlişki')
plt.show()

# Veri setini bölme
X = pd.get_dummies(data.drop(['CO2 Emissions(g/km)'], axis=1), drop_first=True)
y = data['CO2 Emissions(g/km)']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model_egitimi(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Performans metriklerini hesapla
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Performansı:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")
    print("-" * 40)

    return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'R²': r2}

# Kullanılacak modellerin listesi
models = [
    (LinearRegression(), "Linear Regression"),
    (RandomForestRegressor(random_state=42), "Random Forest Regressor"),
    (GradientBoostingRegressor(random_state=42), "Gradient Boosting Regressor"),
    (MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42), "Neural Networks (MLP)")
]

results = []

for model, name in models:
    result = model_egitimi(model, X_train, X_test, y_train, y_test, name)
    results.append(result)

results_df = pd.DataFrame(results)

# MAE Karşılaştırması
plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['MAE'], color='skyblue')
plt.title("Model MAE Karşılaştırması")
plt.ylabel("MAE")
plt.xticks(rotation=45)
plt.show()

# R² Karşılaştırması
plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['R²'], color='lightgreen')
plt.title("Model R² Karşılaştırması")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.show()
