import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def entrenar():
    # --- CORRECCIÓN DE RUTA ---
    # Al estar en la misma carpeta, solo necesitamos el nombre del archivo
    ruta = 'bank-full.csv' 
    
    if not os.path.exists(ruta):
        print(f"❌ Error: No se encuentra el archivo '{ruta}' en la carpeta actual.")
        print(f"Asegúrate de que el CSV esté dentro de la carpeta 'momento3'.")
        return

    print("Reading data...")
    # El archivo original suele usar ';' como separador
    df = pd.read_csv(ruta, sep=';')
    
    # 1. Preprocesamiento de la variable objetivo
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['y'])
    joblib.dump(le, 'LabelEncoder.pkl') 

    # 2. Preparación de características (X) y objetivo (y)
    x = df.drop('y', axis=1)
    y = df['y']
    
    # Convertir variables categóricas a numéricas
    x = pd.get_dummies(x, drop_first=True)
    columnas = x.columns.tolist()
    joblib.dump(columnas, 'columnas_modelo.pkl')

    # 3. División y Escalado
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    joblib.dump(sc, 'escalador.pkl')

    # 4. Entrenamiento de Modelos
    print("Training Logistic Regression...")
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    joblib.dump(log, 'modelo_logistico.pkl')

    print("Training Neural Network...")
    ann = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500, random_state=42)
    ann.fit(X_train, y_train)
    joblib.dump(ann, 'red_neuronal.pkl')

    # 5. Verificación Final
    y_pred = ann.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ ¡Entrenamiento exitoso!")
    print(f"Exactitud del modelo: {acc:.4f}")
    print("Los archivos .pkl han sido generados en la carpeta 'momento3'.")

if __name__ == "__main__":
    entrenar()