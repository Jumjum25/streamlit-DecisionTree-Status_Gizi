import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib  # Tambahkan ini untuk menyimpan/memuat model dengan joblib

# Sidebar untuk navigasi halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Home", "Visualisasi Pohon Keputusan", "Evaluasi Model", "Prediksi Individu"])

# Judul aplikasi
st.title('Klasifikasi Status Gizi Anak dengan Decision Tree')

# Upload file dataset
uploaded_file = st.file_uploader("Upload file CSV dataset", type=["csv"])
if uploaded_file is not None:
    # Baca dataset
    data = pd.read_csv(uploaded_file)
    
    # Menampilkan dataset
    st.write("Dataset:")
    st.write(data.head())

    # Tampilkan nama kolom untuk memastikan
    st.write("Nama kolom dalam dataset:")
    st.write(data.columns.tolist())

    # Memilih fitur dan target
    st.write("Pilih fitur yang akan digunakan:")
    features = st.multiselect(
        "Fitur", 
        data.columns.tolist(),  
        default=['umur', 'berat_badan', 'tinggi_badan', 'zscore_bb_u']  
    )
    
    # Memilih target (label status gizi)
    target = st.selectbox(
        "Pilih Target Status Gizi", 
        [col for col in data.columns.tolist() if 'status_gizi' in col]
    )

    # Memisahkan fitur dan target
    X = data[features]
    y = data[target]

    # Menampilkan jumlah data target yang digunakan
    st.write("Jumlah data untuk masing-masing kategori dalam target:")
    st.write(y.value_counts())

    # Jika target bertipe kategorikal, ubah menjadi numerik
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = le.classes_
        st.write("Mapping kategori ke numerik:")
        st.write(dict(zip(le.classes_, le.transform(le.classes_))))
    elif y.dtype in ['int64', 'float64']:
        st.write("Target berupa data kontinu. Mengubahnya menjadi kategori.")
        bins = st.slider("Pilih jumlah kategori", 2, 10, 4)  
        y = pd.cut(y, bins=bins, labels=False)
        class_names = [f'Kategori {i}' for i in range(bins)]
        st.write(f"Target telah diubah menjadi {str(bins)} kategori.")

    # Memilih metode normalisasi
    st.write("Pilih metode normalisasi:")
    normalization_method = st.radio("Metode Normalisasi", ('None', 'StandardScaler', 'MinMaxScaler'))

    # Normalisasi data jika dipilih
    if normalization_method == 'StandardScaler':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif normalization_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Membagi data menjadi data latih dan data uji
    test_size = st.slider("Proporsi data uji (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Menampilkan jumlah data latih dan uji
    st.write(f"Jumlah data latih: {len(X_train)}")
    st.write(f"Jumlah data uji: {len(X_test)}")

    # Visualisasi Pembagian Data Latih dan Uji
    st.subheader('Visualisasi Pembagian Data')
    data_sizes = [len(X_train), len(X_test)]
    labels = ['Data Latih', 'Data Uji']
    fig, ax = plt.subplots()
    ax.pie(data_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Inisialisasi model Decision Tree
    model = DecisionTreeClassifier()

    # Melatih model
    model.fit(X_train, y_train)

    # Menyimpan model dengan joblib
    model_file_joblib = "decision_tree_model_joblib.pkl"
    joblib.dump(model, model_file_joblib)
    
    # Tampilkan tombol untuk mengunduh model
    st.download_button(label="Unduh Model (Joblib)", data=open(model_file_joblib, 'rb'), file_name=model_file_joblib)

    if page == "Home":
        st.write("Silakan pilih halaman di sidebar untuk memulai.")

    elif page == "Visualisasi Pohon Keputusan":
        st.subheader('Visualisasi Pohon Keputusan')
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, feature_names=features, class_names=class_names, filled=True, ax=ax)
        st.pyplot(fig)

    elif page == "Evaluasi Model":
        st.subheader('Evaluasi Model')
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write(f"Akurasi: {accuracy}")
        st.write(f"Presisi: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1-Score: {f1}")
        st.write("Confusion Matrix:")

        # Visualisasi confusion matrix untuk 3 target
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

    elif page == "Prediksi Individu":
        st.subheader('Prediksi Status Gizi untuk Data Individu')

        # Opsi untuk memuat model dari file joblib
        uploaded_model_file = st.file_uploader("Upload Model (Joblib)", type=["pkl"])
        if uploaded_model_file is not None:
            model = joblib.load(uploaded_model_file)
            st.success("Model berhasil dimuat!")
        
        # Input fitur individu
        input_data = {}
        for feature in features:
            input_data[feature] = st.number_input(f'Masukkan nilai untuk {feature}', value=0.0)
        
        # Prediksi setelah menekan tombol
        if st.button('Prediksi'):
            # Ubah input data menjadi DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Normalisasi input data jika diperlukan
            if normalization_method == 'StandardScaler':
                input_df = scaler.transform(input_df)
            elif normalization_method == 'MinMaxScaler':
                input_df = scaler.transform(input_df)
            
            # Prediksi menggunakan model
            prediksi = model.predict(input_df)
            
            # Tampilkan hasil prediksi
            if class_names is not None:
                st.write(f"Hasil Prediksi: {class_names[prediksi[0]]}")
            else:
                st.write(f"Hasil Prediksi: {prediksi[0]} (kategori ke-{prediksi[0]})")
        else:
            st.write("Silakan upload dataset dan melatih model terlebih dahulu.")

# Informasi tambahan
st.write("**Note:** Pastikan dataset yang Anda upload memiliki kolom yang sesuai dengan pilihan fitur dan target.")