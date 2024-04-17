import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import joblib
import sklearn
print(sklearn.__version__)


@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")
    return df

# Fungsi untuk menampilkan halaman utama
def show_about():
    st.title("Red Wine Quality")
    st.write("""
        Welcome to the homepage!

# Data Understanding

## Business Understanding

### Business objective
jadi tujuan bisnis dari dataset ini adalah untuk menganalisis dataset yang dapat membantu produsen red wine untuk memahami faktor-faktor apa saja yang berkontribusi terhadap kualitas red wine sehingga dapat membuat perubahan yang di perlukan untuk meningkatkan kualitas produk.

### Assess situation
Situasi bisnis yang mendasari analisis ini adalah banyaknya para kompetitor yang bersaing dalam meningkatkan pasar dan dalam meningkatkan kualitas yang lebih baik untuk menarik minat konsumen.

### Tujuan data mining
Tujuan data mining dari dataset red wine quality adalah untuk mengeksplorasi dan mengekstrak pola atau informasi berharga yang tersembunyi dalam data tersebut, dan juga untuk mengoptimasi produksi yaitu menggunakan data untuk mengoptimalkan parameter produksi guna meningkatkan kualitas dan mengurangi variasi dalam hasil red wine, lalu untuk mengidentifikasi tren pasar yaitu untuk menjelajahi data untuk mengidentifikasi tren pasar dan preferensi konsumen yang dapat memberikan keunggulan kompetitif.

### Rencana proyek
Jadi tahapan tahapan yang dilakukan adalah

1. melakukan analisis eksploratif data yaitu melakukan EDA pada dataset red wine quality untuk memahami distribusi fitur, korelasi antar variabel, dll.
2. identifikasi faktor kualitas utama yaitu menggunakan teknik data mining untuk menentukan faktor-faktor kunci yang memengaruhi kualitas red wine.
3. segmentasi pasar yaitu mengidentifikasi segmen pasar yang memiliki preferensi kualitas tertentu.
4. optimasi produk yaitu menggunakan hasil analisis data untuk mengoptimalkan parameter-produksi guna mencapai konsistensi kualitas dan mengurangi variasi hasil anggur.
5. Strategi Pemasaran: Berdasarkan hasil analisis data, susun strategi pemasaran yang dapat menonjolkan keunggulan produk dan menarik segmen pasar yang diidentifikasi.
    """)
    # Path ke file CSV
    file_path = 'DataCleaned.csv'

    # Periksa keberadaan file
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di lokasi yang benar.")
        st.stop()

    # Muat data CSV
    try:
        df = pd.read_csv(file_path)
        st.write(df)  # Tampilkan data jika berhasil dimuat
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")

# Fungsi untuk menampilkan halaman tentang
def show_Distribusi(df):
# Judul dan deskripsi
    st.title("Distribusi Nilai")
    
    st.title("Visualisasi Data Mining menggunakan Streamlit")
    
    st.write("visualisasi diatas menunjukkan distribusi jumlah observasi untuk setiap tingkat kualitas anggur yang ada didalam dataset ini dan dapat dilihat bahwa kualitas anggur banyak yang berada pada nilai 5 dan 6, dapat diasumsi kan bahwa sebagian besar anggur dalam dataset ini memiliki kualitas yang cukup baik, dan sementara untuk kulaitas yang lebih rendah sangat jarang ditemui.")

    st.subheader("Distribusi Alkohol Anggur")
    fig, ax = plt.subplots(figsize=(8, 6))  # Mengubah ukuran figur
    sns.histplot(x='quality', data=df, kde=True, color='royalblue')  # Mengganti kolom 'alcohol' dengan 'quality'
    plt.title("Distribusi Nilai Kualitas Anggur")  # Mengubah judul
    plt.xlabel("Kualitas")  # Mengubah label sumbu x
    plt.ylabel("Jumlah")  # Memastikan label sumbu y tetap relevan
    st.pyplot(fig)  # Menampilkan plot di Streamlit


def show_hubungan(df):
    st.title("Hubungan Nilai")
    st.write("Menu ini menampilkan hubungan antara Fixed acidity dan volatile acidity.")

    # Memilih Plot yang Diminta
    st.write("Scatter plot ini menunjukkan bahwa terdapat hubungan antara nilai keasaman tetap dan nilai keasaman volatil pada anggur. Anggur dengan keasaman tetap tinggi cenderung memiliki keasaman volatil yang lebih tinggi pula.")
    st.subheader("Scatter Plot: Fixed Acidity vs Volatile Acidity")
    fig, ax = plt.subplots()
    sns.scatterplot(x='fixed acidity', y='volatile acidity', data=df.head(200))  # Menggunakan df.head(10)
    ax.set_xlabel("Fixed Acidity")
    ax.set_ylabel("Volatile Acidity")
    st.pyplot(fig)

    st.title("Korelasi")
    st.write("gambar tersebut menunjukkan hubungan negatif yang lemah antara fixed acidity dan volatile acidity. Hal ini dapat dijelaskan dengan beberapa kemungkinan, seperti jenis bahan baku, proses fermentasi, dan kerusakan produk.")
    df_file_corr = df.corr(numeric_only=True)

    # Ambil hanya 10 kolom dan baris pertama
    # df_file_corr_subset = df_file_corr.iloc[:10, :10]

    # Buat heatmap menggunakan seaborn
    # Visualisasi matriks korelasi menggunakan heatmap
    st.subheader("Heatmap Korelasi Antara Fixed Acidity dan Volatile Acidity")
    correlation_matrix = df[['fixed acidity', 'volatile acidity']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Korelasi Antara Fixed Acidity dan Volatile Acidity")
    st.pyplot(fig)

    # Tambahkan teks penjelasan
    text = 'Tabel korelasi diatas menunjukkan bahwa terdapat hubungan yang signifikan antara beberapa variabel. Perusahaan dapat menggunakan informasi ini untuk membuat keputusan yang lebih baik tentang produk/layanan perusahaan, strategi marketing, dan program loyalitas pelanggan.'
    st.markdown(text)
    

# Memuat data
df = load_data()

# Menampilkan hubungan
# show_relationship(df)


# Fungsi untuk menampilkan halaman perbandingan

def show_Perbandingan(df):
    st.title("Perbandingan")
    st.write("""
        Kedua grafik menunjukkan bahwa tingkat SO2 di udara meningkat seiring dengan peningkatan konsentrasi klorida. Hal ini menunjukkan bahwa terdapat hubungan positif antara tingkat SO2 dan konsentrasi klorida.

Peningkatan tingkat SO2 di udara dapat berdampak negatif pada kesehatan manusia dan lingkungan. SO2 dapat menyebabkan iritasi pada saluran pernapasan, penyakit pernapasan, dan masalah jantung. SO2 juga dapat bereaksi dengan bahan kimia lain di atmosfer untuk menghasilkan partikel halus yang dapat menyebabkan kabut asap dan masalah kesehatan lainnya.
    """)
    
    # Membuat subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot pertama: Chlorides
    df['chlorides'].head(10).value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Chlorides', fontweight="bold", size=10)

    # Plot kedua: Free sulfur dioxide
    df['free sulfur dioxide'].head(10).value_counts().plot(kind='bar', ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Free sulfur dioxide', fontweight="bold", size=20)

    # Plot ketiga: Total sulfur dioxide
    df['total sulfur dioxide'].head(10).value_counts().plot(kind='bar', ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Total sulfur dioxide', fontsize=20)

    # Plot keempat: Density
    df['density'].head(10).value_counts().plot(kind='bar', ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Density')

    # Menampilkan plot di Streamlit
    st.pyplot(fig)
# Fungsi untuk menampilkan halaman komposisi
def show_Komposisi(df):
    st.title("Kerentangan pH dalam Redwine")
    st.write("""
       Visualisasi di atas menunjukkan persentase rentang pH redwine. Sebagian besar redwine memiliki pH antara 3 dan 3,5. pH redwine dapat dipengaruhi oleh berbagai faktor, termasuk jenis anggur, proses fermentasi, dan penuaan. pH hanyalah salah satu faktor yang memengaruhi rasa redwine.
    """)

    # Kategorisasi pH
    bins = [0, 2.5, 3, 3.5, float('inf')]
    labels = ['<2.5', '2.5-3', '3-3.5', '>3.5']
    categorized_pH = pd.cut(df['pH'], bins=bins, labels=labels)

    # Menghitung jumlah kemunculan setiap kategori
    categorized_value_counts = categorized_pH.value_counts()

    # Plot bar untuk menunjukkan kerentangan pH
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categorized_value_counts.index, y=categorized_value_counts.values, palette='coolwarm')
    plt.xlabel('Rentang pH')
    plt.ylabel('Jumlah')
    plt.title('Kerentangan pH dalam Redwine')
    plt.xticks(rotation=45)
    st.pyplot(plt)


def predict_cancellation(df):
    st.title("Prediksi Kualitas Anggur")
    st.write("Gunakan fitur-fitur berikut untuk memprediksi apakah anggur buruk, baik, sangat baik:")

    # Select features
    feature_columns = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
        'density', 'pH', 'sulphates', 'alcohol'
    ]

    selected_features = {}
    for feature in feature_columns:
        selected_features[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", sorted(df[feature].unique()))

    data = pd.DataFrame(selected_features, index=[0])

    # Ensure all columns in data are numeric
    data = data.astype(float)

    # Button for prediction
    button = st.button('Prediksi')
    if button:
        try:
            loaded_model = joblib.load('model.pkl')
            predicted = loaded_model.predict(data)
            print(predicted)
            if predicted[0] == 1:
                st.write('buruk')
            elif predicted[0] == 2:
                st.write("baik")
            elif predicted[0] == 3:
                st.write("sangat baik")
            else:
                st.write('Dibatalkan')
        except FileNotFoundError:
            st.write("Model tidak ditemukan. Silakan pastikan bahwa model sudah tersedia.")


# Assuming df is your DataFrame
# df = pd.read_csv('hotel_data.csv')  # Load your data here
# predict_cancellation(df)

# Memuat data
df = load_data()

# Mengatur sidebar
df2 = pd.read_csv('DataCleaned.csv')
nav_options = {
    "About": show_about,
    "Distribution": lambda: show_Distribusi(df),
    "Relations": lambda: show_hubungan(df),
    "Comparison": lambda: show_Perbandingan(df),
    "Composition": lambda: show_Komposisi(df),
    "Predict": lambda: predict_cancellation(df2)
}

# Menampilkan sidebar
st.sidebar.title("Redwine Quality")
selected_page = st.sidebar.radio("Menu", list(nav_options.keys()))

# Menampilkan halaman yang dipilih
nav_options[selected_page]()
