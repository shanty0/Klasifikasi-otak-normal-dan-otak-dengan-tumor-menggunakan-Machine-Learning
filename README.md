# Klasifikasi-otak-normal-dan-otak-dengan-tumor-menggunakan-Machine-Learning
Dilakukan training pada model untuk memprediksi hasil klasifikasi citra otak normal dan otak bertumor dengan convolutional neural network(CNN). 

## Dataset
Digunakan dataset yang mencangkup 3762 citra otak. Masing-masing citra memiliki dataset 5 feature order pertama, 8 feature tekstur dan label 
(Tumor atau non-Tumor) yang terdata dalam file .csv. Dataset dapat diakses [disini](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor). Citra dibagi 
menjadi 2633 citra untuk training, 565 citra untuk validation, dan 564 citra untuk testing.

## Metode
### Feature learning
Dilakukan operasi konvolusi antara filters dan input image untuk mengekstraksi features dalam bentuk feature maps. Feature maps yang dihasilkan ditumpuk menjadi matriks
3 dimensi, dengan panjang dan lebar matriks sama dengan panjang dan lebar citra dan kedalaman matriks sama dengan jumlah feature map yang dihasilkan. Non-linearitas
diimplementasikan pada model dengan activation function ReLU (Rectifier linear units) sehingga weighted inputs yang negative dibuat 0. Kemudian dilakukan pooling dengan 
max pooling. Dengan mengambil setiap nilai maximum dari patch yang digeser di feature map, max pooling mengurangi dimensi feature map dan membuat representasi yang invariant
terhadap perubahan kecil pada gambar (membuat feature spasial lebih compact). 
Proses ini ditumpuk menjadi 3 layer sebagai segmen dari feature learning. 

![image](https://user-images.githubusercontent.com/110709194/183238021-b4920a55-03fd-4f0d-9c2d-cc56b3e8b375.png)

### Classification
Hasil matrix 3 dimensi dari max pooling di flattened menjadi vektor 1 dimensi. Vektor ini kemudian menjadi input ke fully connected neural network dimana setiap neuron 
terkoneksi dengan neuron di layer berikutnya. Activation function yang digunakan adalah ReLu yang performanya lebih baik dan mempercepat training. Dengan weighted 
feature, model berusaha memprediksi hasil klasfikasi. Hasil prediksi kemudian dibandingkan dengan label training data. Model pada saat itu kemudian dijalankan pada 
validation data set. Error dari hasil prediksi training tersebut direpresentasikan dengan loss function (used function : Binary crossentropy).Untuk membuat hasil loss function sekecil mungkin, weights diubah dengan step size/learning rate tertentu kearah dimana hasil loss function lebih rendah (used algorithm :stochastic gradient descent) dan diterapkan ke model dengan backpropogation. Hal ini dilakukan secara iteratif hingga hasil loss function minimum.

### Architecture
Citra input dengan ukuran (224,224,3) dimasukan pada layer berikut :
1. Konvolusi dengan 16 filter, ukuran filter =(3,3), langkah/stride = 2 
2. Fungsi Aktifasi ReLu
3. Konvolusi dengan 16 filter, ukuran filter =(3,3), langkah/stride = 2
4. Fungsi Aktifasi ReLu
5. Max pooling, ukuran=(2,2)
6. Konvolusi dengan 32 filter, ukuran filter =(3,3), langkah/stride = 2 
7. Fungsi Aktifasi ReLu
8. Konvolusi dengan 32 filter, ukuran filter =(3,3), langkah/stride = 2
9. Fungsi Aktifasi ReLu
10. Max pooling, ukuran=(2,2)
11. Konvolusi dengan 64 filter, ukuran filter =(3,3), langkah/stride = 2 
12. Fungsi Aktifasi ReLu
13. Konvolusi dengan 64 filter, ukuran filter =(3,3), langkah/stride = 2
14. Fungsi Aktifasi ReLu
15. Max pooling, ukuran=(2,2)
16. Flatten untuk membuat matriks 3 dimenso menjadi vektor 1 dimensi
17. fully connected neural network 
18. neuron dengan sigmoid activation function (ranges from 0-1, applicable for binary classification)

## Result
![image](https://user-images.githubusercontent.com/110709194/183245257-d5560f58-b2cd-4ad5-99f4-6e4267125068.png)

Training loss menunjukan kinerja model dalam menghasilkan prediksi pada label training data dengan benar (loss function over time). 
Validation loss menunjukan kinerja model dalam menghasilkan prediksi pada label data baru/data yang tidak digunakan untuk training/validation data dengan benar (loss functoiion over time).
Training accuracy mengkomparasi hasil prediksi model dengan label data training dalam persentase.
Validation accuracy mengkomparasi hasil prediksi model dengan label data validasi dalam persentase.
comparing the model predictions with the true values in terms of percentage.

Terlihat validation loss diverge dari training loss disekitar iterasi ke-25. Terlihat validation accuracy diverge dari training accuracy disekitar iterasi ke-25.
hal ini menunjukan overfitting , karena setelah sekitar iterasi ke-25 model sudah tidak cukup general untuk mengola data diluar dataset training dengan baik.

Terlihat adanya loss spike pada loss validation. Hal ini umum ditemukan dalam neural network berkinerja rendah yang dilatih dengan stochastic gradient descent(SGD).
gradients yang backpropogated dari loss yang tinggi dapat menggangu distribusi trainable parameter dan destabilize learning. Loss spike biasanya terjadi untuk batch size yang kecil,
high order loss function dan learning rates tinggi yang tidak stabil [1].

[1]J. Ede and R. Beanland, "Adaptive learning rate clipping stabilizes learning", Machine Learning: Science and Technology, vol. 1, no. 1, p. 015011, 2020. Available: 10.1088/2632-2153/ab81e2.
