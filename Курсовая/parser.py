import librosa
import numpy
import pandas
import os
import sklearn
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from src import configuration
from pydub import AudioSegment

INPUT_DIR = configuration.ConvertWav.CONVERT_DIRECTORY
OUTPUT_DIR = INPUT_DIR + "_wav" + "/"
FILE_TYPE = configuration.ConvertWav.FILE_TYPE

def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_sample_arrays(dataset_dir, folder_name, samp_rate):
    path_of_audios = librosa.util.find_files(dataset_dir + "/" + folder_name)
    audios = []
    for audio in path_of_audios:
        x, sr = librosa.load(audio, sr=samp_rate, duration=5.0)
        audios.append(x)
    audios_numpy = numpy.array(audios)
    return audios_numpy


def extract_features(signal, sample_rate, frame_size, hop_size):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size,
                                                            hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    return [

        numpy.mean(zero_crossing_rate),
        numpy.std(zero_crossing_rate),
        numpy.mean(spectral_centroid),
        numpy.std(spectral_centroid),
        numpy.mean(spectral_contrast),
        numpy.std(spectral_contrast),
        numpy.mean(spectral_bandwidth),
        numpy.std(spectral_bandwidth),
        numpy.mean(spectral_rolloff),
        numpy.std(spectral_rolloff),

        numpy.mean(mfccs[1, :]),
        numpy.std(mfccs[1, :]),
        numpy.mean(mfccs[2, :]),
        numpy.std(mfccs[2, :]),
        numpy.mean(mfccs[3, :]),
        numpy.std(mfccs[3, :]),
        numpy.mean(mfccs[4, :]),
        numpy.std(mfccs[4, :]),
        numpy.mean(mfccs[5, :]),
        numpy.std(mfccs[5, :]),
        numpy.mean(mfccs[6, :]),
        numpy.std(mfccs[6, :]),
        numpy.mean(mfccs[7, :]),
        numpy.std(mfccs[7, :]),
        numpy.mean(mfccs[8, :]),
        numpy.std(mfccs[8, :]),
        numpy.mean(mfccs[9, :]),
        numpy.std(mfccs[9, :]),
        numpy.mean(mfccs[10, :]),
        numpy.std(mfccs[10, :]),
        numpy.mean(mfccs[11, :]),
        numpy.std(mfccs[11, :]),
        numpy.mean(mfccs[12, :]),
        numpy.std(mfccs[12, :]),
        numpy.mean(mfccs[13, :]),
        numpy.std(mfccs[13, :]),
    ]

def createDataset():
    samp_rate = configuration.CreateDataset.SAMPLING_RATE
    frame_size = configuration.CreateDataset.FRAME_SIZE
    hop_size = configuration.CreateDataset.HOP_SIZE
    dataset_dir = configuration.CreateDataset.DATASET_DIRECTORY

    sub_folders = get_subdirectories(dataset_dir)

    labels = []
    is_created = False

    print("Извлечение пространства признаков из айдио-файлов...")
    for sub_folder in sub_folders:
        print(".....Работа в директории:", sub_folder)
        sample_arrays = get_sample_arrays(dataset_dir, sub_folder, samp_rate)
        for sample_array in sample_arrays:
            row = extract_features(sample_array, samp_rate, frame_size, hop_size)
            if not is_created:
                dataset_numpy = numpy.array(row)
                is_created = True
            elif is_created:
                dataset_numpy = numpy.vstack((dataset_numpy, row))

            labels.append(sub_folder)

    print("Нормализация данных...")
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    dataset_numpy = scaler.fit_transform(dataset_numpy)

    Feature_Names = ['meanZCR', 'stdZCR', 'meanSpecCentroid', 'stdSpecCentroid', 'meanSpecContrast', 'stdSpecContrast',
                     'meanSpecBandwidth', 'stdSpecBandwidth', 'meanSpecRollof', 'stdSpecRollof',
                     'meanMFCC_1', 'stdMFCC_1', 'meanMFCC_2', 'stdMFCC_2', 'meanMFCC_3', 'stdMFCC_3',
                     'meanMFCC_4', 'stdMFCC_4', 'meanMFCC_5', 'stdMFCC_5', 'meanMFCC_6', 'stdMFCC_6',
                     'meanMFCC_7', 'stdMFCC_7', 'meanMFCC_8', 'stdMFCC_8', 'meanMFCC_9', 'stdMFCC_9',
                     'meanMFCC_10', 'stdMFCC_10', 'meanMFCC_11', 'stdMFCC_11', 'meanMFCC_12', 'stdMFCC_12',
                     'meanMFCC_13', 'stdMFCC_13'
                     ]
    dataset_pandas = pandas.DataFrame(dataset_numpy, columns=Feature_Names)

    dataset_pandas["genre"] = labels
    dataset_pandas.to_csv("dataset.csv", index=False)
    print("Набор данных сформирован, записан и сохранен в текущей папке с кодом")

def modelTrain():
    data_set = pandas.read_csv('dataset.csv', index_col=False)
    data_set = numpy.array(data_set)
    print("Размер набора данных:", data_set.shape)
    number_of_rows, number_of_cols = data_set.shape

    data_x = data_set[:, :number_of_cols - 1]
    data_y = data_set[:, number_of_cols - 1]

    model = SVC(C=100, gamma=0.08)
    print("Обучение модели.....")
    model.fit(data_x, data_y)

    joblib.dump(model, 'model.pkl')
    print("Модель обучена и сохранена в текущей папке с кодом")


if __name__ == '__main__':
    createDataset()
    modelTrain()