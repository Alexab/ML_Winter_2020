# --- Конфигурация обратной конвертации --- #
class ConvertWav:
    # Путь до папки с музыкой
    CONVERT_DIRECTORY = "./test"

    # audio file format
    FILE_TYPE = "au"


# --- Конфигурация для создания набора данных в .csv,
# который будет использован для распознавания жанров --- #
class CreateDataset:
    # Путь к набору данных для обучения
    DATASET_DIRECTORY = "./train/"

    # Sampling rate (Hz)
    SAMPLING_RATE = 22050

    # Frame size (Samples)
    FRAME_SIZE = 2048

    # Hop Size (Samples)
    HOP_SIZE = 512


class Test:
    # путь к тестовой выборке
    TEST_DATA_PATH = "./test/"