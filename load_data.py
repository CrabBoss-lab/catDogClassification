import pathlib
import tensorflow as tf


def Load_Data(data_dir, batch_size, input_height, input_width):
    """加载数据集"""
    # 数据集
    data_dir = pathlib.Path(data_dir)

    # 拆分数据集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(input_height, input_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(input_height, input_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


if __name__ == '__main__':
    Load_Data(data_dir='dataset/outputs', batch_size=32, input_height=180, input_width=180)
