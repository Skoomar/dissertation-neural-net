import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def undo_one_hot_encoding(encoded):
    non_hot = np.array([], dtype=np.int)
    for i in range(len(encoded)):
        non_hot = np.append(non_hot, int(np.where(encoded[i] == 1)[0]))
    return non_hot


def decode_labels(encoded_y, label_encoder):
    """Reverse the one-hot encoding and numerical encoding of the original alphabetical identifiers for each activity
        class

    Parameters:
        encoded_y (numpy array): array of labels in shape (x, 18) as we have 18 classes
        label_encoder (scikit-learn LabelEncoder): the object that was used to encode this particular set of classes
                                                    during preprocessing

    Returns:
            decoded_y (numpy array): array of values
    """

    # if it's one-hot encoded then reverse it to get numerical identifiers for the target label
    if len(encoded_y.shape) == 2:
        non_hot_y = undo_one_hot_encoding(encoded_y)
    elif len(encoded_y.shape > 2):
        raise ValueError('Array of labels should only have up to 2 dimensions')

    # decode the numerical identifiers to the original alphabetical identifiers
    decoded_y = label_encoder.inverse_transform(non_hot_y)
    return decoded_y


def plot_confusion_matrix(title, true_y, pred_y, label_encoder):
    con_mat = tf.math.confusion_matrix(labels=true_y, predictions=pred_y).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    classes = label_encoder.classes_
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
