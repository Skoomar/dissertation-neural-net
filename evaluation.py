import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(subject_id, true_y, pred_y, label_encoder):
    con_mat = tf.math.confusion_matrix(labels=true_y, predictions=pred_y).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    classes = label_encoder.classes_
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.title('Subject ' + subject_id + ' Confusion Matrix')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
