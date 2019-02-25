import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import _pickle as pk


def write_metadata(log_dir, labels):

    with open(os.path.join(log_dir,'metadata.tsv'), 'w') as file:
        for label in labels:
            file.write(label + '\n')


def visualize_embeddings(models_dir, log_dir, metadata_file, model_file, nw, emb_size, indices_to_proj=None):
    # restore model
    dwe = tf.get_variable(name='dwe', shape=(nw, emb_size), dtype=tf.float32)  # disambiguated word embedding
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, os.path.join(models_dir, model_file))
        print('Model restored.')

        # configure the projector
        embeddings_writer = tf.summary.FileWriter(log_dir, sess.graph)
        config = projector.ProjectorConfig()
        config.model_checkpoint_path = os.path.join(models_dir, model_file) # 'test_model.ckpt'
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = dwe.name
        embedding_conf.metadata_path = os.path.join(log_dir, metadata_file) # 'metadata.tsv'
        projector.visualize_embeddings(embeddings_writer, config)


# model file should have .ckpt extension
def save_model(session, log_dir, file_name):

    saver = tf.train.Saver()
    saver.save(session, os.path.join(log_dir, file_name))


if __name__ == '__main__':
    log_dir = 'C:/Users/admin/PycharmProjects/tensorboard_output'
    models_dir = 'C:/Users/admin/PycharmProjects/models'
    # nw = 82831
    # emb_size = 100
    # visualize_embeddings(models_dir=models_dir, log_dir=log_dir, metadata_file='metadata.tsv',
    #                      model_file='test_model.ckpt', nw=nw, emb_size=emb_size,)

    # write metadata file

    # Path to data Byungkon Kang and Kyung-Ah.
    data_in = 'C:/Users/admin/PycharmProjects/dictionary_bootstrapping_Byungkon_Kyung/simple-data.pkl'
    with open(data_in, 'rb') as f:
        d = pk.load(f, encoding='latin1')

    write_metadata(log_dir, d['id2dw'])
