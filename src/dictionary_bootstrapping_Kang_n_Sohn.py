import tensorflow as tf
import _pickle as pk
import numpy as np
from src.helper_funcs import save_model, visualize_embeddings
from tensorflow.contrib.tensorboard.plugins import projector
import os
import sys


# Path to input data.
data_in = 'C:/Users/admin/PycharmProjects/dictionary_bootstrapping_Byungkon_Kyung/simple-data.pkl'
with open(data_in, 'rb') as f:
    d = pk.load(f, encoding='latin1')

# Path to tensorboard output files
log_dir = 'C:/Users/admin/PycharmProjects/tensorboard_output'
models_dir = 'C:/Users/admin/PycharmProjects/models'

# Path to embeddings initializations
# init_embedd_dir = 'C:/Users/admin/PycharmProjects/NLP/embeddings_initialization_dim_200.npy'

# uncomment the following line if you want to use pre-trained embeddings
# embeddings_init = np.load(init_embedd_dir)

bs = 64  # mini-batch size
td = 150 # embedding dimension
hd = 300 # hidden dimension

nw, mw, ms = d['def'].shape  # total number of words, max num. of words per definition, max num. of senses per word
# nw = len(d['id2dw'])
params = {}                  # hash to hold the trainable parameters
tau = 10
# We need the following 3 lines to account for the discrepancy between
# the true number of words (true_nw) vs. the number of words w/ IDs (nw).
# This discrepancy exists because of some designs choices that have not been modified.
true_nw = nw
maxid = np.max(d['def'])
if maxid >= nw:
    nw = maxid + 1
"""
    Input placeholders and constant values
"""
# placeholders
df = tf.placeholder(name='def', dtype=tf.int32, shape=[None, mw, ms])  # Takes values from d['def']
dm = tf.placeholder(name='dmask', dtype=tf.float32, shape=[None, mw, ms])  # Takes values from d['dmask']
wmask = tf.placeholder(name='wmask', dtype=tf.float32, shape=[None, mw])  # Takes values from d['wmask']
h_d = tf.placeholder(name='idf', dtype=tf.float32, shape=[None, mw]) # Takes values from d['idf']
wi = tf.placeholder(name='wi', dtype=tf.int32)    # batch of word indices
# nwi = tf.placeholder(name='nwi', dtype=tf.int32)  # batch of word indices (negative samples- if Hinge Loss is used)
pr = tf.placeholder(name='sprior', dtype=tf.float32, shape=[None, mw, ms])
lr = tf.placeholder(name='lr', dtype=tf.float64)  # learning rate
beta = tf.placeholder(name='beta', dtype=tf.float32) # beta for fixed point iteration

# constants
# NO constants for now

"""
    Non-trainable parameters
"""
params['dwe'] = tf.get_variable(name='dwe',
                                shape=(nw, td),
                                dtype=tf.float32,
                                initializer=tf.initializers.random_uniform(minval=-0.1, maxval=+0.1),
                                trainable=False)  # disambiguated word embedding shape=(nw, td)

"""
    Trainable parameters
"""
# params['L'] = tf.get_variable('L', shape=(td, ), dtype=tf.float32, initializer=tf.initializers.random_uniform(minval=-0.1, maxval=+0.1))  # td: diagonal entries only (of the td x td matrix)
params['L1'] = tf.get_variable('L1', shape=(td, hd), dtype=tf.float32, initializer=tf.initializers.random_uniform(minval=-0.1, maxval=+0.1))  # td x hd
params['L2'] = tf.get_variable('L2', shape=(hd, td), dtype=tf.float32, initializer=tf.initializers.random_uniform(minval=-0.1, maxval=+0.1))  # hd x td
# params['b1'] = tf.get_variable('bias_1', shape=(hd,), dtype=tf.float32, initializer=tf.initializers.random_uniform(minval=-0.1, maxval=+0.1))
# params['b2'] = tf.get_variable('bias_2', shape=(td,), dtype=tf.float32, initializer=tf.initializers.random_uniform(minval=-0.1, maxval=+0.1))

"""
    Word vectors gathering
"""
with tf.name_scope('Gather_WEs'):
    # ndw = tf.gather(params['dwe'], nwi, name='neg_samp')  # bs x td (negative samples)
    pdw = tf.gather(params['dwe'], wi, name='pos_samp')  # bs x td  (positive samples)

"""
    Expanding tensors to help with operations below  
"""
with tf.name_scope('Expand_dims'):
    wm = tf.expand_dims(wmask, axis=2)  # bs x mw x 1
    idf = tf.expand_dims(h_d, axis=2)  # bs x mw x 1


"""
    Return alpha coefficients
"""
with tf.name_scope('Alpha_coeffs'):

    def calc_alphas(_, args_):
        """
            Returns: alpha coefficients of size mw x ms.
            d: indices of the senses that comprise the definition (mw x ms)
            m: mask of size mw x ms
        """
        d, m = args_

        senses = tf.gather(params['dwe'], d)  # mw x ms x td
        senses_norm = tf.nn.l2_normalize(senses, axis=2) # mw x ms x td, normalized vectors
        cos_sim = tf.reduce_sum(tf.reshape(senses_norm, [mw, ms, 1, 1, td]) * senses_norm, axis=4)  # mw x ms x mw x ms

        # sum the similarities
        logits = tf.reduce_sum(cos_sim, axis=3) # mw x ms x mw

        # calculate the tau / |s(d_m)|
        cnt = tf.reshape(tf.reduce_sum(m, axis=1), [1, 1, mw], name='cnt')  # 1 x 1 x mw (In other words, |s(d_m)|)
        logits = logits / cnt  # mw x ms x mw

        # take rid of the NaN values generated from the above operation
        logits = tf.where(tf.is_nan(logits), tf.zeros_like(logits), logits)  # mw x ms x mw

        # exponentiate and calculate product of context words probs
        logits = tf.exp(tau * logits)  # mw x ms x mw
        logits = tf.reduce_prod(logits, axis=2)  # mw x ms

        # here smooth the average of all senses
        sm = tf.reduce_sum(logits * m, axis=1, keepdims=True)  # mw x 1
        logits = (logits * m) / sm   # mw x ms
        logits = tf.where(tf.logical_or(tf.is_nan(logits), tf.is_inf(logits)), tf.zeros_like(logits), logits)

        return logits


    alphas = tf.scan(calc_alphas, [df, dm], initializer=tf.zeros(shape=[mw, ms]), name='alphas')  # bs x mw x ms

"""
    Calculate the double convex combination of senses of all plain words for a given definition and pass the resulting
    output to a 2-layer NN to get the new embedding
"""
with tf.name_scope('Convex_comb_senses'):

    raw_emb = tf.reduce_sum(tf.expand_dims(alphas, axis=3) * tf.gather(params['dwe'], df), axis=2, name='senses_sum')
    e_i = tf.reduce_sum((raw_emb * idf) * wm, axis=1, name='pl_words_sum')  # bs x td

with tf.name_scope("2_Layer_NN"):

    new_emb = tf.tanh(tf.matmul(e_i, params['L1'], name='L1'))  # bs x hd
    new_emb = tf.matmul(new_emb, params['L2'], name='L2')  # bs x td, after passing through a 2-layer network


"""
    Loss for regression
"""
with tf.name_scope("Loss_calc"):
    # Uncomment the following for NORMALIZATION
    # new_emb_norm = tf.nn.l2_normalize(new_emb, axis=1)
    # pdw_norm = tf.nn.l2_normalize(pdw, axis=1)

    # Uncomment the following for REGULARIZATION
    # l1_l2_reg = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0001, scale_l2=0.0001)
    # reg_penalty = tf.contrib.layers.apply_regularization(l1_l2_reg, [params['L1'], params['L2']])

    # Choose loss function
    # loss = tf.reduce_mean(tf.abs(pdw_norm - new_emb_norm))

    loss = tf.losses.mean_squared_error(pdw, new_emb)


"""
    Optimization
"""

with tf.name_scope('optimization_via_grads'):
    optimizer = tf.train.GradientDescentOptimizer(lr)
    gradients, variables = zip(*optimizer.compute_gradients(loss))


    # clip a small value to deal with vanishing and exploding gradients
    # if i don't add these lines, i get NaN values in gradients
    # gradients = [
    #     None if gradient is None else tf.where(tf.logical_or(tf.is_nan(gradient), tf.is_inf(gradient)), tf.zeros_like(gradient), gradient)
    #     for gradient in gradients]
    # gradients_, _ = tf.clip_by_global_norm(gradients, 1e-10)

    # Summarize all gradients and weights

    for grad, var in zip(gradients, variables):
        tf.summary.histogram(var.name + '/weights', var)
        tf.summary.histogram(var.name + '/gradient', grad)
    train_op = optimizer.apply_gradients(zip(gradients, variables))


"""
    Fixed-point update
"""
with tf.name_scope("fixed_point_update"):
    fp_emb = (1 - beta) * pdw + beta * e_i
    fp_update = tf.scatter_update(params['dwe'], wi, fp_emb)  # we only update a portion of the embeddings
    dwe_diff = tf.reduce_max(tf.abs(fp_emb - pdw))  # maximum increment


"""
    TRAINING PROCESS

"""

with tf.Session() as sess:
    init_all_op = tf.global_variables_initializer()
    sess.run(init_all_op)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    # tensorboard line for the Graph
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    # python variable summary (Mean batch loss value)
    m_b_loss_summ = tf.Summary()

    print('Training on ' + str(true_nw) + " data....")

    # create the "pool" of indices for batch creation
    indices_pool = np.array([i for i in range(true_nw)])

    # basic training parameters
    num_epoch = 10  # total number of epochs to train
    cur_ep = 0
    cur_lr = 5e-2
    num_consec_train = 5  # number of consecutive epochs for SGD
    mode = ['sgd', 'fp']
    cur_mode = 0  # start with 'sgd'. Set this to 1 if you want to start with fp
    tol = 1e-3
    next_schedule = 4  # ( num_consec_train - 1)
    dwe_up_cnt = 0
    tic = 30
    beta_val = 0.7
    # entered_fp = 1

    for epoch in range(num_epoch):

        # Print epoch
        print("################################################################################")
        print("Now processing data for epoch: " + str(epoch))
        print("################################################################################")

        # counter for processed data ( each epoch )
        cnt_processed = 0

        # total loss
        epoch_loss = 0

        # shuffle before slicing
        np.random.shuffle(indices_pool)
        cost = 0
        totTime = 0
        max_diff = -np.inf
        cur_beta = beta_val ** (dwe_up_cnt + 1)
        
        for i_s in range(0, true_nw, bs):

            # i_s and i_e are the Starting and Ending indices of the indices_pool, and their are used to sample our
            # shuffled dataset
            i_e = i_s + bs if i_s + bs < true_nw else true_nw - 1

            # word indices to be trained
            wis = indices_pool[i_s: i_e]
            # indices of the negative sample words
            # cnt = 0
            # nwis = []
            # while cnt < len(wis):
            #     rand_num = np.random.randint(0, true_nw)
            #     if rand_num not in wis:
            #         nwis.append(rand_num)
            #         cnt += 1
            # nwis = np.array(nwis)

            # provide the priors
            priors = np.ones(shape=(len(wis), mw, ms))

            # increase the count of processed data
            cnt_processed += len(wis)

            # initialize batch_loss
            batch_loss = 0

            if mode[cur_mode] == 'sgd':

                feed_d = {wi: wis, df: d['def'][wis], dm: d['dmask'][wis], wmask: d['wmask'][wis], h_d: d['idf'][wis],
                          lr: cur_lr, pr: priors}
                _, batch_loss, merged_summary = sess.run([train_op, loss, merged_summary_op], feed_dict=feed_d)

                # record to Tensorboard
                summary_writer.add_summary(merged_summary, epoch)

            elif mode[cur_mode] == 'fp':

                feed_d = {wi: wis, df: d['def'][wis], dm: d['dmask'][wis], wmask: d['wmask'][wis], h_d: d['idf'][wis],
                          beta: beta_val, pr: priors}
                _, batch_loss, diff, merged_summary = sess.run([fp_update, loss, dwe_diff, merged_summary_op], feed_dict=feed_d)

                # record to Tensorboard
                summary_writer.add_summary(merged_summary, epoch)

                # Difference on update / updates counted
                max_diff = max(max_diff, float(diff))
                dwe_up_cnt += 1

            epoch_loss += batch_loss
            # print mini-batch loss every "tic" time
            if (i_e // bs) % tic == 0:
                print("Current mode:" + mode[cur_mode])
                print(
                    "Accumulated loss (" + str(cnt_processed) + " of " + str(true_nw) + " data): " + str(epoch_loss))

        # End of epoch --> mean batch loss
        mean_batch_loss = epoch_loss / np.ceil((true_nw / bs))
        m_b_loss_summ.value.add(tag='Mean_batch_loss', simple_value= mean_batch_loss)
        summary_writer.add_summary(m_b_loss_summ, epoch)

        # At the end of each epoch determine the transition of the training process
        if cur_mode == 1 and max_diff < tol:
            cur_mode = 0  # switch to SGD
            max_diff = -np.inf
            next_schedule = epoch + num_consec_train
        elif cur_mode == 0 and next_schedule == epoch:
            cur_mode = 1  # switch to fixed-point iteration
            dwe_up_cnt = 0

        # Here you can add a function for saving the model for each epoch
        # TO-DO

    # configure the projector
    embeddings_writer = tf.summary.FileWriter(log_dir, sess.graph)
    config = projector.ProjectorConfig()
    config.model_checkpoint_path = os.path.join(models_dir, 'test_model.ckpt')
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = params['dwe'].name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(embeddings_writer, config)
    # Exit training process, save model
    save_model(sess, models_dir, 'test_model.ckpt')


