'''
Created on December 25th, 2018
@author : maobubu (Huicheng Liu)
Reference : Enhanced LSTM for Natural Language Inference (ACL 2017)
'''
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle as pkl
import pdb
import numpy
import copy
import logging
from tensorflow import logging  as log
# logger = logging.getLogger("tensorflow").setLevel(logging.WARNING)
# logging.getLogger(tensorflow).disabled = True
# log.set_verbosity(logging.INFO)

import warnings
import sys
import time
import pprint

import tensorflow as tf
from collections import OrderedDict
# from nli_prepa import TextIterator
from data_iterator import TextIterator
from tensorflow.contrib import rnn

logger = logging.getLogger(__name__)


# make prefix-appended name
def _s(pp, name):
    return '%s_%s' % (pp, name)


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params
"""
Neural network layer definitions.

The life-cycle of each of these layers is as follows
    1) The param_init of the layer is called, which creates
    the weights of the network.
    2) The feedforward is called which builds that part of the Theano graph
    using the weights created in step 1). This automatically links
    these variables to the graph.

Each prefix is used like a key and should be unique
to avoid naming conflicts when building the graph.
"""

# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')
# batch preparation
def prepare_data(seqs_x, seqs_y, labels, options, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x)
    maxlen_y = numpy.max(lengths_y)

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    l = numpy.zeros((n_samples,)).astype('int64')

    for idx, [s_x, s_y, ll] in enumerate(zip(seqs_x, seqs_y, labels)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx], idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx], idx] = 1.
        l[idx] = ll

    return x, x_mask, y, y_mask, l
        # , maxlen_x, maxlen_y



def fflayer(options, input,params,prefix='ff',  activation_function=None,nin=None, nout=None):

    with tf.variable_scope(prefix,reuse = tf.AUTO_REUSE):
        W = tf.get_variable(
                        _s(prefix,'W'),
                        shape = [nin,nout],
                        initializer=tf.random_normal_initializer(stddev=0.02),
                        dtype=tf.float32
                        )
        bias = tf.get_variable(
                        _s(prefix,'bias'),
                        shape = [nout],
                        initializer = tf.constant_initializer(0.),
                        dtype = tf.float32
                        )
    #     prepare for matmul

    # W_ = tf.tile(W,[tf.shape(input)[0],1])
    # W = tf.reshape(W_,[tf.shape(input)[0],tf.shape(W)[0],tf.shape(W)[1]])
    result = tf.nn.bias_add(tf.tensordot(input,W,[[-1],[0]]),bias)
    if activation_function is None:
        outputs = result
    else:
        outputs = activation_function(result)
    return outputs




def fflayer_2D(options, input,prefix='ff',  activation_function=None,nin=None, nout=None):
    with tf.variable_scope(prefix,reuse = tf.AUTO_REUSE):
        W = tf.get_variable(
                        _s(prefix,'W'),
                        shape = [nin,nout],
                        initializer = tf.random_uniform_initializer( -0.1, 0.1),
                        dtype=tf.float32
                        )
        bias = tf.get_variable(
                        _s(prefix,'bias'),
                        shape = [nout],
                        initializer = tf.constant_initializer(0.),
                        dtype = tf.float32
                        )

        result = tf.nn.bias_add(tf.matmul(input,W),bias)
    if activation_function is None:
        outputs = result
    else:
        outputs = activation_function(result)
    return outputs

def cudnn_lstm_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 8 * hidden_size
    weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return biases + weights

def direction_to_num_directions(direction):
    if direction == "unidirectional":
        return 1
    elif direction == "bidirectional":
        return 2
    else:
        raise ValueError("Unknown direction: %r." % (direction,))
def estimate_cudnn_parameter_size(num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            params += cudnn_lstm_parameter_size(
                isize, hidden_size
            )
        isize = hidden_size * num_directions
    return params



def bilstm_filter(state_below,n_samples,mask,options,keep_prob,prefix='lstm',dim=300,is_training=True):
        with tf.variable_scope(name_or_scope=prefix,reuse=tf.AUTO_REUSE) as scope:
            output1 = bilstm_layer(state_below,n_samples,mask,options,dim,keep_prob,is_training=is_training)
        return output1


def bilstm_layer(state_below,n_samples,mask,options,dim,keep_prob,is_training=True):
    #cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 1,num_units = dim,input_mode="linear_input",direction = "bidirectional",dropout = 0)
    cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=dim, input_size=dim,input_mode="linear_input", direction="bidirectional", dropout=0)
    params_size_t = cell.params_size()
    est_size = estimate_cudnn_parameter_size(
        num_layers=1,
        hidden_size=dim,
        input_size=dim,
        input_mode="linear_input",
        direction="bidirectional")
    c = tf.zeros([2, n_samples, dim],tf.float32)
    h = tf.zeros([2, n_samples, dim],tf.float32)

    rnn_params = tf.get_variable("lstm_params",initializer=tf.random_uniform([est_size], -0.1, 0.1))
    outputs, h, c = cell(state_below, h, c, rnn_params, is_training)
    return outputs

def init_params(options, worddicts):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words'],options['dim_word'])
    # read embedding from GloVe
    if options['embedding']:
        with open(options['embedding'], 'r') as f:
            for line in f:
                tmp = line.split()
                word = tmp[0]
                vector = tmp[1:]
                if word in worddicts and worddicts[word] < options['n_words']:
                    params['Wemb'][worddicts[word], :] = vector
                    # encoder: bidirectional RNN

    return params
# build a training model
def word_embedding(options,params):
    embeddings = tf.get_variable("embeddings", shape= [options['n_words'], options['dim_word']],
                                 initializer=tf.constant_initializer(numpy.array(params['Wemb'])))
    # W = tf.Variable(tf.constant(0.0, shape=[options['n_words'], options['dim_word']]),
    #                 trainable=True, name="W")
    # embedding_placeholder = tf.placeholder(tf.float32, [options['n_words'], options['dim_word']],name='embedding_placeholder')
    # embedding_init = W.assign(embedding_placeholder)
    return embeddings

# build a training model

def dot_production_attention(x_sent,y_sent,x_mask,y_mask):
    weight_matrix = tf.matmul(x_sent, tf.transpose(y_sent, [0, 2, 1]))
    weight_matrix_y = tf.exp(weight_matrix-tf.reduce_max(weight_matrix,axis=2,keep_dims=True))
    weight_matrix_x = tf.exp(tf.transpose((weight_matrix-tf.reduce_max(weight_matrix,axis=1,keep_dims=True)),perm=[0,2,1]))
    weight_matrix_y = weight_matrix_y * y_mask[:,None,:]
    weight_matrix_x = weight_matrix_x * x_mask[:,None,:]
    alpha = weight_matrix_y/(tf.reduce_sum(weight_matrix_y,-1,keep_dims=True)+ 1e-8)
    beta = weight_matrix_x/(tf.reduce_sum(weight_matrix_x,-1,keep_dims=True)+ 1e-8)
    weight_matrix_y = tf.reduce_sum(tf.expand_dims(y_sent,1)*tf.expand_dims(alpha,-1),2)
    weight_matrix_x = tf.reduce_sum(tf.expand_dims(x_sent,1)*tf.expand_dims(beta,-1),2)
    return tf.transpose( weight_matrix_y,[1,0,2]),tf.transpose( weight_matrix_x,[1,0,2])

def build_model(embedding, options,params):
    """ Builds the entire computational graph used for training
    """
    opt_ret = dict()

    # description string: #words x #samples
    x1 = tf.placeholder(tf.int64,shape=[None,None],name='x1')
    x1_mask =  tf.placeholder(tf.float32,shape=[None,None],name='x1_mask')
    x2 = tf.placeholder(tf.int64,shape=[None,None],name='x2')
    x2_mask = tf.placeholder(tf.float32,shape=[None,None],name='x2_mask')
    y = tf.placeholder(tf.int64,shape=[None],name='y')
    keep_rate = tf.placeholder(tf.float32, [],name='keep_rate')


    # description string: #words x #samples

    n_timesteps_x1 = tf.shape(x1)[0]
    n_timesteps_x2 = tf.shape(x2)[0]
    n_samples = tf.shape(x1)[1]


    is_training = True
    #sent1,batchsize,emb
    emb1 = tf.nn.embedding_lookup(embedding,x1)
    if options['use_dropout']:
        emb1 =  tf.nn.dropout(emb1, keep_rate)
    emb2 = tf.nn.embedding_lookup(embedding,x2)
    if options['use_dropout']:
         emb2 = tf.nn.dropout(emb2, keep_rate)
    proj1 = bilstm_filter( emb1,n_samples,x1_mask, options,keep_rate, prefix='encoder',dim=300,is_training=is_training)
    ctx1 = tf.concat(proj1,2)
    proj2 = bilstm_filter( emb2,n_samples,x2_mask ,options,keep_rate, prefix='encoder',dim=300,is_training=is_training)
    ctx2 = tf.concat(proj2,2)
    ctx1 = ctx1 * x1_mask[:,:,None]
    ctx2 = ctx2 * x2_mask[:,:,None]
    weight_y,weight_x = dot_production_attention(tf.transpose(ctx1,[1,0,2]),tf.transpose(ctx2,[1,0,2]),tf.transpose(x1_mask,[1,0]),tf.transpose(x2_mask,[1,0]))
    inp1_ = tf.concat([ctx1, weight_y, tf.multiply(ctx1,weight_y), tf.subtract(ctx1,weight_y)],2)
    inp2_ = tf.concat([ctx2, weight_x, tf.multiply(ctx2,weight_x), tf.subtract(ctx2,weight_x)],2)

    inp1 = fflayer(options,inp1_,params,prefix='projection',activation_function=tf.nn.relu, nin=2400, nout=300)
    inp2 = fflayer(options,inp2_,params,prefix='projection',activation_function=tf.nn.relu, nin=2400, nout=300)

    if options['use_dropout']:
        inp1 = tf.nn.dropout(inp1, keep_rate)
        inp2 = tf.nn.dropout(inp2, keep_rate)


    proj3 = bilstm_filter(inp1,n_samples,x1_mask, options,keep_rate, prefix='decoder',dim=300,is_training=is_training)
    proj4 = bilstm_filter(inp2, n_samples,x2_mask, options, keep_rate, prefix='decoder', dim=300, is_training=is_training)
    ctx3 = tf.concat(proj3,2)
    ctx4 = tf.concat(proj4,2)

    logit1 = tf.reduce_sum(ctx3 * tf.expand_dims(x1_mask,2),0) / tf.expand_dims(tf.reduce_sum(x1_mask,0),1)
    logit2 = tf.reduce_max(ctx3 * tf.expand_dims(x1_mask,2),0)
    logit3 = tf.reduce_sum(ctx4 * tf.expand_dims(x2_mask,2),0) / tf.expand_dims(tf.reduce_sum(x2_mask,0),1)
    logit4 = tf.reduce_max(ctx4 *tf.expand_dims(x2_mask,2),0)

    logit = tf.concat([logit1,logit2,logit3,logit4],1)
    logit = tf.nn.dropout(logit, keep_rate)

    logit = fflayer(options, logit,params, prefix='ff', activation_function=tf.nn.tanh, nin=2400, nout=300)
    if options['use_dropout']:
        logit = tf.nn.dropout(logit, keep_rate)
    pred = fflayer(options, logit,params, prefix='fout', activation_function = None, nin=300, nout=3)


    logger.info('Building f_cost...')
    # todo not same
    # labels = tf.one_hot(y, depth=3, axis=1)
    # pred = tf.nn.softmax(pred,1)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
    # cost =  -tf.reduce_sum(tf.cast(labels,tf.float32)*tf.log(pred))
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=y)
    logger.info('Done')




    return  is_training,cost,x1, x1_mask, x2, x2_mask, y, n_timesteps_x1,n_timesteps_x2,pred

def predict_pro_acc(sess,cost,prepare_data, model_options, iterator, maxlen,correct_pred,pred):
    logger.info('start')
    # fo = open(_s(prefix,'pre.txt'), "w")
    num = 0
    valid_acc = 0
    total_cost = 0
    loss = 0
    result = 0
    for x1_sent, x2_sent, y_sent in iterator:
        num += len(x1_sent)
        data_x1, data_x1_mask, data_x2, data_x2_mask, data_y = prepare_data( x1_sent, x2_sent, y_sent, model_options, maxlen=maxlen)

        loss ,result = sess.run([cost,correct_pred],
                              feed_dict={  'x1:0':data_x1 , 'x1_mask:0':data_x1_mask , 'x2:0':data_x2 , 'x2_mask:0':data_x2_mask , 'y:0':data_y ,'keep_rate:0':1.0})

        valid_acc += result.sum()
        total_cost += loss.sum()


    final_acc =1.0 * valid_acc / num
    final_loss = 1.0 * total_cost  /num


    return final_acc , final_loss


def train(
          dim_word         = 100,  # word vector dimensionality
          dim              = 100,  # the number of GRU units
          encoder          = 'lstm', # encoder model
          decoder          = 'lstm', # decoder model
          patience         = 10,  # early stopping patience
          max_epochs       = 5000,
          finish_after     = 10000000, # finish after this many updates
          decay_c          = 0.,  # L2 regularization penalty
          clip_c           = -1.,  # gradient clipping threshold # TO AVOID GRADIENT EXPLOSION
          lrate            = 0.01,  # learning rate
          n_words          = 100000,  # vocabulary size
          n_words_lemma    = 100000,
          maxlen           = 100,  # maximum length of the description
          optimizer        = 'adadelta',
          batch_size       = 16,
          valid_batch_size = 16,
          save_model       = '../../models/',
          saveto           = 'model.npz',
          dispFreq         = 100,
          validFreq        = 1000,
          saveFreq         = 1000,   # save the parameters after every saveFreq updates
          use_dropout      = False,
          reload_          = False,
          verbose          = False, # print verbose information for debug but slow speed
          datasets         = [],
          valid_datasets   = [],
          test_datasets    = [],
          dictionary       = [],
          kb_dicts         = [],
          embedding        = '', # pretrain embedding file, such as word2vec, GLOVE
          dim_kb           = 5,
          ):

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",filename='./log_result.txt')
    # Model options
    model_options = locals().copy()

    # load dictionary and invert them
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    logger.info("Loading knowledge base ...")
    # with open(kb_dicts, 'rb') as f:
    #     kb_dict = pkl.load(f)

    # reload options
    if reload_ and os.path.exists(saveto):
        logger.info("Reload options")
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    logger.debug(pprint.pformat(model_options))

    logger.info("Loading data")
    train = TextIterator(datasets[0], datasets[1], datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=batch_size)
    train_valid = TextIterator(datasets[0], datasets[1], datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
    valid = TextIterator(valid_datasets[0], valid_datasets[1], valid_datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
    test = TextIterator(test_datasets[0], test_datasets[1], test_datasets[2],
                         dictionary,
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)

    # Initialize (or reload) the parameters using 'model_options'
    # then build the tensorflow graph
    logger.info("init_word_embedding")
    params = init_params(model_options, worddicts)
    embedding = word_embedding(model_options, params)

    is_training,cost,x1, x1_mask, x2, x2_mask, y,n_timesteps_x1,n_timesteps_x2,pred \
    = build_model(embedding, model_options,params)

    # tvars = tf.trainable_variables()

    #cost= tf.reduce_mean(cost)

    lr = tf.Variable(0.0, trainable=False)

    def assign_lr( session, lr_value):
        session.run( tf.assign(lr, lr_value))

    logger.info('Building optimizers...')
    optimizer = tf.train.AdamOptimizer(learning_rate= lr )
    logger.info( 'Done')

    tvars = tf.trainable_variables()
    for var in tvars:
        print(var.name,var.shape)
    #lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tvars if 'embeddings' not in v.name]) * 0.0001
    #cost = cost + lossL2
    # regularization_cost = 0.0003 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tvars])
    # cost = cost + regularization_cost
    grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),model_options['clip_c'])


    train_op = optimizer.apply_gradients(zip(grads,tvars))

    op_loss = tf.reduce_mean(cost)


    logger.info("corret_pred")
    correct_pred = tf.equal(tf.argmax(input=pred,axis=1),y)
    logger.info("Done")


    logger.info("init variables")
    init = tf.global_variables_initializer()
    logger.info("Done")
    # saver
    saver = tf.train.Saver(max_to_keep=15)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        history_errs = []
        # reload history
        if reload_ and os.path.exists(saveto):
            logger.info("Reload history error")
            history_errs = list(numpy.load(saveto)['history_errs'])

        bad_counter = 0

        if validFreq == -1:
            validFreq = len(train[0])/batch_size
        if saveFreq == -1:
            saveFreq = len(train[0])/batch_size


        uidx = 0
        estop = False
        valid_acc_record = []
        test_acc_record = []
        best_num = -1
        best_epoch_num = 0
        lr_change_list = []
        wait_counter = 0
        wait_N = 1
        learning_rate = model_options['lrate']
        assign_lr(sess, learning_rate)
        for eidx in xrange(max_epochs):
            n_samples = 0
            n_samples_2 = 0
            for x1, x2, y in train:
                n_samples += len(x1)
                # n_samples_2 += len(x2)
                uidx += 1
                keep_prob = 0.5
                is_training = True
                data_x1, data_x1_mask, data_x2, data_x2_mask, data_y = prepare_data(x1, x2, y, model_options,
                                                                          maxlen=maxlen)

                if x1 is None:
                    logger.debug('Minibatch with zero sample under length {0}'.format(maxlen))
                    uidx -= 1
                    continue
                ud_start = time.time()
                _,loss=sess.run([train_op,op_loss],feed_dict={ 'x1:0':data_x1 , 'x1_mask:0':data_x1_mask , 'x2:0':data_x2 , 'x2_mask:0':data_x2_mask , 'y:0':data_y ,'keep_rate:0':keep_prob })
                # logger.debug('correct_pre{0} acc{1}'.format(pre,temp))
                ud = time.time() - ud_start
                if numpy.mod(uidx, dispFreq) == 0:
                    logger.debug('Epoch {0} Update {1} Cost {2} UD {3}'.format(eidx, uidx, loss, ud))

                # validate model on validation set and early stop if necessary
                if numpy.mod(uidx, validFreq) == 0:
                    keep_prob = 1
                    is_training = False
                    valid_acc,valid_loss = predict_pro_acc(sess,cost,prepare_data, model_options, valid, maxlen ,correct_pred,pred)
                    test_acc, test_loss = predict_pro_acc(sess, cost, prepare_data, model_options, test, maxlen ,correct_pred,pred)



                    valid_err = 1.0 - valid_acc
                    history_errs.append(valid_err)

                    logger.debug('Epoch  {0}'.format(eidx))
                    logger.debug('Valid cost  {0}'.format(valid_loss))
                    logger.debug('Valid accuracy  {0}'.format(valid_acc))
                    logger.debug('Test cost  {0}'.format(test_loss))
                    logger.debug('Test accuracy  {0}'.format(test_acc))
                    logger.debug('learning_rate:  {0}'.format(learning_rate))
                    # print 'epoch' ,eidx
                    # print 'Valid cost', valid_loss
                    # print 'Valid accuracy', valid_acc
                    # print 'Test cost', test_loss
                    # print 'Test accuracy', test_acc
                    # print 'lrate:', lrate

                    valid_acc_record.append(valid_acc)
                    test_acc_record.append(test_acc)
                    if uidx == 0 or (valid_err <= numpy.array(history_errs).min() and valid_acc > 0.86):
                        best_num = best_num + 1
                        best_epoch_num = eidx
                        wait_counter = 0
                        logger.info("Saving...")
                        saver.save(sess, _s(_s(_s(save_model, "epoch"), str(best_num)), "model.ckpt"))
                        logger.info(_s(_s(_s(save_model, "epoch"), str(best_num)), "model.ckpt"))
                        numpy.savez(saveto, history_errs=history_errs, **params)
                        pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                        logger.info("Done")

                    if valid_err > numpy.array(history_errs).min():
                        wait_counter += 1

                    if wait_counter >= wait_N:
                        logger.info("wait_counter max, need to half the lr")
                        # print 'wait_counter max, need to half the lr'
                        bad_counter += 1
                        wait_counter = 0
                        logger.debug('bad_counter:  {0}'.format(bad_counter))
                        learning_rate = learning_rate * 0.5
                        assign_lr(sess, learning_rate)
                        lr_change_list.append(eidx)
                        logger.debug('lrate change to:   {0}'.format(learning_rate))
                        # print 'lrate change to: ' + str(lrate)

                    if bad_counter > patience:
                        logger.info("Early Stop!")
                        estop = True
                        break

                    if numpy.isnan(valid_err):
                        pdb.set_trace()

                        # finish after this many updates
                if uidx >= finish_after:
                    logger.debug('Finishing after iterations!  {0}'.format(uidx))
                    # print 'Finishing after %d iterations!' % uidx
                    estop = True
                    break
            logger.debug('Seen samples:  {0}'.format(n_samples))
            # print 'Seen %d samples' % n_samples

            if estop:
                    break



    with tf.Session() as sess:
            # Restore variables from disk.
        saver.restore(sess,_s(_s(_s(save_model,"epoch"),str(best_num)),"model.ckpt") )
        keep_prob = 1
        is_training = False
        logger.info('=' * 80)
        logger.info('Final Result')
        logger.info( '=' * 80)
        # logger.debug('best epoch   {0}'.format(best_epoch_num))

        valid_acc, valid_cost = predict_pro_acc(sess, cost, prepare_data, model_options, valid,
                                                         maxlen, correct_pred,pred)
        logger.debug('Valid cost   {0}'.format(valid_cost))
        logger.debug('Valid accuracy   {0}'.format(valid_acc))
                # print 'Valid cost', valid_cost
                # print 'Valid accuracy', valid_acc


        test_acc, test_cost = predict_pro_acc(sess, cost, prepare_data, model_options, test,
                                                       maxlen, correct_pred,pred)
        logger.debug('Test cost   {0}'.format(test_cost))
        logger.debug('Test accuracy   {0}'.format(test_acc))

    # print 'best epoch ', best_epoch_num
        train_acc, train_cost = predict_pro_acc(sess,  cost, prepare_data, model_options, train_valid,
                                               maxlen,correct_pred,pred)
        logger.debug('Train cost   {0}'.format(train_cost))
        logger.debug('Train accuracy   {0}'.format(train_acc))
        # print 'Train cost', train_cost
        # print 'Train accuracy', train_acc



        # print 'Test cost   ', test_cost
        # print 'Test accuracy   ', test_acc


        return None


if __name__ == '__main__':
    pass



