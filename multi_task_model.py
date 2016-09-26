# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:28:22 2016

@author: Bing Liu (liubing@cmu.edu)

Multi-task model: intent, tagging, LM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes

import data_utils
import seq_labeling
from tensorflow.python.ops import array_ops
from generate_encoder_output import generate_embedding_RNN_output_joint


class MultiTaskModel(object):
  def __init__(self, source_vocab_size, lm_vocab_size, tag_vocab_size, 
               label_vocab_size, buckets, 
               word_embedding_size, size, num_layers, max_gradient_norm, batch_size, 
               dropout_keep_prob=1.0, use_lstm=True, lm_cost_weight=1.0,
               DNN_at_output=False, dnn_hidden_layer_size=200, output_emb_size=50,
               use_local_context=False, zero_intent_thres=0,
               forward_only=False):
    """Create the model.

    Args:
      source_vocab_size: int, size of the source vocabulary.
      lm_vocab_size: int, size of the LM vocabulary.
      tag_vocab_size: int, size of the intent tags.
      label_vocab_size: int, size of the slot labels.
      buckets: dummy buckets here, modified from tensorflow translation example.
      word_embedding_size: int, word embedding size
      size: int, number of units in each layer of the model.      
      num_layers: int, number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      dropout_keep_prob: prob to keep during dropout
      use_lstm: use lstm cell
      lm_cost_weight: lm error weight in linear interpolation os multi-task model errors
      DNN_at_output: use DNN at the output layer of each task
      dnn_hidden_layer_size: DNN hidden layer size
      output_emb_size: size of output label/tag embedding
      use_local_context: boolean, if set, apply sampled intent/tag as context to lm.
      zero_intent_thres: int, the intent contribution to context remain zero before this thres, 
                         and linear increase to 1 after that.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.source_vocab_size = source_vocab_size
    self.tag_vocab_size = tag_vocab_size
    self.label_vocab_size = label_vocab_size
    self.lm_vocab_size = lm_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.global_step = tf.Variable(0, trainable=False)

    softmax_loss_function = None

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      # use the customized rnn_cell, --> added trainable option in RNN
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
     
    if not forward_only and dropout_keep_prob < 1.0:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                           input_keep_prob=dropout_keep_prob,
                                           output_keep_prob=dropout_keep_prob)


    # Feeds for inputs.
    self.encoder_inputs = []
    self.encoder_inputs_shiftByOne = []
    self.tag_weights = []
    self.intent_weights = []
    self.lm_weights = []
    self.tags = []    
    self.labels = []    
    self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
    
    self.lm_cost_weight = lm_cost_weight
    
    self.train_with_true_label = tf.placeholder(tf.bool, name="train_with_true_label")
    
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
      self.encoder_inputs_shiftByOne.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder_shiftByOne{0}".format(i)))
      self.lm_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="lm_weight{0}".format(i)))
    for i in xrange(buckets[-1][1]): # output length is 1
      self.tags.append(tf.placeholder(tf.float32, shape=[None], name="tag{0}".format(i)))
      self.tag_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="tag_weight{0}".format(i)))
      self.intent_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="intent_weight{0}".format(i)))
    self.labels.append(tf.placeholder(tf.float32, shape=[None], name="label"))
    self.label_duplicated_sequence = self.labels * buckets[-1][1]

    rnn_outputs = generate_embedding_RNN_output_joint(self.encoder_inputs,
                                                        cell,
                                                        self.source_vocab_size,
                                                        self.label_duplicated_sequence,
                                                        self.label_vocab_size,
                                                        self.tags,
                                                        self.tag_vocab_size,
                                                        word_embedding_size,
                                                        dtype=dtypes.float32,
                                                        scope=None,
                                                        sequence_length=self.sequence_length,
                                                        DNN_at_output=DNN_at_output, 
                                                        dnn_hidden_layer_size=dnn_hidden_layer_size,
                                                        output_emb_size=output_emb_size,
                                                        train_with_true_label=self.train_with_true_label,
                                                        zero_intent_thres=zero_intent_thres,
                                                        forward_only=forward_only)
                                                    
    encoder_outputs, encoder_state, encoder_embedded_inputs, intent_results, tagging_results  = rnn_outputs

    # get tagging loss
    self.tagging_output, sampled_tags, _, self.tagging_loss, _ = seq_labeling.generate_task_output(
          encoder_outputs, None, encoder_state, self.tags, self.sequence_length, self.tag_vocab_size, self.tag_weights,
          buckets, softmax_loss_function=softmax_loss_function, scope='tagging', DNN_at_output=DNN_at_output, tagging_results=tagging_results, forward_only=forward_only)


    # get intent classification loss
      # transform intent classification to a sequence labeling task 
      # output intent class at each step
    self.intent_outputs, _, sampled_intents, self.classification_loss, _ = seq_labeling.generate_task_output(
          encoder_outputs, None, encoder_state, self.label_duplicated_sequence, self.sequence_length, self.label_vocab_size, self.intent_weights,
          buckets, softmax_loss_function=softmax_loss_function, use_attention=False, scope='intent', DNN_at_output=DNN_at_output, train_with_true_label=self.train_with_true_label, intent_results=intent_results, forward_only=forward_only)  
    classification_output_rev = seq_labeling._reverse_seq(self.intent_outputs, self.sequence_length)
    self.classification_output = [classification_output_rev[0]]

      
    # get lm loss, use additional_inputs = sampled_tags_intents
    sampled_tags_intents = [array_ops.concat(1, [sampled_tag, sampled_intent]) for sampled_tag, sampled_intent in zip(sampled_tags, sampled_intents)]
    self.lm_output, _, _, self.lm_loss_withRegularizers, self.lm_loss = seq_labeling.generate_task_output(
            encoder_outputs, sampled_tags_intents, encoder_state, self.encoder_inputs_shiftByOne, self.sequence_length, self.lm_vocab_size, self.lm_weights,
            buckets, softmax_loss_function=softmax_loss_function, use_attention=False, scope='lm', DNN_at_output=DNN_at_output, use_local_context=use_local_context,
            forward_only=forward_only)  
            
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      for i in range(len(params)):
        print (params[i].name)
      opt = tf.train.AdamOptimizer()
      gradients = tf.gradients([self.tagging_loss, self.classification_loss, self.lm_loss_withRegularizers*self.lm_cost_weight], params)
        
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                       max_gradient_norm)
      self.gradient_norm = norm
      self.update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables())


  def joint_step(self, session, encoder_inputs, encoder_inputs_shiftByOne, 
                 lm_weights, tags, tag_weights, labels, intent_weights, 
                 batch_sequence_length, bucket_id, forward_only, 
                 train_with_true_label=True):
    """Run a joint step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      encoder_inputs_shiftByOne: lm output
      lm_weights: list of numpy float vectors to feed as lm target weights.
      tags: list of numpy int vectors to feed as output tags.
      tag_weights: list of numpy float vectors to feed as tag weights.
      labels: list of numpy int vectors to feed as intent labels.
      intent_weights: list of numpy float vectors to feed as intent weights.
      batch_sequence_length: list of numpy int to feed as sequence length.
      bucket_id: which bucket of the model to use. # dummy, always 0.
      forward_only: whether to do the backward step or only forward.
      train_with_true_label: whether to use true label during model training.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, tag_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(tags) != tag_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(tags), tag_size))
    if len(labels) != 1:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(labels), 1))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.tags[l].name] = tags[l]
      input_feed[self.tag_weights[l].name] = tag_weights[l]
      input_feed[self.intent_weights[l].name] = intent_weights[l]
      input_feed[self.encoder_inputs_shiftByOne[l].name] = encoder_inputs_shiftByOne[l]
      input_feed[self.lm_weights[l].name] = lm_weights[l]
    input_feed[self.labels[0].name] = labels[0]
    input_feed[self.train_with_true_label.name] = train_with_true_label

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.lm_loss] # Loss for this batch, report the LM loss here.
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
      output_feed.append(self.classification_output[0])
    else:
      output_feed = [self.lm_loss]
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
      output_feed.append(self.classification_output[0])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3:3+tag_size],outputs[-1]
    else:
      return None, outputs[0], outputs[1:1+tag_size], outputs[-1]


  def tagging_step(self, session, encoder_inputs, tags, tag_weights, batch_sequence_length,
           bucket_id, forward_only, train_with_true_label=True):
    """Run a tagging step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      tags: list of numpy int vectors to feed as output tags.
      tag_weights: list of numpy float vectors to feed as tag weights.
      batch_sequence_length: list of numpy int to feed as sequence length.
      bucket_id: which bucket of the model to use. # dummy, always 0.
      forward_only: whether to do the backward step or only forward.
      train_with_true_label: whether to use true label during model training.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, tag_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(tags) != tag_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(tags), tag_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.tags[l].name] = tags[l]
      input_feed[self.tag_weights[l].name] = tag_weights[l]
    input_feed[self.train_with_true_label.name] = train_with_true_label
    
    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.tagging_loss] # Loss for this batch.
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
    else:
      output_feed = [self.tagging_loss]
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3:3+tag_size]
    else:
      return None, outputs[0], outputs[1:1+tag_size]

  def classification_step(self, session, encoder_inputs, labels, intent_weights, batch_sequence_length,
           bucket_id, forward_only):
    """Run a intent classification step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      labels: list of numpy int vectors to feed as intent labels.
      intent_weights: list of numpy float vectors to feed as intent weights.
      batch_sequence_length: list of numpy int to feed as sequence length.
      bucket_id: which bucket of the model to use. # dummy, always 0.
      forward_only: whether to do the backward step or only forward.
      train_with_true_label: whether to use true label during model training.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, target_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.intent_weights[l].name] = intent_weights[l]
    input_feed[self.labels[0].name] = labels[0]

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.classification_loss,    # Loss for this batch.
                     self.classification_output[0],
                     ]
    else:
      output_feed = [self.classification_loss,
                     self.classification_output[0],
                     ]

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3]  # Gradient norm, loss, outputs.
    else:
      return None, outputs[0], outputs[1]  # No gradient norm, loss, outputs.


  def lm_step(self, session, encoder_inputs, encoder_inputs_shiftByOne, lm_weights, batch_sequence_length, bucket_id, forward_only):
    """Run a joint step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      encoder_inputs_shiftByOne: lm output
      lm_weights: list of numpy float vectors to feed as lm target weights.
      batch_sequence_length: list of numpy int to feed as sequence length.
      bucket_id: which bucket of the model to use. # dummy, always 0.
      forward_only: whether to do the backward step or only forward.
      train_with_true_label: whether to use true label during model training.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    encoder_size, tag_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(encoder_inputs_shiftByOne) != tag_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs_shiftByOne), tag_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.encoder_inputs_shiftByOne[l].name] = encoder_inputs_shiftByOne[l]
      input_feed[self.lm_weights[l].name] = lm_weights[l]

    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.lm_loss] # Loss for this batch.
    else:
      output_feed = [self.lm_loss]

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2]
    else:
      return None, outputs[0]
        
  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, encoder_inputs_shiftByOne, decoder_inputs, labels = [], [], [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_sequence_length_list= list()
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input, label = random.choice(data[bucket_id])
      batch_sequence_length_list.append(len(encoder_input) + 1) # +1 for BOS_ID

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input) - 1) # -1 for BOS_ID 
      #encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
      encoder_inputs.append(list([data_utils.BOS_ID] + encoder_input + encoder_pad))
      encoder_inputs_shiftByOne.append(list(encoder_input + [data_utils.BOS_ID] + encoder_pad)) #BOS_ID and EOS_ID shared the same ID in in_vocab and lm_vocab

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1  # -1 for BOS_ID 
      decoder_inputs.append([data_utils.PAD_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
      labels.append(label)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_encoder_inputs_shiftByOne, batch_decoder_inputs, tagging_batch_weights, intent_batch_weights, lm_batch_weights, batch_labels = [], [], [], [], [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs_shiftByOne.append(
          np.array([encoder_inputs_shiftByOne[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))                        
      # Create target_weights to be 0 for targets that are padding.
      tagging_batch_weight = np.ones(self.batch_size, dtype=np.float32)
      intent_batch_weight = np.ones(self.batch_size, dtype=np.float32)
      lm_batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.

        # set tagging weight in padding position to 0, leave others to 1
        if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          tagging_batch_weight[batch_idx] = 0.0
        # set tagging weight in padding position to 0, leave others to linearly increasing weight
        if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          intent_batch_weight[batch_idx] = 0.0
        else:
          if length_idx <= 4:
            intent_batch_weight[batch_idx] = 0.0
          else:
            intent_batch_weight[batch_idx] = (length_idx - 4) / batch_sequence_length_list[batch_idx]
        if encoder_inputs_shiftByOne[batch_idx][length_idx] == data_utils.PAD_ID:
          lm_batch_weight[batch_idx] = 0.0        
      tagging_batch_weights.append(tagging_batch_weight)
      intent_batch_weights.append(intent_batch_weight)
      lm_batch_weights.append(lm_batch_weight)
    
    batch_labels.append(
      np.array([labels[batch_idx][0]
                for batch_idx in xrange(self.batch_size)], dtype=np.int32))
                        
    batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
    return batch_encoder_inputs, batch_encoder_inputs_shiftByOne, batch_decoder_inputs, tagging_batch_weights, intent_batch_weights, lm_batch_weights, batch_sequence_length, batch_labels


  def get_one(self, data, bucket_id, sample_id):
    """Get a sample data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.
      sample_id: integer, sample id within the bucket specified.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, encoder_inputs_shiftByOne, decoder_inputs, labels = [], [], [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_sequence_length_list= list()
    #for _ in xrange(self.batch_size):
    encoder_input, decoder_input, label = data[bucket_id][sample_id]
    batch_sequence_length_list.append(len(encoder_input) + 1) # +1 for BOS_ID

      # Encoder inputs are padded and then reversed.
    encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input) - 1) # -1 for BOS_ID 
    encoder_inputs.append(list([data_utils.BOS_ID] + encoder_input + encoder_pad))
    encoder_inputs_shiftByOne.append(list(encoder_input + [data_utils.BOS_ID] + encoder_pad)) #BOS_ID and EOS_ID shared the same ID in two vocabs

    # Decoder inputs get an extra "GO" symbol, and are padded then.
    decoder_pad_size = decoder_size - len(decoder_input) - 1 # -1 for BOS_ID 
    decoder_inputs.append([data_utils.PAD_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
    labels.append(label)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_encoder_inputs_shiftByOne, batch_decoder_inputs, tagging_batch_weights, intent_batch_weights, lm_batch_weights, batch_labels = [], [], [], [], [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(1)], dtype=np.int32))

    for length_idx in xrange(encoder_size):
      batch_encoder_inputs_shiftByOne.append(
          np.array([encoder_inputs_shiftByOne[batch_idx][length_idx]
                    for batch_idx in xrange(1)], dtype=np.int32))
                        
    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(1)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      tagging_batch_weight = np.ones(1, dtype=np.float32)
      intent_batch_weight = np.ones(1, dtype=np.float32)
      lm_batch_weight = np.ones(1, dtype=np.float32)
      for batch_idx in xrange(1):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          tagging_batch_weight[batch_idx] = 0.0
        if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          intent_batch_weight[batch_idx] = 0.0
        else:
          if length_idx <= 4:
            intent_batch_weight[batch_idx] = 0.0
          else:
            intent_batch_weight[batch_idx] = (length_idx - 4) / batch_sequence_length_list[batch_idx]
        if encoder_inputs_shiftByOne[batch_idx][length_idx] == data_utils.PAD_ID:
          lm_batch_weight[batch_idx] = 0.0
      tagging_batch_weights.append(tagging_batch_weight)
      lm_batch_weights.append(lm_batch_weight)
      
    batch_labels.append(
      np.array([labels[batch_idx][0]
                for batch_idx in xrange(1)], dtype=np.int32))
                    
    batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
    return batch_encoder_inputs, batch_encoder_inputs_shiftByOne, batch_decoder_inputs, tagging_batch_weights, intent_batch_weights, lm_batch_weights, batch_sequence_length, batch_labels