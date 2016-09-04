# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 16:37:17 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import multi_task_model

import subprocess
import stat


tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 300, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max word vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 123, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("test_file", "/tmp", "Test file.")
tf.app.flags.DEFINE_string("test_output_file", "/tmp", "Test output file.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 300,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 10000,
                            "Max training steps.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("decode_file", False,
                            "Set to True for file decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_integer("max_sequence_length", 0,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")  
tf.app.flags.DEFINE_integer("dnn_hidden_layer_size", 200,
                            "num of nodes in the DNN hidden layer on top of RNN")
tf.app.flags.DEFINE_integer("output_emb_size", 50,
                            "size of output label/tag embedding")
tf.app.flags.DEFINE_float("lm_cost_weight", 1.0,
                          "Weight of LM cost comparing to NLU tasks")  
tf.app.flags.DEFINE_boolean("DNN_at_output", False,
                          "add DNN at the output layer of sequence labeling")     
tf.app.flags.DEFINE_boolean("use_local_context", True,
                          "If false, rnn parameters and emb are locked")    
tf.app.flags.DEFINE_integer("zero_intent_thres", 0,
                            "threshold to linearly increase intent contribution to context")                         
tf.app.flags.DEFINE_string("label_in_training", "true_label", "Options: true_label; predicted_label; scheduled_sampling")                          
FLAGS = tf.app.flags.FLAGS


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
if FLAGS.max_sequence_length == 0:
    print ('Please indicate max sequence length. Exit')
    exit()

# _buckets: modified from tf translation example, just use one dummy bucket here. 
_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]

# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            if wl != '_PAD':
                out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1]) # remove the ending \n on last line
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def read_data(source_path, target_path, label_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source input - word sequence.
    target_path: path to the file with token-ids for the target output - tag sequence;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    label_path: path to the file with token-ids for the sequence classification label
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      with tf.gfile.GFile(label_path, mode="r") as label_file:
        source, target, label = source_file.readline(), target_file.readline(), label_file.readline()
        counter = 0
        while source and target and label and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          label_ids = [int(x) for x in label.split()]
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids, label_ids])
              break
          source, target, label = source_file.readline(), target_file.readline(), label_file.readline()
  return data_set # 3 outputs in each unit: source_ids, target_ids, label_ids 

def read_test_data(source_path):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    source = source_file.readline()
    counter = 0
    while source:
      counter += 1
      if counter % 100000 == 0:
        print("  reading data line %d" % counter)
        sys.stdout.flush()
      source_ids = [int(x) for x in source.split()]
      for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(source_ids) < source_size:
          data_set[bucket_id].append([source_ids, [0]*len(source_ids), [0]])
          break
      source = source_file.readline()
  return data_set # 3 outputs in each unit: source_ids, target_ids, label_ids 

def create_model(session, source_vocab_size, target_vocab_size, label_vocab_size, lm_vocab_size):
  """Create model and initialize or load parameters in session."""
  with tf.variable_scope("model", reuse=None):
    model_train = multi_task_model.MultiTaskModel(
          source_vocab_size, lm_vocab_size, target_vocab_size, label_vocab_size, _buckets,
          FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True, lm_cost_weight=FLAGS.lm_cost_weight,
          forward_only=False,    
          DNN_at_output=FLAGS.DNN_at_output,
          dnn_hidden_layer_size=FLAGS.dnn_hidden_layer_size,
          output_emb_size=FLAGS.output_emb_size, 
          zero_intent_thres=FLAGS.zero_intent_thres)
  with tf.variable_scope("model", reuse=True):
    model_test = multi_task_model.MultiTaskModel(
          source_vocab_size, lm_vocab_size, target_vocab_size, label_vocab_size, _buckets,
          FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True, lm_cost_weight=FLAGS.lm_cost_weight,
          forward_only=True, 
          DNN_at_output=FLAGS.DNN_at_output,
          dnn_hidden_layer_size=FLAGS.dnn_hidden_layer_size,
          output_emb_size=FLAGS.output_emb_size,
          zero_intent_thres=FLAGS.zero_intent_thres)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model_train.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model_train, model_test

def write_results_to_file(result_dict, idx2tag_vocab, result_out_file):
  with open(result_out_file, 'wb') as f:
    for key in sorted(result_dict):
      f.write('%d\t%s\n' % (key, idx2tag_vocab[result_dict[key]]))
        
def train():
  print ('Applying Parameters:')
  for k,v in FLAGS.__dict__['__flags'].iteritems():
    print ('%s: %s' % (k, str(v)))
  print("Preparing trec data in %s" % FLAGS.data_dir)
  vocab_path = ''
  tag_vocab_path = ''
  label_vocab_path = ''

  in_seq_train, out_seq_train, label_train, in_seq_dev, out_seq_dev, label_dev, in_seq_test, out_seq_test, label_test, vocab_path, tag_vocab_path, label_vocab_path = data_utils.prepare_multi_task_data(
    FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)     

  result_dir = FLAGS.train_dir + '/test_results'
  if not os.path.isdir(result_dir):
      os.mkdir(result_dir)

  current_taging_valid_out_file = result_dir + '/taging.valid.hyp.txt'
  current_taging_test_out_file = result_dir + '/taging.test.hyp.txt'
  label_valid_out_file = result_dir + '/label.valid.hyp.txt'
  label_test_out_file = result_dir + '/label.valid.hyp.txt'

  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocabulary(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)
  LM_vocab = vocab.copy()
  assert LM_vocab[data_utils._BOS] == data_utils.BOS_ID
  del LM_vocab[data_utils._BOS]
  LM_vocab[data_utils._BOS] = data_utils.BOS_ID
  rev_LM_vocab = [x for x in rev_vocab]
  rev_LM_vocab[data_utils.BOS_ID] = data_utils._EOS
    
  config = tf.ConfigProto(
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
  )
    
  with tf.Session(config=config) as sess:
    # Create model.
    print("Max sequence length: %d ." % _buckets[0][0])
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    
    model, model_test = create_model(sess, len(vocab), len(tag_vocab), len(label_vocab), len(LM_vocab))
    print ("Creating model with source_vocab_size=%d, target_vocab_size=%d, and label_vocab_size=%d, and lm_vocab_size=%d." % (len(vocab), len(tag_vocab), len(label_vocab), len(LM_vocab)))

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(in_seq_dev, out_seq_dev, label_dev)
    test_set = read_data(in_seq_test, out_seq_test, label_test)
    train_set = read_data(in_seq_train, out_seq_train, label_train)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]

    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    best_valid_score = 0
    best_test_score = 0

    
    if FLAGS.label_in_training == 'true_label':
      print ("Use TRUE label during model training")
      train_with_true_label = True
    elif FLAGS.label_in_training == 'predicted_label':
      print ("Use PREDICTED label during model training")
      train_with_true_label = False
    elif FLAGS.label_in_training == 'scheduled_sampling':
      print ("Use Scheduled Sampling label during model training")
      
    while model.global_step.eval() < FLAGS.max_training_steps:
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, encoder_inputs_shiftByOne, tags, tag_weights, intent_weights, lm_weights, batch_sequence_length, labels = model.get_batch(train_set, bucket_id)

      if FLAGS.label_in_training == 'scheduled_sampling':
          random_number_02 = np.random.random_sample()
          final_training_step = FLAGS.max_training_steps
          if random_number_02 < float(model.global_step.eval())/final_training_step: # use predicted label in training
              train_with_true_label = False    
          else:
              train_with_true_label = True    
      
      _, step_loss, tagging_logits, classification_logits = model.joint_step(sess, encoder_inputs, encoder_inputs_shiftByOne, lm_weights, tags, tag_weights, labels, intent_weights,
                                   batch_sequence_length, bucket_id, False, train_with_true_label=train_with_true_label)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint

      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d step-time %.2f. Training perplexity %.2f" 
            % (model.global_step.eval(), step_time, perplexity))
        sys.stdout.flush()

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        
        def run_eval(data_set, mode): # mode = "Valid", "Test"
        # Run evals on development/test set and print their accyracy.
            word_list = list() 
            ref_tag_list = list() 
            hyp_tag_list = list()
            ref_label_list = list()
            hyp_label_list = list()
            correct_count = 0
            accuracy = 0.0
            for bucket_id in xrange(len(_buckets)): # len(_buckets) = 1 here
              eval_loss = 0.0
              count = 0
              total_word_count = 0
              for i in xrange(len(data_set[bucket_id])):
                count += 1
                eval_encoder_inputs, eval_encoder_inputs_shiftByOne, eval_tags, eval_tag_weights, eval_intent_weights, eval_lm_weights, eval_sequence_length, eval_labels = model_test.get_one(
                  data_set, bucket_id, i)
                eval_intent_weights = eval_tag_weights
                tagging_logits = []
                classification_logits = []
                _, step_loss, tagging_logits, classification_logits = model_test.joint_step(sess, eval_encoder_inputs, eval_encoder_inputs_shiftByOne, eval_lm_weights, eval_tags, eval_tag_weights, eval_labels, eval_intent_weights, 
                                             eval_sequence_length, bucket_id, True)              
                eval_loss += step_loss*(eval_sequence_length[0])
                total_word_count += eval_sequence_length[0] 
                hyp_label = None
                
                # intent results
                ref_label_list.append(rev_label_vocab[eval_labels[0][0]])
                hyp_label = np.argmax(classification_logits[0],0)
                hyp_label_list.append(rev_label_vocab[hyp_label])
                if eval_labels[0] == hyp_label:
                  correct_count += 1
                
                # tagging results
                word_list.append([rev_vocab[x[0]] for x in eval_encoder_inputs[:eval_sequence_length[0]]])
                ref_tag_list.append([rev_tag_vocab[x[0]] for x in eval_tags[:eval_sequence_length[0]]])
                hyp_tag_list.append([rev_tag_vocab[np.argmax(x)] for x in tagging_logits[:eval_sequence_length[0]]])

              eval_perplexity = math.exp(float(eval_loss)/total_word_count)
            print("  %s perplexity: %.2f" % (mode, eval_perplexity))        
            accuracy = float(correct_count)*100/count
            print("  %s accuracy: %.2f %d/%d" % (mode, accuracy, correct_count, count))
            
            tagging_eval_result = dict()
            if mode == 'Valid':
              output_file = current_taging_valid_out_file
            elif mode == 'Test':
              output_file = current_taging_test_out_file
            tagging_eval_result = conlleval(hyp_tag_list, ref_tag_list, word_list, output_file)
            print("  %s f1-score: %.2f" % (mode, tagging_eval_result['f1']))
            sys.stdout.flush()
            return eval_perplexity, tagging_eval_result, hyp_label_list
         
        # run valid
        valid_perplexity, valid_tagging_result, valid_hyp_label_list = run_eval(dev_set, 'Valid')        
        # record best results
        
        if valid_tagging_result['f1'] > best_valid_score:
          best_valid_score = valid_tagging_result['f1']
          subprocess.call(['mv', current_taging_valid_out_file, current_taging_valid_out_file + '.best_f1_%.2f' % best_valid_score])
          with open('%s.best_f1_%.2f' % (label_valid_out_file, best_valid_score), 'wb') as f:
            for i in range(len(valid_hyp_label_list)):
              f.write(valid_hyp_label_list[i] + '\n')
    
        # run test after each validation for development purpose.
        test_perplexity, test_tagging_result, test_hyp_label_list = run_eval(test_set, 'Test')        
        # record best results
        if test_tagging_result['f1'] > best_test_score:
          best_test_score = test_tagging_result['f1']        
          subprocess.call(['mv', current_taging_test_out_file, current_taging_test_out_file + '.best_f1_%.2f' % best_test_score])
          with open('%s.best_f1_%.2f' % (label_test_out_file, best_test_score), 'wb') as f:
            for i in range(len(test_hyp_label_list)):
              f.write(test_hyp_label_list[i] + '\n')

def decode_file(test_file):
  print ('Applying Parameters:')
  for k,v in FLAGS.__dict__['__flags'].iteritems():
    print ('%s: %s' % (k, str(v)))

  vocab_path = FLAGS.data_dir + '/in_vocab_1000.txt'
  tag_vocab_path = FLAGS.data_dir + '/out_vocab_1000.txt'
  label_vocab_path = FLAGS.data_dir + '/label.txt'

  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocabulary(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)
  LM_vocab = vocab.copy()
  assert LM_vocab[data_utils._BOS] == data_utils.BOS_ID
  del LM_vocab[data_utils._BOS]
  LM_vocab[data_utils._BOS] = data_utils.BOS_ID
  rev_LM_vocab = [x for x in rev_vocab]
  rev_LM_vocab[data_utils.BOS_ID] = data_utils._EOS
  
  data_utils.data_to_token_ids(test_file, test_file + '.ids', vocab_path, tokenizer=data_utils.naive_tokenizer)
  
  test_set = read_test_data(test_file + '.ids')
  lm_test_output_file = FLAGS.test_output_file + '.ppl'
  intent_test_output_file = FLAGS.test_output_file + '.intent.hyp'
  tagging_test_output_file = FLAGS.test_output_file + '.tag.hyp'
    
  config = tf.ConfigProto(
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
  )
    
  with tf.Session(config=config) as sess:
    print("Loading model...")
    _, model_test = create_model(sess, len(vocab), len(tag_vocab), len(label_vocab), len(LM_vocab))
    print ("Loaded model with source_vocab_size=%d, target_vocab_size=%d, and label_vocab_size=%d, and lm_vocab_size=%d." % (len(vocab), len(tag_vocab), len(label_vocab), len(LM_vocab)))

    def run_eval(data_set, decode_output_file):
        with open(lm_test_output_file, 'wb') as f_lm:
          with open(intent_test_output_file, 'wb') as f_intent:
            with open(tagging_test_output_file, 'wb') as f_tagging:
              eval_loss = 0.0
              bucket_id = 0
              count = 0
              total_word_count = 0
              for i in xrange(len(data_set[bucket_id])):
                count += 1
                if count % 1000 == 0:
                    print ("Decoding utterance No. %d..." % count)
                eval_encoder_inputs, eval_encoder_inputs_shiftByOne, eval_tags, eval_tag_weights, eval_intent_weights, eval_lm_weights, eval_sequence_length, eval_labels = model_test.get_one(
                  data_set, bucket_id, i)
                eval_intent_weights = eval_tag_weights
    
                tagging_logits = []
                classification_logits = []
                
                _, step_loss, tagging_logits, classification_logits = model_test.joint_step(sess, eval_encoder_inputs, eval_encoder_inputs_shiftByOne, eval_lm_weights, eval_tags, eval_tag_weights, eval_labels, eval_intent_weights, 
                                             eval_sequence_length, bucket_id, True, use_attention=FLAGS.use_attention)
                
                f_lm.write('%.2f\n' % (-step_loss*(eval_sequence_length[0]-1)))
                f_lm.flush()
                eval_loss += step_loss*(eval_sequence_length[0])
                total_word_count += eval_sequence_length[0]                    
                
                hyp_label = None
                # intent results
                hyp_label = np.argmax(classification_logits[0],0)
                f_intent.write('%s\n' % rev_label_vocab[hyp_label])
                f_intent.flush()
                # tagging results
                f_tagging.write('%s\n' % ' '.join([rev_tag_vocab[np.argmax(x,1)] for x in tagging_logits[1:eval_sequence_length[0]]]))
                f_tagging.flush()

          eval_perplexity = math.exp(float(eval_loss)/total_word_count)
        return eval_perplexity
        
    valid_perplexity = run_eval(test_set, FLAGS.test_output_file)        
    print("  Eval perplexity: %.2f" % valid_perplexity)        
    sys.stdout.flush()              


def main(_):
  if FLAGS.decode_file:
    decode_file(FLAGS.test_file)
  else:
    train()

if __name__ == "__main__":
  tf.app.run()


