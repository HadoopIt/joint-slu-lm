Joint Online Spoken Language Understanding and Language Modeling with Recurrent Neural Networks
==================

Tensorflow implementation of joint modeling of SLU (intent & slot filling) and LM with RNNs.

This package implements joint model Number 14 and 15 in Table 1 of the below reference paper [1], i.e. the joint model with *recurrent* (& *local*) intent + slot label context. The other joint models in the paper can be made with simple modifications of generate_encoder_output.py and seq_labeling.py.

**Setup**

* Tensorflow, version >= r0.9 (https://www.tensorflow.org/versions/r0.9/get_started/index.html)

**Usage**:
```bash
data_dir=data/ATIS_samples
model_dir=model_tmp
max_sequence_length=50  # max length for train/valid/test sequence
use_local_context=False # boolean, whether to use local context
DNN_at_output=True # boolean, set to True to use one hidden layer DNN at task output

python run_rnn_joint.py --data_dir $data_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --use_local_context $use_local_context \
      --DNN_at_output $DNN_at_output
```

**Reference**

[1] Bing Liu, Ian Lane, "Joint Online Spoken Language Understanding and Language Modeling with Recurrent Neural Networks", SIGdial, 2016 (<a href="http://www.sigdial.org/workshops/conference17/proceedings/pdf/SIGDIAL03.pdf" target="_blank">PDF</a>)

```
@InProceedings{liu-lane:2016:SIGDIAL,
  author    = {Liu, Bing  and  Lane, Ian},
  title     = {Joint Online Spoken Language Understanding and Language Modeling With Recurrent Neural Networks},
  booktitle = {Proceedings of the 17th Annual Meeting of the Special Interest Group on Discourse and Dialogue},
  month     = {September},
  year      = {2016},
  address   = {Los Angeles},
  publisher = {Association for Computational Linguistics},
  pages     = {22--30},
  url       = {http://www.aclweb.org/anthology/W16-3603}
}
```

**Contact** 

Feel free to email liubing@cmu.edu for any pertinent questions/bugs regarding the code. 
