
Code reproduction of EMNLP-2019 Paper: Enhancing Local Feature Extraction with Global Representation for Neural Text Classification

1. Data download:
    1. download data from here: https://drive.google.com/drive/u/1/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M 
    and put to data/raw_data
    2. download glove.840B.300d.txt and put to data/raw_data

2. requirements

        nltk
        tensorflow 1.4 or later
        python 2.7
    
3. Data preprocess:
    
    in train, split the same number samples with test as the dev dataset

        python preprocess/process_public.py -p data/raw_data/ag_news_csv -n 50000
        -n: vocab
        -p data_path
        
4. Run

        python src/run_drnn.py conf/disconnected_rnn/ag_news.config -b 128
            -b 128: batch_size
            -msl 100: max_seq_len 
            -ebi 2000: eval batch interal
            -o Adadelta: optimizer (Adam/Adadelta)
            -l 1: learning rate=1
            -r 0.1: rnn dropout rate
            -fd 0.2: fc dropout
            -et cnn: global encoder type (cnn rnn attend_rnn other_transformer transformer)
            -at same_init: interaction mode (same_init, attend_init)
            -ft cnn: feature extractor type (rnn(drnn),cnn   only avaliable in drnn, default is drnn, cnn)
            
            -ld: learning_decay
            -fee: fobid eval at the end of epoch
            -l2: use l2 regularization
        
5. details command

        command-details 
  