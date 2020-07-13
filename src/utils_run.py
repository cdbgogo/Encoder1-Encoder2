import os.path
import numpy as np
try:
    from src.utils import data_utils
    from src.model import layer
    from src.utils import config, logger_config
    from src.trainer_multi_label import Trainer_multilabel
    from src.trainer import Trainer

except ImportError:
    from utils import data_utils, config, logger_config
    from model import layer
    from trainer_multi_label import Trainer_multilabel
    from trainer import Trainer
import logging

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def load_vocab(v):
    """
    load vocabulary which is builded previously
    """
    res = {}
    for line in open(v):
        items = line.strip().split('\t')
        res[items[0]] = items[1]
    return res


def run(model_class, args, debug_setting, preprocess_func=None):
    """run"""
    config_path = args.config
    config_file = os.path.join(project_dir, config_path)

    pre_config = config.pre_config(config_file)
    main_config = config.Config(config=pre_config)
    log_file = logger_config.logger_init(project_dir, main_config.model.model_name, main_config.data.dataset)
    main_config.log_file = log_file

    updata_config_by_args(args, main_config)
    logging.info('config_file is: {}'.format(config_file))
    logging.info('config path is: {}'.format(config_path))

    if main_config.data.type == 'single_label':
        trainer_specific = Trainer
    elif main_config.data.type == 'multi_label':
        trainer_specific = Trainer_multilabel
    else:
        raise NotImplementedError

    embedding = None
    if hasattr(main_config.model, 'embedding_initialize') \
            and main_config.model.embedding_initialize == 'pretrained' \
            and hasattr(main_config.data, 'embedding_path'):
        embedding_path = os.path.join(project_dir, main_config.data.embedding_path)
        logging.info('load embedding from {}'.format(embedding_path))
        embedding = np.loadtxt(embedding_path)
        main_config.model.emb_size = embedding.shape[1]
        logging.info('load done')
    else:
        logging.info('use random embedding')

    vocab_path = os.path.join(project_dir, main_config.data.vocab_path)
    vocab = load_vocab(vocab_path)
    main_config.model.vocab_size = len(vocab)
    logging.info('vocab_size is : {}'.format(len(vocab)))

    model = model_class(main_config.model, embedding)
    trainer = trainer_specific(model, main_config, preprocess_func=preprocess_func)
    if main_config.predict_mode:
        logging.info('This is to predict')
        trainer.load_data()
        trainer.load_saver_and_predict()
        exit()
    if debug_setting['debug']:
        logging.info('This is to debug')
        trainer.load_data_debug(debug_setting)
        trainer.train()
    else:
        logging.info('This is to experiment')
        trainer.load_data()
        trainer.train()


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("-e", "--embedding_initialize", type=str, help="embedding initialize")
    parser.add_argument("-es", '--embedding_size', type=int, help="embedding size")
    parser.add_argument("-o", "--optimizer", type=str, help="optimizer")
    parser.add_argument("-l", "--learning_rate", type=float)
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-w", "--window_size", type=int)

    parser.add_argument("-ah", "--attention_head", type=int)
    parser.add_argument("-ab", "--attention_block", type=int)
    parser.add_argument("-ad", "--attention_drop", type=float)
    parser.add_argument("-msl", "--max_sequence_length", type=int)
    parser.add_argument("-ebi", "--eval_batch_interval", type=int, help="eval per patch, default config is in basic config")
    parser.add_argument("-ebs", "--eval_batch_size", type=int, help="eval batch size")
    parser.add_argument("-du", "--disan_units", type=int, help="disan_units")
    # parser.add_argument("-p", '--process_way', type=str, help="clean_str or nltk_tokenizer")
    parser.add_argument("-d", '--debug', action='store_true', help="if --debug, run into debug mode")
    parser.add_argument("-r", '--rnn_dropout', type=float, help="rnn dropout, default=1")
    parser.add_argument("-fd", '--fc_drop', type=float, help="fc dropout, default=0.2")
    parser.add_argument("-f", '--xavier_factor', type=float, help="x = sqrt(factor / (in + out)), default=6")
    # xavier_factor is not used

    parser.add_argument("-fa1", '--fc_activation_1', type=str, help="activation in fc 1")
    parser.add_argument("-fa2", '--fc_activation_2', type=str, help="activation in fc 2")
    parser.add_argument("-aa", '--attention_activation', type=str, help="activation in attention")
    parser.add_argument("-is", '--inception_size', type=str, help="inception size")
    parser.add_argument("-n", '--neighbor_embedding', action='store_true',
                        help="if --neighbor_embedding, use neighbor embedding")
    parser.add_argument("-fbe", '--forbid_batch_eval', action='store_true',
                        help="if --forbid_batch_eval, don't eval in batch interval")
    parser.add_argument("-fee", '--forbid_epoch_eval', action='store_true',
                        help="if --forbid_epoch_eval, don't eval in end of epoch")
    parser.add_argument("-pa", '--pre_attention', action='store_true',
                            help="if --pre_attention, use pre_attention")
    parser.add_argument("-nf", '--no_fast', action='store_true',
                            help="if --nf, use previous disan")
    parser.add_argument("-cv", '--clip_value', action='store_true',
                            help="if --cl use clip_value in grad_clip instead of clip_grad")
    parser.add_argument("-ld", '--lr_decay', action='store_true',
                            help="if --ld lr_decay")
    parser.add_argument("-at", '--attention_type', type=str, help="attention_type")
    parser.add_argument("-et", '--encode_type', type=str, help="encode_type")
    parser.add_argument("-ft", '--feature_type', type=str, help="feature_type")
    # key argument in run_drnn, determines the encoder2.
    # choose from [cnn/dpcnn/rnn], rnn is ths drnn

    parser.add_argument("-l2", '--l2_loss', action='store_true',
                        help="if --l2_loss, use l2_loss")
    parser.add_argument("-gs", '--global_size', type=int, help="global_size to concat")
    parser.add_argument("-efe", '--encoder_fixed_epoch', type=float, default=0.0)
    parser.add_argument("-gt", '--global_type', type=str, default="fake")
    parser.add_argument("-cd", '--checkpoint_dir', type=str)
    parser.add_argument("-pm", '--predict_mode', action='store_true', help="if --pm predict_mode")
    return parser.parse_args()


def updata_config_by_args(args, config):
    # public
    config.predict_mode = args.predict_mode
    if config.predict_mode:
        config.trainer.checkpoint_dir = args.checkpoint_dir
    config.model.pre_attention = args.pre_attention
    config.trainer.clip_value = args.clip_value
    config.trainer.encoder_fixed_epoch = args.encoder_fixed_epoch
    config.model.encoder_fixed_epoch = args.encoder_fixed_epoch
    config.trainer.lr_decay = args.lr_decay
    config.model.l2_loss = args.l2_loss
    config.model.fixedrng = 1996
    config.trainer.debug = args.debug
    if args.embedding_size is not None:
        config.model.emb_size = args.embedding_size
    if args.batch_size is not None:
        config.trainer.batch_size = args.batch_size
    if args.window_size is not None:
        config.model.window_size = args.window_size
    if args.encode_type is not None:
        assert args.encode_type in {"transformer", "disan", "other_transformer", "cnn", "w", "rnn", "attend_rnn"}
        config.model.encode_type = args.encode_type

    if args.attention_type is not None:
        # assert args.attention_type in {"pre_attention", "same_fake_input", "same_init", "diff_init", "attend_init",
        #                                "diff_fake_input", "same_concat", "diff_concat", "same_init_diff_concat", "after"}
        assert args.attention_type in {"same_init", "attend_init"}
        config.model.attention_type = args.attention_type
    else:
        config.model.attention_type = None

    if args.feature_type is not None:
        assert args.feature_type in ["cnn", "rnn", "dpcnn"]
        config.model.feature_type = args.feature_type
    else:
        config.model.feature_type = "rnn"
    str_info = "use {} as the feature extractor".format(config.model.feature_type)
    logging.info(str_info)

    if args.attention_block is not None:
        config.model.attention_block = args.attention_block
    if args.attention_head is not None:
        config.model.attention_head = args.attention_head
    if args.attention_drop is not None:
        config.model.attention_drop = args.attention_drop
    config.model.global_size = args.global_size

    if args.inception_size is not None:
        try:
            inception_size = eval(args.inception_size)
            config.model.inception_size = inception_size
            logging.info('inception size is {}'.format(inception_size))
        except:
            logging.info('use default inception size')
    if args.optimizer is not None:
        if args.optimizer in ['Adadelta', 'Ada', 'ada', 'adadelta']:
            optim = 'Adadelta'
        elif args.optimizer in ['Adam', 'adam']:
            optim = 'Adam'
        elif args.optimizer in ['nAdam', 'Nadam']:
            optim = 'Nadam'
        else:
            optim = args.optimizer
        assert optim in ["Adam", "Adadelta", "Adamdecay", "Nadam"]
        config.trainer.optimizer = optim
    if args.learning_rate is not None:
        config.trainer.learning_rate = args.learning_rate
    if args.eval_batch_interval is not None:
        config.trainer.eval_batch_interval = args.eval_batch_interval
    if args.eval_batch_size is None:
        config.trainer.eval_batch_size = config.trainer.batch_size
    else:
        config.trainer.eval_batch_size = args.eval_batch_size
    if args.forbid_epoch_eval:
        config.trainer.epoch_eval = False
    if args.forbid_batch_eval:
        config.trainer.batch_eval = False

    if args.disan_units is not None:
        config.model.disan_units = args.disan_units
    else:
        config.model.disan_units = 300
    if args.max_sequence_length is not None:
        config.model.max_sequence_length = args.max_sequence_length
    if args.fc_drop is not None:
        config.model.fc_drop = args.fc_drop
    if args.rnn_dropout is not None:
        config.model.rnn_drop = args.rnn_dropout
    if args.xavier_factor is not None:
        config.model.xavier_factor = args.xavier_factor
    if args.no_fast:
        config.model.disan_type = 'origin'

    activation_lst = ['relu', 'leaky_relu', 'tanh', 'p_relu', 'None']
    if args.fc_activation_1 is not None:
        if args.fc_activation_1 not in activation_lst:
            logging.error('activation should be in {}'.format(activation_lst))
            exit()
        config.model.fc_activation_1 = args.fc_activation_1
    if args.fc_activation_2 is not None:
        if args.fc_activation_2 not in activation_lst:
            logging.error('activation should be in {}'.format(activation_lst))
            exit()
        config.model.fc_activation_2 = args.fc_activation_2
    if args.attention_activation is not None:
        if args.attention_activation not in activation_lst:
            logging.error('activation should be in {}'.format(activation_lst))
            exit()
        config.model.attention_activation = args.attention_activation

    # print config.data
    public_dataset = ['ag_news', 'dbpedia', 'yelp_review_full', 'yelp_review_polarity',
                      'yahoo_answers', 'amazon_review_full', 'amazon_review_polarity']
    if config.data.dataset in public_dataset:
        # only for public data
        # config.model.embedding_initialize = 'pretrained'
        config.model.embedding_initialize = 'random'
        if args.embedding_initialize is not None:
            config.model.embedding_initialize = args.embedding_initialize
        # if args.process_way is not None:
        #     if args.process_way not in ['clean_str', 'nltk_tokenizer']:
        #         print("process way should in ['clean_str', 'nltk_tokenizer']")
        #         exit()
        #     config.data.process_way = args.process_way
        # for path_ in ['train_data', 'test_data', 'vocab_path', 'embedding_path']:
        #     origin_p = getattr(config.data, path_)
        #     # print(path_, origin_p)
        #     new_p = origin_p.replace('process_way', config.data.process_way)
        #     setattr(config.data, path_, new_p)
        config.data.neighbor_embedding = args.neighbor_embedding
        config.data.embedding_path = config.data.embedding_path.replace \
            ('neighbor_embedding', str(args.neighbor_embedding))

    config_str = '\n{\n\t\"model\":{\n'
    for arg in vars(config.model):
        val = getattr(config.model, arg)
        config_str += '\t\t{}: {},\n'.format(arg, val)

    config_str += '}\n\t\"data\":{ \n'
    for arg in vars(config.data):
        val = getattr(config.data, arg)
        config_str += '\t\t{}: {},\n'.format(arg, val)
    config_str += '}\n}'

    config_str += '}\n\t\"trainer\":{ \n'
    for arg in vars(config.trainer):
        val = getattr(config.trainer, arg)
        config_str += '\t\t{}: {},\n'.format(arg, val)
    config_str += '}\n}'
    config.log_string = config_str
    config.model.batch_size = config.trainer.batch_size
