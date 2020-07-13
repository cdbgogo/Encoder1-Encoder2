import sys
try:
    from src.model.disconnected_rnn_initial import Disconnected_RNN as Disconnected_RNN_initial
    from src.model.disconnected_rnn import Disconnected_RNN as Disconnected_RNN_fake
    from src.utils_run import run, arg_parse
except Exception:
    from model.disconnected_rnn_initial import Disconnected_RNN as Disconnected_RNN_initial
    from model.disconnected_rnn import Disconnected_RNN as Disconnected_RNN_fake
    from utils_run import run, arg_parse

def main():
    """main"""
    args = arg_parse()
    debug_setting = {
        'mode': 'run',
        'test_number': 300,
        'debug': args.debug,
        'batch_size': 16,
        "max_epoch": 100,
        "eval_batch_interval": 10,
        "show": 10
    }
    global_type = args.global_type
    assert global_type in ["fake", "initial"]
    # fake is to fake the global information as words,
    # initial is to use the global information as the initial_state in rnn,
    # in our paper, we use the fake mode
    if global_type == "fake":
        run(Disconnected_RNN_fake, args, debug_setting)
    elif global_type == "initial":
        run(Disconnected_RNN_initial, args, debug_setting)
    else:
        raise NotImplementedError


if '__main__' == __name__:
    main()

