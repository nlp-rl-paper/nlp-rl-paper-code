import argparse

def build_argparser():
    parser = argparse.ArgumentParser(description="vizdoom trainer")

    parser.add_argument(
        "--REP_TYPE",
        help="representation type: nlp, viz, vec",
        type=str,
        default="nlp"
    )

    parser.add_argument(
        "--LEARNING_RATE",
        help="learning rate of algorithm",
        type=float,
        default=0.00015
    )

    parser.add_argument(
        "--FRAME_REPEAT",
        help="frame repeat of vizdoom simulator",
        type=int,
        #default=12
        default = 4
    )

    parser.add_argument(
        "--EPOCHS",
        help="training epochs",
        type=int,
        #default=50
        default = 100
    )

    parser.add_argument(
        "--PRINT_TRAINING_PROCESS",
        help="boolean indictaing if learning process will be printed",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--SKIP_TEST",
        help="boolean indictaing if testing of trained model should be skipped",
        action='store_false',
        default=True,
    )

    parser.add_argument(
        "--HIDDEN_UNITS",
        help="integer indictaing the number of hidden units within the neural network",
        type=int,
        default=64
    )

    parser.add_argument(
        "--FILTER_COUNT",
        help="integer indictaing the number of convolutional filters of the neural network",
        type=int,
        default=32
    )

    parser.add_argument(
        "--SENTANCE_LEN",
        help="integer indictaing the length of nlp sentances",
        type=int,
        default=200
    )

    parser.add_argument(
        "--BATCH_SIZE",
        help="integer indictaing the batch size of the training process",
        type=int,
        #default=150
        default=50
    )

    parser.add_argument(
        "--SCENARIO",
        help="string indictaing the scenario",
        type=str,
        default="basic"
    )

    parser.add_argument(
        "--ARCH",
        help="string indictaing neural network architecture",
        type=str,
        default="TextCNN"
    )


    parser.add_argument(
        "--ACTION_TYPE",
        help="string indicating the type of actions the agent can take, single button or multiple buttons [single,multy]",
        type=str,
        default="single"
    )

    parser.add_argument(
        "--N_CHANNELS",
        help="integer indicating the number of input channels to be used in visual agent",
        type=int,
        default=1
    )

    parser.add_argument(
        "--SEED",
        help="integer indicating the seed for numpy random",
        type=int,
        default=0
    )

    parser.add_argument(
        "--LEARNING_STEPS_PER_EPOCH",
        help="integer indicating the number of training steps per epoch",
        type=int,
        default=150
    )


    parser.add_argument(
        "--N_PATCHES",
        help="integer indicating the number of patches used in the natural language parser",
        type=int,
        default=3
    )

    parser.add_argument(
        "--LOAD_MODEL",
        help="boolean if a pretrained_model should be loaded",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--REVERSE_GREEN",
        help="boolean if a demon monster should be represented as green instead of red",
        action="store_true",
        default=False
    )

    return parser


def parse_arguments():
    parser = build_argparser()
    return parser.parse_args().__dict__




