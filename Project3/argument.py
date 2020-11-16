def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--action_size', type=int, default=4, help='number of valid actions')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--memory_cap', type=int, default=10000, help='memory capacity')
    parser.add_argument('--n_episode', type=int, default=60000, help='num of total training episodes')
    parser.add_argument('--n_step', type=int, default=5000, help='num of steps to start training')
    parser.add_argument('--update_f', type=int, default=5000, help='network update frequency')
    parser.add_argument('--explore_step', type=int, default=200000, help='steps of epsilon decay')
    parser.add_argument('--load_model', type=bool, default=False, help='whether continue training')    
    parser.add_argument('--algorithm', type=str, default='DDQN', help='type of training algorithm: DQN or DDQN')
    return parser
