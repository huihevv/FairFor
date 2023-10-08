import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run TSF Task.")

    # dataset： electricity PeMSD7 PeMSD8 PeMSD4 traffic metr-la exchange_rate ECG5000 covid-19 solar_energy
    parser.add_argument("--dataset", type=str, default='ECG5000', help="The name of dataset.")
    parser.add_argument("--seq_len", type=int, default=12, help="Based on how many values in the past.")
    parser.add_argument("--horizon", type=int, default=12, help="How many steps to predict in the future.")
    parser.add_argument("--norm_method", type=str, default='cmax', choices={'max01', 'max11', 'std'}, help="Normalized method")
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument("--cluster_num", type=int, default=6, choices={10, 15, 20}, help="The number of cluster.")
    parser.add_argument("--seed", type=int, default=10, choices={1, 2, 3}, help="The random seed.")
    parser.add_argument("--embed_dim", type=int, default=10, choices={2, 4, 6, 8, 10}, help="The embedding dimension.")
    parser.add_argument("--input_dim", type=int, default=1, choices={1, 2}, help="The input dimension.")
    parser.add_argument("--output_dim", type=int, default=1, choices={1, 2}, help="The output dimension.")
    parser.add_argument("--rnn_units", type=int, default=64, choices={32, 64, 128}, help="The hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=2, help="The layers of STCL.")
    parser.add_argument('--cheb_k', default=2, type=int, help="")
    parser.add_argument('--column_wise', default=False, type=eval, help="Normailzed by column.")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=64, choices=[8, 64, 128, 256], help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-3,  choices=[6.25e-5, 8e-5], help="Generator learning rate.")
    parser.add_argument("--lr_d", type=float, default=5e-2,  choices=[5e-2, 8e-2], help="Discriminator learning rate.")
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument("--top_n_nodes", type=int, default=110, help="Select top-n nodes as neighbors.")

    return parser.parse_args()