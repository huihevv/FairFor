import utils.param_parser as parser
import utils.data_helpers as dh
import model.base_model2 as model
import utils.math_utils as ma
import utils.handler as ha
import utils.train_init as tr
import torch
from torch import optim
import time
import numpy as np
import torch.nn as nn

if __name__ == '__main__':
    args = parser.parameter_parser()

    # set seed
    tr.init_seed(args.seed)

    # select cuda
    print(torch.cuda.is_available())
    torch.cuda.set_device(args.cuda)

    # initial parameters  args.data_dir  args.model_dir
    if args.dataset == 'PeMSD8' or args.dataset == 'PeMSD4':
        args.data_dir = 'data/{}/{}.npz'.format(args.dataset, args.dataset)
    elif args.dataset == 'electricity' or args.dataset == 'wiki' or args.dataset == 'traffic' or args.dataset == 'PeMSD7':
        args.data_dir = 'data/{}/{}.npy'.format(args.dataset, args.dataset)
    elif args.dataset == 'solar_energy' or args.dataset == 'exchange_rate':
        args.data_dir = 'data/{}/{}.txt'.format(args.dataset, args.dataset)
    elif args.dataset == 'ECG5000' or args.dataset == 'covid-19':
        args.data_dir = 'data/{}/{}.csv'.format(args.dataset, args.dataset)
    elif args.dataset == 'metr-la':
        args.data_dir = 'data/{}/{}.h5'.format(args.dataset, args.dataset)
    else:
        print('error dataset')
    args.model_dir = 'result/{}'.format(args.dataset)
    # load dataset
    data = dh.load_dataset(args.data_dir, args.dataset)
    # initial parameters
    args.num_node = data.shape[1]

    # initialize model
    Model = model.FairFor(args.seq_len, args.num_node, args.horizon, args.input_dim, args.rnn_units, args.cheb_k,
                          args.embed_dim, args.output_dim, args.num_layers, args.batch_size, args.cluster_num,
                          args.top_n_nodes)
    discriminator1 = model.Discriminator1(args.rnn_units, args.cluster_num)

    # GPU setting
    Model.cuda()
    discriminator1.cuda()

    for p in Model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    tr.print_model_parameters(Model, discriminator1, only_num=False)

    # get data leoader
    train_loader, valid_loader, test_loader, scaler = dh.get_dataloader(data, args)

    # select optimize
    if args.optimizer == 'RMSProp':
        optimizer_G = optim.RMSprop(Model.parameters(), lr=args.lr, eps=1e-08)
    else:
        optimizer_G = optim.Adam(Model.parameters(), lr=args.lr, eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer_D1 = optim.Adam(discriminator1.parameters(), lr=args.lr_d, eps=1e-08, weight_decay=0, amsgrad=False)
    my_lr_scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer=optimizer_G, gamma=args.decay_rate)
    my_lr_scheduler_D1 = optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D1, gamma=args.decay_rate)

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    model_dir_list = []
    # Train the model
    before_train = time.time()
    for epoch in range(args.n_epochs):
        start_time = time.time()
        Model.train()

        # epoch loss
        generator_loss = []
        discriminator1_loss = []
        discriminator2_loss = []

        # train_x, train_y  [batch_size,seq_len,input_size]
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.cuda()
            target = target.cuda()
            valid = torch.autograd.Variable(
                torch.cuda.FloatTensor(inputs.size(0), args.seq_len + args.horizon, 1).fill_(1.0), requires_grad=False)
            fake = torch.autograd.Variable(
                torch.cuda.FloatTensor(inputs.size(0), args.seq_len + args.horizon, 1).fill_(0.0), requires_grad=False)


            Model.zero_grad()
            h, h_hat, h_star, fore, c, H, F = Model(inputs)

            f_loss = ma.Forecast_Loss(fore, target) + ma.Kmeans_Loss(H, F) + ma.OR_Loss(h, h_hat)
            f_loss.backward()
            optimizer_G.step()
            generator_loss.append(f_loss.item())

            discriminator1.zero_grad()
            h_hat_1, c_1 = discriminator1(h_hat, c)
            d1_loss = 0.1 * ma.Adversarial_Loss_1(h_hat_1, c_1)
            optimizer_D1.step()
            discriminator1_loss.append(d1_loss.item())


        print('| end of epoch {:3d} | time: {:5.2f}s | generator_loss {:5.4f} | discriminator1_loss {:5.4f}'.
              format(epoch, (time.time() - start_time), np.mean(generator_loss), np.mean(discriminator1_loss)))

        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = ha.validate(Model, valid_loader, args.norm_method, scaler,
                                              args.num_node, args.seq_len, args.horizon)

            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            # save model
            if is_best_for_now:
                model_dir = ha.set_model_dir(args.seq_len, args.horizon, args.model_dir, epoch + 1)
                model_dir_list.append(model_dir)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break

    ha.save_model(Model, model_dir_list[-1])
    print('Training took time {:5.2f}s.'.format((time.time() - before_train)))


    # Test the model
    before_evaluation = time.time()
    test_model = ha.load_model(model_dir_list[-1])

    performance_metrics = ha.validate(Model, test_loader, args.norm_method, scaler, args.num_node, args.seq_len, args.horizon)
    mae, mape, rmse, smape, wape, var = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse'], \
                                   performance_metrics['smape'], performance_metrics['wape'], performance_metrics['var']
    print(
        'Performance on test set: MAPE: {:5.4f} | MAE: {:5.4f} | RMSE: {:5.4f} | SMAPE: {:5.4f} | WAPE: {:5.4f}| VAR: '
        '{:5.4f}'.format(mape, mae, rmse, smape, wape, var))
    print('Evaluation took time: {:5.2f}s.'.format((time.time() - before_evaluation)))
    print('done')


