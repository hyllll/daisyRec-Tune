import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='test recommender')
    # tuner settings
    parser.add_argument('--score_metric', 
                        type=str, 
                        default='ndcg', 
                        help='use which metric to define hyperopt score')
    parser.add_argument('--tune_epochs', 
                        type=int, 
                        default=30, 
                        help='tuning epochs')
    parser.add_argument('--tune_pack', 
                        type=str, 
                        default='{}', 
                        help='Tuner parameter dictionary, type is JSON string')
    # common settings
    parser.add_argument('--config_file', 
                        type=str, 
                        default='mf_small', 
                        help='config json file')
    parser.add_argument('--device', 
                        type=str, 
                        default='gpu', 
                        help='cpu or gpu')
    parser.add_argument('--problem_type', 
                        type=str, 
                        default='point', 
                        help='pair-wise or point-wise')
    parser.add_argument('--algo_name', 
                        type=str, 
                        default='vae', 
                        help='algorithm to choose')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='10core', 
                        help='dataset preprocess op.: origin/Ncore')
    parser.add_argument('--topk', 
                        type=int, 
                        default=50, 
                        help='top number of recommend list')
    parser.add_argument('--test_method', 
                        type=str, 
                        default='tfo', 
                        help='method for split test,options: ufo/loo/fo/tfo/tloo')
    parser.add_argument('--val_method', 
                        type=str, 
                        default='tfo', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--test_size', 
                        type=float, 
                        default=.2, 
                        help='split ratio for test set')
    parser.add_argument('--val_size', 
                        type=float, 
                        default=.1, help='split ratio for validation set')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=1000, 
                        help='No. of candidates item for predict')
    parser.add_argument('--sample_method', 
                        type=str, 
                        default='uniform', 
                        help='negative sampling method mixed with uniform, options: item-ascd, item-desc')
    parser.add_argument('--sample_ratio', 
                        type=float, 
                        default=0, 
                        help='mix sample method ratio, 0 for all uniform')
    parser.add_argument('--init_method', 
                        type=str, 
                        default='', 
                        help='weight initialization method')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=0, 
                        help='negative sampling number')
    parser.add_argument('--loss_type', 
                        type=str, 
                        default='CL', 
                        help='loss function type')
    parser.add_argument('--tune_testset', 
                        type=int, 
                        default=0, 
                        help='whether directly tune on dataset')
    # algo settings
    parser.add_argument('--factors', 
                        type=int, 
                        default=32, 
                        help='latent factors numbers in the model')
    parser.add_argument('--reg_1', 
                        type=float, 
                        default=0., 
                        help='L1 regularization')
    parser.add_argument('--reg_2', 
                        type=float, 
                        default=0.001, 
                        help='L2 regularization')
    parser.add_argument('--kl_reg', 
                        type=float, 
                        default=0.5, 
                        help='VAE KL regularization')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.5, 
                        help='dropout rate')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001, 
                        help='learning rate')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=50, 
                        help='training epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='batch size for training')
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=2, 
                        help='number of layers in MLP model')
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.02, 
                        help='elastic net parameter')
    parser.add_argument('--elastic', 
                        type=float, 
                        default=0.5, 
                        help='ratio if lasso result, 0 for ridge-regression, 1 for lasso-regression')
    parser.add_argument('--pop_n', 
                        type=int, 
                        default=100, 
                        help='Initial selected number of Most-popular')
    parser.add_argument('--maxk', 
                        type=int, 
                        default=40, 
                        help='The (max) number of neighbors to take into account')
    parser.add_argument('--layer_size', 
                        nargs='?', 
                        default='[64,64,64]',
                        help='NGCF: Output sizes of every layer')
    parser.add_argument('--node_dropout_flag', 
                        type=int, 
                        default=1,
                        help='NGCF: 0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout',
                        type=float,  
                        nargs='?', 
                        default=0.1,
                        help='NGCF: Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', 
                        type=float, 
                        nargs='?', 
                        default=0.1,
                        help='NGCF: Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--act_func', 
                        type=str, 
                        default='relu', 
                        help='activation method in interio layers')
    parser.add_argument('--out_func', 
                        type=str, 
                        default='sigmoid', 
                        help='activation method in output layers')
    parser.add_argument('--no_batch_norm', 
                        action='store_false', 
                        default=True, 
                        help='whether do batch normalization in interior layers')
    args = parser.parse_args()

    return args
