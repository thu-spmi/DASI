import os, argparse, logging, time
from utils.configs import configs
from models.glove.run_model import Run_GloVe
# from models.word2vec.run_model import Run_W2v
from models.dasi.run_model import Run_DASI
from models.counterfitting.run_model import Run_CF

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=['glove', 'w2v', 'AE', 'CPAE', 'CF', 'DASI'], default='AE')
parser.add_argument("--config", type=str, default='AE_pretrain_big_corpus')
parser.add_argument("--pretrain", type=str, default='')
parser.add_argument("--semantic", type=str, default='wordnet')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--syn_coef", type=float, default=1.0)
parser.add_argument("--retrofit_coef", type=float, default=0.0)
parser.add_argument("--CP_coef", type=float, default=32.0)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--use_emb", type=str, default='input')
parser.add_argument("--sem_model", type=str, default='dasi')

# parser.add_argument("--indp", default=False, action="store_true")

args = parser.parse_args()
mapping = {'glove':Run_GloVe,
           # 'w2v':Run_W2v,
           'CF':Run_CF,
           'AE':Run_DASI,
           'CPAE': Run_DASI,
           'DASI':Run_DASI}
training_cls = mapping[args.model]

def init_logging_handler(args, config):
      stderr_handler = logging.StreamHandler()
      if not os.path.exists('./log'):
          os.mkdir('./log')
      log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
      save_dir = os.path.join('./log', args.model ,'{}_{}_{}_{}_sd{}_cp{}_syn{}/'.format(args.pretrain, args.sem_model ,args.semantic,
        args.use_emb, args.seed, args.CP_coef ,config['syn_coef']))
      if not os.path.exists(os.path.join('./log', args.model)):
        os.mkdir(os.path.join('./log', args.model))
      if not os.path.exists(save_dir):
        os.mkdir(save_dir)
      file_handler = logging.FileHandler(save_dir + '{}_{}_{}_{}_sd{}.txt'.format(log_time, args.model, args.config, args.pretrain, args.seed))
      logging.basicConfig(handlers=[stderr_handler, file_handler])
      logger = logging.getLogger()
      logger.setLevel(logging.INFO)
      config['save_dir'] = save_dir

if args.config in configs:
    config = configs[args.config]
    if args.pretrain:
      config['pretrain_embs_path'] = args.pretrain
    config['model'] = args.model
    config['seed'] = args.seed
    config['syn_coef'] = args.syn_coef if args.model != 'CPAE' else 0.
    config['retrofit_coef'] = args.retrofit_coef if args.model != 'CPAE' else 0.
    config['CP_coef'] = args.CP_coef
    config['batch_size'] = args.batch_size
    config['semantic'] = args.semantic
    config['use_emb'] = args.use_emb
    config['sem_model'] = args.sem_model
    init_logging_handler(args, config)
    c = training_cls(**config)
    logging.info(config)
    c.train(args.model)
else:
    print('oops, wrong config name ...')

