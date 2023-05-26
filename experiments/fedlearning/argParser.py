import argparse



class argParser(object):
        def __init__(self):
                self.parser = argparse.ArgumentParser()
                self.__parse()

        def parse_interactive(self,args):
                for key,value in args._get_kwargs():
                        if key=="interactive":continue
                        value = input(key+" default: \t" +str(value) +"\t")
                        # print(args,value)
                return args

        def __parse(self):
                self.parser.add_argument("--interactive", nargs='?',const="0", type=int,help="Choose inputs interactively",
                                        default=0)
                self.parser.add_argument("--nclients", nargs='?',const=2, type=int,help="number connected clients",
                                        default=2)
                self.parser.add_argument("--algorithm", nargs='?',const="fedavg", type=str,help="Federated Learning algorithm",
                                        default="fedavg")
                self.parser.add_argument("--rounds", nargs='?',const=1, type=int,help="number of communication rounds",
                                        default=1)
                ##################################################################################################
                self.parser.add_argument("--dataset_type", nargs='?',const="cifar10", type=str,help="number of local training epochs before uploading to server",
                                        default="cifar10")
                self.parser.add_argument("--alpha", nargs='?',const=2, type=float,help="Dirichlet distribution parameter ",
                                        default=None)
                self.parser.add_argument("--ni", nargs='?',const=2, type=float,help="Degree of non-IID",
                                        default=0.42)
                self.parser.add_argument("--emd", nargs='?',const=2, type=int,help="specify EMD value for distributed client datasets in feature space",
                                        default=None)
                ##################################################################################################
                self.parser.add_argument("--local_epochs", nargs='?',const=2, type=int,help="number of local training epochs before uploading to server ",
                                        default=2)
                self.parser.add_argument("--n_local_runs", nargs='?',const=3, type=int,help="number of local symmetric models ",
                                        default=3)
                self.parser.add_argument("--local_batchsize", nargs='?',const=64, type=int,help="client batchsize",
                                        default=128)
                self.parser.add_argument("--lr", nargs='?',const=0.001, type=float,help="client learning rate",
                                        default=0.001)
                ##################################################################################################
                self.parser.add_argument("--flag_common_weight_init", nargs='?',const=1, type=int,help="common client weight initialization",
                                        default=0)
                self.parser.add_argument("--flag_preload_models", nargs='?',const=2, type=int,help="load trained client models from file",
                                        default=0)
                ##################################################################################################
                self.parser.add_argument("--cf_nruns", nargs='?',const=2, type=int,help="curve finding procedure number of runs",
                                        default=10)
                self.parser.add_argument("--cf_batchsize", nargs='?',const=512, type=int,help="curve finding procedure batch size ",
                                        default=2048)
                self.parser.add_argument("--cf_curve", nargs='?',const=2, type=str,help="type of curve [flat,chain,bezier2,bezier3]",
                                        default="bezier2")
                self.parser.add_argument("--cf_verbose", nargs='?',const=0, type=int,help="curve finding validation monitoring",
                                        default=0)
                self.parser.add_argument("--cf_lr", nargs='?',const=2, type=float,help="curve finding learning rate",
                                        default=0.001)
                self.parser.add_argument("--mc_size", nargs='?',const=128, type=int,help="MonteCarlo sample size",
                                        default=4096)
                self.parser.add_argument("--cf_preload_theta", nargs='?',const=128, type=int,help="preload theta weights from file",
                                        default=0)

                                        
                ##################################################################################################
                self.parser.add_argument("--gamma", nargs='?',const=0.1, type=float,help="FedBN gamma parameter",
                                        default=0.1)
                self.args = self.parser.parse_args()







