import torch
import time
from torch.cuda import random
from logger import Logger
from Node import Node, Global_Node, Select_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Recorder, Summary
from Trainer import Trainer


# init args
args = args_parser()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.split = args.node_num
args.global_model = args.local_model
# args.lr=100
print('Running on', args.device)
Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)
Summary(args)

# logs
logger = Logger(args)   


# init nodes
Global_node = Global_Node(Data.test_all, args)
Edge_nodes = [Node(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]
Select_node = Select_Node(args)

# train
for rounds in range(args.R * args.node_num):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    LR_scheduler(rounds, Edge_nodes, args)
    k = Select_node.random_select()
    for epoch in range(args.E):
        Train(Edge_nodes[k])
        recorder.validate(Edge_nodes[k])
    recorder.printer(Edge_nodes[k])
    print('-------------------------')

    Global_node.update(Edge_nodes[k])       # The server updates the corresponding model parameters. Not that the server only updates its local model, and the global model is not updated. This is done to avoid using intermediate variables.
    Edge_nodes[k].fork(Global_node)         # The node returns directly after reading the global model from the server
    Global_node.processing()                # The server generates a global model based on its local model. It can be seen that the calculation process of the central server and the edge node can be calculated simultaneously. Therefore, it can be considered as parallel computing. 

    # log
    recorder.validate(Global_node)
    recorder.printer(Global_node)
    logger.write(rounds=rounds + 1, test_acc=recorder.val_acc[str(Global_node.num)][rounds])

recorder.finish()
logger.close()

Summary(args)

