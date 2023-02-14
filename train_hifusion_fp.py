from torch import optim 
import torch 
from data import *
from model import *
from sklearn.model_selection import KFold,train_test_split
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm 
from dgllife.utils import EarlyStopping, Meter,RandomSplitter
from prettytable import PrettyTable
import pandas as pd 
import json
from torch.utils.data import DataLoader

import sklearn.metrics as metrics
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.metrics import precision_recall_curve,average_precision_score

from data_split import *
def compute_metrics(y_true, y_prob):
    
    y_pred = np.array(y_prob) > 0.5
    BACC = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    ACC = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred, average = 'binary')
    Prec = precision_score(y_true, y_pred, average= 'binary')
    Rec = recall_score(y_true, y_pred, average = 'binary')
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    ap = average_precision_score(y_true, y_prob)
    return ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap




def compute_kl_loss(p, q, pad_mask=None):
	p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
	q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

	# pad_mask is for seq-level tasks
	if pad_mask is not None:
		p_loss.masked_fill_(pad_mask, 0.)
		q_loss.masked_fill_(pad_mask, 0.)

	# You can choose whether to use function "sum" and "mean" depending on your task
	p_loss = p_loss.mean()
	q_loss = q_loss.mean()

	loss = (p_loss + q_loss) / 2
	return loss






def get_cold_split_data_loader(dataset_name, cold_split_scheme, descript_set):
    label_filenames = data_path + dataset_name + '/labeled_triples_clean.csv'
    smiles_filenames = data_path + dataset_name + '/drug_set.json'
    context_filenames = data_path + dataset_name + '/context_set.json'
    label_df = pd.read_csv(label_filenames)
    drug1 = label_df['drug_1'].values
    drug2 = label_df['drug_2'].values
    context = label_df['context'].values
    labels = label_df['label'].values
    drug2smiles = {}
    with open(smiles_filenames, "r") as read_file:
        dict_d2s = json.load(read_file)
        for key, value in dict_d2s.items():
            drug2smiles[key] = value['smiles']

    context2features = {}
    with open(context_filenames, "r") as read_file:
        context2features = json.load(read_file)
    
    # print(len(context2features[context[0]]))
     
    train_loaders = []
    valid_loaders = []
    test_loaders = []
    for i in range(5):
        if cold_split_scheme == 'cold_drug':
           train_data, valid_data, test_data = cold_drug_split(label_filenames, i)
        elif cold_split_scheme == 'cold_cell':
            train_data, valid_data, test_data = cold_celllines_split(label_filenames, i)
        elif cold_split_scheme == 'cold_drugs':
            train_data, valid_data, test_data =cold_drugpairs_split(label_filenames, i)
        elif cold_split_scheme == 'both_cold':
            train_data, valid_data, test_data =both_cold_split(label_filenames, i)


         

        train_set = FPDataset(train_data['drug_1'].to_list(), 
                               train_data['drug_2'].to_list(), drug2smiles, 
                                train_data['context'].to_list(), context2features, 
                                train_data['label'].to_list(), dataset_name, descript_set=descript_set)
        val_set = FPDataset(valid_data['drug_1'].to_list(), valid_data['drug_2'].to_list(),
                             drug2smiles, valid_data['context'].to_list(), context2features,
                              valid_data['label'].to_list(), dataset_name, descript_set=descript_set)
        test_set = FPDataset(test_data['drug_1'].to_list(), test_data['drug_2'].to_list(), drug2smiles, 
                                test_data['context'].to_list(), context2features, 
                                test_data['label'].to_list(), dataset_name, descript_set=descript_set)

        train_loader = DataLoader(train_set, batch_size= BATCH_SIZE,   shuffle=True, num_workers=8)
        test_loader = DataLoader(val_set, batch_size= BATCH_SIZE, shuffle=False, num_workers=8)
        valid_loader = DataLoader(test_set, batch_size= BATCH_SIZE, shuffle=False, num_workers=8)
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)
        # break

    return train_loaders, valid_loaders, test_loaders






dataset_name = 'drugcombdb'
data_path = '.'
def get_dataloaders(dataset_name, data_path, descript_set):


    label_filenames = data_path + dataset_name + '/labeled_triples_clean.csv'
    smiles_filenames = data_path + dataset_name + '/drug_set.json'
    context_filenames = data_path + dataset_name + '/context_set.json'
    label_df = pd.read_csv(label_filenames)
    drug1 = label_df['drug_1'].values
    drug2 = label_df['drug_2'].values
    context = label_df['context'].values
    labels = label_df['label'].values
    drug2smiles = {}
    with open(smiles_filenames, "r") as read_file:
        dict_d2s = json.load(read_file)
        for key, value in dict_d2s.items():
            drug2smiles[key] = value['smiles']

    context2features = {}
    with open(context_filenames, "r") as read_file:
        context2features = json.load(read_file)
    
     
    train_loaders = []
    valid_loaders = []
    test_loaders = []
    kf = KFold(n_splits=5, shuffle=True)#划分成8:1 之前是5，

    for split, (train_index, test_index) in enumerate( kf.split(labels)):
        test_index, valid_index = train_test_split(test_index, test_size=0.5, random_state=42)
        train_drug_2_cv = np.array(drug2)[train_index]
        train_drug_1_cv = np.array(drug1)[train_index]
        train_context_cv = np.array(context)[train_index]
        train_Y_cv = np.array(labels)[train_index]
        
        test_drug_2_cv = np.array(drug2)[test_index]
        test_drug_1_cv = np.array(drug1)[test_index]
        test_context_cv = np.array(context)[test_index]
        test_Y_cv = np.array(labels)[test_index]

        valid_drug_2_cv = np.array(drug2)[valid_index]
        valid_drug_1_cv = np.array(drug1)[valid_index]
        valid_context_cv = np.array(context)[valid_index]
        valid_Y_cv = np.array(labels)[valid_index]


        train_set = FPDataset(train_drug_1_cv, train_drug_2_cv, drug2smiles, train_context_cv, context2features, train_Y_cv, dataset_name, descript_set=descript_set)
        val_set = FPDataset(valid_drug_1_cv, valid_drug_2_cv, drug2smiles, valid_context_cv, context2features, valid_Y_cv, dataset_name, descript_set=descript_set)
        test_set = FPDataset(test_drug_1_cv, test_drug_2_cv, drug2smiles, test_context_cv, context2features, test_Y_cv, dataset_name, descript_set=descript_set)

         

        train_loader = DataLoader(train_set, batch_size= BATCH_SIZE,   shuffle=True, num_workers=8)
        test_loader = DataLoader(val_set, batch_size= BATCH_SIZE, shuffle=False, num_workers=8)
        valid_loader = DataLoader(test_set, batch_size= BATCH_SIZE, shuffle=False, num_workers=8)

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)
        # break
       
    return train_loaders, valid_loaders, test_loaders




def run_a_train_epoch(device, epoch,num_epochs, model, data_loader, loss_criterion, optimizer, scheduler):
    model.train()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    # print(len(data_loader))
     
        

    for id,  (*x, y) in tbar:
        for i in range(len(x)):
            x[i] = x[i].to(device)
        y = y.to(device)
        batch_size = x[0].size(0)
        optimizer.zero_grad()
        output = model(*x) 
          
        loss = loss_criterion(output.view(-1), y.view(-1))   


        # input_x = copy.deepcopy(x)

        # output1 = model(*x) 

        # output2 = model(*input_x) 

        # ce_loss = 0.5 * (loss_criterion(output1, y.squeeze(1) ) + loss_criterion(output2, y.squeeze(1) ))
        # kl_loss = compute_kl_loss(output1, output2)
        # α = 10
        # loss = ce_loss + α * kl_loss
         
        loss.backward()   
        optimizer.step()
        scheduler.step()
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}')
        # tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f} CELoss={ce_loss.item()  :.3f} KLLoss={kl_loss.item()  :.3f}')


    

def run_an_eval_epoch(model, data_loader):
    model.eval()
    with torch.no_grad():
        preds =  torch.Tensor()
        trues = torch.Tensor()
        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
            logits =  model(*x)
            preds = torch.cat((preds, logits.cpu()), 0)
            # preds_score = torch.nn.functional.softmax(logits)[:,1]
            # preds = torch.cat((preds, preds_score.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)

        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
    return preds, trues
            




BATCH_SIZE = 512
# simple:8
# uncorrelated: 200
# fragment: 101   better 0.846
# graph: 19
# surface: 49 ok 0.844
# druglikeness: 24
# logp: 13 ok: 0.844
# refractivity: 11
# estate: 25
# charge: 18
# general: 12

descript2size = {'simple':8, 'uncorrelated':200, 'fragment':101, 'graph':19, 'surface':49,
                   'logp':13, 'refractivity':11, 'estate':25, 'charge':18, 'general':12, 'surface-logp':13+49}
descript_set = 'fragment'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# train_loaders, valid_loaders, test_loaders = get_dataloaders(dataset_name, data_path, descript_set)
train_loaders, valid_loaders, test_loaders =  get_cold_split_data_loader(dataset_name, 'both_cold', descript_set)


test_ACC=np.zeros((5,1))
test_BACC=np.zeros((5,1)) 
test_Prec=np.zeros((5,1)) 
test_Rec=np.zeros((5,1)) 
test_F1=np.zeros((5,1)) 
test_roc_auc=np.zeros((5,1)) 
test_mcc=np.zeros((5,1)) 
test_kappa=np.zeros((5,1)) 
test_ap=np.zeros((5,1))

t_tables = PrettyTable(['epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC',  'kappa', 'ap'])
t_tables.float_format = '.3' 
for i in range(5):
    model = FPDDS( cell_dim=18046, descriptor_size=descript2size[descript_set]).to(device)
    # A few hyperparamters for the training loop 
    lr = 3e-3
                    
    # drucomb 228, drugcombdb 112
    optimizer = optim.AdamW(model.parameters() )
    loss_criterion = nn.BCELoss()
    stopper = EarlyStopping(mode='higher', patience=15, filename='fp_fusion_fp')
    num_epochs = 100
    all_tables = PrettyTable(['epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC',  'kappa', 'ap'])
    all_tables.float_format = '.3' 

    scheduler =  torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loaders[i]))
    for epoch in range(num_epochs):
            # Train
            run_a_train_epoch(device, epoch,num_epochs, model, train_loaders[i], loss_criterion, optimizer, scheduler)
            # Validation and early stop
            val_pred, val_true = run_an_eval_epoch(model, valid_loaders[i])
            ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = compute_metrics(val_true, val_pred)
            early_stop = stopper.step(roc_auc, model)
            e_tables = PrettyTable(['epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC',  'kappa', 'ap'])
            e_tables.float_format = '.3' 
            row = [epoch,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
            e_tables.add_row(row)
            all_tables.add_row(row)
            print(e_tables)
            if early_stop:
                break
    stopper.load_checkpoint(model)
    test_pred, test_y = run_an_eval_epoch(model, test_loaders[i])
    ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = compute_metrics(test_y, test_pred)
    row = ['test', ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
    t_tables.add_row(row)
    print(t_tables)

    test_ACC[i]=ACC
    test_BACC[i]=BACC
    test_Prec[i]=Prec
    test_Rec[i]=Rec
    test_F1[i]=F1
    test_roc_auc[i]= roc_auc
    test_mcc[i]= mcc
    test_kappa[i]=kappa
    test_ap[i]=ap

row = ['mean', np.mean(test_ACC), np.mean(test_BACC), np.mean(test_Prec), np.mean(test_Rec), np.mean(test_F1), np.mean(test_roc_auc), np.mean(test_mcc), np.mean(test_kappa), np.mean(test_ap)]
t_tables.add_row(row)

row = ['std', np.std(test_ACC), np.std(test_BACC), np.std(test_Prec), np.std(test_Rec), np.std(test_F1), np.std(test_roc_auc), np.std(test_mcc), np.std(test_kappa), np.std(test_ap)]
t_tables.add_row(row)

    
print(t_tables)



# data = SeqDataset(drug1, drug2, drug2smiles, context, context2features, labels, dataset_name,config_drug_feature)
# print(next(iter(data)))

# test_loader = DataLoader(dataset=data, batch_size=2, collate_fn=collate_molgraphs, num_workers=8)
# batch_data  = next(iter(test_loader))
# print(batch_data)
# print(data.max_smi_len)
# bg_drug1, bg_drug2, context_feature,labels = batch_data
# model = DeepCombConvTensorFusion(
#                  node_in_feats=config_drug_feature['node_featurizer'].feat_size(),
#                  edge_in_feats=config_drug_feature['edge_featurizer'].feat_size(),
#                  node_out_feats=32,
#                  edge_hidden_feats=128,
#                  num_step_message_passing=2,
#                  cell_dim=288)
# node_feats1 = bg_drug1.ndata.pop('h') 
# edge_feats1 = bg_drug1.edata.pop('e') 
# node_feats2 = bg_drug2.ndata.pop('h') 
# edge_feats2 = bg_drug2.edata.pop('e')
# output = model(bg_drug1,node_feats1,edge_feats1,bg_drug2,node_feats2,edge_feats2,context_feature)
# print(output.size())
