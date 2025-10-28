import torch
import os
import pickle
from trainer_dif import TrainerMultiple
from pathlib import Path

from model.cnn import build_classifier_from_vit

def From_vit(opt, device):
    print('[  Current classifier : {}  ]\n'.format(opt.classifier))

    root = './pretrained_detectors/human_vit'

    model = build_classifier_from_vit(opt)
    param = torch.load(os.path.join(root, 'model.pt'), map_location = torch.device('cpu'))
    model.load_state_dict(param)
    model = model.to(device)
    return model

def From_dif(device):

    # TrainerMultiple 객체 준비
    
    root = Path('./pretrained_detectors/human_dif')
    
    with open(root / "train_hypers.pt", 'rb') as pickle_file:
        hyper_pars = pickle.load(pickle_file)
    hyper_pars['Device'] = device
    hyper_pars['Batch Size'] = 1
    #hyper_pars['Inp. Channel'] = 3
    trainer = TrainerMultiple(hyper_pars)
    trainer.load_stats(root / f"chk_10.pt")
    
    return trainer

def load(opt, device):
    vit_model = From_vit(opt, device)
    dif_trainer = From_dif(device)
    return vit_model, dif_trainer