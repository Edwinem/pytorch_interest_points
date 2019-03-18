import os
import json
import argparse
import torch
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
import base.lr_schedulers as lr_schedulers
import trainer as module_trainer
from utils import Logger
from utils import util


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def search_module_and_get_instance(module,name,config,**kwargs):
    '''
    Searches through the module _all_ variable and tries to find the specified class.

    WARNING: This function will do wrong things if you have the same class name in different files. It will pick the
    first one found.

    The reasoning for this function is that I wanted to be able to be able to have models, or dataset classes spread
    out in different files, but still be able to have a non specialized train function. Thus we search for the desired
    class.
    :param module:
    :param name:
    :param config:
    :return:
    '''
    for m in module.__all__:
        mod=getattr(module,str(m))
        if hasattr(mod,config[name]['type']):
            print('LOADING \"{}\" of type \"{}\" from \"{}\" '.format(name,config[name]['type'],mod))
            return get_instance(mod,name,config,**kwargs)
    print('ERROR: Could not find {} in any of the modules under {}. Please make sure that the class exists '
          'and you spelled it correctly'.format(config[name]['type'],module))
    exit(1)

def search_module_and_get_attr(module,name,config,**kwargs):
    '''
    Searches through the module _all_ variable and tries to find the specified class. Same as
    search_module_and_get_instance expect it returns an attr instead

    :param module:
    :param name:
    :param config:
    :return:
    '''
    for m in module.__all__:
        mod=getattr(module,str(m))
        if hasattr(mod,config[name]['type']):
            print('LOADING \"{}\" of type \"{}\" from \"{}\" '.format(name,config[name]['type'],mod))
            return getattr(mod,config[name]['type'])
    print('ERROR: Could not find {} in any of the modules under {}. Please make sure that the class exists '
          'and you spelled it correctly'.format(config[name]['type'],module))
    exit(1)

def main(config, resume):
    train_logger = Logger()

    data_loader = search_module_and_get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = search_module_and_get_instance(module_arch, 'model', config)
    print(model)

    # get loss and metrics
    losses = {
        entry['type']: (
            getattr(module_loss, entry['type'])(**entry['args']),
            entry['weight']
        )
        for entry in config['losses']
    }
    #loss = get_instance(module_loss, 'loss', config)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)



    Trainer=search_module_and_get_attr(module_trainer,'trainer',config)
    trainer = Trainer(model, losses, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--experiment_name', default=None, type=str,
                        help='save the experiment results under the given name')
    args = parser.parse_args()

    if args.config:
        # load config file
        json_file = open(args.config)
        #Remove comments from Json file
        cleaned_json=util.json_remove_comments(json_file)
        config =json.loads(cleaned_json)
        json_file.close()
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.config and args.resume:
        config['finetune'] = True

    if args.experiment_name:
        config['name'] = args.name

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
