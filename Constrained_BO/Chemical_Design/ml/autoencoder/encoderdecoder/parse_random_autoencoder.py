import os
import re
from pprint import pprint
from operator import itemgetter
import ast
import argparse

DEFAULT_FOLDER = '/n/home05/rgbombarelli'
EXPERIMENT_FOLDER = 'random_autoencoder_experiment'
KEY = 'val acc'
DELAY_UNTIL_VAE = 4


def parse_dict(text):
    dictio_text = re.findall("{(.*)}", text)
    if dictio_text:
        dictio_text = dictio_text[0]
    else:
        return
    dictio_text = dictio_text.replace('array([', '')
    dictio_text = dictio_text.replace('])', '')
    dictio = ast.literal_eval("{" + dictio_text + "}")
    return dictio


def parse_accs(text):
    hyper_pars = parse_dict(text)

    val_accs = re.findall("val_acc: 0(?:\.\d+)?", text)
    if val_accs:
        val_accs = [k.split()[-1] for k in val_accs]
    else:
        return
    best_acc = max(val_accs)
    result_dict = {'val acc': best_acc}
    if 'do_vae' in hyper_pars and (hyper_pars['do_vae'] is True or hyper_pars['do_vae'] == 1):
        anneal_epoch = hyper_pars['vae_annealer_start']
        vae_accs_from = anneal_epoch + int(min(DELAY_UNTIL_VAE, 0.25 * anneal_epoch))
        if len(val_accs) > vae_accs_from:
            best_vae_acc = max(val_accs[vae_accs_from - 1:])
            result_dict['vae val acc'] = best_vae_acc
    else:
        result_dict['vae val acc'] = float('nan')
    return result_dict, hyper_pars


def parse_slurm_outputs(parent_folders=[DEFAULT_FOLDER],
                        experiment_folders=[EXPERIMENT_FOLDER],
                        dates=[], required_completed=False):
    errors = []
    results = []

    if type(parent_folders) is str:
        parent_folders = parent_folders.split(',')
    if type(experiment_folders) is str:
        experiment_folders = experiment_folders.split(',')
    if type(dates) is str:
        dates = dates.split(',')

    for parent_folder in parent_folders:
        for experiment_folder in experiment_folders:
            folders = [os.path.join(parent_folder, i) for i in os.listdir(parent_folder) if i.startswith(experiment_folder)]

            files = []
            for i in folders:
                for j in os.listdir(i):
                    if j.startswith('slurm-'):
                        files.append(os.path.join(i, j))
            if dates:
                files = [file for file in files if any([(date in file) for date in dates])]

            files.sort()
            for file in files:
                with open(file, 'r') as f:
                    text = f.read()
                if required_completed and 'All done' not in text[-100:]:
                    continue
                try:
                    result_dict, hyper_pars = parse_accs(text)
                    result_dict['file'] = file
                    results.append((result_dict, hyper_pars))
                except (TypeError, ValueError) as e:
                    errors.append((e, file, text[-100:]))
                    continue

    return results, errors


if __name__ == '__main__':
    desc = """Parsing output files from random autoencoder experiments"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--parent', dest='parent', type=str, default=DEFAULT_FOLDER,
                        help="common parent folder")
    parser.add_argument('--exp', dest='experiment', type=str, default=EXPERIMENT_FOLDER,
                        help="string in all experiment folders")
    parser.add_argument('--dates', dest='dates', type=str, default="",
                        help="comma-separated strings in selected folders")
    parser.add_argument('--comp_only', dest='completed',
                        action='store_true', default=False,
                        help="consider only completed experiments")
    parser.add_argument('--key', dest='key', type=str,
                        default=KEY,
                        help="what key to look for in result dict: 'val acc' or 'vae val acc'")

    clargs = parser.parse_args()
    key = clargs.key

    results, errors = parse_slurm_outputs(dates=clargs.dates,
                                          parent_folder=clargs.parent,
                                          experiment_folder=clargs.experiment,
                                          required_completed=clargs.completed)

    sorted_results = sorted(results, key=lambda (v): v[0][key], reverse=True)
    pprint([(j, i[0][key], i[1]['hidden_dim']) for j,i in enumerate(sorted_results[:10])])
