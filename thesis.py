import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
from evaluation import evaluate, evaluate2
from matplotlib import rcParams

def plot_experts():
    dir_path = '/home/giovani/hacer/dril/dril/demo_data'
    seeds = [68 , 66, 66]
    titles = ['PPO-Beta (Bounded)', 'PPO-Gaussian (Unbounded)', 'PPO-Gaussian (Clipped)']
    fig, ax = plt.subplots(2, len(seeds),figsize=(10, 5))
    actions = ['Left | Right', 'Brake | Throttle']
    for j, seed in enumerate(seeds):
        acs_file_name = f'acs_CarRacing-v0_seed={seed}_ntraj=1.npy'
        acs = np.load(os.path.join(dir_path, acs_file_name))
        for i in range(acs.shape[1]):
            act = np.clip(acs[:, i],-1,1) if j==2 else acs[:, i]
            ax[i, j].scatter(range(acs.shape[0]),act, s=1, color=f"{'blue' if i==0 else 'orange'}")
            ax[i, j].set_ylabel(f'{actions[i]}') if j==0 else None
            ax[i, j].set_ylim(-4.2, 4.2)
            ax[i, j].fill_between(range(acs.shape[0]), -1*np.ones(acs.shape[0]), 1*np.ones(acs.shape[0]),
                                  facecolor='gray', alpha=0.3, label='Action space')


            if i==0:
                ax[i, j].set_title(titles[j])
            else:
                ax[i, j].set_xlabel('Episode steps')

    ax[0,0].legend(loc='upper left')


    fig_name = "clipped_bounded_action_expert.jpg"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='jpg')
    plt.show()

def plot_gaussian_expert():
    dir_path = '/home/giovani/hacer/dril/dril/demo_data'
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    actions = ['Left | Right', 'Brake | Throttle']
    acs_file_name = f'acs_CarRacing-v0_seed=66_ntraj=1.npy'
    acs = np.load(os.path.join(dir_path, acs_file_name))
    for i in range(acs.shape[1]):
        ax[i].scatter(range(acs.shape[0]),acs[:, i], s=1, color=f"{'blue' if i==0 else 'orange'}")
        ax[i].set_ylim(-4.2, 4.2)
        ax[i].fill_between(range(acs.shape[0]), -1*np.ones(acs.shape[0]), 1*np.ones(acs.shape[0]),
                              facecolor='gray', alpha=0.3, label='Action space')
        ax[i].set_title(f'Action dimension {i}')
        ax[i].set_ylabel(f'{actions[i]}')
        ax[i].set_xlabel('Steps')
        ax[i].legend(loc='lower left')

    plt.show()


    acs_total = 0

    for i in range(acs.shape[0]):
        if acs[i, 0]>1 or acs[i,0]<-1 or acs[i, 1]>1 or acs[i,1]<-1:
            acs_total += 1

    print(acs_total/acs.shape[0])

def plot_alt_experts():
    dir_path = os.getcwd() + '/trained_models/dril/'
    device = 'cuda:0'
    file_names = {'66gaussian':'dril_CarRacing-v0_d=gaussian_policy_ntrajs=1_seed=66_ppoepoch=4_lr3e-05_steps=4096_det=False_nup=151.pt',
                  '66beta'    :'dril_CarRacing-v0_d=beta_policy_ntrajs=1_seed=66_ppoepoch=4_lr3e-05_steps=4096_det=False_nup=51.pt',
                  '67gaussian':'dril_CarRacing-v0_d=gaussian_policy_ntrajs=1_seed=67_ppoepoch=4_lr3e-05_steps=4096_det=False_nup=151.pt',
                  '67beta'    :'dril_CarRacing-v0_d=beta_policy_ntrajs=1_seed=67_ppoepoch=4_lr3e-05_steps=4096_det=False_nup=20.pt'}

    #file_names = {'66gaussian':'dril_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=66_ppoepoch=4_lr3e-05_steps=4096_det=False_nup=151.pt',
    #              '66beta'    :'dril_CarRacing-v0_d=beta_policy_ntrajs=20_seed=66_ppoepoch=4_lr3e-05_steps=4096_det=False_nup=151.pt',
    #              '67gaussian':'dril_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=67_ppoepoch=4_lr3e-05_steps=4096_det=False_nup=151.pt',
    #              '67beta'    :'dril_CarRacing-v0_d=beta_policy_ntrajs=20_seed=67_ppoepoch=4_lr3e-05_steps=4096_det=False_nup=143.pt'}



    fig, ax = plt.subplots(2,2,figsize=(15,10))
    for i, (seed, file_name) in enumerate(file_names.items()):
        path = os.path.join(dir_path, file_name)
        actor_critic, obs_rms, args, scr_train, utrain = torch.load(path, map_location=device)
        eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
        ax[i // 2, i % 2].scatter(range(len(scr_train)), scr_train, color='blue', s=2)
        ax[i // 2, i % 2].scatter(range(len(scr_train)), moving_avg(scr_train,100), color='orange', s=2)
        ax[i // 2, i % 2].set_title(seed)

    plt.show()


def pack_trajs(ntraj):
    dir_path = '/home/giovani/article/expert/gaussian/'
    file_names = os.listdir(dir_path)

    selected_files = []
    for fname in file_names:
        if 'demo' in fname:
            selected_files.append(fname)


    obs_l = []
    acs_l = []

    for file_name in selected_files[0:ntraj]:
        rollout = torch.load(os.path.join(dir_path, file_name), map_location='cuda:0')
        obs_l.extend(rollout['obs'])
        acs_l.extend(rollout['acs'])

    acs_pth = torch.cat(acs_l, dim=0)
    obs_pth = torch.cat(obs_l, dim=0)

    acs_pth = torch.clip(acs_pth, -1, 1)

    acs = acs_pth.cpu().numpy()
    obs = obs_pth.cpu().numpy()

    if "beta" in file_names[0]:
        print(f'rescaling actions from beta to -1 to +1')
        acs = acs * 2 - 1

    print(acs.shape)
    print(obs.shape)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(acs[:, 0])
    ax[0].set_ylabel('Left | Right')
    ax[1].plot(acs[:, 1])
    ax[1].set_ylabel('Brake   | Throttle ')
    ax[1].set_xlabel('Steps')
    plt.show()

    save_path = '/home/giovani/hacer/dril/dril/demo_data'

    acs_file_name = f'acs_CarRacing-v0_seed=67_ntraj={len(file_names[0:ntraj])}.npy'
    obs_file_name = f'obs_CarRacing-v0_seed=67_ntraj={len(file_names[0:ntraj])}.npy'

    print(f'Saving expert demos at {save_path}')
    np.save(os.path.join(save_path,acs_file_name), acs)
    np.save(os.path.join(save_path,obs_file_name), obs)


def moving_avg(a, n):
    mv = np.zeros_like(a)

    for i in range(n):
        mv[i] = np.average(a[0:i])

    for i in range(0, len(a)-(n-1)):
        mv[i+n-1]=np.average(a[i:i+n])

    return mv


def plot_train_dril_cr_gaussian():
    dir_path = os.getcwd() + '/trained_models/dril/'
    file_names = \
        [f'dril_CarRacing-v0_d=gaussian_NO_PRET_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=gaussian_NO_PRET_ntrajs=1_seed=68_ppoepoch=10_lr=0_steps=4096_det=False_nup=212.pt',
         f'dril_CarRacing-v0_d=gaussian_NO_PRET_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=gaussian_NO_PRET_ntrajs=20_seed=68_ppoepoch=10_lr=0_steps=4096_det=False_nup=304.pt']

    ncol = 2

    fig, ax = plt.subplots(max(2,len(file_names)//ncol), ncol, figsize=(25, 16))
    char_break = 48
    titles = [fname[0:char_break] + "\n" + fname[char_break:] for fname in file_names]

    """
    titles = ['ntraj=1, NP ppo (epoch=10  \n lr=2.5e-4, steps=4096)',
              'ntraj=1, NP ppo (epoch=10  \n lr=2.5e-4, steps=4096)',
              'ntraj=20, NP, ppo (epoch=10 \n lr=2.5e-4, steps=4096)',
              'ntraj=20, NP, ppo (epoch=10 \n lr=2.5e-4, steps=4096)']
    """

    fig.suptitle('DRIL-Gaussian, Expert demonstration from PPO-Beta')


    for idx, fname in enumerate(file_names):
        path = os.path.join(dir_path, fname)
        ac, obs_rms, args, scr_train, utrain = torch.load(path, map_location='cpu')
        N = len(scr_train)
        line_plot = scr_train[0:N]

        ax[idx//ncol, idx % ncol].scatter(range(len(line_plot)), line_plot, marker='.')
        ax[idx//ncol, idx % ncol].plot(range(len(line_plot)), moving_avg(line_plot, 100), color='orange', linewidth=3)

        ax[idx//ncol, idx % ncol].set_title(titles[idx])
        ax[idx//ncol, idx % ncol].set_ylim(-200, 1000)

        if idx//ncol == 1:
            ax[idx//ncol, idx % ncol].set_xlabel('Episodes')

        if idx % ncol != 0:
            ax[idx // ncol, idx % ncol].yaxis.set_visible(False)
            ax[idx // ncol, idx % ncol].yaxis.set_visible(False)
        else:
            ax[idx // ncol, idx % ncol].set_ylabel('Score')

    plt.show()


def plot_train_dril_cr_beta():
    dir_path = os.getcwd() + '/trained_models/dril/'
    file_names = \
        [f'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=1_seed=68_ppoepoch=1_lr=0.00025_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=1_seed=68_ppoepoch=1_lr=2.5e-07_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=1_seed=68_ppoepoch=1_lr=2.5e-09_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=1_seed=68_ppoepoch=1_lr=0_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=20_seed=68_ppoepoch=1_lr=0.00025_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=20_seed=68_ppoepoch=1_lr=2.5e-07_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=20_seed=68_ppoepoch=1_lr=2.5e-09_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=20_seed=68_ppoepoch=1_lr=0_steps=4096_det=False_nup=304.pt']

    ncol = 4

    fig, ax = plt.subplots(max(2,len(file_names)//ncol), ncol, figsize=(25, 16))
    char_break = 48
    titles = [fname[0:char_break] + "\n" + fname[char_break:] for fname in file_names]

    titles = ['ntraj=1, NP ppo (epoch=4  \n lr=2.5e-4, steps=4096)',
              'ntraj=1, NP, ppo (epoch=1 \n lr=2.5e-7, steps=4096)',
              'ntraj=1, NP, ppo (epoch=1 \n lr=2.5e-9, steps=4096)',
              'ntraj=1, NP, ppo (epoch=1 \n lr=0, steps=4096)',

              'ntraj=20, NP, ppo (epoch=4 \n lr=2.5e-4, steps=4096)',
              'ntraj=20, NP, ppo (epoch=1 \n lr=2.5e-7, steps=4096)',
              'ntraj=20, NP, ppo (epoch=1 \n lr=2.5e-9, steps=4096)',
              'ntraj=20, NP, ppo (epoch=1 \n lr=0, steps=4096)']

    fig.suptitle(f'DRIL-Beta, Expert demonstration from PPO-Beta, Without BC pre-training \n'  
                 f'Effect of lowering PPO lr and number of PPO epochs')


    for idx, fname in enumerate(file_names):
        path = os.path.join(dir_path, fname)
        ac, obs_rms, args, scr_train, utrain = torch.load(path, map_location='cpu')
        N = len(scr_train)
        line_plot = scr_train[0:N]
        ma = moving_avg(line_plot, 100)
        ma_num = np.nan_to_num(ma)
        ma_max = np.max(ma_num)
        ax[idx//ncol, idx % ncol].scatter(range(len(line_plot)), line_plot, marker='.')
        ax[idx//ncol, idx % ncol].plot(range(len(line_plot)), moving_avg(line_plot, 100), color='orange', \
                                       linewidth=3, label=f'Max mov avg ={ma_max:.0f}')

        ax[idx//ncol, idx % ncol].set_title(titles[idx])
        ax[idx//ncol, idx % ncol].legend(loc='lower right')
        ax[idx // ncol, idx % ncol].get_xaxis().set_visible(False) if idx // ncol == 0 else None

        ax[idx//ncol, idx % ncol].set_ylim(-200, 1000)
        ax[idx//ncol, idx % ncol].set_xlabel('Training Episodes') if idx//ncol == 1 else None

        if idx % ncol != 0:
            ax[idx // ncol, idx % ncol].yaxis.set_visible(False)
            ax[idx // ncol, idx % ncol].yaxis.set_visible(False)
        else:
            ax[idx // ncol, idx % ncol].set_ylabel('Score')

    plt.show()


def args_items():
    dir_path = os.getcwd() + '/trained_models/dril/'
    file_names = \
        [f'dril_CarRacing-v0_d=beta_LP_ntrajs=1_seed=68_ppoepoch=4_lr=2.5e-06_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_LP_ntrajs=1_seed=68_ppoepoch=1_lr=2.5e-06_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_LP_ntrajs=20_seed=68_ppoepoch=4_lr=2.5e-06_steps=4096_det=False_nup=304.pt',
         f'dril_CarRacing-v0_d=beta_LP_ntrajs=20_seed=68_ppoepoch=1_lr=2.5e-06_steps=4096_det=False_nup=304.pt',]


    for fname in file_names:
        path = os.path.join(dir_path, fname)
        ac, obs_rms, args, scr_train, utrain = torch.load(path, map_location='cpu')

    for k,v in args.__dict__.items():
        print(f'{k:30} = {v}')


def plot_train_scr_u_dril_cr():

    DISTRIBUTION = 'beta'
    EXPERT = 'bounded'
    if DISTRIBUTION == 'gaussian':
        if EXPERT == 'clipped':
            dir_path = os.getcwd() + '/trained_models/dril/backup/dril_g_exp_g/'
            file_names = \
                [f'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=53.122974799999994.pt',
                 f'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=248.5544372.pt']
            color_points = 'gray'
            fig_name = "cr_dril_gaussian_plot_train_scr_u.eps"
        else:
            dir_path = os.getcwd() + '/trained_models/dril/backup/dril_g_exp_b/'
            file_names = \
                [
                    f'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=233.7758806.pt',
                    f'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=284.10052320000005.pt']
            color_points = 'green'
            fig_name = "cr_dril_beta_plot_train_scr_u.eps"

    else:
        if EXPERT == 'clipped':
            pass
        else:
            dir_path = os.getcwd() + '/trained_models/dril/backup/dril_b_exp_b/'
            file_names = \
                [
                    f'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=102.51775520000001.pt',
                    f'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=129.1200822.pt']
            color_points = 'black'
            fig_name = "cr_dril_beta_bounded_plot_train_scr_u.eps"







    fig, ax = plt.subplots(2, len(file_names), figsize=(10, 5))
    m = 48
    titles =[ fname[0:m]+"\n"+fname[m:] for fname in file_names]
    titles = ['a) 1 trajectory', 'b) 20 trajectories']


    for j, fname in enumerate(file_names):
        path = os.path.join(dir_path, fname)
        ac, obs_rms, args, scr_train, utrain = torch.load(path, map_location='cpu')
        N = len(scr_train)
        scr_train = scr_train[0:N]
        utrain = utrain[0:N]

        for i, line_plot in enumerate([scr_train, utrain]):
            ax[i, j].scatter(range(len(line_plot)), line_plot, marker='.', color=color_points, s=3)
            ax[i, j].plot(range(len(line_plot)), moving_avg(line_plot, 100), color='orange', linewidth=3, label='100-episode moving average')

            if i == 0:
                ax[i, j].set_title(titles[j])
                ax[i, j].set_ylim(-200, 1000)
            else:
                ax[i, j].set_xlabel('Episodes')
                ax[i, j].set_ylim(0.9, 1.01)

    ax[1, 0].set_ylabel('U_signal')
    ax[0, 0].set_ylabel('Training score')
    ax[0, 1].yaxis.set_visible(False)
    ax[1, 1].yaxis.set_visible(False)
    ax[0, 0].legend()


    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='eps')
    plt.show()



def plot_training_ens():
    dir_path = '~/hacer/dril/dril/trained_results/ensemble'
    d = 'beta'
    file_names = [f'ensemble_CarRacing-v0_n=5_hu=512_d={d}_policy_ntrajs=1_seed=66.perf',
                  f'ensemble_CarRacing-v0_n=5_hu=512_d={d}_policy_ntrajs=20_seed=66.perf']
    titles = ['a) Dataset with 1 trajectory', 'b) Dataset with 20 trajectories']

    plt.rcParams.update({'font.size': 11})
    plt.rcParams['figure.figsize'] = [8, 3.1]
    plt.rcParams['figure.subplot.bottom'] = 0.0
    plt.rcParams.update({'figure.autolayout': True})

    fig, ax = plt.subplots(1, 2, sharey=True)

    for i, file_name in enumerate(file_names):
        file_path = os.path.join(dir_path, file_name)
        data = pd.read_csv(file_path)

        ax[i].plot(data['epoch'], data['trloss'], label='Training Loss')
        ax[i].plot(data['epoch'], data['teloss'], label='Validation Loss')
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Epochs')

    ax[0].set_ylabel('Loss')
    ax[1].legend(loc='upper right')


    fig_name = 'cr_ensemble_gaussian_seed66.eps'
    save_dir = '/home/giovani/faire/eps/'
    save_path = os.path.join(save_dir, fig_name)
    plt.savefig(save_path, format='eps', bbox_inches='tight')

    plt.show()



def plot_var_npol_ens():
    dir_path = '~/hacer/dril/dril/trained_results/ensemble'
    D = ['gaussian', 'beta']
    d = D[1]
    NPOLICIES = [5, 20, 50]

    fig, ax = plt.subplots(1, len(NPOLICIES), figsize=(10, 5), sharey=True)
    for j, npol in enumerate(NPOLICIES):
        fname = f'ensemble_CarRacing-v0_n={npol}_hu=512_d={d}_policy_ntrajs=1_seed=80.perf'

        file_path = os.path.join(dir_path, fname)
        data = pd.read_csv(file_path)

        ax[j].plot(data['epoch'], data['trloss'], label='Training Loss')
        ax[j].plot(data['epoch'], data['teloss'], label='Test Loss')
        ax[j].set_title(f'{npol} policies, {d}')
        ax[j].set_xlabel('Epochs')

    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper right')
    plt.show()


def plot_ens_acs_var_npols():
    NTRAJS = [1, 20]
    NPOLS = [5, 20, 50]

    d = 'beta'
    fig, ax = plt.subplots(len(NTRAJS), len(NPOLS), figsize=(10, 5), sharey=True)

    for i, ntrajs in enumerate(NTRAJS):
        for j, npols in enumerate(NPOLS):
            fname = f'CarRacing-v0_n={npols}_hu=512_ntraj={ntrajs}_seed=80_d=beta_actions.pt'
            dir_path_ens = '/home/giovani/hacer/dril/dril/trained_models/ensemble/'
            file_path_ens = os.path.join(dir_path_ens, fname)
            ens_acs = torch.load(file_path_ens)

            #print(ens_acs)

            variance = ens_acs['variance'][0]
            quantiles = ens_acs['quantiles'][0]

            ax[i, j].scatter(range(len(variance)), variance, label="Ensemble's Actions Variance", s=5)
            ax[i, j].plot(range(len(variance)), quantiles*np.ones(len(variance)), label='98th quantile', color='orange')

            if i == 0:
                ax[i, j].set_title(f'{npols} policies')
            else:
                ax[i, j].set_xlabel('States')

            if j == 0:
                ax[i, j].set_ylabel(f'{NTRAJS[i]} Trajectories')

    plt.show()


def plot_ens_signal():

    d = 'gaussian'
    seed=66
    NTRAJ = [1, 20]
    fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    for j, ntraj in enumerate(NTRAJ):
        fname = f'CarRacing-v0_n=5_hu=512_ntraj={ntraj}_seed={seed}_d={d}_signals.pt'

        dir_path_ens = '/home/giovani/hacer/dril/dril/trained_models/ensemble/'
        fpath = os.path.join(dir_path_ens, fname)
        data = torch.load(fpath)
        rl, ensemble_variance_hist, raw_reward_hist, episode_rewards = data

        print(episode_rewards)

        fname = f'CarRacing-v0_n=5_hu=512_ntraj={ntraj}_seed={seed}_d={d}_actions.pt'
        dir_path = '/home/giovani/hacer/dril/dril/trained_models/ensemble/'
        file_path = os.path.join(dir_path, fname)
        ens_acs = torch.load(file_path)

        variance = ens_acs['variance'][0]
        threshold = ens_acs['quantiles'][0]

        acs = rl.actions.squeeze().squeeze().cpu().numpy()


        N = 1000
        ax[0, j].scatter(range(len(acs))[:N], acs[:, 0][:N], color='blue', label='Left/Right', s=1)
        ax[0, j].scatter(range(len(acs))[:N], acs[:, 1][:N], color='orange', label='Brake/Throttle', s=1)
        ax[0, j].fill_between(range(len(acs))[:N], -1*np.ones_like(acs[:, 0])[:N],
                          +1*np.ones_like(acs[:, 0])[:N], color='gray', alpha=0.5, label='Valid range')

        ax[1, j].bar(range(len(ensemble_variance_hist))[:N], ensemble_variance_hist[:N], color='black', label='Ensemble variance')
        ax[1, j].plot(range(len(ensemble_variance_hist))[:N], threshold*np.ones(len(ensemble_variance_hist))[:N], color='red', label='Variance threshold')
        ax[2, j].bar(range(len(raw_reward_hist))[:N], raw_reward_hist[:N], color='gray')


        if j ==0:
            ax[0, j].set_ylabel('Action')
            ax[1, j].set_ylabel('Variance')
            ax[2, j].set_ylabel('Ensemble Reward')

        else:
            ax[0, j].legend(loc='lower right', ncol=2)
            ax[1, j].legend(loc='upper right', ncol=1)


        ax[2, j].set_xlabel('Time step')
        ax[0, j].set_ylim([-4.0, 4.0])
        ax[1, j].set_ylim([0.0, 2.0])


        ax[0, j].set_title(f'{NTRAJ[j]} {"trajectory" if j==0 else "trajectories"} | '
                           f'training score: {episode_rewards[0]:.0f}')


    SHOW_OBS = False
    if SHOW_OBS:
        ncols = 11
        nrows = 8
        frames = 1000
        step = int(frames / (ncols*nrows))
        start_frame = 0
        fig, ax = plt.subplots(nrows, ncols, figsize=(25, 16), sharex=True)
        for i in range(nrows*ncols):
            frame_shown = i*step+start_frame
            ax[i // ncols, i % ncols].imshow(rl.obs[frame_shown,0,0,:,:].cpu().numpy(), cmap='gray')
            ax[i // ncols, i % ncols].set_title(f'Time step {frame_shown}')

    plt.show()


def eval_bc():
    device = 'cpu' # 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dir_ac = os.getcwd() +'/trained_models/dril/cleanup/'

    for det in [True, False]:
        #file_ac = 'dril_CarRacing-v0_d=gaussian_NO_PRET_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=4096_det=False_nup=8.pt'
        file_ac = 'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=20_seed=68_ppoepoch=1_lr=0.00025_steps=4096_det=False_nup=304.pt'
        path_ac = os.path.join(dir_ac, file_ac)
        actor_critic, obs_rms, args, _, _ = torch.load(path_ac, map_location=device)
        args.num_processes = 8

        score_history = {}
        for ntraj in [1, 20]:
            dir_bc = os.getcwd() + '/trained_models/bc/'
            #file_bc = f'bc_CarRacing-v0_d=gaussian_policy_ntrajs={ntraj}_seed=66.model.pth'
            file_bc= f'bc_CarRacing-v0_d=beta_policy_ntrajs={ntraj}_seed=68.model.pth'
            path_bc = os.path.join(dir_bc, file_bc)
            bc_model_params = torch.load(path_bc)
            actor_critic.load_state_dict(bc_model_params)

            eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
            scores = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                           device, num_episodes=100, atari_max_steps=None, fname='return_vector', det=det)

            score_history[ntraj] = scores

        bc_eval_path = os.path.join(dir_bc, f'{file_bc}_NO_PRET_eval_det={det}.pt')
        print(f'Saving BC evaluation at {bc_eval_path}')
        torch.save(score_history, bc_eval_path)


def plot_eval_bc():
    dir_bc = os.getcwd() + '/trained_models/bc/'

    plt.rcParams.update({'font.size': 9})
    plt.rcParams['figure.figsize'] = [20, 8]
    plt.rcParams.update({'figure.autolayout': True})

    fnames = ['bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=True.pt',
              'bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=True.pt',
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=True.pt',
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=True.pt',
              'bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=False.pt',
              'bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=False.pt',
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=False.pt',
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=False.pt']

    dir_bc = os.getcwd() + '/trained_models/bc/'

    'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=True.pt'
    'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=False.pt'

    ncols = 8

    fig, ax = plt.subplots(2, ncols, sharey=True, sharex=True)
    plt.suptitle(f'Behavior Cloning Deterministic & Stochastic', fontsize=16)

    titles= [ 'BCP Gaus, Exp Gaus,  1T',
              'BCP Gaus, Exp Gaus, 20T',
              'BCP Gaus, Exp Beta,  1T',
              'BCP Gaus, Exp Beta, 20T',
              'BCP Beta, Exp Gaus,  1T',
              'BCP Beta, Exp Gaus, 20T',
              'BCP Beta, Exp Beta,  1T',
              'BCP Beta, Exp Beta, 20T']



    color_pack =['blue',  'green', 'black', 'purple']
    for pointer, fname in enumerate(fnames):
        bc_eval_path = os.path.join(dir_bc,fname)
        score_history = torch.load(bc_eval_path)
        N = 100
        expert = 900

        for i, k in enumerate([1, 20]):
            print(f'{bc_eval_path}')
            print(f'k={k}, len={len(score_history[k])}')

            score_history[k] = score_history[k][0:N]
            exp_level = [score_history[k] >= np.ones(N) * expert]
            print(f'Expert level reached in {np.sum(exp_level)} out of {N} episodes for {k} trajectories')

        for i, k in enumerate([1, 20]):
            idx = pointer * 2 + i
            ax[idx //ncols, idx % ncols].scatter(range(N), score_history[k], color=color_pack[idx%ncols//2])
            mean = np.average(score_history[k])
            std = np.std(score_history[k])
            ax[idx //ncols, idx % ncols].plot(range(N), np.average(score_history[k])*np.ones(N), color='orange', linewidth=3, \
                       label=f'scr={mean :.0f} \u00B1 {std:.0f}')
            ax[1, idx % ncols].set_xlabel('Eval episodes')
            ax[idx //ncols, idx % ncols].set_ylim(-200, 1000)
            ax[idx //ncols, 0].set_ylabel(f'Score, {"Deterministic" if idx //ncols == 0 else "Stochastic"}')
            print(f'Min/max score for {k} trajectories: {np.min(score_history[k]):0f}/{np.max(score_history[k]):0f}')
            #ax[j, i].legend(bbox_to_anchor=(0.05, 0.55), loc='lower left', ncol=1, borderaxespad=0., framealpha=0.1)
            ax[0, idx % ncols].set_title(titles[idx % ncols])
            ax[idx //ncols, idx % ncols].legend()


    fig_name = 'cr_bc_eval_summary.eps'
    save_dir = '/home/giovani/hacer/dril/dril/trained_models/bc'
    save_path = os.path.join(save_dir, fig_name)
    plt.savefig(save_path, format='eps')

    plt.show()


def get_reward_history():
    device = 'cuda:0' # 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dir_ac = os.getcwd() +'/trained_models/dril/'
    det = False

    #file_ac = 'dril_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=68_ppoepoch=10_lr0.0003_steps=2048_det=False_nup=2440.pt'
    file_ac = 'dril_CarRacing-v0_d=beta_policy_ntrajs=1_seed=80_nup=100_ppoepoch=1_lr3e-05_steps=128_det=False.pt'
    path_ac = os.path.join(dir_ac, file_ac)
    actor_critic, obs_rms, args, _, _ = torch.load(path_ac, map_location=device)
    args.num_processes = 1

    score_history = {}
    for ntraj in [20]:
        dir_bc = os.getcwd() + '/trained_models/bc/'
        #file_bc = f'bc_CarRacing-v0_d=gaussian_policy_ntrajs={ntraj}_seed=68.model.pth'
        file_bc= f'bc_CarRacing-v0_d=beta_policy_ntrajs={ntraj}_seed=68.model.pth'
        path_bc = os.path.join(dir_bc, file_bc)
        bc_model_params = torch.load(path_bc)
        actor_critic.load_state_dict(bc_model_params)

        eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
        scores = evaluate2(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                       device, num_episodes=10, atari_max_steps=None, fname='return_vector', det=det)

        score_history[ntraj] = scores

    bc_eval_path = os.path.join(dir_bc, f'{file_bc}_eval_det={det}.pt')
    print(f'Saving BC evaluation at {bc_eval_path}')
    torch.save(score_history, bc_eval_path)


def plot_reward_history():
    dcr_eval_dril_gaussian_peak_final_seed66.jpgir_name = '/home/giovani/hacer/dril/dril/expert'
    file_names =[ 'score_hist_scr=943.pt',
        'score_hist_scr=928.pt',
        'score_hist_scr=892.pt',
        'score_hist_scr=271.pt']
    plt.rcParams.update({'font.size': 16})
    fig, ax =  plt.subplots(1,len(file_names),figsize=(25,5), sharey=True)

    titles = ['a) Perfect lap',
              'b) Missed tile early',
              'c) Missed tile late' ,
              'd) Missed early curve'  ]

    expert_level = 900
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(dir_name, file_name)
        reward = torch.load(file_path)
        reward =[x.squeeze() for x in reward]
        reward = [np.sum(reward[0:i]) for i in range(len(reward))]
        ax[i].plot(range(len(reward)), reward, label = f'Sccore={int(reward[-1])}')
        ax[i] .plot(range(len(reward)),np.ones_like(reward)*expert_level, linestyle='--',color='red', label='Expert')
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Steps')
        ax[i].legend(loc='lower right')

    ax[0].set_ylabel('Score')

    plt.show()


def eval_dril_cc_final():
    dir_path = os.getcwd() + '/trained_models/dril/'

    file_final_1 = 'dril_CarRacing-v0_d=beta_policy_ntrajs=1_seed=68_ppoepoch=4_lr3e-06_steps=16384_det=False_nup=304.pt'
    file_final_20 = 'dril_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68_ppoepoch=4_lr3e-06_steps=16384_det=False_nup=304.pt'

    file_names = { 'final_1': file_final_1,
                  'final_20': file_final_20}
    fig, ax = plt.subplots(2, len(file_names.keys()), figsize=(15, 10))

    scr_hist = {}
    device = 'cuda:0'
    for i, file_name in file_names.items():
        print(f'Starting evaluation of {i}: {file_name}')

        path = os.path.join(dir_path, file_name)
        actor_critic, obs_rms, args, scr_train, utrain = torch.load(path, map_location=device)
        eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
        args.num_processes = 4
        det = True
        scores = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                          device, num_episodes=100, atari_max_steps=None, fname='return_vector', det=det)
        scr_hist[i] = scores

    dril_eval_path = os.path.join(dir_path, f'{file_final_20}_eval_det={det}.pt')
    print(f'Saving DRIL evaluation at {dril_eval_path}')
    torch.save(scr_hist, dril_eval_path)


def eval_dril_cc_final():
    dir_path = os.getcwd() + '/trained_models/dril/'

    file_final_a = 'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=34_scr=814.3645798.pt'
    file_final_b = 'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=33_scr=777.2216255000001.pt'

    file_names = { 'final_1': file_final_a,
                  'final_20': file_final_b}
    fig, ax = plt.subplots(2, len(file_names.keys()), figsize=(15, 10))

    scr_hist = {}
    device = 'cpu'
    for i, file_name in file_names.items():
        print(f'Starting evaluation of {i}: {file_name}')

        path = os.path.join(dir_path, file_name)
        actor_critic, obs_rms, args, scr_train, utrain = torch.load(path, map_location=device)
        eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
        args.num_processes = 4
        det = True
        scores = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                          device, num_episodes=100, atari_max_steps=None, fname='return_vector', det=det)
        scr_hist[i] = scores

    dril_eval_path = os.path.join(dir_path, f'{file_final_a}peak_search_eval_det={det}.pt')
    print(f'Saving DRIL evaluation at {dril_eval_path}')
    torch.save(scr_hist, dril_eval_path)


def plot_eval_dril_std():
    dir_dril = os.getcwd() + '/trained_models/dril/'
    file_name = 'dril_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=66_ppoepoch=10_lr0.0003_steps=2048_det=False_nup=2440.pt_eval_det=True.pt'

    dril_path = os.path.join(dir_dril, file_name)
    score = torch.load(dril_path)

    N = 100
    expert = 900

    plt.rcParams.update({'font.size': 11})
    plt.rcParams['figure.figsize'] = [8, 3.1]
    plt.rcParams['figure.subplot.bottom'] = 0.0
    plt.rcParams.update({'figure.autolayout': True})

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)


    for i,  k in enumerate(score.keys()):
        ax[i].plot(np.ones(N) * expert, label='Expert threshold (900)', color='red', linewidth=3, linestyle='--')

        ax[i].scatter(range(len(score[k])), score[k],label='DRIL score per episode', color='blue', s=6)
        mean = np.average(score[k])
        std = np.std(score[k])
        ax[i].plot(range(len(score[k])), np.average(score[k])*np.ones(len(score[k])), color='orange', linewidth=3,\
                   label=f'DRIL avg score = {mean:.0f} \u00B1 {std:.0f}')

        ax[i].set_title(f'{"a) Dataset with 1 trajectory" if i==0 else "b) Dataset with 20 trajectories"}')
        ax[i].set_xlabel('Episodes')
        ax[i].set_ylim(-200, 1000)

        ax[i].legend(bbox_to_anchor=(0.05, 0.55), loc='lower left',
                     ncol=1, borderaxespad=0.,
                     fancybox=False,
                     framealpha=0.5)

    ax[0].set_ylabel('Score')


    fig_name = 'dril_eval_std.eps'
    save_dir = '/home/giovani/faire/eps/'
    save_path = os.path.join(save_dir, fig_name)
    plt.savefig(save_path, format='eps', bbox_inches='tight')
    plt.show()


def plot_peaks():
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    plt.rcParams.update({'font.size': 11})
    plt.rcParams['figure.figsize'] = [16, 3.1]
    plt.rcParams['figure.subplot.bottom'] = 0.0
    plt.rcParams.update({'figure.autolayout': True})

    device = 'cpu'
    dir_dril = os.getcwd() + '/trained_models/dril/'
    fnames =['dril_CarRacing-v0_d=beta_LP_ntrajs=20_seed=68_ppoepoch=4_lr=2.5e-06_steps=4096_det=False_nup=304.pt',
             'dril_CarRacing-v0_d=beta_LP_ntrajs=20_seed=68_ppoepoch=4_lr=2.5e-06_steps=4096_det=False_nup=304.pt']

    for i, fname in enumerate(fnames):
        dril_path = os.path.join(dir_dril, fname)
        actor_critic, obs_rms, args, scr_train, utrain = torch.load(dril_path, map_location=device)
        mov_avg_limit = min(len(scr_train), 100)
        max_len_plot = min(len(scr_train), 500)
        scr_train = scr_train[0:max_len_plot]

        ax[i].plot(scr_train)
        ax[i].plot(moving_avg(scr_train, mov_avg_limit))

    plt.show()
    #print(actor_critic)



if __name__ == '__main__':
    # 66 ==> gaussian, 67 ==> gaussian clipped, 68 ==> beta

    #plot_peaks()

    #plot_experts()
    #plot_alt_experts()

    #load_death_tracks()
    #plot_ens_acs_var_npols()
    #plot_var_npol_ens()


    #eval_bc()
    #plot_eval_bc()
    #plot_training_ens()
    #plot_ens_acs_cr()
    #plot_ens_signal()

    plot_train_scr_u_dril_cr()
    #plot_train_dril_cr_gaussian()
    #plot_train_dril_cr_beta()

    #args_items()
    #plot_eval_dril_std()
    #eval_dril_cc_final()

    #for n in [1,20]:
    #    pack_trajs(n)

    #plot_gaussian_expert()
    #get_reward_history()
    #plot_reward_history()