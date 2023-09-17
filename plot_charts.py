import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
from evaluation import evaluate, evaluate2
from matplotlib import rcParams

def moving_avg(a, n):
    mv = np.zeros_like(a)

    for i in range(n):
        mv[i] = np.average(a[0:i])

    for i in range(0, len(a)-(n-1)):
        mv[i+n-1]=np.average(a[i:i+n])

    return mv

############################
########### BC #############
############################
def eval_bc():
    device = 'cpu' # 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dir_ac = os.getcwd() +'/trained_models/dril/cleanup/'

    for det in [True, False]:
        file_ac = 'dril_CarRacing-v0_d=gaussian_NO_PRET_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=4096_det=False_nup=8.pt'
        #file_ac = 'dril_CarRacing-v0_d=beta_NO_PRET_ntrajs=20_seed=68_ppoepoch=1_lr=0.00025_steps=4096_det=False_nup=304.pt'
        path_ac = os.path.join(dir_ac, file_ac)
        actor_critic, obs_rms, args, _, _ = torch.load(path_ac, map_location=device)
        args.num_processes = 8

        score_history = {}
        for ntraj in [1, 20]:
            dir_bc = os.getcwd() + '/trained_models/bc/'
            file_bc = f'bc_CarRacing-v0_d=gaussian_policy_ntrajs={ntraj}_seed=68.model.pth'
            #file_bc= f'bc_CarRacing-v0_d=beta_policy_ntrajs={ntraj}_seed=68.model.pth'
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

    fnames = ['bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=True.pt',  # gauss/clipped
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=True.pt',      # beta/clipped
              'bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=True.pt',  # gauss/bounded
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=True.pt',      # beta/bounded

              'bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=False.pt',  # gauss/clipped
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=False.pt',      # beta/clipped
              'bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=False.pt',  # gauss/bounded
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=False.pt'       # beta/bounded
              ]



    ncols = 8

    fig, ax = plt.subplots(2, ncols, sharey=True, sharex=True)
    plt.suptitle(f'Behavior Cloning Policy Evaluation in Deterministic & Stochastic mode', fontsize=16)

    titles= [ 'Clipped, Gaussian, 1T',
              'Clipped, Gaussian, 20T',
              'Clipped, Beta, 1T',
              'Clipped, Beta, 20T',
              'Bounded, Gaussian, 1T',
              'Bounded, Gaussian, 20T',
              'Bounded, Beta, 1T',
              'Bounded, Beta, 20T']



    color_pack =['gray', 'black', 'green', 'purple']
    for pointer, fname in enumerate(fnames):
        bc_eval_path = os.path.join(dir_bc,fname)
        score_history = torch.load(bc_eval_path)
        N = 100
        expert = 900

        for i, k in enumerate([1, 20]):
            score_history[k] = score_history[k][0:N]
            exp_level = [score_history[k] >= np.ones(N) * expert]
            #print(f'Expert level reached in {np.sum(exp_level)} out of {N} episodes for {k} trajectories')

        for i, k in enumerate([1, 20]):
            idx = pointer * 2 + i
            ax[idx //ncols, idx % ncols].scatter(range(N), score_history[k], color=color_pack[idx%ncols//2])
            print(f'k={k}, len={len(score_history[k])}')

            mean = np.average(score_history[k])
            std = np.std(score_history[k])
            ax[idx //ncols, idx % ncols].plot(range(N), np.average(score_history[k])*np.ones(N), color='orange', linewidth=3, \
                       label=f'scr={mean :.0f} \u00B1 {std:.0f}')
            ax[1, idx % ncols].set_xlabel('Eval episodes')
            ax[idx //ncols, idx % ncols].set_ylim(-200, 1000)
            ax[idx //ncols, 0].set_ylabel(f'Evaluation Score, {"Deterministic" if idx //ncols == 0 else "Stochastic"}')
            #print(f'Min/max score for {k} trajectories: {np.min(score_history[k]):0f}/{np.max(score_history[k]):0f}')
            #ax[j, i].legend(bbox_to_anchor=(0.05, 0.55), loc='lower left', ncol=1, borderaxespad=0., framealpha=0.1)
            ax[0, idx % ncols].set_title(titles[idx % ncols])
            ax[idx //ncols, idx % ncols].legend()


    fig_name = 'cr_bc_eval_summary.eps'
    save_dir = '/home/giovani/hacer/dril/dril/trained_models/bc'
    save_path = os.path.join(save_dir, fig_name)
    plt.savefig(save_path, format='eps')

    fig_name = 'cr_bc_eval_summary.jpg'
    save_dir = '/home/giovani/hacer/dril/dril/trained_models/bc'
    save_path = os.path.join(save_dir, fig_name)
    plt.savefig(save_path, format='jpg')



    plt.show()


############################
######## DRIL BETA #########
############################
def cr_eval_dril_beta():
    EXPERT = 'gaussian'
    device = 'cuda:0'

    if EXPERT == 'gaussian':
        dir_path = os.getcwd() + '/trained_models/dril/backup/dril_b_exp_g'
        file_names = [
            'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=6_scr=507.746515.pt',
            'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=101_scr=130.39531250000002.pt',
            'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=4_scr=755.7266466000001.pt',
            'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=118_scr=439.2860697.pt']

    else:
        dir_path = os.getcwd() + '/trained_models/dril/backup/dril_b_exp_b'
        file_names = [
            'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=2_scr=281.78713569999996.pt',
            'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=102.51775520000001.pt',
            'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=0_scr=693.323746875.pt',
            'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=129.1200822.pt'
        ]


    for det in [True, False]:
        scr_hist = {}
        for file_name in file_names:
            print(f'Starting evaluation of: {file_name} \nin {"Deterministic" if det==True else "Stochastic"} mode')
            path = os.path.join(dir_path, file_name)
            actor_critic, obs_rms, args, scr_train, utrain = torch.load(path, map_location=device)
            eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
            args.num_processes = 8
            scores = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                              device, num_episodes=100, atari_max_steps=None, fname='return_vector', det=det)
            scr_hist[file_name] = scores

        dril_eval_path = os.path.join(dir_path, f'{file_name}_eval_det={det}.pt')
        print(f'Saving DRIL evaluation at {dril_eval_path}')
        torch.save(scr_hist, dril_eval_path)

def plot_dril_beta_p_gaussian_exp():
    # DRIL evaluation files
    files = ['dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=118_scr=439.2860697.pt_eval_det=True.pt',
             'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=118_scr=439.2860697.pt_eval_det=False.pt']

    # keys to evaluation files
    ks = [
        'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=6_scr=507.746515.pt',
        'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=101_scr=130.39531250000002.pt',
        'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=4_scr=755.7266466000001.pt',
        'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=118_scr=439.2860697.pt'
    ]

    titles = ['c) DRIL peak',
              'd) DRIL final',
              'g) DRIL peak',
              'h) DRIL final']

    fig, ax = plt.subplots(3,6, sharey=True, figsize=(16, 10))
    gs = ax[0, 0].get_gridspec()
    for rem_ax in ax[0, 0:6]:
        rem_ax.remove()
    axbig1 = fig.add_subplot(gs[0, 0:3])
    axbig2 = fig.add_subplot(gs[0, 3:])
    axbig = [axbig1, axbig2]

    plt.suptitle(f'DRIL with Beta, Clipped Expert', fontsize=16)
    fontsize = 12
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times New Roman']
    rcParams['font.size'] = fontsize
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.titlesize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    rcParams['text.usetex'] = True
    fig.tight_layout(pad=2.0)



    for j, fname in enumerate(files, start=1):
        dir_path = '/home/giovani/hacer/dril/dril/trained_models/dril/backup/dril_b_exp_g'
        file_path = os.path.join(dir_path, fname)
        scores = torch.load(file_path)
        N=100
        for i,  k in enumerate(ks):
            scores[k] = scores[k][0:N]
            ax[j, i+i//2+1].scatter(range(len(scores[k])), scores[k], s=5, color='black')
            ax[j, i+i//2+1].plot(range(len(scores[k])), np.average(scores[k])*np.ones_like(scores[k]), color='orange', \
                          label=f'Scr: {np.average(scores[k]):.0f} \u00B1 {np.std(scores[k]):.0f}')
            ax[j, i+i//2+1].set_title(titles[i]) if j == 1 else None
            ax[j, i+i//2+1].set_ylim([-200, 1000])
            ax[j, i+i//2+1].legend(loc='lower left')
            #ax[j, i].legend(loc='lower right', fontsize=fontsize)
            ax[2, i+i//2+1].set_xlabel('Eval Episode')
            ax[1, i+i//2+1].get_xaxis().set_visible(False)
            ax[j, 0].set_ylabel(f'Score, {"Deterministic" if j == 1 else "Stochastic"}')
            ax[j,i+i//2+1].set_ylim([-300, 1000])

    final_files = [ks[1], ks[3]]
    for i, fname in enumerate(final_files):
        actor_critic, obs_rms, args, train_hist, u_hist = torch.load(os.path.join(dir_path, fname), map_location='cpu')
        axbig[i].scatter(range(len(train_hist)), train_hist, color='black', s=5, label='Training episode score')
        axbig[i].plot(range(len(train_hist)), moving_avg(train_hist, 100), color='orange',
                      label='100-episode moving average', linewidth=3)
        axbig[i].set_ylim([-300, 1000])
        axbig[i].set_title(f'{"a) 1 Trajectory" if i == 0 else "e) 20 Trajectories"}')
        #axbig[i].legend(loc='upper right')

        axbig[i].get_yaxis().set_visible(False) if i==1 else None
        axbig[i].set_ylabel("Training score") if i == 0 else None
        #ax[j, i].set_xlabel('Training Episode') if j == 0 else None

    fnames = ['bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=True.pt',
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=False.pt']

    for j, fname in enumerate(fnames, start=1):
        dir_path_bc = '/home/giovani/hacer/dril/dril/trained_models/bc/'
        file_path = os.path.join(dir_path_bc, fname)
        scores = torch.load(file_path)

        for i, k in enumerate([1,20]):
            scores[k] = scores[k][0:N]
            ax[j, i*3].scatter(range(len(scores[k])), scores[k], s=5, color='black')
            ax[j, i*3].plot(range(len(scores[k])), np.average(scores[k]) * np.ones_like(scores[k]), color='orange', \
                               label=f'Scr: {np.average(scores[k]):.0f} \u00B1 {np.std(scores[k]):.0f}')
            ax[2, i * 3].set_xlabel('Eval Episode')
            ax[j, i * 3].legend(loc='lower left')
            ax[1, i * 3].set_title(f'{"b) BC" if i == 0 else "f) BC"}')
            ax[1, i * 3].get_xaxis().set_visible(False)

    fig_name = "dril_b_exp_g.eps"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='eps')

    fig_name = "dril_b_exp_g.jpg"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='jpg')

    plt.show()

def plot_dril_beta_p_beta_exp():
    # DRIL evaluation files
    files = ['dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=129.1200822.pt_eval_det=True.pt',
             'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=129.1200822.pt_eval_det=False.pt']

    # keys to evaluation files
    ks = [
        'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=2_scr=281.78713569999996.pt',
        'dril_CarRacing-v0_d=beta_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=102.51775520000001.pt',
        'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=0_scr=693.323746875.pt',
        'dril_CarRacing-v0_d=beta_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=129.1200822.pt'
    ]
    titles = ['c) DRIL peak',
              'd) DRIL final',
              'g) DRIL peak',
              'h) DRIL final']

    fig, ax = plt.subplots(3,6, sharey=True, figsize=(16, 10))
    gs = ax[0, 0].get_gridspec()
    for rem_ax in ax[0, 0:6]:
        rem_ax.remove()
    axbig1 = fig.add_subplot(gs[0, 0:3])
    axbig2 = fig.add_subplot(gs[0, 3:])
    axbig = [axbig1, axbig2]

    plt.suptitle(f'DRIL-Beta, Bounded Expert', fontsize=16)
    fontsize = 12
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times New Roman']
    rcParams['font.size'] = fontsize
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.titlesize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    rcParams['text.usetex'] = True
    fig.tight_layout(pad=2.0)


    N = 100
    for j, fname in enumerate(files, start=1):
        dir_path = '/home/giovani/hacer/dril/dril/trained_models/dril/backup/dril_b_exp_b'
        file_path = os.path.join(dir_path, fname)
        print(file_path)
        scores = torch.load(file_path)

        for i,  k in enumerate(ks):
            scores[k] = scores[k][0:N]

            ax[j, i+i//2+1].scatter(range(len(scores[k])), scores[k], s=5, color='purple')
            ax[j, i+i//2+1].plot(range(len(scores[k])), np.average(scores[k])*np.ones_like(scores[k]), color='orange', \
                          label=f'Scr: {np.average(scores[k]):.0f} \u00B1 {np.std(scores[k]):.0f}')
            ax[j, i+i//2+1].set_title(titles[i]) if j == 1 else None
            ax[j, i+i//2+1].set_ylim([-200, 1000])
            ax[j, i+i//2+1].legend(loc='lower left')
            #ax[j, i].legend(loc='lower right', fontsize=fontsize)
            ax[2, i+i//2+1].set_xlabel('Eval Episode')
            ax[1, i+i//2+1].get_xaxis().set_visible(False)
            ax[j, 0].set_ylabel(f'Score, {"Deterministic" if j == 1 else "Stochastic"}')
            ax[j,i+i//2+1].set_ylim([-300, 1000])

    final_files = [ks[1], ks[3]]
    for i, fname in enumerate(final_files):
        actor_critic, obs_rms, args, train_hist, u_hist = torch.load(os.path.join(dir_path, fname), map_location='cpu')
        axbig[i].scatter(range(len(train_hist)), train_hist, color='purple', s=5, label='Training episode score')
        axbig[i].plot(range(len(train_hist)), moving_avg(train_hist, 100), color='orange',
                      label='100-episode moving average', linewidth=3)
        axbig[i].set_ylim([-300, 1000])
        axbig[i].set_title("a) 1 Trajectory" if i == 0 else "e) 20 Trajectories")
        #axbig[i].legend(loc='upper right')

        axbig[i].get_yaxis().set_visible(False) if i==1 else None
        axbig[i].set_ylabel("Training score") if i == 0 else None
        #ax[j, i].set_xlabel('Training Episode') if j == 0 else None

    fnames = ['bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=True.pt',
              'bc_CarRacing-v0_d=beta_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=False.pt']

    for j, fname in enumerate(fnames, start=1):
        dir_path_bc = '/home/giovani/hacer/dril/dril/trained_models/bc/'
        file_path = os.path.join(dir_path_bc, fname)
        scores = torch.load(file_path)

        for i, k in enumerate([1,20]):
            scores[k] = scores[k][0:N]
            print(f'scores avg ={np.average(scores[k])}, N={len(scores[k])}')
            ax[j, i*3].scatter(range(len(scores[k])), scores[k], s=5, color='purple')
            ax[j, i*3].plot(range(len(scores[k])), np.average(scores[k]) * np.ones_like(scores[k]), color='orange', \
                               label=f'Scr: {np.average(scores[k]):.0f} \u00B1 {np.std(scores[k]):.0f}')
            ax[2, i * 3].set_xlabel('Eval Episode')
            ax[j, i * 3].legend(loc='lower left')
            ax[1, i * 3].set_title(f'{"b) BC" if i == 0 else "f) BC"}')

            ax[1, i * 3].get_xaxis().set_visible(False)

    fig_name = "dril_b_exp_b.eps"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='eps')

    fig_name = "dril_b_exp_b.jpg"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='jpg')

    plt.show()


###########################
###### DRIL Gaussian ######
###########################
def cr_eval_dril_gaussian():
    device = 'cuda:0'
    EXPERT = 'beta'

    if EXPERT == 'gaussian':
        dir_path = os.getcwd() + '/trained_models/dril/backup/dril_g_exp_b'
        file_names = [
            'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=70_scr=514.3864867.pt',
            'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=233.7758806.pt',
            'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=34_scr=814.3645798.pt',
            'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=284.10052320000005.pt']
    else:
        dir_path = os.getcwd() + '/trained_models/dril/backup/dril_g_exp_g'
        file_names =[
            'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=51_scr=532.4610768.pt',
            'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=53.122974799999994.pt',
            'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=34_scr=879.3055492.pt',
            'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=248.5544372.pt']

    for det in [True, False]:
        scr_hist = {}
        for file_name in file_names:
            print(f'Starting evaluation of: {file_name} \nin {"Deterministic" if det==True else "Stochastic"} mode')
            path = os.path.join(dir_path, file_name)
            actor_critic, obs_rms, args, scr_train, utrain = torch.load(path, map_location=device)
            eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
            args.num_processes = 8
            scores = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                              device, num_episodes=100, atari_max_steps=None, fname='return_vector', det=det)
            scr_hist[file_name] = scores

        dril_eval_path = os.path.join(dir_path, f'{file_name}_eval_det={det}.pt')
        print(f'Saving DRIL evaluation at {dril_eval_path}')
        torch.save(scr_hist, dril_eval_path)

def plot_dril_gaussian_p_gaussian_exp():
    # DRIL evaluation files
    files = ['dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=248.5544372.pt_eval_det=True.pt',
             'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=248.5544372.pt_eval_det=False.pt']

    # keys to evaluation files
    ks = [
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=51_scr=532.4610768.pt',
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=53.122974799999994.pt',
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=34_scr=879.3055492.pt',
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=66_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=248.5544372.pt']

    titles = ['c) DRIL peak',
              'd) DRIL final',
              'g) DRIL peak',
              'h) DRIL final']

    fig, ax = plt.subplots(3,6, sharey=True, figsize=(16, 10))
    gs = ax[0, 0].get_gridspec()
    for rem_ax in ax[0, 0:6]:
        rem_ax.remove()
    axbig1 = fig.add_subplot(gs[0, 0:3])
    axbig2 = fig.add_subplot(gs[0, 3:])
    axbig = [axbig1, axbig2]

    plt.suptitle(f'DRIL-Gaussian, Clipped Expert', fontsize=14)
    fontsize = 12
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times New Roman']
    rcParams['font.size'] = fontsize
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.titlesize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    rcParams['text.usetex'] = True
    fig.tight_layout(pad=2.0)


    N=100
    for j, fname in enumerate(files, start=1):
        dir_path = '/home/giovani/hacer/dril/dril/trained_models/dril/backup/dril_g_exp_g'
        file_path = os.path.join(dir_path, fname)
        print(file_path)
        scores = torch.load(file_path)

        for i,  k in enumerate(ks):
            scores[k] = scores[k][0:N]
            ax[j, i+i//2+1].scatter(range(len(scores[k])), scores[k], s=5, color='gray')
            ax[j, i+i//2+1].plot(range(len(scores[k])), np.average(scores[k])*np.ones_like(scores[k]), color='orange', \
                          label=f'Scr: {np.average(scores[k]):.0f} \u00B1 {np.std(scores[k]):.0f}')
            ax[j, i+i//2+1].set_title(titles[i]) if j == 1 else None
            ax[j, i+i//2+1].set_ylim([-200, 1000])
            ax[j, i+i//2+1].legend(loc='lower left')
            #ax[j, i].legend(loc='lower right', fontsize=fontsize)
            ax[2, i+i//2+1].set_xlabel('Eval Episode')
            ax[1, i+i//2+1].get_xaxis().set_visible(False)
            ax[j, 0].set_ylabel(f'Score, {"Deterministic" if j == 1 else "Stochastic"}')
            ax[j,i+i//2+1].set_ylim([-300, 1000])

    final_files = [ks[1], ks[3]]
    for i, fname in enumerate(final_files):
        actor_critic, obs_rms, args, train_hist, u_hist = torch.load(os.path.join(dir_path, fname), map_location='cpu')
        axbig[i].scatter(range(len(train_hist)), train_hist, color='gray', s=5, label='Training episode score')
        axbig[i].plot(range(len(train_hist)), moving_avg(train_hist, 100), color='orange',
                      label='100-episode moving average', linewidth=3)
        axbig[i].set_ylim([-300, 1000])
        axbig[i].set_title("a) 1 Trajectory" if i==0 else "e) 20 Trajectories")
        #axbig[i].legend(loc='upper right')
        axbig[i].get_yaxis().set_visible(False) if i==1 else None
        axbig[i].set_ylabel("Training score") if i == 0 else None
        #ax[j, i].set_xlabel('Training Episode') if j == 0 else None

    fnames = ['bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=True.pt',
              'bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=66.model.pth_NO_PRET_eval_det=False.pt']

    for j, fname in enumerate(fnames, start=1):
        dir_path_bc = '/home/giovani/hacer/dril/dril/trained_models/bc/'
        file_path = os.path.join(dir_path_bc, fname)
        scores = torch.load(file_path)

        for i, k in enumerate([1,20]):
            scores[k] = scores[k][0:N]
            ax[j, i*3].scatter(range(len(scores[k])), scores[k], s=5, color='gray')
            ax[j, i*3].plot(range(len(scores[k])), np.average(scores[k]) * np.ones_like(scores[k]), color='orange', \
                               label=f'Scr: {np.average(scores[k]):.0f} \u00B1 {np.std(scores[k]):.0f}')
            ax[2, i * 3].set_xlabel('Eval Episode')
            ax[j, i * 3].legend(loc='lower left')
            ax[1, i * 3].set_title(f'{"b) BC" if i==0 else "f) BC"}')
            ax[1, i * 3].get_xaxis().set_visible(False)

    fig_name = "dril_g_exp_g.eps"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='eps')

    fig_name = "dril_g_exp_g.jpg"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='jpg')

    plt.show()

def plot_dril_gaussian_p_beta_exp():
    # DRIL evaluation files
    files = [
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=284.10052320000005.pt_eval_det=True.pt',
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=284.10052320000005.pt_eval_det=False.pt']

    # keys to evaluation files
    ks = [
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=70_scr=514.3864867.pt',
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=1_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=233.7758806.pt',
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=34_scr=814.3645798.pt',
        'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=284.10052320000005.pt']

    titles = ['c) DRIL peak',
              'd) DRIL final',
              'g) DRIL peak',
              'h) DRIL final']

    fig, ax = plt.subplots(3,6, sharey=True, figsize=(16, 10))
    gs = ax[0, 0].get_gridspec()
    for rem_ax in ax[0, 0:6]:
        rem_ax.remove()
    axbig1 = fig.add_subplot(gs[0, 0:3])
    axbig2 = fig.add_subplot(gs[0, 3:])
    axbig = [axbig1, axbig2]

    plt.suptitle(f'DRIL-Gaussian, Bounded Expert', fontsize=16)
    fontsize = 12
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times New Roman']
    rcParams['font.size'] = fontsize
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.titlesize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    rcParams['text.usetex'] = True
    fig.tight_layout(pad=2.0)


    N=100
    for j, fname in enumerate(files, start=1):
        dir_path = '/home/giovani/hacer/dril/dril/trained_models/dril/backup/dril_g_exp_b'
        file_path = os.path.join(dir_path, fname)
        print(file_path)
        scores = torch.load(file_path)

        for i,  k in enumerate(ks):
            scores[k] = scores[k][0:N]
            ax[j, i+i//2+1].scatter(range(len(scores[k])), scores[k], s=5, color='green')
            ax[j, i+i//2+1].plot(range(len(scores[k])), np.average(scores[k])*np.ones_like(scores[k]), color='orange', \
                          label=f'Scr: {np.average(scores[k]):.0f} \u00B1 {np.std(scores[k]):.0f}')
            ax[j, i+i//2+1].set_title(titles[i]) if j == 1 else None
            ax[j, i+i//2+1].set_ylim([-200, 1000])
            ax[j, i+i//2+1].legend(loc='lower left')
            #ax[j, i].legend(loc='lower right', fontsize=fontsize)
            ax[2, i+i//2+1].set_xlabel('Eval Episode')
            ax[1, i+i//2+1].get_xaxis().set_visible(False)
            ax[j, 0].set_ylabel(f'Score, {"Deterministic" if j == 1 else "Stochastic"}')
            ax[j,i+i//2+1].set_ylim([-300, 1000])

    final_files = [ks[1], 'dril_CarRacing-v0_d=gaussian_BN3_ntrajs=20_seed=68_ppoepoch=10_lr=0.00025_steps=2048_det=False_nup=365_scr=284.10052320000005.pt']
    for i, fname in enumerate(final_files):
        actor_critic, obs_rms, args, train_hist, u_hist = torch.load(os.path.join(dir_path, fname), map_location='cpu')
        axbig[i].scatter(range(len(train_hist)), train_hist, color='green', s=5, label='Training episode score')
        axbig[i].plot(range(len(train_hist)), moving_avg(train_hist, 100), color='orange',
                      label='100-episode moving average', linewidth=3)
        axbig[i].set_ylim([-300, 1000])
        axbig[i].set_title("a) 1 Trajectory" if i == 0 else "e) 20 Trajectories")
        axbig[i].get_yaxis().set_visible(False) if i==1 else None
        axbig[i].set_ylabel("Training score") if i == 0 else None
        #ax[j, i].set_xlabel('Training Episode') if j == 0 else None

    fnames = ['bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=True.pt',
              'bc_CarRacing-v0_d=gaussian_policy_ntrajs=20_seed=68.model.pth_NO_PRET_eval_det=False.pt']

    for j, fname in enumerate(fnames, start=1):
        dir_path_bc = '/home/giovani/hacer/dril/dril/trained_models/bc/'
        file_path = os.path.join(dir_path_bc, fname)
        scores = torch.load(file_path)

        for i, k in enumerate([1,20]):
            scores[k] = scores[k][0:N]
            ax[j, i*3].scatter(range(len(scores[k])), scores[k], s=5, color ='green')
            ax[j, i*3].plot(range(len(scores[k])), np.average(scores[k]) * np.ones_like(scores[k]), color='orange', \
                               label=f'Scr: {np.average(scores[k]):.0f} \u00B1 {np.std(scores[k]):.0f}')
            ax[2, i * 3].set_xlabel('Eval Episode')
            ax[j, i * 3].legend(loc='lower left')
            ax[1, i * 3].set_title(f'{"b) BC" if i == 0 else "f) BC"}')

            ax[1, i * 3].get_xaxis().set_visible(False)

    fig_name = "dril_g_exp_b.eps"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='eps')

    fig_name = "dril_g_exp_b.jpg"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='jpg')

    plt.show()




if __name__ == '__main__':

    #eval_bc()
    #plot_eval_bc()

    #cr_eval_dril_beta()
    plot_dril_beta_p_gaussian_exp()
    #plot_dril_beta_p_beta_exp()

    #cr_eval_dril_gaussian()
    #plot_dril_gaussian_p_gaussian_exp()
    #plot_dril_gaussian_p_beta_exp()
