import matplotlib.pyplot as plt

def plot_dirty_input(idx, size, x, save_path):
    plt.figure(figsize=(18, 3))
    for i in range(size):
        plt.plot(x[idx, :, -i].cpu().flatten(), lw=2)
    plt.grid(color='k', ls ='--', lw=1)
    plt.title('Dirty Input', fontsize=24)
    plt.tick_params('both', labelsize=18)
    plt.ylim((-2, 2))
    plt.savefig(save_path+f"dirty_input.png", dpi=300)
    plt.show()

def plot_original_data(idx, size, data, save_path):
    plt.figure(figsize=(18, 3))
    for i in range(size):
        plt.plot(data[idx, :, -i].cpu().flatten(), lw=2)
    plt.grid(color='k', ls ='--', lw=1)
    plt.title('Original Data', fontsize=24)
    plt.tick_params('both', labelsize=18)
    plt.ylim((-2, 2))
    plt.savefig(save_path+f"original.png", dpi=300)
    plt.show()

def plot_reconstruction(idx, size, trend, season, r, save_path):
    plt.figure(figsize=(18, 3))
    for i in range(size):
        plt.plot((trend + season + r)[idx, :, -i].cpu().detach().numpy().flatten(), lw=2)
    plt.grid(color='k', ls ='--', lw=1)
    plt.title('Reconstruction', fontsize=24)
    plt.tick_params('both', labelsize=18)
    plt.ylim((-2, 2))
    plt.savefig(save_path+f"reconstruction.png", dpi=300)
    plt.show()

def plot_original_season(idx, size, season_r, save_path):
    plt.figure(figsize=(18, 3))
    for i in range(size):
        plt.plot(season_r[idx, :, -i].cpu().flatten(), lw=2)
    plt.grid(color='k', ls ='--', lw=1)
    plt.title('Original Season', fontsize=24)
    plt.tick_params('both', labelsize=18)
    plt.ylim((-1, 1))
    plt.savefig(save_path+f"orig_season.png", dpi=300)
    plt.show()

def plot_learnt_season(idx, size, season, save_path):
    plt.figure(figsize=(18, 3))
    for i in range(size):
        plt.plot(season[idx, :, -i].cpu().detach().numpy().flatten(), lw=2)
    plt.grid(color='k', ls ='--', lw=1)
    plt.title('Learnt Season', fontsize=24)
    plt.tick_params('both', labelsize=18)
    plt.ylim((-1, 1))
    plt.savefig(save_path+f"learnt_season.png", dpi=300)
    plt.show()

def plot_original_trend(idx, size, trend_r, save_path):
    plt.figure(figsize=(18, 3))
    for i in range(size):
        plt.plot(trend_r[idx, :, -i].cpu().flatten(), lw=2)
    plt.grid(color='k', ls ='--', lw=1)
    plt.title('Original Trend', fontsize=24)
    plt.tick_params('both', labelsize=18)
    plt.ylim((-1, 1))
    plt.savefig(save_path+f"orig_trend.png", dpi=300)
    plt.show()

def plot_learnt_trend(idx, size, trend, save_path):
    plt.figure(figsize=(18, 3))
    for i in range(size):
        plt.plot(trend[idx, :, -i].cpu().detach().numpy().flatten(), lw=2)
    plt.grid(color='k', ls ='--', lw=1)
    plt.title('Learnt Trend', fontsize=24)
    plt.tick_params('both', labelsize=18)
    plt.ylim((-1, 1))
    plt.savefig(save_path+f"learnt_trend.png", dpi=300)
    plt.show()

def plot_learnt_residual(idx, size, r, save_path):
    plt.figure(figsize=(18, 3))
    for i in range(size):
        plt.plot(r[idx, :, -i].cpu().detach().numpy().flatten(), lw=2)
    plt.grid(color='k', ls ='--', lw=1)
    plt.title('Learnt Residual', fontsize=24)
    plt.tick_params('both', labelsize=18)
    plt.ylim((-1, 1))
    plt.savefig(save_path+f"learnt_residual.png", dpi=300)
    plt.show()