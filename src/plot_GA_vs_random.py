import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def plot_ga_vs_random(gen, max_, random_fitnesses, best_set, save_path='plots'):
    plt.figure(figsize=(15, 6))
    plt.plot(gen, max_, color='blue', label='GA', linewidth=1)
    plt.plot(random_fitnesses, color='orange', label='Random baseline', alpha=0.7, linewidth=1)
    plt.plot(best_set, color='red', label='Best random set', alpha=0.7, linewidth=1)  # New line for best set
    plt.xlabel('Number of generations', fontsize=18)
    plt.ylabel("NMF mean Pearson correlation score", fontsize=18)
    # plt.title('Dataset 1: Go et al., 2021', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.ylim(0,1)
    plt.xlim(1,1000)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path}/GA_vs_Random_plot.png", dpi=300)
    plt.savefig(f"{save_path}/GA_vs_Random_plot.svg", dpi=300)
    plt.savefig(f"{save_path}/GA_vs_Random_plot.pdf", dpi=300)
    # plt.show()  # Show the plot in addition to saving it
    plt.clf()



