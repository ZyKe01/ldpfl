import matplotlib.pyplot as plt
import matplotlib as mpl


def openfile(filepath):
    file = open(filepath)
    y = []
    while 1:
        line = file.readline()
        if line.rstrip('\n') == '':
            break
        y.append(float(line.rstrip('\n')))
        if not line:
            break
        pass
    file.close()
    return y

def draw_lamb(epsilon):
    lamb_list = [0.0, 0.1, 0.25, 0.5, 0.75]
    attack_list = ['flip']
    for attack in attack_list:
        plt.figure()
        plt.ylabel('Testing Accuracy')
        plt.xlabel('Global Round')
        for lamb in lamb_list:
            y = openfile(
                f'./test_log/accfile_fed_mnist_cnn_100_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_0.01_robust_mine_{attack}_lambda_{lamb}_before25.dat')
            plt.plot(range(100), y, label=f'lambda = {lamb}')
        
        plt.title(f'Mnist Gaussian $\epsilon={epsilon}$')
        plt.legend()
        plt.savefig(f'image/mnist_gaussian_{epsilon}_robust_{attack}_ablation_{ablation}.png')
        
def draw_begin_group(epsilon):
    iter_list = [0, 25, 100]
    attack_list = ['flip']
    for attack in attack_list:
        plt.figure()
        plt.ylabel('Testing Accuracy')
        plt.xlabel('Global Round')
        for iter in iter_list:
            y = openfile(
                f'./test_log/accfile_fed_mnist_cnn_100_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_0.01_robust_mine_{attack}_lambda_0.75_before{iter}.dat')
            plt.plot(range(100), y, label=f'iter = {iter}')
        
        plt.title(f'Mnist Gaussian $\epsilon={epsilon}$')
        plt.legend()
        plt.savefig(f'image/mnist_gaussian_{epsilon}_robust_{attack}_ablation_{ablation}.png')

# test_log/accfile_fed_mnist_cnn_100_iidFalse_dp_Gaussian_epsilon_4.0_top1.0.dat
if __name__ == '__main__':
    
    ablation = 'begin_group'
    epsilon = 1.0
    if ablation == 'lambda':
        draw_lamb(epsilon)
    elif ablation == 'begin_group':
        draw_begin_group(epsilon)
