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


# test_log/accfile_fed_mnist_cnn_100_iidFalse_dp_Gaussian_epsilon_4.0_top1.0.dat
if __name__ == '__main__':
    
    epsilon = 1.0
    attack_list = ['random', 'negative', 'flip']
    attack_list = ['random']
    epoch = 150
    for attack in attack_list:
        plt.figure()
        plt.ylabel('Testing Accuracy')
        plt.xlabel('Global Round')
        y = openfile(
            f'./test_log/accfile_fed_mnist_cnn_{epoch}_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_None_robust_no_{attack}.dat')
        plt.plot(range(epoch), y, label=f'{attack} no robust')
        y = openfile(
            f'./test_log/accfile_fed_mnist_cnn_{epoch}_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_0.005_robust_mine_{attack}_lambda_0.75_before25.dat')
        plt.plot(range(epoch), y, label=f'{attack} ours')
        y = openfile(
            f'./test_log/accfile_fed_mnist_cnn_{epoch}_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_None_robust_Krum_{attack}.dat')
        plt.plot(range(epoch), y, label=f'{attack} Krum')
        y = openfile(
            f'./test_log/accfile_fed_mnist_cnn_{epoch}_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_None_robust_FLtrust_{attack}.dat')
        plt.plot(range(epoch), y, label=f'{attack} FLtrust')
        y = openfile(
            f'./test_log/accfile_fed_mnist_cnn_{epoch}_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_None_robust_RoWA_{attack}.dat')
        plt.plot(range(epoch), y, label=f'{attack} RoWA')
        y = openfile(
            f'./test_log/accfile_fed_mnist_cnn_{epoch}_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_None_robust_tdsc_{attack}.dat')
        plt.plot(range(epoch), y, label=f'{attack} Zhou')

        plt.title(f'Mnist Gaussian $\epsilon={epsilon}$')
        plt.legend()
        plt.savefig(f'result/mnist_gaussian_{epsilon}_robust_{attack}_comp.png')
