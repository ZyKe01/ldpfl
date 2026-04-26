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
    plt.figure()
    dataset = 'mnist'
    epsilon = 5.0
    k_array = ['0.0', '0.001', '0.003', '0.005', '0.007', '0.009']
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Global Round')
    for k in k_array:
        y = openfile(
            f'./test_log/accfile_fed_{dataset}_cnn_150_iidTrue_dp_Gaussian_epsilon_{epsilon}_threshold_{k}_robust_no_no.dat')
        plt.plot(range(150), y, label=r'$k={}$'.format(k))
    plt.title(f'{dataset} Gaussian $\epsilon={epsilon}$')
    plt.legend()
    plt.savefig(f'result/{dataset}_gaussian_{epsilon}__threshold.png')
