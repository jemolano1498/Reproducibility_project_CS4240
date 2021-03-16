from data_loader import *
def run():
    D = load_data('data/ihdp_npci_1-100.train.npz')
    print('Data loaded')

if __name__ == '__main__':
    run()