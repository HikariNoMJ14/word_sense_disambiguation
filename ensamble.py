import pandas as pd


def ensamble_results(filename_1, filename_2, filename_out, w=0.5):
    file_1 = pd.read_csv(filename_1, delimiter=' ', header=None, names=['pred', 'prob_0', 'prob_1'])
    file_2 = pd.read_csv(filename_2, delimiter=' ', header=None, names=['pred', 'prob_0', 'prob_1'])

    print(file_1.head(10))
    print(file_2.head(10))

    file_1['ensamble_prob_0'] = w * file_1['prob_0'] + (1-w) * file_2['prob_0']
    file_1['ensamble_prob_1'] = w * file_1['prob_1'] + (1-w) * file_2['prob_1']
    file_1['ensamble_pred'] = file_1['ensamble_prob_0'] < file_1['ensamble_prob_1']
    file_1['ensamble_pred'] = file_1['ensamble_pred'].astype('int')

    print(file_1[['ensamble_pred', 'ensamble_prob_0', 'ensamble_prob_1']].head(10))

    file_1[['ensamble_pred', 'ensamble_prob_0', 'ensamble_prob_1']].to_csv(filename_out, sep=" ", header=False, index=None)


if __name__ == "__main__":

    w = 0.5

    results_1 = '../GlossBERT/results/context_aug/1314/1/results.txt'
    results_2 = '../GlossBERT/results/de_best/1314/1/results.txt'
    results_out = f'../GlossBERT/results/ensamble/1314/1/{str(w).replace(".", "_")}/results.txt'

    print(results_out)

    ensamble_results(results_1, results_2, results_out, w)
