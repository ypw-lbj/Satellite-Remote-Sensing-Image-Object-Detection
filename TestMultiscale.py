import os
import tqdm

epoch_list = tqdm.tqdm(range(500,800,10))
for i in epoch_list:
    os.system('python refinedet_train_test.py --subsize ' + str(i))