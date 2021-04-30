from scipy import io as scio 
import numpy as np 
import os 

train_ratio = 0.7

folder = '/root/data/SynthText'

if __name__ == '__main__':

    print('Loading data')
    imnames = np.load(os.path.join(folder, 'imnames.npy'), allow_pickle=True)
    charBB = np.load(os.path.join(folder, 'charBB.npy'),allow_pickle=True)
    aff_charBB = np.load(os.path.join(folder, 'aff_charBB.npy'), allow_pickle=True)

    num_image_full = imnames.shape[0]

    image_index = np.array([x for x in range(num_image_full)])

    #train random choice 

    num_train = int(num_image_full*train_ratio)
    train_index = np.random.choice(num_image_full, num_train, replace=False)

    #
    index_without_train = np.delete(image_index, train_index)

    #
    val_ratio = (1-train_ratio)/2
    num_val = int(num_image_full*val_ratio)

    val_index_ = np.random.choice(num_image_full - num_train , num_val , replace=False)
    val_index = index_without_train[val_index_]

    test_index = np.delete(index_without_train, val_index_)
    num_test = test_index.shape[0]

    num_sum = num_train + num_val + num_test 

    print('Number of Full image : {}'.format(num_image_full))
    print('train : {} + valid : {} + test : {}  = {}'.format(num_train, num_val, num_test, num_sum))

    #
    print('Split index...')
    train_imnames = imnames[train_index]
    train_charBB = charBB[train_index]
    train_aff_charBB = aff_charBB[train_index]

    val_imnames = imnames[val_index]
    val_charBB = charBB[val_index]
    val_aff_charBB = aff_charBB[val_index]

    test_imnames = imnames[test_index]
    test_charBB = charBB[test_index]
    test_aff_charBB = aff_charBB[test_index]

    front = ['train' , 'val' , 'test']
    rear = ['imnames' , 'charBB', 'aff_charBB']

    for f in front: 
        for r in rear: 
            np.save('/root/data/SynthText/splitdata/{}_{}'.format(f,r) , vars()['{}_{}'.format(f,r)] )

    print('Completed')