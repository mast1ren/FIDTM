import os
for set in ['train', 'test', 'val']:
    sequence = []
    gt = []
    for root, dirnames, filenames in os.walk(os.path.join('.', set)):
        for file in filenames:
            if file.endswith('jpg'):
                seq = file[3:6]
                if seq not in sequence:
                    sequence.append(seq)
            if file.endswith('h5'):
                seq = file[6:9]
                if seq not in gt:
                    gt.append(seq)
    sequence.sort()
    gt.sort()
    print(sequence, gt)