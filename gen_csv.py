import os
import glob
import random
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_csv(root, filename, name2label):
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            images += glob.glob(os.path.join(root, name, '*.png'))
        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                label = name2label[name]
                writer.writerow([img, label])
            print('written into csv file:', filename)

    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label)
            img = './' + str(img)
            images.append(img)
            labels.append(label)
    return images, labels


def load_train(root, mode='train'):
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys())
    images, labels = load_csv(root, 'traindata.csv', name2label)
    if mode == 'train':
        images = images
        labels = labels
    return images, labels, name2label


def load_test(root, mode='test'):
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys())
    images, labels = load_csv(root, 'testdata.csv', name2label)
    if mode == 'test':
        images = images
        labels = labels
    return images, labels, name2label


def load_val(root, mode='val'):
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys())
    images, labels = load_csv(root, 'valdata.csv', name2label)
    if mode == 'val':
        images = images
        labels = labels
    return images, labels, name2label


def main():
    images1, labels1, table1 = load_test('train', 'train')
    print('images:', len(images1), images1)
    print('labels:', len(labels1), labels1)
    print('table:', table1)


if __name__ == '__main__':
    main()
