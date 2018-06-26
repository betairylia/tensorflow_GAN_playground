import numpy as np

def load_data(path):
    files = os.listdir(path)
    numImages = len(files)

    data = np.zeros((numImages, height, width, 3), dtype=np.uint8)
    cnt = 0

    for file in files:
        with Image.open(os.path.join(path, file)) as im:
            w, h = im.size
            x = random.randint(0, w-input_crop_size-1)
            y = random.randint(0, h-input_crop_size-1)
            im_crop = im.crop(x, y, x+input_crop_size, y+input_crop_size)

            data[cnt, :, :, :] = np.array(im_crop)[...,:3]
            cnt = cnt + 1

            if cnt % 20 == 0:
                print(cnt)

    np.random.shuffle(data)
    return data

def create_batch(t, batch_size, src, gray=False):
    X = np.zeros((batch_size, height, width, 3 if not gray else 1), dtype=np.float32)

    for k, image in enumerate(src[t:t+batch_size]):
        if gray:
            X[k, :, :, :] = rgb2gray(image)
        else:
            X[k, :, :, :] = image / 255.0

    return X

def create_batch_Effecient(filePath):
    X = np.zeros((batch_size, input_crop_size, input_crop_size, 3), dtype=np.float32)

    files = os.listdir(filePath)
    numImages = len(files)

    for i in range(batch_size):
        index = random.randint(0, numImages - 1)
        with Image.open(os.path.join(filePath, files[index])) as im:
            w, h = im.size
            x = random.randint(0, w-input_crop_size-1)
            y = random.randint(0, h-input_crop_size-1)
            im_crop = im.crop(x, y, x+input_crop_size, y+input_crop_size)
            X[i, :, :, :] = np.array(im_crop)[...,:3] / 255.0

    return X
