import os
import glob
import random
import argparse

CLASS_NAME = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Q2 and Q3 TODO : select "better" images from Q2 folder
def select_imaegs(image_paths, images_num=200):
    """
    :param image_paths: --> ['your_folder/images1.jpg', 'your_folder/images2.jpg', ...]
    :param images_num: choose the number of images
    :return :
        selected_image_paths = ['your_folder/images10.jpg', 'your_folder/images12.jpg', ...]
    """
    # TODO : select images
    ############################
    #########  Q2  #########
    ############################
    #ver1
    '''
    num_folders = 6
    num_images_per_folder = 200
    total_images = num_folders * num_images_per_folder
    
    # 先隨機取每個資料夾中的30張
    selected_image_paths = []

    for folder_start in range(0, total_images, num_images_per_folder):
        folder_paths = image_paths[folder_start : folder_start + num_images_per_folder]
        random.shuffle(folder_paths)
        selected_image_paths.extend(folder_paths[:30])

    remaining_images = list(set(image_paths) - set(selected_image_paths))
    remaining_images_number = len(remaining_images)

    for folder_start in range(0, remaining_images_number, 180):
        folder_paths = remaining_images[folder_start : folder_start + 180]
        random.shuffle(folder_paths)
        selected_image_paths.extend(folder_paths[30:33])

    # 再隨機取剩下的未被選上的20張
    selct_2 = list(set(image_paths) - set(selected_image_paths))
    random.shuffle(selct_2)
    selected_image_paths.extend(selct_2[:2])
    '''
    #ver2
    '''
    selected_image_paths = []
    
    camera_173_paths = image_paths[:200]
    random.shuffle(camera_173_paths)

    camera_398_paths = image_paths[200:400]
    random.shuffle(camera_398_paths)

    camera_170_paths = image_paths[400:600]
    random.shuffle(camera_170_paths)

    camera_410_paths = image_paths[600:800]
    random.shuffle(camera_410_paths)

    camera_511_paths = image_paths[800:1000]
    random.shuffle(camera_511_paths)

    camera_495_paths = image_paths[1000:]
    random.shuffle(camera_495_paths)

    selected_image_paths.extend(camera_173_paths[:20])
    selected_image_paths.extend(camera_398_paths[:35])
    selected_image_paths.extend(camera_170_paths[:45])
    selected_image_paths.extend(camera_410_paths[:45])
    selected_image_paths.extend(camera_511_paths[:20])
    selected_image_paths.extend(camera_495_paths[:35])
    '''

    ############################
    #########  Q3  #########
    ############################

    num_folders = 6
    num_images_per_folder = 200
    total_images = num_folders * num_images_per_folder
    
    # 先隨機取每個資料夾中的30張
    selected_image_paths = []

    for folder_start in range(0, total_images, num_images_per_folder):
        folder_paths = image_paths[folder_start : folder_start + num_images_per_folder]
        random.shuffle(folder_paths)
        selected_image_paths.extend(folder_paths[:30])

    remaining_images = list(set(image_paths) - set(selected_image_paths))
    remaining_images_number = len(remaining_images)

    for folder_start in range(0, remaining_images_number, 180):
        folder_paths = remaining_images[folder_start : folder_start + 180]
        random.shuffle(folder_paths)
        selected_image_paths.extend(folder_paths[30:33])

    # 再隨機取剩下的未被選上的20張
    selct_2 = list(set(image_paths) - set(selected_image_paths))
    random.shuffle(selct_2)
    selected_image_paths.extend(selct_2[:2])

    with open('/content/drive/MyDrive/CV_hw4/Complete/Q2/train_Q2_ver2.txt', 'r') as file1:
        paths1 = file1.readlines()
    with open('/content/drive/MyDrive/CV_hw4/Complete/Q2/val_Q2_ver2.txt', 'r') as file2:
        paths2 = file2.readlines()

    Q2_select = paths1 + paths2
    selected_image_paths.extend(Q2_select)

    return selected_image_paths

# TODO : split train and val images
def split_train_val_path(all_image_paths, train_val_ratio=0.9):
    """
    :param all_image_paths: all image paths for question in the data folder
    :param train_val_ratio: ratio of image paths used to split training and validation
    :return :
        train_image_paths = ['your_folder/images1.jpg', 'your_folder/images2.jpg', ...]
        val_image_paths = ['your_folder/images3.jpg', 'your_folder/images4.jpg', ...]
    """
    # TODO : split train and val

    ############################
    #########  Q1  #########
    ############################
    # ver1
    '''
    train_image_paths = all_image_paths[: int(len(all_image_paths) * train_val_ratio)]  # just an example
    val_image_paths = all_image_paths[int(len(all_image_paths) * train_val_ratio):]  # just an example
    '''
    # ver2
    '''
    camera_398_paths = all_image_paths[:160]
    random.shuffle(camera_398_paths)

    camera_173_paths = all_image_paths[160:]
    random.shuffle(camera_173_paths)

    train_image_paths = camera_398_paths[: int(len(camera_398_paths) * train_val_ratio)] + camera_173_paths[: int(len(camera_173_paths) * train_val_ratio)]
    val_image_paths = camera_398_paths[int(len(camera_398_paths) * train_val_ratio):] + camera_173_paths[int(len(camera_173_paths) * train_val_ratio):]
    '''

    ############################
    #########  Q2  #########
    ############################
    # ver1
    '''
    random.shuffle(all_image_paths)

    
    train_image_paths = all_image_paths[: int(len(all_image_paths) * train_val_ratio)]  # just an example
    val_image_paths = all_image_paths[int(len(all_image_paths) * train_val_ratio):]  # just an example
    '''
    # ver2
    
    import re
    folders = {'173': [], '398': [], '170': [], '410': [], '511': [], '495': []}

    # 提取文件路径中的数字并进行分类
    for path in all_image_paths:
        match = re.search(r'/(\d+)-', path)
        if match:
            folder_number = match.group(1)
            if folder_number in folders:
                folders[folder_number].append(path)

    random.shuffle(folders['173'])
    camera_173_paths_train = folders['173'][: int(len(folders['173']) * train_val_ratio)]
    camera_173_paths_val = folders['173'][int(len(folders['173']) * train_val_ratio):]

    random.shuffle(folders['398'])
    camera_398_paths_train = folders['398'][: int(len(folders['398']) * train_val_ratio)]
    camera_398_paths_val = folders['398'][int(len(folders['398']) * train_val_ratio):]

    random.shuffle(folders['170'])
    camera_170_paths_train = folders['170'][: int(len(folders['170']) * train_val_ratio)]
    camera_170_paths_val = folders['170'][int(len(folders['170']) * train_val_ratio):]

    random.shuffle(folders['410'])
    camera_410_paths_train = folders['410'][: int(len(folders['410']) * train_val_ratio)]
    camera_410_paths_val = folders['410'][int(len(folders['410']) * train_val_ratio):]

    random.shuffle(folders['511'])
    camera_511_paths_train = folders['511'][: int(len(folders['511']) * train_val_ratio)]
    camera_511_paths_val = folders['511'][int(len(folders['511']) * train_val_ratio):]

    random.shuffle(folders['495'])
    camera_495_paths_train = folders['495'][: int(len(folders['495']) * train_val_ratio)]
    camera_495_paths_val = folders['495'][int(len(folders['495']) * train_val_ratio):]


    train_image_paths = camera_173_paths_train + camera_398_paths_train + camera_170_paths_train + camera_410_paths_train + camera_511_paths_train + camera_495_paths_train
    val_image_paths = camera_173_paths_val + camera_398_paths_val + camera_170_paths_val + camera_410_paths_val + camera_511_paths_val + camera_495_paths_val
    
    return train_image_paths, val_image_paths
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./data/CityCam', help='path of CityCam datasets folder')
    parser.add_argument('--ques', type=str, default='Q1', choices=['Q1', 'Q2', 'Q3'], help='question in data_folder')
    args = parser.parse_args()
    print(args)

    # Get whole and Test image paths
    all_image_paths = glob.glob(os.path.join(args.data_folder, args.ques, '*', '*.jpg'))
    # print('all', all_image_paths)
    test_image_paths = glob.glob(os.path.join(args.data_folder, 'test', '*' + os.sep + '*.jpg'))

    # for Q2 and Q3 : select images
    if args.ques == 'Q2' or args.ques == 'Q3':
        selected_image_paths = select_imaegs(all_image_paths, images_num=200)
    else:
        selected_image_paths = all_image_paths
    # split Train and Val
    train_image_paths, val_image_paths = split_train_val_path(selected_image_paths)

    # write train/val/test info
    train_path = os.path.join(args.data_folder, 'train.txt')
    val_path = os.path.join(args.data_folder, 'val.txt')
    test_path = os.path.join(args.data_folder, 'test.txt')
    with open(train_path, 'w') as f:
        for image_path in train_image_paths:
            f.write(os.path.abspath(image_path) + '\n')
    with open(val_path, 'w') as f:
        for image_path in val_image_paths:
            f.write(os.path.abspath(image_path) + '\n')
    with open(test_path, 'w') as f:
        for image_path in test_image_paths:
            f.write(os.path.abspath(image_path) + '\n')

    # write training YAML file
    with open('./data/citycam.yaml', 'w') as f:
        f.write("train: " + os.path.abspath(train_path) + "\n")
        f.write("val: " + os.path.abspath(val_path) + "\n")
        f.write("test: " + os.path.abspath(test_path) + "\n")
        # number of classes
        f.write('nc: 80\n')
        # class names
        f.write('names: ' + str(CLASS_NAME))

    # delete cache
    if os.path.exists(os.path.join(args.data_folder, 'train.cache')):
        os.remove(os.path.join(args.data_folder, 'train.cache'))
    if os.path.exists(os.path.join(args.data_folder, 'val.cache')):
        os.remove(os.path.join(args.data_folder, 'val.cache'))
    """
    if os.path.exists(os.path.join(args.data_folder, 'test.cache')):
        os.remove(os.path.join(args.data_folder, 'test.cache'))
        """
