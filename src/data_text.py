import os
import os.path
from tqdm import tqdm


def get_files_list(dir, label):
    """
    实现遍历dir目录下,所有文件
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    :parent:父目录, filenames:该目录下所有文件夹, filenames:该目录下的文件名
    """
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):

        for filename in filenames:
            # print("\nparent is: " + parent)
            # print("\nfilename is: " + filename)
            # print(os.path.join(parent, filename))
            # curr_file = parent.split(os.sep)[-1]
            # 分离出子文件夹的名字

            files_list.append([os.path.join(parent, filename), label])

    # print('\nfiles_list is :', files_list)
    # print('\ntotal image number is %d' % len(files_list))
    # print('\n')
    return files_list


def write_txt(content, filename, mode='w'):

    """
    保存 text 数据,写入对应的文件中
    mode= ‘w’ => 写文件
    :param content:需要保存的数据,type => list
    :param filename:文件名
    :return: void
    """
    with open(filename, mode) as f:
        for line in tqdm(content):  # tqdm => 显示进度条工具
            str_line = ""
            for col, data in enumerate(line):
                # 使用enumerate函数，会将line拆解成 序号+列表内容 的形式
                # col代表序号 0和 1, 0 => jpg ,1 => label
                if not col == len(line) - 1:
                    str_line = str_line + str(data) + " "  # 以空格作为分隔符
                else:
                    str_line = str_line + str(data) + "\n"

                # data = jpg +“ ” + label.
            f.write(str_line)

if __name__ == '__main__':
    # # 少量数据集进行测试
    # trainCat_dir = 'D:\\desktop\\Pyexp\\data\\train\\cat'
    # trainCat_txt = 'D:\\desktop\\Pyexp\\src\\dataset\\test\\train.txt'
    # trainCat_data = get_files_list(trainCat_dir, 0)
    #
    # trainDog_dir = 'D:\\desktop\\Pyexp\\data\\dataset\\train\\dog'
    # trainDog_txt = 'D:\\desktop\\Pyexp\\src\\dataset\\train.txt'
    # trainDog_data = get_files_list(trainDog_dir, 1)
    # train_data = trainCat_data + trainDog_data
    # write_txt(train_data, trainDog_txt, mode='w')
    #
    # test_dir = 'D:\\desktop\\Pyexp\\data\\dataset\\test1'
    # test_txt = 'D:\\desktop\\Pyexp\\src\\dataset\\test.txt'
    # test_data = get_files_list(test_dir, "")
    # write_txt(test_data, test_txt, mode='w')

    # 官网数据 => 12500
    trainCat_dir = 'D:\\desktop\\Pyexp\\data\\train\\cat'
    trainCat_txt = 'D:\\desktop\\Pyexp\\src\\dataset\\train.txt'
    trainCat_data = get_files_list(trainCat_dir, 0)

    trainDog_dir = 'D:\\desktop\\Pyexp\\data\\train\\dog'
    trainDog_txt = 'D:\\desktop\\Pyexp\\src\\dataset\\train.txt'
    trainDog_data = get_files_list(trainDog_dir, 1)
    train_data = trainCat_data + trainDog_data
    write_txt(train_data, trainDog_txt, mode='w')

    test_dir = 'D:\\desktop\\Pyexp\\data\\test'
    test_txt = 'D:\\desktop\\Pyexp\\src\\dataset\\test.txt'
    test_data = get_files_list(test_dir, "")
    write_txt(test_data, test_txt, mode='w')
