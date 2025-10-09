import json
import os
import matplotlib.pyplot as plt


def get_log(path):
    epoch = []
    train_lr = []
    train_loss = []
    test_loss = []
    test_acc1 = []
    decoder = json.JSONDecoder()
    log = open(os.path.join(path, 'log.txt'), encoding='utf-8')
    data = log.readlines()
    for data_line in data:
        data_line = data_line.strip('\n')
        data_line = decoder.raw_decode(data_line)
        print(data_line)
        data_line = data_line[0]
        epoch_line = data_line['epoch']
        epoch.append(epoch_line)
        lr_line = data_line['train_lr']
        train_lr.append(lr_line)
        loss_line = data_line['train_loss']
        train_loss.append(loss_line)
        test_los_line = data_line['test_loss']
        test_loss.append(test_los_line)
        acc1_line = data_line['test_acc1']
        test_acc1.append(acc1_line)
    log.close()
    return epoch, train_lr, train_loss, test_loss, test_acc1


path = 'output_dir_finetune/'
path_noise = 'output_dir_finetune/'
epoch, train_lr, train_loss, test_loss, test_acc1 = get_log(path)
epoch_noise, train_lr_noise, train_loss_noise, test_loss_noise, test_acc1_noise = get_log(path_noise)
# 绘制test_acc1的曲线图
plt.figure()
plt.plot(test_acc1, color='r', label='test accuracy of multi-task pre-trained')
plt.plot(test_acc1_noise, color='b', label='test accuracy of none pre-trained')
# plt.title('Test Accuracy Over Time')
plt.xlabel('Epoch')
# plt.ylabel('test accuracy')
plt.legend()
plt.show()
plt.savefig(os.path.join(path, 'acd_acc.png'))


plt.figure()
plt.plot(train_loss, color='r', label='train loss of multi-task pre-trained')
plt.plot(train_loss_noise, color='b', label='train loss of none pre-trained')
# plt.title('Test Accuracy Over Time')
plt.xlabel('Epoch')
# plt.ylabel('test accuracy')
plt.legend()
plt.show()
plt.savefig(os.path.join(path, 'acd_loss.png'))
