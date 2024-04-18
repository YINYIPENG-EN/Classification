import os

import scipy.signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        self.train_acc = []
        self.val_acc = []
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):  # 训练的loss和测试的loss
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()
    def append_acc(self, train_acc, val_acc):
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        with open(os.path.join(self.save_path, "epoch_train_acc_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(train_acc))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_acc_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_acc))
            f.write("\n")
        self.acc_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

    def acc_plot(self):
        iters = range(len(self.train_acc))
        plt.figure()
        plt.plot(iters, self.train_acc, 'red', linewidth=2, label='train acc')
        plt.plot(iters, self.val_acc, 'coral', linewidth=2, label='val acc')
        try:
            if len(self.train_acc) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.train_acc, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val acc')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.save_path, "epoch_acc_" + str(self.time_str) + ".png"))
        plt.cla()
        plt.close("all")
