from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle


class LRFinder(Callback):
    def __init__(self, min_lr, max_lr, mom=0.98, stop_multiplier=None,
                 reload_weights=True, batches_lr_update=1,
                 plot_filepath='./lrfind_plot.png',
                 accuracy_plot_filepath=None,
                 lr_filepath='./lrfind_suggestion.pkl',
                 n_lr_upgrade=None):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        self.plot_filepath = plot_filepath
        self.accuracy_plot_filepath = accuracy_plot_filepath
        self.lr_filepath = lr_filepath
        self.min_lr_suggested = min_lr
        self.max_lr_suggested = max_lr
        self.min_lr_suggested_idx = -1
        self.max_lr_suggested_idx = -1
        self.n_lr_upgrade = n_lr_upgrade
        self.lr_init = min_lr

        if stop_multiplier is None:
            self.stop_multiplier = -20 * self.mom / 3 + 10  # 4 if mom=0.9
            # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier

        self.min_to_max_factor = 1. / 10

    def on_train_begin(self, logs={}):
        if not self.n_lr_upgrade:
            p = self.params
            try:
                self.n_lr_upgrade = p['epochs'] * p['samples'] // p['batch_size']
            except:
                self.n_lr_upgrade = p['steps'] * p['epochs']

        n_in_zero_one = np.linspace(0, 1, num=self.n_lr_upgrade // self.batches_lr_update)
        self.learning_rates = self.min_lr * (self.max_lr / self.min_lr) ** n_in_zero_one

        self.losses = []
        self.mov_avg = 0
        self.mov_avg_acc = 0
        self.iteration = 0
        self.best_loss = 0
        self.accuracies = []
        if self.reload_weights:
            self.model.save_weights('tmp.hdf5')
        self.lr_init = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        acc = logs.get('accuracy')

        # Make loss smoother using momentum
        self.mov_avg_acc = self.mom * self.mov_avg_acc + (1 - self.mom) * acc
        acc = self.mov_avg_acc / (1 - self.mom ** (self.iteration + 1))

        self.mov_avg = self.mom * self.mov_avg + (1 - self.mom) * loss
        loss = self.mov_avg / (1 - self.mom ** (self.iteration + 1))

        if self.iteration == 0 or loss < self.best_loss:
            self.best_loss = loss

        if self.iteration % self.batches_lr_update == 0:  # Evaluate each lr over 5 epochs

            lr = self.learning_rates[self.iteration // self.batches_lr_update]
            K.set_value(self.model.optimizer.lr, lr)

            self.losses.append(loss)
            self.accuracies.append(acc)

        if loss > self.best_loss * self.stop_multiplier or self.iteration == self.n_lr_upgrade - 1:  # Stop criteria
            self.model.stop_training = True

        self.iteration += 1

    def on_train_end(self, logs=None):
        if self.reload_weights:
            self.model.load_weights('tmp.hdf5')
        K.set_value(self.model.optimizer.lr, self.lr_init)

        self.compute_range()

        self.plot_results()

    def compute_range(self, skip_begin=5, skip_end=1):

        n_losses = len(self.losses)
        if n_losses <= skip_begin + skip_end:
            skip_begin = 0
            skip_end = -n_losses
        lrs = self.learning_rates[:n_losses]
        lrs = lrs[skip_begin:-skip_end]
        losses = self.losses[skip_begin:-skip_end]
        min_loss_idx = np.argmin(losses)
        min_loss_lr_div_10 = lrs[min_loss_idx]/10
        self.max_lr_suggested_idx = (np.abs(lrs - min_loss_lr_div_10)).argmin()
        self.max_lr_suggested = lrs[self.max_lr_suggested_idx]
        try:
            # get the learning rate at the steepest descent (minimum gradient)
            # between the point where the loss starts to decrease and the point
            # of minimum loss
            loss_grad = np.gradient(np.array(losses))
            reverse_losses = np.flip(loss_grad[:min_loss_idx])
            start_descent_idx = min_loss_idx - np.argmax(reverse_losses >= 0)
            self.min_lr_suggested_idx = start_descent_idx + (loss_grad[start_descent_idx:min_loss_idx]).argmin()
            self.min_lr_suggested = lrs[self.min_lr_suggested_idx]
            assert(self.min_lr_suggested < self.max_lr_suggested)
            assert(self.min_lr_suggested/self.max_lr_suggested <= self.min_to_max_factor)
            print(f"Min numerical gradient: {self.min_lr_suggested:.2E}")
        except:
            print("Failed to compute the min LR as the one en the steepest descent of the loss.")
            self.min_lr_suggested = self.min_to_max_factor * self.max_lr_suggested
            self.min_lr_suggested_idx = (np.abs(lrs - self.min_lr_suggested)).argmin()
            self.min_lr_suggested = lrs[self.min_lr_suggested_idx]
            print(f"Setting the min LR to {self.min_to_max_factor:.2} the max LR: {self.min_lr_suggested:.2E}")

        print(f"Max LR corresponding to the min loss LR divided by 10: {self.max_lr_suggested:.2E}")

        lr_range = dict()
        lr_range['lr_low'] = self.min_lr_suggested
        lr_range['lr_high'] = self.max_lr_suggested

        with open(self.lr_filepath, 'wb') as fp:
            pickle.dump(lr_range, fp, protocol=pickle.HIGHEST_PROTOCOL)

        self.min_lr_suggested_idx += skip_begin
        self.max_lr_suggested_idx += skip_begin

        return

    def plot_results(self):

        plt.figure(figsize=(12, 6))
        lrs = self.learning_rates[:len(self.losses)]

        plt.plot(lrs, self.losses)
        plt.plot(self.min_lr_suggested, self.losses[self.min_lr_suggested_idx], markersize=10, marker='*', color='red')
        plt.plot(self.max_lr_suggested, self.losses[self.max_lr_suggested_idx], markersize=10, marker='*', color='red')
        plt.annotate(xy=(self.min_lr_suggested, self.losses[self.min_lr_suggested_idx]),
                     s=f'{self.min_lr_suggested:.2E}')
        plt.annotate(xy=(self.max_lr_suggested, self.losses[self.max_lr_suggested_idx]),
                     s=f'{self.max_lr_suggested:.2E}')
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')

        min_loss = np.min(self.losses)
        gap = self.losses[0] - min_loss
        plt.ylim(bottom=min_loss - 0.25 * gap, top=self.losses[0] + 0.25 * gap)

        p = self.params
        try:
            steps_per_epoch = p['samples'] // p['batch_size']
        except:
            steps_per_epoch = p['steps']

        upgrade_epoch_ratio = np.float(self.n_lr_upgrade) / steps_per_epoch
        plt.title(f"LR Range test loss on {upgrade_epoch_ratio:.3} epochs")

        plt.savefig(self.plot_filepath)
        plt.close()

        if self.accuracy_plot_filepath:
            plt.figure(figsize=(12, 6))
            plt.plot(lrs, self.accuracies)
            plt.plot(self.min_lr_suggested, self.accuracies[self.min_lr_suggested_idx], markersize=10, marker='*', color='red')
            plt.plot(self.max_lr_suggested, self.accuracies[self.max_lr_suggested_idx], markersize=10, marker='*', color='red')
            plt.annotate(xy=(self.min_lr_suggested, self.accuracies[self.min_lr_suggested_idx]),
                         s=f'{self.min_lr_suggested:.2E}')
            plt.annotate(xy=(self.max_lr_suggested, self.accuracies[self.max_lr_suggested_idx]),
                         s=f'{self.max_lr_suggested:.2E}')
            plt.xlabel("Learning Rate")
            plt.ylabel("Accuracy")
            plt.xscale('log')

            plt.title(f"LR Range test accuracy on {upgrade_epoch_ratio:.3} epochs")

            plt.savefig(self.accuracy_plot_filepath)
            plt.close()

