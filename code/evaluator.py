# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/12/19 3:13 PM

from utils import convert_to_dataset_friendly_scores, get_logger
from metrics import *
import csv
import pathlib
import subprocess

logger = get_logger("Evaluate stats")


class Evaluator():
    def __init__(
            self,
            prompt_id,
            out_dir,
            modelname,
            train_x,
            dev_x,
            test_x,
            train_y,
            dev_y,
            test_y,
            ts,
            mode,
            fold,
            batch_size,
            args
    ):

        self.prompt_id = prompt_id
        self.train_x, self.dev_x, self.test_x = train_x, dev_x, test_x
        self.train_y, self.dev_y, self.test_y = train_y, dev_y, test_y
        self.train_y_org = convert_to_dataset_friendly_scores(train_y, self.prompt_id)
        self.dev_y_org = convert_to_dataset_friendly_scores(dev_y, self.prompt_id)
        self.test_y_org = convert_to_dataset_friendly_scores(test_y, self.prompt_id)
        self.out_dir = out_dir
        self.modelname = modelname
        self.ts = ts
        self.mode = mode
        self.fold = fold
        self.batch_size = batch_size
        self.args = args
        self.best_dev = -1
        self.best_test = -1
        self.best_loss = 10000
        self.commit_short = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])

    def calc_kappa(self, dev_pred, test_pred, weight='quadratic'):
        # self.train_qwk = kappa(self.train_y_org, train_pred, weight)
        self.dev_qwk = kappa(self.dev_y_org, dev_pred, weight)
        self.test_qwk = kappa(self.test_y_org, test_pred, weight)

    def evaluate(self, model, epoch, print_info=False):
        # train_pred = model.predict(self.train_x, batch_size=self.batch_size, verbose=False).squeeze()
        dev_pred = model.predict(self.dev_x, batch_size=self.batch_size, verbose=False).squeeze()
        test_pred = model.predict(self.test_x, batch_size=self.batch_size, verbose=False).squeeze()

        # train_pred_int = convert_to_dataset_friendly_scores(train_pred, self.prompt_id)
        dev_pred_int = convert_to_dataset_friendly_scores(dev_pred, self.prompt_id)
        test_pred_int = convert_to_dataset_friendly_scores(test_pred, self.prompt_id)

        self.calc_kappa(dev_pred_int, test_pred_int)

        if self.dev_qwk > self.best_dev:
            # self.best_train = self.train_qwk
            self.best_dev = self.dev_qwk
            self.best_test = self.test_qwk
            self.best_dev_epoch = epoch
            model.save_weights(self.out_dir + '/' + self.modelname + '.hdf5', overwrite=True)
            with open(self.out_dir + '/' + self.modelname + '_pred_test_score.txt', 'w') as file:
                for x in test_pred_int:
                    file.write(str(x) + "\n")

        if print_info:
            self.print_info()

    def evaluate_by_loss(self, model, epoch, current_loss=9999, print_info=False):
        # train_pred = model.predict(self.train_x, batch_size=self.batch_size, verbose=False).squeeze()
        dev_pred = model.predict(self.dev_x, batch_size=self.batch_size, verbose=False).squeeze()
        test_pred = model.predict(self.test_x, batch_size=self.batch_size, verbose=False).squeeze()

        # train_pred_int = convert_to_dataset_friendly_scores(train_pred, self.prompt_id)
        dev_pred_int = convert_to_dataset_friendly_scores(dev_pred, self.prompt_id)
        test_pred_int = convert_to_dataset_friendly_scores(test_pred, self.prompt_id)

        self.calc_kappa(dev_pred_int, test_pred_int)

        # self.dev_qwk = current_loss
        # self.test_qwk = kappa(self.test_y_org, test_pred, 'quadratic')

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_dev = current_loss
            self.best_test = self.test_qwk
            self.best_dev_epoch = epoch
            model.save_weights(self.out_dir + '/' + self.modelname + '.hdf5', overwrite=True)
            with open(self.out_dir + '/' + self.modelname + '_pred_test_score.txt', 'w') as file:
                for x in test_pred_int:
                    file.write(str(x) + "\n")

        if print_info:
            self.print_info()

    def print_info(self):
        # logger.info('[TRAIN] QWK:  %.3f (Best @ %i: {{%.3f}})' % (self.train_qwk, self.best_dev_epoch, self.best_train))
        logger.info('[DEV]   QWK:  %.3f (Best @ %i: {{%.3f}})' % (self.dev_qwk, self.best_dev_epoch, self.best_dev))
        logger.info('[TEST]  QWK:  %.3f (Best @ %i: {{%.3f}})' % (self.test_qwk, self.best_dev_epoch, self.best_test))
        logger.info('-------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        self.record()
        logger.info('-------------------------------------------------------------------------------------------------')
        logger.info('Best @ Epoch %i:' % self.best_dev_epoch)
        logger.info('  [DEV]  QWK: %.3f' % self.best_dev)
        logger.info('  [TEST] QWK: %.3f' % self.best_test)

    def record(self):
        path = pathlib.Path(self.out_dir + '/' + self.mode + '_' + str(self.fold) + '_record.csv')
        if not path.exists():
            row = [
                'ts',
                'mode',
                'concatenation_mode',
                'features',
                'prompt',
                'fold',
                'batch',
                'best_dev_epoch',
                'best_dev',
                'best_test',
                'modelname',
                'commit_short',
                'commit_hash',
                'argstr'
            ]
            with open(path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            f.close()

        row = [
            self.ts,
            self.mode,
            self.args.concatenation_mode,
            self.args.feature,
            self.prompt_id,
            self.fold,
            self.batch_size,
            self.best_dev_epoch,
            self.best_dev,
            self.best_test,
            self.modelname,
            self.commit_short,
            self.commit_hash,
            str(self.args)
        ]
        with open(path, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

