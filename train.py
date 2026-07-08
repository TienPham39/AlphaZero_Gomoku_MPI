# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:31:39 2018

@author: initial-h
"""


from __future__ import print_function
import random
import csv
import numpy as np
import os
import time
from collections import defaultdict, deque
from game_board import Board,Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorlayer import PolicyValueNet

class TrainPipeline():
    metric_columns = [
        'batch_index',
        'train_step',
        'train_steps',
        'mini_batch_size',
        'loss',
        'entropy',
        'kl',
        'explained_var_old',
        'explained_var_new',
        'episode_len',
        'data_buffer_size'
    ]
    batch_metric_columns = [
        'batch_index',
        'episode_len',
        'augmented_samples',
        'data_buffer_size',
        'total_simulations_run',
        'avg_simulations_per_move',
        'avg_simulations_per_second',
        'did_train',
        'elapsed_hours',
        'collect_data_time_hours',
        'train_data_time_hours',
        'evaluate_time_hours',
        'did_eval',
        'pure_mcts_playout_num',
        'eval_games',
        'win_count',
        'lose_count',
        'tie_count',
        'win_ratio',
        'best_win_ratio',
        'is_new_best',
        'model_saved'
    ]

    def __init__(self, init_model=None,transfer_model=None):
        self.resnet_block = 19  # num of block structures in resnet
        # params of the board and the game
        self.board_width = 11
        self.board_height = 11
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 1e-3
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 500000 # memory size
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1 # play n games for each network training
        self.check_freq = 50
        self.game_batch_num = 100000 # total game to train
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 200
        self.episode_len = 0
        self.metrics_path = os.path.join('log', 'train_metrics.csv')
        self.batch_metrics_path = os.path.join('log', 'batch_metrics.csv')
        self.last_train_metrics = {}
        if (init_model is not None) and os.path.exists(init_model+'.index'):
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,block=self.resnet_block,init_model=init_model,cuda=True)
        elif (transfer_model is not None) and os.path.exists(transfer_model+'.index'):
            # start training from a pre-trained policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,block=self.resnet_block,transfer_model=transfer_model,cuda=True)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,block=self.resnet_block,cuda=True)

        self.mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                       action_fc=self.policy_value_net.action_fc_test,
                                       evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                       c_puct=self.c_puct,
                                       n_playout=self.n_playout,
                                       is_selfplay=True)

    def format_metric_float(self, value):
        return '{:.6f}'.format(float(value))

    def append_csv_row(self, csv_path, columns, row):
        log_dir = os.path.dirname(csv_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        if file_exists and not self.csv_header_matches(csv_path, columns):
            backup_path = self.next_backup_path(csv_path)
            os.rename(csv_path, backup_path)
            print('metrics header changed, moved old log to {}'.format(backup_path))
            file_exists = False

        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def csv_header_matches(self, csv_path, columns):
        with open(csv_path, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, [])
        return header == list(columns)

    def next_backup_path(self, csv_path):
        backup_path = csv_path + '.bak'
        if not os.path.exists(backup_path):
            return backup_path

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_path = '{}.{}.bak'.format(csv_path, timestamp)
        counter = 1
        while os.path.exists(backup_path):
            backup_path = '{}.{}.{}.bak'.format(csv_path, timestamp, counter)
            counter += 1
        return backup_path

    def log_train_metrics(self, batch_index, train_step, train_steps,
                          mini_batch_size, loss, entropy, kl,
                          explained_var_old, explained_var_new):
        try:
            row = {
                'batch_index': int(batch_index),
                'train_step': int(train_step),
                'train_steps': int(train_steps),
                'mini_batch_size': int(mini_batch_size),
                'loss': self.format_metric_float(loss),
                'entropy': self.format_metric_float(entropy),
                'kl': self.format_metric_float(kl),
                'explained_var_old': self.format_metric_float(explained_var_old),
                'explained_var_new': self.format_metric_float(explained_var_new),
                'episode_len': int(self.episode_len),
                'data_buffer_size': int(len(self.data_buffer))
            }
            self.append_csv_row(self.metrics_path, self.metric_columns, row)
        except Exception as exc:
            print('write train metrics failed: {}'.format(exc))

    def optional_metric_int(self, value):
        if value is None or value == '':
            return ''
        return int(value)

    def optional_metric_float(self, value):
        if value is None or value == '':
            return ''
        return self.format_metric_float(value)

    def metric_flag(self, value):
        return int(bool(value))

    def new_batch_metrics(self, batch_index):
        return {
            'batch_index': int(batch_index),
            'did_train': 0,
            'did_eval': 0,
            'is_new_best': 0,
            'model_saved': 0
        }

    def log_batch_metrics(self, metrics):
        try:
            row = {
                'batch_index': int(metrics.get('batch_index')),
                'episode_len': self.optional_metric_int(metrics.get('episode_len', '')),
                'augmented_samples': self.optional_metric_int(metrics.get('augmented_samples', '')),
                'data_buffer_size': self.optional_metric_int(metrics.get('data_buffer_size', '')),
                'total_simulations_run': self.optional_metric_int(metrics.get('total_simulations_run', '')),
                'avg_simulations_per_move': self.optional_metric_int(metrics.get('avg_simulations_per_move', '')),
                'avg_simulations_per_second': self.optional_metric_int(metrics.get('avg_simulations_per_second', '')),
                'did_train': self.metric_flag(metrics.get('did_train', 0)),
                'elapsed_hours': self.optional_metric_float(metrics.get('elapsed_hours', '')),
                'collect_data_time_hours': self.optional_metric_float(metrics.get('collect_data_time_hours', '')),
                'train_data_time_hours': self.optional_metric_float(metrics.get('train_data_time_hours', '')),
                'evaluate_time_hours': self.optional_metric_float(metrics.get('evaluate_time_hours', '')),
                'did_eval': self.metric_flag(metrics.get('did_eval', 0)),
                'pure_mcts_playout_num': self.optional_metric_int(metrics.get('pure_mcts_playout_num', '')),
                'eval_games': self.optional_metric_int(metrics.get('eval_games', '')),
                'win_count': self.optional_metric_int(metrics.get('win_count', '')),
                'lose_count': self.optional_metric_int(metrics.get('lose_count', '')),
                'tie_count': self.optional_metric_int(metrics.get('tie_count', '')),
                'win_ratio': self.optional_metric_float(metrics.get('win_ratio', '')),
                'best_win_ratio': self.optional_metric_float(metrics.get('best_win_ratio', '')),
                'is_new_best': self.metric_flag(metrics.get('is_new_best', 0)),
                'model_saved': self.metric_flag(metrics.get('model_saved', 0))
            }
            self.append_csv_row(self.batch_metrics_path,
                                self.batch_metric_columns,
                                row)
        except Exception as exc:
            print('write batch metrics failed: {}'.format(exc))

    def get_equi_data(self, play_data):
        '''
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        '''
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                #rotate counterclockwise 90*i
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                #np.flipud like A[::-1,...]
                #https://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.flipud.html
                # change the reshaped numpy
                # 0,1,2,
                # 3,4,5,
                # 6,7,8,
                # as
                # 6 7 8
                # 3 4 5
                # 0 1 2
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                #这个np.fliplr like m[:, ::-1]
                #https://docs.scipy.org/doc/numpy/reference/generated/numpy.fliplr.html
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        '''
        collect self-play data for training
        '''
        total_episode_len = 0
        total_augmented_samples = 0
        total_simulations_run = 0
        total_self_play_seconds = 0.0

        for i in range(n_games):
            self_play_start_time = time.time()
            _winner, play_data = self.game.start_self_play(self.mcts_player,is_shown=False)
            self_play_elapsed_seconds = time.time() - self_play_start_time
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            simulations_per_move = getattr(self.mcts_player.mcts,
                                           '_n_playout',
                                           self.n_playout)
            episode_simulations_run = int(self.episode_len * simulations_per_move)
            # augment the data
            augmented_play_data = self.get_equi_data(play_data)
            augmented_samples = len(augmented_play_data)
            self.data_buffer.extend(augmented_play_data)

            total_episode_len += self.episode_len
            total_augmented_samples += augmented_samples
            total_simulations_run += episode_simulations_run
            total_self_play_seconds += self_play_elapsed_seconds

        avg_simulations_per_move = 0
        if total_episode_len > 0:
            avg_simulations_per_move = int(total_simulations_run / total_episode_len)

        avg_simulations_per_second = 0
        if total_self_play_seconds > 0:
            avg_simulations_per_second = int(total_simulations_run / total_self_play_seconds)

        return {
            'episode_len': int(total_episode_len),
            'augmented_samples': int(total_augmented_samples),
            'data_buffer_size': int(len(self.data_buffer)),
            'total_simulations_run': int(total_simulations_run),
            'avg_simulations_per_move': int(avg_simulations_per_move),
            'avg_simulations_per_second': int(avg_simulations_per_second)
        }

    def policy_update(self, batch_index=None):
        '''
        update the policy-value net
        '''
        # play_data: [(state, mcts_prob, winner_z), ..., ...]
        # train an epoch

        tmp_buffer = np.array(self.data_buffer)
        np.random.shuffle(tmp_buffer)
        steps = len(tmp_buffer)//self.batch_size
        print('tmp buffer: {}, steps: {}'.format(len(tmp_buffer),steps))
        for i in range(steps):
            mini_batch = tmp_buffer[i*self.batch_size:(i+1)*self.batch_size]
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]

            old_probs, old_v = self.policy_value_net.policy_value(state_batch=state_batch,
                                                                  actin_fc=self.policy_value_net.action_fc_test,
                                                                  evaluation_fc=self.policy_value_net.evaluation_fc2_test)
            loss, entropy = self.policy_value_net.train_step(state_batch,
                                                             mcts_probs_batch,
                                                             winner_batch,
                                                             self.learn_rate)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch=state_batch,
                                                                  actin_fc=self.policy_value_net.action_fc_test,
                                                                  evaluation_fc=self.policy_value_net.evaluation_fc2_test)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )

            explained_var_old = (1 -
                                 np.var(np.array(winner_batch) - old_v.flatten()) /
                                 np.var(np.array(winner_batch)))
            explained_var_new = (1 -
                                 np.var(np.array(winner_batch) - new_v.flatten()) /
                                 np.var(np.array(winner_batch)))
            self.last_train_metrics = {
                'loss': loss,
                'entropy': entropy,
                'kl': kl,
                'explained_var_old': explained_var_old,
                'explained_var_new': explained_var_new
            }
            if batch_index is not None:
                self.log_train_metrics(batch_index=batch_index,
                                       train_step=i,
                                       train_steps=steps,
                                       mini_batch_size=len(mini_batch),
                                       loss=loss,
                                       entropy=entropy,
                                       kl=kl,
                                       explained_var_old=explained_var_old,
                                       explained_var_new=explained_var_new)

            if steps<10 or (i%(steps//10)==0):
                # print some information, not too much
                print('batch: {},length: {}'
                      'kl:{:.5f},'
                      'loss:{},'
                      'entropy:{},'
                      'explained_var_old:{:.3f},'
                      'explained_var_new:{:.3f}'.format(i,
                                                        len(mini_batch),
                                                        kl,
                                                        loss,
                                                        entropy,
                                                        explained_var_old,
                                                        explained_var_new))

        return loss, entropy

    def policy_evaluate(self, n_games=10):
        '''
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        '''
        current_mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                       action_fc=self.policy_value_net.action_fc_test,
                                       evaluation_fc=self.policy_value_net.evaluation_fc2_test,
                                       c_puct=5,
                                       n_playout=400,
                                       is_selfplay=False)

        test_player = MCTS_Pure(c_puct=5,
                                n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(player1=current_mcts_player,
                                          player2=test_player,
                                          start_player=i % 2,
                                          is_shown=0,
                                          print_prob=False)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return {
            'eval_games': int(n_games),
            'win_count': int(win_cnt[1]),
            'lose_count': int(win_cnt[2]),
            'tie_count': int(win_cnt[-1]),
            'win_ratio': win_ratio
        }

    def run(self):
        '''
        run the training pipeline
        '''
        # make dirs first
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        if not os.path.exists('model'):
            os.makedirs('model')

        # record time for each part
        start_time = time.time()
        collect_data_time = 0
        train_data_time = 0
        evaluate_time = 0

        try:
            for i in range(self.game_batch_num):
                batch_index = i + 1
                batch_metrics = self.new_batch_metrics(batch_index)

                # collect self-play data
                collect_data_start_time = time.time()
                batch_metrics.update(self.collect_selfplay_data(self.play_batch_size))
                collect_data_time += time.time()-collect_data_start_time
                print("batch i:{}, episode_len:{}".format(
                        batch_index, self.episode_len))

                if len(self.data_buffer) > self.batch_size*5:
                    # train collected data
                    train_data_start_time = time.time()
                    loss, entropy = self.policy_update(batch_index=batch_index)
                    train_data_time += time.time()-train_data_start_time
                    batch_metrics['did_train'] = 1

                    # print some training information
                    print('now time : {}'.format((time.time() - start_time) / 3600))
                    print('collect_data_time : {}, train_data_time : {},evaluate_time : {}'.format(
                        collect_data_time / 3600, train_data_time / 3600,evaluate_time/3600))

                if batch_index % self.check_freq == 0 :

                    # save current model for evaluating
                    self.policy_value_net.save_model('tmp/current_policy.model')
                    if batch_index % (self.check_freq*2) == 0:
                        print("current self-play batch: {}".format(batch_index))
                        evaluate_start_time = time.time()

                        # evaluate current model
                        eval_games = 10
                        evaluate_playout_num = self.pure_mcts_playout_num
                        eval_metrics = self.policy_evaluate(n_games=eval_games)
                        evaluate_time += time.time()-evaluate_start_time
                        win_ratio = eval_metrics['win_ratio']
                        is_new_best = win_ratio > self.best_win_ratio
                        model_saved = False
                        best_win_ratio_for_log = self.best_win_ratio
                        if win_ratio > self.best_win_ratio:
                            # save best model
                            print("New best policy!!!!!!!!")
                            self.best_win_ratio = win_ratio
                            best_win_ratio_for_log = self.best_win_ratio
                            self.policy_value_net.save_model('model/best_policy.model')
                            model_saved = True

                        batch_metrics.update({
                            'did_eval': 1,
                            'pure_mcts_playout_num': evaluate_playout_num,
                            'eval_games': eval_metrics['eval_games'],
                            'win_count': eval_metrics['win_count'],
                            'lose_count': eval_metrics['lose_count'],
                            'tie_count': eval_metrics['tie_count'],
                            'win_ratio': win_ratio,
                            'best_win_ratio': best_win_ratio_for_log,
                            'is_new_best': is_new_best,
                            'model_saved': model_saved
                        })

                        if is_new_best:
                            if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                                # increase playout num and  reset the win ratio
                                self.pure_mcts_playout_num += 100
                                self.best_win_ratio = 0.0
                            if self.pure_mcts_playout_num ==5000:
                                # reset mcts pure playout num
                                self.pure_mcts_playout_num = 1000
                                self.best_win_ratio = 0.0

                batch_metrics.update({
                    'elapsed_hours': (time.time() - start_time) / 3600,
                    'collect_data_time_hours': collect_data_time / 3600,
                    'train_data_time_hours': train_data_time / 3600,
                    'evaluate_time_hours': evaluate_time / 3600
                })
                self.log_batch_metrics(batch_metrics)

        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model='model/best_policy.model',transfer_model=None)
    # training_pipeline = TrainPipeline(init_model=None, transfer_model='transfer_model/best_policy.model')
    # training_pipeline = TrainPipeline()
    training_pipeline.run()
