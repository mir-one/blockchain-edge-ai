#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_users', type=int, default=100,
                        help="количество пользователей: К")
    parser.add_argument('--global_round', type=int, default=10,
                        help="номер раунда")
    parser.add_argument('--idx', type=int, default=1,
                        help="идентификатор обучающего клиента, idx")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="количество эпох: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="размер пакета обновлений: B")

    parser.add_argument('--model', type=str, default='mlp', help='имя модели')

    parser.add_argument('--iid', type=int, default=1,
                        help='Распределение данных среди пользователей. По умолчанию установлен IID. Установите 0 при отсутствии IID')
    parser.add_argument('--unequal', type=int, default=0,
                        help='Используется в настройках, отличных от iid. Возможность разделить данные между пользователями поровну или неравномерно. По умолчанию установлено значение 0 для равных долей. Установите 1 для неравных долей')

    args = parser.parse_args()
    return args
