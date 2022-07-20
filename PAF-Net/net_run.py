# -*- coding: utf-8 -*-
# from __future__ import print_function, division
import sys
from util.parse_config import parse_config
from net_run.agent_cls import ClassificationAgent
from agent_seg import SegmentationAgent


def main():
    stage = 'train'
    cfg_file ='./dataset/config/train_test.cfg'
    config = parse_config(cfg_file)
    task = config['dataset']['task_type']
    assert task in ['cls', 'cls_nexcl', 'seg']
    if task == 'cls' or task == 'cls_nexcl':
         agent = ClassificationAgent(config, stage)
    else:
        agent = SegmentationAgent(config, stage)
    agent.run()
             

if __name__ == "__main__":
    main()
    