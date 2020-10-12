# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:29:02 2020

@author: William Bankes
"""

from setuptools import setup


setup(
      name='drlAgents',
      url='https://github.com/williambankes/deep_reinforcement_learning',
      author='William Bankes',
      author_email='williamjamesbankes@gmail.com',
      packages=['drlAgents'],
      install_requires=['numpy', 'torch'],
      description='An implementation of Deep Reinforcement Learning Algorithms'     
      )