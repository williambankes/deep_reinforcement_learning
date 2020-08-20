# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 21:29:02 2020

@author: William Bankes
"""

from setuptools import setup


setup(
      name='drl_agents',
      url='https://github.com/williambankes/deep_reinforcement_learning',
      author='William Bankes',
      author_email='williamjamesbankes@gmail.com',
      # Needed to actually package something
      packages=['drl_agents'],
      # Needed for dependencies
      install_requires=['numpy', 'torch'],
      # *strongly* suggested for sharing
      version='0.1',
      # The license can be anything you like
      license='MIT',
      description='An example of a python package from pre-existing code'     
      )