# -*- coding: utf-8 -*-
"""
Create a logging class to collate information and

Created on Wed Aug 19 20:19:15 2020

@author: William Bankes

"""
#%%

class Logger:
    
    __instance = None
       
    @staticmethod 
    def getInstance():
      """ Static access method. """
      if Logger.__instance == None:
          Logger()
      return Logger.__instance
      
    
    def __init__(self):
      """ Virtually private constructor. """
      
      if Logger.__instance != None:
          raise Exception("This class is a singleton!")
      else:    
          self.__data = dict()    
          Logger.__instance = self
     
    def add(self, name, value):
        
        if name in self.__data.keys():
            self.__data[name].append(value)

        else:
            self.__data[name] = [value]
        
        
    def get(self, name):
        if name in self.__data.keys():
            return self.__data[name]
        else:
            raise Exception("{} does not exist".format(name))
        

if __name__ == '__main__':
    for i in range(10):
        Logger.getInstance().add('testing', i)
        
    Logger.getInstance().get('testing')