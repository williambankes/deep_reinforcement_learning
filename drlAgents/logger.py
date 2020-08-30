# -*- coding: utf-8 -*-
"""
Create a logging class to collate information and

- Add comet_ml integration to pass experiement across modules

Created on Wed Aug 19 20:19:15 2020

@author: William Bankes

"""


def rescale_stepwise(data, duration):
    
    current_index = 0
    rescaled_data = list()
    
    for dur in duration:
        
        rescaled_data.append(np.mean(data[current_index:dur + current_index]))
        
        current_index += dur
        
    return rescaled_data


class Logger:
    """
    A Singleton class that can be called throughout the package to record
    variables and other metrics during run time. A comet_ml experiement object
    can be passed to it and (hopefully) will collect the same metrics during
    run time and pass them to the comet_ml dashboard...   
    
    n.b. doesn't need to be instantiated
    
    Example:
    >>>>Logger.getInstance().add('name_of_var', 10)
    >>>>Logger.getInstance().get('name_of_var')
    [10]
    """
    
    
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
          self.__comet_exp = None
          self.__comet = False
          self.__data = dict()    
          Logger.__instance = self


          
    def check_name(self, name):
        return name in self.__data.keys()

    
    
    def add_comet_experiement(self, exp):
        self.__comet_exp = exp
        self.__comet = True
        
    
    
    def clear_comet_experiement(self, exp):
        self.__comet_exp = None
        self.__comet = False      
        
        
     
    def add(self, name, value):
        if self.check_name(name):
            self.__data[name].append(value)
        else:
            self.__data[name] = [value]
            
        if self.__comet:
            self._comet_exp.log_metric(name, value)


        
    def get(self, name):
        if self.check_name(name):
            return self.__data[name]
        else:
            raise Exception("{} does not exist".format(name))
    

        
    def get_names(self):
        return self.__data.keys()
    
    
    
    def clear(self, name=None):
        if name is not None:    
            if name in self.__data.keys():    
                self.__data[name] = list()
            else:
                raise Exception("{} does not exist".format(name))
        else:
            self.__data = dict()
