#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:29:58 2019

@author: pi
"""
from utils import create, new_face, recog

print("Input new face?")
if(input() is 'y'):
    new_face()
data = create()
recog(data)
