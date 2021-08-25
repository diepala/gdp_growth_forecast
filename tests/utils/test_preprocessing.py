#!/usr/bin/env python

import sys
sys.path.append("..") 
import pandas as pd
import pytest
import utils.preprocessing


def test_remove_na_cols():
    df_test = pd.DataFrame([{'Year': 1960, 'a': 1, 'b':2},{'Year': 1961, 'a': 3, 'b':4},{'Year': 1962, 'a': 5}])
    df_correct_test = pd.DataFrame([{'Year': 1960, 'a': 1},{'Year': 1961, 'a': 3},{'Year': 1962, 'a': 5}])
    assert utils.preprocessing.remove_na_cols(df_test,0.4,1961).equals(df_correct_test)
    
def test_drop_high_corr():
    df_test = pd.DataFrame([{'a': 1, 'b': 2, 'c':3}, {'a':10, 'b': 20, 'c': -3}, {'a': 1, 'b': 2, 'c':-1000}]);
    df_correct_test = pd.DataFrame([{'a':1,'c':3},{'a':10,'c':-3},{'a':1,'c':-1000}]);
    assert utils.preprocessing.drop_high_corr(df_test,0.5).equals(df_correct_test)