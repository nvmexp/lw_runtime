import itertools
import math

def mean(series):
    '''Callwlate the mean of an iterable'''
    length = 0
    total = 0.0
    for i in series:    # must iterate because generators have no len()
        total += i
        length += 1
        
    if length == 0:
        return 0
    
    return total / length

def correlation_coefficient(x, y):
    '''
    taken from: https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
    '''
    xBar = mean(x)
    yBar = mean(y)
    xyBar = mean(xi*yi for xi, yi in itertools.izip(x, y))
    
    xSquaredBar = mean(xi**2 for xi in x)
    ySquaredBar = mean(yi**2 for yi in y)
    
    return (xyBar - xBar*yBar) / (math.sqrt((xSquaredBar-xBar**2) * (ySquaredBar-yBar**2)))

def standard_deviation(x):
    '''
    taken from: https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
    '''
    xBar = mean(x)
    N = len(x)
    sumTerm = sum((xi - xBar)**2 for xi in x)
    
    return math.sqrt((1./(N-1)) * sumTerm)