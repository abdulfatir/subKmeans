from __future__ import print_function


def log_time(message, time, tabs=1):
    message = (tabs * '\t') + '[t] ' + message
    print('%s: %.4fs' % (message, time))
