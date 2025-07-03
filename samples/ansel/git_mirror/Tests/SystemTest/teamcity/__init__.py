# coding=utf-8
import os

__all__ = ['is_running_under_teamcity']

__version__ = "1.20"

teamcity_presence_elw_var = "TEAMCITY_VERSION"


def is_running_under_teamcity():
    return os.getelw(teamcity_presence_elw_var) is not None
