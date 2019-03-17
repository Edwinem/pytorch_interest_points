from os.path import dirname, basename, isfile
import glob


modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

folder_name=basename(dirname(__file__))

for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        __import__(folder_name+'.'+basename(f)[:-3], globals(),locals())

del modules
del folder_name