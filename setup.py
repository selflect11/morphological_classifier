try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description' : 'Morphological classifier for the Portuguese language',
    'author' : 'Pedro Volpi Nacif',
    'url' : '',
    'download_url': '',
    'author_email' : 'pedrovolpi@poli.ufrj.br',
    'verion' : '0.1',
    'install_requires' : ['pip'],
    'packages' : ['morphological_classifier'],
    'scripts' : [],
    'name' : 'Morphological Classifier',
    'license' : 'MIT',
    'test_suite' : 'morphological_classifier.tests',
}

setup(**config)
