from setuptools import find_packages, setup


setup(name='ctlearn_optimizer',
      version='1.0.0',
      description='Optimization framework for CTLearn',
      url='https://github.com/ctlearn-project/ctlearn',
      license='BSD-3-Clause',
      packages=['ctlearn_optimizer'],
      package_dir={'ctlearn_optimizer': 'src/ctlearn_optimizer'},
      include_package_data=True,
      dependencies=[],
      dependency_links=[],
      zip_safe=False)
