from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> list[str]:
    with open(file_path, 'r') as f:
        f.readlines()
        requirements = [req.replace('\n', '') for req in f.readlines()]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(name='loan_default_prediction',
      version='0.0.2',
      author='Amit',
      author_email='amitkumars342@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt')
)