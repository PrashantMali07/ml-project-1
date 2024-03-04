from setuptools import find_packages, setup
from typing import List

hyphen_instance = '-e .'

def install_requirements(file_path:str)->List[str]:
    '''
    This function can return the name modules required for this project
    which is written inside the requirements.txt file.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[mod.replace("\n", " ") for mod in requirements]

        if hyphen_instance in requirements:
            requirements.remove(hyphen_instance)
    
    return requirements

setup(
    name='mlproject1', 
    version='1.0.01',
    author='prashant',
    author_email='er.prashantkmrmali@gmail.com',
    packages=find_packages(),
    install_requires = install_requirements('requirements.txt')
    )