from setuptools import find_packages, setup 
from typing import List # type: ignore
hyphen='-e .'
def get_requirements(file_path:str)->List[str]:
    reqs=[]
    with open(file_path) as file:
        reqs=file.readlines()
        reqs=[n.replace("\n","") for n in reqs]
        if hyphen in reqs:
            reqs.remove(hyphen)
    return reqs
setup(
    name= 'MLPROJECT',
    author= 'srivanth',
    version='0.0.1',
    author_email='srivanthsail@gmail.com',
    packages=find_packages(),
    requires=get_requirements('requirements.txt')
)