from setuptools import find_packages,setup
from typing import List

def get_requirements(File_path :str) ->List[str]:
    """ This function will return the list of the requirements package """
    requirements =[]

    with open(File_path) as file_obj:
        requirements =file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
    name='ML Model for Assignment',
    version = '0.0.1',
    author = 'Nishant G',
    author_email = '2026aj05092@wilp.bits-pilani.ac.in',
    install_requires= get_requirements("requirements.txt")
)
