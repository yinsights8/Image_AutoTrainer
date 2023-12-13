from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def get_requirements(file_path:str)->List[str]:
    """
    This function returns a list of requrirements
    """
    requrirements = []

    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", " ") for req in requrirements]

        if HYPEN_E_DOT in requirements:
            requrirements.remove(HYPEN_E_DOT)

    return requrirements

setup(
name="Image_AutoTrainer",
version='1.0.0',
author="Yash Dhakade",
author_email = "yinsights8@gmail.com",
packages=find_packages(),
include_package_data=True,
license='MIT',
  description = "It's an auto image classification library",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/entbappy/ImageSeeker-Package",
  author = 'Yash Dhakade',
  author_email = 'yinsights8@gmail.com',
  keywords = ['autotrainer'],
  install_requires=get_requirements("requirements.txt"),
  classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',

  ],
  entry_points={
        "console_scripts": [
            "autotrainer = Image_AutoTrainer.app:start_app",
        ]},
)