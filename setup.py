from setuptools import setup,find_packages

setup(
    name="vUtils",
    version="0.0.1",
    author="gaobiaoli",
    url="https://github.com/gaobiaoli/vUtils",
    author_email="gaobiaoli@tongji.edu.cn",
    packages=find_packages(),
    install_requires=[
        "numpy",         
        "opencv-python>=4.10", 
        "opencv-contrib-python>=4.10", 
    ],
)

