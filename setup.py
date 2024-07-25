from setuptools import setup, find_packages

setup(
    name='MIPLS2',
    version='0.7',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # List your package dependencies here
        # e.g., 'numpy', 'pandas',
    ],
    author='Ashkan Tashk',
    author_email='author@example.com',
    description='Machine learning utility based on PLS2 for imputing missing values',
    url='https://github.com/ashkantashk/MIPLS2',
)
