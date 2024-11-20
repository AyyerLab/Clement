import setuptools

with open('README.md', 'r') as fptr:
    long_desc = fptr.read()

setuptools.setup(
    name='clement',
    version='0.3.0',
    author='CNI Group at MPSD',
    author_email='kartik.ayyer@mpsd.mpg.de',
    description='GUI for Correlative Light and Electron Microscopy',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/AyyerLab/Clement',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Visualization',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy>=1.0.0',
        'scikit-image',
        'scikit-learn',
        'pyqt5',
        'numexpr',
        'mrcfile',
        'read-lif==0.4.0',
        'tifffile',
        'matplotlib',
        'pyqtgraph',
        'pyyaml',
        'xmltodict',
    ],
    package_data={'clement': ['styles/*.qss']},
    entry_points={'gui_scripts': [
        'clement = clement.gui:main',
        ],
    },
)
