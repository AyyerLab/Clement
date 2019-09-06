import setuptools

with open('README.md', 'r') as fptr:
    long_desc = fptr.read()

setuptools.setup(
    name='Clement',
    version='0.1',
    author='CNI Group at MPSD',
    author_email='kartik.ayyer@mpsd.mpg.de',
    description='GUI for Correlative Light and Electron Microscopy',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/kartikayyer/Clement',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'pyqt5',
        'numexpr',
        'mrcfile',
        'read-lif',
        'pyqtgraph @ git+https://github.com/pyqtgraph/pyqtgraph.git@develop',        
    ],
    package_data={'Clement': ['styles/*.qss']},
    entry_points={'gui_scripts': [
        'clement = clement.gui:main',
        ],
    },
)
