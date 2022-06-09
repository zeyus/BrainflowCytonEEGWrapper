from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='BrainflowCyton',
    version='0.1.1',    
    description='Python wrapper for BrainFlow API + Cyton SDK for collecting realtime or offline (SDCard) EEG, EMG, EKG data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/zeyus/BrainflowCytonEEGWrapper',
    project_urls={
        'Bug Tracker': 'https://github.com/zeyus/BrainflowCytonEEGWrapper/issues',
    },
    author='zeyus',
    author_email='dev@zeyus.com',
    license='MIT',
    packages=['BrainflowCyton'],
    install_requires=['pandas',
                      'numpy',
                      'nptyping',
                      'scipy',
                      'sounddevice',
                      'brainflow',
                      'samplerate',
                      'pyxdf',                     
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux', 
        'Natural Language :: English',      
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: System :: Hardware',
        'Topic :: Utilities',
    ],
)