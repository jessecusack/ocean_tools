from distutils.core import setup

setup(name='ocean_tools',
      version='0.1.0',
      description='Miscillaneous python tools for oceanographic analysis',
      long_description=open('README.rst').read(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Oceanography :: Data analysis',
      ],
      url='http://github.com/jessecusack/ocean_tools',
      author='Jesse Cusack',
      author_email='jesse.cusack@noc.soton.ac.uk',
      license='MIT',
      packages=['ocean_tools'],
      install_requires=[
          'numpy', 'scipy', 'gsw', 'seawater',
      ],
      python_requires='>=2.7',
      zip_safe=False)
