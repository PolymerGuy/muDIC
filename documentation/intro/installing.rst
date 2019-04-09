Getting started with µDIC
=======================================
In order to get started with µDIC, you need to install it on your computer.
There are two main ways to to this:

*   You can install it via a package manager like PIP or Conda
*   You can  clone the repo


Installing via a package manager:
----------------------------------
Prerequisites:
    This toolkit is only tested on Python 2.7x but will be ported to 3.7 later

On the command line, check if python is available::

$ python --version


If this command does not return the verstion of you python installation, 
you need to fix this first.

If everything seems to work, you install the package in your global python 
environment (Not recommend) via pip::

$ pip install muDIC

and you are good to go!

We recommend that you always use virtual environments, either by virtualenv or by Conda env.

Virtual env::

$ cd /path/to/your/project
$ virtualenv myproject
$ source /myproject/bin/activate
$ pip install muDIC



Conda env::

$ cd /path/to/your/project
$ conda create -n envname python=2.7
$ source activate envname
$ conda install muDIC





By cloning the repo:
-------------------

These instructions will get you a copy of the project up and running on your 
local machine for development and testing purposes.

Prerequisites:
    This toolkit is only tested on Python 2.7x but will be ported to 3.7 later

Installing:
    Start to clone this repo to your preferred location::

    $ git init
    $ git clone https://github.com/PolymerGuy/myDIC.git



    We recommend that you always use virtual environments, either by virtualenv or by Conda env

    Virtual env::
    
    $ virtualenv myproject
    $ source /myproject/bin/activate
    $ pip install -r requirements.txt
    

    Conda env::
    
    $ conda create -n envname python=2.7
    $ source activate envname
    $ conda install --yes --file requirements.txt
    

    You can now run an example::

    $ python path_to_myDIC/Examples/phantomImages.py


Running the tests
    The tests should always be launched to check your installation.
    These tests are integration and unit tests::

    $ unittests /tests

