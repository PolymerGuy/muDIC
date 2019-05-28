Getting started with µDIC
=======================================
In order to get started with µDIC, you need to install it on your computer.
There are two main ways to to this:

*   You can install it via a package manager like PIP or Conda
*   You can  clone the repo


Installing via a package manager:
----------------------------------
Prerequisites:
    This toolkit is tested on Python 2.7x and Python 3.7

On the command line, check if python is available::

$ python --version


If this command does not return the version of you python installation,
you need to fix this first.

If everything seems to work, you install the package in your global python 
environment (Not recommend) via pip::

$ pip install muDIC

and you are good to go!

We recommend that you always use virtual environments by virtualenv or by Conda env.

Virtual env::

$ cd /path/to/your/project
$ python -m virtualenv env
$ source ./env/bin/activate #On Linux and Mac OS
$ env\Scripts\activate.bat #On Windows
$ pip install muDIC


By cloning the repo:
---------------------

These instructions will get you a copy of the project up and running on your 
local machine for development and testing purposes.

Prerequisites:
    This toolkit is tested on Python 2.7x and Python 3.7

Installing:
    Start to clone this repo to your preferred location::

    $ git init
    $ git clone https://github.com/PolymerGuy/myDIC.git



    We recommend that you always use virtual environments, either by virtualenv or by Conda env

    Virtual env::
    
    $ python -m virtualenv env
    $ source ./env/bin/activate #On Linux and Mac OS
    $ env\Scripts\activate.bat #On Windows
    $ pip install -r requirements.txt


    You can now run an example::
    $ python path_to_muDIC/Examples/quick_start.py



Running the tests
------------------
The tests should always be launched to check your installation.
These tests are integration and unit tests

If you installed via a package manger::

    $ nosetests muDIC

If you cloned the repo, you have to call nosetests from within the folder::

    $ nosetests muDIC

