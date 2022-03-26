Positron Cross Section Analysis
===============================

This is a python application that assists with the analysis and plotting of positron cross sections

|pre-commit| |pypi|

Installation
============

To install the positron cross section application, ensure that you have python version 3.7 or greater with the command

.. code:: bash

    python --version

and then install the positron-cross-section application with

.. code:: bash

    python -m pip install --upgrade pip
    python -m pip install positron-cross-section

You can then run the application using the command

.. code:: bash

    positron-cross-section cross_section_data.csv

which will produce total cross section plots in the `output` folder.

Development environment installation instructions
-------------------------------------------------

Setting up Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The python code in this application requires a development environment with
python 3.9 or greater. To maintain the integrity of other python applications on your system, it
is highly recommended to use a separate python environment for the
positron cross section application, however it can also be installed directly if
your python version meets the requirements.

**Installing a python environment**

To set up a separate python environment (recommended), we will use
`pyenv <https://github.com/pyenv/pyenv>`__ which allows us to isolate
the positron cross section development environment and python
version. To install pyenv, please follow the instructions detailed
`here <https://realpython.com/intro-to-pyenv/>`__. During this
installation, you will get the warning

.. code:: bash

    WARNING: seems you still have not added 'pyenv' to the load path.
    # Load pyenv automatically by adding
    # the following to ~/.bashrc:

To add this text to ~./bashrc, run the command

.. code:: bash

    echo 'export PATH="$HOME/.pyenv/bin:$PATH"
    export PATH="$HOME/.pyenv/shims:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

You now need to reload your shell which can be done by restarting your terminal
or running the command

.. code:: bash

    exec $SHELL

To create a pyenv environment called positrons for this application with
python version 3.9.5, run the commands

1. ``pyenv install 3.9.5``
2. ``pyenv virtualenv 3.9.5 positrons``

Then, prior to following the installation steps below and before each
time using the ``positron-cross-section`` application, you will need
to enter the positrons python environment using the command

``pyenv activate positrons``

Downloading and installing the positron cross section application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To download the source code and install the application, please open a
terminal, navigate to the folder in which you would like to perform the
installation and run the commands

1. ``git clone https://github.com/ryanstocks00/positron-cross-section``
2. ``cd positron-cross-section``
3. ``source tools/install-dev-env``

Congratulations! The poistron cross section application is now
(hopefully) successfully installed and can be run using the command

.. code:: bash

    positron-cross-section --help

in a new terminal window.

.. |pre-commit| image:: https://github.com/ryanstocks00/positron-cross-section/actions/workflows/python-3.9-pre-commit.yml/badge.svg
   :target: https://github.com/ryanstocks00/positron-cross-section/actions/workflows/python-3.9-pre-commit.yml
.. |pypi| image:: https://badge.fury.io/py/positron-cross-section.svg
   :target: https://pypi.org/project/positron-cross-section/
