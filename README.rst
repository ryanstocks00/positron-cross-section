Positron Cross Section Analysis
===============================

This is a python application that assists with the analysis and plotting of positron cross sections

|pre-commit|

Installation
============

This application has been designed to run on either a windows or a unix
style operating system or terminal. The windows installation will be
sufficient if active development is not required, however active
development will be much easier on a unix style system. On windows
systems, this can be achieved using the windows subsystem for linux
(WSL). The windows subsystem for linux can be installed by following the
steps detailed
`here <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.

Windows Installation Instructions
---------------------------------

On a windows system, we will install the drone simulation application in
Windows PowerShell. To open Windows PowerShell as an administrator,
right-click the windows start button and select
``Windows PowerShell (Admin)`` and then select "Yes" when prompted.

Installing Chocolatey
~~~~~~~~~~~~~~~~~~~~~

We will use the package manager Chocolatey to install the correct
versions of python. To install Chocolatey, open Windows PowerShell as an
administrator and run the command:

.. code:: powershell

    Set-ExecutionPolicy Bypass -Scope Process -Force;
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
    iwr https://chocolatey.org/install.ps1 -UseBasicParsing | iex

You may need to reopen PowerShell before using choco (Chocolatey) in the
next step.

Installing Python
~~~~~~~~~~~~~~~~~

As this is a python application, we first need to install python. We
will use Chocolatey to install version 3.9.5. Open Windows PowerShell as
an administrator and run the command:

.. code:: powershell

    choco install -y python3 --version=3.9.5 --force

If the installation is successful, python 3.9 can then be accessed using
the command ``py -3.9`` (and exited using the command ``exit()``).

Installing Git
~~~~~~~~~~~~~~

By default, Windows PowerShell does not come with the useful version
control system git. Hence if git is not installed on your system (this
can be checked using the command ``git --version`` in PowerShell, if
PowerShell displays a version of git this indicates git is already
installed), please install it using the following command:

.. code:: powershell

    choco install -y git.install --params "/GitAndUnixToolsOnPath /SChannel /NoAutoCrlf"

Setting the Execution Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also need to change the PowerShell execution policy to allow us to
run external scripts by running the following command in PowerShell
(with admin):

.. code:: powershell

    set-executionpolicy remotesigned

Downloading and Installing the Positron Cross Section Analysis package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have python and git installed, open a PowerShell window in
the folder that you would like to download the positron cross section analysis
application. This can be done by opening File Explorer, navigating to
the folder in which you want to start PowerShell, then typing powershell
into the address bar and hitting enter. Alternately, you can start
WindowsPowershell as before and then navigating to the target folder
using ``cd`` (change directory). You can now run the following commands
to download and install the application:

1. ``git clone https://github.com/ryanstocks00/positron-cross-section-analysis``
2. ``cd positron-cross-section-analysis``
3. ``.\tools\windows-install.ps1``

Congratulations! The poistron cross section analysis application is now
(hopefully) successfully installed and can be run using the command

.. code:: powershell

    positron-cross-section --help

in a new powershell window.

Unix installation instructions (Including WSL)
----------------------------------------------

Setting up Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The python code in this application requires python 3.9 or greater. To
maintain the integrity of other python applications on your system, it
is highly recommended to use a separate python environment for the
positron cross section analysis acpplication, however it can also be installed directly if
your python version meets the requirements.

**Installing a python environment**

To set up a separate python environment (recommended), we will use
`pyenv <https://github.com/pyenv/pyenv>`__ which allows us to isolate
the positron cross section analysis development environment and python
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

Downloading and installing the positron cross section analysis application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To download the source code and install the application, please open a
terminal, navigate to the folder in which you would like to perform the
installation and run the commands

1. ``git clone https://github.com/ryanstocks00/positron-cross-section-analysis``
2. ``cd positron-cross-section-analysis``
3. ``source tools/install-dev-env``

Congratulations! The poistron cross section analysis application is now
(hopefully) successfully installed and can be run using the command

.. code:: bash

    positron-cross-section --help

in a new terminal window.

.. |pre-commit| image:: https://github.com/ryanstocks00/positron-cross-section-analysis/actions/workflows/python-3.9-pre-commit.yml/badge.svg
   :target: https://github.com/ryanstocks00/positron-cross-section-analysis/actions/workflows/python-3.9-pre-commit.yml
