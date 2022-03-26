# Update pip
py -3.8 -m pip install --upgrade pip setuptools wheel

# Install pre-commit
py -3.8 -m pip install --upgrade pre-commit
pre-commit install --install-hooks
pre-commit autoupdate

# Install pylint
py -3.8 -m pip install --upgrade pylint

# Install bushfire_drone_simulation
py -3.8 -m pip install -e .

Write-Host "Congratulations! The Positron Cross Section Analysis application is now installed."
