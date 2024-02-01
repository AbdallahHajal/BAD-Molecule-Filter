echo Installing Development Server....

python -m venv venv

python -m pip install cython
python -m pip install -r requirements.txt --progress-bar off

start server.bat

END
