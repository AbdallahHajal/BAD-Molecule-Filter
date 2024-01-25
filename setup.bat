:: This batch file installs file and dependencies and run server
:: ECHO OFF
ECHO Installing And Running Development Server....
python  -m pip install -r requirements.txt --progress-bar off
start /wait "" http://localhost:5000/
flask run
PAUSE