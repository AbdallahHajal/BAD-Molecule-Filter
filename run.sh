#!/bin/bash

echo "Installing And Running Development Server...."

if command -v python > /dev/null; then
    python -m pip install -r requirements.txt --progress-bar off
else
    sudo apt update -y && sudo apt install python3 -y && pip install virtualenv
    

python -m venv venv

source venv/bin/activate

python -m pip install -r requirements.txt --progress-bar off

# Run Flask in development mode
flask run -h localhost -p 5000

if command -v open > /dev/null; then
    open http://localhost:5000/
elif command -v xdg-open > /dev/null; then
    xdg-open http://localhost:5000/
elif command -v start > /dev/null; then
    start http://localhost:5000/
else
    echo "Unable to open the browser. Please open your browser manually and navigate to http://localhost:5000/"
fi

read -p "Press Enter to exit..."
