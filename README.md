# Simple pdf chat

## Python version: 3.11

## Steps to run the project

1. Clone the repository
2. Create a virtual environment and install the dependencies
```bash
pip install virtualenv
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
3. Run the project
```bash
uvicorn main:app --host 127.0.0.1 --port 3000 --reload
```