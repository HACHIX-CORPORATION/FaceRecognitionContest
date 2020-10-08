source /home/hachix/.bashrc

source activate talent5_

gunicorn --bind 0.0.0.0:5000 main:app
