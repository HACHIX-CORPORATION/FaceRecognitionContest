import multiprocessing
# Config for main.py
# workers = multiprocessing.cpu_count() * 2 + 1
workers = multiprocessing.cpu_count() + 1

# Config for main_socket.py 
# workers = 1
# worker_class = "eventlet"

# Common config
timeout = 60*60*12
