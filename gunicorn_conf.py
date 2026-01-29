"""
Gunicorn configuration file for TinyBERT Service.

This configuration is optimized for production deployment with multiple workers.
"""

import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1  # Common formula: (2 x CPU cores) + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000  # Restart workers after this many requests (helps with memory leaks)
max_requests_jitter = 1000  # Add randomness to max_requests to prevent all workers restarting at once
timeout = 120  # Workers silent for more than this many seconds are killed and restarted
keepalive = 5  # Wait this many seconds for requests on Keep-Alive connections

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "tinybert_api"

# Server mechanics
daemon = False  # Don't daemonize (required for Docker)
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed, uncomment and configure)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Pre/post fork hooks
def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info("Worker received SIGINT or SIGQUIT")

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT")
