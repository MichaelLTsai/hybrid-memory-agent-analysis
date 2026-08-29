#!/usr/bin/env python3
"""Run a command in its own session, fully detached from this terminal.

macOS ships no setsid(1), and a plain background job stays in this shell's
process group: when the parent (here, a Claude Code task) is torn down the
whole group is signalled and the job dies with it. That is exactly how the
first attempt died at 10/21. Double-fork + os.setsid() reparents to init, so
the run survives the shell, the task, and the Claude Code process exiting.

    ./detach.py <logfile> <command> [args...]
"""
import os, sys

log, cmd = sys.argv[1], sys.argv[2:]
if os.fork():                      # parent returns to the shell immediately
    sys.exit(0)
os.setsid()                        # new session, detached from the terminal
if os.fork():                      # second fork: cannot reacquire a terminal
    os._exit(0)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.execvp(cmd[0], cmd)
