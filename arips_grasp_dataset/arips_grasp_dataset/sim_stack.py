"""Start and stop the headless lu simulation with the navigation stack (--start-sim)."""

import os
import signal
import subprocess
import time

SERVER_LINE = 'Gazebo Sim Server'


def descendants(pid):
    """PIDs of all processes below pid (by parent pid, crosses setsid'd children)."""
    children = {}
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat') as f:
                stat = f.read()
        except OSError:
            continue
        # comm may contain spaces; the fields after its closing parenthesis are fixed
        ppid = int(stat.rsplit(')', 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry))
    result, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            result.append(child)
            stack.append(child)
    return result


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # zombies count as gone
        with open(f'/proc/{pid}/stat') as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def _signal(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


class SimStack:
    """ros2 launch arips_gazebo startup.launch.py world:=lu in its own session."""

    def __init__(self, log_dir, logger, gui=False):
        self._log_dir = log_dir
        self._gui = gui
        self._logger = logger
        self._proc = None
        self._log_path = None
        self._launches = 0

    def start(self, attempts=3, server_timeout=60.0):
        """Launch the sim; retry when Gazebo comes up without its server (startup race)."""
        os.makedirs(self._log_dir, exist_ok=True)
        for attempt in range(1, attempts + 1):
            self._launches += 1
            self._log_path = os.path.join(self._log_dir, f'sim_launch_{self._launches}.log')
            log = open(self._log_path, 'w')
            self._logger.info(f'Starting the simulation (attempt {attempt}), log: '
                              f'{self._log_path}')
            self._proc = subprocess.Popen(
                ['ros2', 'launch', 'arips_gazebo', 'startup.launch.py', 'world:=lu',
                 f'gui:={str(self._gui).lower()}'],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
            log.close()
            deadline = time.monotonic() + server_timeout
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    break
                with open(self._log_path, errors='replace') as f:
                    if SERVER_LINE in f.read():
                        return True
                time.sleep(1.0)
            self._logger.warning('Gazebo server did not start, restarting the simulation')
            self.stop()
        return False

    def stop(self, timeout=30.0):
        """SIGINT the launch and everything below it, SIGKILL what does not exit."""
        if self._proc is None:
            return
        # Collected first: the component manager starts its launches in their own sessions,
        # so they are not in the launch's process group and may outlive it.
        pids = descendants(self._proc.pid)
        try:
            os.killpg(self._proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        pids = [pid for pid in pids if _alive(pid)]
        if pids:
            self._logger.info(f'Stopping {len(pids)} leftover processes')
            _signal(pids, signal.SIGINT)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
                time.sleep(0.5)
            _signal([pid for pid in pids if _alive(pid)], signal.SIGKILL)
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._proc = None
