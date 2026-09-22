import json
import multiprocessing
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
import yaml

from ament_index_python.packages import get_package_share_directory

from component_manager_msgs.msg import ComponentsState
from component_manager_msgs.srv import (
    GetComponentLog,
    ListComponents,
    StartComponent,
    StopComponent,
)


@dataclass
class ComponentConfig:
    package: str
    file: str
    dependencies: list[str]


@dataclass
class ComponentState:
    config: ComponentConfig
    state: int = ComponentsState.OFF
    process: multiprocessing.Process | None = None
    started_by: set[str] = field(default_factory=set)
    log: deque = field(default_factory=deque)
    # Set right before we deliberately terminate the process, so the monitor
    # thread can distinguish a requested stop from an unexpected exit.
    expected_shutdown: bool = False
    pending_final_state: int | None = None
    pending_cascade: str | None = None


class ComponentManagerNode(Node):
    def __init__(self):
        super().__init__('component_manager')
        self.declare_parameter('config_file', '')
        self.declare_parameter('log_line_limit', 10000)

        config_path = self.get_parameter('config_file').get_parameter_value().string_value
        if not config_path:
            self.get_logger().fatal('Parameter "config_file" is required')
            raise SystemExit(1)

        self._log_line_limit = (
            self.get_parameter('log_line_limit').get_parameter_value().integer_value)

        self._components: dict[str, ComponentState] = {}
        self._dependents: dict[str, set[str]] = {}
        self._lock = threading.Lock()

        self._load_config(config_path)

        self._start_srv = self.create_service(
            StartComponent, '~/start_component', self._handle_start)
        self._stop_srv = self.create_service(
            StopComponent, '~/stop_component', self._handle_stop)
        self._list_srv = self.create_service(
            ListComponents, '~/list_components', self._handle_list)
        self._log_srv = self.create_service(
            GetComponentLog, '~/get_component_log', self._handle_get_log)

        self._state_pub = self.create_publisher(
            ComponentsState, '~/components_state', 10)

        self.get_logger().info(
            f'Component manager ready with {len(self._components)} components')

    # ── Config loading ──────────────────────────────────────────────

    def _load_config(self, path: str) -> None:
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        components_raw = raw.get('components', {})
        if not components_raw:
            self.get_logger().fatal('No components defined in config')
            raise SystemExit(1)

        # Build configs
        for name, cfg in components_raw.items():
            self._components[name] = ComponentState(
                config=ComponentConfig(
                    package=cfg['package'],
                    file=cfg['file'],
                    dependencies=cfg.get('dependencies', []),
                ),
                log=deque(maxlen=self._log_line_limit),
            )

        # Validate all dependency references exist
        for name, state in self._components.items():
            for dep in state.config.dependencies:
                if dep not in self._components:
                    self.get_logger().fatal(
                        f'Component "{name}" depends on unknown component "{dep}"')
                    raise SystemExit(1)

        # Detect cycles
        self._detect_cycles()

        # Build reverse-dependency map (dependents) for failure propagation
        self._dependents = {name: set() for name in self._components}
        for name, state in self._components.items():
            for dep in state.config.dependencies:
                self._dependents[dep].add(name)

        self.get_logger().info(f'Loaded config from {path}')

    def _detect_cycles(self) -> None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in self._components}

        def dfs(node: str, path: list[str]) -> None:
            color[node] = GRAY
            path.append(node)
            for dep in self._components[node].config.dependencies:
                if color[dep] == GRAY:
                    cycle = path[path.index(dep):]
                    self.get_logger().fatal(
                        f'Dependency cycle detected: {" -> ".join(cycle + [dep])}')
                    raise SystemExit(1)
                if color[dep] == WHITE:
                    dfs(dep, path)
            path.pop()
            color[node] = BLACK

        for name in self._components:
            if color[name] == WHITE:
                dfs(name, [])

    # ── Dependency resolution ───────────────────────────────────────

    def _resolve_dependencies(self, name: str) -> list[str]:
        """Return transitive dependencies in bottom-up order (deepest first),
        excluding *name* itself."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            for dep in self._components[n].config.dependencies:
                visit(dep)
            order.append(n)

        for dep in self._components[name].config.dependencies:
            visit(dep)

        return order  # bottom-up: leaves first

    @staticmethod
    def _is_alive(state: ComponentState) -> bool:
        return state.process is not None and state.process.is_alive()

    # ── Launch helpers ──────────────────────────────────────────────

    @staticmethod
    def _run_launch(package: str, launch_file: str, log_write_fd: int) -> None:
        """Entry point for the child process. Runs a LaunchService."""
        # Redirect stdout/stderr before anything else so descendant
        # subprocesses launched by the LaunchService inherit it too.
        os.dup2(log_write_fd, 1)
        os.dup2(log_write_fd, 2)
        os.close(log_write_fd)

        from launch import LaunchDescription, LaunchService
        from launch.actions import IncludeLaunchDescription
        from launch.launch_description_sources import (
            PythonLaunchDescriptionSource,
        )

        pkg_share = get_package_share_directory(package)
        path = os.path.join(pkg_share, launch_file)

        ld = LaunchDescription([
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(path),
            ),
        ])

        ls = LaunchService(argv=[])
        ls.include_launch_description(ld)
        ls.run()

    def _launch_component(self, name: str) -> None:
        """Start (or restart) a component's process. Must be called while
        holding ``self._lock``."""
        state = self._components[name]
        cfg = state.config

        state.log.clear()
        state.expected_shutdown = False
        state.pending_final_state = None
        state.pending_cascade = None

        read_fd, write_fd = os.openpty()
        proc = multiprocessing.Process(
            target=self._run_launch,
            args=(cfg.package, cfg.file, write_fd),
            name=f'launch-{name}',
            daemon=True,
        )
        proc.start()
        os.close(write_fd)

        state.process = proc
        state.state = ComponentsState.RUNNING
        self.get_logger().info(
            f'Launched component "{name}" in process {proc.pid}')

        threading.Thread(
            target=self._read_log, args=(name, read_fd),
            daemon=True, name=f'log-{name}').start()
        threading.Thread(
            target=self._monitor, args=(name, proc),
            daemon=True, name=f'mon-{name}').start()

    def _monitor(self, name: str, proc: multiprocessing.Process) -> None:
        """Waits for a launched process to exit and updates its state
        accordingly. The sole caller of ``proc.join()`` for this process."""
        proc.join()
        with self._lock:
            state = self._components[name]
            if state.process is not proc:
                return  # superseded by a newer launch of this component
            state.process = None

            if state.expected_shutdown:
                final = (
                    state.pending_final_state
                    if state.pending_final_state is not None
                    else ComponentsState.OFF)
                cascade = state.pending_cascade
                state.state = final
                state.expected_shutdown = False
                state.pending_final_state = None
                state.pending_cascade = None
                if final == ComponentsState.OFF:
                    state.started_by.clear()
                self.get_logger().info(f'Component "{name}" stopped')
                if cascade == 'stop_unused_deps':
                    self._cascade_stop_dependencies(name)
            else:
                state.state = ComponentsState.FAILURE
                state.started_by.clear()
                self.get_logger().warn(
                    f'Component "{name}" exited unexpectedly')
                self._propagate_failure(name)

            self._publish_state()

    def _begin_shutdown(self, name: str, final_state: int, cascade: str | None) -> None:
        """Request a graceful shutdown of a running component. Non-blocking;
        must be called while holding ``self._lock``."""
        state = self._components[name]
        if state.state == ComponentsState.SHUTTING_DOWN or not self._is_alive(state):
            return
        state.state = ComponentsState.SHUTTING_DOWN
        state.expected_shutdown = True
        state.pending_final_state = final_state
        state.pending_cascade = cascade
        proc = state.process
        threading.Thread(
            target=self._signal_and_escalate, args=(proc,),
            daemon=True, name=f'kill-{name}').start()

    @staticmethod
    def _signal_and_escalate(proc: multiprocessing.Process) -> None:
        """Send SIGINT, then escalate to SIGKILL if it doesn't exit in time."""
        try:
            os.kill(proc.pid, signal.SIGINT)
        except OSError:
            print(f'Failed to send SIGINT to process {proc.pid}')
            pass
        deadline = time.monotonic() + 5.0
        while proc.is_alive() and time.monotonic() < deadline:
            time.sleep(0.2)
        if proc.is_alive():
            try:
                print(f'Escalating to SIGKILL for process {proc.pid}')
                proc.kill()
            except OSError:
                print(f'Failed to kill process {proc.pid}')
                pass

    def _cascade_stop_dependencies(self, name: str) -> None:
        """After *name* has stopped, remove it as a source from its
        dependencies and shut down any that become sourceless. Must be
        called while holding ``self._lock``."""
        for dep_name in self._components[name].config.dependencies:
            dep_state = self._components[dep_name]
            dep_state.started_by.discard(name)
            if not dep_state.started_by and dep_state.state == ComponentsState.RUNNING:
                self._begin_shutdown(dep_name, ComponentsState.OFF, 'stop_unused_deps')

    def _collect_running_dependents(self, name: str) -> set[str]:
        """Return the transitive dependents of *name* that are currently
        RUNNING (they cannot function once *name* is stopped). Must be
        called while holding ``self._lock``."""
        stack = list(self._dependents.get(name, ()))
        seen: set[str] = set()
        while stack:
            dep = stack.pop()
            if dep in seen:
                continue
            seen.add(dep)
            stack.extend(self._dependents.get(dep, ()))
        return {d for d in seen if self._components[d].state == ComponentsState.RUNNING}

    def _propagate_failure(self, name: str) -> None:
        """Flag transitive dependents of a failed component as
        DEPENDENCY_FAILURE, without touching their processes. Must be called
        while holding ``self._lock``."""
        stack = list(self._dependents.get(name, ()))
        seen: set[str] = set()
        while stack:
            dep = stack.pop()
            if dep in seen:
                continue
            seen.add(dep)
            dep_state = self._components[dep]
            if dep_state.state == ComponentsState.RUNNING:
                dep_state.state = ComponentsState.DEPENDENCY_FAILURE
                stack.extend(self._dependents.get(dep, ()))
            elif dep_state.state == ComponentsState.SHUTTING_DOWN:
                dep_state.pending_final_state = ComponentsState.DEPENDENCY_FAILURE

    # ── Log capture ──────────────────────────────────────────────────

    def _read_log(self, name: str, read_fd: int) -> None:
        buf = b''
        try:
            while True:
                try:
                    chunk = os.read(read_fd, 4096)
                except OSError:
                    break  # pty raises EIO once all write ends are closed
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    self._append_log(name, line.decode('utf-8', errors='replace'))
        finally:
            if buf:
                self._append_log(name, buf.decode('utf-8', errors='replace'))
            os.close(read_fd)

    def _append_log(self, name: str, line: str) -> None:
        with self._lock:
            state = self._components.get(name)
            if state is not None:
                state.log.append(line)

    # ── State publishing ────────────────────────────────────────────

    def _build_state_msg(self) -> ComponentsState:
        """Build a ComponentsState message from the current state. Must be
        called while holding ``self._lock``."""
        msg = ComponentsState()
        for name in sorted(self._components):
            state = self._components[name]
            msg.names.append(name)
            msg.state.append(state.state)
            msg.started_by.append(json.dumps(sorted(state.started_by)))
            msg.dependencies.append(json.dumps(state.config.dependencies))
        return msg

    def _publish_state(self) -> None:
        """Publish current components state. Must be called while holding
        ``self._lock``."""
        self._state_pub.publish(self._build_state_msg())

    # ── Service handlers ────────────────────────────────────────────

    def _handle_start(
        self,
        request: StartComponent.Request,
        response: StartComponent.Response,
    ) -> StartComponent.Response:
        name = request.name
        with self._lock:
            if name not in self._components:
                response.success = False
                response.message = f'Unknown component "{name}"'
                return response

            state = self._components[name]
            if state.state == ComponentsState.SHUTTING_DOWN:
                response.success = False
                response.message = f'Component "{name}" is currently shutting down'
                return response

            # Resolve transitive deps (bottom-up order)
            deps = self._resolve_dependencies(name)

            for dep_name in deps:
                if self._components[dep_name].state == ComponentsState.SHUTTING_DOWN:
                    response.success = False
                    response.message = (
                        f'Dependency "{dep_name}" is currently shutting down')
                    return response

            if state.state == ComponentsState.RUNNING and 'user' in state.started_by:
                response.success = True
                response.message = (
                    f'Component "{name}" is already running '
                    f'(started by: {state.started_by})')
                return response

            started: list[str] = []

            # Start dependencies first, only relaunching the ones actually
            # not running; ones that are still alive but mislabeled (e.g.
            # DEPENDENCY_FAILURE) just have their state healed.
            for dep_name in deps:
                dep_state = self._components[dep_name]
                if not self._is_alive(dep_state):
                    self._launch_component(dep_name)
                    started.append(dep_name)
                elif dep_state.state != ComponentsState.RUNNING:
                    dep_state.state = ComponentsState.RUNNING
                dep_state.started_by.add(name)

            # Start the component itself
            if not self._is_alive(state):
                self._launch_component(name)
                started.append(name)
            elif state.state != ComponentsState.RUNNING:
                state.state = ComponentsState.RUNNING
            state.started_by.add('user')

            parts = [f'Component "{name}" started']
            if started:
                parts.append(f'(launched: {", ".join(started)})')
            response.success = True
            response.message = ' '.join(parts)
            self._publish_state()
            return response

    def _handle_stop(
        self,
        request: StopComponent.Request,
        response: StopComponent.Response,
    ) -> StopComponent.Response:
        name = request.name
        with self._lock:
            if name not in self._components:
                response.success = False
                response.message = f'Unknown component "{name}"'
                return response

            state = self._components[name]
            if state.state != ComponentsState.RUNNING:
                response.success = False
                response.message = f'Component "{name}" is not running'
                return response

            # Anything depending on *name*, directly or transitively, can no
            # longer function once it stops — stop those first regardless of
            # who originally started *name* or its dependents.
            affected = self._collect_running_dependents(name)
            for dep_name in affected:
                self._begin_shutdown(dep_name, ComponentsState.OFF, 'stop_unused_deps')
            self._begin_shutdown(name, ComponentsState.OFF, 'stop_unused_deps')

            response.success = True
            if affected:
                response.message = (
                    f'Shutdown of "{name}" initiated '
                    f'(also stopping dependents: {", ".join(sorted(affected))})')
            else:
                response.message = f'Shutdown of "{name}" initiated'
            self._publish_state()
            return response

    def _handle_list(
        self,
        request: ListComponents.Request,
        response: ListComponents.Response,
    ) -> ListComponents.Response:
        with self._lock:
            response.state = self._build_state_msg()
        return response

    def _handle_get_log(
        self,
        request: GetComponentLog.Request,
        response: GetComponentLog.Response,
    ) -> GetComponentLog.Response:
        with self._lock:
            if request.name not in self._components:
                response.success = False
                response.message = f'Unknown component "{request.name}"'
                response.log = ''
                return response

            state = self._components[request.name]
            response.success = True
            response.message = ''
            response.log = '\n'.join(state.log)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ComponentManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
