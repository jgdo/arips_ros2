import json
import multiprocessing
import os
import signal
import threading
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
import yaml

from ament_index_python.packages import get_package_share_directory

from component_manager_msgs.msg import ComponentsState
from component_manager_msgs.srv import (
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
    running: bool = False
    process: multiprocessing.Process | None = None
    started_by: set[str] = field(default_factory=set)


class ComponentManagerNode(Node):

    def __init__(self):
        super().__init__('component_manager')
        self.declare_parameter('config_file', '')

        config_path = self.get_parameter('config_file').get_parameter_value().string_value
        if not config_path:
            self.get_logger().fatal('Parameter "config_file" is required')
            raise SystemExit(1)

        self._components: dict[str, ComponentState] = {}
        self._lock = threading.Lock()

        self._load_config(config_path)

        self._start_srv = self.create_service(
            StartComponent, '~/start_component', self._handle_start)
        self._stop_srv = self.create_service(
            StopComponent, '~/stop_component', self._handle_stop)
        self._list_srv = self.create_service(
            ListComponents, '~/list_components', self._handle_list)

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

    # ── Launch helpers ──────────────────────────────────────────────

    @staticmethod
    def _run_launch(package: str, launch_file: str) -> None:
        """Entry point for the child process. Runs a LaunchService."""
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
        state = self._components[name]
        cfg = state.config

        proc = multiprocessing.Process(
            target=self._run_launch,
            args=(cfg.package, cfg.file),
            name=f'launch-{name}',
            daemon=True,
        )
        proc.start()

        state.process = proc
        state.running = True
        self.get_logger().info(
            f'Launched component "{name}" in process {proc.pid}')

        # Monitor process in a background thread so we can detect
        # unexpected exits without blocking.
        def _monitor() -> None:
            proc.join()
            with self._lock:
                if state.process is proc:
                    state.running = False
                    state.process = None
                    self.get_logger().info(
                        f'Component "{name}" process exited')

        threading.Thread(
            target=_monitor, daemon=True, name=f'mon-{name}').start()

    def _shutdown_component(self, name: str) -> None:
        state = self._components[name]
        if state.process is not None and state.process.is_alive():
            self.get_logger().info(f'Shutting down component "{name}"')
            # Send SIGINT first for graceful shutdown, then SIGKILL as fallback
            try:
                os.kill(state.process.pid, signal.SIGINT)
            except OSError:
                pass
            state.process.join(timeout=5.0)
            if state.process.is_alive():
                state.process.kill()
                state.process.join(timeout=2.0)
        state.process = None
        state.running = False
        state.started_by.clear()

    # ── State publishing ────────────────────────────────────────────

    def _build_state_msg(self) -> ComponentsState:
        """Build a ComponentsState message from the current state. Must be
        called while holding ``self._lock``."""
        msg = ComponentsState()
        for name in sorted(self._components):
            state = self._components[name]
            msg.names.append(name)
            msg.running.append(state.running)
            msg.started_by.append(json.dumps(sorted(state.started_by)))
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
            if state.running and 'user' in state.started_by:
                response.success = True
                response.message = (
                    f'Component "{name}" is already running '
                    f'(started by: {state.started_by})')
                return response

            # Resolve transitive deps (bottom-up order)
            deps = self._resolve_dependencies(name)

            started: list[str] = []

            # Start dependencies first
            for dep_name in deps:
                dep_state = self._components[dep_name]
                if not dep_state.running:
                    self._launch_component(dep_name)
                    started.append(dep_name)
                # Record that *name* caused this dep to be running
                dep_state.started_by.add(name)

            # Start the component itself
            if not state.running:
                self._launch_component(name)
                started.append(name)
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
            if not state.running:
                response.success = False
                response.message = f'Component "{name}" is not running'
                return response

            if 'user' not in state.started_by:
                response.success = False
                response.message = (
                    f'Component "{name}" was not started by the user '
                    f'(started by: {state.started_by})')
                return response

            # Remove user as a start source
            state.started_by.discard('user')

            stopped: list[str] = []

            if not state.started_by:
                # No other source keeps it alive — shut it down
                self._shutdown_component(name)
                stopped.append(name)
                # Cascade to dependencies
                self._cascade_stop(name, stopped)
            else:
                response.success = True
                response.message = (
                    f'User source removed from "{name}" but it stays running '
                    f'(still needed by: {state.started_by})')
                return response

            response.success = True
            if stopped:
                response.message = f'Stopped: {", ".join(stopped)}'
            else:
                response.message = f'Component "{name}" stop requested'
            self._publish_state()
            return response

    def _cascade_stop(self, name: str, stopped: list[str]) -> None:
        """After stopping *name*, remove it as a source from its dependencies
        and recursively stop any that become sourceless."""
        for dep_name in self._components[name].config.dependencies:
            dep_state = self._components[dep_name]
            dep_state.started_by.discard(name)
            if dep_state.running and not dep_state.started_by:
                self._shutdown_component(dep_name)
                stopped.append(dep_name)
                self._cascade_stop(dep_name, stopped)

    def _handle_list(
        self,
        request: ListComponents.Request,
        response: ListComponents.Response,
    ) -> ListComponents.Response:
        with self._lock:
            response.state = self._build_state_msg()
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
