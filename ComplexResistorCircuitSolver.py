import numpy as np
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional, Callable, Union
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


@dataclass
class Component:
    name: str
    type: str
    value: Union[float, Callable[[float], float]]
    nodes: Tuple[int, int]
    current: float = 0.0
    is_time_varying: bool = False
    initial_condition: float = 0.0


class DynamicCircuitSolver:
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.nodes: Set[int] = set()
        self.reference_node: Optional[int] = None
        self.voltage_sources: List[str] = []
        self.inductors: List[str] = []
        self.capacitors: List[str] = []
        self.time = 0.0
        self.state = None
        self.dt = 0.0

        # Simulation parameters
        self.abstol = 1e-8
        self.reltol = 1e-6
        self.max_step = 1e-5
        self.method = 'LSODA'

    def set_simulation_parameters(self, abstol: float = 1e-8, reltol: float = 1e-6,
                                  max_step: float = 1e-5, method: str = 'LSODA'):
        """Set simulation parameters for numerical stability."""
        self.abstol = abstol
        self.reltol = reltol
        self.max_step = max_step
        self.method = method

    def add_resistor(self, name: str, resistance: float, node1: int, node2: int):
        """Add a resistor to the circuit."""
        if resistance <= 0:
            raise ValueError(f"Resistance must be positive. Got {resistance} for {name}.")
        self.components[name] = Component(name, 'resistor', resistance, (node1, node2))
        self.nodes.update([node1, node2])

    def add_inductor(self, name: str, inductance: float, node1: int, node2: int, initial_current: float = 0.0):
        """Add an inductor to the circuit."""
        if inductance <= 0:
            raise ValueError(f"Inductance must be positive. Got {inductance} for {name}.")
        self.components[name] = Component(name, 'inductor', inductance, (node1, node2),
                                          initial_condition=initial_current)
        self.nodes.update([node1, node2])
        self.inductors.append(name)

    def add_capacitor(self, name: str, capacitance: float, node1: int, node2: int, initial_voltage: float = 0.0):
        """Add a capacitor to the circuit."""
        if capacitance <= 0:
            raise ValueError(f"Capacitance must be positive. Got {capacitance} for {name}.")
        self.components[name] = Component(name, 'capacitor', capacitance, (node1, node2),
                                          initial_condition=initial_voltage)
        self.nodes.update([node1, node2])
        self.capacitors.append(name)

    def add_voltage_source(self, name: str, voltage: Union[float, Callable[[float], float]], node1: int, node2: int):
        """Add a voltage source to the circuit."""
        is_time_varying = callable(voltage)
        self.components[name] = Component(name, 'voltage_source', voltage, (node1, node2),
                                          is_time_varying=is_time_varying)
        self.nodes.update([node1, node2])
        self.voltage_sources.append(name)

    def add_current_source(self, name: str, current: Union[float, Callable[[float], float]], node1: int, node2: int):
        """Add a current source to the circuit."""
        is_time_varying = callable(current)
        self.components[name] = Component(name, 'current_source', current, (node1, node2),
                                          is_time_varying=is_time_varying)
        self.nodes.update([node1, node2])

    def set_reference_node(self, node: int):
        """Set the reference node (ground) for the circuit."""
        if node not in self.nodes:
            raise ValueError(f"Node {node} does not exist in the circuit.")
        self.reference_node = node

    def get_source_value(self, component: Component, t: float) -> float:
        """Get the value of a source at time t."""
        if component.is_time_varying:
            return component.value(t)
        return component.value

    def build_system_matrices(self, t: float, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build the system matrices with improved conditioning."""
        n_nodes = len(self.nodes) - 1
        n_l = len(self.inductors)
        n_vs = len(self.voltage_sources)
        size = n_nodes + n_l + n_vs

        # Initialize matrices
        A = np.zeros((size, size))
        b = np.zeros(size)

        # Add small conductance for numerical stability
        gmin = 1e-12

        # Create node mapping
        node_map = {node: idx for idx, node in enumerate(sorted(self.nodes - {self.reference_node}))}

        # Extract state variables
        inductor_currents = state[:n_l]

        # Add minimum conductances
        for node in self.nodes - {self.reference_node}:
            idx = node_map[node]
            A[idx, idx] += gmin

        # Add component stamps
        for name, component in self.components.items():
            n1, n2 = component.nodes

            if component.type == 'resistor':
                g = 1.0 / component.value
                self._add_conductance_stamp(A, g, n1, n2, node_map)

            elif component.type == 'voltage_source':
                v = self.get_source_value(component, t)
                idx = n_nodes + self.voltage_sources.index(name)
                self._add_voltage_source_stamp(A, b, v, n1, n2, idx, node_map)

            elif component.type == 'inductor':
                idx = self.inductors.index(name)
                i_L = inductor_currents[idx]
                curr_idx = n_nodes + n_vs + idx

                if n1 != self.reference_node:
                    i = node_map[n1]
                    A[i, curr_idx] += 1
                    A[curr_idx, i] += 1

                if n2 != self.reference_node:
                    i = node_map[n2]
                    A[i, curr_idx] -= 1
                    A[curr_idx, i] -= 1

                L = component.value
                A[curr_idx, curr_idx] = -L / self.dt
                b[curr_idx] = -L / self.dt * i_L

            elif component.type == 'capacitor':
                C = component.value
                g_eq = C / self.dt
                self._add_conductance_stamp(A, g_eq, n1, n2, node_map)
                i_eq = C / self.dt * self._get_capacitor_voltage(component, state)
                if n1 != self.reference_node:
                    b[node_map[n1]] += i_eq
                if n2 != self.reference_node:
                    b[node_map[n2]] -= i_eq

        return A, b

    def simulate(self, t_start: float, t_end: float, dt: float) -> Dict[str, List[float]]:
        """Simulate the circuit using scipy's solve_ivp."""
        if not self.components:
            raise ValueError("Circuit is empty. Add some components before solving.")

        if self.reference_node is None:
            self.reference_node = min(self.nodes)

        self.dt = dt

        # Initialize state vector
        n_l = len(self.inductors)
        n_c = len(self.capacitors)
        initial_state = np.zeros(n_l + n_c)

        # Set initial conditions
        for i, ind_name in enumerate(self.inductors):
            initial_state[i] = self.components[ind_name].initial_condition
        for i, cap_name in enumerate(self.capacitors):
            initial_state[n_l + i] = self.components[cap_name].initial_condition

        # Initialize results with branch voltages instead of node voltages
        results = {
            'time': [],
            'branch_voltages': {name: [] for name in self.components},  # Track voltage across each component
            'currents': {name: [] for name in self.components},
            'state': []
        }

        try:
            solution = solve_ivp(
                fun=self.get_state_derivatives,
                t_span=(t_start, t_end),
                y0=initial_state,
                method=self.method,
                t_eval=np.arange(t_start, t_end + dt / 2, dt),
                atol=self.abstol,
                rtol=self.reltol,
                max_step=self.max_step
            )

            if not solution.success:
                raise ValueError(f"Integration failed: {solution.message}")

            for t_idx, t in enumerate(solution.t):
                state = solution.y[:, t_idx]
                self._store_results(t, state, results)
                results['time'].append(float(t))
                results['state'].append(state.copy())

        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            if len(results['time']) > 0:
                print(f"Returning partial results up to t={results['time'][-1]}")
                return results
            raise

        return results

    def get_state_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """Calculate derivatives with improved numerical stability."""
        try:
            A, b = self.build_system_matrices(t, state)

            # Check condition number
            cond = np.linalg.cond(A)
            if cond > 1e12:
                epsilon = 1e-12
                A = A + epsilon * np.eye(A.shape[0])

            solution = np.linalg.solve(A, b)

            derivatives = np.zeros_like(state)
            n_l = len(self.inductors)
            n_c = len(self.capacitors)

            # Calculate derivatives with bounds
            for i, ind_name in enumerate(self.inductors):
                L = self.components[ind_name].value
                n1, n2 = self.components[ind_name].nodes
                v_L = self._get_voltage_difference(solution, n1, n2)
                derivatives[i] = np.clip(v_L / L, -1e6, 1e6)

            for i, cap_name in enumerate(self.capacitors):
                C = self.components[cap_name].value
                n1, n2 = self.components[cap_name].nodes
                i_C = self._get_current_through(solution, n1, n2)
                derivatives[n_l + i] = np.clip(i_C / C, -1e6, 1e6)

            return derivatives

        except np.linalg.LinAlgError as e:
            print(f"Warning: Linear algebra error at t={t}: {str(e)}")
            return np.zeros_like(state)

    def _store_component_currents(self, solution: np.ndarray, state: np.ndarray, results: Dict[str, List[float]]):
        """Store currents through all components."""
        node_map = {node: idx for idx, node in enumerate(sorted(self.nodes - {self.reference_node}))}
        n_nodes = len(self.nodes) - 1

        for name, component in self.components.items():
            n1, n2 = component.nodes
            current = 0.0

            if component.type == 'resistor':
                # I = V/R for resistors
                v1 = 0.0 if n1 == self.reference_node else solution[node_map[n1]]
                v2 = 0.0 if n2 == self.reference_node else solution[node_map[n2]]
                current = (v1 - v2) / component.value

            elif component.type == 'inductor':
                # Get current directly from state for inductors
                idx = self.inductors.index(name)
                current = state[idx]

            elif component.type == 'capacitor':
                # I = C * dV/dt for capacitors
                idx = len(self.inductors) + self.capacitors.index(name)
                current = self.get_state_derivatives(self.time, state)[idx] * component.value

            elif component.type == 'voltage_source':
                # Get current from the solution vector
                idx = n_nodes + self.voltage_sources.index(name)
                current = solution[idx]

            elif component.type == 'current_source':
                # Use the source value directly
                current = self.get_source_value(component, self.time)

            # Store the current with bounds checking
            current = np.clip(current, -1e6, 1e6)
            results['currents'][name].append(float(current))

    def _calculate_branch_voltage(self, component: Component, solution: np.ndarray) -> float:
        """Calculate voltage across a component."""
        n1, n2 = component.nodes
        node_map = {node: idx for idx, node in enumerate(sorted(self.nodes - {self.reference_node}))}

        v1 = 0.0 if n1 == self.reference_node else solution[node_map[n1]]
        v2 = 0.0 if n2 == self.reference_node else solution[node_map[n2]]

        return v1 - v2

    def _store_results(self, t: float, state: np.ndarray, results: Dict[str, List[float]]):
        """Store results with error checking."""
        try:
            self.time = t
            A, b = self.build_system_matrices(t, state)

            cond = np.linalg.cond(A)
            if cond > 1e12:
                epsilon = 1e-12
                A = A + epsilon * np.eye(A.shape[0])

            solution = np.linalg.solve(A, b)

            # Store branch voltages for each component
            for name, component in self.components.items():
                voltage = self._calculate_branch_voltage(component, solution)
                voltage = np.clip(voltage, -1e6, 1e6)
                results['branch_voltages'][name].append(float(voltage))

            self._store_component_currents(solution, state, results)

        except np.linalg.LinAlgError as e:
            print(f"Warning: Error storing results at t={t}: {str(e)}")
            for name in self.components:
                if results['branch_voltages'][name]:
                    results['branch_voltages'][name].append(results['branch_voltages'][name][-1])
                else:
                    results['branch_voltages'][name].append(0.0)
                if results['currents'][name]:
                    results['currents'][name].append(results['currents'][name][-1])
                else:
                    results['currents'][name].append(0.0)

    def _add_conductance_stamp(self, A: np.ndarray, g: float, n1: int, n2: int, node_map: Dict[int, int]):
        """Add conductance stamps."""
        if n1 != self.reference_node:
            i = node_map[n1]
            A[i, i] += g
            if n2 != self.reference_node:
                j = node_map[n2]
                A[i, j] -= g
                A[j, i] -= g
                A[j, j] += g

    def _add_voltage_source_stamp(self, A: np.ndarray, b: np.ndarray, v: float, n1: int, n2: int,
                                  idx: int, node_map: Dict[int, int]):
        """Add voltage source stamps."""
        if n1 != self.reference_node:
            i = node_map[n1]
            A[i, idx] += 1
            A[idx, i] += 1
        if n2 != self.reference_node:
            i = node_map[n2]
            A[i, idx] -= 1
            A[idx, i] -= 1
        b[idx] = v

    def _get_capacitor_voltage(self, component: Component, state: np.ndarray) -> float:
        """Get capacitor voltage from state."""
        idx = len(self.inductors) + self.capacitors.index(component.name)
        return state[idx]

    def _get_voltage_difference(self, solution: np.ndarray, n1: int, n2: int) -> float:
        """Get voltage difference between nodes."""
        node_map = {node: idx for idx, node in enumerate(sorted(self.nodes - {self.reference_node}))}
        if n1 == self.reference_node:
            return -solution[node_map[n2]]
        if n2 == self.reference_node:
            return solution[node_map[n1]]
        return solution[node_map[n1]] - solution[node_map[n2]]

    def _get_current_through(self, solution: np.ndarray, n1: int, n2: int) -> float:
        """Get current through a branch."""
        if n1 == self.reference_node:
            return -solution[n2]
        if n2 == self.reference_node:
            return solution[n1]
        return solution[n1] - solution[n2]

    def plot_results(self, results: Dict[str, List[float]], components_to_plot: List[str] = None,
                     figsize: Tuple[int, int] = (12, 8)):
        """Plot simulation results with branch voltages."""
        times = np.array(results['time']) * 1000  # Convert to milliseconds

        if components_to_plot is None:
            components_to_plot = list(self.components.keys())

        # Verify data exists for the requested components
        components_to_plot = [comp for comp in components_to_plot
                            if comp in results['branch_voltages'] and results['branch_voltages'][comp] and
                               comp in results['currents'] and results['currents'][comp]]

        if not components_to_plot:
            raise ValueError("No valid data to plot")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot branch voltages
        for component in components_to_plot:
            voltages = np.array(results['branch_voltages'][component])
            if len(voltages) == len(times):
                ax1.plot(times, voltages, label=f'{component}')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Branch Voltages')
        ax1.grid(True)
        ax1.legend()

        # Plot branch currents
        for component in components_to_plot:
            currents = np.array(results['currents'][component])
            if len(currents) == len(times):
                ax2.plot(times, currents, label=component)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('Branch Currents')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        return fig, (ax1, ax2)

def main():
    # Example circuit with time-varying sources
    circuit = DynamicCircuitSolver()

    def sinusoidal_voltage(t):
        return 10 * np.sin(2 * np.pi * 1591.549 * t)  # 10V peak, 1kHz

    # Add components with more reasonable values for the frequency
    '''
    circuit.add_voltage_source('V1', sinusoidal_voltage, 0, 1)
    circuit.add_resistor('R1', 100.0, 1, 2)  # 100Ω
    circuit.add_inductor('L1', 0.01, 2, 3)  # 10mH - reduced from 0.1H
    circuit.add_capacitor('C1', 1e-6, 3, 0)  # 1µF
    '''
    circuit.add_resistor('R1', 4.0, 1, 2)  # 100Ω
    circuit.add_resistor('R2', 10.0, 1, 3)
    circuit.add_inductor('L1', 2, 2, 0)  # 10mH - reduced from 0.1H
    circuit.add_capacitor('C1', 0.05, 3, 0)  # 1µF
    circuit.add_voltage_source('V1', 3, 0, 1)

    circuit.set_reference_node(0)

    # Simulation parameters
    t_start = 0.0
    t_end = 5  # 5ms simulation
    dt = 1e-2  # Smaller timestep for better stability

    try:
        results = circuit.simulate(t_start, t_end, dt)
        fig, (ax1, ax2) = circuit.plot_results(
            results,
            components_to_plot=['R1', 'L1', 'C1', 'R2']
        )
        plt.show()
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
