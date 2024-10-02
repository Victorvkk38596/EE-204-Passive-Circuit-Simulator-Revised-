import numpy as np

class ComplexResistorCircuitSolver:
    def __init__(self):
        self.components = {}
        self.nodes = set()
        self.reference_node = None
        self.voltage_sources = []
        self.solution = None

    def add_resistor(self, name, resistance, node1, node2):
        """Add a resistor to the circuit."""
        if resistance <= 0:
            raise ValueError(f"Resistance must be positive. Got {resistance} for {name}.")
        self.components[name] = {'type': 'resistor', 'value': resistance, 'nodes': (node1, node2)}
        self.nodes.update([node1, node2])

    def add_voltage_source(self, name, voltage, node1, node2):
        """Add a voltage source to the circuit."""
        self.components[name] = {'type': 'voltage_source', 'value': voltage, 'nodes': (node1, node2)}
        self.nodes.update([node1, node2])
        self.voltage_sources.append(name)

    def add_current_source(self, name, current, node1, node2):
        """Add a current source to the circuit."""
        self.components[name] = {'type': 'current_source', 'value': current, 'nodes': (node1, node2)}
        self.nodes.update([node1, node2])

    def set_reference_node(self, node):
        """Set the reference node (ground) for the circuit."""
        if node not in self.nodes:
            raise ValueError(f"Reference node {node} is not in the circuit.")
        self.reference_node = node

    def build_matrices(self):
        """Build the matrices needed for solving the circuit using Modified Nodal Analysis."""
        if self.reference_node is None:
            self.reference_node = min(self.nodes)

        nodes = sorted(self.nodes - {self.reference_node})
        node_indices = {node: i for i, node in enumerate(nodes)}

        n = len(nodes)
        m = len(self.voltage_sources)
        size = n + m

        A = np.zeros((size, size))
        z = np.zeros(size)

        # Fill in the conductance matrix and current vector
        for component in self.components.values():
            n1, n2 = component['nodes']
            if component['type'] == 'resistor':
                g = 1 / component['value']
                if n1 != self.reference_node:
                    i = node_indices[n1]
                    A[i, i] += g
                if n2 != self.reference_node:
                    i = node_indices[n2]
                    A[i, i] += g
                if n1 != self.reference_node and n2 != self.reference_node:
                    i, j = node_indices[n1], node_indices[n2]
                    A[i, j] -= g
                    A[j, i] -= g
            elif component['type'] == 'current_source':
                if n1 != self.reference_node:
                    i = node_indices[n1]
                    z[i] -= component['value']
                if n2 != self.reference_node:
                    i = node_indices[n2]
                    z[i] += component['value']

        # Add voltage sources to the matrix
        for i, v_source in enumerate(self.voltage_sources):
            component = self.components[v_source]
            n1, n2 = component['nodes']
            if n1 != self.reference_node:
                j = node_indices[n1]
                A[n+i, j] = 1
                A[j, n+i] = 1
            if n2 != self.reference_node:
                j = node_indices[n2]
                A[n+i, j] = -1
                A[j, n+i] = -1
            z[n+i] = component['value']

        return A, z, node_indices

    def solve(self):
        """Solve the circuit and return node voltages."""
        if not self.components:
            raise ValueError("Circuit is empty. Add some components before solving.")

        if self.reference_node is None:
            raise ValueError("Reference node not set. Use set_reference_node() before solving.")

        A, z, node_indices = self.build_matrices()

        # Check for disconnected nodes
        connected_nodes = set()
        for component in self.components.values():
            connected_nodes.update(component['nodes'])
        if connected_nodes != self.nodes:
            disconnected = self.nodes - connected_nodes
            raise ValueError(f"Disconnected nodes found: {disconnected}")

        try:
            self.solution = np.linalg.solve(A, z)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Unable to solve circuit equations: {e}")

        voltages = {self.reference_node: 0}
        for node, index in node_indices.items():
            voltages[node] = self.solution[index]

        return voltages

    def get_branch_currents(self):
        """Calculate branch currents based on node voltages."""
        if self.solution is None:
            raise ValueError("Circuit must be solved before calculating branch currents.")

        voltages = self.solve()  # Ensure we have the latest solution
        currents = {}
        for name, component in self.components.items():
            n1, n2 = component['nodes']
            v1 = voltages[n1]
            v2 = voltages[n2]
            if component['type'] == 'resistor':
                current = (v1 - v2) / component['value']
                currents[name] = current
            elif component['type'] == 'voltage_source':
                index = len(voltages) - 1 + self.voltage_sources.index(name)
                currents[name] = self.solution[index]
            elif component['type'] == 'current_source':
                currents[name] = component['value']
        return currents

def main():
    # Create a complex resistor circuit
    circuit = ComplexResistorCircuitSolver()

    # Add components
    circuit.add_resistor('R1', 10, 1, 2)
    circuit.add_resistor('R2', 10, 2, 0)
    circuit.add_resistor('R3', 10, 2, 3)
    circuit.add_resistor('R4', 10, 3, 0)
    circuit.add_voltage_source('V1', 10, 0, 1)

    # Set reference node
    circuit.set_reference_node(0)

    # Solve the circuit
    try:
        node_voltages = circuit.solve()
        branch_currents = circuit.get_branch_currents()

        print("Node Voltages:")
        for node, voltage in node_voltages.items():
            print(f"Node {node}: {voltage:.3f} V")

        print("\nBranch Currents:")
        for branch, current in branch_currents.items():
            print(f"{branch}: {current:.3f} A")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()