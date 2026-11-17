"""Fleet coordinator: the single writer of fleet state.

Phase 2a holds only the decision layer, ported from couliglig-gamesim:
    coordinator.spec       world snapshot, observation builder, action decoding
    coordinator.policies   RL (ONNX) and heuristic decision policies
The service itself (bus, registry, rounds, tasks) comes in later parts.
"""
