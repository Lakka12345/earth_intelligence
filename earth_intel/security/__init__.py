"""
security package — standalone security modules for the Earth Intelligence
multi-agent platform.

Modules in this package are intentionally agent-independent: they accept
plain values (paths, strings, dicts) rather than agent-specific schema
objects, so they can be imported by any current or future agent (Agent 4
dataset download, Agent 5 dataset processing, etc.) without creating a
dependency on those agents' code.
"""
