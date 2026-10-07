"""Modules shared by the taxonomy generator and the world package: code that neither owns.

``boolean`` holds truth tables, the Shepard, Hovland, and Jenkins types, and the minimal
disjunctive normal form. It moved here from ``semantic_world.taxonomy`` in stage a1 of
``docs/specs/WORLD_AND_LANGUAGE.md``, so that the world package's matrices can read truth tables
without importing the taxonomy generator (which imports the world package in turn).
"""
