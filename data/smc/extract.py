"""Extracts the module dependency graph (MDG) of the Python stdlib `email` package.

Statically parses each top-level module of `email` with the `ast` module and
records an edge module_a -> module_b whenever module_a imports module_b, for
imports that stay inside the `email` package (intra-package imports only).
Writes the result as a plain edge-list to data/smc/email.mdg.
"""
import ast
import os
import email

pkgdir = os.path.dirname(email.__file__)
modules = sorted(f[:-3] for f in os.listdir(pkgdir) if f.endswith(".py") and f != "__init__.py")

edges = set()
for module in modules:
    path = os.path.join(pkgdir, module + ".py")
    tree = ast.parse(open(path).read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("email."):
                    name = name.split(".")[1]
                if name in modules and name != module:
                    edges.add((module, name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            name = node.module
            if node.level > 0:
                # relative import: "from . import x" or "from .x import y"
                if name == "":
                    for alias in node.names:
                        if alias.name in modules and alias.name != module:
                            edges.add((module, alias.name))
                    continue
            if name.startswith("email."):
                name = name.split(".")[1]
            elif name == "email":
                for alias in node.names:
                    if alias.name in modules and alias.name != module:
                        edges.add((module, alias.name))
                continue
            if name in modules and name != module:
                edges.add((module, name))

outpath = os.path.join(os.path.dirname(__file__), "email.mdg")
with open(outpath, "w") as f:
    f.write("# module dependency graph of the Python stdlib email package\n")
    f.write("# modules: " + " ".join(modules) + "\n")
    for a, b in sorted(edges):
        f.write(f"{a} {b}\n")

print(f"{len(modules)} modules, {len(edges)} edges -> {outpath}")
