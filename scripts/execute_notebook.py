"""Execute the walkthrough, optionally without kernel sockets on restricted hosts.

Default mode uses nbclient/Jupyter. --in-process runs the identical code cells
in an IPython InteractiveShell, capturing rich display output and exceptions.
It is intended for this notebook's pure-Python cells, not arbitrary notebooks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nbformat


def execute_in_process(path: Path) -> None:
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        count += 1
        compile(cell.source, f"{path.name}:cell{index}", "exec")
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=False)
        cell.execution_count = count
        cell.outputs = []
        for name, text in (("stdout", captured.stdout), ("stderr", captured.stderr)):
            if text:
                cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
        if not result.success:
            nbformat.write(notebook, path)
            raise RuntimeError(f"Notebook cell {index} failed:\n{captured.stdout}\n{captured.stderr}") from (result.error_before_exec or result.error_in_exec)
        print(f"Executed cell {index} ({count})", flush=True)
    notebook.metadata["execution_verification"] = {
        "method": "IPython InteractiveShell, socket-free in-process execution",
        "python": sys.version.split()[0],
        "code_cells": count,
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args()
    if args.in_process:
        execute_in_process(args.notebook.resolve())
    else:
        from nbclient import NotebookClient

        notebook = nbformat.read(args.notebook, as_version=4)
        NotebookClient(notebook, timeout=600, kernel_name="python3").execute()
        nbformat.write(notebook, args.notebook)
    print(f"Notebook completed: {args.notebook}")


if __name__ == "__main__":
    main()
