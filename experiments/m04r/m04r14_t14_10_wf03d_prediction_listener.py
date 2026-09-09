"""Wait once for the D4 producer, then launch its canonical verifier."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
from typing import Sequence

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_prediction_store as producer


class PredictionListenerError(RuntimeError): pass


def _wait_once(pid: int)->str:
    try: descriptor=os.pidfd_open(pid,0)
    except ProcessLookupError: return "already_exited"
    except AttributeError as error: raise PredictionListenerError("pidfd_open is unavailable") from error
    try:
        waiter=select.poll(); waiter.register(descriptor,select.POLLIN); events=waiter.poll()
        if not events: raise PredictionListenerError("process listener returned without event")
    finally: os.close(descriptor)
    return "process_exit_observed"


def execute(repository: Path, producer_pid: int)->Path:
    repository=repository.resolve(strict=True)
    if producer._git(repository,"status","--porcelain","--untracked-files=all"):
        raise PredictionListenerError("listener requires clean worktree")
    _wait_once(producer_pid)
    seal_path=repository/producer.OUTPUT_RELATIVE/"SEALED.json"
    if not seal_path.is_file():
        raise PredictionListenerError("producer exited without canonical prediction seal")
    seal=base._read(seal_path)
    if not producer._valid_seal(seal,timing=True) or seal.get("passed") is not True:
        raise PredictionListenerError("canonical prediction seal is invalid")
    command=[str(repository/".venv/bin/python"),"-m","experiments.m04r.verify_m04r14_t14_10_wf03d_prediction_store","--repository",str(repository)]
    result=subprocess.run(command,cwd=repository,text=True,capture_output=True,check=False)
    if result.returncode:
        raise PredictionListenerError(result.stderr.strip() or result.stdout.strip() or "verifier failed")
    verified=repository/producer.VERIFICATION_RELATIVE/"VERIFIED.json"
    if not verified.is_file(): raise PredictionListenerError("verifier exited without receipt")
    return verified


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository",required=True,type=Path); parser.add_argument("--producer-pid",required=True,type=int); args=parser.parse_args(argv)
    path=execute(args.repository,args.producer_pid); print(json.dumps({"status":"verified","path":str(path)})); return 0


if __name__=="__main__": raise SystemExit(main())
