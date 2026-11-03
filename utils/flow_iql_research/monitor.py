"""Read-only experiment monitoring; no queue selection or score substitution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    while not (args.root/'STOP_MONITOR').exists():
        subprocess.run([sys.executable,'-m','utils.flow_iql_research.audit',str(args.root)],check=False)
        try:
            status=json.loads((args.root/'scheduler_status.json').read_text())
            audit=json.loads((args.root/'analysis/live_audit.json').read_text())
            print(json.dumps(dict(time=time.time(),running=len(status['running']),pending=status['pending'],
                                 finished=status['finished'],evaluated_runs=sum(r['last_evaluation_step'] is not None for r in audit['rows']))),flush=True)
        except (OSError,ValueError) as error:
            print(type(error).__name__,str(error),flush=True)
        time.sleep(60)


if __name__=='__main__':main()
