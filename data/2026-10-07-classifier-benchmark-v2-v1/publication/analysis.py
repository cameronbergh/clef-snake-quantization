# SPDX-License-Identifier: Apache-2.0
"""Run byte-preserved frozen statistics with stronger corrected inventory auditor."""
import argparse,json
from pathlib import Path
from .audit import audit
from experiments.classifier_benchmark_v2_v1 import analysis as frozen

def analyze(data,out):
    original=frozen.audit
    try:
        frozen.audit=audit
        return frozen.analyze(data,out)
    finally:frozen.audit=original

def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();s=analyze(a.data,a.out);print(json.dumps({k:{'micro':v['micro']['accuracy'],'macro':v['macro_accuracy']} for k,v in s['conditions'].items()},indent=2))
if __name__=='__main__':main()
