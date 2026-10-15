#!/usr/bin/env python3
"""Verify portable H3 against frozen baselines, or rerun an explicit source checkout."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def verify(baselines):
    from rooflang.programs.models.minimax_h3 import create_workflow
    from rooflang.programs.models.minimax_h3.hardware import A5PRServer
    from rooflang.runtime.metrics_export import _operator_rows, _semantic_stage
    def equal(actual, expected, label):
        if isinstance(expected, (int,float)) and not isinstance(expected,bool):
            ok=math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-8)
        else: ok=actual==expected
        if not ok: raise AssertionError(f'{label}: {actual!r} != {expected!r}')
    for workflow in ('fl2va','ref2va'):
        reference=json.loads((baselines/workflow/'summary.json').read_text())
        model=create_workflow(A5PRServer(),{'workflow':workflow}); results=model.run()
        for key in ('shape_summary','operator_graph_multipliers','transformer_forward_count',
                    'persistent_tensor_bytes','indexed_known_weight_bytes','all_resident_weight_bytes',
                    'all_resident_memory_bytes','peak_stage_memory_bytes','operator_graph_weight_proxy_bytes'):
            equal(model.metadata[key],reference[key],workflow+'/'+key)
        total=0; count=0
        for name,stage in results.items():
            _,metrics=_semantic_stage(name,stage,'MiniMax-H3','A5_pr_server')
            expected=next(s for s in reference['stages'] if s['name']==name)
            for key,value in expected.items():
                equal(metrics[key],value,workflow+'/'+name+'/'+key)
            total+=metrics['execution_time_ms']
            for graph in stage.graphs:
                actual=list(_operator_rows(graph.result,graph.label,'MiniMax-H3','A5_pr_server'))
                with (baselines/workflow/(graph.label+'.csv')).open() as f: old=list(csv.DictReader(f))
                equal(len(actual),len(old),graph.label+'/row_count')
                for new,previous in zip(actual,old):
                    for key in ('node_name','operator'): equal(new[key],previous[key],graph.label+'/'+key)
                    for key in ('layer_idx','execution_time_ms','compute_ms','memory_ms','network_ms',
                                'flops','memory_traffic_bytes','weight_saved'):
                        equal(new[key],float(previous[key] or 0),graph.label+'/'+new['node_name']+'/'+key)
                count+=len(actual)
        equal(total,reference['execution_time_ms'],workflow+'/execution_time_ms')
        print(f'{workflow}: PASS; {count} operator rows; {total:.12f} ms')


SOURCE_RUNNER = r'''
import sys, pathlib, json, hashlib, subprocess, shutil
import yaml
from src.configs import load_task_config
from src.configs.minimax_h3_config import MiniMaxH3MultimodalConfig
from src.configs.task_converter import convert_task_config_to_workload_and_plan
from src.optimal.task import Task
out=pathlib.Path(sys.argv[1]).resolve()
for workflow in ('fl2va','ref2va'):
    config=load_task_config('configs/task/minimax_h3.yaml')
    raw=yaml.safe_load(pathlib.Path('configs/workload/workload_minimax_h3.yaml').read_text())['workload']['multimodal_config']
    raw['workflow']=workflow
    config.workload.multimodal_config=MiniMaxH3MultimodalConfig.from_dict(raw)
    config.output_path=str(out)
    workload,plan=convert_task_config_to_workload_and_plan(config)
    result=Task(workload=workload,plan=plan).execute()
    print(workflow,result.execution_time_ms,result.is_valid)
files=['configs/models/minimax_h3.yaml','configs/workload/workload_minimax_h3.yaml',
       'configs/deployment/minimax_h3.yaml','configs/task/minimax_h3.yaml',
       'configs/chips/A5_pr_server.yaml','configs/op_config/op_config.yaml']
for name in files:
    dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
manifest={'source_repository':'model_Sim','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'baseline_python':sys.version,'files':{f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in files}}
(out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines',type=Path,default=ROOT/'baselines/minimax_h3')
    parser.add_argument('--source',type=Path,help='Optional model_Sim checkout to rerun, without editing it')
    parser.add_argument('--source-python',default=sys.executable,help='Interpreter with the source dependencies installed')
    parser.add_argument('--output',type=Path,help='New/empty directory for fresh source results')
    args=parser.parse_args()
    if args.source:
        if args.output is None: parser.error('--source requires --output')
        output=args.output.resolve()
        if output.exists() and any(output.iterdir()): parser.error('--output must be new or empty')
        source=args.source.resolve()
        if output==source or source in output.parents: parser.error('--output must be outside the source checkout')
        output.mkdir(parents=True,exist_ok=True)
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(source)}
        subprocess.run([args.source_python,'-c',SOURCE_RUNNER,str(output)],cwd=source,env=env,check=True)
        verify(output)
    else:
        if args.output: parser.error('--output is only used with --source')
        verify(args.baselines)

if __name__=='__main__': main()
