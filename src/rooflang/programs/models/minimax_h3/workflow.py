"""MiniMax-H3 semantic stages on RoofLang's native simulation runtime."""
from functools import partial
from pathlib import Path
import json
import math
from copy import deepcopy
from rooflang.language.hardware.component import Compute
from rooflang.programs.experiments.simulation import simulate
from rooflang.programs.experiments.workflow import Workflow
from rooflang.runtime.stages import GraphResult, StageResult
from .config import MiniMaxH3MultimodalConfig
from .bridge import build_component, component_nodes

ROOT = Path(__file__).parent

def _load(name):
    return json.loads((ROOT/name).read_text())

def resolve_config(overrides):
    overrides=deepcopy(overrides)
    for key in ('batch_size','dp','tp','cp','pp'):
        value = overrides.pop(key, 1)
        if type(value) is not int or value != 1:
            raise ValueError(f'MiniMax-H3 currently requires {key}=1')
    if overrides.pop('compute_dtype','fp16') != 'fp16':
        raise ValueError('MiniMax-H3 reproduction supports compute_dtype=fp16')
    sp=overrides.pop('sp',4)
    if type(sp) is not int or sp not in (1,2,4,8):
        raise ValueError('MiniMax-H3 SP must be one of 1,2,4,8')
    data=_load('workload.json')
    workflow=overrides.get('workflow',data['workflow'])
    if workflow not in data['workflows']:
        raise ValueError('MiniMax-H3 workflow must be fl2va or ref2va')
    for key in ('text_condition_rows','keyframes','references'):
        if key in overrides:
            data['workflows'][workflow][key]=overrides.pop(key)
    target=data['target'].copy()
    for key in ('frames','height','width'):
        if key in overrides: target[key]=overrides.pop(key)
    if 'target' in overrides: target.update(overrides.pop('target'))
    allowed={'workflow','num_inference_steps','assume_stage_unload','workflows','type'}
    unknown=set(overrides)-allowed
    if unknown: raise ValueError(f'Unknown MiniMax-H3 options: {sorted(unknown)}')
    data.update(overrides); data['target']=target
    for key in ('frames', 'height', 'width'):
        if type(data['target'][key]) is not int:
            raise ValueError(f'target.{key} must be an integer')
    if type(data['num_inference_steps']) is not int:
        raise ValueError('num_inference_steps must be an integer')
    if type(data['assume_stage_unload']) is not bool:
        raise ValueError('assume_stage_unload must be a boolean')
    if type(data['workflows'][workflow]['text_condition_rows']) is not int:
        raise ValueError('text_condition_rows must be an integer')
    cfg=MiniMaxH3MultimodalConfig.from_dict(data)
    return cfg,sp,data

def video_vae_tiles(height,width):
    return min(height,256),min(width,256),math.prod(max(1,math.ceil((x-64)/192)) for x in (height,width))

def _persistent(mm):
    rows=mm['packed_rows']
    return {'prompt_embeds':2*mm['text_rows']*5120,
        'video_packed_rows':2*(mm['condition_video_rows']+mm['target_video_rows'])*96,
        'audio_packed_rows':2*(mm['condition_audio_rows']+mm['target_audio_rows'])*32,
        'position_ids':8*rows*3,'token_tags':8*rows,'packed_indices':8*rows,
        'all_step_timestep_indices':8*mm['num_inference_steps']*rows}

def _graph_stats(result):
    kernels=dict.fromkeys(e.kernel for e in result.trace)
    return {'weight':sum(k.csv_metadata.get('source_weight_saved',0) for k in kernels),
            'activation':max((k.csv_metadata.get('activation_peak_bytes_proxy',0) for k in kernels),default=0)}

def create_workflow(hardware, config):
    cfg,sp,resolved=resolve_config(config)
    architecture=_load('architecture.json'); mm=cfg.runtime_multimodal_config()
    devices=[n for n in hardware.nodes if isinstance(n,Compute) and n.kind=='gpu']
    if len(devices)<sp: raise ValueError(f'MiniMax-H3 SP={sp} requires at least {sp} devices')
    for device in devices[:sp]:
        missing={'fp16:cube','fp16:vector'}-set(device.tflops)
        if missing: raise ValueError('MiniMax-H3 requires explicit fp16:cube and fp16:vector throughput; use A5_pr_server or provide both')
        if any(not math.isfinite(device.tflops[key]) or device.tflops[key] <= 0 for key in ('fp16:cube', 'fp16:vector')):
            raise ValueError('MiniMax-H3 Cube/Vector throughput must be finite and positive')
        fabric = hardware.find_fabric(device, hardware.find_local_memory(device))
        read_bw, write_bw = fabric.dst_to_src_bandwidth_gbs, fabric.src_to_dst_bandwidth_gbs
        if not math.isfinite(read_bw) or read_bw <= 0 or not math.isclose(read_bw, write_bw):
            raise ValueError('MiniMax-H3 aggregate traffic requires equal positive HBM read/write bandwidth')
    transformer='transformer' if cfg.workflow=='fl2va' else 'transformer_ref'
    # component, label, sequence rows, updates, repeat count, distributed
    jobs={'qwen_vision_encode':[]}
    for kind in ('image','video'):
        segments=[g['height']*g['width'] for g in mm['qwen_visual_grids'] if g['kind']==kind for _ in range(g['temporal'])]
        if segments: jobs['qwen_vision_encode'].append(('qwen_vision',f'qwen_vision_{kind}',sum(segments),{'qwen_attention_segments':segments},1,True))
    jobs['qwen_text_condition']=[('qwen_text','qwen_text_condition',mm['text_rows'],{},1,True)]
    jobs['condition_vae_encode']=[]
    for index,condition in enumerate(mm['visual_conditions']):
        h,w,tiles=video_vae_tiles(condition['height'],condition['width'])
        chunks=1 if condition['frames']==1 else math.ceil(condition['frames']/17)
        updates={'visual_conditions':[{'frames':1 if condition['frames']==1 else 17,'height':h,'width':w}]}
        jobs['condition_vae_encode'].append(('video_vae_encoder',f'condition_video_{index}',1,updates,chunks*tiles,False))
    for index,length in enumerate(mm['audio_reference_latents']):
        jobs['condition_vae_encode'].append(('audio_vae_encoder',f'condition_audio_{index}',1,{'audio_reference_latents':[length]},1,False))
    jobs['transformer_text_refiner']=[('transformer_text_refiner','transformer_text_refiner',mm['text_rows'],{},cfg.num_inference_steps,False)]
    jobs['transformer_denoise']=[(transformer,'transformer_denoise',mm['packed_rows'],{},cfg.num_inference_steps,True)]
    h,w,tiles=video_vae_tiles(mm['target_height'],mm['target_width'])
    calls=(math.ceil((mm['target_latent_frames']+3)/5)-1)*tiles
    jobs['video_vae_decode']=[('video_vae_decoder','video_vae_decode',7*(h//16)*(w//16)+5,{},calls,False)]
    jobs['audio_vae_decode']=[('audio_vae_decoder','audio_vae_decode',1,{},1,False)]

    def run_stage(name):
        result=StageResult(); weights={}; peak=0
        for component,label,seq,updates,multiplier,distributed in jobs[name]:
            graph,placement=build_component(component,architecture,{**mm,**updates},seq,hardware,sp if distributed else 1)
            simulation=simulate(graph,placement,hardware)
            result.graphs.append(GraphResult(label,simulation,multiplier))
            stats=_graph_stats(simulation)
            weights.setdefault(component,stats['weight']);peak=max(peak,stats['activation'])
        result.metrics.update(total_weight_bytes=sum(weights.values()),peak_activation_bytes=peak,
            timing_status='modeled_main_operators_only')
        return result

    def summarize(results):
        persistent=_persistent(mm); ptotal=sum(persistent.values())
        complete=set(results)==set(jobs)
        summary={'operator_graph_multipliers':{g.label:g.multiplier for s in results.values() for g in s.graphs},
            'operator_graph_stages':{g.label:name for name,s in results.items() for g in s.graphs},
            'request_count':1 if complete else None,
            'transformer_forward_count':cfg.num_inference_steps if 'transformer_denoise' in results else 0,
            'persistent_tensor_bytes':persistent,'persistent_tensor_total_bytes':ptotal,
            'shape_summary':mm['shape_summary'],'transformer_component':transformer}
        if not complete:
            known = architecture['indexed_weight_bytes']
            phase_weights = {'qwen_vision_encode': known['qwen_conditioner'],
                'qwen_text_condition': known['qwen_conditioner'],
                'condition_vae_encode': known['video_vae'],
                'transformer_text_refiner': known[transformer],
                'transformer_denoise': known[transformer],
                'video_vae_decode': known['video_vae']}
            if 'audio_vae_decode' in results and mm['audio_reference_latents']:
                audio_nodes = component_nodes('audio_vae_encoder', architecture,
                    {**mm, 'audio_reference_latents': mm['audio_reference_latents'][:1]}, 1)
                phase_weights['audio_vae_decode'] = (results['audio_vae_decode'].metrics['total_weight_bytes']
                    + sum(n.operator.get_memory_details().get('weight_saved', 0) for n in audio_nodes))
            for name, stage in results.items():
                stage.metrics['peak_memory_bytes'] = (
                    phase_weights.get(name, stage.metrics['total_weight_bytes'])
                    + stage.metrics['peak_activation_bytes'] + ptotal)
            peak_memory = max(s.metrics['peak_memory_bytes'] for s in results.values())
            capacity = min(hardware.find_local_memory(d).capacity_gb*1e9 for d in devices[:sp])
            valid = peak_memory <= capacity
            summary.update(peak_stage_memory_bytes=peak_memory, fits_peak_stage_memory=valid,
                memory_validity_basis='selected_stage_memory', memory_scope='selected_stages_only',
                is_valid=valid, fail_reason='' if valid else 'Selected-stage working-set proxy exceeds HBM capacity',
                all_resident_weight_bytes=None, all_resident_memory_bytes=None, fits_all_resident_memory=None,
                operator_graph_weight_proxy_bytes=None, audio_vae_weight_proxy_bytes=None,
                packed_rows_per_second=None, transformer_processed_rows_per_second=None)
            return summary
        indexed={k:architecture['indexed_weight_bytes'][k] for k in ('qwen_conditioner',transformer,'video_vae')}
        audio_enc=max((_graph_stats(g.result)['weight'] for g in results['condition_vae_encode'].graphs if g.label.startswith('condition_audio')),default=0)
        audio=audio_enc+results['audio_vae_decode'].metrics['total_weight_bytes']
        peak=lambda name: results[name].metrics['peak_activation_bytes']
        resident=sum(indexed.values())+audio
        all_memory=resident+ptotal+max(peak(name) for name in results)
        stage_peaks={
            'qwen_vision_encode':indexed['qwen_conditioner']+peak('qwen_vision_encode')+ptotal,
            'qwen_text_condition':indexed['qwen_conditioner']+peak('qwen_text_condition')+ptotal,
            'condition_vae_encode':indexed['video_vae']+peak('condition_vae_encode')+ptotal,
            'transformer_text_refiner':indexed[transformer]+peak('transformer_text_refiner')+ptotal,
            'transformer_denoise':indexed[transformer]+peak('transformer_denoise')+ptotal,
            'video_vae_decode':indexed['video_vae']+peak('video_vae_decode')+ptotal,
            'audio_vae_decode':audio+peak('audio_vae_decode')+ptotal}
        for name,value in stage_peaks.items(): results[name].metrics['peak_memory_bytes']=value
        peak_memory=max(stage_peaks.values())
        capacity=min(hardware.find_local_memory(d).capacity_gb*1e9 for d in devices[:sp])
        fits_peak=peak_memory<=capacity;fits_all=all_memory<=capacity
        valid=fits_peak if cfg.assume_stage_unload else fits_all
        time_ms=sum(g.result.total_time_us/1000*g.multiplier for s in results.values() for g in s.graphs)
        rps=1000/time_ms
        summary.update(memory_scope='selected_workflow', indexed_known_weight_bytes=indexed,audio_vae_weight_proxy_bytes=audio,
            operator_graph_weight_proxy_bytes=sum(s.metrics['total_weight_bytes'] for s in results.values()),
            all_resident_weight_bytes=resident,all_resident_memory_bytes=all_memory,
            peak_stage_memory_bytes=peak_memory,fits_peak_stage_memory=fits_peak,fits_all_resident_memory=fits_all,
            memory_validity_basis='peak_stage_memory' if cfg.assume_stage_unload else 'all_resident_memory',
            is_valid=valid,fail_reason='' if valid else 'Analytical working-set proxy exceeds HBM capacity',
            packed_rows_per_second=rps*mm['packed_rows'],
            transformer_processed_rows_per_second=rps*mm['packed_rows']*cfg.num_inference_steps)
        return summary

    return Workflow({name:partial(run_stage,name) for name in jobs},{
        'workflow':cfg.workflow,'batch_size':1,'prompt_len':mm['text_rows'],'total_cards':sp,
        'main_sp':sp,'main_cp':1,'main_tp':1,'main_dp':1,'compute_dtype':'fp16','weight_storage_dtype':'fp16',
        'num_diffusion_steps':cfg.num_inference_steps,'cfg':1,'resolved_shape_summary':mm['shape_summary'],
        'assume_stage_unload':cfg.assume_stage_unload,
        '_resolved_config':{'workload':resolved,'architecture':architecture,'sp':sp,'batch_size':1,'compute_dtype':'fp16'},
        'source_commit':_load('SOURCES.json')['source_commit'],
        'latency_scope':'modeled_main_operators_only','compute_scope':'critical_rank_per_stage_not_global_device_sum',
        'timing_backend':'rooflang_discrete_event_simulator','end_to_end_timing_complete':False,
        'memory_status':'indexed_weights_plus_persistent_plus_peak_operator_activation_proxy',
        'capacity_verdict_is_provisional':True},summarize)
