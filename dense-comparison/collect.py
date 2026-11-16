"""Audit saved measurements and assemble evidence for PR #67844; no GPU work."""
import ast
import argparse
import dataclasses
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace-root', type=Path, default=HERE.parents[1])
parser.add_argument('--output', type=Path, default=HERE/'dense-comparison')
args = parser.parse_args()
ROOT = args.workspace_root.resolve()
OUT = args.output.resolve()
OUT.mkdir(exist_ok=True)

def read(p):
    return json.loads(p.read_text())

def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

comparisons = ROOT / 'vllm-bench-collection/comparisons/gfx1201/qwen3.8-27b-fp8'
runs = {}
source_checks = []
common = []
for pair, folder in [('bf16', 'octave-comparison-20261109'), ('fp8', 'fp8-segmented-vs-octave-20261110')]:
    comparison = read(comparisons / folder / 'comparison.json')
    for arm in ['baseline', 'candidate']:
        archived = comparison[arm]
        src = Path(archived['folder'])
        report = src / 'agent-multiturn-benchmarks.json'
        assert sha(report) == archived['sha256'] == read(src / 'success.json')['report_sha256']
        raw = read(report)
        name = pair if arm == 'baseline' else 'octave_' + pair + '_pair'
        common.append(raw['config']['spec'])
        records = {}
        results = {}
        for b in raw['benchmarks']:
            streams = int(b['config']['strategy']['streams'])
            requests = b['requests']
            assert not requests.get('errored') and not requests.get('incomplete')
            rows = requests['successful']
            assert len(rows) == streams * 8
            assert all(r['output_tokens'] == 512 for r in rows)
            conversations = {r['info']['conversation_id'] for r in rows}
            assert len(conversations) == streams
            for c in conversations:
                assert sorted(r['info']['turn_index'] for r in rows if r['info']['conversation_id'] == c) == list(range(8))
            window = max(r['request_end_time'] for r in rows) - min(r['request_start_time'] for r in rows)
            result = {
                'requests': len(rows),
                'aggregate_tps': sum(r['output_tokens'] for r in rows) / window,
                'output_tps': statistics.mean(r['output_tokens_per_second'] for r in rows),
                'ttft_ms': statistics.mean(r['time_to_first_token_ms'] for r in rows),
                'e2e_s': statistics.mean(r['request_latency'] for r in rows),
                'itl_ms': statistics.mean(r['inter_token_latency_ms'] for r in rows),
                'window_s': window,
            }
            for k, v in result.items():
                assert math.isclose(v, archived['results'][str(streams)][k], rel_tol=1e-10), (name, streams, k)
            results[str(streams)] = result
            fields = ['request_start_time', 'request_end_time', 'request_latency', 'prompt_tokens', 'output_tokens', 'output_tokens_per_second', 'time_to_first_token_ms', 'inter_token_latency_ms']
            records[str(streams)] = [{**{k:r[k] for k in fields}, 'conversation_id':r['info']['conversation_id'], 'turn_index':r['info']['turn_index']} for r in rows]
        assert sorted(map(int, results)) == [1, 2, 4, 8]
        log = (src / 'server.log').read_text()
        capacity = int(re.search(r'GPU KV cache size: ([\d,]+) tokens', log)[1].replace(',', ''))
        assert capacity == archived['capacity_tokens']
        assert not archived['timed_jit_monitor_warnings']
        assert archived['metrics_delta']['vllm:num_preemptions_total'] == 0
        inspection = archived['inspection']
        assert inspection['kv_cache_memory_bytes'] == 8393906176
        runs[name] = {
            'backend': archived['backend'], 'capacity_tokens':capacity,
            'attention_block_size': archived['attention_block_size'],
            'prefix_hit_rate': archived['metrics_delta']['vllm:prefix_cache_hits_total'] / archived['metrics_delta']['vllm:prefix_cache_queries_total'],
            'vram_gib_after': {k:v['vram']/2**30 for k,v in archived['hardware_after_measured'].items()},
            'results': results,
        }
        write(name + '-requests.json', records)
        write(name + '-provenance.json', {
            'original_report': str(report.relative_to(ROOT)), 'original_report_sha256':sha(report),
            'setup':read(src/'setup.json'), 'workers':read(src/'inspection.json'),
            'workload':read(src/'agent-multiturn-benchmarks-workload.json'),
            'server_log_sha256':sha(src/'server.log'),
            'capacity_log_lines':[line for line in log.splitlines() if 'GPU KV cache size:' in line or 'reserved 7.82 GiB' in line],
            'timed_jit_warnings':archived['timed_jit_monitor_warnings'],
            'metrics_delta':archived['metrics_delta'],
        })
        source_checks.append({'run':name,'report_sha256':sha(report),'comparison_sha256':sha(comparisons/folder/'comparison.json'),'requests':sum(v['requests'] for v in results.values())})
for field in ['data','profile','constraints','seed']:
    assert all(c[field] == common[0][field] for c in common), field

# Run the PR's actual pure-Python format definition without importing GPU libraries.
fmt_path = ROOT/'vllm-rdna4-octave-kv-pr/vllm/v1/attention/ops/rocm_octave.py'
tree = ast.parse(fmt_path.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OctaveFormat' or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ['HADAMARD_BLOCK','ROPE_BITS','NOPE_BITS'] for t in n.targets)]
namespace = {'dataclass':dataclasses.dataclass}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(fmt_path), 'exec'), namespace)
slots = {'bf16':2*256*2,'fp8':2*256}
for kind in ['octave_k3v4','octave_k3v3','octave_k3v3_compact']:
    slots[kind] = namespace['OctaveFormat'].from_cache_dtype(kind,256,64).slot_bytes
assert slots == {'bf16':1024,'fp8':512,'octave_k3v4':248,'octave_k3v3':216,'octave_k3v3_compact':196}

# Independently confirm the ordinary TRITON_ATTN FP8 capacity from the accuracy run.
accuracy = {}
for name,folder,expected in [('bf16','bf16-retry',236700),('fp8','fp8',446202),('octave','octave',786432)]:
    src=ROOT/'reports/rdna4-octave-gsm8k-20261110/arms'/folder
    lines=(src/'server.log').read_text().splitlines()
    line=next(x for x in lines if 'GPU KV cache size:' in x)
    count=int(re.search(r'GPU KV cache size: ([\d,]+)',line)[1].replace(',',''))
    assert count==expected
    accuracy[name]={'capacity_tokens':count,'capacity_log_line':line,'server_log_sha256':sha(src/'server.log'),'workers':read(src/'worker-settings.json')}
write('accuracy-capacity-provenance.json',accuracy)
write('summary.json',{'historical':True,'new_gpu_measurements':False,'model':'Qwen/Qwen3.8-27B-FP8','hardware':'2 x Radeon AI PRO R9700 (gfx1201), TP2','kv_budget_bytes_per_gpu':8393906176,'kv_budget_gib_per_gpu':8393906176/2**30,'slots_bytes_per_token_per_kv_head':slots,'format_source_sha256':sha(fmt_path),'runs':runs,'input_checks':source_checks})
print(json.dumps({'checked_requests':sum(x['requests'] for x in source_checks),'slots':slots,'runs':{k:{'capacity':v['capacity_tokens'],'tps':[round(r['aggregate_tps'],2) for r in v['results'].values()]} for k,v in runs.items()}},indent=2))
