"""Regenerate the PR figure and verify metrics from the included request rows."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'summary.json').read_text())
runs = data['runs']
for name, run in runs.items():
    requests = json.loads((ROOT/(name+'-requests.json')).read_text())
    for streams, rows in requests.items():
        window = max(r['request_end_time'] for r in rows)-min(r['request_start_time'] for r in rows)
        value = sum(r['output_tokens'] for r in rows)/window
        assert math.isclose(value, run['results'][streams]['aggregate_tps'], rel_tol=1e-12)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#172033','axes.edgecolor':'#cbd5e1','savefig.facecolor':'white'})
colors = ['#3778b5','#db8c2c','#168a77']
fig, (left,right) = plt.subplots(1,2,figsize=(13.5,5.4),gridspec_kw={'width_ratios':[1,1.25]})
fig.subplots_adjust(left=.09,right=.98,top=.76,bottom=.22,wspace=.34)
fig.suptitle('Octave fits more KV tokens; this workload runs slower',x=.06,y=.98,ha='left',fontsize=19,weight='bold')
fig.text(.06,.90,'Qwen3.8-27B-FP8  ·  2 × Radeon AI PRO R9700  ·  TP2  ·  Same 7.82 GiB KV budget per GPU',fontsize=11,color='#475569')
names=['bf16','fp8','octave_bf16_pair']
labels=['BF16\nTriton','FP8\nSegmented Triton','Octave\nK3V4']
bars=left.bar(range(3),[runs[n]['capacity_tokens'] for n in names],color=colors,width=.63)
left.set_xticks(range(3),labels)
left.set_ylim(0,900000)
left.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:.0f}K'))
left.set_ylabel('Reported KV token capacity')
left.set_title('Cache capacity ↑',loc='left',pad=15)
left.set_axisbelow(True); left.grid(axis='y',color='#e2e8f0')
for bar in bars:
    left.text(bar.get_x()+bar.get_width()/2,bar.get_height()+17000,f'{bar.get_height():,.0f}',ha='center',weight='bold',fontsize=11)
right.set_title('Serving throughput ↑',loc='left',pad=15)
for name,label,color,style,marker in [
    ('bf16','BF16 · Triton',colors[0],'-','o'),
    ('fp8','FP8 · segmented Triton',colors[1],'-','s'),
    ('octave_bf16_pair','Octave · BF16 pair',colors[2],'-','o'),
    ('octave_fp8_pair','Octave · FP8 pair','#284f46','--','x'),
]:
    right.plot([1,2,4,8],[runs[name]['results'][str(s)]['aggregate_tps'] for s in [1,2,4,8]],label=label,color=color,linestyle=style,marker=marker,linewidth=2,markersize=6)
right.set_xticks([1,2,4,8]); right.set_ylim(0,170)
right.set_xlabel('Concurrent conversations');right.set_ylabel('Aggregate output tokens / second')
right.grid(color='#e2e8f0');right.set_axisbelow(True)
right.legend(frameon=False,fontsize=9,loc='upper left')
fig.text(.06,.085,'Historical measurements, Nov 9–10: two separate matched pairs; one warmed sweep per arm, 120 requests each.',fontsize=10,color='#475569')
fig.text(.06,.045,'8 turns × 2,048 new prompt tokens + 512 output tokens. Includes prefill. Capacity is not total VRAM savings.',fontsize=10,color='#475569')
fig.savefig(ROOT/'capacity-throughput.png',dpi=160)
print('Verified all 16 throughput points and rendered capacity-throughput.png')
