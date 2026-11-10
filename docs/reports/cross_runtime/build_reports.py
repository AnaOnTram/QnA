#!/usr/bin/env python3
"""Reconcile retained measurements and render the cross-runtime report evidence.

No inference or attacks. From the archive root:
    python docs/reports/cross_runtime/build_reports.py
Requires matplotlib and numpy. Report prose is maintained in the Markdown files.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
REPO = REPORTS.parents[1]
FIGURES = HERE / 'figures'
FIGURES.mkdir(exist_ok=True)
BLUE, ORANGE = '#2166a5', '#c26b16'
FAMILY_MODEL = {'ResNetV2-50': 'resnetv2_50', 'MobileNetV2': 'mobilenetv2'}
MAIN = ['fgsm@1', 'pgd@1', 'pgd_stronger@1', 'pgd@8', 'square@4', 'square@8']
HEADERS = ['FGSM 1', 'PGD 1', 'PGD+ 1', 'PGD 8', 'Square 4', 'Square 8']
sources = {}

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    path = path.resolve()
    sources[str(path.relative_to(REPO))] = sha(path)
    return json.loads(path.read_text())

def save(name, data):
    (HERE/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    denom = 1+z*z/n
    mid = (p+z*z/(2*n))/denom
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0,100*(mid-radius)), min(100,100*(mid+radius))]

def metric(k, n):
    assert isinstance(k, int) and isinstance(n, int) and 0 <= k <= n and n > 0
    return dict(fooled=k, eligible=n, success_pct=100*k/n, wilson95_pct=wilson(k,n))

def link(path, start):
    return os.path.relpath(path, start)

# The archive manifest resolves original filenames; it is not a measurement
# input. Hashing it here would create a cycle when it catalogs these outputs.
manifest = json.loads((REPORTS/'manifest.json').read_text())
sources[str(Path(__file__).resolve().relative_to(REPO))] = sha(Path(__file__))
source_map = {str(Path('/home/ross/QA')/a['source_path_relative_to_qa']): REPORTS/a['path']
              for a in manifest['files'] if a.get('source_path_relative_to_qa')}
master_path = REPORTS/'mobilenetv2/TPU/evasion/results/comparisons/all_deployments.json'
master = read(master_path)
records = []
for item in master:
    original = Path(item['path'])
    if original.is_relative_to('/home/ross/QA/model-weights'):
        source = REPO/original.relative_to('/home/ross/QA/model-weights')
    elif item['family'] == 'ResNetV2-50' and item['label'] == 'QNN HTP INT8':
        source = HERE/'inputs/resnetv2_50_qnn_summary.json'
    else:
        source = source_map[str(original)]
    summary = read(source)
    assert sha(source) == item['sha256'], source
    legacy = item['family'] == 'ResNetV2-50' and item['label'] == 'QNN HTP INT8'
    clean = summary['clean']['npu'] if legacy else summary['clean']
    assert clean['n'] == item['n'] == 100 and clean['correct'] == item['clean_correct']
    conditions = {}
    for attack in summary['attacks']:
        if legacy and attack.get('model') != 'npu':
            continue
        key = f"{attack['attack']}@{attack['eps_255']:g}"
        assert key not in conditions
        k, n = int(attack['fooled']), int(attack['clean_correct'])
        assert n == clean['correct']
        retained = item['conditions'][key]
        assert (k,n) == (retained['fooled'],retained['n']), (source,key)
        assert abs(retained['pct']-100*k/n) < 1e-8
        conditions[key] = metric(k,n) | dict(robust_correct=clean['correct']-k,
            robust_accuracy_pct=clean['correct']-k)
    assert set(conditions) == set(item['conditions'])
    records.append(dict(model=FAMILY_MODEL[item['family']], family=item['family'],
        deployment=item['label'].replace(' (new)',''), clean_correct=clean['correct'],
        population_n=100, source=str(source.relative_to(REPORTS)), source_sha256=sha(source),
        conditions=conditions))
assert Counter(a['model'] for a in records) == {'mobilenetv2':20,'resnetv2_50':19}
lookup = {(a['model'],a['deployment']):a for a in records}

# Verify image identity, without claiming identical model-specific crop tensors.
resnet_samples = read(HERE/'inputs/resnetv2_50_samples.json')
mobile_samples = read(REPORTS/'mobilenetv2/TPU/results/samples.json')
identities = lambda rows: [(x['index'],x['sha256'],x['label']) for x in rows]
assert identities(resnet_samples) == identities(mobile_samples)
assert len(resnet_samples) == len({x['sha256'] for x in resnet_samples}) == 100

core = {model:read(REPORTS/f'{model}/evasion_coreml/results/comparisons/matrix.json')
        for model in FAMILY_MODEL.values()}
seeds = {model:read(REPORTS/f'{model}/evasion_coreml/results/supplementary/square_seeds.json')
         for model in FAMILY_MODEL.values()}
for model, data in core.items():
    assert data['validation']['passed']
    for arm,a in data['arms'].items():
        precision,unit = arm.split('_',1)
        label=f"Core ML {precision.upper()} "+{'cpu_only':'CPU','cpu_and_gpu':'CPU+GPU','cpu_and_ne':'CPU+NE'}[unit]
        own=lookup[model,label]
        assert own['clean_correct'] == a['clean']['correct']
        for key,m in a['conditions'].items():
            assert (m['fooled'],m['n']) == (own['conditions'][key]['fooled'],own['conditions'][key]['eligible'])
        for key,m in a['shared'].items():
            assert m['n'] == data['shared_clean_n']
            assert 0 <= m['fooled'] <= m['n']
            assert m['fooled'] <= own['conditions'][key]['fooled']

old_resnet = read(REPORTS/'resnetv2_50/evasion_onnx/results/comparisons/timm_matrix.json')
mobile_rknn = read(REPORTS/'mobilenetv2/evasion_rknn/results/comparisons/matrix.json')
mobile_trt = read(REPORTS/'mobilenetv2/evasion_trt/results/comparisons/matrix.json')

# Compare quantization on each study's common clean-correct mask. These masks
# differ across studies/models; their rate differences are descriptive interactions.
effects=[]
for key in ['pgd@1','pgd_stronger@1','square@8']:
    attack,eps = key.split('@')
    a = old_resnet['matrix']['rknn_fp16'][attack][eps]
    b = old_resnet['matrix']['rknn_int8'][attack][eps]
    assert a['eligible'] == b['eligible'] == old_resnet['shared_clean_n'] == 71
    effects.append(dict(model='resnetv2_50',pair='RKNN FP16 → INT8',condition=key,
        float_fooled=a['evaded'],quantized_fooled=b['evaded'],shared_n=a['eligible'],
        reduction_pp=100*(a['evaded']-b['evaded'])/a['eligible'], source='resnetv2_50/evasion_onnx/results/comparisons/timm_matrix.json'))
    a=mobile_rknn['models']['rknn_fp16']['conditions'][f'{attack}_eps{eps}']
    b=mobile_rknn['models']['rknn_int8']['conditions'][f'{attack}_eps{eps}']
    assert a['shared_n'] == b['shared_n'] == mobile_rknn['shared_clean_correct'] == 80
    effects.append(dict(model='mobilenetv2',pair='RKNN FP16 → INT8',condition=key,
        float_fooled=a['shared_fooled'],quantized_fooled=b['shared_fooled'],shared_n=a['shared_n'],
        reduction_pp=100*(a['shared_fooled']-b['shared_fooled'])/a['shared_n'],source='mobilenetv2/evasion_rknn/results/comparisons/matrix.json'))
    for model in FAMILY_MODEL.values():
        for unit,label in [('cpu_only','CPU'),('cpu_and_gpu','CPU+GPU'),('cpu_and_ne','CPU+NE')]:
            a=core[model]['arms'][f'fp16_{unit}']['shared'][key]
            b=core[model]['arms'][f'int8_{unit}']['shared'][key]
            assert a['n']==b['n']==core[model]['shared_clean_n']
            effects.append(dict(model=model,pair=f'Core ML {label}: FP16 → W8A16',condition=key,
                float_fooled=a['fooled'],quantized_fooled=b['fooled'],shared_n=a['n'],
                reduction_pp=100*(a['fooled']-b['fooled'])/a['n'], source=f'{model}/evasion_coreml/results/comparisons/matrix.json'))

seed_results=[]
for model,data in seeds.items():
    for arm,a in data['arms'].items():
        for eps,cell in a['square'].items():
            counts=[x['fooled'] for x in cell['runs']]
            rates=[100*k/a['clean_correct'] for k in counts]
            assert len(counts)==3 and all(0<=k<=a['clean_correct'] for k in counts)
            seed_results.append(dict(model=model,arm=arm,eps_255=int(eps),n=a['clean_correct'],
                seeds=[x['seed'] for x in cell['runs']],fooled_by_seed=counts,
                mean_pct=100*sum(counts)/(len(counts)*a['clean_correct']),
                min_pct=min(rates),max_pct=max(rates)))

common_labels=[x['deployment'] for x in records if x['model']=='resnetv2_50' and
              ('mobilenetv2',x['deployment']) in lookup]
common_labels.append('TensorRT INT8 (different recipe/hardware)')
cross=[]
for label in common_labels:
    rlabel='TensorRT INT8 Q/DQ' if label.startswith('TensorRT INT8 (') else label
    mlabel='TensorRT INT8' if label.startswith('TensorRT INT8 (') else label
    a,b=lookup['resnetv2_50',rlabel],lookup['mobilenetv2',mlabel]
    for key in MAIN:
        x,y=a['conditions'].get(key),b['conditions'].get(key)
        cross.append(dict(deployment=label,condition=key,resnet=x,mobilenet=y,
            mobile_minus_resnet_pp=None if x is None or y is None else y['success_pct']-x['success_pct']))

save('normalized_results.json',dict(as_of='2026-11-10',population_n=100,
    cohort_identity_verified=True,preprocessing_identical_between_models=False,
    rate_definition='fooled / each deployment own clean-correct',
    robust_accuracy_definition='(clean_correct - fooled) / 100',deployments=records))
save('quantization_effects.json',effects)
save('cross_model_comparisons.json',cross)
save('square_seed_results.json',seed_results)
save('source_index.json',dict(inputs=sources,methods=dict(normalization='Read each pinned source summary; reconcile all counts with the historical registry; recompute rates and Wilson intervals.',
    shared_masks='Reuse retained common-clean matrix counts. Do not combine masks or fabricate an all-deployment common cohort.',
    intervals='95% Wilson per cell; seed ranges are observed ranges of three extra attack seeds, not confidence intervals.',
    differences='Percentage-point differences are descriptive. Paired difference uncertainty is unavailable without per-image discordances.',
    experiment_status='Reporting only; no hardware inference or attacks performed.')))

with (HERE/'all_conditions.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['model','deployment','condition','clean_correct','fooled','eligible','success_pct','wilson95_low_pct','wilson95_high_pct','robust_correct','robust_accuracy_pct','source','source_sha256'],lineterminator='\n')
    writer.writeheader()
    for a in records:
        for condition,m in a['conditions'].items():
            writer.writerow(dict(model=a['model'],deployment=a['deployment'],condition=condition,
                clean_correct=a['clean_correct'],fooled=m['fooled'],eligible=m['eligible'],
                success_pct=m['success_pct'],wilson95_low_pct=m['wilson95_pct'][0],
                wilson95_high_pct=m['wilson95_pct'][1],robust_correct=m['robust_correct'],
                robust_accuracy_pct=m['robust_accuracy_pct'],source=a['source'],source_sha256=a['source_sha256']))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':15,
    'axes.labelsize':11,'figure.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'savefig.facecolor':'white','svg.fonttype':'none'})
chart_data={}
for family,model in FAMILY_MODEL.items():
    subset=[a for a in records if a['model']==model]
    values=np.array([[a['conditions'][k]['success_pct'] if k in a['conditions'] else np.nan for k in MAIN] for a in subset])
    fig,ax=plt.subplots(figsize=(12.8,0.4*len(subset)+2.4))
    ax.imshow(values,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    ax.set_yticks(range(len(subset)),[a['deployment'] for a in subset])
    ax.set_xticks(range(len(MAIN)),HEADERS)
    ax.tick_params(length=0,pad=8)
    for i,a in enumerate(subset):
        for j,key in enumerate(MAIN):
            m=a['conditions'].get(key)
            text='Not run' if m is None else f"{m['fooled']}/{m['eligible']}\n{m['success_pct']:.1f}%"
            ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if m and m['success_pct']>62 else '#182532')
    ax.set_title(f'{family}: retained evasion success',loc='left',pad=20)
    fig.text(0.02,0.025,'Own clean-correct denominators; higher = more evasion. Budgets in /255. PGD+ = stronger 1/255 check.\nRows use different clean-correct sets; the full matrix is descriptive, not an all-deployment paired comparison.',fontsize=10)
    fig.subplots_adjust(left=0.34,right=0.98,top=0.91,bottom=0.12)
    fig.savefig(FIGURES/f'{model}_matrix.png',dpi=170)
    plt.close(fig)
    chart_data[f'{model}_matrix']=[dict(deployment=a['deployment'],conditions={k:a['conditions'].get(k) for k in MAIN}) for a in subset]

# The same axes and uncertainty definition for both models in the direct view.
fig,axes=plt.subplots(1,2,figsize=(15,9.3),sharey=True)
for ax,key,title in zip(axes,['pgd@1','square@8'],['PGD at 1/255','Square at 8/255 (500 queries)']):
    rows=[x for x in cross if x['condition']==key]
    for i,row in enumerate(rows):
        for offset,field,color,marker in [(-.12,'resnet',BLUE,'o'),(.12,'mobilenet',ORANGE,'s')]:
            m=row[field]
            low,high=m['wilson95_pct']
            ax.errorbar(m['success_pct'],i+offset,xerr=[[max(0,m['success_pct']-low)],[max(0,high-m['success_pct'])]],fmt=marker,color=color,capsize=2,markersize=5)
    ax.set_title(title,loc='left',pad=12)
    ax.set_xlim(0,105);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Attack success (%)')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
axes[0].set_yticks(range(len(common_labels)),[x.replace(' (different recipe/hardware)','*') for x in common_labels])
axes[0].invert_yaxis()
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',linestyle='',label='ResNetV2-50'),Line2D([],[],color=ORANGE,marker='s',linestyle='',label='MobileNetV2')],loc='upper center',ncol=2,bbox_to_anchor=(.64,.985),frameon=False)
fig.text(.02,.025,'Intervals: 95% Wilson for each own-denominator rate; they do not measure uncertainty in model differences.\nSame image identities, model-specific crops and clean sets. *TensorRT uses different GPUs, versions and PTQ recipes.',fontsize=10)
fig.subplots_adjust(left=.29,right=.98,top=.89,bottom=.13,wspace=.12)
fig.savefig(FIGURES/'cross_model_rates.png',dpi=170);plt.close(fig)
chart_data['cross_model_rates']=[x for x in cross if x['condition'] in ['pgd@1','square@8']]

pair_names=['RKNN FP16 → INT8','Core ML CPU: FP16 → W8A16','Core ML CPU+GPU: FP16 → W8A16','Core ML CPU+NE: FP16 → W8A16']
fig,axes=plt.subplots(1,2,figsize=(14,5.4),sharey=True)
for ax,key,title in zip(axes,['pgd@1','square@8'],['PGD at 1/255','Square at 8/255']):
    for model,offset,color in [('resnetv2_50',-.17,BLUE),('mobilenetv2',.17,ORANGE)]:
        vals=[next(x['reduction_pp'] for x in effects if x['model']==model and x['pair']==pair and x['condition']==key) for pair in pair_names]
        ax.barh(np.arange(len(vals))+offset,vals,height=.28,color=color,label=model)
        for i,v in enumerate(vals):ax.text(v+(.5 if v>=0 else -.5),i+offset,f'{v:+.1f}',ha='left' if v>=0 else 'right',va='center',fontsize=10)
    ax.axvline(0,color='#283848',linewidth=1)
    ax.set_xlim(-7,33);ax.set_title(title,loc='left');ax.set_xlabel('Reduction in attack success (percentage points)')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
axes[0].set_yticks(range(4),['RKNN FP16 → W8A8','Core ML CPU: FP16 → W8A16','Core ML GPU: FP16 → W8A16','Core ML NE: FP16 → W8A16']);axes[0].invert_yaxis()
fig.legend(handles=[Line2D([],[],color=BLUE,lw=7,label='ResNetV2-50'),Line2D([],[],color=ORANGE,lw=7,label='MobileNetV2')],loc='upper center',ncol=2,frameon=False)
fig.text(.02,.025,'Positive = fewer measured evasions after quantization. Common-clean masks within each source study:\nRKNN: ResNet n=71, MobileNet n=80; Core ML: ResNet n=81, MobileNet n=82. Descriptive differences, no paired-effect intervals.',fontsize=9.5)
fig.subplots_adjust(left=.3,right=.98,top=.83,bottom=.23,wspace=.12)
fig.savefig(FIGURES/'quantization_model_interaction.png',dpi=170);plt.close(fig)
chart_data['quantization_model_interaction']=effects

fig,axes=plt.subplots(1,2,figsize=(13,5.8),sharey=True)
for ax,model,title in zip(axes,['resnetv2_50','mobilenetv2'],['ResNetV2-50','MobileNetV2']):
    for precision,color,marker in [('fp16',BLUE,'o'),('int8',ORANGE,'s')]:
        cells=[next(x for x in seed_results if x['model']==model and x['arm']==f'{precision}_{unit}' and x['eps_255']==8) for unit in ['cpu_only','cpu_and_ne','cpu_and_gpu']]
        means=[x['mean_pct'] for x in cells]
        positions=np.arange(3)+(-.025 if precision=='fp16' else .025)
        ax.errorbar(positions,means,yerr=[[x['mean_pct']-x['min_pct'] for x in cells],[x['max_pct']-x['mean_pct'] for x in cells]],color=color,marker=marker,linestyle='-' if precision=='fp16' else '--',capsize=4,label='FP16' if precision=='fp16' else 'INT8 weights / FP16 compute')
    ax.set_xticks(range(3),['CPU','CPU + NE','CPU + GPU']);ax.set_ylim(40,95)
    ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0].set_ylabel('Square success at 8/255 (%)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
fig.text(.02,.025,'Means and observed min–max over three supplementary seeds; ranges are not confidence intervals.\nOwn clean-correct populations (ResNet FP16 n=82, W8A16 n=81; MobileNet n=83). Each unit is a compute-unit request.',fontsize=10)
fig.subplots_adjust(left=.08,right=.98,top=.83,bottom=.23,wspace=.12)
fig.savefig(FIGURES/'coreml_square_seed_ranges.png',dpi=170);plt.close(fig)
chart_data['coreml_square_seed_ranges']=[x for x in seed_results if x['eps_255']==8]
save('chart_data.json',chart_data)

# Compact table fragments support the authored reports and can be regenerated.
def cell(m):
    return 'Not run' if m is None else f"{m['fooled']}/{m['eligible']} ({m['success_pct']:.1f}%)"
for model in FAMILY_MODEL.values():
    lines=['| Deployment | Clean /100 | '+' | '.join(HEADERS)+' |','| --- | ---: | '+' | '.join(['---:']*len(MAIN))+' |']
    for a in records:
        if a['model']!=model:continue
        lines.append(f"| [{a['deployment']}]({link(REPORTS/a['source'], HERE)}) | {a['clean_correct']} | "+' | '.join(cell(a['conditions'].get(k)) for k in MAIN)+' |')
    (HERE/f'{model}_table.md').write_text('\n'.join(lines)+'\n')
    lines=['| Core ML deployment | PGD 1 | PGD+ 1 | Square 4 | Square 8 |','| --- | ---: | ---: | ---: | ---: |']
    for arm,a in core[model]['arms'].items():
        lines.append('| `'+arm+'` | '+' | '.join(cell(metric(a['shared'][k]['fooled'],a['shared'][k]['n'])) for k in ['pgd@1','pgd_stronger@1','square@4','square@8'])+' |')
    (HERE/f'{model}_coreml_shared.md').write_text('\n'.join(lines)+'\n')
lines=['| Deployment | ResNet PGD 1 | MobileNet PGD 1 | MobileNet − ResNet (pp) | ResNet Square 8 | MobileNet Square 8 |','| --- | ---: | ---: | ---: | ---: | ---: |']
for label in common_labels:
    a=next(x for x in cross if x['deployment']==label and x['condition']=='pgd@1')
    b=next(x for x in cross if x['deployment']==label and x['condition']=='square@8')
    lines.append(f"| {label} | {cell(a['resnet'])} | {cell(a['mobilenet'])} | {a['mobile_minus_resnet_pp']:+.1f} | {cell(b['resnet'])} | {cell(b['mobilenet'])} |")
(HERE/'cross_model_table.md').write_text('\n'.join(lines)+'\n')
save('validation.json',dict(passed=True,deployment_counts=dict(Counter(a['model'] for a in records)),
    conditions_reconciled=sum(len(a['conditions']) for a in records),
    original_summary_hashes_verified=len(records),cohort_sha256_and_label_matches=100,
    coreml_matrices_checked=2,shared_effect_cells=len(effects),supplementary_seed_cells=len(seed_results),
    figures_generated=5,full_combined_shared_population_available=False,
    environment=dict(python=sys.version.split()[0],numpy=np.__version__,matplotlib=matplotlib.__version__),
    limitations=['No fresh inference or candidate replay performed.','No per-image Core ML data in this archive; no combined 19/20-arm clean intersection or paired model-effect confidence interval is claimed.']))
print((HERE/'validation.json').read_text())
