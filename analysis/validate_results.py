#!/usr/bin/env python3
"""Independent numerical checks, data conservation, and report-link checks.

The key ANCOVA estimates and HC3 intervals are recalculated using NumPy
matrix algebra rather than the statsmodels fitting code in run_analysis.py.
Known missing primary evidence is a limitation, not a failed software test.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = root/'analysis'
    manifest = json.loads((analysis/'manifest.json').read_text(encoding='utf-8'))
    checks = []
    def require(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    sources = list((root/'csv').glob('*.csv'))
    require(len(sources)==16, '16 source CSV files')
    for path in sources:
        require(hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==manifest['input_sha256'][path.name], 'Source SHA-256 (LF normalized): '+path.name)
    p = pd.read_csv(root/'csv'/'Участники.csv',encoding='utf-8-sig')
    require(len(p)==56 and p['Код участника'].nunique()==56, '56 unique participants')
    g = p['Код группы'].to_numpy(float)
    require(int(g.sum())==27 and int((g==0).sum())==29, 'Group sizes 27 and 29')
    x = np.column_stack([np.ones(len(p)),g,p['Входной балл (T0)'].to_numpy(float)])
    xtx_inv = np.linalg.inv(x.T@x)
    leverage = np.sum((x@xtx_inv)*x,axis=1)
    ancova = pd.read_csv(analysis/'tables'/'04_ancova.csv',encoding='utf-8-sig')
    for outcome,column in [('post','Итоговый балл (T1)'),('delayed','Отложенный балл (T2)')]:
        y = p[column].to_numpy(float)
        beta = np.linalg.lstsq(x,y,rcond=None)[0]
        residual = y-x@beta
        omega = (residual/(1-leverage))**2
        covariance = xtx_inv @ (x.T @ (x*omega[:,None])) @ xtx_inv
        se = np.sqrt(covariance[1,1])
        critical = stats.t.ppf(.975,len(y)-x.shape[1])
        row = ancova[(ancova.outcome==outcome)&(ancova.model=='primary_pre_HC3')].iloc[0]
        expected = [beta[1],se,beta[1]-critical*se,beta[1]+critical*se]
        actual = row[['estimate','se','ci_low','ci_high']].to_numpy(float)
        require(np.allclose(expected,actual,rtol=1e-9,atol=1e-9), 'Independent NumPy ANCOVA + HC3: '+outcome)
    expert = pd.read_csv(root/'csv'/'Оценки_экспертов.csv',encoding='utf-8-sig')
    rubrics = pd.read_csv(analysis/'tables'/'06_h3_rubrics.csv',encoding='utf-8-sig')
    for criterion in ['sources','provenance','reproducibility','model_choice']:
        by_person = expert[expert['Код критерия']==criterion].groupby(['Код участника','Группа'])['Оценка'].mean().reset_index()
        diff = by_person[by_person['Группа']=='experimental']['Оценка'].mean()-by_person[by_person['Группа']=='comparison']['Оценка'].mean()
        actual = rubrics[rubrics.criterion==criterion].iloc[0]
        require(np.isclose(diff,actual.difference,rtol=1e-10), 'Participant-level rubric difference: '+criterion)
    primary = ancova[(ancova.model=='primary_pre_HC3')&ancova.outcome.isin(['post','delayed'])]
    primary_rubrics = rubrics[rubrics.primary_H3]
    pvalues = np.r_[primary.p.to_numpy(),primary_rubrics.p.to_numpy()]
    order = np.argsort(pvalues)
    adjusted = np.empty(6)
    adjusted[order] = np.minimum(1,np.maximum.accumulate(pvalues[order]*np.arange(6,0,-1)))
    reported = np.r_[primary.p_holm_primary_family.to_numpy(),primary_rubrics.p_holm_primary_family.to_numpy()]
    require(np.allclose(adjusted,reported,rtol=1e-9,atol=1e-13),'Six-test Holm family recalculated independently')
    scores = pd.read_csv(analysis/'tables'/'13_score_reconstruction.csv',encoding='utf-8-sig')
    require(len(scores)==280 and scores.difference.abs().max()<1e-8, 'All 280 participant scores reconciled with item totals')
    metrics = pd.read_csv(analysis/'tables'/'16_participant_metrics.csv',encoding='utf-8-sig')
    control = metrics[metrics['group']==0]
    require(control[['quality','completeness','requests']].isna().all().all(), 'Absent comparison logs stay missing')
    corr = pd.read_csv(analysis/'tables'/'08_h4_correlations.csv',encoding='utf-8-sig')
    require(corr.n.eq(27).all(), 'H4 uses 27 participants, not 962 events')
    for r in corr.itertuples():
        exp = metrics[metrics['group']==1]
        require(np.isclose(np.corrcoef(exp.transfer,exp[r.predictor])[0,1],r.pearson_r,atol=1e-10),'H4 Pearson independently: '+r.predictor)
    coding = pd.read_csv(root/'csv'/'Кодирование.csv',encoding='utf-8-sig')
    themes = pd.read_csv(analysis/'tables'/'12_h5_themes.csv',encoding='utf-8-sig')
    for r in themes.itertuples():
        block = coding[coding['Тема']==r.theme]
        require(r.coder1_positive==block['Кодировщик 1'].sum() and r.coder2_positive==block['Кодировщик 2'].sum(), 'Both coder counts: '+r.theme)
    require(len(manifest['output_tables'])==18 and len(manifest['output_figures'])==12, '18 CSV tables and 6 PNG/SVG figure pairs')
    for name in manifest['output_tables']:
        path = analysis/'tables'/name
        require(path.read_bytes().startswith(b'\xef\xbb\xbf'),'UTF-8 BOM: '+name)
        pd.read_csv(path,encoding='utf-8-sig')
    for name in manifest['output_figures']:
        path=analysis/'figures'/name
        require(path.stat().st_size>1000,'Nonempty figure: '+name)
        if path.suffix=='.png':
            with Image.open(path) as img:
                img.verify()
    report=(analysis/'REPORT.md').read_text(encoding='utf-8')
    for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report):
        require((analysis/target).is_file(),'Report image link: '+target)
    require('не содержит ответов' in report and 'реконструкц' in report.lower(), 'Report exposes primary-data limitations')
    for script in ['run_analysis.py','validate_results.py']:
        compile((analysis/script).read_text(encoding='utf-8'),script,'exec')
    require(True,'Python scripts compile')
    result={'status':'PASS','checks_passed':len(checks),'checks':checks,
            'scope':'Numerical implementation, input hashes, units, output integrity. Does not establish authentic primary data.'}
    (analysis/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'PASS','checks_passed':len(checks)},ensure_ascii=False))


if __name__=='__main__':
    main()
