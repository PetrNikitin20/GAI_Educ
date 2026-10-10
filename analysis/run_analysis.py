#!/usr/bin/env python3
"""Reproduce the H1--H5 analysis without changing any source CSV.

Run: python analysis/run_analysis.py --root .
Only participants are independent analysis units. The source history and
template-labelled bibliography raise provenance concerns; statistics describe this table
and do not establish an authentic randomized experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20261010
SOURCE_COMMIT = 'a2a5246960889eaad21681d432c2dfc13839d2f0'
RECONSTRUCTION_MARKER_COMMIT = 'd6a7855a96292864a47dc92d8cd83257f9a6da67'
REPO_URL = 'https://github.com/PetrNikitin20/GAI_Educ'
GROUP_NAMES = {1: 'Структурированная программа', 0: 'Группа сравнения'}
COLS = {
    'Код участника': 'id', 'Код группы': 'group', 'Группа': 'group_name',
    'Входной балл (T0)': 'pre', 'Итоговый балл (T1)': 'post',
    'Отложенный балл (T2)': 'delayed', 'ИИ-грамотность': 'ai_literacy',
    'Балл переноса навыков': 'transfer', 'Опыт программирования': 'experience',
    'Всего ссылок': 'links_total', 'Проверенные ссылки': 'links_verified',
    'Проверено слов': 'words', 'Неподтверждённые утверждения': 'claims',
    'Успешное воспроизведение': 'reproduced', 'Выбор модели обоснован': 'model_justified',
    'Полнота журнала': 'completeness', 'Качество проверки': 'quality',
    'Количество запросов к ИИ': 'requests', 'Обнаружена ошибка ИИ': 'ai_error',
}
CRITERIA = {
    'question': 'Исследовательский вопрос', 'hypothesis': 'Гипотеза и опровержение',
    'sources': 'Проверка источников', 'design': 'Дизайн исследования',
    'provenance': 'Происхождение данных', 'reproducibility': 'Воспроизводимость',
    'model_choice': 'Выбор модели', 'reflection': 'Ограничения и рефлексия',
}
H3_KEYS = ['sources', 'provenance', 'reproducibility', 'model_choice']
THEMES = {
    'AI_as_critic': 'ИИ как критик', 'alternative_generation': 'Генерация альтернатив',
    'authorship_awareness': 'Осознание авторства', 'overtrust_risk': 'Риск чрезмерного доверия',
    'time_cost': 'Затраты времени', 'verification_habit': 'Привычка проверять',
}
MEASURES = {
    'pre': 'Входной балл T0 (0–100)', 'post': 'Итоговый балл T1 (0–100)',
    'delayed': 'Отложенный балл T2 (0–100)', 'transfer': 'Перенос TR (0–20)',
    'ai_literacy': 'ИИ-грамотность (0–40)', 'experience': 'Опыт программирования (шкала не описана)',
    'artifact_total': 'Сумма экспертной рубрики (0–32)',
    'links_fraction': 'Доля проверенных ссылок (0–1)',
    'claims_rate': 'Неподтверждённые утверждения на 1000 слов',
    'completeness': 'Полнота журнала (0–1)', 'quality': 'Качество проверки (0–1)',
    'requests': 'Число запросов к ИИ',
}


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def table_md(frame, columns=None, rename=None, digits=3):
    """Markdown tables without an optional tabulate dependency."""
    d = frame.copy() if columns is None else frame.loc[:, columns].copy()
    if rename:
        d = d.rename(columns=rename)
    def fmt(v):
        if pd.isna(v):
            return 'н.д.'
        if isinstance(v, (float, np.floating)):
            if v != 0 and abs(v) < 0.001:
                return f'{v:.2e}'
            return f'{v:.{digits}f}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, d.columns)) + ' |', '| ' + ' | '.join(['---'] * len(d.columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in d.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def fit_ols(data, outcome, predictors, robust=True):
    x = sm.add_constant(data[predictors].astype(float), has_constant='add')
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError(f'Rank-deficient model: {outcome}, {predictors}')
    model = sm.OLS(data[outcome].astype(float), x, missing='raise')
    return model.fit(cov_type='HC3', use_t=True) if robust else model.fit()


def estimate(model, term):
    ci = model.conf_int().loc[term]
    return {'estimate': float(model.params[term]), 'se': float(model.bse[term]),
            'ci_low': float(ci.iloc[0]), 'ci_high': float(ci.iloc[1]),
            't': float(model.tvalues[term]), 'p': float(model.pvalues[term]),
            'n': int(model.nobs), 'df': float(model.df_resid)}


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb)**2 / ((va / na)**2 / (na-1) + (vb / nb)**2 / (nb-1))
    difference = a.mean() - b.mean()
    half = stats.t.ppf(0.975, df) * se
    pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    correction = 1 - 3 / (4*(na+nb)-9)
    return {'n_exp': na, 'n_comp': nb, 'mean_exp': a.mean(), 'sd_exp': a.std(ddof=1),
            'mean_comp': b.mean(), 'sd_comp': b.std(ddof=1), 'difference': difference,
            'ci_low': difference-half, 'ci_high': difference+half,
            'p': float(stats.ttest_ind(a, b, equal_var=False).pvalue),
            'hedges_g': correction*difference/pooled, 'welch_df': df,
            'p_mannwhitney': float(stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic').pvalue)}


def row_corr(x, y):
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(x*x, axis=1) * np.sum(y*y, axis=1))
    return np.divide(np.sum(x*y, axis=1), denom, out=np.full(len(x), np.nan), where=denom > 0)


def bootstrap_corr(values, samples, rank=False):
    d = values[samples]
    if rank:
        d = stats.rankdata(d, axis=1)
    return np.column_stack([row_corr(d[:, :, 0], d[:, :, j]) for j in range(1, d.shape[2])])


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    observed = np.mean(a == b)
    expected = np.mean(a == 1)*np.mean(b == 1) + np.mean(a == 0)*np.mean(b == 0)
    return np.nan if np.isclose(expected, 1) else (observed-expected)/(1-expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--resamples', type=int, default=20000)
    args = parser.parse_args()
    if args.resamples < 1000:
        raise ValueError('At least 1000 resamples are required.')
    root = args.root.resolve()
    output = root / 'analysis'
    tables_dir, figures_dir = output/'tables', output/'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sources = sorted((root/'csv').glob('*.csv'))
    if len(sources) != 16:
        raise ValueError(f'Expected 16 source CSVs, got {len(sources)}')
    raw = {p.stem: pd.read_csv(p, encoding='utf-8-sig') for p in sources}
    # Git checkouts can use CRLF on Windows. Normalize only CRLF for portable
    # text checksums; preserve source bytes and BOM on disk.
    raw_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    hashes = {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sources}
    expected_hashes = json.loads((output/'source_sha256.json').read_text(encoding='utf-8'))
    if hashes != expected_hashes:
        raise ValueError('Source CSVs differ from the documented snapshot. Review the analysis plan and update source_sha256.json before analysing new data.')
    tables = {}
    def save(name, frame):
        frame = pd.DataFrame(frame)
        frame.to_csv(tables_dir/f'{name}.csv', index=False, encoding='utf-8-sig', float_format='%.12g', lineterminator='\n')
        tables[name] = frame
        return frame
    print('Loaded 16 source CSV files.', flush=True)

    inventory = []
    dictionary = []
    for path in sources:
        d = raw[path.stem]
        inventory.append({'file': path.name, 'rows': len(d), 'columns': d.shape[1],
                          'missing_cells': int(d.isna().sum().sum()), 'sha256': hashes[path.name]})
        for c in d.columns:
            dictionary.append({'file': path.name, 'column': c, 'dtype': str(d[c].dtype),
                               'missing': int(d[c].isna().sum()), 'distinct_nonmissing': int(d[c].nunique())})
    save('01_data_inventory', inventory)
    save('18_data_dictionary', dictionary)

    data = raw['Участники'].rename(columns=COLS).copy()
    if data.id.duplicated().any() or data['group'].isna().any() or not set(data['group']) == {0, 1}:
        raise ValueError('Invalid participant keys or group coding.')
    for name in ['pre', 'post', 'delayed', 'ai_literacy', 'transfer', 'experience']:
        if data[name].isna().any():
            raise ValueError(f'Missing required outcome/covariate {name}; do not silently impute.')
    if not (data['group_name'] == data['group'].map({1:'experimental',0:'comparison'})).all():
        raise ValueError('Group labels disagree.')
    expert = raw['Оценки_экспертов']
    if expert.duplicated(['Код участника','Код эксперта','Код критерия']).any():
        raise ValueError('Duplicate expert records.')
    if (expert.groupby(['Код участника','Код критерия']).size() != 2).any():
        raise ValueError('Every participant/criterion requires two expert ratings.')
    if not expert['Оценка'].between(0,4).all():
        raise ValueError('Expert rating outside 0–4.')
    artifact = expert.pivot_table(index='Код участника', columns='Код критерия', values='Оценка', aggfunc='mean')
    artifact = artifact[list(CRITERIA)]
    artifact['artifact_total'] = artifact.sum(axis=1, min_count=8)
    data = data.merge(artifact, left_on='id', right_index=True, validate='one_to_one', how='left')
    if data['artifact_total'].isna().any():
        raise ValueError('Unmatched artifact ratings.')
    data['links_fraction'] = data.links_verified / data.links_total
    data['claims_rate'] = data.claims / data.words * 1000
    save('16_participant_metrics', data[['id','group','group_name']+list(MEASURES)+['reproduced','model_justified']+list(CRITERIA)])
    missing = []
    for g, block in data.groupby('group'):
        for c in list(COLS.values()):
            missing.append({'group': GROUP_NAMES[g], 'variable': c, 'n': len(block), 'missing': int(block[c].isna().sum())})
    save('17_missingness', missing)

    audit = []
    def check(name, ok, detail):
        audit.append({'check': name, 'result': 'PASS' if ok else 'FAIL', 'detail': detail})
    registry = raw['Реестр']
    check('Идентификаторы реестра совпадают', set(registry['Код участника']) == set(data.id), 'Сопоставление по идентификатору')
    check('Все участники включены в реестр', registry['Включён в выборку'].eq(1).all(), 'Проверена отметка включения')
    check('Группы реестра совпадают', registry.set_index('Код участника')['Код группы'].reindex(data.id).to_numpy().tolist() == data['group'].tolist(), '0 / 1')
    responses = raw['Ответы_тестов']
    check('Нет повторных ответов на задания', not responses.duplicated(['Код участника','Код инструмента','Код задания']).any(), 'Ключ: участник × инструмент × задание')
    check('Баллы заданий в пределах шкалы', (responses['Балл за задание'].ge(0) & responses['Балл за задание'].le(responses['Максимальный балл за задание'])).all(), 'Включая дробные баллы')
    score_checks = []
    for tool, measure in [('T0','pre'),('T1','post'),('T2','delayed'),('AIL','ai_literacy'),('TR','transfer')]:
        totals = responses[responses['Код инструмента']==tool].groupby('Код участника')['Балл за задание'].sum()
        for row in data.itertuples():
            calculated = float(totals.loc[row.id])
            stated = float(getattr(row, measure))
            score_checks.append({'id': row.id, 'instrument': tool, 'sum_items': calculated,
                                 'participant_score': stated, 'difference': calculated-stated})
    score_checks = save('13_score_reconstruction', score_checks)
    check('Суммы заданий воспроизводят баллы', (score_checks.difference.abs()<1e-8).all(), f'Максимальная ошибка: {score_checks.difference.abs().max():.3g}')
    # Check expert long table against all 16 wide rating columns, not just totals.
    expert_differences = []
    for code, label in CRITERIA.items():
        for j in [1,2]:
            long = expert[(expert['Код критерия']==code)&(expert['Код эксперта']==f'E{j}')].set_index('Код участника')['Оценка']
            differences = data[f'{label} (эксперт {j})'].to_numpy() - long.reindex(data.id).to_numpy()
            expert_differences.extend(differences)
    check('Экспертные оценки long/wide совпадают', np.max(np.abs(expert_differences)) < 1e-8, '896 оценок')
    links = raw['Проверка_ссылок'].groupby('Код участника').agg(n=('Код ссылки','size'), yes=('Ссылка проверена','sum'))
    check('Число и проверка ссылок согласованы', np.array_equal(links.reindex(data.id).n, data.links_total) and np.array_equal(links.reindex(data.id).yes, data.links_verified), '678 строк; статистические единицы — 56 участников')
    for filename, mappings in [('Аудит_текста', {'Проверено слов':'words','Неподтверждённые утверждения':'claims'}),
                               ('Воспроизведение', {'Успешное воспроизведение':'reproduced','Выбор модели обоснован':'model_justified'})]:
        block = raw[filename].set_index('Код участника').reindex(data.id)
        for original, alias in mappings.items():
            check(f'{filename}: {original}', np.allclose(block[original],data[alias]), 'Согласование таблиц по идентификатору')
    audit += [
        {'check':'Происхождение наблюдений', 'result':'LIMITATION', 'detail':'В версии d6a7855 сводка названа «Воспроизводимая реконструкция чисел статьи»; в a2a5246 заголовок удалён без изменения чисел; первичное происхождение не подтверждено'},
        {'check':'Проверки ссылок', 'result':'LIMITATION', 'detail':'Макеты DOI/каталога и REF:...; нет реальных DOI/URL для независимой проверки'},
        {'check':'Тексты интервью', 'result':'MISSING', 'detail':'Ответы_интервью.csv содержит только вопросы; нет ответов, цитат и дат'},
        {'check':'Протокол эксперимента', 'result':'MISSING', 'detail':'Нет описания назначения групп, преподавателей/кластеров, интервала T1–T2 или текста новой задачи'},
        {'check':'Данные процесса в группе сравнения', 'result':'MISSING', 'detail':'29 пропусков в полноте, качестве проверки, числе запросов и обнаружении ошибки; нулями не заменяются'},
    ]

    descriptive = []
    for variable, label in MEASURES.items():
        for g in [1,0]:
            x = data.loc[data['group']==g, variable].dropna()
            n = len(x)
            half = stats.t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
            descriptive.append({'variable':variable,'measure':label,'group':GROUP_NAMES[g], 'n':n,
                                'mean':x.mean(),'sd':x.std(ddof=1),'median':x.median(),
                                'min':x.min(),'max':x.max(),'ci_low':x.mean()-half,'ci_high':x.mean()+half})
    save('03_descriptive', descriptive)

    ancova, diagnostics = [], []
    models = {}
    for outcome in ['post','delayed','transfer','ai_literacy']:
        for name, covariates, robust in [
            ('primary_pre_HC3',['group','pre'], True),
            ('sensitivity_pre_experience_HC3',['group','pre','experience'], True),
            ('reconstruct_pre_experience_OLS',['group','pre','experience'], False),
        ]:
            model = fit_ols(data, outcome, covariates, robust)
            row = {'outcome':outcome,'model':name,'covariance':'HC3' if robust else 'OLS', **estimate(model,'group'), 'r_squared':model.rsquared}
            # Marginal predictions at common sample-mean baseline and experience.
            for g in [0,1]:
                design = {c:[float(data[c].mean())] for c in covariates}
                design['group'] = [g]
                new = sm.add_constant(pd.DataFrame(design), has_constant='add')[model.model.exog_names]
                pred = model.get_prediction(new).summary_frame(alpha=.05).iloc[0]
                row[f'adjusted_mean_{g}'] = float(pred['mean'])
                row[f'adjusted_ci_low_{g}'] = float(pred['mean_ci_lower'])
                row[f'adjusted_ci_high_{g}'] = float(pred['mean_ci_upper'])
            ancova.append(row)
            if name=='primary_pre_HC3':
                models[outcome] = model
                ordinary = fit_ols(data,outcome,covariates,False)
                interacted = data.assign(group_x_pre=data['group']*(data.pre-data.pre.mean()))
                interaction_model = fit_ols(interacted,outcome,['group','pre','group_x_pre'])
                diagnostics.append({'outcome':outcome,'n':len(data),'slope_interaction_p':estimate(interaction_model,'group_x_pre')['p'],
                                    'breusch_pagan_p':float(het_breuschpagan(ordinary.resid,ordinary.model.exog)[1]),
                                    'shapiro_residual_p':float(stats.shapiro(ordinary.resid).pvalue),
                                    'max_cooks_distance':float(ordinary.get_influence().cooks_distance[0].max()),
                                    'max_leverage':float(ordinary.get_influence().hat_matrix_diag.max())})
    ancova = save('04_ancova', ancova)
    save('05_model_diagnostics', diagnostics)

    rubric = []
    for key in list(CRITERIA)+['artifact_total']:
        a,b = [data.loc[data['group']==g,key] for g in [1,0]]
        rubric.append({'criterion':key, 'label':CRITERIA.get(key,'Сумма рубрики 0–32'), 'primary_H3':key in H3_KEYS, **welch(a,b)})
    rubric = pd.DataFrame(rubric)
    # One primary family: H1, H2, and four pre-specified H3 domains.
    primary_p = [models[o].pvalues['group'] for o in ['post','delayed']] + rubric.set_index('criterion').loc[H3_KEYS,'p'].tolist()
    corrected = multipletests(primary_p, alpha=.05, method='holm')[1]
    ancova['p_holm_primary_family'] = np.nan
    for i,o in enumerate(['post','delayed']):
        ancova.loc[(ancova.outcome==o)&(ancova.model=='primary_pre_HC3'),'p_holm_primary_family'] = corrected[i]
    save('04_ancova', ancova)
    rubric['p_holm_primary_family'] = np.nan
    for i,key in enumerate(H3_KEYS):
        rubric.loc[rubric.criterion==key,'p_holm_primary_family'] = corrected[i+2]
    save('06_h3_rubrics', rubric)

    operations = []
    for variable,label in [('links_fraction','Доля проверенных ссылок'),('claims_rate','Неподтверждённые утверждения на 1000 слов'),('reproduced','Успешное воспроизведение'),('model_justified','Обоснованный выбор модели')]:
        a,b = [data.loc[data['group']==g,variable].to_numpy(float) for g in [1,0]]
        n1,n0 = len(a),len(b)
        boot = a[rng.integers(n1,size=(args.resamples,n1))].mean(axis=1)-b[rng.integers(n0,size=(args.resamples,n0))].mean(axis=1)
        ci = np.quantile(boot,[.025,.975])
        row = {'variable':variable,'label':label,'n_exp':n1,'n_comp':n0,'mean_exp':a.mean(),'mean_comp':b.mean(),
               'difference':a.mean()-b.mean(),'ci_low':ci[0],'ci_high':ci[1], 'ci_method':'stratified participant percentile bootstrap', 'odds_ratio':np.nan}
        if variable in ['reproduced','model_justified']:
            counts = [[int(a.sum()),n1-int(a.sum())],[int(b.sum()),n0-int(b.sum())]]
            fisher = stats.fisher_exact(counts)
            row.update({'p':float(fisher.pvalue),'test':'Fisher exact (participants)','odds_ratio':float(fisher.statistic),'events_exp':int(a.sum()),'events_comp':int(b.sum())})
        else:
            row.update({'p':float(stats.mannwhitneyu(a,b,alternative='two-sided',method='asymptotic').pvalue),'test':'Mann–Whitney (participant measures)','events_exp':np.nan,'events_comp':np.nan})
        operations.append(row)
    operations = pd.DataFrame(operations)
    operations['p_holm_exploratory'] = multipletests(operations.p,method='holm')[1]
    save('07_h3_operational', operations)

    # H4: complete cases, only experimental group, never turn absent logs into 0.
    exp = data.loc[data['group']==1].copy()
    if exp[['quality','completeness','requests']].isna().any().any():
        raise ValueError('Experimental process metrics contain missing values.')
    predictors = ['quality','completeness','requests']
    samples = rng.integers(len(exp), size=(args.resamples,len(exp)))
    matrix = exp[['transfer']+predictors].to_numpy(float)
    boot_r = bootstrap_corr(matrix,samples)
    boot_s = bootstrap_corr(matrix,samples,rank=True)
    correlations = []
    for j,var in enumerate(predictors):
        pearson = stats.pearsonr(exp.transfer,exp[var])
        rank_x = stats.rankdata(exp[var].to_numpy())
        rank_y = stats.rankdata(exp.transfer.to_numpy())
        rho = float(stats.pearsonr(rank_x,rank_y).statistic)
        # Permute pairing, preserving ties and the subject count.
        perm_y = np.array([rng.permutation(rank_y) for _ in range(args.resamples)])
        perm_r = row_corr(np.broadcast_to(rank_x,perm_y.shape),perm_y)
        p_perm = (np.sum(np.abs(perm_r)>=abs(rho)-1e-12)+1)/(args.resamples+1)
        ci_r = np.nanquantile(boot_r[:,j],[.025,.975])
        ci_s = np.nanquantile(boot_s[:,j],[.025,.975])
        correlations.append({'predictor':var,'n':len(exp),'pearson_r':float(pearson.statistic),'pearson_p':float(pearson.pvalue),
                             'pearson_ci_low':ci_r[0],'pearson_ci_high':ci_r[1],'spearman_rho':rho,
                             'spearman_p_permutation':p_perm,'spearman_ci_low':ci_s[0],'spearman_ci_high':ci_s[1]})
    corr = pd.DataFrame(correlations)
    corr['pearson_p_holm_exploratory'] = multipletests(corr.pearson_p,method='holm')[1]
    save('08_h4_correlations', corr)
    standardized = exp.copy()
    for c in ['transfer','pre','experience']+predictors:
        standardized[c] = (exp[c]-exp[c].mean())/exp[c].std(ddof=1)
    regression = fit_ols(standardized,'transfer',predictors+['pre','experience'])
    x = regression.model.exog
    regression_table = []
    for i,term in enumerate(regression.model.exog_names):
        regression_table.append({'term':term,**estimate(regression,term),'vif':variance_inflation_factor(x,i) if term!='const' else np.nan,
                                 'r_squared':regression.rsquared,'adjusted_r_squared':regression.rsquared_adj})
    save('09_h4_regression', regression_table)
    contrasts = []
    for j,variable in enumerate(predictors[:2]):
        contrast = np.zeros(len(regression.params))
        contrast[regression.model.exog_names.index(variable)] = 1
        contrast[regression.model.exog_names.index('requests')] = -1
        test = regression.t_test(contrast)
        interval = test.conf_int()[0]
        contrasts.append({'comparison':variable+' minus requests','type':'standardized_beta_signed_HC3',
                          'difference':float(np.asarray(test.effect).item()),'ci_low':interval[0],'ci_high':interval[1],'p':float(test.pvalue)})
        for absolute in [False,True]:
            dist = (np.abs(boot_r[:,j])-np.abs(boot_r[:,2])) if absolute else (boot_r[:,j]-boot_r[:,2])
            point = (abs(corr.pearson_r.iloc[j])-abs(corr.pearson_r.iloc[2])) if absolute else (corr.pearson_r.iloc[j]-corr.pearson_r.iloc[2])
            interval = np.nanquantile(dist,[.025,.975])
            contrasts.append({'comparison':variable+' minus requests','type':'pearson_absolute_paired_bootstrap' if absolute else 'pearson_signed_paired_bootstrap',
                              'difference':point,'ci_low':interval[0],'ci_high':interval[1],'p':np.nan})
    contrasts = pd.DataFrame(contrasts)
    contrasts['p_holm_two_beta_contrasts'] = np.nan
    mask = contrasts.type=='standardized_beta_signed_HC3'
    contrasts.loc[mask,'p_holm_two_beta_contrasts'] = multipletests(contrasts.loc[mask,'p'],method='holm')[1]
    save('10_h4_contrasts', contrasts)
    logs = raw['Журнал_ИИ'].groupby('Код участника').agg(log_rows=('Номер взаимодействия','size'),
        completeness_from_log=('Оценка полноты журнала','mean'),quality_from_log=('Оценка качества проверки','mean'))
    log_audit = exp[['id','requests','completeness','quality']].merge(logs,left_on='id',right_index=True,how='left',validate='one_to_one')
    log_audit['requests_difference'] = log_audit.log_rows-log_audit.requests
    log_audit['completeness_difference'] = log_audit.completeness_from_log-log_audit.completeness
    log_audit['quality_difference'] = log_audit.quality_from_log-log_audit.quality
    save('11_h4_log_audit', log_audit)
    check('Число запросов согласовано с журналом', log_audit.requests_difference.eq(0).all(), 'Проверены только 27 участников с журналом')
    check('Полнота равна среднему по журналу', log_audit.completeness_difference.abs().lt(1e-6).all(), f'Максимальное расхождение {log_audit.completeness_difference.abs().max():.6f}; агрегирующая методика не описана')
    check('Качество равно среднему по журналу', log_audit.quality_difference.abs().lt(1e-6).all(), f'Максимальное расхождение {log_audit.quality_difference.abs().max():.6f}; агрегирующая методика не описана')
    # Sensitivity to the participant vs event-level aggregate discrepancy.
    alternative = exp[['id','transfer','pre','experience']].merge(logs,left_on='id',right_index=True,validate='one_to_one')
    alt_correlations = {v:float(stats.pearsonr(alternative.transfer, alternative[c]).statistic) for v,c in [('quality','quality_from_log'),('completeness','completeness_from_log'),('requests','log_rows')]}

    coding = raw['Кодирование']
    if coding.duplicated(['Код участника','Тема']).any():
        raise ValueError('Duplicate qualitative coding key.')
    topic_counts = raw['Темы_интервью'].set_index('Тема')
    theme_rows = []
    for theme,label in THEMES.items():
        block = coding[coding['Тема']==theme]
        n = len(block)
        a,b = [block[c].to_numpy(int) for c in ['Кодировщик 1','Кодировщик 2']]
        ci = proportion_confint(a.sum(),n,alpha=.05,method='wilson')
        theme_rows.append({'theme':theme,'label':label,'n_interviews':n,'coder1_positive':int(a.sum()),'coder2_positive':int(b.sum()),
                           'both_positive':int(((a==1)&(b==1)).sum()),'disagreements':int((a!=b).sum()),
                           'coder1_proportion':a.mean(),'wilson_ci_low':ci[0],'wilson_ci_high':ci[1],
                           'agreement':np.mean(a==b),'kappa':kappa(a,b)})
        check('Частота темы: '+theme, a.sum()==topic_counts.loc[theme,'Количество отмеченных тем'], 'Сводная частота соответствует кодировщику 1, а не согласованному решению двух кодировщиков')
    themes = save('12_h5_themes',theme_rows)
    pooled_kappa = float(kappa(coding['Кодировщик 1'],coding['Кодировщик 2']))
    check('Отметки согласия кодировщиков согласованы', np.array_equal((coding['Кодировщик 1']==coding['Кодировщик 2']).astype(int),coding['Согласие кодировщиков']), '72 пары меток')
    if (data[data.id.isin(coding['Код участника'])]['group'] != 1).any():
        raise ValueError('Unexpected interview group.')
    save('02_data_audit', audit)

    consistency = []
    alias = {'pre_score':'pre','post_score':'post','delayed_score':'delayed','ai_literacy':'ai_literacy','transfer_score':'transfer'}
    for row in raw['Описательная'].itertuples(index=False,name=None):
        key,group,n,mean,sd = row
        if key not in alias:
            continue
        block = data[data.group_name==group][alias[key]]
        consistency.append({'source':'Описательная.csv','metric':key+' '+group,'stated':mean,'recalculated':block.mean(),'difference':block.mean()-mean})
        consistency.append({'source':'Описательная.csv','metric':key+' '+group+' SD','stated':sd,'recalculated':block.std(ddof=1),'difference':block.std(ddof=1)-sd})
    for row in raw['Артефакты'].itertuples(index=False,name=None):
        key,mean1,sd1,mean0,sd0,g = row
        key = 'artifact_total' if key=='total_0_32' else key
        rr = rubric[rubric.criterion==key].iloc[0]
        for col,stated in [('mean_exp',mean1),('sd_exp',sd1),('mean_comp',mean0),('sd_comp',sd0),('hedges_g',g)]:
            consistency.append({'source':'Артефакты.csv','metric':key+' '+col,'stated':stated,'recalculated':rr[col],'difference':rr[col]-stated})
    # Published aggregate ANCOVA figures are re-evaluated, never used as analysis inputs.
    summary = raw['Сводка']
    for phrase,outcome in [('итогового балла','post'),('отложенного балла','delayed'),('ИИ грамотности','ai_literacy'),('переноса','transfer')]:
        source_row = summary[summary.iloc[:,0].astype(str).str.contains('Скорректированная разность '+phrase,regex=False)].iloc[0]
        recalculated = ancova[(ancova.outcome==outcome)&(ancova.model=='reconstruct_pre_experience_OLS')].iloc[0]
        stated = float(source_row.iloc[1])
        consistency.append({'source':'Сводка.csv','metric':outcome+' ANCOVA OLS + experience','stated':stated,'recalculated':recalculated.estimate,'difference':recalculated.estimate-stated})
    save('14_summary_consistency',consistency)

    h1,h2 = [ancova[(ancova.outcome==o)&(ancova.model=='primary_pre_HC3')].iloc[0] for o in ['post','delayed']]
    h3 = rubric.set_index('criterion').loc[H3_KEYS]
    hypotheses = [
        {'hypothesis':'H1','status':'Численно поддерживается в представленной таблице', 'result':f'ΔT1={h1.estimate:.3f}; 95% ДИ [{h1.ci_low:.3f}; {h1.ci_high:.3f}]; p Holm={h1.p_holm_primary_family:.3g}', 'limitation':'Первичное происхождение и рандомизация не подтверждены'},
        {'hypothesis':'H2','status':'Различие T2 поддерживается; новая задача не верифицирована', 'result':f'ΔT2={h2.estimate:.3f}; 95% ДИ [{h2.ci_low:.3f}; {h2.ci_high:.3f}]; p Holm={h2.p_holm_primary_family:.3g}', 'limitation':'Нет текста новой задачи и интервала отложенного теста'},
        {'hypothesis':'H3','status':'Четыре оценки рубрики численно поддерживают гипотезу', 'result':f'Все четыре Δ>0; максимальное p Holm={h3.p_holm_primary_family.max():.3g}', 'limitation':'Нет исходных артефактов, реальных DOI, данных или исполняемого кода участников'},
        {'hypothesis':'H4','status':'Не поддерживается', 'result':'; '.join(f'r({r.predictor})={r.pearson_r:.3f}' for r in corr.itertuples()), 'limitation':'n=27; ДИ различий связей включают 0; результаты не доказывают отсутствия эффекта'},
        {'hypothesis':'H5','status':'Темы представлены; переход не может быть установлен', 'result':'Кодировщик 1: критик 8/12, альтернативы 7/12, время 8/12, доверие 4/12', 'limitation':'Нет текстов ответов, цитат, временных сравнений и контрольных интервью'},
    ]
    hypotheses = save('15_hypotheses',hypotheses)
    write_json(output/'results.json',{
        'source_commit':SOURCE_COMMIT,'seed':SEED,'resamples':args.resamples,
        'n_total':len(data),'n_experimental':len(exp),'n_comparison':int((data['group']==0).sum()),
        'h1':{k:float(h1[k]) for k in ['estimate','ci_low','ci_high','p','p_holm_primary_family']},
        'h2':{k:float(h2[k]) for k in ['estimate','ci_low','ci_high','p','p_holm_primary_family']},
        'h4_raw_r':{r.predictor:r.pearson_r for r in corr.itertuples()},'h4_log_mean_r':alt_correlations,
        'h4_model_r_squared':float(regression.rsquared),'h4_model_adjusted_r_squared':float(regression.rsquared_adj),
        'h5_pooled_kappa':pooled_kappa,'h5_overall_agreement':float(coding['Согласие кодировщиков'].mean()),
        'audit_failures':int(sum(r['result']=='FAIL' for r in audit)),
        'data_authenticity':'Not established; historical snapshot d6a7855 contained a reconstruction heading, current bibliography entries explicitly describe templates',
    })
    print('Models and tables calculated. Rendering figures.',flush=True)
    make_figures(data,ancova,rubric,corr,themes,models,figures_dir)
    make_report(output,data,ancova,rubric,operations,corr,contrasts,regression_table,themes,hypotheses,audit,pooled_kappa,alt_correlations,args.resamples)
    versions = {name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','statsmodels','matplotlib']}
    write_json(output/'manifest.json',{
        'source_repository':REPO_URL,'source_commit':SOURCE_COMMIT,'source_directory':'csv',
        'input_hash_method':'SHA-256 of UTF-8 bytes (BOM preserved), CRLF normalized to LF for cross-platform Git checkouts',
        'input_sha256':hashes,'python_version':platform.python_version(),'packages':versions,
        'seed':SEED,'resamples':args.resamples,'primary_family':['H1 T1','H2 T2']+H3_KEYS,
        'inference':'two-sided; alpha=.05; HC3 t intervals for ANCOVA; Holm for six primary tests',
        'output_tables':sorted(p.name for p in tables_dir.glob('*.csv')),
        'output_figures':sorted(p.name for p in figures_dir.glob('*')),
    })
    if len(tables)!=18:
        raise AssertionError(f'Expected 18 result tables, got {len(tables)}')
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=raw_hashes[path.name]:
            raise AssertionError(f'Source file changed: {path}')
    print(json.dumps({'tables':len(tables),'H1_delta':h1.estimate,'H2_delta':h2.estimate,'H4_supported':False,'audit_failures':sum(r['result']=='FAIL' for r in audit)},ensure_ascii=False),flush=True)


def make_figures(data,ancova,rubric,corr,themes,models,directory):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'GAI_Educ_20261010'})
    colors = {1:'#205493',0:'#a3541e'}
    rng = np.random.default_rng(SEED)
    def finish(fig,name):
        fig.text(.5,.01,'Происхождение данных не подтверждено; выводы относятся к представленным таблицам.',ha='center',fontsize=8,color='#555555')
        fig.tight_layout(rect=[0,.035,1,1])
        fig.savefig(directory/f'{name}.png',dpi=200,facecolor='white',metadata={'Software':'GAI_Educ analysis'})
        fig.savefig(directory/f'{name}.svg',facecolor='white',metadata={'Date':None})
        svg_path = directory/f'{name}.svg'
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
        plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,4.3))
    for ax,var,title in zip(axes,['pre','post','delayed'],['Входной тест T0','Итоговый тест T1','Отложенный тест T2']):
        for i,g in enumerate([1,0]):
            y = data.loc[data['group']==g,var].to_numpy()
            ax.scatter(i+rng.normal(0,.055,len(y)),y,s=24,color=colors[g],alpha=.65)
            half = stats.t.ppf(.975,len(y)-1)*y.std(ddof=1)/np.sqrt(len(y))
            ax.errorbar(i,y.mean(),yerr=half,fmt='D',color='black',capsize=5,markersize=6)
        ax.set_xticks([0,1],['Программа\nn=27','Сравнение\nn=29'])
        ax.set_ylim(0,100); ax.set_ylabel('Балл (0–100)'); ax.set_title(title); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Индивидуальные баллы; ромб — среднее и 95% ДИ',fontsize=14)
    finish(fig,'01_scores')

    fig,ax = plt.subplots(figsize=(9,4.4))
    for i,(out,label) in enumerate([('post','H1 · итоговый T1'),('delayed','H2 · отложенный T2')]):
        r = ancova[(ancova.outcome==out)&(ancova.model=='primary_pre_HC3')].iloc[0]
        ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',color=colors[1],capsize=5)
        ax.text(r.ci_high+.4,i,f'{r.estimate:.2f} [{r.ci_low:.2f}; {r.ci_high:.2f}]',va='center')
    ax.set_yticks([0,1],['H1 · итоговый T1','H2 · отложенный T2']); ax.invert_yaxis()
    ax.axvline(0,color='#777777',linestyle='--');ax.set_xlim(-1,21)
    ax.set_xlabel('Разность: программа − сравнение, баллы (95% ДИ HC3)')
    ax.set_title('Различия с контролем входного балла T0')
    ax.grid(axis='x',alpha=.2)
    finish(fig,'02_adjusted_effects')

    fig,ax = plt.subplots(figsize=(10,4.8))
    primary = rubric[rubric.criterion.isin(H3_KEYS)].copy()
    for i,r in enumerate(primary.itertuples()):
        ax.errorbar(r.difference,i,xerr=[[r.difference-r.ci_low],[r.ci_high-r.difference]],fmt='o',color=colors[1],capsize=5)
        ax.text(r.ci_high+.06,i,f'Δ={r.difference:.2f}; g={r.hedges_g:.2f}',va='center',fontsize=10)
    ax.set_yticks(range(len(primary)),primary.label); ax.invert_yaxis(); ax.axvline(0,color='#777777',ls='--')
    ax.set_xlim(-.15,2.55);ax.set_xlabel('Разность средних оценок на шкале 0–4 (95% ДИ Welch)')
    ax.set_title('H3 · среднее двух экспертов для каждого участника');ax.grid(axis='x',alpha=.2)
    finish(fig,'03_artifact_domains')

    fig,axes=plt.subplots(1,3,figsize=(12,4.4))
    for ax,var,label in zip(axes,['quality','completeness','requests'],['Качество проверки (0–1)','Полнота журнала (0–1)','Число запросов к ИИ']):
        block=data[data['group']==1];r=corr[corr.predictor==var].iloc[0]
        ax.scatter(block[var],block.transfer,s=40,alpha=.75,color=colors[1])
        line=np.polyfit(block[var],block.transfer,1);grid=np.linspace(block[var].min(),block[var].max(),100)
        ax.plot(grid,np.polyval(line,grid),color='#444444')
        ax.set_xlabel(label);ax.set_ylabel('Перенос TR (0–20)');ax.set_ylim(0,21)
        ax.set_title(f'r={r.pearson_r:.3f}; p={r.pearson_p:.3f}');ax.grid(alpha=.15)
    fig.suptitle('H4 · экспериментальная группа, n=27',fontsize=14)
    finish(fig,'04_process_transfer')

    fig,ax=plt.subplots(figsize=(10,4.8))
    y=np.arange(len(themes))
    ax.barh(y-.15,themes.coder1_positive,height=.3,color=colors[1],label='Кодировщик 1')
    ax.barh(y+.15,themes.coder2_positive,height=.3,color=colors[0],label='Кодировщик 2')
    ax.set_yticks(y,themes.label);ax.invert_yaxis();ax.set_xlim(0,13)
    ax.set_xticks(range(0,13,2));ax.set_xlabel('Интервью с отмеченной темой, из 12');ax.legend(loc='upper right')
    ax.set_title('H5 · наличие тем; временной переход не измерен');ax.grid(axis='x',alpha=.2)
    finish(fig,'05_interview_themes')

    fig,axes=plt.subplots(2,2,figsize=(10,7.3))
    for j,out in enumerate(['post','delayed']):
        ordinary=fit_ols(data,out,['group','pre'],False)
        ax=axes[j,0]
        for g in [1,0]:
            mask=data['group']==g
            ax.scatter(ordinary.fittedvalues[mask],ordinary.resid[mask],color=colors[g],alpha=.7,label=GROUP_NAMES[g])
        ax.axhline(0,color='black',ls='--');ax.set_xlabel('Предсказанный балл');ax.set_ylabel('Остаток')
        ax.set_title(('T1' if out=='post' else 'T2')+' · остатки');ax.legend(fontsize=7)
        sm.qqplot(ordinary.resid,line='45',fit=True,ax=axes[j,1],markerfacecolor=colors[1],markeredgecolor=colors[1],alpha=.6)
        axes[j,1].set_title(('T1' if out=='post' else 'T2')+' · Q–Q остатков')
    finish(fig,'06_diagnostics')


def make_report(directory,data,ancova,rubric,operations,corr,contrasts,regression_table,themes,hypotheses,audit,pooled_kappa,alt_correlations,resamples):
    primary=ancova[(ancova.model=='primary_pre_HC3')&ancova.outcome.isin(['post','delayed','transfer'])].copy()
    primary['outcome']=primary.outcome.map({'post':'T1, итоговый','delayed':'T2, отложенный','transfer':'TR, перенос (дополнительно)'})
    h3=rubric[rubric.criterion.isin(H3_KEYS)]
    h4=corr.copy(); h4['predictor']=h4.predictor.map({'quality':'Качество проверки','completeness':'Полнота журнала','requests':'Число запросов'})
    contrast_display = contrasts.copy()
    contrast_display['comparison'] = contrast_display.comparison.map({'quality minus requests':'Качество − запросы','completeness minus requests':'Полнота − запросы'})
    contrast_display['type'] = contrast_display.type.map({'standardized_beta_signed_HC3':'β HC3','pearson_signed_paired_bootstrap':'r, парный bootstrap','pearson_absolute_paired_bootstrap':'|r|, парный bootstrap'})
    regression_display = pd.DataFrame(regression_table)
    regression_display = regression_display[regression_display.term!='const'].copy()
    regression_display['term'] = regression_display.term.map({'quality':'Качество проверки','completeness':'Полнота журнала','requests':'Число запросов','pre':'Входной балл T0','experience':'Опыт программирования'})
    g1,g0=[data[data['group']==g] for g in [1,0]]
    rows=[]
    for var in ['pre','post','delayed','transfer','ai_literacy','artifact_total']:
        rows.append({'Показатель':MEASURES[var], 'Программа, n=27':f'{g1[var].mean():.2f} ± {g1[var].std(ddof=1):.2f}', 'Сравнение, n=29':f'{g0[var].mean():.2f} ± {g0[var].std(ddof=1):.2f}'})
    failures=[r for r in audit if r['result']=='FAIL']
    report=f'''# Проверка H1–H5 по 16 CSV

Анализ выполнен 10 октября 2026 года. Исходный снимок: [{SOURCE_COMMIT[:7]}]({REPO_URL}/tree/{SOURCE_COMMIT}). Все 16 исходных CSV сохранены без изменений. Таблицы пересчитаны из строк участников, ответов и экспертных оценок; готовые сводки использованы только для проверки согласованности.

## Главный вывод и происхождение данных

**Таблица численно поддерживает H1, различие T2 в H2 и четыре показателя H3. H4 не поддерживается. Для H5 доступны частоты тем, но переход к иной стратегии использования ИИ установить нельзя. Эти результаты не доказывают реальный эффект программы: первичное происхождение наблюдений не подтверждено.**

Это ограничение следует из структуры файлов и истории набора:

- В [историческом снимке {RECONSTRUCTION_MARKER_COMMIT[:7]}]({REPO_URL}/blob/{RECONSTRUCTION_MARKER_COMMIT}/csv/Сводка.csv) первая строка `csv/Сводка.csv`: «Воспроизводимая реконструкция чисел статьи». В текущем исходном снимке {SOURCE_COMMIT[:7]} этот заголовок и одна пустая строка удалены; численные строки и остальные 15 CSV не изменены. Поэтому заголовок цитируется как свидетельство истории набора, а не как строка текущего файла.
- `csv/Проверка_ссылок.csv`: способ — «макет проверки DOI или каталога», примечание — «структурная запись для воспроизведения числителя и знаменателя». Идентификаторы `REF:...` не являются реальными DOI или адресами источников.
- `csv/Ответы_интервью.csv` содержит 72 вопроса для 12 участников, но **не содержит ответов**. В тестах есть коды ответов и баллы, но нет текстов заданий/ответов.
- Нет протокола назначения групп, дат, интервала T1–T2, содержания новой задачи, данных о преподавателях и учебных кластерах. Метки `experimental`/`comparison` интерпретированы в соответствии с формулировкой пользователя; реализацию двух программ файлы не описывают.
- Для 29 участников группы сравнения отсутствуют все четыре процессных показателя ИИ; пропуски не заменялись нулями. Нет исходных артефактов или скриптов независимого запуска участников.

Следовательно, p-значения и доверительные интервалы ниже — условные характеристики предоставленной таблицы при принятых статистических предпосылках. Если наблюдения реконструированы под опубликованные числа, эти p-значения не дают новых независимых эмпирических доказательств. Этот отчёт нельзя представлять как верификацию первичного эксперимента.

{table_md(hypotheses,['hypothesis','status','limitation'],{'hypothesis':'Гипотеза','status':'Результат проверки','limitation':'Граница вывода'})}

## Данные и единицы анализа

56 участников: 27 в экспериментальной группе и 29 в группе сравнения. Во всех таблицах группы и наблюдения соединены по коду участника. Единица статистического анализа — **участник**, а не отдельный запрос к ИИ, ссылка или оценка эксперта. Оценки двух экспертов сначала усреднены внутри участника и критерия. Шкалы T0/T1/T2 — 0–100, перенос TR — 0–20, ИИ-грамотность — 0–40, экспертные критерии — 0–4. Максимумы тестовых шкал получены из сумм максимальных баллов заданий.

{table_md(pd.DataFrame(rows))}

Здесь M ± SD. Пропусков T0/T1/T2/TR или опыта программирования среди включённых 56 участников нет. Из файлов нельзя установить набор до включения и фактическое выбытие участников. Таблицы `01_data_inventory.csv`, `17_missingness.csv`, `18_data_dictionary.csv` описывают структуру и пропуски; словарь фиксирует исходные поля и типы, а не заменяет отсутствующий исследовательский протокол.

![Распределения баллов](figures/01_scores.png)

## H1: итоговый балл при контроле входного уровня

Основная модель: `T1 = β0 + β1·группа + β2·T0 + ε`, где группа 1 — структурированная программа, 0 — сравнение. Применены стандартные ошибки **HC3** и t-интервалы с остаточными степенями свободы. β1 — скорректированная разность групп. Входной уровень включён независимо от его статистической значимости. Аналогичная модель применена для H2.

## H2: отложенный тест

Основной исход H2 — балл T2 с контролем T0. Балл TR анализируется отдельно как дополнительный исход переноса; он не подменяет T2. Файл называет T2 отложенным тестом, но не позволяет проверить, через сколько времени и на какой новой задаче он выполнен. Поэтому установлено численное сохранение различия T2, а полная формулировка H2 требует протокола задач и дат.

{table_md(primary,['outcome','estimate','ci_low','ci_high','p','p_holm_primary_family'],{'outcome':'Исход','estimate':'Разность групп','ci_low':'95% ДИ: низ','ci_high':'95% ДИ: верх','p':'p HC3','p_holm_primary_family':'p Holm'})}

![Скорректированные различия](figures/02_adjusted_effects.png)

Шесть основных проверок образуют одну семью: T1, T2 и четыре критерия H3. Для них использована поправка **Holm**. Проверки двусторонние, α=0,05; обычные 95% ДИ не являются одновременными интервалами для всей семьи. TR и ИИ-грамотность — дополнительные исходы. Наличие контроля T0 не устраняет различия в мотивации, преподавателях или других неизмеренных факторах.

В `04_ancova.csv` также приведены модели с опытом программирования и классическими OLS-ошибками. Модель с T0 и опытом воспроизводит старую сводку: ΔT1=9,962, ΔT2=9,925; основная модель выше контролирует только T0 и использует HC3. Различие результатов объясняется набором ковариат и оценкой ошибок. OLS-модель служит проверкой чисел уже существующей сводки. Проверки остатков, взаимодействия группа×T0 и влияния наблюдений сохранены в `05_model_diagnostics.csv`. Незначимый тест взаимодействия не доказывает равенство наклонов. Если группы соответствуют отдельным классам, индивидуальные HC3-ошибки не учитывают кластеризацию; идентификаторов классов в данных нет.

## H3: качество проектных артефактов

Основные показатели — четыре заранее названных в гипотезе критерия. Для каждого участника взято среднее оценок двух экспертов. Сравнение средних выполнено тестом Welch, ДИ разности построен с его степенями свободы. **g Хеджеса** — нескорректированный стандартизованный эффект, рассчитанный через объединённое SD участников с поправкой на малую выборку. Четыре p-значения включены в основную семью Holm вместе с H1/H2. Остальные критерии и сумма рубрики — описательный дополнительный анализ. Проверка Mann–Whitney сохранена как чувствительность к порядковой природе шкалы.

{table_md(h3,['label','mean_exp','mean_comp','difference','ci_low','ci_high','hedges_g','p_holm_primary_family'],{'label':'Критерий','mean_exp':'Программа, M','mean_comp':'Сравнение, M','difference':'Δ','ci_low':'ДИ низ','ci_high':'ДИ верх','hedges_g':'g','p_holm_primary_family':'p Holm'})}

![Оценки артефактов](figures/03_artifact_domains.png)

Все четыре разности положительны. Это свидетельство различий **зарегистрированных оценок**, но не независимая проверка источников, происхождения данных или реального запуска проектов: исходных проектов и доказательств этих действий в репозитории нет.

Дополнительные операционные показатели рассчитаны на уровне участников:

{table_md(operations,['label','mean_exp','mean_comp','difference','ci_low','ci_high','p_holm_exploratory'],{'label':'Показатель','mean_exp':'Программа','mean_comp':'Сравнение','difference':'Δ','ci_low':'ДИ низ','ci_high':'ДИ верх','p_holm_exploratory':'p Holm, доп.'})}

Доли в таблице выражены от 0 до 1; разность 0,10 означает 10 процентных пунктов. Проверенные ссылки: {int(g1.links_verified.sum())}/{int(g1.links_total.sum())} и {int(g0.links_verified.sum())}/{int(g0.links_total.sum())} в сумме по группам; значения таблицы — средние индивидуальных долей, поэтому могут отличаться от этих агрегированных отношений. Воспроизведение: {int(g1.reproduced.sum())}/27 и {int(g0.reproduced.sum())}/29; обоснование модели: {int(g1.model_justified.sum())}/27 и {int(g0.model_justified.sum())}/29. Для бинарных признаков использован точный тест Fisher, для доли ссылок и частоты утверждений — Mann–Whitney. Их p-значения скорректированы Holm в отдельной дополнительной семье из четырёх проверок. ДИ разностей — стратифицированный по группе bootstrap участников, {resamples:,} повторов. ДИ относится к разности средних, а Mann–Whitney проверяет распределения, поэтому они отвечают на разные статистические вопросы.

## H4: связь процесса использования ИИ с переносом

Анализ ограничен **27 участниками экспериментальной группы**. Полнота, качество проверки и число запросов не измерены в группе сравнения. Основные процессные показатели взяты из `Участники.csv`; это агрегаты без опубликованной формулы их получения.

{table_md(h4,['predictor','pearson_r','pearson_ci_low','pearson_ci_high','pearson_p','spearman_rho','spearman_p_permutation'],{'predictor':'Показатель','pearson_r':'r Pearson','pearson_ci_low':'ДИ низ','pearson_ci_high':'ДИ верх','pearson_p':'p Pearson','spearman_rho':'ρ Spearman','spearman_p_permutation':'p перестановки'})}

95% ДИ корреляций — bootstrap с повторной выборкой целых участников; p Spearman — перестановочный тест пар, {resamples:,} повторов, с сохранением связок рангов. Это разведочные проверки. Все три линейные связи близки к нулю; отсутствие значимости не устанавливает отсутствия связи в популяции.

![Процесс и перенос](figures/04_process_transfer.png)

Для сравнения «сильнее, чем число запросов» недостаточно сравнить два отдельных p-значения. Прямо оценены зависимые разности `r(качество, TR) − r(запросы, TR)` и `r(полнота, TR) − r(запросы, TR)`, а также разности абсолютных корреляций, с совместным bootstrap участников. Кроме того, рассчитана стандартизованная множественная регрессия `TR ~ качество + полнота + запросы + T0 + опыт` с HC3. Стандартизация всех пяти предикторов и исхода выполнена внутри этой группы, SD с ddof=1. Две разности условных β проверены непосредственно Wald t-тестами и скорректированы Holm. Подписанные β оценивают направление условной связи, а разности абсолютных r — силу связи без направления.

{table_md(regression_display,['term','estimate','ci_low','ci_high','p'],{'term':'Предиктор','estimate':'Стандартизованный β','ci_low':'ДИ низ','ci_high':'ДИ верх','p':'p HC3'})}

{table_md(contrast_display,['comparison','type','difference','ci_low','ci_high','p_holm_two_beta_contrasts'],{'comparison':'Сравнение','type':'Оценка','difference':'Разность','ci_low':'ДИ низ','ci_high':'ДИ верх','p_holm_two_beta_contrasts':'p Holm'})}

**H4 не поддерживается:** интервалы прямых разностей включают ноль. Детали β, VIF и R² сохранены в `09_h4_regression.csv`. n=27 при пяти предикторах даёт широкую неопределённость; процессные показатели наблюдательные и могут зависеть от способностей и сложности задания. Они не устанавливают причинный механизм программы.

Средние оценки из `Журнал_ИИ.csv` немного расходятся с агрегатами участника. Формула агрегирования отсутствует; основной анализ не исправлял исходные значения. Чувствительность к средним по журналу: r качества={alt_correlations['quality']:.3f}, r полноты={alt_correlations['completeness']:.3f}, r числа строк={alt_correlations['requests']:.3f}. Вывод H4 остаётся тем же. Сопоставление по каждому участнику — `11_h4_log_audit.csv`. Всего 962 строки журнала; они не считаются 962 независимыми участниками.

## H5: темы интервью и пределы качественного вывода

[План сбора подлинных интервью, сценарий и учебные примеры](../h5_interviews/README.md). Все DEMO вымышлены и не входят в результаты ниже.

Доступны 12 интервью **только экспериментальной группы**, каждое с шестью бинарными темами и двумя кодировщиками. Нет текстов ответов, начальных интервью или контрольной группы. Поэтому можно описать наличие тем, но нельзя реконструировать цитаты или доказать **переход** к использованию ИИ как критика/источника альтернатив. «Затраты времени» — метка темы, а не измерение затраченных минут. Код ошибки в журнале не даёт времени или контекста изменения стратегии.

{table_md(themes,['label','coder1_positive','coder2_positive','both_positive','disagreements','kappa'],{'label':'Тема','coder1_positive':'Кодировщик 1 /12','coder2_positive':'Кодировщик 2 /12','both_positive':'Оба отметили /12','disagreements':'Разногласий','kappa':'κ'})}

Сводка `Темы_интервью.csv` воспроизводит **кодировщика 1**, а не согласованное итоговое решение. Разногласия нельзя автоматически разрешать в пользу подтверждения гипотезы. Особенно различается риск чрезмерного доверия: 4/12 у первого и 8/12 у второго кодировщика; окончательное число требует согласования с транскриптами.

Общее согласие — {raw_agreement(themes):.1%}, объединённая κ по 72 меткам — {pooled_kappa:.3f}. Объединение разных тем может скрывать тематические разногласия; эти метки также зависимы внутри интервью. У темы «Привычка проверять» оба кодировщика отметили все интервью, поэтому κ не определена из-за отсутствия вариации, а не равна единице. Wilson-интервалы частот приведены в CSV как описательные ориентиры; неизвестный способ отбора 12 интервью не позволяет переносить их на всех участников.

![Темы и кодировщики](figures/05_interview_themes.png)

**Итог H5:** коды согласуются с наличием критики, альтернатив, затрат времени и риска доверия. Качественное свидетельство изменения стратегии требует транскриптов, фрагментов с датами/идентификаторами, правила отбора интервью и описанного процесса согласования кодирования. В этом анализе цитаты не выдумывались.

## Проверка данных и моделей

Суммы баллов по 2 352 строкам заданий совпадают с 280 итоговыми значениями пяти инструментов. 896 экспертных оценок и 678 записей ссылок согласуются с таблицей участников; это подтверждает внутреннюю согласованность, но не первичную достоверность. Существующие 62 строки `Проверки.csv` не использованы как внешнее подтверждение; исходные агрегаты пересчитаны самостоятельно.

Найдены следующие расхождения при проверке простого среднего журнальных оценок (они могут объясняться другой, неописанной агрегацией):

{table_md(pd.DataFrame(failures))}

Диагностика H1/H2:

![Диагностика моделей](figures/06_diagnostics.png)

## Воспроизведение и файлы

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/run_analysis.py --root .
python analysis/validate_results.py --root .
```

Использован Python 3.12, seed={SEED}, {resamples:,} bootstrap/перестановочных повторов. Скрипт пересоздаёт только файлы в `analysis/`, проверяет ключи и шкалы, не редактирует `csv/` или старые `figures/`. Пакеты и хеши SHA-256 исходников записаны в `manifest.json`, проверенный снимок закреплён в `source_sha256.json`. При расчёте хешей CRLF нормализуется в LF, BOM сохраняется; это обеспечивает одинаковую проверку Git-выгрузок в Windows и Linux, не меняя исходные файлы. `results.json` содержит ключевые численные результаты. `tables/` содержит 18 CSV с UTF-8 BOM; `figures/` — шесть рисунков в PNG и SVG. Все оценки рассчитаны в Python, CSV — сохранённые результаты. Повторный запуск воспроизводит этот снимок; если исходные данные заменены, код останавливается до пересмотра плана анализа и обновления хешей. Текст отчёта и выводы написаны для данного снимка. Проверки и методы выбраны после ознакомления с доступными файлами, регистрации плана до эксперимента нет.

## Источники методов

- [statsmodels: HC3 и робастная ковариация](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html) — выбор ковариационной матрицы для линейной модели.
- [statsmodels: поправки множественных проверок](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) — Holm.
- [SciPy: bootstrap и парное ресэмплирование](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) — принцип сохранения пар; здесь ресэмплирование реализовано явно для совместных разностей.
- [SciPy: Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) — перестановочная проверка для небольшой выборки.

Для полноценной проверки заявленного реального эксперимента нужны первичные обезличенные наблюдения с документированным происхождением, протокол распределения групп и задач, исходные артефакты и транскрипты. Наличие этих материалов изменит доказательную силу анализа; оно не может быть заменено точным воспроизведением сводных чисел.
'''
    (directory/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')


def raw_agreement(themes):
    return 1 - themes.disagreements.sum()/themes.n_interviews.sum()


if __name__=='__main__':
    main()
