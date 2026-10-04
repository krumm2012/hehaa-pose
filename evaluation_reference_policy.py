"""Qualify reference agreement without treating assisted review as ground truth."""

from observation_policy import finite_number

POLICY_VERSION = 'evaluation_reference_v1_review_agreement'
REFERENCE_NOTE = '显示参考对照结果；独立准确性未验证。触球容差按源帧号，曝光未核验。'


def reference_note(evaluation):
    method = (evaluation.get('reference_provenance') or {}).get('declared_method')
    labels = {'model_assisted_review': '模型辅助复核',
              'independent_annotation': '声明为独立标注（未核验）',
              'automated_contract_fixture': '自动测试参考'}
    return '参考来源：' + labels.get(method, '来源未记录') + '。' + REFERENCE_NOTE


def qualify_reference_comparison(report, annotations):
    summary = report['summary']
    summary['stroke_type_match_ratio'] = summary.get('stroke_type_accuracy')
    summary['contact_within_tolerance_ratio'] = summary.get('contact_accuracy')
    # Reserved compatibility keys: the present evaluator cannot verify independent truth.
    summary['stroke_type_accuracy'] = None
    summary['contact_accuracy'] = None
    source = annotations.get('source')
    source = source if isinstance(source, dict) else {}
    method = source.get('reference_method')
    method = method if isinstance(method, str) else 'legacy_unspecified'
    complete = summary.get('metrics_finalized') is True and summary.get('provisional') is False
    report['reference_provenance'] = {
        'policy_version': POLICY_VERSION,
        'declared_method': method,
        'model_predictions_visible': source.get('model_predictions_visible')
            if type(source.get('model_predictions_visible')) is bool else None,
        'comparison_review_complete': complete,
        'independence_verified': False,
        'accuracy_validated': False,
        'reasons': ['independent_reference_not_verified'] + ([] if complete else ['reference_review_incomplete_or_unrecorded']),
        'semantics': 'agreement_with_reference_not_independent_accuracy',
    }
    return report


def comparison_metric_rows(evaluation):
    summary = evaluation.get('summary') or {}
    complete = summary.get('metrics_finalized') is True and summary.get('provisional') is False
    values = [('事件Precision（对照）', summary.get('precision')),
              ('事件Recall（对照）', summary.get('recall')),
              ('事件F1（对照）', summary.get('f1')),
              ('类型匹配比例', summary.get('stroke_type_match_ratio', summary.get('stroke_type_accuracy'))),
              ('触球帧容差匹配比例', summary.get('contact_within_tolerance_ratio', summary.get('contact_accuracy')))]
    rows = []
    for label, value in values:
        number = finite_number(value)
        text = '待复核' if not complete else (f'{number:.1%}' if number is not None and 0 <= number <= 1 else '不可核验')
        rows.append((label, text))
    return rows


def reference_metric_script():
    """Shared display contract for generated live reports, including historical JSON."""
    return r'''
    function reviewMetricRows(evaluation) {
      const summary = evaluation?.summary || {};
      const complete = summary.metrics_finalized === true && summary.provisional === false;
      const has = (key) => Object.prototype.hasOwnProperty.call(summary, key);
      const ratio = (key, oldKey) => has(key) ? summary[key] : summary[oldKey];
      const percent = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
        ? `${(value * 100).toFixed(1)}%` : '不可核验';
      return [
        ['事件Precision（对照）', summary.precision],
        ['事件Recall（对照）', summary.recall],
        ['事件F1（对照）', summary.f1],
        ['类型匹配比例', ratio('stroke_type_match_ratio','stroke_type_accuracy')],
        ['触球帧容差匹配比例', ratio('contact_within_tolerance_ratio','contact_accuracy')],
      ].map(([label,value]) => [label, complete ? percent(value) : '待复核']);
    }
    function reviewReferenceNote(evaluation) {
      const labels = {model_assisted_review:'模型辅助复核', independent_annotation:'声明为独立标注（未核验）', automated_contract_fixture:'自动测试参考'};
      const method = evaluation?.reference_provenance?.declared_method;
      const source = labels[method] || '来源未记录';
      return `参考来源：${source}。显示参考对照结果；独立准确性未验证。触球容差按源帧号，曝光未核验。`;
    }
'''
