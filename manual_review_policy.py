"""Business rules for human review, separate from model evidence and coaching."""
from html import escape
from practice_scoring import POLICY, score_review

VERSION = 'human_coach_review_v3'
CONTACT_STATES = {'unreviewed', 'confirmed', 'uncertain', 'not_applicable'}
VERDICTS = {'unreviewed', 'supported', 'rejected', 'insufficient'}
COACH_TEXT_FIELDS = ('reviewer', 'observed_fact', 'suggestion', 'evidence')


def validate_review_fields(annotation):
    contact = annotation.get('contact_review', 'unreviewed')
    if not isinstance(contact, str) or contact not in CONTACT_STATES:
        raise ValueError('触球审核状态不支持')
    if contact == 'confirmed' and (not annotation.get('valid_hit', True)
            or (annotation.get('frames') or {}).get('contact') is None):
        raise ValueError('确认触球需要有效事件和源帧号')
    if contact == 'not_applicable' and annotation.get('valid_hit', True):
        raise ValueError('非击球事件请同时取消有效击球，避免审核结论矛盾')
    coach = annotation.get('coach_review', {})
    if not isinstance(coach, dict):
        raise ValueError('教练审核必须是对象')
    verdict = coach.get('verdict', 'unreviewed')
    if not isinstance(verdict, str) or verdict not in VERDICTS:
        raise ValueError('教练建议审核状态不支持')
    for key in COACH_TEXT_FIELDS:
        if key in coach and (not isinstance(coach[key], str) or len(coach[key]) > 4000):
            raise ValueError('教练审核文本必须是至多 4000 字符的字符串')
    if verdict != 'unreviewed' and not all(str(coach.get(k) or '').strip()
                                        for k in ('reviewer', 'observed_fact', 'evidence')):
        raise ValueError('提交教练判断需要审核人、可见事实及证据帧/视角')
    review = annotation.get('practice_review')
    if review is not None:
        if not isinstance(review, dict) or type(review.get('confirmed', False)) is not bool:
            raise ValueError('教练评分和确认必须使用对象及布尔值')
        result = score_review(review)
        if review.get('confirmed'):
            if result['coverage'] != 1:
                raise ValueError('确认总分需要五项完整评分；看不清的项目请留空并保留草稿')
            if verdict == 'unreviewed' or contact != 'confirmed' or annotation.get('needs_review'):
                raise ValueError('确认总分前需完成事件、触球和教练建议审核')
    return contact


def review_summary(annotations):
    rows = list(annotations)
    return {'policy_version': VERSION, 'event_count': len(rows),
            'contact_confirmed_count': sum(r.get('contact_review') == 'confirmed' for r in rows),
            'coach_reviewed_count': sum((r.get('coach_review') or {}).get('verdict', 'unreviewed') != 'unreviewed' for r in rows),
            'coach_score_confirmed_count': sum((r.get('practice_review') or {}).get('confirmed') is True for r in rows)}


def review_fields_html():
    ratings = []
    for dimension in POLICY['dimensions']:
        options = '<option value="">看不清 / 未评</option>' + ''.join(
            f'<option value="{i}">{i} · {escape(anchor)}</option>'
            for i, anchor in enumerate(dimension['anchors'], 1))
        ratings.append(f'<label>{escape(dimension["label"])}<select data-rating="{dimension["id"]}">{options}</select></label>')
    return '''<div class="human-review-fields">
      <label>触球证据审核<select data-field="contact_review">
        <option value="unreviewed">未审核（帧号仅为候选）</option><option value="confirmed">已从连续帧确认触球</option>
        <option value="uncertain">遮挡 / 看不清，无法确认</option><option value="not_applicable">非击球事件 / 不适用</option></select></label>
      <details class="human-coach-editor"><summary>人工教练审核与五维评分</summary>
      <p>事件有效、类型明确、触球确认分别审核。评分留空表示证据不足；不自动用模型分填充。</p>
      <label>审核人<input data-coach="reviewer" maxlength="4000" placeholder="教练姓名或审核代号"></label>
      <label>系统建议审核<select data-coach="verdict"><option value="unreviewed">未审核</option><option value="supported">证据支持</option><option value="rejected">不采纳</option><option value="insufficient">证据不足，暂不判断</option></select></label>
      <label>可见事实<textarea data-coach="observed_fact" maxlength="4000" rows="2"></textarea></label>
      <label>证据帧与视角<input data-coach="evidence" maxlength="4000" placeholder="例如 F95–F108，正面；镜像辅助"></label>
      <label>本次教练建议<textarea data-coach="suggestion" maxlength="4000" rows="2" placeholder="一条可执行建议；无可靠结论可留空"></textarea></label>
      <div class="manual-rating-grid">''' + ''.join(ratings) + '''</div>
      <label><input type="checkbox" data-field="practice_confirmed">确认五维评分（须五项完整、触球已确认、事件复核完成）</label>
      <small>人工评分使用现有练习量表，尚待独立验证，不代表专业等级；不会自动训练模型或修改实时规则。</small>
      </details></div>'''


def review_contract_script():
    import json
    return '''
    function humanReviewFields() {
      return ''' + json.dumps(review_fields_html(), ensure_ascii=False) + ''';
    }
    function humanReviewFromCard(card) {
      const field = card.querySelector('[data-field="contact_review"]');
      const coach = {};
      card.querySelectorAll('[data-coach]').forEach(input => { coach[input.dataset.coach] = input.value.trim(); });
      const ratings = {};
      card.querySelectorAll('[data-rating]').forEach(input => { ratings[input.dataset.rating] = input.value === '' ? null : Number(input.value); });
      return {contact_review:field?.value || 'unreviewed', coach_review:coach,
        practice_review:{ratings, confirmed:card.querySelector('[data-field="practice_confirmed"]')?.checked === true}};
    }
    function applyHumanReview(card, annotation) {
      const field = card.querySelector('[data-field="contact_review"]');
      if (field) field.value = annotation.contact_review || 'unreviewed';
      card.querySelectorAll('[data-coach]').forEach(input => {
        input.value = annotation.coach_review?.[input.dataset.coach] ?? (input.dataset.coach === 'verdict' ? 'unreviewed' : '');
      });
      card.querySelectorAll('[data-rating]').forEach(input => {
        input.value = annotation.practice_review?.ratings?.[input.dataset.rating] ?? '';
      });
      const confirmed = card.querySelector('[data-field="practice_confirmed"]');
      if (confirmed) confirmed.checked = annotation.practice_review?.confirmed === true;
    }
    function validateHumanReview(annotation) {
      const contact = Object.prototype.hasOwnProperty.call(annotation, 'contact_review') ? annotation.contact_review : 'unreviewed';
      if (!['unreviewed','confirmed','uncertain','not_applicable'].includes(contact)) throw new Error('触球审核状态不支持');
      if (contact === 'confirmed' && (annotation.valid_hit === false || annotation.frames?.contact == null)) throw new Error('确认触球需要有效事件和源帧号');
      if (contact === 'not_applicable' && annotation.valid_hit !== false) throw new Error('非击球事件请同时取消有效击球');
      const coach = Object.prototype.hasOwnProperty.call(annotation, 'coach_review') ? annotation.coach_review : {};
      if (coach === null || typeof coach !== 'object' || Array.isArray(coach)) throw new Error('教练审核必须是对象');
      const verdict = coach.verdict ?? 'unreviewed';
      if (!['unreviewed','supported','rejected','insufficient'].includes(verdict)) throw new Error('教练建议审核状态不支持');
      for (const key of ['reviewer','observed_fact','suggestion','evidence']) {
        if (Object.prototype.hasOwnProperty.call(coach,key) && (typeof coach[key] !== 'string' || coach[key].length > 4000)) throw new Error('教练审核文本必须是至多4000字符的字符串');
      }
      if (verdict !== 'unreviewed' && !['reviewer','observed_fact','evidence'].every(k => coach[k]?.trim())) throw new Error('提交教练判断需要审核人、可见事实及证据帧/视角');
      const practice = annotation.practice_review;
      if (practice != null) {
        if (typeof practice !== 'object' || Array.isArray(practice) || (practice.confirmed !== undefined && typeof practice.confirmed !== 'boolean')) throw new Error('教练评分及确认格式不正确');
        const ratings = practice.ratings ?? {};
        if (typeof ratings !== 'object' || Array.isArray(ratings)) throw new Error('评分必须是对象');
        const dimensions = ['preparation','positioning','contact','coordination','recovery'];
        for (const [key,value] of Object.entries(ratings)) {
          if (!dimensions.includes(key) || (value !== null && (!Number.isInteger(value) || value < 1 || value > 5))) throw new Error('评分须为1至5整数；看不清请留空');
        }
        if (practice.confirmed && (dimensions.some(k => ratings[k] == null) || contact !== 'confirmed' || verdict === 'unreviewed' || annotation.needs_review)) throw new Error('确认总分需要五项完整评分及事件、触球、教练建议审核');
      }
    }
'''
