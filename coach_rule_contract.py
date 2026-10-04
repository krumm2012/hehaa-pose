"""Code-owned approval for independently validated automatic teaching rules.

Observation eligibility, a model confidence, historical labels, and manual
scores cannot approve an automatic rule. Keep capture/review guidance available.
"""
POLICY_VERSION = 'automatic_coach_v2_validated_rules_only'
EXCLUSION_REASON = 'automatic_technique_rule_not_independently_validated'
VALIDATED_TECHNIQUE_RULES = frozenset()


def technique_rule_approved(rule_id):
    return rule_id in VALIDATED_TECHNIQUE_RULES


def automatic_coach_policy():
    return {'policy_version': POLICY_VERSION,
            'validated_rule_ids': sorted(VALIDATED_TECHNIQUE_RULES),
            'approval_source': 'code_owned_independent_validation',
            'exclusion_reason': EXCLUSION_REASON,
            'accuracy_validated': False}
