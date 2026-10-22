import json
from pathlib import Path

from eval.business_oracles import expected_answers, grade_business_answers

ROOT = Path(__file__).resolve().parent.parent


def test_retail_numbers_must_match_even_when_report_and_charts_complete():
    sources = [str(ROOT / 'data/retail/orders_eu_2011.csv'), str(ROOT / 'data/retail/products.csv')]
    answer = expected_answers('retail-gmv', sources)
    assert answer['gmv_months'] == ['2011-10', '2011-11']
    assert answer['top_return_sku'] == '22423'
    def report(value):
        return '# Completed report\n```json\n' + json.dumps({'business_answers': value}) + '\n```'
    assert grade_business_answers('retail-gmv', report(answer), sources)['passed']
    wrong = {**answer, 'gmv_growth_pct': -74.4}
    assert not grade_business_answers('retail-gmv', report(wrong), sources)['passed']
    assert not grade_business_answers('retail-gmv', '# Completed report, three charts', sources)['passed']


def test_toy_oracle_is_independent_of_agent_output():
    sources = [str(ROOT / 'data/customers.csv'), str(ROOT / 'data/orders.csv')]
    expected = expected_answers('toy-regions', sources)
    assert expected == {'high_value_sales_by_region': {'East': 470.0}}
    assert not grade_business_answers('toy-regions', '```json\n{"business_answers":{"high_value_sales_by_region":{"East":471}}}\n```', sources)['passed']
