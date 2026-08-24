import main


def test_run_with_valid_payload():
    payload = {'dry_run': False}
    result = main.run(payload)
    assert isinstance(result, dict), 'run() did not return a dictionary'
    assert 'ok' in result and 'error' in result, 'run() returned an unexpected format'


def test_run_with_dry_run():
    payload = {'dry_run': True}
    result = main.run(payload)
    assert isinstance(result, dict), 'run() did not return a dictionary'
    assert 'ok' in result and 'error' in result, 'run() returned an unexpected format'
    payload = {'dry_run': True}
    result = main.run(payload)
    assert isinstance(result, dict), 'run() did not return a dictionary'
    assert 'ok' in result and 'error' in result, 'run() returned an unexpected format'