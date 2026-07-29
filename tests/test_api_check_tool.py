from tools.check_cqlib2_api import main


def test_cqlib2_api_check_uses_supported_statevector_expectation(capsys):
    assert main() == 0
    output = capsys.readouterr().out
    assert "[PASS] optional native Statevector.apply_pauli_rotation is available" in output
