from expv2.beta_estimation import build_interval_rows


def _row(step, trace, signal, noisy):
    return dict(seed=42, run_id="x", method="dp_sgd", learning_rate=0.1, step=step, layer="fc1",
                r=0.01, dimension=100, trace_A=1, trace_G=trace, trace_F=trace,
                clean_signal_energy=signal, noisy_gradient_energy=noisy,
                expected_noise_energy=1, noise_debiased_energy_raw=noisy - 1,
                beta_oracle_step=signal / trace, beta_dp_step_raw=(noisy - 1) / trace,
                beta_dp_step_positive=max((noisy - 1) / trace, 0), beta_dp_step_negative=False)


def test_pooled_ratio_of_sums():
    rows = build_interval_rows([_row(0, 1, 1, 2), _row(1, 100, 50, 51)], 2)
    pooled = rows[0]
    assert pooled["beta_oracle"] == 51 / 101
    assert pooled["beta_oracle"] != (1 / 1 + 50 / 100) / 2


def test_partial_final_window():
    rows = []
    for step in range(1170):
        rows.append(_row(step, 10, 5, 6))
    intervals = build_interval_rows(rows, 50)
    assert len(intervals) == 24
    last = intervals[-1]
    assert last["start_step"] == 1150 and last["end_step"] == 1169 and last["n_steps"] == 20
