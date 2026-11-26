from experiments.ego_immune_recognition.riskbench_metadata.guarded_epoch_sweep import (
    checkpoint_selection_key,
    parse_eval_epochs,
)


def _row(
    epoch: int,
    method: str,
    *,
    recall: float,
    auroc: float,
    fpr: float,
) -> dict:
    return {
        "epoch": epoch,
        "split": "validation",
        "method": method,
        "event_recall_at_threshold": recall,
        "frame_auroc": auroc,
        "normal_false_alarm_rate_at_threshold": fpr,
    }


def test_parse_eval_epochs_keeps_final_epoch() -> None:
    assert parse_eval_epochs("1,3,5,10,20", max_epoch=10) == [1, 3, 5, 10]


def test_selection_rejects_excess_validation_fpr() -> None:
    rows = [
        _row(3, "ttc_percentile", recall=0.80, auroc=0.80, fpr=0.05),
        _row(3, "persistent_guarded", recall=0.90, auroc=0.90, fpr=0.08),
        _row(5, "ttc_percentile", recall=0.80, auroc=0.80, fpr=0.05),
        _row(5, "persistent_guarded", recall=0.82, auroc=0.82, fpr=0.055),
    ]

    assert checkpoint_selection_key(rows, 5) > checkpoint_selection_key(rows, 3)


def test_selection_prefers_guarded_event_recall_when_eligible() -> None:
    rows = [
        _row(3, "ttc_percentile", recall=0.80, auroc=0.80, fpr=0.05),
        _row(3, "persistent_guarded", recall=0.82, auroc=0.85, fpr=0.055),
        _row(5, "ttc_percentile", recall=0.80, auroc=0.80, fpr=0.05),
        _row(5, "persistent_guarded", recall=0.84, auroc=0.82, fpr=0.058),
    ]

    assert checkpoint_selection_key(rows, 5) > checkpoint_selection_key(rows, 3)
