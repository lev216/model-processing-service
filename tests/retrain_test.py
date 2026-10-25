import json
from datetime import UTC, datetime, timedelta

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app import retrain


def write_model(models_dir, name, uid, interval):
    path = models_dir / name
    path.mkdir(parents=True)
    (path / "meta.json").write_text(json.dumps({"instrument_uid": uid, "interval": interval}))


def test_trained_pairs_lists_serving_models_only(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain.settings, "models_dir", str(tmp_path))
    write_model(tmp_path, "uid-1__1DAY", "uid-1", "1DAY")
    write_model(tmp_path, "uid-1__1DAY__previous", "uid-1", "1DAY")
    write_model(tmp_path, "uid-1__1DAY__candidate", "uid-1", "1DAY")
    write_model(tmp_path, "uid-2__1HOUR", "uid-2", "1HOUR")
    (tmp_path / "broken__1DAY").mkdir()

    assert retrain.trained_pairs() == [("uid-1", "1DAY"), ("uid-2", "1HOUR")]


async def test_retrain_all_models_skips_pairs_without_new_candles(monkeypatch):
    last_runs = {"old": datetime(2026, 10, 1, tzinfo=UTC), "fails": None, "fresh": None}
    retrained = []

    async def last_run(uid, interval):
        return last_runs[uid]

    async def has_new_candles(uid, interval, since):
        return False

    async def run(uid, interval):
        retrained.append(uid)
        if uid == "fails":
            raise RuntimeError("boom")
        return {"status": "done"}

    monkeypatch.setattr(
        retrain, "trained_pairs", lambda: [("old", "1DAY"), ("fails", "1DAY"), ("fresh", "1DAY")]
    )
    monkeypatch.setattr(retrain, "fetch_last_run_started_at", last_run)
    monkeypatch.setattr(retrain, "has_candles_since", has_new_candles)
    monkeypatch.setattr(retrain, "run_retrain", run)

    await retrain.retrain_all_models()

    assert retrained == ["fails", "fresh"]


async def test_run_retrain_refuses_parallel_training(monkeypatch):
    async def active(uid, interval):
        return 7

    monkeypatch.setattr(retrain, "fetch_active_run", active)

    with pytest.raises(retrain.TrainingInProgress):
        await retrain.run_retrain("uid", "1DAY")


@pytest.mark.parametrize("status, cache_dropped", [("done", True), ("rejected", False)])
async def test_run_retrain_drops_cache_only_when_model_replaced(monkeypatch, status, cache_dropped):
    audits = {}

    async def no_active(uid, interval):
        return None

    async def candles(uid, interval):
        return "candles"

    async def start(**kwargs):
        return 42

    async def done(**kwargs):
        audits.update(kwargs)

    def retrain_model(**kwargs):
        return {
            "status": status,
            "reason": None if status == "done" else "worse",
            "comparison": {"candidate_mape": 0.01},
            "training_samples": 100,
            "candles_from": "2026-01-01T00:00:00",
            "candles_to": "2026-10-01T00:00:00",
            "best_model": "ModelA",
            "mape": 0.01,
            "leaderboard": [],
        }

    monkeypatch.setattr(retrain, "fetch_active_run", no_active)
    monkeypatch.setattr(retrain, "fetch_candles", candles)
    monkeypatch.setattr(retrain, "audit_start", start)
    monkeypatch.setattr(retrain, "audit_done", done)
    monkeypatch.setattr(retrain, "retrain_model", retrain_model)
    monkeypatch.setitem(retrain._PREDICTOR_CACHE, "uid_1DAY", object())

    result = await retrain.run_retrain("uid", "1DAY")

    assert result["run_id"] == 42
    assert result["status"] == status
    assert audits["status"] == status
    assert ("uid_1DAY" not in retrain._PREDICTOR_CACHE) is cache_dropped


def test_schedule_retrain_registers_nightly_job(monkeypatch):
    scheduler = AsyncIOScheduler()
    monkeypatch.setattr(retrain, "scheduler", scheduler)
    monkeypatch.setattr(retrain.settings, "retrain_cron", "0 3 * * *")
    monkeypatch.setattr(retrain.settings, "retrain_timezone", "Europe/Moscow")

    retrain.schedule_retrain()

    job = scheduler.get_job(retrain.RETRAIN_JOB_ID)
    assert job is not None
    assert str(job.trigger.timezone) == "Europe/Moscow"
    fields = {field.name: str(field) for field in job.trigger.fields}
    assert (fields["hour"], fields["minute"], fields["day"]) == ("3", "0", "*")


def test_schedule_retrain_can_be_disabled(monkeypatch):
    scheduler = AsyncIOScheduler()
    monkeypatch.setattr(retrain, "scheduler", scheduler)
    monkeypatch.setattr(retrain.settings, "retrain_cron", "")
    monkeypatch.setattr(retrain.settings, "retrain_error_check_cron", "")

    retrain.schedule_retrain()

    assert scheduler.get_jobs() == []


def test_schedule_retrain_registers_error_check(monkeypatch):
    scheduler = AsyncIOScheduler()
    monkeypatch.setattr(retrain, "scheduler", scheduler)
    monkeypatch.setattr(retrain.settings, "retrain_cron", "")
    monkeypatch.setattr(retrain.settings, "retrain_error_check_cron", "15 * * * *")

    retrain.schedule_retrain()

    job = scheduler.get_job(retrain.ERROR_CHECK_JOB_ID)
    fields = {field.name: str(field) for field in job.trigger.fields}
    assert (fields["hour"], fields["minute"]) == ("*", "15")
    assert scheduler.get_job(retrain.RETRAIN_JOB_ID) is None


@pytest.fixture
def error_check(monkeypatch):
    """Stubs for check_realized_error; tests adjust `state` and read `retrained`."""
    state = {"mape": 0.05, "points": 15, "last_run": None, "version": "2026-10-01T03:00:00"}
    retrained = []

    async def fill(uid, interval):
        return 0

    async def realized(uid, interval, version, window):
        assert version == state["version"]
        return state["mape"], state["points"]

    async def last_run(uid, interval):
        return state["last_run"]

    async def run(uid, interval):
        retrained.append(uid)
        return {"status": "done"}

    monkeypatch.setattr(retrain, "fill_prediction_actuals", fill)
    monkeypatch.setattr(retrain, "fetch_realized_error", realized)
    monkeypatch.setattr(retrain, "fetch_last_run_started_at", last_run)
    monkeypatch.setattr(retrain, "run_retrain", run)
    monkeypatch.setattr(retrain, "_serving_model_version", lambda uid, interval: state["version"])
    monkeypatch.setattr(retrain.settings, "retrain_max_realized_mape", 0.03)
    monkeypatch.setattr(retrain.settings, "realized_error_min_points", 10)
    monkeypatch.setattr(retrain.settings, "retrain_cooldown_hours", 24)
    return state, retrained


async def test_high_real_error_starts_retrain(error_check):
    state, retrained = error_check

    assert await retrain.check_realized_error("uid", "1DAY") is True
    assert retrained == ["uid"]
    gauge = retrain.MODEL_REALIZED_MAPE.labels(instrument_uid="uid", interval="1DAY")
    assert gauge._value.get() == pytest.approx(0.05)


@pytest.mark.parametrize(
    "change",
    [
        {"mape": 0.02},
        {"points": 9},
        {"mape": None, "points": 0},
        {"version": None},
        {"last_run": datetime.now(UTC) - timedelta(hours=2)},
    ],
    ids=["error below limit", "too few forecasts", "nothing matched yet", "no model", "cooldown"],
)
async def test_real_error_check_does_not_retrain(error_check, change):
    state, retrained = error_check
    state.update(change)

    assert await retrain.check_realized_error("uid", "1DAY") is False
    assert retrained == []


async def test_real_error_check_runs_after_cooldown(error_check):
    state, retrained = error_check
    state["last_run"] = datetime.now(UTC) - timedelta(hours=25)

    assert await retrain.check_realized_error("uid", "1DAY") is True


async def test_record_prediction_never_fails(monkeypatch):
    async def broken(result):
        raise RuntimeError("database is down")

    monkeypatch.setattr(retrain, "save_prediction", broken)

    await retrain.record_prediction({"instrument_uid": "uid"})
