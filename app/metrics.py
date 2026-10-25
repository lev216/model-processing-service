from prometheus_client import Counter, Gauge, Histogram

TRAINING_RUNS_TOTAL = Counter(
    "model_training_runs_total",
    "Количество запусков обучения",
    labelnames=["instrument_uid", "interval", "status"],
)

TRAINING_DURATION_SECONDS = Histogram(
    "model_training_duration_seconds",
    "Время обучения модели в секундах",
    labelnames=["instrument_uid", "interval"],
    buckets=[30, 60, 90, 120, 180, 300, 600, 1200, 1800, 3600],
)

MODEL_BEST_MAPE = Gauge(
    "model_best_mape",
    "MAPE лучшей модели после последнего обучения",
    labelnames=["instrument_uid", "interval", "best_model"],
)

MODEL_TRAINING_SAMPLES = Gauge(
    "model_training_samples_total",
    "Количество свечей в обучающей выборке",
    labelnames=["instrument_uid", "interval"],
)

MODEL_LEADERBOARD_POSITION = Gauge(
    "model_leaderboard_mape",
    "MAPE каждой модели из leaderboard последнего запуска",
    labelnames=["instrument_uid", "interval", "model_name"],
)


PREDICTION_TOTAL = Counter(
    "model_prediction_total",
    "Количество прогнозов",
    labelnames=["instrument_uid", "interval", "status"],
)

PREDICTION_DURATION_SECONDS = Histogram(
    "model_prediction_duration_seconds",
    "Время выполнения инференса в секундах",
    labelnames=["instrument_uid", "interval"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

PREDICTED_CHANGE_PCT = Histogram(
    "model_predicted_change_pct",
    "Распределение предсказанного изменения цены в %",
    labelnames=["instrument_uid", "interval"],
    buckets=[-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10],
)

PREDICTED_CLOSE_PRICE = Gauge(
    "model_predicted_close_price",
    "Последняя предсказанная цена закрытия",
    labelnames=["instrument_uid", "interval"],
)


SCHEDULER_JOB_RUNS_TOTAL = Counter(
    "scheduler_job_runs_total",
    "Количество запусков задач планировщика",
    labelnames=["instrument_uid", "interval", "status"],
)

SCHEDULER_JOB_DURATION_SECONDS = Histogram(
    "scheduler_job_duration_seconds",
    "Время выполнения задачи планировщика в секундах",
    labelnames=["instrument_uid", "interval"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

SCHEDULER_JOBS_ACTIVE = Gauge(
    "scheduler_jobs_active",
    "Количество активных задач в планировщике",
)


MODEL_REALIZED_MAPE = Gauge(
    "model_realized_mape",
    "MAPE последних прогнозов рабочей модели относительно фактических цен закрытия",
    labelnames=["instrument_uid", "interval"],
)
