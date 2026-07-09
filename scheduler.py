"""
scheduler.py
--------------
DataOps requirement: runs the full pipeline automatically every 2 minutes,
logging every run. This uses APScheduler (pip install apscheduler) which
works on any machine/cloud VM without needing Airflow.

Run with:  python scheduler.py
Stop with: Ctrl+C

CLOUD ALTERNATIVES (mention in your report if you deploy differently):
  - Apache Airflow DAG with schedule_interval='*/2 * * * *'
  - AWS Lambda + EventBridge rule: rate(2 minutes)
  - A plain cron job: */2 * * * * /usr/bin/python3 /path/to/run_pipeline.py
"""

import logging
import time

from run_pipeline import run_once, logger

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False


def scheduled_job():
    logger.info("SCHEDULER: triggering pipeline run")
    run_once()


if __name__ == "__main__":
    if APSCHEDULER_AVAILABLE:
        scheduler = BlockingScheduler()
        scheduler.add_job(scheduled_job, "interval", minutes=2, next_run_time=__import__("datetime").datetime.now())
        logger.info("SCHEDULER: started, running every 2 minutes (APScheduler). Press Ctrl+C to stop.")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("SCHEDULER: stopped by user")
    else:
        # Fallback: plain infinite loop, no extra dependency needed
        logger.warning("SCHEDULER: apscheduler not installed, falling back to a simple loop "
                        "(pip install apscheduler for a more robust scheduler)")
        try:
            while True:
                scheduled_job()
                time.sleep(120)  # 2 minutes
        except KeyboardInterrupt:
            logger.info("SCHEDULER: stopped by user")
