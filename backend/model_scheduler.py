import schedule
import time
from datetime import datetime
from continuous_learning import ContinuousLearning
from alert_system import AlertSystem

class ModelScheduler:
    def __init__(self):
        self.cl = ContinuousLearning(retrain_frequency_days=7)
        self.alerts = AlertSystem()

    def daily_model_check(self):
        print(f"\n[{datetime.now()}] Running daily model check...")

        try:
            should_retrain, reason = self.cl.should_retrain()
            print(f"Check result: {reason}")

            if should_retrain:
                print("Triggering retrain...")
                model_id = self.cl.run_retraining_cycle()

                if model_id:
                    self.alerts.send_email(
                        "Model Retrained Successfully",
                        f"New model deployed: {model_id}\nReason: {reason}"
                    )
                else:
                    self.alerts.send_email(
                        "Model Retrain Attempted",
                        f"Retrain attempted but new model rejected\nReason: {reason}"
                    )
            else:
                print("No retrain needed")

        except Exception as e:
            print(f"Error in model check: {e}")
            self.alerts.alert_system_error("Model Check Failed", str(e))

    def weekly_full_retrain(self):
        print(f"\n[{datetime.now()}] Running weekly forced retrain...")

        try:
            model_id = self.cl.run_retraining_cycle(force=True)

            if model_id:
                self.alerts.send_email(
                    "Weekly Model Retrain Complete",
                    f"New model deployed: {model_id}\nScheduled weekly retrain"
                )
            else:
                self.alerts.send_email(
                    "Weekly Model Retrain Failed",
                    "Weekly retrain did not produce a better model"
                )

        except Exception as e:
            print(f"Error in weekly retrain: {e}")
            self.alerts.alert_system_error("Weekly Retrain Failed", str(e))

    def run(self):
        print("Model Scheduler Started")
        print("Schedule:")
        print("  - Daily check: 2:00 AM")
        print("  - Weekly retrain: Sunday 1:00 AM")

        schedule.every().day.at("02:00").do(self.daily_model_check)
        schedule.every().sunday.at("01:00").do(self.weekly_full_retrain)

        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    scheduler = ModelScheduler()
    scheduler.run()
