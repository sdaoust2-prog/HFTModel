import schedule
import time
from datetime import datetime
from continuous_learning import ContinuousLearning
from online_learning import OnlineLearner

class EnhancedLearningScheduler:
    def __init__(self):
        self.cl = ContinuousLearning()
        self.ol = OnlineLearner()

    def hourly_online_update(self):
        print(f"\n{'='*60}")
        print(f"HOURLY ONLINE LEARNING - {datetime.now()}")
        print(f"{'='*60}")

        try:
            self.ol.print_learning_report(lookback_hours=1)

            self.ol.reinforcement_learning_update(lookback_hours=1)

        except Exception as e:
            print(f"error in hourly online update: {e}")

    def end_of_day_learning(self):
        print(f"\n{'='*60}")
        print(f"END OF DAY LEARNING - {datetime.now()}")
        print(f"{'='*60}")

        try:
            print("\n1. Analyzing today's trading performance...")
            self.ol.print_learning_report(lookback_hours=24)

            print("\n2. Applying reinforcement learning from today's outcomes...")
            self.ol.reinforcement_learning_update(lookback_hours=24)

            print("\n3. Incremental model update with trade outcomes...")
            self.ol.incremental_update(lookback_hours=24, min_samples=10)

            print("\n4. Checking if full retrain needed...")
            should_retrain, reason = self.cl.should_retrain()

            if should_retrain:
                print(f"\nFull retrain triggered: {reason}")
                self.cl.retrain_and_deploy()
            else:
                print(f"\nFull retrain not needed: {reason}")

        except Exception as e:
            print(f"error in end of day learning: {e}")

    def weekly_full_retrain(self):
        print(f"\n{'='*60}")
        print(f"WEEKLY FULL RETRAIN - {datetime.now()}")
        print(f"{'='*60}")

        try:
            print("1. Collecting past week's trading data...")
            self.ol.print_learning_report(lookback_hours=168)

            print("\n2. Forcing full model retrain...")
            self.cl.retrain_and_deploy(force=True)

        except Exception as e:
            print(f"error in weekly retrain: {e}")

    def morning_prep(self):
        print(f"\n{'='*60}")
        print(f"MORNING MARKET PREP - {datetime.now()}")
        print(f"{'='*60}")

        try:
            print("Checking overnight model updates...")
            model_age_days = (datetime.now() - datetime.fromtimestamp(
                self.cl.get_current_model_age())).days

            print(f"Current model age: {model_age_days} days")

            print("\nReviewing yesterday's learning...")
            self.ol.print_learning_report(lookback_hours=24)

        except Exception as e:
            print(f"error in morning prep: {e}")

    def run(self):
        print(f"{'='*60}")
        print("ENHANCED LEARNING SCHEDULER STARTED")
        print(f"{'='*60}")
        print(f"\nSchedule:")
        print(f"  Hourly:       Online learning update (every trading hour)")
        print(f"  9:00 AM:      Morning market prep")
        print(f"  4:30 PM:      End-of-day learning + incremental update")
        print(f"  Sunday 1 AM:  Weekly full retrain")
        print(f"\nPress Ctrl+C to stop")
        print(f"{'='*60}\n")

        schedule.every().hour.do(self.hourly_online_update)

        schedule.every().day.at("09:00").do(self.morning_prep)

        schedule.every().day.at("16:30").do(self.end_of_day_learning)

        schedule.every().sunday.at("01:00").do(self.weekly_full_retrain)

        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    import sys

    scheduler = EnhancedLearningScheduler()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test-hourly':
            scheduler.hourly_online_update()
        elif sys.argv[1] == 'test-eod':
            scheduler.end_of_day_learning()
        elif sys.argv[1] == 'test-weekly':
            scheduler.weekly_full_retrain()
        elif sys.argv[1] == 'test-morning':
            scheduler.morning_prep()
        else:
            print("usage: python enhanced_learning_scheduler.py [test-hourly|test-eod|test-weekly|test-morning]")
    else:
        scheduler.run()
