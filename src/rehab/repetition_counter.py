"""
Simple 2-state repetition counter (REST <-> FLEXED) driven by
exercise-specific thresholds from config/exercise_config.py.

A "correct" rep is one that reaches the flexed threshold before
returning to rest. A rep that reverses early (never reaches the
threshold) is counted as "incomplete".
"""


class RepetitionCounter:
    def __init__(self, exercise_config):
        self.cfg = exercise_config
        self.state = "REST"
        self.reached_target = False
        self.total_reps = 0
        self.correct_reps = 0
        self.incorrect_reps = 0

    def update(self, angle):
        if angle is None:
            return self.snapshot()

        rest_min = self.cfg["rest_angle_min"]
        flex_max = self.cfg["flexed_angle_max"]
        invert = self.cfg.get("invert", False)

        is_at_rest = angle >= rest_min if not invert else angle <= rest_min
        is_flexed = angle <= flex_max if not invert else angle >= flex_max

        if self.state == "REST" and is_flexed:
            self.state = "FLEXED"
            self.reached_target = True

        elif self.state == "FLEXED" and is_at_rest:
            # completed one full cycle back to rest
            self.total_reps += 1
            if self.reached_target:
                self.correct_reps += 1
            else:
                self.incorrect_reps += 1
            self.state = "REST"
            self.reached_target = False

        return self.snapshot()

    def snapshot(self):
        return {
            "state": self.state,
            "total_reps": self.total_reps,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
        }

    def reset(self):
        self.__init__(self.cfg)
